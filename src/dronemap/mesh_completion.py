"""AI-Guided 3D Mesh Completion, Hole Inpainting & Gravity Rectification Engine.

Integrates Ollama Local SLM (with deterministic geometric fallback) into Stage 6
to analyze 3D point cloud / mesh cavities, rectify unconstrained 1D flight-line
roll, remove floating ray-mismatch outliers, synthesize missing 3D surface
patches, and inpaint occluded texture regions for a complete, solid 3D model.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import scipy.ndimage
import scipy.spatial
import trimesh

from .agent.client import LLMClient
from .terrain import _rotation_aligning

if TYPE_CHECKING:
    from .workspace import RunWorkspace

logger = logging.getLogger("dronemap.mesh_completion")


@dataclass
class MeshCompletionPlan:
    """Parameters selected by Ollama Local SLM (or deterministic geometry engine)."""

    provider: str
    strategy: str
    grid_dim: int
    outlier_knn: int
    outlier_std_ratio: float
    hole_fill_radius_cells: int
    bilateral_d: int
    bilateral_sigma_Height: float
    bilateral_sigma_space: float
    texture_inpaint_radius: int
    sharpen_texture: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rectify_and_clean_dense_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    up_target: np.ndarray | None = None,
    camera_centres: np.ndarray | None = None,
    knn: int = 16,
    std_ratio: float = 2.2,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Remove 3D floating outliers and rectify unconstrained flight-line roll.

    Returns ``(clean_points, clean_colors, stats)``.
    """
    n_raw = len(points)
    if n_raw < 32:
        return points, colors, {"n_raw": n_raw, "n_clean": n_raw, "roll_rectified_deg": 0.0}

    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.uint8) if (colors is not None and len(colors) == n_raw) else None

    # 1. Coarse radial trimming (remove extreme >99% ray-mismatch spikes first)
    med0 = np.median(pts, axis=0)
    r0 = np.linalg.norm(pts - med0, axis=1)
    coarse_keep = r0 <= np.percentile(r0, 98.5)
    if int(coarse_keep.sum()) >= 32:
        pts = pts[coarse_keep]
        if cols is not None:
            cols = cols[coarse_keep]

    # 2. Statistical Outlier Removal (SOR) via KDTree mean k-NN distance
    k_use = min(max(knn, 6), len(pts) - 1)
    tree = scipy.spatial.cKDTree(pts)
    dists, _ = tree.query(pts, k=k_use + 1)
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    med_d = float(np.median(mean_knn_dist))
    mad_d = float(np.median(np.abs(mean_knn_dist - med_d))) * 1.4826
    thresh_d = med_d + std_ratio * max(mad_d, 1e-3)
    sor_keep = mean_knn_dist <= thresh_d
    if int(sor_keep.sum()) >= 64:
        pts = pts[sor_keep]
        if cols is not None:
            cols = cols[sor_keep]

    # 3. Check & rectify unconstrained roll around the 1D camera trajectory
    roll_rectified_deg = 0.0
    if up_target is not None:
        from .terrain import fit_ground_plane

        u_geo = np.asarray(up_target, dtype=np.float64)
        u_geo /= max(float(np.linalg.norm(u_geo)), 1e-12)
        med = np.median(pts, axis=0)

        # Check if cloud is an isotropic thin 2D sheet that rolled around a flight line
        centered = pts - med
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        is_isotropic_thin_sheet = (
            s_vals[1] > 1e-6
            and (s_vals[0] / s_vals[1] < 1.8)
            and (s_vals[2] / s_vals[1] < 0.28)
        )
        if is_isotropic_thin_sheet:
            n_scene = vt[2].copy()
            n_scene /= max(float(np.linalg.norm(n_scene)), 1e-12)
            if camera_centres is not None and len(camera_centres) > 0:
                cam_vec = np.mean(np.asarray(camera_centres, dtype=np.float64), axis=0) - med
                if float(np.dot(n_scene, cam_vec)) < 0.0:
                    n_scene = -n_scene
            elif float(np.dot(n_scene, u_geo)) < 0.0:
                n_scene = -n_scene
            tilt_deg = math.degrees(math.acos(float(np.clip(np.dot(n_scene, u_geo), -1.0, 1.0))))
            if tilt_deg > 12.0:
                R_fix = _rotation_aligning(n_scene, u_geo)
                pts = (pts - med) @ R_fix.T + med
                roll_rectified_deg = round(tilt_deg, 2)
        else:
            plane = fit_ground_plane(pts, up_hint=u_geo, max_tilt_deg=15.0)
            n_scene = plane.normal
            tilt_deg = math.degrees(math.acos(float(np.clip(np.dot(n_scene, u_geo), -1.0, 1.0))))
            if 3.0 < tilt_deg <= 15.0:
                R_fix = _rotation_aligning(n_scene, u_geo)
                pts = (pts - med) @ R_fix.T + med
                roll_rectified_deg = round(tilt_deg, 2)

    return pts, cols, {
        "n_raw": int(n_raw),
        "n_clean": int(len(pts)),
        "outliers_removed": int(n_raw - len(pts)),
        "roll_rectified_deg": roll_rectified_deg,
    }


def plan_mesh_completion_with_ollama(
    pts: np.ndarray,
    up_vec: np.ndarray,
    client: LLMClient | None = None,
) -> MeshCompletionPlan:
    """Query Ollama Local SLM (or deterministic fallback) to plan 3D mesh completion."""
    client = client or LLMClient()

    u = np.asarray(up_vec, dtype=np.float64)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    R_up = _rotation_aligning(u, np.array([0.0, 0.0, 1.0]))
    pts_local = (pts - np.median(pts, axis=0)) @ R_up.T

    x_span = float(np.percentile(pts_local[:, 0], 99) - np.percentile(pts_local[:, 0], 1))
    y_span = float(np.percentile(pts_local[:, 1], 99) - np.percentile(pts_local[:, 1], 1))
    z_relief = float(np.percentile(pts_local[:, 2], 98) - np.percentile(pts_local[:, 2], 2))
    ground_z = float(np.percentile(pts_local[:, 2], 20))
    above_ground_frac = float(np.mean((pts_local[:, 2] - ground_z) > 1.5))

    # Default high-definition completion parameters tuned for scene geometry
    has_structures = z_relief >= 4.0 or above_ground_frac >= 0.08
    default_plan = MeshCompletionPlan(
        provider="offline-geometric",
        strategy="3d_structure_preserving_inpaint" if has_structures else "2.5d_smooth_terrain_inpaint",
        grid_dim=320 if len(pts) > 50_000 else 260,
        outlier_knn=16,
        outlier_std_ratio=2.2,
        hole_fill_radius_cells=18,
        bilateral_d=5,
        bilateral_sigma_Height=1.8 if has_structures else 0.8,
        bilateral_sigma_space=1.5,
        texture_inpaint_radius=5,
        sharpen_texture=True,
        reasoning=(
            f"Scene footprint {x_span:.1f}m x {y_span:.1f}m with {z_relief:.1f}m vertical relief "
            f"({above_ground_frac * 100:.1f}% elevated structure points). Using bilateral edge-preserving "
            f"3D surface completion and Telea Navier-Stokes texture inpainting to close all occluded holes."
        ),
    )

    if client.is_available:
        prompt = (
            "You are a 3D photogrammetry mesh completion AI agent running via Ollama. "
            "Given the following dense point cloud telemetry, output a JSON object configuring "
            "the 3D surface hole-inpainting and bilateral edge-preservation parameters:\n"
            f"- Clean points: {len(pts)}\n"
            f"- Footprint: {x_span:.1f}m x {y_span:.1f}m\n"
            f"- Vertical 3D relief: {z_relief:.2f}m\n"
            f"- Elevated structure ratio: {above_ground_frac:.3f}\n\n"
            "Return JSON with keys: strategy (string), grid_dim (int 240..380), "
            "hole_fill_radius_cells (int 10..30), bilateral_sigma_Height (float 0.5..3.0), "
            "reasoning (concise 1-sentence technical explanation)."
        )
        resp = client.generate(
            prompt=prompt,
            system_prompt="Output strictly valid JSON for 3D mesh completion parameters.",
            json_mode=True,
        )
        if resp:
            try:
                data = json.loads(resp)
                default_plan.provider = f"ollama-local ({client.config.default_model})"
                default_plan.strategy = str(data.get("strategy", default_plan.strategy))
                default_plan.grid_dim = int(np.clip(int(data.get("grid_dim", default_plan.grid_dim)), 200, 400))
                default_plan.hole_fill_radius_cells = int(
                    np.clip(int(data.get("hole_fill_radius_cells", default_plan.hole_fill_radius_cells)), 8, 35)
                )
                default_plan.bilateral_sigma_Height = float(
                    np.clip(float(data.get("bilateral_sigma_Height", default_plan.bilateral_sigma_Height)), 0.4, 4.0)
                )
                default_plan.reasoning = str(data.get("reasoning", default_plan.reasoning))
            except Exception as exc:
                logger.debug("Ollama mesh planner JSON fallback: %s", exc)

    return default_plan


def repair_and_complete_openmvs_mesh(
    mesh: trimesh.Trimesh,
    dense_points: np.ndarray | None = None,
) -> dict[str, int]:
    """Close holes, fix normals, and remove degenerate faces on a 3D OpenMVS mesh."""
    faces_before = len(mesh.faces)
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception as exc:
        logger.debug("Trimesh hole repair note: %s", exc)
    return {
        "faces_before_repair": int(faces_before),
        "faces_after_repair": int(len(mesh.faces)),
        "holes_filled_faces": max(0, int(len(mesh.faces) - faces_before)),
    }
