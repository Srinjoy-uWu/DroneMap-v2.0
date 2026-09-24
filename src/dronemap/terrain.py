"""2.5D Terrain surface mesh reconstruction engine.

Reconstructs a solid, continuous 2.5D Digital Surface Model (TIN mesh) from
sparse or dense point clouds.  This is ideal for nadir/terrain drone flights
where 3D Delaunay volumetric carving (OpenMVS ReconstructMesh) may struggle
with single-viewpoint parallax or produce airborne floaters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.spatial
import trimesh
from PIL import Image

if TYPE_CHECKING:
    from .workspace import RunWorkspace


# A ground plane recovered from a georeferenced cloud cannot be steeply tilted:
# UTM +Z *is* up by construction, so anything beyond a modest slope means the
# fit locked onto facades or floaters rather than the ground.  15 deg is well
# past any real terrain gradient a survey flight covers and far short of the
# 60-85 deg the unconstrained PCA fit was producing.
MAX_GROUND_TILT_DEG = 15.0


@dataclass(frozen=True)
class GroundPlane:
    """A measured ground plane, with the evidence for trusting it.

    ``rotation`` maps world coordinates into a frame whose +Z is the plane
    normal, so the 2.5D grid can be laid out in the plane.  The remaining
    fields exist so a caller can tell a fit that *worked* from one that was
    rejected -- the previous version returned only the rotation, and a rotation
    is equally well-formed whether it is right or 70 degrees wrong.
    """

    center: np.ndarray
    normal: np.ndarray
    rotation: np.ndarray
    tilt_deg: float
    rms_residual_m: float
    inlier_fraction: float
    source: str  # "fitted" | "prior" | "fit_rejected"


def _rotation_aligning(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper rotation (det = +1) taking unit ``source`` onto unit ``target``.

    The antiparallel case matters here.  The previous implementation returned
    ``-np.eye(3)`` for it, which is a *reflection*: determinant -1, so it
    mirrors the scene and inverts every face winding.  A 180 deg rotation about
    any axis perpendicular to ``source`` is the correct answer and keeps the
    handedness of the mesh intact.
    """
    v = np.cross(source, target)
    c = float(np.dot(source, target))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 deg about any perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s**2))


def fit_ground_plane(
    points: np.ndarray,
    up_hint: np.ndarray | None = None,
    max_tilt_deg: float = MAX_GROUND_TILT_DEG,
) -> GroundPlane:
    """Recover the ground plane of a dense cloud, robustly and measurably.

    The plane that PCA finds is the axis of least variance of *every* point,
    and in an urban dense cloud that is not the ground.  Building facades,
    vegetation and reconstruction floaters all contribute variance along the
    vertical, and on a single-pass orbit the horizontal footprint is narrow in
    one direction, so the least-variance axis routinely comes out nearly
    horizontal.  Measured on the runs in this repository, the old fit reported
    ground normals tilted 60-85 deg from vertical on five of the six real
    scenes -- ``fixture_orbit_hires`` at 59.9 deg, ``fixture_corridor_v1`` at
    70.5 deg -- which is why the 2.5D product came out on a plane that had
    nothing to do with the ground.  Rotating a scene by that much collapses its
    true height: the corridor's 228 m of relief became 22 m of "elevation" in
    the rotated frame.

    Two further defects made it worse.  The eigenvector's sign is arbitrary, so
    the normal pointed *down* on four runs, inverting the surface -- buildings
    became pits.  And the antiparallel branch returned a reflection rather than
    a rotation, mirroring the mesh.

    This replaces that with three things:

    1.  **A prior.**  In a georeferenced cloud the vertical is known -- UTM +Z
        is up -- so the fit refines a known answer instead of searching for it.
    2.  **A robust estimate.**  The plane is fitted to the *lower envelope*
        (points between the 2nd and 30th height percentiles), which is ground
        by construction, not to the full cloud.  The low tail is excluded so
        sub-surface floaters cannot drag it.
    3.  **A rejection test.**  A fit tilted further than ``max_tilt_deg`` from
        the prior is discarded in favour of the prior, and says so in
        ``source``.  A wrong plane is worse than no plane, and silently
        shipping one is what produced the misaligned product.
    """
    up = np.array([0.0, 0.0, 1.0]) if up_hint is None else np.asarray(up_hint, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-12)

    if len(points) < 16:
        # Too few points to fit anything defensible; use the prior and say so.
        return GroundPlane(
            center=points.mean(axis=0) if len(points) else np.zeros(3),
            normal=up,
            rotation=_rotation_aligning(up, np.array([0.0, 0.0, 1.0])),
            tilt_deg=0.0,
            rms_residual_m=float("nan"),
            inlier_fraction=0.0,
            source="prior",
        )

    normal = up.copy()
    center = np.median(points, axis=0)
    band = points

    # Three passes is enough: the height ordering barely changes once the
    # normal is within a few degrees, and more iterations only re-fit noise.
    for _ in range(3):
        height = (points - center) @ normal
        lo, hi = np.percentile(height, [2.0, 30.0])
        band = points[(height >= lo) & (height <= hi)]
        if len(band) < 16:
            band = points
            break
        center = band.mean(axis=0)
        centred = band - center
        cov = centred.T @ centred / len(band)
        evals, evecs = np.linalg.eigh(cov)
        candidate = evecs[:, 0]
        candidate /= max(float(np.linalg.norm(candidate)), 1e-12)
        # Resolve the eigenvector's arbitrary sign against the prior.
        if float(np.dot(candidate, up)) < 0.0:
            candidate = -candidate
        normal = candidate

    tilt = math.degrees(math.acos(float(np.clip(np.dot(normal, up), -1.0, 1.0))))
    source = "fitted"

    residual = (band - center) @ normal
    rms = float(np.sqrt(np.mean(residual**2))) if len(band) else float("nan")
    # "Inlier" is relative to the spread of the band itself, so the number
    # means the same thing on a flat airfield and on a sloping hillside.
    tol = max(3.0 * rms, 0.05) if np.isfinite(rms) else np.inf
    inliers = float(np.mean(np.abs(residual) <= tol)) if len(band) else 0.0

    if not np.isfinite(tilt) or tilt > max_tilt_deg:
        normal = up
        center = band.mean(axis=0) if len(band) else points.mean(axis=0)
        source = "fit_rejected"
        tilt = 0.0

    return GroundPlane(
        center=center,
        normal=normal,
        rotation=_rotation_aligning(normal, np.array([0.0, 0.0, 1.0])),
        tilt_deg=float(tilt),
        rms_residual_m=rms,
        inlier_fraction=round(inliers, 4),
        source=source,
    )


def _fit_ground_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backwards-compatible 3-tuple wrapper around :func:`fit_ground_plane`."""
    plane = fit_ground_plane(points)
    return plane.center, plane.normal, plane.rotation


def up_hint_from_cameras(images_txt: Path) -> np.ndarray | None:
    """Derive the vertical from where the cameras were looking.

    Without telemetry, COLMAP's world frame is arbitrary: its +Z has no
    relation to gravity, so defaulting the ground-plane prior to +Z is a guess
    dressed up as a prior.  But the poses themselves carry the information.  A
    survey flight points its camera at the ground, so the mean optical axis is
    a good estimate of *down*, and its negation of up.

    This is weaker than telemetry and is not a substitute for it: an orbit that
    tilts the gimbal to 45 degrees biases the estimate by roughly that much.
    It is used only as the prior a fit is measured against, and the fit is
    still allowed to reject it.  Returns ``None`` when the poses cannot be
    read or disagree with each other too much to mean anything -- which is a
    distinct outcome from "up is +Z".
    """
    if not images_txt.exists():
        for candidate in [
            images_txt.parent / "txt" / "images.txt",
            images_txt.parent / "0" / "txt" / "images.txt",
            images_txt.parent / "sparse" / "0" / "txt" / "images.txt",
            images_txt.parent.parent / "0" / "txt" / "images.txt",
            images_txt.parent.parent / "sparse" / "0" / "txt" / "images.txt",
        ]:
            if candidate.exists():
                images_txt = candidate
                break
    if not images_txt.exists():
        return None

    axes: list[np.ndarray] = []
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
    try:
        with images_txt.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                # Image header lines in COLMAP images.txt have exactly 10 tokens:
                # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                if len(parts) != 10:
                    continue
                name = parts[9].lower()
                if not any(name.endswith(ext) for ext in IMAGE_EXTS):
                    continue
                qw, qx, qy, qz = (float(v) for v in parts[1:5])
                n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                if n < 1e-9:
                    continue
                qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
                # Third row of R (world->camera) transposed is R^T @ [0,0,1]:
                # the camera's viewing direction expressed in world coordinates.
                axes.append(np.array([
                    2.0 * (qx * qz + qw * qy),
                    2.0 * (qy * qz - qw * qx),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ]))
    except Exception:
        return None

    if len(axes) < 3:
        return None

    stacked = np.asarray(axes)
    mean_axis = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean_axis))
    # A norm near zero means the views cancel out -- cameras pointing every
    # which way, so there is no consistent "down" to extract.  0.5 keeps an
    # orbit with a 60 deg spread and discards a genuinely incoherent set.
    if norm < 0.5:
        return None
    return -mean_axis / norm


def reconstruct_terrain_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    output_obj: Path,
    output_ply: Path | None = None,
    grid_dim: int = 250,
    max_grid_dim: int = 400,
    k_neighbors: int = 8,
    up_hint: np.ndarray | None = None,
    max_tilt_deg: float = MAX_GROUND_TILT_DEG,
    texture_size: int = 2048,
) -> dict[str, int | float | str]:
    """Build a solid, non-deformed 2.5D terrain surface mesh from 3D points.

    Key improvements over naive regular-grid IDW:
    1. Data Support Masking: Discards grid nodes and triangles that lie outside
       the actual survey footprint (distance to nearest point > 2.5x cell size).
       This completely eliminates sagging border curtains, corner droops, and
       30m stretched triangles.
    2. Regularized Elevation & Outlier Suppression: Avoids singular 1/d^2 spikes
       and removes local height outliers so noise/vegetation points cannot create
       artificial cones.
    3. Analytical Surface Normals: Computes gradients directly on the elevation
       field to bake smooth vertex normals into the OBJ/PLY/GLB, guaranteeing
       continuous PBR shading in 3D viewers.
    4. High-Resolution Texture Atlas: Upsamples and antialiases the surface
       texture to 2048x2048 for crisp, photorealistic terrain appearance.
    """
    import cv2
    import scipy.ndimage

    if len(points) < 10:
        raise RuntimeError(f"Cannot reconstruct terrain mesh: only {len(points)} points available.")

    output_obj.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fit ground plane and rotate into canonical horizontal coordinates
    plane = fit_ground_plane(points, up_hint=up_hint, max_tilt_deg=max_tilt_deg)
    center, R = plane.center, plane.rotation
    pts_rot = (points - center) @ R.T

    # 2. Compute 2D bounding box with 0.5-99.5 percentile trimming to suppress extreme outliers
    x_min, x_max = float(np.percentile(pts_rot[:, 0], 0.5)), float(np.percentile(pts_rot[:, 0], 99.5))
    y_min, y_max = float(np.percentile(pts_rot[:, 1], 0.5)), float(np.percentile(pts_rot[:, 1], 99.5))

    dx = max(x_max - x_min, 1e-3)
    dy = max(y_max - y_min, 1e-3)

    aspect = dy / dx
    if aspect >= 1.0:
        ny = min(max_grid_dim, max(40, int(grid_dim * aspect)))
        nx = min(max_grid_dim, max(40, int(grid_dim)))
    else:
        nx = min(max_grid_dim, max(40, int(grid_dim / aspect)))
        ny = min(max_grid_dim, max(40, int(grid_dim)))

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    GX, GY = np.meshgrid(xi, yi)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
    cell_size = max(dx / max(nx - 1, 1), dy / max(ny - 1, 1))

    # 3. KDTree queries: nearest distance (for support masking) and k-neighbors (for IDW)
    tree = scipy.spatial.cKDTree(pts_rot[:, :2])
    dists_1, _ = tree.query(grid_xy, k=1)
    max_support_dist = max(2.5 * cell_size, 0.6)
    valid_node = (dists_1 <= max_support_dist)

    k_query = min(max(k_neighbors, 8), len(points))
    dists, idxs = tree.query(grid_xy, k=k_query)
    if k_query == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Regularized inverse distance weighting to prevent singular division
    reg_eps = max(0.25 * cell_size, 0.05)
    weights = 1.0 / (np.maximum(dists, 1e-4) + reg_eps)**2

    # Local outlier rejection on Z to suppress aerial floaters and sub-surface spikes
    local_z = pts_rot[idxs, 2]
    med_z = np.median(local_z, axis=1, keepdims=True)
    dev_z = np.abs(local_z - med_z)
    mad_z = np.median(dev_z, axis=1, keepdims=True) * 1.4826
    outlier_mask = dev_z > np.maximum(3.0 * mad_z, 0.5)
    weights[outlier_mask] = 0.0
    weight_sum = weights.sum(axis=1, keepdims=True)
    zero_weights = (weight_sum[:, 0] == 0)
    if np.any(zero_weights):
        weights[zero_weights] = 1.0 / (dists[zero_weights] + reg_eps)**2
        weight_sum = weights.sum(axis=1, keepdims=True)
    weights /= np.maximum(weight_sum, 1e-12)

    gz = np.sum(weights * local_z, axis=1)

    # 4. Normalised 2D Gaussian smoothing on valid elevation grid
    Z_grid = gz.reshape((ny, nx))
    Valid_grid = valid_node.reshape((ny, nx))
    blurred_z = scipy.ndimage.gaussian_filter(Z_grid * Valid_grid, sigma=0.75)
    blurred_w = scipy.ndimage.gaussian_filter(Valid_grid.astype(float), sigma=0.75)
    safe_w = np.maximum(blurred_w, 1e-6)
    Z_smooth = np.where(blurred_w > 1e-3, blurred_z / safe_w, Z_grid)
    gz = np.where(valid_node, Z_smooth.ravel(), gz)

    # 5. Compute analytical smooth surface vertex normals
    dz_dy, dz_dx = np.gradient(Z_smooth, dy / max(ny - 1, 1), dx / max(nx - 1, 1))
    N_rot = np.column_stack([-dz_dx.ravel(), -dz_dy.ravel(), np.ones_like(gz)])
    N_rot /= np.maximum(np.linalg.norm(N_rot, axis=1, keepdims=True), 1e-12)
    N_orig = N_rot @ plane.rotation

    # 6. Transform grid vertices back to original 3D coordinates
    V_rot = np.column_stack([grid_xy, gz])
    V_orig = V_rot @ plane.rotation + center

    # 7. Triangulation with strict boundary support masking
    faces = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            i0 = r * nx + c
            i1 = r * nx + (c + 1)
            i2 = (r + 1) * nx + c
            i3 = (r + 1) * nx + (c + 1)
            if valid_node[i0] and valid_node[i1] and valid_node[i2]:
                faces.append([i0, i1, i2])
            if valid_node[i1] and valid_node[i3] and valid_node[i2]:
                faces.append([i1, i3, i2])

    if not faces:
        # Fallback in degenerate case: keep regular triangulation
        for r in range(ny - 1):
            for c in range(nx - 1):
                i0 = r * nx + c
                i1 = r * nx + (c + 1)
                i2 = (r + 1) * nx + c
                i3 = (r + 1) * nx + (c + 1)
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])

    faces_arr = np.array(faces, dtype=np.int32)
    used_indices = np.unique(faces_arr)
    index_map = np.full(len(grid_xy), -1, dtype=np.int32)
    index_map[used_indices] = np.arange(len(used_indices), dtype=np.int32)
    faces_remapped = index_map[faces_arr]

    # UV coordinates
    u = (GX.ravel() - x_min) / dx
    v = (GY.ravel() - y_min) / dy
    uv = np.column_stack([u, v])

    V_used = V_orig[used_indices]
    N_used = N_orig[used_indices]
    UV_used = uv[used_indices]

    # 8. High-resolution texture atlas generation
    if colors is not None and len(colors) == len(points):
        gr = np.clip(np.sum(weights * colors[idxs, 0], axis=1), 0, 255).astype(np.uint8)
        gg = np.clip(np.sum(weights * colors[idxs, 1], axis=1), 0, 255).astype(np.uint8)
        gb = np.clip(np.sum(weights * colors[idxs, 2], axis=1), 0, 255).astype(np.uint8)
    else:
        gr = np.full(len(gz), 180, dtype=np.uint8)
        gg = np.full(len(gz), 180, dtype=np.uint8)
        gb = np.full(len(gz), 180, dtype=np.uint8)

    base_tex = np.column_stack([gr, gg, gb]).reshape((ny, nx, 3))
    tex_res = min(max(texture_size, 1024), 4096)
    if (ny, nx) != (tex_res, tex_res):
        tex_img = cv2.resize(base_tex, (tex_res, tex_res), interpolation=cv2.INTER_CUBIC)
    else:
        tex_img = base_tex

    # Flip vertically to match glTF UV convention (v=0 at bottom of image)
    tex_img = np.flipud(tex_img)
    pil_img = Image.fromarray(tex_img)
    visual = trimesh.visual.TextureVisuals(uv=UV_used, image=pil_img)

    mesh = trimesh.Trimesh(
        vertices=V_used,
        faces=faces_remapped,
        vertex_normals=N_used,
        visual=visual,
        process=False,
    )

    # 9. Export OBJ + MTL + Texture
    obj_str, files = trimesh.exchange.obj.export_obj(mesh, return_texture=True)
    output_obj.write_text(obj_str, encoding="utf-8")
    for filename, content in files.items():
        file_path = output_obj.parent / filename
        if filename.endswith(".mtl"):
            # Ensure full diffuse reflectance (Kd 1.0) so glTF export does not dim by 60%
            if isinstance(content, bytes):
                content = content.replace(
                    b"Kd 0.40000000 0.40000000 0.40000000",
                    b"Kd 1.00000000 1.00000000 1.00000000",
                )
            elif isinstance(content, str):
                content = content.replace(
                    "Kd 0.40000000 0.40000000 0.40000000",
                    "Kd 1.00000000 1.00000000 1.00000000",
                )
        if isinstance(content, str):
            file_path.write_text(content, encoding="utf-8")
        else:
            file_path.write_bytes(content)

    if output_ply is not None:
        mesh.export(str(output_ply))

    return {
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "grid_nx": nx,
        "grid_ny": ny,
        "ground_tilt_deg": round(plane.tilt_deg, 3),
        "ground_plane_source": plane.source,
        "ground_rms_residual_m": (
            round(plane.rms_residual_m, 4) if np.isfinite(plane.rms_residual_m) else None
        ),
        "ground_inlier_fraction": plane.inlier_fraction,
        "ground_normal": [round(float(v), 6) for v in plane.normal],
        "relief_m": round(float(pts_rot[:, 2].max() - pts_rot[:, 2].min()), 3),
        "cloud_z_span_m": round(float(points[:, 2].max() - points[:, 2].min()), 3),
    }

