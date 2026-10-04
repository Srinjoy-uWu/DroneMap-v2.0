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
    force_prior: bool = False,
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

    if force_prior or len(points) < 16:
        return GroundPlane(
            center=points.mean(axis=0) if len(points) else np.zeros(3),
            normal=up,
            rotation=_rotation_aligning(up, np.array([0.0, 0.0, 1.0])),
            tilt_deg=0.0,
            rms_residual_m=0.0,
            inlier_fraction=1.0,
            source="datum_vertical" if force_prior else "prior",
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
    grid_dim: int = 500,
    max_grid_dim: int = 800,
    k_neighbors: int = 12,
    up_hint: np.ndarray | None = None,
    max_tilt_deg: float = MAX_GROUND_TILT_DEG,
    texture_size: int = 4096,
    is_georef: bool = False,
    labels: np.ndarray | None = None,
) -> dict[str, int | float | str]:
    """Build a solid, non-deformed 2.5D terrain surface mesh from 3D points.

    Key improvements:
    1. Data Support Masking: Discards grid nodes and triangles outside survey footprint.
    2. Road & Ground Datum Leveling: Robust percentile estimation extracts the true continuous
       ground and road surface without being warped by aerial floaters or roadside trees.
    3. Elevated Structure Preservation: Coherent planar building rooftops are preserved at true
       elevations while isolated vegetation canopy spikes are de-spiked.
    4. Direct Point-Cloud 4K/8K Texture Atlas: Accumulates all dense points directly into the
       texture image with hierarchical inpainting, capturing crisp road markings, asphalt texture,
       and roof details.
    """
    import cv2
    import scipy.ndimage

    if len(points) < 10:
        raise RuntimeError(f"Cannot reconstruct terrain mesh: only {len(points)} points available.")

    output_obj.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fit ground plane and rotate into canonical horizontal coordinates
    plane = fit_ground_plane(points, up_hint=up_hint, max_tilt_deg=max_tilt_deg, force_prior=is_georef)
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
    direct_valid = (dists_1 <= max_support_dist).reshape((ny, nx))

    # Close interior occlusion holes/cavities while preserving exterior boundary footprint
    closed_valid = scipy.ndimage.binary_closing(
        direct_valid, structure=np.ones((3, 3), dtype=bool), iterations=2
    )
    filled_valid = scipy.ndimage.binary_fill_holes(closed_valid)
    inpainted_mask = (filled_valid & ~direct_valid).astype(np.uint8)
    valid_node = filled_valid.ravel()

    k_query = min(max(k_neighbors, 16), len(points))
    dists, idxs = tree.query(grid_xy, k=k_query)
    if k_query == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    local_z = pts_rot[idxs, 2]

    # Surface height estimation:
    # 1. Base ground and road datum:
    # Ground/road returns always occupy the lower envelope of point returns.
    # The 20th percentile captures the smooth, continuous road and ground surface without tree interference.
    z_ground = np.percentile(local_z, 20, axis=1)

    # 2. Elevated structure detection:
    # Identify cells that represent genuine planar buildings or structures (e.g. flat/sloped roofs)
    # vs scattered tree foliage or isolated spikes.
    mask_elev = local_z > (z_ground[:, None] + 1.2)
    count_elev = np.sum(mask_elev, axis=1)

    sum_elev = np.sum(np.where(mask_elev, local_z, 0.0), axis=1)
    mean_elev = sum_elev / np.maximum(count_elev, 1)
    sq_diff_elev = np.sum(np.where(mask_elev, (local_z - mean_elev[:, None]) ** 2, 0.0), axis=1)
    std_elev = np.sqrt(sq_diff_elev / np.maximum(count_elev, 1))

    # Coherent structures have multiple points agreeing tightly on elevation (std <= 0.65m)
    # Trees have wide vertical scatter (leaves across meters, std > 1.0m) and are de-spiked.
    is_structure = (count_elev >= 3) & (std_elev <= 0.65)

    if labels is not None and len(labels) == len(points):
        # CLASS_STRUCTURE = 4, CLASS_VEGETATION = 3, CLASS_TERRAIN = 2
        local_labels = labels[idxs]
        struct_pts_count = np.sum(local_labels == 4, axis=1)
        veg_pts_count = np.sum(local_labels == 3, axis=1)
        is_structure = is_structure | ((struct_pts_count >= 3) & (struct_pts_count > veg_pts_count) & (count_elev >= 2))
        is_structure = is_structure & ~((veg_pts_count >= 4) & (struct_pts_count == 0))

    gz = np.where(is_structure, mean_elev, z_ground)

    # 4. Edge-preserving 3D Bilateral Filtering
    Z_grid = gz.reshape((ny, nx)).astype(np.float32)
    Valid_grid = filled_valid
    try:
        # Sharp edge-preserving threshold: 0.65m prevents smoothing across vertical building walls (>= 1m)
        # while planarizing flat roofs and silky smooth roads.
        Z_bilat = cv2.bilateralFilter(
            Z_grid, d=5, sigmaColor=0.65, sigmaSpace=1.5
        )
        Z_smooth = np.where(Valid_grid, Z_bilat, Z_grid).astype(np.float64)
    except Exception:
        blurred_z = scipy.ndimage.gaussian_filter(Z_grid * Valid_grid, sigma=0.65)
        blurred_w = scipy.ndimage.gaussian_filter(Valid_grid.astype(float), sigma=0.65)
        Z_smooth = np.where(blurred_w > 1e-3, blurred_z / np.maximum(blurred_w, 1e-6), Z_grid)
    gz = np.where(valid_node, Z_smooth.ravel(), gz)

    # 5. Compute analytical smooth surface vertex normals
    dz_dy, dz_dx = np.gradient(Z_smooth, dy / max(ny - 1, 1), dx / max(nx - 1, 1))
    N_rot = np.column_stack([-dz_dx.ravel(), -dz_dy.ravel(), np.ones_like(gz)])
    N_rot /= np.maximum(np.linalg.norm(N_rot, axis=1, keepdims=True), 1e-12)
    N_orig = N_rot @ plane.rotation

    # 6. Transform grid vertices back to original 3D coordinates
    V_rot = np.column_stack([grid_xy, gz])
    V_orig = V_rot @ plane.rotation + center

    # 7. Triangulation: Shorter 3D Diagonal Delaunay Quad Splitting
    # Eliminates directional diagonal banding and produces an aesthetic, uniform wireframe
    # that naturally hugs ridgelines, roads, and structure steps.
    r_idx = np.arange(ny - 1)[:, None]
    c_idx = np.arange(nx - 1)[None, :]

    i0 = (r_idx * nx + c_idx).ravel()
    i1 = (r_idx * nx + (c_idx + 1)).ravel()
    i2 = ((r_idx + 1) * nx + c_idx).ravel()
    i3 = ((r_idx + 1) * nx + (c_idx + 1)).ravel()

    # Valid quad when all 4 corner nodes have support
    valid_quad = valid_node[i0] & valid_node[i1] & valid_node[i2] & valid_node[i3]

    if np.any(valid_quad):
        i0_v, i1_v, i2_v, i3_v = i0[valid_quad], i1[valid_quad], i2[valid_quad], i3[valid_quad]
        d03_sq = np.sum((V_orig[i0_v] - V_orig[i3_v]) ** 2, axis=1)
        d12_sq = np.sum((V_orig[i1_v] - V_orig[i2_v]) ** 2, axis=1)
        use_03 = d03_sq < d12_sq

        t1_a = np.column_stack([i0_v[use_03], i1_v[use_03], i3_v[use_03]])
        t1_b = np.column_stack([i0_v[use_03], i3_v[use_03], i2_v[use_03]])
        t2_a = np.column_stack([i0_v[~use_03], i1_v[~use_03], i2_v[~use_03]])
        t2_b = np.column_stack([i1_v[~use_03], i3_v[~use_03], i2_v[~use_03]])

        faces_arr = np.vstack([t1_a, t1_b, t2_a, t2_b]).astype(np.int32)
    else:
        # Fallback for sparse bounds
        faces = []
        for r in range(ny - 1):
            for c in range(nx - 1):
                j0 = r * nx + c
                j1 = r * nx + (c + 1)
                j2 = (r + 1) * nx + c
                j3 = (r + 1) * nx + (c + 1)
                faces.append([j0, j1, j2])
                faces.append([j1, j3, j2])
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

    # 8. High-Resolution Direct Point-Cloud Texture Atlas (No Downsampling Blur)
    tex_res = min(max(texture_size, 512), 8192)
    if colors is not None and len(colors) == len(points):
        u_pts = (pts_rot[:, 0] - x_min) / max(dx, 1e-6)
        v_pts = (pts_rot[:, 1] - y_min) / max(dy, 1e-6)
        valid_pts = (u_pts >= 0.0) & (u_pts <= 1.0) & (v_pts >= 0.0) & (v_pts <= 1.0)

        col_pts = np.clip((u_pts[valid_pts] * (tex_res - 1)).astype(np.int32), 0, tex_res - 1)
        row_pts = np.clip(((1.0 - v_pts[valid_pts]) * (tex_res - 1)).astype(np.int32), 0, tex_res - 1)
        lin_idx = row_pts * tex_res + col_pts
        val_colors = colors[valid_pts]

        count = np.bincount(lin_idx, minlength=tex_res * tex_res)
        has_sample = count > 0

        r_acc = np.bincount(lin_idx, weights=val_colors[:, 0], minlength=tex_res * tex_res)
        g_acc = np.bincount(lin_idx, weights=val_colors[:, 1], minlength=tex_res * tex_res)
        b_acc = np.bincount(lin_idx, weights=val_colors[:, 2], minlength=tex_res * tex_res)

        tex_flat = np.full((tex_res * tex_res, 3), 160, dtype=np.uint8)
        tex_flat[has_sample, 0] = np.clip(r_acc[has_sample] / count[has_sample], 0, 255).astype(np.uint8)
        tex_flat[has_sample, 1] = np.clip(g_acc[has_sample] / count[has_sample], 0, 255).astype(np.uint8)
        tex_flat[has_sample, 2] = np.clip(b_acc[has_sample] / count[has_sample], 0, 255).astype(np.uint8)

        tex_img = tex_flat.reshape((tex_res, tex_res, 3))
        tex_mask = (has_sample.reshape((tex_res, tex_res))).astype(np.uint8) * 255

        # Multi-scale full-resolution dilation: NEVER downsamples to 256x256,
        # ensuring 100% 4K/8K razor-sharp texture clarity across middle and edge areas.
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil1 = cv2.dilate(tex_img, kernel5, iterations=2)
        tex_filled = np.where(tex_mask[:, :, None] > 0, tex_img, dil1)

        mask_dil1 = cv2.dilate(tex_mask, kernel5, iterations=2)
        if int((mask_dil1 == 0).sum()) > 0:
            kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dil2 = cv2.dilate(tex_filled, kernel7, iterations=3)
            tex_img = np.where(mask_dil1[:, :, None] > 0, tex_filled, dil2)
        else:
            tex_img = tex_filled
    else:
        tex_img = np.full((tex_res, tex_res, 3), 180, dtype=np.uint8)

    # In OBJ/glTF conventions, v=0 maps to image bottom and v=1 to image top.
    # Since row 0 is v=1 and row tex_res-1 is v=0, tex_img already matches UV coordinates directly.
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

