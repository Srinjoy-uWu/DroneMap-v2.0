"""Validation harness against the Blender ground-truth fixture (WP9).

What this measures, and what each number is allowed to claim
-----------------------------------------------------------
Every accuracy figure here names the transform applied before it was taken,
because that is what decides its meaning. The previous version of this script
fitted a Sim(3) to the trajectory, applied it to the dense cloud, and printed
the result as "3D Surface Geometry" - silently removing the scale and
georeferencing error it claimed to be measuring.

So results come as an explicit ladder, and every removed quantity is disclosed:

    georeferenced       the run brought into the fixture's frame by *geodesy
                        alone* - undo the local-origin shift the georef stage
                        applied, then ECEF -> ENU about the published datum.
                        Both steps are closed-form frame changes with nothing
                        fitted to ground truth, so this is the real accuracy of
                        the product, and it is the headline number.
    - mean bias         the constant offset removed. Rigid: changes no distance,
                        no dimension, no scale. A diagnostic that separates a
                        systematic datum shift from random error.
    - yaw               plus grid north vs true north. Also rigid. The fitted
                        yaw is printed next to the CRS's own grid convergence,
                        so a real orientation bug cannot hide behind a
                        legitimate ~1 deg convergence.
    + Sim(3)            scale and full pose fitted to ground truth. NOT
                        accuracy. Reported only so a scale error can be told
                        apart from a shape error.

Dimensions and marker separations are measured from the reconstruction, never
echoed from ground truth, and are taken in the unfitted georeferenced frame, so
they remain independent evidence about scale.

Usage
-----
    python scripts/evaluate_synthetic.py --run-dir data/runs/fixture_orbit_v4
    python scripts/evaluate_synthetic.py --run-dir <run> --fixture-dir <fixture>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d

# How close a reconstructed surface has to be for the run to count as validated.
# Standalone GNSS on a consumer drone is a 1-3 m instrument; metre-level is the
# honest bar, and anything tighter would be claiming accuracy the sensor cannot
# deliver.
GEOREF_RMSE_PASS_M = 3.0
SHAPE_SCALE_PASS_PCT = 2.0
# Above this, two point sets are not the same frame at all (a projected CRS
# against local metres), so the raw RMSE is a frame mismatch and not an error.
CRS_OFFSET_M = 1_000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) scale, rotation, translation aligning X to Y: Y ~= s * R @ X + t."""
    n, m = X.shape
    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    sigma_x2 = float(np.var(X, axis=0).sum())
    Sigma_yx = (Y - mu_y).T @ (X - mu_x) / n

    U, D, Vt = np.linalg.svd(Sigma_yx)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S)) / max(sigma_x2, 1e-9)
    t = mu_y - scale * R @ mu_x
    return float(scale), R, t


def fit_yaw_translation(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Rigid yaw-about-vertical plus translation carrying X onto Y.

    Deliberately *not* a general rotation and *not* scaled. A projected CRS
    differs from a true-north local frame by a grid convergence angle about the
    vertical only; allowing tilt or scale here would quietly absorb real
    reconstruction error.
    """
    mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
    a = X[:, :2] - mu_x[:2]
    b = Y[:, :2] - mu_y[:2]
    yaw = math.atan2(
        float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
        float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])),
    )
    R = rot_z(yaw)
    return yaw, mu_y - R @ mu_x


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def grid_convergence_deg(crs: str, lat_deg: float, lon_deg: float) -> float | None:
    """Angle between projected-grid north and true north at a point.

    Printed beside the fitted yaw so the reader can tell a legitimate frame
    convention apart from a georeferencing bug. Only meaningful for a projected
    CRS; ENU and ECEF are true-north frames and return None.
    """
    m = re.fullmatch(r"EPSG:32[67](\d{2})", (crs or "").strip().upper())
    if not m:
        return None
    zone = int(m.group(1))
    if not 1 <= zone <= 60:
        return None
    central_meridian = 6.0 * zone - 183.0
    d_lon = math.radians(lon_deg - central_meridian)
    return math.degrees(math.atan(math.tan(d_lon) * math.sin(math.radians(lat_deg))))


# WGS-84
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def wgs84_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return np.array([
        (N + h_m) * cos_lat * math.cos(lon),
        (N + h_m) * cos_lat * math.sin(lon),
        (N * (1.0 - _WGS84_E2) + h_m) * sin_lat,
    ])


def ecef_to_enu(pts_ecef: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Exact geodetic ECEF -> local ENU about a datum.

    Nothing here is fitted to ground truth: it is the closed-form change of
    basis between two published frames. So a residual measured after this
    conversion is still a full accuracy claim - unlike a fitted translation,
    which at least has to be disclosed.
    """
    lat, lon = math.radians(lat0), math.radians(lon0)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    R = np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    return (R @ (pts_ecef - wgs84_to_ecef(lat0, lon0, alt0)).T).T


# ---------------------------------------------------------------------------
# Reading COLMAP models (binary or text)
# ---------------------------------------------------------------------------

def _centers_from_text(images_file: Path) -> dict[int, np.ndarray]:
    """Parse images.txt and compute camera centres C = -R^T * T."""
    centers: dict[int, np.ndarray] = {}
    valid_exts = (".jpg", ".jpeg", ".png")
    with open(images_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and any(parts[-1].lower().endswith(e) for e in valid_exts):
                name = parts[-1]
                qw, qx, qy, qz = (float(p) for p in parts[1:5])
                tx, ty, tz = (float(p) for p in parts[5:8])
                R_cam = np.array([
                    [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
                ])
                C = -R_cam.T @ np.array([tx, ty, tz])
                m = re.search(r"\d+", name)
                if m:
                    centers[int(m.group(0))] = C
    return centers


def read_camera_centers(model_dir: Path) -> tuple[dict[int, np.ndarray], str]:
    """Camera centres keyed by source frame index, from a text or binary model.

    The georef stage writes only .bin, so a text-only reader silently reports
    "no georeferenced trajectory" on a run that has one - which would downgrade
    a good run to an unmeasured one.
    """
    if not model_dir.exists():
        return {}, "missing"

    for candidate in (model_dir / "images.txt", model_dir / "txt" / "images.txt"):
        if candidate.exists():
            return _centers_from_text(candidate), f"text:{candidate.name}"

    if (model_dir / "images.bin").exists():
        try:
            import pycolmap

            rec = pycolmap.Reconstruction(str(model_dir))
            centers: dict[int, np.ndarray] = {}
            for image in rec.images.values():
                m = re.search(r"\d+", image.name)
                if m:
                    centers[int(m.group(0))] = np.asarray(image.projection_center(), dtype=float)
            return centers, "binary:pycolmap"
        except Exception as exc:  # pragma: no cover - environment dependent
            return {}, f"binary:unreadable ({exc})"

    return {}, "no model"


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


# ---------------------------------------------------------------------------
# Measurement: object dimensions
# ---------------------------------------------------------------------------

def measure_object(
    pts: np.ndarray,
    centre_xy: tuple[float, float],
    gt_w: float,
    gt_l: float,
    gt_h: float,
) -> dict | None:
    """Measure one object's extent from the reconstruction.

    Ground truth supplies only *where to look* - the centre - never the answer.
    Height is taken above a ground level estimated from a ring around the
    object, so a global vertical datum offset cannot flatter it.
    """
    cx, cy = centre_xy
    footprint = max(gt_w, gt_l)

    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    radial = np.hypot(dx, dy)

    ring = (radial > footprint * 0.5 + 3.0) & (radial < footprint * 0.5 + 12.0)
    if int(ring.sum()) < 50:
        return None
    ground_z = float(np.percentile(pts[ring, 2], 5))

    margin = 1.5
    box = (np.abs(dx) <= gt_w * 0.5 + margin) & (np.abs(dy) <= gt_l * 0.5 + margin)
    above = box & (pts[:, 2] > ground_z + max(1.0, 0.30 * gt_h))
    n_above = int(above.sum())
    if n_above < 80:
        return None

    obj = pts[above]
    # Robust percentiles, not min/max: one flyaway point would otherwise set
    # the dimension.
    meas_w = float(np.percentile(obj[:, 0], 98) - np.percentile(obj[:, 0], 2))
    meas_l = float(np.percentile(obj[:, 1], 98) - np.percentile(obj[:, 1], 2))
    meas_h = float(np.percentile(obj[:, 2], 99) - ground_z)

    return {
        "gt_width_m": round(gt_w, 3),
        "gt_length_m": round(gt_l, 3),
        "gt_height_m": round(gt_h, 3),
        "measured_width_m": round(meas_w, 3),
        "measured_length_m": round(meas_l, 3),
        "measured_height_m": round(meas_h, 3),
        "width_error_m": round(meas_w - gt_w, 3),
        "length_error_m": round(meas_l - gt_l, 3),
        "height_error_m": round(meas_h - gt_h, 3),
        "n_points": n_above,
        "local_ground_z": round(ground_z, 3),
    }


def measure_reference_distances(
    pts: np.ndarray,
    markers: dict[str, list[float]],
    gt_distances: list[dict],
) -> tuple[list[dict], dict | None]:
    """Recover marker centroids from the cloud and check their separations.

    This is the scale check that needs no RTK: the separations are surveyed
    quantities, and they are compared in a rigid frame, so nothing fitted to
    ground truth can have absorbed a scale error.
    """
    found: dict[str, np.ndarray] = {}
    for name, marker_xyz in markers.items():
        mx, my = float(marker_xyz[0]), float(marker_xyz[1])
        near = np.hypot(pts[:, 0] - mx, pts[:, 1] - my) < 4.0
        if int(near.sum()) < 30:
            continue
        local = pts[near]
        ground_z = float(np.percentile(local[:, 2], 20))
        plate = local[np.abs(local[:, 2] - ground_z) < 1.5]
        if len(plate) < 20:
            continue
        found[name] = plate.mean(axis=0)

    results: list[dict] = []
    for entry in gt_distances:
        a, b = entry["from"], entry["to"]
        if a not in found or b not in found:
            continue
        measured = float(np.linalg.norm(found[a] - found[b]))
        gt = float(entry["distance_m"])
        results.append({
            "from": a, "to": b,
            "gt_distance_m": round(gt, 3),
            "measured_distance_m": round(measured, 3),
            "error_m": round(measured - gt, 3),
            "error_pct": round((measured - gt) / gt * 100.0, 3) if gt else None,
        })

    summary = None
    if results:
        errs = np.array([r["error_pct"] for r in results if r["error_pct"] is not None])
        abs_m = np.array([abs(r["error_m"]) for r in results])
        summary = {
            "n_pairs": len(results),
            "n_markers_recovered": len(found),
            "mean_signed_scale_error_pct": round(float(errs.mean()), 3) if len(errs) else None,
            "max_abs_error_m": round(float(abs_m.max()), 3),
        }
    return results, summary


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_run(run_dir: Path, fixture_dir: Path) -> dict:
    print("\n" + "=" * 72)
    print(f"[*] GROUND-TRUTH EVALUATION: {run_dir.name}")
    print(f"    fixture: {fixture_dir}")
    print("=" * 72)

    gt_mesh_path = fixture_dir / "gt_mesh.ply"
    gt_dims_path = fixture_dir / "gt_dimensions.json"

    est_georef, georef_src = read_camera_centers(run_dir / "03b_georef" / "sparse_enu")
    est_raw, raw_src = read_camera_centers(run_dir / "03_pose" / "sparse" / "0")

    gt_dict: dict[int, dict] = {}
    gt_poses_path = fixture_dir / "gt_poses.json"
    if gt_poses_path.exists():
        for k, v in json.loads(gt_poses_path.read_text(encoding="utf-8")).items():
            m = re.search(r"\d+", k)
            if m:
                gt_dict[int(m.group(0))] = v

    # CRS and datum. Needed both to bring the run into the fixture's frame by
    # pure geodesy and to say whether a fitted yaw is a legitimate convention.
    crs = ""
    coordinate_frame = ""
    pipeline_rmse = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gm = manifest.get("stages", {}).get("georef", {}).get("metrics", {}) or {}
        crs = str(gm.get("crs", "") or "")
        pipeline_rmse = gm.get("alignment_rmse_m")
    tj = run_dir / "03b_georef" / "sparse_enu" / "transform.json"
    model_offset: np.ndarray | None = None
    if tj.exists():
        tdata = json.loads(tj.read_text(encoding="utf-8"))
        coordinate_frame = str(tdata.get("coordinate_frame", "") or "")
        crs = crs or str(tdata.get("crs", "") or "")
        off = tdata.get("model_offset_m")
        if off:
            model_offset = np.asarray(off, dtype=float)

    datum = {}
    fixture_meta = {}
    if (fixture_dir / "fixture.json").exists():
        fixture_meta = json.loads((fixture_dir / "fixture.json").read_text(encoding="utf-8"))
        datum = fixture_meta.get("reference_datum") or {}

    traj: dict = {
        "n_gt_cameras": len(gt_dict),
        "georef_model_source": georef_src,
        "relative_model_source": raw_src,
        "crs": crs or None,
        "coordinate_frame": coordinate_frame or None,
        "pipeline_reported_alignment_rmse_m": pipeline_rmse,
    }
    sim3: tuple[float, np.ndarray, np.ndarray] | None = None
    # A second Sim(3), fitted in the *georeferenced* frame. `sim3` above is
    # fitted on the pre-georef model, whose arbitrary COLMAP frame the dense
    # cloud is not in -- applying it to the georeferenced dense cloud mixes two
    # frames and reported a meaningless 731 m. A shape figure for the surface
    # must use the transform fitted in the surface's own frame.
    sim3_geo: tuple[float, np.ndarray, np.ndarray] | None = None
    # Exact frame change applied to the run before any comparison, if any.
    geodetic: dict | None = None

    if est_georef and gt_dict:
        matched = sorted(k for k in est_georef if k in gt_dict)
        if len(matched) >= 4:
            P_est = np.array([est_georef[k] for k in matched])
            P_gt = np.array([[gt_dict[k]["x"], gt_dict[k]["y"], gt_dict[k]["z"]] for k in matched])
            traj["n_matched_georeferenced"] = len(matched)

            # Bring the run into the fixture's frame by geodesy alone. The georef
            # stage emits geocentric metres (EPSG:4978) shifted to a local origin
            # so OpenMVS's float32 geometry can hold them; undoing that shift and
            # converting ECEF->ENU fits nothing, so the residual afterwards is a
            # real accuracy figure - not something a Sim(3) massaged into shape.
            if coordinate_frame == "ECEF" and datum:
                lat0 = float(datum["lat"]); lon0 = float(datum["lon"])
                alt0 = float(datum.get("alt_m", 0.0))
                if model_offset is not None:
                    P_est = P_est + model_offset
                P_est = ecef_to_enu(P_est, lat0, lon0, alt0)
                geodetic = {
                    "applied": "un-shift local origin, then ECEF (EPSG:4978) -> "
                               "local ENU about the fixture datum",
                    "model_offset_m": model_offset.tolist() if model_offset is not None else None,
                    "datum": {"lat": lat0, "lon": lon0, "alt_m": alt0},
                    "fitted": False,
                }
                traj["geodetic_conversion"] = geodetic

            traj["rmse_georeferenced_m"] = round(_rmse(P_est, P_gt), 3)
            traj["mean_abs_error_georeferenced_m"] = round(
                float(np.mean(np.linalg.norm(P_est - P_gt, axis=1))), 3)

            bias = P_gt.mean(axis=0) - P_est.mean(axis=0)
            traj["mean_bias_enu_m"] = [round(float(v), 3) for v in bias]
            traj["mean_bias_magnitude_m"] = round(float(np.linalg.norm(bias)), 3)
            traj["frames_differ_by_crs"] = bool(np.linalg.norm(bias) > CRS_OFFSET_M)
            traj["rmse_after_translation_m"] = round(_rmse(P_est + bias, P_gt), 3)

            yaw, t_rigid = fit_yaw_translation(P_est, P_gt)
            R_rigid = rot_z(yaw)
            traj["rmse_after_yaw_translation_m"] = round(
                _rmse((R_rigid @ P_est.T).T + t_rigid, P_gt), 3)
            traj["fitted_yaw_deg"] = round(math.degrees(yaw), 4)
            conv = grid_convergence_deg(
                crs, float(datum.get("lat", 0.0)), float(datum.get("lon", 0.0))) if datum else None
            traj["grid_convergence_deg"] = round(conv, 4) if conv is not None else None
            traj["grid_convergence_note"] = (
                "n/a - ENU/ECEF are true-north frames, so any fitted yaw is real error"
                if conv is None else
                "projected CRS: this much yaw is a frame convention, not an error")

            # Fitted in the georeferenced frame, for the dense surface to use.
            s_g, R_g, t_g = umeyama_alignment(P_est, P_gt)
            sim3_geo = (s_g, R_g, t_g)
            traj["sim3_scale_factor_georeferenced"] = round(s_g, 6)

    poses_for_shape = est_raw or est_georef
    if poses_for_shape and gt_dict:
        matched = sorted(k for k in poses_for_shape if k in gt_dict)
        if len(matched) >= 4:
            P_est = np.array([poses_for_shape[k] for k in matched])
            P_gt = np.array([[gt_dict[k]["x"], gt_dict[k]["y"], gt_dict[k]["z"]] for k in matched])
            scale, R_a, t_a = umeyama_alignment(P_est, P_gt)
            sim3 = (scale, R_a, t_a)
            traj["n_matched_relative"] = len(matched)
            traj["rmse_after_sim3_m"] = round(_rmse((scale * (R_a @ P_est.T)).T + t_a, P_gt), 3)
            traj["sim3_scale_factor"] = round(scale, 6)

    print("\n-- Trajectory ---------------------------------------------------------")
    print(f"   models            : georef={georef_src}  relative={raw_src}")
    print(f"   frame             : {coordinate_frame or '?'} ({crs or '?'})")
    if geodetic:
        print(f"   geodetic step     : {geodetic['applied']}  [exact, nothing fitted]")
    print(f"   cameras matched   : "
          f"{max(traj.get('n_matched_georeferenced', 0), traj.get('n_matched_relative', 0))}"
          f" / {len(gt_dict)}")
    print(f"   [ACCURACY] RMSE   : {traj.get('rmse_georeferenced_m')} m   "
          f"<-- nothing fitted to ground truth")
    print(f"   [ACCURACY] mean   : {traj.get('mean_abs_error_georeferenced_m')} m")
    print(f"   pipeline reported : {pipeline_rmse} m  (COLMAP model_aligner)")
    print(f"   mean bias (E,N,U) : {traj.get('mean_bias_enu_m')} m  "
          f"(|{traj.get('mean_bias_magnitude_m')}| m)")
    print(f"   [diag] -bias      : {traj.get('rmse_after_translation_m')} m   "
          f"(rigid; scale untouched)")
    print(f"   [diag] -bias -yaw : {traj.get('rmse_after_yaw_translation_m')} m   "
          f"(yaw {traj.get('fitted_yaw_deg')} deg; grid convergence "
          f"{traj.get('grid_convergence_deg')})")
    print(f"   [SHAPE]  +Sim(3)  : {traj.get('rmse_after_sim3_m')} m   "
          f"(scale fitted - NOT accuracy)")

    # --- surface geometry -------------------------------------------------
    #
    # Accuracy and completeness are separate measurements and must not be
    # averaged into one "Chamfer" figure.
    #
    # The earlier symmetric Chamfer reported 16.3 m on a run whose trajectory
    # was accurate to 1.65 m and whose seven buildings measured within 0.7 m --
    # an internal contradiction that traced entirely to the *ground truth* being
    # bigger than the flight. gt_mesh.ply spans 300x300 m; this orbit observed
    # 32.8% of it. Every GT sample over ground the drone never flew past has no
    # nearby reconstructed point, so the GT->est direction measured "we did not
    # fly there", and averaging that into a number labelled [ACCURACY] read as
    # "the surface is 16 m wrong". It was not: est->GT alone is 1.05 m.
    #
    # So, following ETH3D / Tanks-and-Temples practice:
    #   accuracy      est -> GT, over all reconstructed points. How wrong is
    #                 what we built? Reported as median and p90 as well as mean,
    #                 because a handful of flyaway points must not set the
    #                 headline.
    #   completeness  GT -> est, restricted to the footprint actually observed,
    #                 with the observed fraction stated next to it so the
    #                 restriction can never be mistaken for full coverage.
    surface: dict = {
        "dense_points": None,
        "accuracy_georeferenced_m": None,          # est -> GT, mean
        "accuracy_median_m": None,
        "accuracy_p90_m": None,
        "gt_area_observed_pct": None,
        "completeness_mean_m_observed_area": None,
        "completeness_25cm_pct_observed_area": None,
        "completeness_50cm_pct_observed_area": None,
        "completeness_25cm_pct_debiased": None,    # diagnostic, labelled
        "chamfer_shape_only_m": None,
        "frame_used_for_measurement": None,
        "mesh": None,
    }
    pts_measure: np.ndarray | None = None
    measure_frame = None

    dense_ply = _first_existing(
        run_dir / "05_dense" / "scene_dense.ply",
        run_dir / "05_dense" / "scene_dense_mesh_refine.ply",
    )
    if dense_ply and gt_mesh_path.exists():
        pcd_est = o3d.io.read_point_cloud(str(dense_ply))
        pts_est = np.asarray(pcd_est.points)
        pts_est = pts_est[np.isfinite(pts_est).all(axis=1)]
        surface["dense_points"] = int(len(pts_est))

        gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))
        pcd_gt = gt_mesh.sample_points_uniformly(number_of_points=200_000)
        gt_pts = np.asarray(pcd_gt.points)

        def observed_mask(points: np.ndarray, cell: float = 4.0) -> np.ndarray:
            """GT samples lying in a ground cell the reconstruction covers.

            A 4 m occupancy grid over the reconstructed cloud's plan view. Coarse
            on purpose: this decides *where the flight looked*, not how good the
            surface is, and a fine grid would start excluding genuine holes --
            which are completeness failures that must stay counted.
            """
            occupied = set(map(tuple, np.floor(points[:, :2] / cell).astype(int)))
            keys = np.floor(gt_pts[:, :2] / cell).astype(int)
            return np.fromiter((tuple(k) in occupied for k in keys),
                               dtype=bool, count=len(keys))

        if len(pts_est) > 500:
            def distances(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(points)
                return (np.asarray(pc.compute_point_cloud_distance(pcd_gt)),   # est -> GT
                        np.asarray(pcd_gt.compute_point_cloud_distance(pc)))   # GT  -> est

            # Accuracy: the dense cloud gets exactly the same *exact* geodetic
            # step the trajectory got, and nothing else. If georeferencing never
            # ran there is no such number, and none is invented.
            if geodetic is not None:
                d = geodetic["datum"]
                pts_geo = pts_est + model_offset if model_offset is not None else pts_est
                pts_geo = ecef_to_enu(pts_geo, d["lat"], d["lon"], d["alt_m"])
                d_est, d_gt = distances(pts_geo)
                inside = observed_mask(pts_geo)

                surface["accuracy_georeferenced_m"] = round(float(d_est.mean()), 3)
                surface["accuracy_median_m"] = round(float(np.median(d_est)), 3)
                surface["accuracy_p90_m"] = round(float(np.percentile(d_est, 90)), 3)
                surface["gt_area_observed_pct"] = round(float(inside.mean() * 100.0), 1)
                if int(inside.sum()) > 0:
                    obs = d_gt[inside]
                    surface["completeness_mean_m_observed_area"] = round(float(obs.mean()), 3)
                    surface["completeness_25cm_pct_observed_area"] = round(float(np.mean(obs <= 0.25) * 100.0), 1)
                    surface["completeness_50cm_pct_observed_area"] = round(float(np.mean(obs <= 0.50) * 100.0), 1)

                # Diagnostic: the same completeness after removing the constant
                # georeferencing offset. Tight thresholds are otherwise dominated
                # by a ~1.4 m standalone-GNSS bias, which says nothing about how
                # much surface detail was recovered. Rigid, and labelled as
                # fitted wherever it is printed.
                #
                # `mean_bias_enu_m` is stored as the *correction* (GT mean minus
                # estimate mean), so it is added, matching how
                # `rmse_after_translation_m` applies it.
                if traj.get("mean_bias_enu_m"):
                    shifted = pts_geo + np.asarray(traj["mean_bias_enu_m"], dtype=float)
                    _, d_gt_deb = distances(shifted)
                    ins_deb = observed_mask(shifted)
                    if int(ins_deb.sum()) > 0:
                        surface["completeness_25cm_pct_debiased"] = round(
                            float(np.mean(d_gt_deb[ins_deb] <= 0.25) * 100.0), 1)

                pts_measure = pts_geo
                measure_frame = ("georeferenced, un-shifted and ECEF->ENU by geodesy only "
                                 "(nothing fitted - the dimensions below are independent "
                                 "evidence about scale)")

            # Shape only. Labelled as such everywhere it appears.
            #
            # The transform and the points must come from the same frame. A
            # Sim(3) fitted on the pre-georef model does not describe the
            # georeferenced dense cloud, and pairing them produced a meaningless
            # 731 m. So each frame uses its own fit, and when no fit exists in
            # the cloud's frame no number is invented.
            shape_fit = sim3_geo if geodetic is not None else sim3
            shape_src = pts_geo if geodetic is not None else pts_est
            if shape_fit is not None:
                scale, R_a, t_a = shape_fit
                d_est_s, _ = distances((scale * (R_a @ shape_src.T)).T + t_a)
                surface["chamfer_shape_only_m"] = round(float(d_est_s.mean()), 3)
                if pts_measure is None:
                    pts_measure = (scale * (R_a @ shape_src.T)).T + t_a
                    measure_frame = ("Sim(3)-aligned to GT (scale FITTED - dimensions here "
                                     "are NOT independent evidence about scale)")

        surface["frame_used_for_measurement"] = measure_frame

    # Report on the *delivered* asset first. A judge receives 07_export/model.glb,
    # not OpenMVS's working PLYs, and the working PLYs carry no texture -- so
    # reading them made a fully textured deliverable look like bare geometry.
    glb_path = run_dir / "07_export" / "model.glb"
    if glb_path.exists():
        try:
            import trimesh

            scene = trimesh.load(str(glb_path))
            geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
            verts = sum(len(g.vertices) for g in geoms)
            tris = sum(len(g.faces) for g in geoms)
            has_uv = any(getattr(g.visual, "uv", None) is not None for g in geoms)
            textured = any(
                getattr(getattr(g.visual, "material", None), "baseColorTexture", None) is not None
                for g in geoms)
            all_v = np.vstack([np.asarray(g.vertices) for g in geoms])
            surface["mesh"] = {
                "path": "07_export/model.glb",
                "vertices": int(verts),
                "triangles": int(tris),
                "has_uvs": bool(has_uv),
                "has_texture": bool(textured),
                "extent_m": [round(float(v), 2) for v in (all_v.max(axis=0) - all_v.min(axis=0))],
                "max_abs_coord_m": round(float(np.abs(all_v).max()), 2),
                # The reason a georeferenced mesh must be delivered in a local
                # frame: glTF stores positions as float32, whose spacing at
                # geocentric magnitude (5.5e6) is 0.5 m.
                "float32_spacing_m": float(np.spacing(np.float32(np.abs(all_v).max()))),
            }
        except Exception as exc:  # noqa: BLE001 - report, never fail the eval on it
            surface["mesh"] = {"path": "07_export/model.glb", "error": str(exc)}

    if surface["mesh"] is None:
        mesh_ply = _first_existing(
            run_dir / "06_mesh" / "mesh_textured.ply",
            run_dir / "06_mesh" / "scene_dense_mesh_refine_texture.ply",
            run_dir / "06_mesh" / "scene_dense_mesh_clean.ply",
            run_dir / "06_mesh" / "scene_dense_mesh.ply",
            run_dir / "06_mesh" / "mesh.ply",
        )
        if mesh_ply:
            m = o3d.io.read_triangle_mesh(str(mesh_ply))
            surface["mesh"] = {
                "path": mesh_ply.name,
                "vertices": int(len(m.vertices)),
                "triangles": int(len(m.triangles)),
                "has_vertex_colors": bool(len(m.vertex_colors) > 0),
                "has_uvs": bool(len(m.triangle_uvs) > 0),
            }

    print("\n-- Dense surface ------------------------------------------------------")
    print(f"   dense points      : {surface['dense_points']}")
    print(f"   [ACCURACY] est->GT: mean {surface['accuracy_georeferenced_m']} m, "
          f"median {surface['accuracy_median_m']} m, p90 {surface['accuracy_p90_m']} m"
          f"   <-- nothing fitted")
    print(f"   GT area observed  : {surface['gt_area_observed_pct']} % of the ground-truth "
          f"mesh (the rest was never flown over)")
    print(f"   completeness      : mean {surface['completeness_mean_m_observed_area']} m, "
          f"<25 cm {surface['completeness_25cm_pct_observed_area']} %, "
          f"<50 cm {surface['completeness_50cm_pct_observed_area']} %"
          f"   (within observed area)")
    if surface["completeness_25cm_pct_debiased"] is not None:
        print(f"   [diag] <25 cm     : {surface['completeness_25cm_pct_debiased']} % after "
              f"removing the constant georeferencing bias (rigid; fitted)")
    print(f"   [shape]  est->GT  : {surface['chamfer_shape_only_m']} m  (Sim(3) fitted - NOT accuracy)")
    if surface["mesh"]:
        mi = surface["mesh"]
        if mi.get("error"):
            print(f"   mesh              : {mi['path']} unreadable - {mi['error']}")
        else:
            print(f"   mesh ({mi['path']}): {mi['vertices']} verts, {mi['triangles']} tris, "
                  f"uvs={mi.get('has_uvs')}, textured={mi.get('has_texture', mi.get('has_vertex_colors'))}")
            if mi.get("extent_m"):
                print(f"   mesh extent       : {mi['extent_m']} m (E,up,-N), "
                      f"max |coord| {mi['max_abs_coord_m']} m -> float32 spacing "
                      f"{mi['float32_spacing_m']:.6f} m")

    # --- measured dimensions and surveyed distances -----------------------
    dimensions: list[dict] = []
    ref_distances: list[dict] = []
    ref_summary = None
    gt_dims_data: dict = {}
    if gt_dims_path.exists():
        gt_dims_data = json.loads(gt_dims_path.read_text(encoding="utf-8"))

    if pts_measure is not None and gt_dims_data:
        for obj in gt_dims_data.get("ground_truth_objects", []):
            centre = obj.get("centre_xyz_m")
            if not centre:
                continue
            m = measure_object(
                pts_measure, (float(centre[0]), float(centre[1])),
                float(obj["width_x_m"]), float(obj["length_y_m"]), float(obj["height_z_m"]),
            )
            row: dict = {"name": obj["name"]}
            if m is None:
                row.update({
                    "measured": False,
                    "reason": "too few reconstructed points above local ground",
                    "gt_width_m": obj["width_x_m"],
                    "gt_length_m": obj["length_y_m"],
                    "gt_height_m": obj["height_z_m"],
                })
            else:
                row.update({"measured": True, **m})
            dimensions.append(row)

        ref_distances, ref_summary = measure_reference_distances(
            pts_measure,
            gt_dims_data.get("reference_markers", {}),
            gt_dims_data.get("reference_distances", []),
        )

    print("\n-- Measured dimensions ------------------------------------------------")
    print(f"   frame: {measure_frame or 'n/a - no dense cloud to measure'}")
    for row in dimensions:
        if row.get("measured"):
            print(f"   {row['name']:<16} "
                  f"W {row['measured_width_m']:>6.2f}/{row['gt_width_m']:<5.1f} "
                  f"L {row['measured_length_m']:>6.2f}/{row['gt_length_m']:<5.1f} "
                  f"H {row['measured_height_m']:>6.2f}/{row['gt_height_m']:<5.1f}  "
                  f"err {row['width_error_m']:+.2f}/{row['length_error_m']:+.2f}/"
                  f"{row['height_error_m']:+.2f} m")
        else:
            print(f"   {row['name']:<16} not measurable - {row.get('reason')}")

    if ref_summary:
        print("\n-- Surveyed distances (RTK-free scale check) ---------------------------")
        for r in ref_distances:
            print(f"   {r['from']}->{r['to']:<8} "
                  f"{r['measured_distance_m']:>8.2f} / {r['gt_distance_m']:<8.2f} m  "
                  f"({r['error_pct']:+.2f} %)")
        print(f"   markers recovered {ref_summary['n_markers_recovered']}, "
              f"mean scale error {ref_summary['mean_signed_scale_error_pct']:+.2f} %")

    # --- verdict ----------------------------------------------------------
    # The tier is awarded on the *unfitted* number only.
    #
    # An earlier draft of this block graded on `rmse_after_yaw_translation_m`.
    # That transform is rigid, so it preserves every distance and the scale --
    # but it is still *fitted to ground truth*, and what it fits away is
    # absolute position and heading, which is exactly what a georeferencing
    # accuracy claim asserts. Grading on it would have reproduced the very
    # defect this harness was rewritten to remove: subtracting the error before
    # measuring it. The rigid figures stay below as diagnostics, where they
    # answer a different question (is this a systematic offset or scattered
    # noise?) without inflating the claim.
    geo_rmse = traj.get("rmse_georeferenced_m")
    if geo_rmse is not None and geo_rmse <= GEOREF_RMSE_PASS_M:
        tier = "validated_georeferenced_3d"
        tier_reason = (f"georeferenced trajectory RMSE {geo_rmse} m within "
                       f"{GEOREF_RMSE_PASS_M} m, measured after geodetic frame "
                       f"changes alone with nothing fitted to ground truth")
    elif geo_rmse is not None:
        tier = "georeferenced_but_outside_tolerance"
        tier_reason = (f"georeferenced trajectory RMSE {geo_rmse} m exceeds "
                       f"{GEOREF_RMSE_PASS_M} m (nothing fitted to ground truth)")
    elif traj.get("rmse_after_sim3_m") is not None:
        tier = "shape_validated_not_georeferenced"
        tier_reason = ("shape matches ground truth after Sim(3), but no "
                       "georeferenced trajectory was available to measure - no "
                       "metric accuracy claim is supported")
    else:
        tier = "not_validated"
        tier_reason = "no trajectory could be matched to ground truth"

    scorecard = {
        "run_id": run_dir.name,
        "fixture_dir": str(fixture_dir),
        "fixture": fixture_meta or None,
        "evaluation_tier": tier,
        "evaluation_tier_reason": tier_reason,
        "claim_semantics": {
            "raw": "no transform; meaningless across differing CRSs, flagged when so",
            "after_translation": "origin removed only; rigid, preserves every distance "
                                 "and the scale - a metric accuracy claim",
            "after_yaw_translation": "origin and grid-north convention removed; still "
                                     "rigid - a metric accuracy claim, with the fitted "
                                     "yaw disclosed against the CRS grid convergence",
            "shape_only_sim3": "scale and pose fitted to ground truth - shape fidelity "
                               "only, NOT an accuracy claim",
        },
        "trajectory": traj,
        "surface": surface,
        "dimensions": dimensions,
        "reference_distances": ref_distances,
        "reference_distance_summary": ref_summary,
        "thresholds": {
            "georef_rmse_pass_m": GEOREF_RMSE_PASS_M,
            "shape_scale_pass_pct": SHAPE_SCALE_PASS_PCT,
        },
    }

    # Into the run, not the fixture: two runs against one fixture must not
    # overwrite each other's scorecard.
    out_json = run_dir / "evaluation_scorecard.json"
    out_json.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"[SCORECARD] {out_json}")
    print(f"    tier: {tier}")
    print(f"    why : {tier_reason}")
    print("=" * 72)
    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="pipeline run directory to evaluate")
    parser.add_argument("--fixture-dir", default="",
                        help="ground-truth fixture directory (default: inferred from the "
                             "run manifest, else data/synthetic_flight)")
    args = parser.parse_args()

    run_path = Path(args.run_dir).resolve()
    if not run_path.exists():
        raise SystemExit(f"run directory not found: {run_path}")

    if args.fixture_dir:
        fixture_path = Path(args.fixture_dir).resolve()
    else:
        # Infer the fixture from the video the run was given, so evaluating the
        # corridor fixture cannot silently score it against the orbit's truth.
        fixture_path = Path("data/synthetic_flight").resolve()
        manifest_file = run_path / "manifest.json"
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            video = str(data.get("input", {}).get("video", "")
                        or data.get("inputs", {}).get("video", ""))
            if video and (Path(video).parent / "gt_poses.json").exists():
                fixture_path = Path(video).parent.resolve()

    if not (fixture_path / "gt_poses.json").exists():
        raise SystemExit(f"no ground truth in fixture directory: {fixture_path}")

    evaluate_run(run_path, fixture_path)
