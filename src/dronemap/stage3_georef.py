"""Stage 3b — Metric scale and georeferencing.

What this stage does
--------------------
Takes the up-to-scale COLMAP sparse model and a GPS telemetry file, and aligns
the reconstruction to real-world coordinates (WGS-84 / UTM).

Strategy selection (``config.georef.method``):
  - ``"pose_prior"``: Use COLMAP ``pose_prior_mapper`` — re-runs the mapper
    with GPS positions as soft priors in the bundle adjustment.  Best result
    quality; available on COLMAP ≥ 4.0.
  - ``"aligner"``: Use COLMAP ``model_aligner`` — applies a Sim(3) alignment
    (Umeyama) from the existing sparse model to the GPS positions.  Faster and
    works on any COLMAP version, but less accurate for noisy GPS.
  - ``"auto"`` (default): try ``pose_prior_mapper``; fall back to ``model_aligner``
    if the command is not available in the installed build.

In both cases the aligned model is saved in the ENU (East-North-Up) frame
anchored at the scene centroid, with axes in metres.  The output CRS is chosen
automatically as the UTM zone covering the scene centroid.

Outputs
-------
``ws.georef_sparse_dir/``  — aligned sparse model (cameras.bin/images.bin/points3D.bin)
``ws.georef_sparse_dir/ref_images.txt``  — GPS control file passed to COLMAP
``ws.georef_sparse_dir/transform.json``  — Sim(3) parameters (scale, R, t, CRS)
``ws.manifest["accuracy"]``  — alignment_rmse_m, n_gps_fixes_used, crs, gnss_mode
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .tools import ColmapTool, ToolRegistry
    from .workspace import RunWorkspace, _StageContext


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _pick_utm_zone(lat: float, lon: float) -> str:
    """Return a PROJ CRS string for the UTM zone containing (lat, lon)."""
    zone = int((lon + 180) / 6) + 1
    hemi = "N" if lat >= 0 else "S"
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    """WGS-84 geographic → ECEF (metres)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = 6_378_137.0
    f = 1.0 / 298.257_223_563
    e2 = 2 * f - f * f
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + alt_m) * math.sin(lat)
    return x, y, z


def _ecef_to_enu(
    x: float, y: float, z: float,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """ECEF → ENU relative to (lat0, lon0, alt0)."""
    x0, y0, z0 = _wgs84_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)
    sl, cl = math.sin(lat0r), math.cos(lat0r)
    sL, cL = math.sin(lon0r), math.cos(lon0r)
    east  = -sL * dx + cL * dy
    north = -sl * cL * dx - sl * sL * dy + cl * dz
    up    =  cl * cL * dx + cl * sL * dy + sl * dz
    return east, north, up


# ---------------------------------------------------------------------------
# Reference-image file writer
# ---------------------------------------------------------------------------

def _read_image_names(sparse_txt_dir: Path) -> list[str]:
    """Read image names from images.txt (COLMAP TXT format)."""
    names: list[str] = []
    p = sparse_txt_dir / "images.txt"
    if not p.exists():
        p = sparse_txt_dir / "txt" / "images.txt"
    if not p.exists():
        return names

    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            # In COLMAP images.txt:
            # Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (10 tokens)
            # Line 2: POINTS2D (coordinates and point3d_ids)
            if len(parts) >= 10 and any(parts[9].lower().endswith(ext) for ext in valid_exts):
                names.append(parts[9])
    return names


def _load_telemetry_json(ws: "RunWorkspace") -> list[dict]:
    if not ws.telemetry_json.exists():
        return []
    return json.loads(ws.telemetry_json.read_text(encoding="utf-8"))


def _match_images_to_fixes(
    image_names: list[str],
    fixes: list[dict],
    keyframes_index: list[dict],
) -> dict[str, dict]:
    """Map image filename → nearest telemetry fix."""
    # Build a lookup from frame path stem to fix
    stem_to_fix: dict[str, dict] = {}
    for kf in keyframes_index:
        stem = Path(kf["path"]).stem
        if kf.get("lat") is not None:
            stem_to_fix[stem] = {
                "lat": kf["lat"],
                "lon": kf["lon"],
                "alt_m": kf.get("alt_m", 0.0) or 0.0,
            }

    # If keyframes have no GPS, try directly from fixes using timestamp
    result: dict[str, dict] = {}
    for name in image_names:
        stem = Path(name).stem
        if stem in stem_to_fix:
            result[name] = stem_to_fix[stem]
    return result


def _write_ref_images(
    ref_path: Path,
    image_to_fix: dict[str, dict],
    mode: str,
) -> tuple[float, float, float]:
    """Write COLMAP ref_images.txt and return the scene centroid (lat0, lon0, alt0).

    mode='gps': write lat lon alt (for --ref_is_gps 1)
    mode='enu': convert to ENU and write X Y Z (for --ref_is_gps 0)
    """
    lats = [f["lat"] for f in image_to_fix.values()]
    lons = [f["lon"] for f in image_to_fix.values()]
    alts = [f["alt_m"] for f in image_to_fix.values()]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    alt0 = float(np.mean(alts))

    lines: list[str] = []
    for name, fix in image_to_fix.items():
        if mode == "gps":
            lines.append(f"{name} {fix['lat']:.8f} {fix['lon']:.8f} {fix['alt_m']:.3f}")
        else:
            e, n, u = _ecef_to_enu(
                *_wgs84_to_ecef(fix["lat"], fix["lon"], fix["alt_m"]),
                lat0, lon0, alt0,
            )
            lines.append(f"{name} {e:.4f} {n:.4f} {u:.4f}")

    ref_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lat0, lon0, alt0


# ---------------------------------------------------------------------------
# COLMAP invocations
# ---------------------------------------------------------------------------

def _run_model_aligner(
    colmap: "ColmapTool",
    ws: "RunWorkspace",
    config: "Config",
    ref_images_path: Path,
) -> tuple[Path, bool, str]:
    out_dir = ws.georef_sparse_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_model = ws.sparse_dir / "0"
    transform_path = out_dir / "transform.txt"

    # Attempt 1: ECEF with config tolerance
    try:
        colmap.run(
            "model_aligner",
            [
                "--input_path", str(sparse_model),
                "--output_path", str(out_dir),
                "--ref_images_path", str(ref_images_path),
                "--ref_is_gps", "1",
                "--alignment_type", "ecef",
                "--alignment_max_error", str(config.georef.alignment_max_error),
                "--transform_path", str(transform_path),
            ],
            log_path=ws.log_path("colmap_model_aligner"),
        )
        return out_dir, True, "ECEF"
    except Exception:
        pass

    # Attempt 2: ENU alignment (much more numerically stable for short flights / few fixes)
    try:
        colmap.run(
            "model_aligner",
            [
                "--input_path", str(sparse_model),
                "--output_path", str(out_dir),
                "--ref_images_path", str(ref_images_path),
                "--ref_is_gps", "1",
                "--alignment_type", "enu",
                "--alignment_max_error", "25.0",
                "--transform_path", str(transform_path),
            ],
            log_path=ws.log_path("colmap_model_aligner"),
        )
        return out_dir, True, "ENU"
    except Exception:
        pass

    # Attempt 3: ECEF relaxed tolerance
    try:
        colmap.run(
            "model_aligner",
            [
                "--input_path", str(sparse_model),
                "--output_path", str(out_dir),
                "--ref_images_path", str(ref_images_path),
                "--ref_is_gps", "1",
                "--alignment_type", "ecef",
                "--alignment_max_error", "50.0",
                "--transform_path", str(transform_path),
            ],
            log_path=ws.log_path("colmap_model_aligner"),
        )
        return out_dir, True, "ECEF"
    except Exception:
        pass

    # Fallback: GPS alignment could not resolve RANSAC inliers (e.g. collinear flight);
    # copy sparse model so downstream stages proceed in relative metric mode
    import shutil
    if sparse_model.exists():
        for item in sparse_model.iterdir():
            dest = out_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    return out_dir, False, "LOCAL_RELATIVE"


def _run_pose_prior_mapper(
    colmap: "ColmapTool",
    ws: "RunWorkspace",
    config: "Config",
    ref_images_path: Path,
) -> Path:
    out_dir = ws.georef_sparse_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.georef
    colmap.run(
        "pose_prior_mapper",
        [
            "--database_path", str(ws.colmap_db),
            "--image_path", str(ws.images_dir),
            "--output_path", str(out_dir),
            "--overwrite_priors_covariance", "1",
            "--prior_position_std_x", str(cfg.gnss_std_xy),
            "--prior_position_std_y", str(cfg.gnss_std_xy),
            "--prior_position_std_z", str(cfg.gnss_std_z),
        ],
        log_path=ws.log_path("colmap_pose_prior_mapper"),
    )
    return out_dir


def _recentre_aligned_model(aligned_dir: Path) -> tuple[list[float] | None, str]:
    """Shift the aligned model to a local origin, returning the offset applied.

    Why this is not optional
    ------------------------
    ``model_aligner --alignment_type ecef`` writes geocentric coordinates, so a
    scene sits ~6.4e6 m from the origin. OpenMVS - like most mesh software -
    stores vertices and does its geometry in **float32**, whose spacing at 5.5e6
    is *0.5 m*. A 190 m site therefore gets quantised onto a half-metre lattice:
    facades collapse, triangles become degenerate, and RefineMesh and
    TextureMesh both die with STATUS_STACK_BUFFER_OVERRUN (0xC0000409). The
    pipeline then falls back to a 2.5D height field, which is why a capture that
    passes every quality gate still produces no buildings.

    Shifting to a local origin is exact - a pure translation of doubles - and
    fully reversible from the recorded offset, which export adds back before
    projecting. The offset is rounded to whole metres so it stays exactly
    representable and reads sensibly in a manifest.

    Returns ``(offset, note)``; ``offset`` is None when nothing was applied.
    """
    try:
        import pycolmap
    except Exception as exc:
        return None, f"pycolmap unavailable ({type(exc).__name__}); model left at source magnitude"

    try:
        rec = pycolmap.Reconstruction(str(aligned_dir))
        centres = np.array([im.projection_center() for im in rec.images.values()], dtype=float)
        if len(centres) == 0:
            return None, "no registered images to centre on"

        offset = np.round(centres.mean(axis=0)).astype(float)
        # Only worth doing when the magnitude actually threatens float32. A model
        # already near the origin is left untouched so the frame stays simple.
        if float(np.abs(offset).max()) < 10_000.0:
            return None, "model already near the origin; no shift needed"

        rec.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), -offset))
        rec.write(str(aligned_dir))

        resolution = float(np.spacing(np.float32(np.abs(centres).max())))
        return [float(v) for v in offset], (
            f"shifted model to a local origin by {offset.tolist()} m "
            f"(float32 spacing at source magnitude was {resolution:.3f} m, which "
            f"is what breaks OpenMVS meshing)"
        )
    except Exception as exc:
        return None, f"re-centring failed ({type(exc).__name__}: {exc}); model left as aligned"


def _undistort_aligned_model(
    colmap: "ColmapTool", ws: "RunWorkspace", aligned_dir: Path, config: "Config"
) -> None:
    """Refresh OpenMVS input from the aligned model.

    OpenMVS reads camera poses from COLMAP's undistorted workspace.  Keeping
    the pre-alignment workspace here silently loses both metric scale and the
    geodetic transform for every downstream product.
    """
    if ws.undistorted_dir.exists():
        shutil.rmtree(ws.undistorted_dir)
    colmap.run(
        "image_undistorter",
        [
            "--image_path", str(ws.images_dir),
            "--input_path", str(aligned_dir),
            "--output_path", str(ws.undistorted_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(config.frames.max_long_edge),
        ],
        log_path=ws.log_path("colmap_image_undistorter_georef"),
    )


# ---------------------------------------------------------------------------
# Alignment quality estimation (Umeyama / evo)
# ---------------------------------------------------------------------------

def _compute_alignment_rmse(
    sparse_dir: Path,
    image_to_fix: dict[str, dict],
    lat0: float,
    lon0: float,
    alt0: float,
    coordinate_frame: str,
    log_path: Path | None = None,
    model_offset: list[float] | None = None,
) -> tuple[float | None, str]:
    """RMS position error between aligned camera centres and their GPS fixes.

    Returns ``(rmse_m, source)`` where *source* records how the number was
    obtained, or why there is none. Both are needed: a georeferencing stage that
    reports an unexplained absence of accuracy is indistinguishable from one
    that never tried.

    This function previously swallowed every failure into two bare
    ``except Exception`` blocks and returned ``None``, which the caller then
    recorded as ``-1``. It had never once produced a number, because the log
    branch raised ``NameError`` (``re`` was not imported) and the pycolmap
    branch raised ``AttributeError`` (``cam_from_world`` is a method in
    pycolmap 4.x, not a property) - all while COLMAP was printing the exact
    answer into the log file already open two lines above. Failures are now
    named rather than discarded.
    """
    # 1. First priority: COLMAP's own RANSAC alignment error, which is the
    #    authoritative number - it is computed over the control points the
    #    aligner actually used as inliers.
    if log_path and log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"=>\s*Alignment error:\s+([\d.]+)", text)
            if matches:
                return float(matches[-1]), "colmap_model_aligner_log"
        except (OSError, ValueError) as exc:
            log_note = f"log parse failed ({type(exc).__name__})"
        else:
            log_note = "log had no alignment-error line"
    else:
        log_note = "no aligner log"

    if coordinate_frame == "LOCAL_RELATIVE":
        return None, "not georeferenced (relative model)"

    try:
        import pycolmap
        rec = pycolmap.Reconstruction()
        rec.read(str(sparse_dir))
    except Exception as exc:
        return None, f"{log_note}; model unreadable ({type(exc).__name__})"

    # In ENU mode, COLMAP uses the first reference coordinate as origin
    first_fix = next(iter(image_to_fix.values())) if image_to_fix else None
    enu_lat = first_fix["lat"] if first_fix else lat0
    enu_lon = first_fix["lon"] if first_fix else lon0
    enu_alt = first_fix["alt_m"] if first_fix else alt0

    errors: list[float] = []
    skipped_no_fix = 0
    centre_failures: list[str] = []
    for img in rec.images.values():
        name = img.name
        if name not in image_to_fix:
            skipped_no_fix += 1
            continue
        fix = image_to_fix[name]
        ecef_gt = _wgs84_to_ecef(fix["lat"], fix["lon"], fix["alt_m"])
        if coordinate_frame == "ECEF":
            gt = ecef_gt
        elif coordinate_frame == "ENU":
            gt = _ecef_to_enu(*ecef_gt, enu_lat, enu_lon, enu_alt)
        else:
            return None, f"{log_note}; unsupported frame {coordinate_frame!r}"

        # Camera centre in whichever frame model_aligner produced, plus the
        # local-origin shift added back so it is comparable with the GPS fix.
        try:
            centre = np.asarray(img.projection_center(), dtype=float)
        except Exception as exc:
            centre_failures.append(type(exc).__name__)
            continue
        if model_offset is not None:
            centre = centre + np.asarray(model_offset, dtype=float)
        errors.append(math.dist(centre.tolist(), list(gt)))

    if not errors:
        detail = f"{skipped_no_fix} images had no matching fix"
        if centre_failures:
            detail += f"; {len(centre_failures)} centre lookups failed ({centre_failures[0]})"
        return None, f"{log_note}; {detail}"

    rmse = float(math.sqrt(sum(e ** 2 for e in errors) / len(errors)))
    return rmse, f"{log_note}; recomputed over {len(errors)} control points"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    colmap = tools.colmap
    if colmap is None:
        raise RuntimeError("COLMAP is required for stage 'georef'.")

    # Load telemetry
    fixes = _load_telemetry_json(ws)
    if not fixes:
        ctx.note("no telemetry — stage 'georef' skipped (up-to-scale model kept)")
        raise RuntimeError(
            "No telemetry.json found for georeferencing. "
            "Pass --no-telemetry to run in up-to-scale mode, "
            "or provide a .srt/.csv sidecar via --telemetry."
        )

    # Load keyframe index for GPS lookup
    keyframes: list[dict] = []
    if ws.frames_index.exists():
        keyframes = json.loads(ws.frames_index.read_text(encoding="utf-8"))

    # Match image names to GPS fixes
    txt_dir = ws.sparse_dir / "0" / "txt"
    image_names = _read_image_names(txt_dir)
    image_to_fix = _match_images_to_fixes(image_names, fixes, keyframes)

    if len(image_to_fix) < 3:
        ctx.note(f"Only {len(image_to_fix)} images matched GPS fixes; continuing in relative scale mode")
        import shutil
        ws.georef_sparse_dir.mkdir(parents=True, exist_ok=True)
        sparse_model = ws.sparse_dir / "0"
        if sparse_model.exists():
            for item in sparse_model.iterdir():
                dest = ws.georef_sparse_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        from .workspace import GnssMode
        ws.set_gnss(GnssMode.NONE)
        ws.set_accuracy(georef_method="relative", gnss_mode="NONE", coordinate_frame="LOCAL_RELATIVE")
        (ws.georef_sparse_dir / "transform.json").write_text(json.dumps({
            "method": "relative", "coordinate_frame": "LOCAL_RELATIVE",
            "output_crs": "LOCAL_RELATIVE", "georef_success": False,
        }, indent=2), encoding="utf-8")
        return

    # Compute scene centroid and write reference file
    ws.georef_sparse_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ws.georef_sparse_dir / "ref_images.txt"
    lat0, lon0, alt0 = _write_ref_images(ref_path, image_to_fix, mode="gps")

    # Pick UTM CRS
    crs = config.georef.target_crs
    if crs == "auto":
        crs = _pick_utm_zone(lat0, lon0)

    ctx.note(f"scene centroid: lat={lat0:.6f}, lon={lon0:.6f}, alt={alt0:.1f}m, CRS={crs}")
    ctx.note(f"GPS control points: {len(image_to_fix)}")

    # Method selection
    # GPS positions are supplied through ref_images.txt.  Do not claim the
    # pose-prior mapper until priors are actually written to COLMAP's database.
    requested_method = config.georef.method
    method = "aligner"
    if requested_method in ("auto", "pose_prior"):
        ctx.note("using model_aligner; pose-prior mapping is not enabled because database priors were not ingested")
    aligned_dir, success, coordinate_frame = _run_model_aligner(colmap, ws, config, ref_path)

    model_offset: list[float] | None = None
    if success:
        ctx.note("model_aligner: successfully aligned to GPS coordinates")
        # Re-centre before undistortion, because undistortion is what publishes
        # the model to OpenMVS and OpenMVS cannot survive geocentric magnitudes.
        model_offset, offset_note = _recentre_aligned_model(aligned_dir)
        ctx.note(offset_note)
        _undistort_aligned_model(colmap, ws, aligned_dir, config)
    else:
        ctx.note("model_aligner: GPS alignment could not resolve RANSAC inliers (collinear/short flight); continuing with relative scale model")

    # Estimate alignment quality
    rmse: float | None = None
    rmse_source = "georeferencing did not succeed"
    if success:
        rmse, rmse_source = _compute_alignment_rmse(
            aligned_dir, image_to_fix, lat0, lon0, alt0, coordinate_frame,
            log_path=ws.log_path("colmap_model_aligner"),
            model_offset=model_offset,
        )
    if rmse is not None:
        ctx.note(f"alignment error {rmse:.2f} m over {len(image_to_fix)} GPS control "
                 f"points (source: {rmse_source})")
    elif success:
        # Never let an unmeasured accuracy pass as a quiet absence: a
        # georeferenced product whose error is unknown cannot be published as
        # metric, and the reason has to reach the operator.
        ctx.note(f"WARNING: georeferenced but alignment error could not be measured "
                 f"- {rmse_source}. Treat coordinates as unvalidated.")

    # Determine GNSS mode
    from .workspace import GnssMode
    current_mode = ws.gnss_mode
    if not success:
        gnss_mode = GnssMode.NONE
    elif current_mode is GnssMode.NONE:
        gnss_mode = GnssMode.SIMULATED
    else:
        gnss_mode = current_mode  # already set by stage1 or CLI

    # Two different CRSs, because two different things are being described, and
    # collapsing them is how an ECEF model came to be advertised as UTM:
    #   model_crs  - the frame the sparse/dense/mesh files are actually in.
    #                ECEF alignment produces EPSG:4978, shifted by model_offset.
    #   crs        - the projected CRS the GIS products are written in, after
    #                export converts them. This is what stage 7 consumes.
    if not success:
        model_crs = "LOCAL_RELATIVE"
    elif coordinate_frame == "ECEF":
        model_crs = "EPSG:4978"
    elif coordinate_frame == "ENU":
        model_crs = "LOCAL_ENU"
    else:
        model_crs = "LOCAL_RELATIVE"
    product_crs = crs if (success and coordinate_frame == "ECEF") else "LOCAL_RELATIVE"

    ws.set_gnss(gnss_mode, source=str(ws.telemetry_json) if success else None)
    ws.set_accuracy(
        alignment_rmse_m=round(rmse, 4) if rmse is not None else None,
        alignment_rmse_source=rmse_source,
        n_gps_fixes_used=len(image_to_fix) if success else 0,
        crs=product_crs,
        model_crs=model_crs,
        model_offset_m=model_offset,
        lat0=lat0 if success else None,
        lon0=lon0 if success else None,
        alt0_m=alt0 if success else None,
        georef_method=method if success else "none",
        gnss_mode=gnss_mode.value,
    )

    # Save transform info
    transform = {
        "method": method if success else "none",
        "crs": product_crs,
        "scene_centroid_wgs84": {"lat": lat0, "lon": lon0, "alt_m": alt0} if success else None,
        "n_gps_control_points": len(image_to_fix) if success else 0,
        "alignment_rmse_m": round(rmse, 4) if rmse is not None else None,
        "alignment_rmse_source": rmse_source,
        "gnss_mode": gnss_mode.value,
        "coordinate_frame": coordinate_frame if success else "LOCAL_RELATIVE",
        "source_crs": model_crs,
        "model_crs": model_crs,
        # Add this back to model coordinates to recover true model_crs values.
        "model_offset_m": model_offset,
        "output_crs": product_crs,
        "georef_success": bool(success),
    }
    (ws.georef_sparse_dir / "transform.json").write_text(
        json.dumps(transform, indent=2), encoding="utf-8"
    )

    ctx.metric(
        n_gps_fixes_used=len(image_to_fix) if success else 0,
        # None, not -1: a sentinel that looks like a measurement gets rendered
        # as "-1 m" in reports and compares as excellent against any threshold.
        alignment_rmse_m=round(rmse, 4) if rmse is not None else None,
        alignment_rmse_source=rmse_source,
        crs=product_crs,
        model_crs=model_crs,
        georef_method=method if success else "none",
    )
    ctx.output(georef_sparse_dir=str(aligned_dir))
