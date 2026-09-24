"""Stage 3a — Structure-from-Motion with COLMAP.

What this stage does
--------------------
1.  Runs COLMAP ``feature_extractor`` (SIFT, GPU if available).
2.  Runs ``sequential_matcher`` (primary strategy for video — images are
    ordered, so sequential matching is far faster than exhaustive).
3.  Runs ``mapper`` to reconstruct camera poses and a sparse point cloud.
4.  If the registered-frame fraction is below ``min_registered_fraction``,
    escalates through a ladder:
      a. ``exhaustive_matcher`` + re-mapper
      b. ``vocab_tree_matcher`` + re-mapper  (if the vocab tree is present)
5.  Picks the largest sub-model and undistorts it (output for OpenMVS in
    Stage 5 and export in Stage 7).
6.  Converts the sparse model to TXT format for downstream stages.

The escalation ladder is deliberate: for a single forward pass with low
overlap, sequential matching is almost always sufficient, but a failed flight
(missed loop, strong wind) may need exhaustive or retrieval-based matching
to connect all the sub-models.

Outputs
-------
``ws.sparse_dir/0/``       — cameras.bin, images.bin, points3D.bin (COLMAP binary)
``ws.sparse_dir/0/txt/``   — cameras.txt, images.txt, points3D.txt
``ws.undistorted_dir/``    — undistorted images + sparse for OpenMVS
``ws.colmap_db``           — feature database (kept for potential re-matching)
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .tools import ColmapTool, ToolRegistry
    from .workspace import RunWorkspace, _StageContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_images(images_dir: Path) -> int:
    return sum(1 for _ in images_dir.glob("*.jpg"))


def _largest_model_dir(sparse_dir: Path) -> Path | None:
    """Return the sub-model directory with the most registered images."""
    best: Path | None = None
    best_count = 0
    for sub in sorted(sparse_dir.iterdir()):
        if not sub.is_dir():
            continue
        images_bin = sub / "images.bin"
        if not images_bin.exists():
            continue
        # Quick heuristic: file size correlates with number of images
        size = images_bin.stat().st_size
        if size > best_count:
            best_count = size
            best = sub
    return best


def _count_registered_cameras(model_dir: Path | None) -> int:
    """Read exact registered camera count from COLMAP model directory."""
    if not model_dir or not model_dir.exists():
        return 0
    images_bin = model_dir / "images.bin"
    if images_bin.exists():
        try:
            with images_bin.open("rb") as f:
                import struct
                return struct.unpack("<Q", f.read(8))[0]
        except Exception:
            pass
    images_txt = model_dir / "txt" / "images.txt"
    if images_txt.exists():
        try:
            with images_txt.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip() and not line.startswith("#")) // 2
        except Exception:
            pass
    return 0


def _parse_mapper_summary(log_text: str, n_total: int = 0) -> dict[str, float]:
    """Extract registered-image count and reprojection error from mapper log."""
    metrics: dict[str, float] = {}

    # Modern COLMAP 3.9+ outputs: num_reg_frames=X
    reg_matches = [int(x) for x in re.findall(r"num_reg_frames=(\d+)", log_text)]
    if reg_matches:
        n_reg = max(reg_matches)
        metrics["n_registered"] = n_reg
        if n_total > 0:
            metrics["n_total"] = n_total
            metrics["registered_fraction"] = n_reg / max(n_total, 1)

    # Legacy COLMAP summary output
    m = re.search(r"Registered:\s+(\d+)\s*/\s*(\d+)", log_text)
    if m:
        metrics["n_registered"] = int(m.group(1))
        metrics["n_total"] = int(m.group(2))
        metrics["registered_fraction"] = int(m.group(1)) / max(int(m.group(2)), 1)

    m2 = re.search(r"Mean reprojection error:\s+([\d.]+)", log_text)
    if m2:
        metrics["mean_reproj_error"] = float(m2.group(1))
    return metrics


def _parse_points3d_count(sparse_model_dir: Path) -> int:
    """Count 3D points by counting lines in points3D.txt."""
    txt = sparse_model_dir / "txt" / "points3D.txt"
    if not txt.exists():
        bin_path = sparse_model_dir / "points3D.bin"
        if bin_path.exists():
            # Binary: first 8 bytes = uint64 count
            try:
                with bin_path.open("rb") as f:
                    import struct
                    count_bytes = f.read(8)
                    if len(count_bytes) == 8:
                        return struct.unpack("<Q", count_bytes)[0]
            except Exception:
                pass
        return 0
    count = 0
    with txt.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                count += 1
    return count


# ---------------------------------------------------------------------------
# COLMAP step runners
# ---------------------------------------------------------------------------

def _extract_features(colmap: "ColmapTool", ws: "RunWorkspace", config: "Config") -> None:
    cfg = config.pose
    masks_dir = ws.masks_dir if ws.masks_dir.exists() and any(ws.masks_dir.iterdir()) else None
    args = [
        "--database_path", str(ws.colmap_db),
        "--image_path", str(ws.images_dir),
        "--ImageReader.single_camera", "1" if cfg.single_camera else "0",
        "--ImageReader.camera_model", cfg.camera_model,
        "--SiftExtraction.max_num_features", str(cfg.max_num_features),
    ]
    camera_params = getattr(cfg, "camera_params", None)
    if camera_params:
        args += ["--ImageReader.camera_params", str(camera_params)]
    # These flags were removed / renamed between COLMAP releases; check before using.
    if colmap.supports("feature_extractor", "--SiftExtraction.use_gpu"):
        args += ["--SiftExtraction.use_gpu", "1" if cfg.use_gpu else "0"]
    if colmap.supports("feature_extractor", "--SiftExtraction.max_image_size"):
        args += ["--SiftExtraction.max_image_size", str(config.frames.max_long_edge)]
    if masks_dir:
        args += ["--ImageReader.mask_path", str(masks_dir)]
    colmap.run(
        "feature_extractor", args,
        log_path=ws.log_path("colmap_feature_extractor"),
    )



def _match_sequential(colmap: "ColmapTool", ws: "RunWorkspace", config: "Config") -> None:
    cfg = config.pose
    vocab = _find_vocab_tree(config)
    args = [
        "--database_path", str(ws.colmap_db),
        "--SequentialMatching.overlap", str(cfg.sequential_overlap),
        "--SequentialMatching.quadratic_overlap", "1" if cfg.quadratic_overlap else "0",
    ]
    if cfg.loop_detection and vocab:
        args += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", str(vocab),
            "--SequentialMatching.loop_detection_period", str(cfg.loop_detection_period),
        ]
    else:
        args += ["--SequentialMatching.loop_detection", "0"]
    colmap.run("sequential_matcher", args, log_path=ws.log_path("colmap_sequential_matcher"))


def _match_exhaustive(colmap: "ColmapTool", ws: "RunWorkspace") -> None:
    colmap.run(
        "exhaustive_matcher",
        ["--database_path", str(ws.colmap_db)],
        log_path=ws.log_path("colmap_exhaustive_matcher"),
    )


def _is_faiss_vocab_tree(path: Path) -> bool:
    """Return True if the vocab tree uses the faiss format (COLMAP >= May 2025).

    COLMAP's faiss-format trees have file_version == 1 or 2 at byte offset 0.
    The old flann-based trees had a different header (e.g. 32762) and will
    crash COLMAP 4.x with exit code 3221226505 (CHECK failed).
    """
    try:
        with path.open("rb") as f:
            header = f.read(4)
        if len(header) < 4:
            return False
        import struct
        file_version = struct.unpack_from("<I", header, 0)[0]
        return file_version in (1, 2)  # faiss format; flann = other values
    except Exception:
        return False


def _match_vocab_tree(colmap: "ColmapTool", ws: "RunWorkspace", config: "Config") -> None:
    vocab = _find_vocab_tree(config)
    if not vocab:
        raise RuntimeError("vocab tree not found — cannot run vocab_tree_matcher")
    if not _is_faiss_vocab_tree(vocab):
        raise RuntimeError(
            f"Vocab tree at {vocab} uses the old flann format. COLMAP >= May 2025 "
            "requires a faiss-format tree. Re-run `python scripts/bootstrap.py` to "
            "download the updated tree, or build one with:\n"
            "  colmap vocab_tree_builder --descriptor_path <path> --vocab_tree_path <out>"
        )
    colmap.run(
        "vocab_tree_matcher",
        [
            "--database_path", str(ws.colmap_db),
            "--VocabTreeMatching.vocab_tree_path", str(vocab),
        ],
        log_path=ws.log_path("colmap_vocab_tree_matcher"),
    )


def _find_vocab_tree(config: "Config") -> Path | None:
    tools_root = Path(config.tools_root)
    # Prefer a faiss-format tree (larger file version, typically has "faiss" in name)
    all_trees = sorted(tools_root.glob("**/vocab_tree*.bin"))
    faiss_trees = [t for t in all_trees if _is_faiss_vocab_tree(t)]
    return faiss_trees[0] if faiss_trees else (all_trees[0] if all_trees else None)



def _run_mapper(
    colmap: "ColmapTool", ws: "RunWorkspace", config: "Config"
) -> tuple[str, int]:
    """Run COLMAP mapper; return its log tail **and its exit code**.

    ``check=False`` is deliberate -- a mapper that runs to completion and
    registers too few frames is a *result* to be measured, not an exception. But
    the exit code used to be dropped on the floor here, and that made the two
    genuinely different outcomes indistinguishable downstream:

      * the mapper ran, finished, and could not reconstruct the scene, versus
      * the mapper never finished -- killed by the OS, out of memory, or crashed.

    Both arrive as "no model on disk", and the caller then asserted the first
    one, blaming overlap, motion blur or frame count. That was wrong at least
    once on this project: a run whose log showed 87 of 130 frames registered
    and healthy (~1800 of ~3100 points seen per image) was reported as "too
    little overlap ... or too few frames" when in fact a concurrent OpenMVS
    texturing job had exhausted memory and the process was terminated
    mid-registration. The advice that followed -- lower ``init_min_tri_angle``,
    add keyframes -- would have made memory pressure *worse*.

    So return the exit code and let the caller diagnose from evidence.
    """
    ws.sparse_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.pose
    result = colmap.run(
        "mapper",
        [
            "--database_path", str(ws.colmap_db),
            "--image_path", str(ws.images_dir),
            "--output_path", str(ws.sparse_dir),
            "--Mapper.init_min_tri_angle", str(cfg.init_min_tri_angle),
            "--Mapper.ba_refine_principal_point", "1" if cfg.ba_refine_principal_point else "0",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.multiple_models", "0",
        ],
        log_path=ws.log_path("colmap_mapper"),
        check=False,  # don't hard-fail; we parse the log for registered fraction
    )
    return result.tail, result.returncode


def _diagnose_empty_model(
    *,
    log_text: str,
    mapper_rc: int,
    n_images: int,
    run_id: str,
    log_path: "Path",
) -> str:
    """Explain an empty sparse dir from evidence rather than by assertion.

    Two genuinely different outcomes both leave no model on disk, and telling
    them apart matters because the remedies are opposites:

      * **Process failure** -- the mapper was killed or crashed. Needs *less*
        load (stop the concurrent job, then resume).
      * **Capture failure** -- the mapper ran to completion and could not solve
        the scene. Needs *more* signal (denser keyframes, less blur).

    The old message asserted the second unconditionally, listing "too little
    overlap, severe motion blur, or too few frames". That misdiagnosed a real
    run on this project: the log showed 87 of 130 frames registered and healthy
    when a concurrent OpenMVS texturing job exhausted memory and the process was
    terminated. Its advice -- lower ``init_min_tri_angle``, add keyframes --
    would have increased the memory pressure that actually caused the failure.

    The exit code and the log are the evidence. COLMAP prints ``num_reg_frames``
    as it registers and writes the model only at the end, so substantial
    progress with no model means interruption, not refusal.
    """
    progress = [int(x) for x in re.findall(r"num_reg_frames=(\d+)", log_text)]
    # `num_reg_frames=N` is printed *before* the Nth registration completes, so
    # the count reached is one more than the largest value seen.
    reached = max(progress) + 1 if progress else 0

    if mapper_rc != 0:
        # On Windows a terminated process surfaces as a large NTSTATUS
        # (0xC0000409 stack buffer overrun, 0xC0000005 access violation,
        # 0xC0000017 no-memory); POSIX signals arrive negative. Print both forms
        # so the code is searchable whichever platform produced it.
        detail = (
            f"COLMAP mapper exited with code {mapper_rc} "
            f"(0x{mapper_rc & 0xFFFFFFFF:08X}) after registering {reached} of "
            f"{n_images} frames, and wrote no model.\n"
            "This is a PROCESS failure, not a capture failure: the mapper was "
            "terminated or crashed rather than rejecting the imagery."
        )
        if reached >= 3:
            detail += (
                f"\n\nRegistration was progressing normally to {reached} frames, so "
                "the capture is not the problem. The usual cause is memory "
                "exhaustion from another photogrammetry job running at the same "
                "time - OpenMVS DensifyPointCloud, RefineMesh and TextureMesh are "
                "the heavy ones. Check for a concurrent run, wait for it to "
                "finish, then resume:\n"
                f"  dronemap run --run-id {run_id}\n"
                "Do NOT lower pose.init_min_tri_angle or add keyframes for this "
                "failure - both increase memory pressure."
            )
        return f"{detail}\nFull log: {log_path}"

    # Clean exit with no model: the mapper really did decline to reconstruct.
    return (
        f"COLMAP mapper exited cleanly but produced no reconstruction "
        f"(it registered at most {reached} of {n_images} frames).\n"
        "This is a CAPTURE failure: too little overlap between consecutive "
        "keyframes, severe motion blur, or a featureless/repetitive surface that "
        "defeats SIFT matching.\n"
        "Raise keyframe density by increasing frames.target_overlap AND lowering "
        "frames.sharpness_window together - the window bounds the overlap, so "
        "raising overlap alone silently does nothing. If the flight has very "
        "little parallax, also lower pose.init_min_tri_angle.\n"
        f"Full log: {log_path}"
    )


def _convert_to_txt(colmap: "ColmapTool", ws: "RunWorkspace", model_dir: Path) -> None:
    txt_dir = model_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    colmap.run(
        "model_converter",
        [
            "--input_path", str(model_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ],
        log_path=ws.log_path("colmap_model_converter"),
    )


def _undistort(colmap: "ColmapTool", ws: "RunWorkspace", model_dir: Path, config: "Config") -> None:
    ws.undistorted_dir.mkdir(parents=True, exist_ok=True)
    colmap.run(
        "image_undistorter",
        [
            "--image_path", str(ws.images_dir),
            "--input_path", str(model_dir),
            "--output_path", str(ws.undistorted_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(config.frames.max_long_edge),
        ],
        log_path=ws.log_path("colmap_image_undistorter"),
    )


def _assess_capture_quality(
    ws: "RunWorkspace",
    config: "Config",
    ctx: "_StageContext",
    model_dir: Path,
    n_images: int,
) -> None:
    """Measure capture geometry and record an honest verdict.

    Never raises on its own internal failure: a quality gate that crashes the
    pipeline is worse than no gate.  It either records a verdict or notes that
    it could not.  Only a REJECT verdict stops the run, and only when
    ``quality.block_on_reject`` is set.
    """
    qcfg = getattr(config, "quality", None)
    if qcfg is None or not qcfg.enabled:
        return

    try:
        import pycolmap

        from .quality import Verdict, assess_pose, measure_sparse_geometry

        recon = pycolmap.Reconstruction(str(model_dir))
        geom = measure_sparse_geometry(recon, n_total=n_images)
        assessment = assess_pose(geom, qcfg)
    except Exception as exc:  # pragma: no cover - diagnostic path only
        ctx.note(f"capture-quality assessment unavailable ({exc}); continuing ungated")
        return

    ctx.metric(
        median_tri_angle_deg=round(geom.median_tri_angle_deg, 3),
        baseline_depth_ratio=round(geom.baseline_depth_ratio, 5),
        depth_dynamic_range=round(geom.depth_dynamic_range, 3),
        path_efficiency=round(geom.path_efficiency, 4),
        forward_motion_ratio=round(geom.forward_motion_ratio, 4),
        mean_track_length=round(geom.mean_track_length, 2),
        capture_verdict=assessment.verdict.value,
    )

    # stage_dir() keys off the stage name ("pose"), not the directory it maps to.
    report_path = ws.stage_dir("pose") / "capture_quality.json"
    try:
        report_path.write_text(
            json.dumps(assessment.to_dict(), indent=2), encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - disk-level failure
        ctx.note(f"could not write capture_quality.json ({exc})")

    ctx.note(
        f"capture quality: {assessment.headline} "
        f"(triangulation {geom.median_tri_angle_deg:.1f}deg, "
        f"baseline/depth {geom.baseline_depth_ratio:.2f}x, "
        f"depth range {geom.depth_dynamic_range:.2f}x)"
    )
    for reason in assessment.reasons:
        ctx.note(f"  - {reason}")
    for warning in assessment.warnings:
        ctx.note(f"  ! {warning}")

    if assessment.verdict is Verdict.REJECT and qcfg.block_on_reject:
        detail = " ".join(assessment.reasons) or "capture geometry is inadequate"
        advice = " ".join(assessment.advice)
        raise RuntimeError(
            f"Capture rejected before dense reconstruction. {detail} {advice} "
            "DroneMap stopped here rather than spend the dense and mesh budget "
            "producing a model that would not represent the scene."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    cfg = config.pose

    # -----------------------------------------------------------------------
    # Stretch backend: MASt3R
    # -----------------------------------------------------------------------
    if cfg.backend == "mast3r":
        ctx.note("pose.backend=mast3r — using MASt3R SfM (Stretch path)")
        from .stage3_pose_mast3r import run_mast3r
        run_mast3r(ws, config, tools, ctx)
        return

    # -----------------------------------------------------------------------
    # Core backend: COLMAP
    # -----------------------------------------------------------------------
    colmap = tools.colmap
    if colmap is None:
        raise RuntimeError("COLMAP is required for stage 'pose'. Run `python scripts/bootstrap.py`.")

    n_images = _count_images(ws.images_dir)
    if n_images == 0:
        raise RuntimeError(f"No images in {ws.images_dir} — did stage 'frames' complete?")
    ctx.metric(n_images=n_images)

    # Ensure clean database and sparse output when stage is executed
    if ws.colmap_db.exists():
        ws.colmap_db.unlink()
    import shutil
    if ws.sparse_dir.exists():
        shutil.rmtree(ws.sparse_dir, ignore_errors=True)
    ws.sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Feature extraction
    ctx.note("extracting SIFT features")
    _extract_features(colmap, ws, config)

    # Step 2: Sequential matching (primary — best for video)
    ctx.note("sequential matching")
    _match_sequential(colmap, ws, config)

    # Step 3: Mapping
    ctx.note("mapping (first attempt)")
    log_text, mapper_rc = _run_mapper(colmap, ws, config)
    model_dir = _largest_model_dir(ws.sparse_dir)
    n_reg = _count_registered_cameras(model_dir)

    metrics = _parse_mapper_summary(log_text, n_images)
    if n_reg > metrics.get("n_registered", 0):
        metrics["n_registered"] = n_reg
        metrics["registered_fraction"] = n_reg / max(n_images, 1)

    registered_fraction = metrics.get("registered_fraction", n_reg / max(n_images, 1))
    ctx.metric(**metrics)

    # Escalation ladder
    if registered_fraction < cfg.min_registered_fraction and cfg.escalate_on_failure:
        import shutil
        backup_dir = ws.sparse_dir.parent / "sparse_backup_sequential"
        if ws.sparse_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(ws.sparse_dir, backup_dir)

        # Step 3b: Neural matcher boost on low-inlier pairs (< 50 matches)
        try:
            from .matching_neural import enhance_sparse_matches
            boost_res = enhance_sparse_matches(ws, config, ctx, min_inliers=50)
            if boost_res.get("n_boosted_pairs", 0) > 0:
                log_text, mapper_rc = _run_mapper(colmap, ws, config)
                boosted_model_dir = _largest_model_dir(ws.sparse_dir)
                boosted_n_reg = _count_registered_cameras(boosted_model_dir)
                if boosted_n_reg > n_reg:
                    ctx.note(f"neural matcher boost improved registration: {n_reg} -> {boosted_n_reg}")
                    model_dir = boosted_model_dir
                    n_reg = boosted_n_reg
                    metrics = _parse_mapper_summary(log_text, n_images)
                    registered_fraction = metrics.get("registered_fraction", n_reg / max(n_images, 1))
                    ctx.metric(**metrics)
        except Exception as e:
            ctx.note(f"neural matcher boost exception ({e}); continuing escalation")

        # Cap exhaustive matching to N <= 150 to prevent O(N^2) latency explosion
        if registered_fraction < cfg.min_registered_fraction and n_images <= 150:
            ctx.note(
                f"registered fraction {registered_fraction:.1%} < "
                f"threshold {cfg.min_registered_fraction:.1%} (N={n_images} <= 150) — escalating to exhaustive matching"
            )
            # Re-use the database (features already extracted); just re-match
            _match_exhaustive(colmap, ws)
            log_text, mapper_rc = _run_mapper(colmap, ws, config)
            new_model_dir = _largest_model_dir(ws.sparse_dir)
            new_n_reg = _count_registered_cameras(new_model_dir)

            # Never replace a superior sequential reconstruction with a degenerate exhaustive one!
            if new_n_reg < n_reg and backup_dir.exists():
                ctx.note(
                    f"exhaustive matching yielded fewer cameras ({new_n_reg} < {n_reg}); "
                    "preserving superior sequential reconstruction"
                )
                shutil.rmtree(ws.sparse_dir)
                shutil.copytree(backup_dir, ws.sparse_dir)
                model_dir = _largest_model_dir(ws.sparse_dir)
            else:
                model_dir = new_model_dir
                n_reg = new_n_reg
                metrics = _parse_mapper_summary(log_text, n_images)
                registered_fraction = metrics.get("registered_fraction", n_reg / max(n_images, 1))
                ctx.metric(**metrics)
        else:
            ctx.note(
                f"registered fraction {registered_fraction:.1%} < threshold {cfg.min_registered_fraction:.1%}, "
                f"but N={n_images} > 150 (exhaustive would incur {n_images * (n_images - 1) // 2} pairs); "
                "bypassing exhaustive matching to preserve near-real-time throughput"
            )

        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        if registered_fraction < cfg.min_registered_fraction:
            vocab = _find_vocab_tree(config)
            if vocab and _is_faiss_vocab_tree(vocab):
                ctx.note("exhaustive matching insufficient — trying vocab_tree_matcher")
                _match_vocab_tree(colmap, ws, config)
                log_text, mapper_rc = _run_mapper(colmap, ws, config)
                new_model_dir = _largest_model_dir(ws.sparse_dir)
                new_n_reg = _count_registered_cameras(new_model_dir)
                if new_n_reg >= n_reg:
                    model_dir = new_model_dir
                    metrics = _parse_mapper_summary(log_text, n_images)
                    ctx.metric(**metrics)
                    registered_fraction = metrics.get("registered_fraction", new_n_reg / max(n_images, 1))
            elif vocab and not _is_faiss_vocab_tree(vocab):
                ctx.note(
                    "⚠ vocab tree is flann-format (incompatible with COLMAP ≥ May 2025) — "
                    "skipping vocab_tree escalation. Re-run bootstrap.py to get a faiss tree."
                )
            else:
                ctx.note("no vocab tree available for fallback matching")

    if model_dir is None:
        raise RuntimeError(
            _diagnose_empty_model(
                log_text=log_text,
                mapper_rc=mapper_rc,
                n_images=n_images,
                run_id=ws.run_id,
                log_path=ws.log_path("colmap_mapper"),
            )
        )

    n_registered_cams = _count_registered_cameras(model_dir)
    if n_registered_cams < 3:
        raise RuntimeError(
            f"COLMAP could only register {n_registered_cams} camera pose(s). "
            "At least 3 overlapping cameras are required to reconstruct a 3D scene. "
            "Please ensure the drone video has steady forward motion, continuous visual overlap, "
            "and does not spin or point at featureless surfaces."
        )

    # The mapper log format differs between COLMAP releases.  The binary model
    # is the authoritative source for registered-camera count.
    registered_fraction = n_registered_cams / max(n_images, 1)
    ctx.metric(
        n_registered=n_registered_cams,
        n_total=n_images,
        registered_fraction=round(registered_fraction, 4),
    )

    if registered_fraction < cfg.min_registered_fraction:
        raise RuntimeError(
            f"Only {n_registered_cams}/{n_images} frames ({registered_fraction:.1%}) registered; "
            f"the minimum for a coherent scene is {cfg.min_registered_fraction:.0%}. "
            "DroneMap stopped before dense reconstruction to avoid showing a partial or "
            "unrecognisable mesh. Re-capture with slower movement, 70–80% forward overlap, "
            "and textured buildings/ground visible throughout the pass."
        )

    # Step 4: Convert to TXT (needed by stage3_georef and stage7_export)
    ctx.note("converting sparse model to TXT format")
    _convert_to_txt(colmap, ws, model_dir)

    # Step 4b: Capture-quality assessment.
    #
    # Registration counts say nothing about whether the imagery can support 3D
    # structure: a hovering drone registers 100% of its frames and still yields
    # a flat sheet.  Measure the actual capture geometry here -- before the
    # expensive undistort/dense/mesh path -- so a hover or a fly-through is
    # named while the user is still watching, rather than after twenty minutes
    # of MVS produces a road carpet labelled "3D model".
    _assess_capture_quality(ws, config, ctx, model_dir, n_images)

    # Step 5: Undistort (needed by stage5_dense via OpenMVS)
    ctx.note("undistorting images for OpenMVS")
    _undistort(colmap, ws, model_dir, config)

    # Record 3D point count
    n_points = _parse_points3d_count(model_dir)
    ctx.metric(n_points3d=n_points)

    if n_points < cfg.min_sparse_points:
        ctx.note(
            f"WARNING: COLMAP recovered {n_points} sparse 3D points (threshold: {cfg.min_sparse_points}). "
            "OpenMVS dense stereo will densify the scene from the registered cameras."
        )

    ctx.output(
        sparse_dir=str(model_dir),
        undistorted_dir=str(ws.undistorted_dir),
    )
