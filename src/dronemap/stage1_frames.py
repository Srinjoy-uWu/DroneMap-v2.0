"""Stage 1 — Frame extraction, sharpness filtering, baseline-aware keyframe selection.

What this stage does
--------------------
1.  Decode the video with a streaming 2-pass architecture (O(1) memory footprint):
    - Pass 1 (Analysis): Stream frame-by-frame without storing raw images in RAM.
      Compute Laplacian-variance sharpness and sample optical flow on the fly.
    - Select keyframes using windowed sharpness filtering and baseline-aware stride.
    - Pass 2 (Extraction): Stream/seek only to the selected keyframe indices,
      downscale to ``max_long_edge``, apply CLAHE if enabled, and write directly
      to disk as JPEG.
2.  Within non-overlapping windows of ``sharpness_window`` frames, keep only
    the sharpest frame — this eliminates groups of blurred frames efficiently.
3.  Apply a global ``min_sharpness_ratio`` threshold (fraction of the clip
    median) to discard frames that are blurred even within their window.
4.  Apply baseline-aware stride: thin the stream until consecutive keyframes
    have roughly ``target_overlap`` forward overlap. When telemetry is
    available, stride is derived from GPS ground speed; otherwise it falls back
    to optical-flow magnitude estimation.
5.  Enforce the ``max_keyframes`` cap with uniform subsampling.
6.  Write ``keyframes.json`` with per-frame metadata for downstream stages.
7.  If a telemetry sidecar is present, parse it and write ``telemetry.json``.

Outputs
-------
``ws.images_dir/*.jpg``  — selected keyframes, named ``frame_NNNNNN.jpg``
``ws.frames_index``      — JSON list of per-frame dicts
``ws.telemetry_json``    — JSON list of GPS fixes (if telemetry available)
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .tools import ToolRegistry
    from .workspace import RunWorkspace, _StageContext


def _laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness score: higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calculate_exposure_score(gray: np.ndarray) -> float:
    """Measures exposure quality (0.0 to 1.0).
    Penalizes mean luma deviation from 128 and severe clipping (< 10 or > 245).
    """
    mean_val = float(np.mean(gray))
    luma_score = max(0.0, 1.0 - abs(mean_val - 128.0) / 128.0)
    under_exposed = float(np.mean(gray < 10))
    over_exposed = float(np.mean(gray > 245))
    clipping_penalty = min(1.0, 2.0 * (under_exposed + over_exposed))
    return round(max(0.0, luma_score - clipping_penalty), 4)


def _calculate_contrast_score(gray: np.ndarray) -> float:
    """Measures RMS / standard deviation contrast normalized to [0, 1]."""
    std_val = float(np.std(gray))
    return round(min(1.0, std_val / 64.0), 4)


def _calculate_entropy_score(gray: np.ndarray) -> float:
    """Calculates Shannon entropy of edge gradients normalized to [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hist, _ = np.histogram(mag, bins=32, range=(0, 255), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    entropy = -float(np.sum(hist * np.log2(hist)))
    return round(min(1.0, entropy / 5.0), 4)


def _resize_to_long_edge(img: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe_obj.apply(gray)


# ---------------------------------------------------------------------------
# Video probing & streaming helpers (constant memory O(1))
# ---------------------------------------------------------------------------

def _probe_video(video_path: Path, ffmpeg_bin: str | None = None) -> dict[str, float | int]:
    """Extract width, height, fps, and frame count without loading video into RAM."""
    info: dict[str, float | int] = {"width": 0, "height": 0, "fps": 30.0, "frame_count": 0}

    # 1. Try OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if w > 0 and h > 0:
                info["width"] = w
                info["height"] = h
                info["fps"] = fps if fps > 0 else 30.0
                info["frame_count"] = count
                return info
    except Exception:
        pass

    # 2. Try ffprobe (if available on PATH or next to ffmpeg_bin)
    if ffmpeg_bin:
        probe_candidates = [
            Path(ffmpeg_bin).parent / ("ffprobe" + Path(ffmpeg_bin).suffix),
            Path("ffprobe"),
        ]
        which_ffprobe = shutil.which("ffprobe")
        if which_ffprobe:
            probe_candidates.insert(0, Path(which_ffprobe))

        for candidate in probe_candidates:
            if candidate.is_file() or shutil.which(str(candidate)):
                try:
                    probe_cmd = [
                        str(candidate), "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                        "-of", "json",
                        str(video_path),
                    ]
                    res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
                    data = json.loads(res.stdout or "{}")
                    streams = data.get("streams", [])
                    if streams:
                        s = streams[0]
                        info["width"] = int(s.get("width", 1920))
                        info["height"] = int(s.get("height", 1080))
                        # r_frame_rate e.g. "30/1" or "2997/100"
                        rate_str = s.get("r_frame_rate", "30/1")
                        if "/" in rate_str:
                            num, den = rate_str.split("/")
                            info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
                        else:
                            info["fps"] = float(rate_str)
                        info["frame_count"] = int(s.get("nb_frames", 0))
                        return info
                except Exception:
                    pass

        # 3. Fall back to parsing ffmpeg -i stderr
        try:
            res = subprocess.run([ffmpeg_bin, "-i", str(video_path)], capture_output=True, text=True, timeout=15)
            match = re.search(r"Stream.*Video:.*?(\d{2,5})x(\d{2,5})", res.stderr)
            if match:
                info["width"] = int(match.group(1))
                info["height"] = int(match.group(2))
            fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", res.stderr)
            if fps_match:
                info["fps"] = float(fps_match.group(1))
        except Exception:
            pass

    return info


class FlowSampler:
    """Collects sparse optical flow samples across the stream using O(1) memory."""

    def __init__(self, target_samples: int = 30, total_hint: int = 300) -> None:
        self.sample_interval = max(5, total_hint // target_samples) if total_hint > 0 else 10
        self.flow_magnitudes: list[float] = []
        self.prev_gray: np.ndarray | None = None
        self.armed_idx: int | None = None

    def observe(self, idx: int, gray: np.ndarray) -> None:
        if self.armed_idx is not None and idx == self.armed_idx + 1:
            if self.prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=5)
                if pts is not None and len(pts) > 0:
                    pts2, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
                    if pts2 is not None:
                        good = status.flatten() == 1
                        if good.any():
                            diffs = pts2[good] - pts[good]
                            mag = float(np.linalg.norm(diffs, axis=1).mean())
                            self.flow_magnitudes.append(mag)
            self.prev_gray = None
            self.armed_idx = None
        elif idx % self.sample_interval == 0:
            self.prev_gray = gray.copy()
            self.armed_idx = idx

    def compute_stride(self, frame_height: int) -> int:
        if not self.flow_magnitudes:
            return 10
        median_flow = float(np.median(self.flow_magnitudes))
        target_displacement = frame_height * 0.25
        if median_flow < 0.5:
            return 30  # camera nearly static
        return max(2, int(round(target_displacement / median_flow)))


# ---------------------------------------------------------------------------
# Telemetry alignment
# ---------------------------------------------------------------------------

def _load_telemetry(path: str | Path) -> list[dict]:
    from .telemetry import load as tel_load
    return tel_load(path)


def _align_telemetry_to_frames(
    fixes: list[dict],
    frame_timestamps: list[float],
    time_offset_s: float = 0.0,
) -> list[dict | None]:
    """For each frame timestamp, find the nearest telemetry fix.

    Returns a list parallel to ``frame_timestamps`` — ``None`` where no fix is
    close enough (> 2 seconds gap).
    """
    if not fixes:
        return [None] * len(frame_timestamps)

    fix_times = [f["timestamp_s"] + time_offset_s for f in fixes]
    result: list[dict | None] = []
    for ts in frame_timestamps:
        idx = int(np.searchsorted(fix_times, ts))
        candidates = []
        for ci in (idx - 1, idx):
            if 0 <= ci < len(fixes):
                candidates.append((abs(fix_times[ci] - ts), ci))
        if not candidates:
            result.append(None)
            continue
        best_dt, best_ci = min(candidates)
        result.append(fixes[best_ci] if best_dt <= 2.0 else None)
    return result


# ---------------------------------------------------------------------------
# Stride estimation
# ---------------------------------------------------------------------------

def _estimate_stride_from_speed(
    speed_mps: float,
    fps: float,
    altitude_m: float,
    target_overlap: float,
) -> int:
    """How many frames to skip to achieve ``target_overlap``."""
    if fps <= 0 or speed_mps <= 0:
        return 10
    footprint_m = altitude_m * 0.8  # rough nadir footprint assumption
    overlap_fraction = min(max(target_overlap, 0.3), 0.95)
    stride_m = footprint_m * (1 - overlap_fraction)
    stride_frames = stride_m / speed_mps * fps
    return max(2, int(round(stride_frames)))


def _estimate_stride_from_turn_rate(
    aligned_fixes: list[dict | None],
    fps: float,
    max_turn_deg: float,
    accuracy_m: float = 2.0,
) -> tuple[int | None, float]:
    """Bound the stride by how fast the platform is *turning*.

    Overlap and viewpoint change are the same thing only for a camera flying in
    a straight line.  On an orbit they come apart: the gimbal tracks the target,
    so the subject barely moves in the image and every image-space estimator
    says "safe to thin aggressively" - while the true viewing direction sweeps
    degrees per frame.  Thin on that advice and consecutive keyframes are tens
    of degrees apart, which is where incremental SfM stops being able to match
    them.  This is the failure mode that left the previous fixture registering
    3 of 30 images.

    The turn is measured from the *shape* of the GPS track rather than by summing
    per-step heading changes.  Summing headings is the obvious approach and it
    does not work: heading error goes as (GNSS error / step length), so on a
    9 m step with 2.5 m accuracy each increment carries ~20 deg of noise and the
    sum accumulates it linearly - it reported 4 deg/frame on a track genuinely
    turning at 1.2.  Path length and the widest chord are instead sums and
    extremes over hundreds of metres, where the same noise is a few percent.
    For an arc of sweep ``theta``, ``path / chord`` is a monotone function of
    ``theta`` alone, independent of radius, so inverting it recovers the sweep.

    Returns ``(stride, deg_per_frame)``, or ``(None, 0.0)`` when the track is too
    short to say anything.
    """
    idx_fix = [(i, f) for i, f in enumerate(aligned_fixes)
               if f and f.get("lat") is not None and f.get("lon") is not None]
    if len(idx_fix) < 12 or fps <= 0:
        return None, 0.0

    lat0 = float(np.median([f["lat"] for _i, f in idx_fix]))
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    lon0, lat_ref = idx_fix[0][1]["lon"], idx_fix[0][1]["lat"]
    track = [
        (i, (f["lon"] - lon0) * m_per_deg_lon, (f["lat"] - lat_ref) * m_per_deg_lat)
        for i, f in idx_fix
    ]

    # Accumulate into segments several times the GNSS error, so summed path
    # length is not inflated by per-step jitter.
    min_seg_m = max(4.0 * max(accuracy_m, 0.5), 8.0)
    seg_stride = max(1, int(round(fps)))
    nodes = track[::seg_stride]
    for _attempt in range(6):
        if len(nodes) < 6:
            break
        steps = [math.dist((a[1], a[2]), (b[1], b[2])) for a, b in zip(nodes, nodes[1:])]
        if float(np.median(steps)) >= min_seg_m:
            break
        seg_stride *= 2
        nodes = track[::seg_stride]
    if len(nodes) < 6:
        return None, 0.0

    path_m = sum(math.dist((a[1], a[2]), (b[1], b[2])) for a, b in zip(nodes, nodes[1:]))
    chord_m = max(
        math.dist((a[1], a[2]), (b[1], b[2]))
        for k, a in enumerate(nodes) for b in nodes[k + 1:]
    )
    spanned = nodes[-1][0] - nodes[0][0]
    if path_m < 20.0 or chord_m < min_seg_m or spanned <= 0:
        return None, 0.0          # too little travel to infer a shape

    ratio = path_m / chord_m
    if ratio <= 1.05:
        return None, 0.0          # straight: no angular constraint to add

    # Invert ratio = theta / (2 sin(theta/2)) for theta <= pi; beyond a
    # half-turn the chord saturates at the diameter and ratio = theta / 2.
    if ratio >= math.pi / 2.0:
        theta = 2.0 * ratio
    else:
        lo, hi = 1e-4, math.pi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mid / (2.0 * math.sin(mid / 2.0)) < ratio:
                lo = mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)

    deg_per_frame = math.degrees(theta) / spanned
    if deg_per_frame <= 1e-3:
        return None, deg_per_frame
    return max(2, int(round(max_turn_deg / deg_per_frame))), deg_per_frame


# ---------------------------------------------------------------------------
# Main entry point — Streaming 2-Pass Execution
# ---------------------------------------------------------------------------

def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    cfg = config.frames
    video_path_str: str = ws.manifest.get("input", {}).get("video", "")
    if not video_path_str:
        raise RuntimeError("No video path in run manifest. Pass --video when starting a new run.")
    video_path = Path(video_path_str)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path}")

    ws.images_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous selection. Keyframe names encode the source frame
    # index, so a re-run with different settings does not overwrite the old
    # set - it unions with it, and SfM then runs on a mixture of two selection
    # policies with no record that it happened.
    stale = sorted(ws.images_dir.glob("frame_*.jpg"))
    for old in stale:
        old.unlink()
    if stale:
        ctx.note(f"cleared {len(stale)} keyframes from a previous selection")

    ffmpeg_bin = str(tools.ffmpeg.binary) if tools.ffmpeg else "ffmpeg"
    video_info = _probe_video(video_path, ffmpeg_bin)
    fps = float(video_info.get("fps", 30.0))
    width = int(video_info.get("width", 1920))
    height = int(video_info.get("height", 1080))
    total_frames_hint = int(video_info.get("frame_count", 0))

    # Compute target resolution scaling for analysis & extraction
    long_edge = max(width, height)
    if long_edge > cfg.max_long_edge:
        scale = cfg.max_long_edge / long_edge
        out_w = int(math.floor(width * scale / 2) * 2)
        out_h = int(math.floor(height * scale / 2) * 2)
    else:
        out_w, out_h = width, height

    # Choose decoder backend
    use_opencv = True
    if cfg.decoder == "ffmpeg":
        use_opencv = False
    elif cfg.decoder in ("auto", "opencv"):
        cap_test = cv2.VideoCapture(str(video_path))
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            cap_test.release()
            if not ret:
                use_opencv = False
        else:
            use_opencv = False

    ctx.note(f"video: {width}x{height} @ {fps:.1f} fps (decoder: {'opencv' if use_opencv else 'ffmpeg'})")

    # ==================================================================
    # Pass 1 — Analysis (Zero frame storage in RAM)
    # ==================================================================
    ctx.note("Pass 1: Streaming sharpness and optical flow analysis...")
    sharpness_scores: list[float] = []
    flow_sampler = FlowSampler(target_samples=30, total_hint=total_frames_hint)

    if use_opencv:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if long_edge > cfg.max_long_edge:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness_scores.append(_laplacian_variance(gray))
            flow_sampler.observe(idx, gray)
            idx += 1
        cap.release()
    else:
        vf = f"scale={out_w}:{out_h}" if long_edge > cfg.max_long_edge else "null"
        cmd = [
            ffmpeg_bin, "-v", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-pix_fmt", "gray",
            "-f", "rawvideo",
            "-",
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert process.stdout
        frame_bytes = out_w * out_h
        idx = 0
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            gray = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w))
            sharpness_scores.append(_laplacian_variance(gray))
            flow_sampler.observe(idx, gray)
            idx += 1
        process.wait()

    n_total = len(sharpness_scores)
    if n_total == 0:
        raise RuntimeError("No frames could be decoded from the video.")

    ctx.metric(n_frames_decoded=n_total)
    duration_s = n_total / fps if fps > 0 else 0.0
    ctx.metric(video_duration_s=round(duration_s, 2))

    median_sharpness = float(np.median(sharpness_scores))
    ctx.metric(median_sharpness=round(median_sharpness, 2))

    # ==================================================================
    # Keyframe Selection
    # ==================================================================
    # Window-based best-frame selection
    target_min_candidates = min(n_total, 25)
    window = max(1, cfg.sharpness_window)
    if n_total // window < target_min_candidates:
        window = max(1, n_total // target_min_candidates)

    window_best: list[int] = []
    for start in range(0, n_total, window):
        end = min(start + window, n_total)
        best_in_window = max(range(start, end), key=lambda i: sharpness_scores[i])
        window_best.append(best_in_window)

    min_sharpness = median_sharpness * cfg.min_sharpness_ratio
    window_best = [i for i in window_best if sharpness_scores[i] >= min_sharpness]

    if not window_best:
        raise RuntimeError(
            "All frames failed the sharpness threshold — the video may be extremely blurry. "
            f"Try lowering `frames.min_sharpness_ratio` (currently {cfg.min_sharpness_ratio})."
        )

    # Telemetry parsing & alignment
    telemetry_raw_path = ws.manifest.get("input", {}).get("telemetry")
    fixes: list[dict] = []
    if telemetry_raw_path and Path(telemetry_raw_path).exists():
        try:
            fixes = _load_telemetry(telemetry_raw_path)
            ws.telemetry_json.write_text(
                json.dumps(fixes, indent=2, default=str), encoding="utf-8"
            )
            ctx.note(f"telemetry: {len(fixes)} fixes from {Path(telemetry_raw_path).name}")
            # A fabricated timebase is the quietest way to lose telemetry: the
            # 10 Hz fallback ends before a slower real log does, and every fix
            # past its end is dropped by frame alignment with nothing to show
            # for it. Say so, and say how far it reaches.
            if fixes and fixes[0].get("time_source") == "synthetic_10hz":
                ctx.note(
                    f"! no recognised time column - timestamps assumed at 10 Hz, "
                    f"spanning {fixes[-1]['timestamp_s']:.1f} s for a "
                    f"{n_total / fps:.1f} s video. Fixes are aligned on this "
                    f"assumption and may not correspond to the right frames."
                )
        except Exception as exc:
            ctx.note(f"telemetry parse failed: {exc}; continuing without GPS data")

    frame_timestamps = [idx / fps for idx in range(n_total)]
    aligned_fixes = _align_telemetry_to_frames(fixes, frame_timestamps, config.georef.time_offset_s)
    if fixes:
        covered = sum(1 for f in aligned_fixes if f)
        if covered < 0.8 * n_total:
            ctx.note(
                f"! telemetry covers only {covered}/{n_total} decoded frames "
                f"({covered / max(n_total, 1):.0%}); georeferencing will rest on "
                f"the covered span only. Check the log's time column and "
                f"georef.time_offset_s."
            )

    # Baseline stride estimation
    #
    # Three independent bounds, and the densest wins. Each catches a different
    # way of thinning a good capture into an unreconstructable one:
    #
    #   overlap  ground footprint from height and speed. Needs height *above
    #            ground*: a log reporting MSL (a 200-300 m datum is ordinary)
    #            inflates the footprint several-fold and the stride with it, so
    #            only a genuine relative altitude is used here.
    #   flow     apparent image motion. Scale-free, so immune to the datum
    #            question - but blind to a tracking gimbal.
    #   turn     course change from the GPS track. The only one of the three
    #            that sees an orbit, where the gimbal holds the subject still in
    #            frame while the viewing direction sweeps.
    flow_stride = flow_sampler.compute_stride(out_h)
    accuracies = [f["accuracy_m"] for f in aligned_fixes
                  if f and f.get("accuracy_m")] if fixes else []
    turn_stride, turn_deg_per_frame = _estimate_stride_from_turn_rate(
        aligned_fixes, fps, cfg.max_keyframe_turn_deg,
        accuracy_m=float(np.median(accuracies)) if accuracies else 2.0,
    ) if fixes else (None, 0.0)

    if fixes and any(f and f.get("speed_mps") for f in aligned_fixes):
        speeds = [f["speed_mps"] for f in aligned_fixes if f and f.get("speed_mps")]
        agl = [f["agl_m"] for f in aligned_fixes if f and f.get("agl_m")]
        alts = [f["alt_m"] for f in aligned_fixes if f and f.get("alt_m")]
        avg_speed = float(np.median(speeds))
        if agl:
            avg_alt, alt_note = float(np.median(agl)), "AGL"
        elif alts:
            # Absolute altitude may be an MSL datum rather than flying height.
            # Trust it, but let the other two bounds override an implausible
            # stride rather than thinning on a number that may be 5x too large.
            avg_alt, alt_note = float(np.median(alts)), "absolute (may include datum)"
        else:
            avg_alt, alt_note = 50.0, "assumed"
        gps_stride = _estimate_stride_from_speed(avg_speed, fps, avg_alt, cfg.target_overlap)
        stride = min(gps_stride, flow_stride)
        detail = (f"stride from GPS speed: {avg_speed:.1f} m/s, "
                  f"alt: {avg_alt:.0f} m {alt_note} -> {gps_stride}")
        if stride != gps_stride:
            detail += f"; capped by optical flow -> {stride}"
        if turn_stride is not None and turn_stride < stride:
            stride = turn_stride
            detail += (f"; capped by turn rate {turn_deg_per_frame:.2f} deg/frame "
                       f"-> {stride}")
        ctx.note(detail)
    else:
        stride = flow_stride
        detail = f"stride from optical flow -> {stride}"
        if turn_stride is not None and turn_stride < stride:
            stride = turn_stride
            detail += (f"; capped by turn rate {turn_deg_per_frame:.2f} deg/frame "
                       f"-> {stride}")
        ctx.note(detail)

    stride = max(cfg.min_frame_stride, min(cfg.max_frame_stride, stride))

    # `target_overlap` is silently bounded by `sharpness_window`, and the run
    # must say so rather than let the config imply an overlap it never got.
    #
    # Windowed best-frame selection above already spaced the candidates `window`
    # frames apart, so `apply_stride` can only ever thin them *further*. When the
    # requested stride is the smaller of the two it does nothing at all: asking
    # for 90% overlap on this project's orbit fixture computed a stride of 3,
    # but a `sharpness_window` of 8 held the real spacing at 8 and produced 55
    # keyframes instead of ~160 -- a 3x shortfall in the dense sampling that the
    # reported GSD is derived from, with nothing in the run saying the request
    # had been overridden.
    #
    # Report the overlap the spacing actually implies, inverting the same
    # footprint model `_estimate_stride_from_speed` used, so the two numbers are
    # comparable rather than merely adjacent.
    effective_spacing = max(window, stride)
    if stride < window:
        achieved = ""
        if fixes and any(f and f.get("speed_mps") for f in aligned_fixes):
            footprint_m = avg_alt * 0.8
            if footprint_m > 0 and fps > 0:
                advance_m = effective_spacing / fps * avg_speed
                achieved = (f"; achieved overlap ~{max(0.0, 1.0 - advance_m / footprint_m):.2f} "
                            f"vs requested {cfg.target_overlap:.2f}")
        ctx.note(
            f"sharpness_window ({window}) exceeds the overlap stride ({stride}), so it - "
            f"not target_overlap - sets keyframe spacing{achieved}. "
            f"Lower frames.sharpness_window to honour the requested overlap."
        )

    def apply_stride(candidates: list[int], frame_stride: int) -> list[int]:
        chosen: list[int] = []
        for idx_val in candidates:
            if not chosen or idx_val - chosen[-1] >= frame_stride:
                chosen.append(idx_val)
        return chosen

    strided: list[int] = apply_stride(window_best, stride)

    # Safety floor: COLMAP needs at least ~20 frames to find an initial pair
    min_useful = min(n_total, 20)
    while len(strided) < min_useful and stride > 1:
        stride -= 1
        strided = apply_stride(window_best, stride)
    if len(strided) < min_useful:
        strided = window_best

    if len(strided) > cfg.max_keyframes:
        step = len(strided) / cfg.max_keyframes
        strided = [strided[int(round(i * step))] for i in range(cfg.max_keyframes)]

    ctx.metric(
        n_keyframes=len(strided),
        mean_sharpness=round(float(np.mean([sharpness_scores[i] for i in strided])), 2),
    )

    # ==================================================================
    # Pass 2 — Extraction (Direct-to-Disk)
    # ==================================================================
    ctx.note(f"Pass 2: Extracting {len(strided)} keyframes to disk...")
    target_indices = set(strided)
    extracted_frames: dict[int, np.ndarray] = {}

    if use_opencv:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            if idx in target_indices:
                ret, frame = cap.read()
                if not ret:
                    break
                extracted_frames[idx] = frame
            else:
                ret = cap.grab()
                if not ret:
                    break
            idx += 1
            if len(extracted_frames) == len(target_indices):
                break
        cap.release()
    else:
        vf = f"scale={out_w}:{out_h}" if long_edge > cfg.max_long_edge else "null"
        cmd = [
            ffmpeg_bin, "-v", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-",
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert process.stdout
        frame_bytes = out_w * out_h * 3
        idx = 0
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            if idx in target_indices:
                bgr = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
                extracted_frames[idx] = bgr
            idx += 1
            if len(extracted_frames) == len(target_indices):
                break
        process.kill()
        process.wait()

    keyframe_records: list[dict] = []
    for rank, frame_idx in enumerate(strided):
        bgr = extracted_frames.get(frame_idx)
        if bgr is None:
            continue
        bgr = _resize_to_long_edge(bgr, cfg.max_long_edge)

        gray_eval = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if cfg.clahe:
            enhanced = _clahe(gray_eval, cfg.clahe_clip)
            bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        stem = f"frame_{frame_idx:06d}"
        out_path = ws.images_dir / f"{stem}.jpg"
        cv2.imwrite(
            str(out_path),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
        )

        exp_score = _calculate_exposure_score(gray_eval)
        cont_score = _calculate_contrast_score(gray_eval)
        ent_score = _calculate_entropy_score(gray_eval)

        fix = aligned_fixes[frame_idx] if frame_idx < len(aligned_fixes) else None
        record: dict = {
            "rank": rank,
            "frame_idx": frame_idx,
            "timestamp_s": round(frame_timestamps[frame_idx], 4),
            "path": str(out_path),
            "sharpness": round(sharpness_scores[frame_idx], 2),
            "exposure_score": exp_score,
            "contrast_score": cont_score,
            "entropy_score": ent_score,
            "lat": fix["lat"] if fix else None,
            "lon": fix["lon"] if fix else None,
            "alt_m": fix["alt_m"] if fix else None,
            "speed_mps": fix["speed_mps"] if fix else None,
            "gimbal_pitch": fix["gimbal_pitch"] if fix else None,
        }
        keyframe_records.append(record)

    ws.frames_index.write_text(json.dumps(keyframe_records, indent=2, default=str), encoding="utf-8")
    ctx.output(images_dir=str(ws.images_dir), frames_index=str(ws.frames_index))

    # Update GNSS mode in workspace
    n_with_gps = sum(1 for r in keyframe_records if r["lat"] is not None)
    if n_with_gps > 0:
        from .workspace import GnssMode
        if ws.gnss_mode is GnssMode.NONE:
            ws.set_gnss(GnssMode.STANDALONE, source=str(telemetry_raw_path), reason="telemetry sidecar parsed")
        ctx.note(f"{n_with_gps}/{len(keyframe_records)} keyframes have GPS fixes")
    else:
        ctx.note("no GPS fixes aligned to keyframes — up-to-scale mode")
