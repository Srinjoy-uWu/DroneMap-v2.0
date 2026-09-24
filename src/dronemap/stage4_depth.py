"""Stage 4a — Monocular metric depth estimation (Stretch).

Produces per-frame depth maps that downstream stages can use to:
  - Provide geometric priors for georeferencing (scale disambiguation)
  - Export as a secondary dense product alongside the MVS cloud
  - Fuse with the dense point cloud for gap-filling in occluded areas

Two backends
------------
``depth_anything``
    Depth Anything V2 (Hugging Face hub: ``depth-anything/Depth-Anything-V2-Small-hf``
    and siblings).  Produces *relative* depth — the scale is unknown without
    external reference.  We scale it to metric using the telemetry altitude
    (nadir-only approximation: depth ≈ altitude / cos(gimbal_pitch)).  This
    works well for nadir imagery but degrades on oblique passes.

``metric3d``
    Metric3D v2 (``JUGGHNU/metric3d_vit_small``).  Produces *absolute* metric
    depth from a single image using a universal prior, without telemetry.
    Higher quality, especially on oblique views, but ~3× slower.

Output format
-------------
Depth maps are saved as 16-bit PNGs (millimetres, uint16).  A value of 0
means invalid / no depth.  Max representable depth = 65.535 m — sufficient
for drone altitude.  The companion ``depth_scale.json`` records the per-frame
altitude used for Depth Anything scaling (not applicable for Metric3D).

Outputs
-------
``ws.stage_dir("depth")/raw/<stem>_depth.png``   — 16-bit depth PNG (mm)
``ws.stage_dir("depth")/depth_index.json``       — per-frame metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .tools import ToolRegistry
    from .workspace import RunWorkspace, _StageContext


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

_DA_HUB = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _run_depth_anything(
    images_bgr: list[np.ndarray],
    model_size: str,
    device: str,
) -> list[np.ndarray]:
    """Return relative depth maps, float32, same H×W as inputs (0=nearest, 1=farthest)."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise RuntimeError("transformers is required for Depth Anything V2 — run `uv sync --extra ml`") from exc

    hub_id = _DA_HUB.get(model_size, _DA_HUB["small"])
    pipe = hf_pipeline(
        task="depth-estimation",
        model=hub_id,
        device=0 if device == "cuda" else -1,
    )

    results = []
    for bgr in images_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)
        out = pipe(pil)
        depth_arr = np.array(out["depth"], dtype=np.float32)
        # Normalise to [0, 1]
        dmin, dmax = depth_arr.min(), depth_arr.max()
        if dmax > dmin:
            depth_arr = (depth_arr - dmin) / (dmax - dmin)
        results.append(depth_arr)

    return results


def _scale_relative_depth(
    rel_depth: np.ndarray,
    altitude_m: float,
    gimbal_pitch_deg: float | None,
) -> np.ndarray:
    """Scale relative depth (0–1) to metric (metres) using nadir altitude.

    For nadir imagery: slant range ≈ altitude / cos(pitch).
    ``rel_depth == 1`` corresponds to the ground (furthest from camera).
    """
    if altitude_m <= 0:
        return rel_depth * 0.0  # unknown scale
    pitch_rad = abs((gimbal_pitch_deg or -90.0) * np.pi / 180.0)
    # cos(90°) = 0 for straight-down; keep nadir_range = altitude
    nadir_range = altitude_m / max(abs(np.cos(pitch_rad)), 0.01)
    # rel_depth=1 → ground at nadir_range; rel_depth=0 → camera (0 m)
    metric = rel_depth * nadir_range
    return metric.astype(np.float32)


# ---------------------------------------------------------------------------
# Metric3D v2
# ---------------------------------------------------------------------------

_M3D_HUB = {
    "vit_small": "JUGGHNU/metric3d_vit_small",
    "vit_large": "JUGGHNU/metric3d_vit_large",
}


def _run_metric3d(
    images_bgr: list[np.ndarray],
    model_key: str,
    device: str,
) -> list[np.ndarray]:
    """Return absolute metric depth (metres, float32) for each image."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers+torch required for Metric3D — run `uv sync --extra ml`") from exc

    hub_id = _M3D_HUB.get(model_key, _M3D_HUB["vit_small"])
    processor = AutoImageProcessor.from_pretrained(hub_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(hub_id, trust_remote_code=True).to(device)
    model.eval()

    results = []
    for bgr in images_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Metric3D returns depth in metres, shape (1, 1, H, W)
        depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        # Resize to original resolution if needed
        h, w = bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        results.append(depth)

    return results


# ---------------------------------------------------------------------------
# Depth PNG writer (16-bit, millimetres)
# ---------------------------------------------------------------------------

def _save_depth_png(depth_m: np.ndarray, out_path: Path) -> None:
    """Save float32 depth (metres) as uint16 PNG (millimetres, max 65.535 m)."""
    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_mm)


def _load_depth_png(path: Path) -> np.ndarray:
    """Load a 16-bit depth PNG back to float32 metres."""
    mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return mm / 1000.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    cfg = config.depth

    if not ws.frames_index.exists():
        raise RuntimeError("keyframes.json not found — run stage 'frames' first.")
    keyframes: list[dict] = json.loads(ws.frames_index.read_text(encoding="utf-8"))

    depth_dir = ws.stage_dir("depth")
    raw_dir = depth_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(cfg.device)
    ctx.note(f"depth backend: {cfg.backend}, device: {device}")

    stride = max(1, cfg.stride)
    selected_kfs = keyframes[::stride]
    ctx.note(f"running depth on {len(selected_kfs)}/{len(keyframes)} frames (stride={stride})")

    depth_index: list[dict] = []
    total_valid_px = 0
    mean_depths: list[float] = []

    # Process in batches of up to 8 (VRAM budget)
    batch_size = 4 if device == "cuda" else 1
    kf_batches = [selected_kfs[i:i + batch_size] for i in range(0, len(selected_kfs), batch_size)]

    try:
        for batch in kf_batches:
            images_bgr: list[np.ndarray] = []
            valid_kfs: list[dict] = []

            for kf in batch:
                img_path = Path(kf["path"])
                if not img_path.exists():
                    continue
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                images_bgr.append(bgr)
                valid_kfs.append(kf)

            if not images_bgr:
                continue

            # Run depth estimation
            if cfg.backend == "depth_anything":
                rel_depths = _run_depth_anything(images_bgr, cfg.depth_anything_model, device)
                depth_maps: list[np.ndarray] = []
                for i, rel in enumerate(rel_depths):
                    kf = valid_kfs[i]
                    alt = kf.get("alt_m") or 50.0
                    pitch = kf.get("gimbal_pitch")
                    depth_maps.append(_scale_relative_depth(rel, alt, pitch))
            else:
                depth_maps = _run_metric3d(images_bgr, cfg.metric3d_model, device)

            # Save outputs
            for kf, depth_m in zip(valid_kfs, depth_maps):
                stem = Path(kf["path"]).stem
                out_path = raw_dir / f"{stem}_depth.png"

                if cfg.save_depth_png:
                    _save_depth_png(depth_m, out_path)

                valid_mask = depth_m > 0.01
                n_valid = int(valid_mask.sum())
                total_valid_px += n_valid
                mean_d = float(depth_m[valid_mask].mean()) if n_valid else 0.0
                mean_depths.append(mean_d)

                depth_index.append({
                    "frame_idx": kf.get("frame_idx"),
                    "stem": stem,
                    "depth_png": str(out_path) if cfg.save_depth_png else None,
                    "backend": cfg.backend,
                    "mean_depth_m": round(mean_d, 3),
                    "n_valid_px": n_valid,
                    "alt_m": kf.get("alt_m"),
                })
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

    (depth_dir / "depth_index.json").write_text(
        json.dumps(depth_index, indent=2, default=str), encoding="utf-8"
    )

    overall_mean = float(np.mean(mean_depths)) if mean_depths else 0.0
    ctx.metric(
        n_frames_with_depth=len(depth_index),
        backend=cfg.backend,
        mean_depth_m=round(overall_mean, 2),
        total_valid_px=total_valid_px,
    )
    ctx.output(
        depth_dir=str(depth_dir),
        depth_index=str(depth_dir / "depth_index.json"),
    )
    ctx.note(
        f"mean scene depth: {overall_mean:.1f} m — "
        + ("consistent with GPS altitude ✓" if abs(overall_mean - (depth_index[0].get('alt_m') or 0)) < overall_mean * 0.3 else "check GPS altitude alignment")
    )
