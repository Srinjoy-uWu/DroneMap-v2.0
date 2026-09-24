"""Stage 2 — Dynamic object masking with YOLOv8-seg.

What this stage does
--------------------
1.  Loads the YOLOv8 segmentation model (auto-download on first run).
2.  Runs inference on every keyframe.
3.  Unions the instance masks for classes in ``config.masks.dynamic_classes``.
4.  Dilates each union mask by ``config.masks.dilate_px`` pixels so that the
    soft edge of a moving object (motion blur halo) is also excluded.
5.  Frames where the mask covers more than ``config.masks.max_masked_fraction``
    of the image are *dropped* — they contribute nothing useful to SfM.
6.  Saves one binary PNG mask per surviving keyframe (255 = masked / excluded,
    0 = valid), matching the filename stem of the keyframe image.

Why we keep PNG masks rather than applying them in-place
---------------------------------------------------------
COLMAP's ``feature_extractor`` accepts a ``--ImageReader.mask_path`` directory
and reads same-stem PNG masks from it directly — no in-place modification of
the source images.  This means Stage 1 images are never mutated and the masks
can be regenerated cheaply if the config changes.

Outputs
-------
``ws.masks_dir/<stem>.png``  — one binary mask per surviving keyframe
``ws.frames_index``          — updated to remove dropped frames
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .tools import ToolRegistry
    from .workspace import RunWorkspace, _StageContext


def _load_model(model_name: str, device: str):
    """Load a YOLOv8 segmentation model, picking device automatically."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for dynamic masking. "
            "Run `uv sync --extra ml` to install it."
        ) from exc

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model = YOLO(model_name)
    model.to(device)
    return model, device


def _build_dynamic_mask(
    result,
    dynamic_classes: set[str],
    conf_threshold: float,
    image_hw: tuple[int, int],
    dilate_px: int,
    speed_mps: float = 0.0,
) -> np.ndarray:
    """Build a binary mask from a YOLOv8 result with velocity-adaptive dilation.

    Returns uint8 array (same size as image) with 255 where an object from
    ``dynamic_classes`` was detected with confidence >= conf_threshold, 0
    elsewhere.
    """
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return mask

    names: dict[int, str] = result.names

    for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
        conf = float(result.boxes.conf[i].cpu())
        if conf < conf_threshold:
            continue
        cls_name = names.get(cls_id, "")
        if cls_name.lower() not in dynamic_classes:
            continue

        seg = result.masks.data[i].cpu().numpy()
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[seg > 0.5] = 255

    # Velocity-adaptive dilation: expand kernel for fast moving UAV / vehicles
    effective_dilate = dilate_px
    if speed_mps > 0 and dilate_px > 0:
        effective_dilate = int(round(dilate_px + min(dilate_px * 1.5, speed_mps * 0.8)))

    if effective_dilate > 0 and mask.any():
        kernel_size = 2 * effective_dilate + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel)

    return mask


def _generate_preview_overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Generate visual verification overlay blending original image with a transparent red mask overlay."""
    overlay = bgr.copy()
    if mask.any():
        red_tint = np.zeros_like(bgr)
        red_tint[:] = (0, 0, 220)  # Red in BGR
        idx = mask > 0
        overlay[idx] = cv2.addWeighted(bgr[idx], 0.55, red_tint[idx], 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    cfg = config.masks

    # Load keyframe index
    if not ws.frames_index.exists():
        raise RuntimeError("keyframes.json not found — run stage 'frames' first.")
    keyframes: list[dict] = json.loads(ws.frames_index.read_text(encoding="utf-8"))

    ws.masks_dir.mkdir(parents=True, exist_ok=True)

    dynamic_classes = {c.lower() for c in cfg.dynamic_classes}

    model, device = _load_model(cfg.model, cfg.device)
    ctx.note(f"YOLOv8 model: {cfg.model}, device: {device}")

    n_masked = 0
    n_dropped = 0
    total_fraction = 0.0
    surviving: list[dict] = []

    try:
        for kf in keyframes:
            img_path = Path(kf["path"])
            if not img_path.exists():
                ctx.note(f"skipping missing image: {img_path.name}")
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                ctx.note(f"could not read image: {img_path.name}")
                continue

            h, w = bgr.shape[:2]

            # Run inference (returns a list; we pass a single image)
            results = model(bgr, verbose=False, conf=cfg.conf)
            result = results[0]

            speed_mps = float(kf.get("speed_mps") or 0.0)
            mask = _build_dynamic_mask(
                result,
                dynamic_classes,
                conf_threshold=cfg.conf,
                image_hw=(h, w),
                dilate_px=cfg.dilate_px,
                speed_mps=speed_mps,
            )

            masked_fraction = float(mask.sum()) / (255 * h * w)
            total_fraction += masked_fraction

            if masked_fraction >= cfg.max_masked_fraction:
                # Frame is mostly masked — drop it entirely
                img_path.unlink(missing_ok=True)
                n_dropped += 1
                ctx.note(f"dropped {img_path.name}: {masked_fraction:.1%} masked")
                continue

            # In COLMAP convention for --ImageReader.mask_path:
            # >0 (255) = VALID pixels where features are extracted.
            # 0        = MASKED OUT pixels (dynamic objects) to be ignored.
            # We invert the dynamic mask so background is 255 and dynamic objects are 0.
            colmap_mask = 255 - mask

            # Save the mask PNG (COLMAP expects <image_name>.png, e.g. frame_000000.jpg.png)
            mask_path_ext = ws.masks_dir / f"{img_path.name}.png"
            mask_path_stem = ws.masks_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(mask_path_ext), colmap_mask)
            cv2.imwrite(str(mask_path_stem), colmap_mask)

            # Generate visual verification preview overlay (Before vs After)
            previews_dir = ws.masks_dir / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            preview_file = previews_dir / f"{img_path.stem}_overlay.jpg"
            overlay = _generate_preview_overlay(bgr, mask)
            cv2.imwrite(str(preview_file), overlay)

            if mask.any():
                n_masked += 1

            kf["masked_fraction"] = round(masked_fraction, 4)
            kf["preview_path"] = str(preview_file)
            surviving.append(kf)

        if not surviving:
            raise RuntimeError(
                "All keyframes were dropped by the masking stage — the scene may be "
                "entirely dynamic objects. Lower `masks.max_masked_fraction` or "
                "set `masks.enabled: false` to bypass masking."
            )

        mean_fraction = total_fraction / max(len(keyframes), 1)

        # Update keyframe index (drop removed frames)
        ws.frames_index.write_text(json.dumps(surviving, indent=2), encoding="utf-8")

        ctx.metric(
            n_keyframes_in=len(keyframes),
            n_masked_frames=n_masked,
            n_dropped_frames=n_dropped,
            n_keyframes_out=len(surviving),
            mean_masked_fraction=round(mean_fraction, 4),
        )
        ctx.output(masks_dir=str(ws.masks_dir))
    finally:
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()
