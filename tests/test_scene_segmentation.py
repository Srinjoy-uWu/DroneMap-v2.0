"""Unit tests for 5-class aerial scene segmentation (Sky, Water, Terrain, Vegetation, Structure)."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from dronemap.scene_segmentation import (
    CLASS_SKY,
    CLASS_STRUCTURE,
    CLASS_TERRAIN,
    CLASS_VEGETATION,
    CLASS_WATER,
    condition_and_classify_3d_cloud,
    render_semantic_overlay,
    segment_frame_semantics_and_depth,
)
from dronemap.stage2_masks import _build_sky_mask
from dronemap.workspace import RunWorkspace


def _make_synthetic_coastal_castle_frame(h: int = 240, w: int = 360) -> np.ndarray:
    """Create a synthetic oblique coastal castle image with Sky, Water, Vegetation, Terrain, and a 3D Castle."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    # 1. Sky at top 14% (bright pale blue-white, low saturation)
    sky_h = int(h * 0.14)
    bgr[:sky_h, :] = (235, 225, 215)

    # 2. Water in lower foreground & right bay (blue-grey, low ExG)
    bgr[sky_h:, :] = (145, 95, 65)

    # 3. Peninsula terrain strip in middle (sandy brown, ExG = 2*100 - 145 - 75 = -20)
    bgr[int(h * 0.30) : int(h * 0.68), int(w * 0.08) : int(w * 0.85)] = (75, 100, 145)

    # 4. Vegetation meadow on peninsula (high green channel -> high ExG)
    bgr[int(h * 0.38) : int(h * 0.60), int(w * 0.15) : int(w * 0.42)] = (45, 155, 65)

    # 5. Stone Castle in center with strong masonry texture/edges
    y0, y1 = int(h * 0.28), int(h * 0.56)
    x0, x1 = int(w * 0.40), int(w * 0.65)
    rng = np.random.default_rng(42)
    castle_patch = rng.integers(55, 195, size=(y1 - y0, x1 - x0), dtype=np.uint8)
    bgr[y0:y1, x0:x1, 0] = castle_patch
    bgr[y0:y1, x0:x1, 1] = castle_patch
    bgr[y0:y1, x0:x1, 2] = np.clip(castle_patch.astype(np.int32) + 12, 0, 255).astype(np.uint8)
    return bgr


def test_segment_frame_differentiates_five_classes():
    bgr = _make_synthetic_coastal_castle_frame()
    analysis = segment_frame_semantics_and_depth(bgr, use_neural=False)

    assert analysis.labels.shape == bgr.shape[:2]
    assert analysis.disparity.shape == bgr.shape[:2]
    assert set(analysis.class_fractions.keys()) == {
        "sky",
        "water",
        "terrain",
        "vegetation",
        "structure",
    }
    # Verify every class is detected in the synthetic coastal castle frame
    assert analysis.class_fractions["sky"] > 0.05
    assert analysis.class_fractions["water"] > 0.15
    assert analysis.class_fractions["terrain"] > 0.05
    assert analysis.class_fractions["vegetation"] > 0.03
    assert analysis.class_fractions["structure"] > 0.02

    # Top rows should be sky, green meadow should be vegetation, textured castle should be structure
    assert int(np.median(analysis.labels[5:15, :])) == CLASS_SKY
    assert int(np.median(analysis.labels[105:130, 70:120])) == CLASS_VEGETATION
    assert np.any(analysis.labels[75:125, 150:220] == CLASS_STRUCTURE)


def test_render_semantic_overlay_adds_legend_and_contours():
    bgr = _make_synthetic_coastal_castle_frame()
    analysis = segment_frame_semantics_and_depth(bgr, use_neural=False)
    overlay = render_semantic_overlay(
        bgr, analysis.labels, analysis.class_fractions, include_legend=True
    )
    assert overlay.shape[1] == bgr.shape[1]
    assert overlay.shape[0] > bgr.shape[0]  # includes bottom legend banner
    assert overlay.dtype == np.uint8


def test_stage2_sky_mask_captures_upper_sky_only():
    bgr = _make_synthetic_coastal_castle_frame()
    sky_mask = _build_sky_mask(bgr)
    assert sky_mask.shape == bgr.shape[:2]
    # Top sky band should be masked (> 80% True)
    assert float(sky_mask[:20, :].mean()) > 0.80
    # Lower half (ground/water/castle) must NEVER be masked by sky detector
    assert float(sky_mask[120:, :].mean()) == 0.0


def test_3d_cloud_conditioning_strips_sky_and_preserves_castle_height(tmp_path: Path):
    from dronemap.config import load_config

    ws = RunWorkspace.create(tmp_path, "castle_test", load_config())
    rng = np.random.default_rng(7)

    # 1. Sky floater points (bright white/blue-grey)
    pts_sky = np.column_stack([
        rng.uniform(-20, 20, 100),
        rng.uniform(-20, 20, 100),
        rng.uniform(15, 25, 100),
    ])
    cols_sky = np.full((100, 3), (230, 232, 240), dtype=np.uint8)

    # 2. Water points at z ~ 0.0m (blue-grey)
    pts_water = np.column_stack([
        rng.uniform(-30, 30, 300),
        rng.uniform(-30, 30, 300),
        rng.normal(0.0, 0.08, 300),
    ])
    cols_water = np.full((300, 3), (70, 95, 130), dtype=np.uint8)

    # 3. Terrain points at z ~ 1.2m (brownish)
    pts_terrain = np.column_stack([
        rng.uniform(-15, 15, 300),
        rng.uniform(-15, 15, 300),
        rng.normal(1.2, 0.15, 300),
    ])
    cols_terrain = np.full((300, 3), (140, 115, 80), dtype=np.uint8)

    # 4. Vegetation points at z ~ 1.8m (green)
    pts_veg = np.column_stack([
        rng.uniform(-12, 12, 200),
        rng.uniform(-12, 12, 200),
        rng.normal(1.8, 0.2, 200),
    ])
    cols_veg = np.full((200, 3), (55, 150, 60), dtype=np.uint8)

    # 5. Castle structure points rising z = 4.0 .. 14.0m (stone grey)
    pts_castle = np.column_stack([
        rng.uniform(-5, 5, 250),
        rng.uniform(-5, 5, 250),
        rng.uniform(4.0, 14.0, 250),
    ])
    cols_castle = np.full((250, 3), (135, 130, 125), dtype=np.uint8)

    pts = np.vstack([pts_sky, pts_water, pts_terrain, pts_veg, pts_castle])
    cols = np.vstack([cols_sky, cols_water, cols_terrain, cols_veg, cols_castle])

    res = condition_and_classify_3d_cloud(ws, pts, cols, use_neural=False)

    # All 100 sky points should be stripped
    assert res.summary["sky_points_removed"] == 100
    assert len(res.points) == 1050
    assert not np.any(res.labels == CLASS_SKY)

    # Castle structure points must have substantially higher mean elevation than terrain & water
    assert res.summary["3d_cloud_class_counts"]["structure"] >= 200
    assert (
        res.summary["mean_elevation_above_datum_m"]["structure"]
        > res.summary["mean_elevation_above_datum_m"]["terrain"] + 4.0
    )
    assert (ws.export_dir / "semantics.json").exists()
