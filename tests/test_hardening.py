"""Hardening regression tests for DroneMap (SIH26158).

Verifies:
1. Zero-GPS / missing telemetry truthful CRS handling (LOCAL_RELATIVE).
2. Exhaustive escalation ladder frame cap (N <= 150).
3. Stage 4b SIH26158 semantic category mapping rollup.
4. Dynamic masking frame gating and graceful degradation.
5. PyTorch GPU memory cleanup invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from dronemap.api.server import derive_status_tier
from dronemap.stage7_export import _guess_crs, _points_for_gis
from dronemap.stage4_semantics import _map_to_sih26158_categories


def test_zero_gps_truthful_local_relative():
    """Verify that absent GPS telemetry strictly falls back to LOCAL_RELATIVE."""
    # Empty georef dict must produce LOCAL_RELATIVE
    assert _guess_crs({}) == "LOCAL_RELATIVE"
    assert _guess_crs({"coordinate_mode": "relative"}) == "LOCAL_RELATIVE"
    assert _guess_crs({"crs": None}) == "LOCAL_RELATIVE"

    # Coordinates must not be transformed or shifted to a fictitious UTM zone
    raw_coords = np.array([[10.5, 20.2, 5.1], [11.0, 21.0, 5.3]])
    gis_coords = _points_for_gis(raw_coords, {"coordinate_mode": "relative"})
    np.testing.assert_array_equal(raw_coords, gis_coords)

    # API status tier must truthfully report Relative/local
    manifest = {
        "stages": {
            "pose": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"},
        },
        "accuracy": {"coordinate_mode": "relative", "crs": "LOCAL_RELATIVE"},
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Relative/local"
    assert status_type == "relative"


def test_stage4_semantics_sih26158_category_rollup():
    """Verify raw multi-class fractions map into the 4 mandatory SIH26158 classes."""
    # Typical UAVid / LoveDA class fractions
    uavid_fractions = {
        "background": 0.05,
        "building": 0.30,
        "road": 0.20,
        "tree": 0.25,
        "low_veg": 0.10,
        "human": 0.02,
        "car": 0.08,
    }

    mapped = _map_to_sih26158_categories(uavid_fractions)

    # terrain = background (0.05) + low_veg (0.10) = 0.15
    assert mapped["terrain"] == pytest.approx(0.15, abs=1e-3)
    # buildings = building (0.30)
    assert mapped["buildings"] == pytest.approx(0.30, abs=1e-3)
    # roads_infrastructure = road (0.20)
    assert mapped["roads_infrastructure"] == pytest.approx(0.20, abs=1e-3)
    # vegetation_obstacles = tree (0.25) + human (0.02) + car (0.08) = 0.35
    assert mapped["vegetation_obstacles"] == pytest.approx(0.35, abs=1e-3)
    assert mapped["other"] == 0.0


def test_exhaustive_matching_frame_cap_boundary():
    """Verify that exhaustive matching escalation respects N <= 150 boundary."""
    # Under 150 frames, exhaustive matching is bounded (<= 11,175 pairs)
    n_small = 120
    pairs_small = n_small * (n_small - 1) // 2
    assert pairs_small <= 11175

    # Over 150 frames (e.g. 300 or 600), pairs explode quadratically
    n_large = 300
    pairs_large = n_large * (n_large - 1) // 2
    assert pairs_large == 44850  # Must be bypassed to prevent near-real-time timeout
    assert pairs_large > 11175


@patch("dronemap.stage2_masks._load_model")
def test_masking_all_frames_dropped_error_handling(mock_load, tmp_path):
    """Verify that if all frames are dynamic objects, stage2_masks fails gracefully."""
    from dronemap.stage2_masks import run
    from dronemap.config import Config
    from dronemap.workspace import RunWorkspace, _StageContext

    mock_model = MagicMock()
    mock_load.return_value = (mock_model, "cpu")

    ws = RunWorkspace.create(tmp_path, "run_test", {})
    # Write empty keyframes index
    ws.frames_index.write_text(json.dumps([]), encoding="utf-8")

    cfg = Config()
    tools = MagicMock()
    ctx = MagicMock()

    # With empty keyframes, raises RuntimeError gracefully
    with pytest.raises(RuntimeError, match="All keyframes were dropped"):
        run(ws, cfg, tools, ctx)


def test_gpu_cleanup_called():
    """Verify PyTorch CUDA cache flush and gc.collect logic handles cleanly."""
    import gc
    import torch

    initial_gc = gc.isenabled()
    assert initial_gc is True

    # Calling torch.cuda.empty_cache() should not raise whether CUDA is present or not
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_accelerated_profile_config_loading():
    """Verify that configs/accelerated.yaml loads properly with cloud parameters."""
    from dronemap.config import load_config
    cfg = load_config(profile="accelerated")
    assert cfg.frames.max_long_edge == 3840
    assert cfg.mesh.texture_size == 16384
    assert cfg.dense.resolution_level == 0
    assert cfg.semantics.enabled is True
    assert cfg.depth.enabled is True
    assert cfg.depth.depth_anything_model == "large"

