"""Tests for dronemap.agent package (Offline Deterministic Core, Local SLM & Diagnostics)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dronemap.agent.config import AgentConfig
from dronemap.agent.client import LLMClient
from dronemap.agent.intelligence import (
    IntelligenceReport,
    generate_intelligence_report,
)
from dronemap.agent.diagnostics import (
    analyze_reconstruction_diagnostics,
    SelfHealingReport,
)
from dronemap.agent.tools import (
    calculate_3d_distance,
    explain_pipeline_stage,
    handle_spatial_chat,
)


def test_agent_config_routing():
    """Verify local SLM vs offline deterministic routing in AgentConfig."""
    cfg_offline = AgentConfig(preferred_provider="offline")
    assert cfg_offline.active_provider == "offline"
    assert cfg_offline.default_model == "offline-deterministic"

    cfg_local = AgentConfig(preferred_provider="local", model_name="qwen2.5:1.5b")
    assert cfg_local.active_provider == "local"
    assert cfg_local.default_model == "qwen2.5:1.5b"


def test_llm_client_offline_degradation():
    """Verify LLMClient gracefully returns None when offline without raising exceptions."""
    cfg = AgentConfig(preferred_provider="offline")
    client = LLMClient(cfg)
    assert not client.is_available
    result = client.generate("Hello model")
    assert result is None


def test_llm_client_local_slm_mocked():
    """Verify LLMClient properly formats local OpenAI-compatible SLM (Ollama) call."""
    cfg = AgentConfig(preferred_provider="local", local_endpoint="http://localhost:11434/v1")
    client = LLMClient(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"result": "local_slm_ok"}'}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        out = client.generate("Test prompt", json_mode=True)
        assert out == '{"result": "local_slm_ok"}'


def test_offline_intelligence_generation(tmp_path: Path):
    """Verify offline deterministic intelligence report generation."""
    ws = MagicMock()
    ws.run_id = "mission_alpha"
    ws.export_dir = tmp_path / "export"
    ws.export_dir.mkdir(parents=True, exist_ok=True)
    ws.images_dir = tmp_path / "images"
    ws.images_dir.mkdir(parents=True, exist_ok=True)

    ws.manifest = {
        "stages": {
            "frames": {"metrics": {"n_keyframes": 45, "n_frames_decoded": 220}},
            "pose": {"metrics": {"registered_fraction": 0.95, "mean_reproj_error": 0.52}},
            "dense": {"metrics": {"n_dense_points": 85000}},
            "mesh": {"metrics": {"n_faces": 150000}},
            "export": {"metrics": {"gsd_m": 0.038, "laz_observed_fraction": 0.78, "laz_estimated_fraction": 0.16, "laz_inferred_fraction": 0.06}},
        },
        "accuracy": {"crs": "EPSG:32643", "coordinate_mode": "georeferenced"},
    }

    offline_client = LLMClient(AgentConfig(preferred_provider="offline"))
    report = generate_intelligence_report(ws, client=offline_client)
    assert isinstance(report, IntelligenceReport)
    assert report.run_id == "mission_alpha"
    assert report.is_ai_boosted is False
    assert "85,000" in " ".join(report.critical_infrastructure)
    assert report.confidence_breakdown["observed_fraction"] == 0.78

    saved_json = ws.export_dir / "intelligence_report.json"
    assert saved_json.exists()
    data = json.loads(saved_json.read_text(encoding="utf-8"))
    assert data["run_id"] == "mission_alpha"


def test_diagnostics_healthy_run(tmp_path: Path):
    """Verify diagnostician identifies healthy pipeline status."""
    ws = MagicMock()
    ws.run_id = "healthy_run"
    ws.logs_dir = tmp_path / "logs"
    ws.logs_dir.mkdir(parents=True, exist_ok=True)

    ws.manifest = {
        "stages": {
            "frames": {"status": "ok", "metrics": {"n_frames_decoded": 200, "n_keyframes": 40}},
            "masks": {"status": "ok", "metrics": {"mean_masked_fraction": 0.02, "n_dropped_frames": 0}},
            "pose": {"status": "ok", "metrics": {"registered_fraction": 0.96, "mean_reproj_error": 0.61}},
        }
    }

    diag = analyze_reconstruction_diagnostics(ws)
    assert isinstance(diag, SelfHealingReport)
    assert diag.overall_health == "HEALTHY"
    assert len(diag.detected_bottlenecks) == 0


def test_diagnostics_low_registration_recommends_neural(tmp_path: Path):
    """Verify diagnostician detects low registration and recommends neural matching."""
    ws = MagicMock()
    ws.run_id = "degraded_run"
    ws.logs_dir = tmp_path / "logs"
    ws.logs_dir.mkdir(parents=True, exist_ok=True)

    ws.manifest = {
        "stages": {
            "frames": {"status": "ok", "metrics": {"n_frames_decoded": 300, "n_keyframes": 50}},
            "masks": {"status": "ok", "metrics": {"mean_masked_fraction": 0.05, "n_dropped_frames": 1}},
            "pose": {"status": "ok", "metrics": {"registered_fraction": 0.58, "mean_reproj_error": 1.45}},
        }
    }

    diag = analyze_reconstruction_diagnostics(ws)
    assert diag.overall_health == "DEGRADED"
    assert any("Incomplete camera registration" in b for b in diag.detected_bottlenecks)
    assert any("High reprojection error" in b for b in diag.detected_bottlenecks)

    rec_params = [action.parameter for action in diag.tuning_recommendations]
    assert "neural_matching" in rec_params
    assert "bundle_adjustment_loss" in rec_params


def test_spatial_tools_and_chat(tmp_path: Path):
    """Verify spatial tools and natural language chat query handler."""
    dist = calculate_3d_distance((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))
    assert dist["distance_2d_horizontal_m"] == 5.0
    assert dist["distance_3d_m"] == 13.0
    assert dist["delta_elevation_m"] == 12.0

    stage_exp = explain_pipeline_stage("stage 2")
    assert "Velocity-Adaptive Dynamic Object Masking" in stage_exp

    ws = MagicMock()
    ws.run_id = "chat_run"
    ws.export_dir = tmp_path / "export"
    ws.export_dir.mkdir(parents=True, exist_ok=True)
    ws.manifest = {
        "stages": {
            "frames": {"metrics": {"n_keyframes": 30}},
            "pose": {"metrics": {"registered_fraction": 0.94, "mean_reproj_error": 0.62}},
            "export": {"metrics": {"gsd_m": 0.045, "laz_observed_fraction": 0.72}},
        },
        "accuracy": {"crs": "EPSG:32643", "coordinate_mode": "georeferenced", "alignment_rmse_m": 1.42},
    }

    res_gsd = handle_spatial_chat(ws, "What is the ground sampling distance?")
    assert "4.5 cm/pixel" in res_gsd["reply"]

    res_acc = handle_spatial_chat(ws, "What is the georeferencing accuracy and RMSE?")
    assert "1.42 m RMS" in res_acc["reply"]
    assert "EPSG:32643" in res_acc["reply"]

    res_conf = handle_spatial_chat(ws, "Show me confidence tiers")
    assert "Class 1 (Directly Observed)" in res_conf["reply"]
    assert res_conf["action"]["type"] == "highlight_confidence"


def test_ollama_mesh_completion_and_roll_rectification():
    """Verify 3D flight-line roll rectification, outlier removal, and Ollama mesh completion planning."""
    import numpy as np
    from dronemap.mesh_completion import (
        plan_mesh_completion_with_ollama,
        rectify_and_clean_dense_cloud,
    )
    from dronemap.terrain import _rotation_aligning

    rng = np.random.default_rng(42)
    x = rng.uniform(-30, 30, 600)
    y = rng.uniform(-30, 30, 600)
    z = rng.normal(0.0, 0.05, 600)
    z[:100] += 8.0
    pts_upright = np.column_stack([x, y, z])

    tilted_normal = np.array([0.6, 0.5, -0.6245])
    tilted_normal /= np.linalg.norm(tilted_normal)
    R_tilt = _rotation_aligning(np.array([0.0, 0.0, 1.0]), tilted_normal)
    pts_tilted = pts_upright @ R_tilt.T

    outliers = rng.uniform(400, 600, (5, 3))
    pts_noisy = np.vstack([pts_tilted, outliers])
    cols = rng.integers(40, 220, (len(pts_noisy), 3), dtype=np.uint8)

    cam_centres = np.array([[0.0, 0.0, 40.0]]) @ R_tilt.T
    clean_pts, clean_cols, stats = rectify_and_clean_dense_cloud(
        pts_noisy, cols, up_target=np.array([0.0, 0.0, 1.0]), camera_centres=cam_centres
    )
    assert stats["outliers_removed"] >= 5
    assert stats["roll_rectified_deg"] > 100.0
    assert len(clean_pts) == len(clean_cols)

    plan = plan_mesh_completion_with_ollama(clean_pts, np.array([0.0, 0.0, 1.0]))
    assert plan.grid_dim >= 200
    assert plan.hole_fill_radius_cells >= 8
    assert "inpaint" in plan.strategy or "completion" in plan.strategy

