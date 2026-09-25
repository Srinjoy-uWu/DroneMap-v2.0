"""Tests for dronemap.agent package (LLM Booster Layer & Diagnostics)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from dronemap.agent.config import AgentConfig
from dronemap.agent.client import LLMClient
from dronemap.agent.intelligence import (
    IntelligenceReport,
    generate_intelligence_report,
    _extract_survey_facts,
    _generate_offline_intelligence,
)
from dronemap.agent.diagnostics import (
    analyze_reconstruction_diagnostics,
    SelfHealingReport,
)
from dronemap.agent.tools import (
    calculate_3d_distance,
    explain_pipeline_stage,
    handle_spatial_chat,
    query_regional_confidence,
)


def test_agent_config_routing():
    """Verify provider routing and offline fallback in AgentConfig."""
    # No keys -> offline
    cfg_offline = AgentConfig(gemini_api_key=None, openai_api_key=None, anthropic_api_key=None)
    assert not cfg_offline.has_api_keys
    assert cfg_offline.active_provider == "offline"

    # Gemini key present
    cfg_gemini = AgentConfig(gemini_api_key="mock_gemini_key", openai_api_key=None)
    assert cfg_gemini.has_api_keys
    assert cfg_gemini.active_provider == "gemini"
    assert "gemini" in cfg_gemini.default_model

    # Explicit provider selection
    cfg_forced = AgentConfig(
        gemini_api_key="mock_gemini",
        openai_api_key="mock_openai",
        preferred_provider="openai",
    )
    assert cfg_forced.active_provider == "openai"


def test_llm_client_offline_degradation():
    """Verify LLMClient gracefully returns None when offline without raising exceptions."""
    cfg = AgentConfig(gemini_api_key=None, openai_api_key=None)
    client = LLMClient(cfg)
    assert not client.is_available
    result = client.generate("Hello model")
    assert result is None


def test_llm_client_gemini_mocked():
    """Verify LLMClient properly formats Gemini API call."""
    cfg = AgentConfig(gemini_api_key="test_key", preferred_provider="gemini")
    client = LLMClient(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"result": "success"}'}]}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        out = client.generate("Test prompt", json_mode=True)
        assert out == '{"result": "success"}'


def test_llm_client_openai_mocked():
    """Verify LLMClient properly formats OpenAI API call."""
    cfg = AgentConfig(openai_api_key="test_key", preferred_provider="openai")
    client = LLMClient(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Tactical summary ready."}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        out = client.generate("Briefing request")
        assert out == "Tactical summary ready."


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

    # Verify JSON file written
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
    # 3D Distance calculation
    dist = calculate_3d_distance((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))
    assert dist["distance_2d_horizontal_m"] == 5.0
    assert dist["distance_3d_m"] == 13.0
    assert dist["delta_elevation_m"] == 12.0

    # Pipeline stage explanation
    stage_exp = explain_pipeline_stage("stage 2")
    assert "Velocity-Adaptive Dynamic Object Masking" in stage_exp

    # Spatial chat mock
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

    # Query GSD
    res_gsd = handle_spatial_chat(ws, "What is the ground sampling distance?")
    assert "4.5 cm/pixel" in res_gsd["reply"]

    # Query Accuracy
    res_acc = handle_spatial_chat(ws, "What is the georeferencing accuracy and RMSE?")
    assert "1.42 m RMS" in res_acc["reply"]
    assert "EPSG:32643" in res_acc["reply"]

    # Query Confidence
    res_conf = handle_spatial_chat(ws, "Show me confidence tiers")
    assert "Class 1 (Directly Observed)" in res_conf["reply"]
    assert res_conf["action"]["type"] == "highlight_confidence"
