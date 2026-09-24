"""Tests for Stage 7 HTML report and confidence embedding."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from dronemap.stage7_export import (
    _compute_regional_confidence,
    _write_accuracy_report,
    _write_html_report,
)


def test_compute_regional_confidence():
    """Verify 3D regional confidence classifies points into 1 (Observed), 2 (Estimated), 3 (Inferred)."""
    # Create cluster of dense points and scattered sparse points
    np.random.seed(42)
    dense_pts = np.random.uniform(0, 1, size=(200, 3))
    sparse_pts = np.random.uniform(10, 50, size=(50, 3))
    pts = np.vstack([dense_pts, sparse_pts])

    conf = _compute_regional_confidence(pts)
    assert len(conf) == len(pts)
    assert set(conf).issubset({1, 2, 3})
    assert np.any(conf == 1)  # dense points
    assert np.any(conf == 3)  # sparse points


def test_write_accuracy_report_with_confidence(tmp_path: Path):
    """Verify accuracy report JSON captures confidence tiers when provided."""
    ws = MagicMock()
    ws.root = tmp_path
    ws.run_id = "test_run_123"
    ws.manifest = {
        "created_utc": "2026-09-24T12:00:00Z",
        "gnss": {"mode": "STANDALONE"},
        "accuracy": {},
        "stages": {
            "pose": {"metrics": {"registered_fraction": 0.95, "mean_reproj_error": 0.65}},
            "dense": {"metrics": {"n_dense_points": 25000}},
        },
    }
    ws.georef_sparse_dir = tmp_path / "georef"
    ws.georef_sparse_dir.mkdir(parents=True, exist_ok=True)

    out_json = tmp_path / "accuracy_report.json"
    conf_summary = {
        "observed_fraction": 0.72,
        "estimated_fraction": 0.21,
        "inferred_fraction": 0.07,
    }

    report = _write_accuracy_report(ws, out_json, conf_summary)
    assert out_json.exists()
    assert report["run_id"] == "test_run_123"
    assert "confidence_tiers" in report
    assert report["confidence_tiers"]["observed_fraction"] == 0.72


def test_write_html_report_contains_before_after_and_confidence(tmp_path: Path):
    """Verify HTML report includes the Before vs After quality audit and 3D confidence tiers."""
    ws = MagicMock()
    ws.root = tmp_path
    ws.run_id = "test_run_456"
    ws.export_dir = tmp_path / "export"
    ws.export_dir.mkdir(parents=True, exist_ok=True)
    ws.manifest = {
        "created_utc": "2026-09-24T12:00:00Z",
        "input": {"video": "sample.mp4"},
        "stages": {
            "frames": {"metrics": {"n_frames_decoded": 300, "n_keyframes": 60}},
            "masks": {"status": "ok", "metrics": {"mean_masked_fraction": 0.045, "n_dropped_frames": 2}},
            "pose": {"metrics": {"neural_fallback_used": True}},
            "mesh": {"metrics": {"n_faces": 120000, "n_vertices": 60500}},
        },
    }

    out_html = tmp_path / "report.html"
    report = {
        "run_id": "test_run_456",
        "created_utc": "2026-09-24T12:00:00Z",
        "registered_fraction": 0.98,
        "mean_reproj_error_px": 0.58,
        "n_dense_points": 50000,
        "coordinate_mode": "georeferenced",
    }
    conf_summary = {
        "observed_fraction": 0.75,
        "estimated_fraction": 0.18,
        "inferred_fraction": 0.07,
    }

    _write_html_report(ws, out_html, report, gsd=0.042, crs="EPSG:32643", conf_summary=conf_summary)
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")

    # Assert Before vs After Audit Card
    assert "Stage-by-Stage Quality &amp; Hardening Audit" in content
    assert "Multi-Factor Frame Selection" in content
    assert "300 frames" in content
    assert "60 keyframes" in content
    assert "Velocity-Adaptive Masking" in content
    assert "4.5% masked" in content
    assert "Feature Escalation Ladder" in content
    assert "Hybrid Neural (LightGlue Escalation)" in content
    assert "Watertight PBR" in content

    # Assert 3D Regional Confidence classification
    assert "3D Regional Confidence Classification (ASPRS LAZ Tiers)" in content
    assert "Class 1: Directly Observed" in content
    assert "75.0%" in content
    assert "Class 2: Estimated Surface" in content
    assert "18.0%" in content
    assert "Class 3: Inferred Zone" in content
    assert "7.0%" in content
