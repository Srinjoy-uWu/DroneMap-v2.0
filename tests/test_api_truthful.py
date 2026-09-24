"""Tests for truthful georeferencing, status tiers, and measurement endpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from dronemap.api.server import app, derive_status_tier
from dronemap.workspace import RunWorkspace


@pytest.fixture
def client():
    return TestClient(app)


def test_derive_status_tier_capture_rejected():
    manifest = {
        "stages": {
            "pose": {"status": "failed"}
        }
    }
    tier, status_type = derive_status_tier(manifest, has_model=False)
    assert tier == "Capture rejected"
    assert status_type == "rejected"


def test_derive_status_tier_terrain_2_5d():
    manifest = {
        "stages": {
            "pose": {"status": "ok"},
            "mesh": {
                "status": "ok",
                "metrics": {"mesh_type": "terrain_2.5d"}
            },
            "export": {"status": "ok"}
        }
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Terrain-only (2.5D)"
    assert status_type == "terrain"


def test_derive_status_tier_relative_local():
    manifest = {
        "stages": {
            "pose": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"}
        },
        "accuracy": {"coordinate_mode": "relative", "crs": "LOCAL_RELATIVE"}
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Relative/local"
    assert status_type == "relative"


def test_derive_status_tier_validated_3d():
    """"Validated" requires a *measured* alignment error, not just georef success.

    This test previously passed a manifest with no ``alignment_rmse_m`` and
    asserted "Validated 3D", which encoded the defect it was meant to guard:
    georeferencing *succeeding* and its error being *measured* are different
    facts, and the RMSE went unmeasured for the whole history of this project
    while every run still badged itself validated.  The measured number is now
    part of the fixture because it is part of the claim.
    """
    manifest = {
        "stages": {
            "pose": {"status": "ok"},
            "georef": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"}
        },
        "accuracy": {"coordinate_mode": "georeferenced", "crs": "EPSG:32630",
                     "alignment_rmse_m": 1.42}
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Validated 3D"
    assert status_type == "validated"


def test_derive_status_tier_georeferenced_but_unmeasured():
    """Georeferenced with no RMSE is its own tier - not validated, not relative.

    The model genuinely carries world coordinates, so calling it "Relative/local"
    would understate it; but nothing measured how wrong those coordinates are, so
    "Validated 3D" would overstate it.  ``None`` here means "could not be
    measured" and must not be confused with a small error.
    """
    manifest = {
        "stages": {
            "pose": {"status": "ok"},
            "georef": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"}
        },
        "accuracy": {"coordinate_mode": "georeferenced", "crs": "EPSG:32630",
                     "alignment_rmse_m": None}
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Georeferenced (unvalidated)"
    assert status_type == "georeferenced"


def test_derive_status_tier_verdict_outranks_files():
    """A recorded ``reject`` verdict wins even when a mesh exists on disk.

    File existence is not evidence of validity: the float32-at-ECEF era produced
    .glb files that opened to a half-metre lattice.  The gate's verdict is the
    measurement, so it outranks the artefact.
    """
    manifest = {
        "stages": {
            "pose": {"status": "ok", "metrics": {"capture_verdict": "reject"}},
            "georef": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"}
        },
        "accuracy": {"coordinate_mode": "georeferenced", "crs": "EPSG:32630",
                     "alignment_rmse_m": 0.9}
    }
    tier, status_type = derive_status_tier(manifest, has_model=True)
    assert tier == "Capture rejected"
    assert status_type == "rejected"


def test_derive_status_tier_ignores_run_name():
    """The product type comes from the run's metrics, never from its name.

    ``"terrain" in run_name.lower()`` used to badge any run the operator happened
    to call ``terrain_survey`` as 2.5D regardless of what it built.  A name is
    operator prose, not evidence.
    """
    manifest = {
        "stages": {
            "pose": {"status": "ok", "metrics": {"capture_verdict": "accept_3d"}},
            "georef": {"status": "ok"},
            "mesh": {"status": "ok", "metrics": {"mesh_type": "openmvs_3d"}},
            "export": {"status": "ok"}
        },
        "accuracy": {"coordinate_mode": "georeferenced", "crs": "EPSG:32630",
                     "alignment_rmse_m": 1.1}
    }
    tier, _ = derive_status_tier(manifest, has_model=True, run_name="terrain_test_orbit")
    assert tier == "Validated 3D"


def test_measurement_unaligned_run(tmp_path: Path, client: TestClient, monkeypatch):
    ws = RunWorkspace.create(tmp_path, "run_unaligned", config={})
    ws.set_accuracy(crs="LOCAL_RELATIVE", coordinate_mode="relative")
    ws.manifest["stages"]["georef"]["status"] = "skipped"
    ws._write()

    import dronemap.api.server as srv
    monkeypatch.setattr(srv, "_data_root", tmp_path)

    resp = client.post(
        "/api/runs/run_unaligned/measure",
        json={"a": [0.0, 0.0, 0.0], "b": [3.0, 4.0, 0.0]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_metric"] is False
    assert data["distance_m"] == 5.0
    assert data["horizontal_m"] is None
    assert data["vertical_m"] is None
    assert data["bearing_deg"] is None


def test_measurement_georeferenced_run(tmp_path: Path, client: TestClient, monkeypatch):
    ws = RunWorkspace.create(tmp_path, "run_aligned", config={})
    ws.set_accuracy(crs="EPSG:32630", coordinate_mode="georeferenced")
    ws.manifest["stages"]["georef"]["status"] = "ok"
    ws._write()

    import dronemap.api.server as srv
    monkeypatch.setattr(srv, "_data_root", tmp_path)

    # In Three.js coordinates: Y is vertical, X and Z are horizontal
    resp = client.post(
        "/api/runs/run_aligned/measure",
        json={"a": [0.0, 0.0, 0.0], "b": [3.0, 10.0, 4.0]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_metric"] is True
    assert data["distance_m"] == pytest.approx(math.sqrt(3**2 + 10**2 + 4**2), 0.01)
    assert data["vertical_m"] == pytest.approx(10.0, 0.01)
    assert data["horizontal_m"] == pytest.approx(5.0, 0.01)  # sqrt(3^2 + 4^2) = 5
    assert data["bearing_deg"] is not None


def test_api_chat_and_stages_endpoints(tmp_path: Path, client: TestClient, monkeypatch):
    ws = RunWorkspace.create(tmp_path, "run_ai_test", config={})
    ws.manifest["stages"]["frames"]["metrics"] = {"n_keyframes": 24}
    ws.manifest["stages"]["pose"]["status"] = "ok"
    ws.manifest["stages"]["pose"]["metrics"] = {"registered_fraction": 0.92, "mean_reproj_error": 0.55}
    ws.manifest["stages"]["export"]["metrics"] = {"gsd_m": 0.042, "laz_observed_fraction": 0.80}
    ws.set_accuracy(crs="EPSG:32643", coordinate_mode="georeferenced", alignment_rmse_m=0.88)
    ws._write()

    import dronemap.api.server as srv
    monkeypatch.setattr(srv, "_data_root", tmp_path)

    # 1. Test /api/runs/{id}/stages
    stages_resp = client.get("/api/runs/run_ai_test/stages")
    assert stages_resp.status_code == 200
    sdata = stages_resp.json()
    assert sdata["run_id"] == "run_ai_test"
    assert "stages" in sdata

    # 2. Test /api/runs/{id}/chat
    chat_resp = client.post(
        "/api/runs/run_ai_test/chat",
        json={"query": "What is the ground sampling distance?"}
    )
    assert chat_resp.status_code == 200
    cdata = chat_resp.json()
    assert "4.2 cm/pixel" in cdata["reply"]

    # 3. Test /api/runs/{id}/diagnostics
    diag_resp = client.get("/api/runs/run_ai_test/diagnostics")
    assert diag_resp.status_code == 200
    ddata = diag_resp.json()
    assert ddata["overall_health"] in ("HEALTHY", "DEGRADED")

    # 4. Test /api/runs/{id}/intelligence
    intel_resp = client.get("/api/runs/run_ai_test/intelligence")
    assert intel_resp.status_code == 200
    idata = intel_resp.json()
    assert idata["run_id"] == "run_ai_test"
    assert "trafficability_wheeled" in idata

