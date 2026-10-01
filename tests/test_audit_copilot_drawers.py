"""Tests for Audit Drawer and AI Copilot endpoints and frontend bindings."""

from pathlib import Path
from fastapi.testclient import TestClient

from dronemap.api.server import app, _resolve_workspace


def test_resolve_workspace_default():
    """Verify that _resolve_workspace resolves 'default' to the latest run."""
    ws = _resolve_workspace("default")
    assert ws is not None
    assert ws.run_id != ""
    assert (ws.root / "manifest.json").exists()


def test_api_chat_spatial_copilot():
    """Verify that copilot chat endpoint works and generates actions."""
    client = TestClient(app)

    # 1. GSD query
    res = client.post("/api/runs/default/chat", json={"query": "What is the ground sampling distance?"})
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "Ground Sampling Distance" in data["reply"] or "GSD" in data["reply"]

    # 2. Confidence query -> triggers highlight_confidence action
    res = client.post("/api/runs/default/chat", json={"query": "Show me 3D regional confidence tiers"})
    assert res.status_code == 200
    data = res.json()
    assert "action" in data
    assert data["action"] == {"type": "highlight_confidence"}

    # 3. Dynamic masks query -> triggers toggle_masks action
    res = client.post("/api/runs/default/chat", json={"query": "What dynamic objects were removed by velocity masking?"})
    assert res.status_code == 200
    data = res.json()
    assert "action" in data
    assert data["action"] == {"type": "toggle_masks"}


def test_api_stages_audit_and_diagnostics():
    """Verify that stages audit and diagnostics return comprehensive pipeline metrics."""
    client = TestClient(app)

    ws = _resolve_workspace("default")
    assert ws is not None

    res = client.get(f"/api/runs/{ws.run_id}/stages")
    assert res.status_code == 200
    data = res.json()
    assert "stages" in data
    assert "has_model" in data
    assert "mask_previews" in data

    res_diag = client.get(f"/api/runs/{ws.run_id}/diagnostics")
    assert res_diag.status_code == 200
    diag = res_diag.json()
    assert "overall_health" in diag
    assert "diagnostic_summary" in diag


def test_mask_preview_file_serving():
    """Verify that mask preview files are served properly if present."""
    client = TestClient(app)
    ws = _resolve_workspace("run_drone_ihasalu_li_20261001_180048")
    if ws is not None:
        res = client.get(f"/api/runs/{ws.run_id}/stages")
        assert res.status_code == 200
        previews = res.json().get("mask_previews", [])
        if previews:
            preview_res = client.get(f"/api/runs/{ws.run_id}/masks/preview/{previews[0]}")
            assert preview_res.status_code == 200
            assert preview_res.headers["content-type"] == "image/jpeg"
            assert len(preview_res.content) > 1000


def test_static_assets_pointer_events_and_drawers():
    """Verify static assets have pointer-events auto and drawer interactivity."""
    css_path = Path("src/dronemap/api/static/style.css")
    js_path = Path("src/dronemap/api/static/app.js")
    html_path = Path("src/dronemap/api/static/index.html")

    assert css_path.exists()
    assert js_path.exists()
    assert html_path.exists()

    css_text = css_path.read_text(encoding="utf-8")
    js_text = js_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")

    # CSS contains top-bar-actions with pointer-events: auto
    assert ".top-bar-actions" in css_text
    assert "pointer-events: auto" in css_text
    assert ".sliding-drawer" in css_text

    # HTML contains the drawer buttons and elements
    assert 'id="btn-toggle-audit"' in html_text
    assert 'id="btn-toggle-copilot"' in html_text
    assert 'id="audit-drawer"' in html_text
    assert 'id="copilot-drawer"' in html_text

    # JS contains open/close drawer handlers and active class management
    assert "openAuditDrawer" in js_text
    assert "openCopilotDrawer" in js_text
    assert "sendCopilotQuery" in js_text
    assert "Escape" in js_text
