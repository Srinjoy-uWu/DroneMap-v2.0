"""FastAPI web server — measurement viewer and run API.

Routes
------
GET  /                         → viewer HTML (serves static/index.html)
GET  /api/runs                 → list of all runs with status summary
GET  /api/runs/{run_id}        → full manifest JSON
GET  /api/runs/{run_id}/model.glb    → serve the GLB mesh
GET  /api/runs/{run_id}/cloud.laz    → serve the LAZ point cloud
GET  /api/runs/{run_id}/dsm.tif      → serve the DSM GeoTIFF
GET  /api/runs/{run_id}/orthomosaic.tif  → orthomosaic GeoTIFF
GET  /api/runs/{run_id}/report       → accuracy report JSON
POST /api/runs/{run_id}/measure      → {a: [x,y,z], b: [x,y,z]} → {distance_m, bearing_deg}

Static files are served from src/dronemap/api/static/.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..config import Config

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="dronemap viewer & processor", version="0.2.0")


# ---------------------------------------------------------------------------
# State (config + data_root set by serve())
# ---------------------------------------------------------------------------

_data_root: Path = Path("data")
_active_jobs: dict[str, dict[str, Any]] = {}


def _runs_dir() -> Path:
    return _data_root / "runs"


def _uploads_dir() -> Path:
    return _data_root / "uploads"


def _run_export_dir(run_id: str) -> Path:
    return _runs_dir() / run_id / "07_export"


def _run_manifest(run_id: str) -> dict[str, Any]:
    p = _runs_dir() / run_id / "manifest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _export_file(run_id: str, filename: str) -> Path:
    p = _run_export_dir(run_id) / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found for run {run_id!r}")
    return p


# ---------------------------------------------------------------------------
# Background Pipeline Worker
# ---------------------------------------------------------------------------

def _run_job_worker(
    job_id: str,
    run_id: str,
    video_path: Path,
    telemetry_path: Path | None,
    profile: str,
    no_telemetry: bool,
    quality: str,
) -> None:
    from ..config import load_config
    from ..tools import ToolRegistry
    from ..workspace import RunWorkspace
    from ..pipeline import STAGE_GRAPH, plan_stages, run_pipeline

    job = _active_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.utcnow().isoformat()

    try:
        # Apply quality preset overrides as key=value strings
        # The web UI must not silently weaken the geometry requirements used
        # by the CLI; poor camera geometry should be rejected, not hidden.
        overrides_list: list[str] = []
        # Optimized presets for fast execution and high quality without stalling
        if quality == "fast":
            overrides_list.extend([
                "frames.max_long_edge=1600",
                "frames.max_keyframes=22",
                "pose.max_num_features=8192",
                "dense.resolution_level=2",
                "dense.max_resolution=1280",
                "mesh.texture_size=2048",
            ])
        elif quality == "hd":
            overrides_list.extend([
                "frames.max_long_edge=1920",
                "frames.max_keyframes=42",
                "pose.max_num_features=10240",
                "dense.resolution_level=1",
                "dense.max_resolution=1920",
                "mesh.texture_size=4096",
            ])
        else:  # "balanced" / standard default
            overrides_list.extend([
                "frames.max_long_edge=1600",
                "frames.max_keyframes=26",
                "pose.max_num_features=8192",
                "dense.resolution_level=2",
                "dense.max_resolution=1600",
                "mesh.texture_size=4096",
            ])

        cfg = load_config(
            profile=profile if profile and profile != "default" else None,
            overrides=overrides_list,
        )
        cfg.data_root = _data_root
        if no_telemetry or not telemetry_path:
            cfg.georef.method = "none"

        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Resolving external tools (COLMAP, OpenMVS, FFmpeg)...")
        tools = ToolRegistry.resolve(cfg)

        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing workspace for run '{run_id}'...")
        ws = RunWorkspace.create(cfg.data_root, run_id, cfg.model_dump(mode="json"))
        ws.set_input(
            video=str(video_path.resolve()),
            telemetry=str(telemetry_path.resolve()) if telemetry_path else None,
        )

        specs = plan_stages(cfg)
        job["total_stages"] = len(specs)
        for s in specs:
            job["stages"][s.key] = "pending"

        def on_stage_callback(spec: Any, status: str) -> None:
            job["current_stage"] = spec.key
            job["current_stage_title"] = spec.title
            job["stages"][spec.key] = status
            timestamp = datetime.now().strftime("%H:%M:%S")
            job["logs"].append(f"[{timestamp}] Stage '{spec.key}': {status} ({spec.title})")

        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting photogrammetry pipeline ({len(specs)} stages)...")
        run_pipeline(ws, cfg, tools, specs=specs, on_stage=on_stage_callback)

        job["status"] = "completed"
        job["finished_at"] = datetime.utcnow().isoformat()
        job["has_model"] = (_run_export_dir(run_id) / "model.glb").exists()
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 3D Reconstruction complete! Model ready for inspection.")
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = datetime.utcnow().isoformat()
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Pipeline failed: {exc}")


# ---------------------------------------------------------------------------
# Job execution API (GUI upload and pipeline runner)
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    telemetry: Optional[UploadFile] = File(None),
    run_id: Optional[str] = Form(None),
    profile: str = Form("default"),
    no_telemetry: bool = Form(False),
    quality: str = Form("hd"),
) -> JSONResponse:
    # Generate run_id if not provided
    if not run_id or not run_id.strip():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(video.filename or "flight").stem
        clean_stem = re.sub(r"[^a-zA-Z0-9_]", "_", stem)[:16]
        run_id = f"run_{clean_stem}_{timestamp}"
    else:
        run_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", run_id.strip())

    job_id = f"job_{uuid.uuid4().hex[:8]}"

    # Setup upload directory
    upload_run_dir = _uploads_dir() / run_id
    upload_run_dir.mkdir(parents=True, exist_ok=True)

    # Save video file
    video_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    saved_video_path = upload_run_dir / f"input_video{video_ext}"
    with saved_video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Save telemetry file if uploaded
    saved_telemetry_path: Path | None = None
    if telemetry and telemetry.filename:
        telem_ext = Path(telemetry.filename).suffix or ".srt"
        saved_telemetry_path = upload_run_dir / f"input_telemetry{telem_ext}"
        with saved_telemetry_path.open("wb") as buffer:
            shutil.copyfileobj(telemetry.file, buffer)

    # Register job state
    _active_jobs[job_id] = {
        "job_id": job_id,
        "run_id": run_id,
        "status": "queued",
        "current_stage": "queued",
        "current_stage_title": "Queued for processing",
        "stages": {},
        "total_stages": 7,
        "logs": [
            f"[{datetime.now().strftime('%H:%M:%S')}] Job '{job_id}' queued for run '{run_id}'",
            f"[{datetime.now().strftime('%H:%M:%S')}] Video uploaded: {video.filename} ({saved_video_path.stat().st_size / 1e6:.1f} MB)",
            f"[{datetime.now().strftime('%H:%M:%S')}] Telemetry: {telemetry.filename if telemetry and telemetry.filename else ('Disabled' if no_telemetry else 'None (relative mode)')}",
            f"[{datetime.now().strftime('%H:%M:%S')}] Profile: {profile} | Quality: {quality}",
        ],
        "error": None,
        "has_model": False,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Launch processing thread
    thread = threading.Thread(
        target=_run_job_worker,
        args=(
            job_id,
            run_id,
            saved_video_path,
            saved_telemetry_path,
            profile,
            no_telemetry,
            quality,
        ),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "job_id": job_id,
        "run_id": run_id,
        "status": "queued",
        "message": f"Processing started for run '{run_id}'",
    })


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    return JSONResponse(list(_active_jobs.values()))


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> JSONResponse:
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return JSONResponse(job)


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> PlainTextResponse:
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return PlainTextResponse("\n".join(job.get("logs", [])))


def derive_status_tier(manifest: dict, has_model: bool = True, run_name: str = "") -> tuple[str, str]:
    """Derive the status badge from what the run *measured*.

    Every branch here must trace to a recorded measurement.  Two earlier
    shortcuts did not:

    * ``"terrain" in run_name.lower()`` inferred the product from the run's
      *name*.  A run the operator happened to call ``terrain_test_orbit`` was
      badged 2.5D no matter what it actually built, and renaming a run silently
      changed its reported capability.  The name is operator prose, not
      evidence; the mesh stage records ``mesh_type`` and the pose stage records
      ``capture_verdict``, and those are the evidence.
    * ``georef_ok`` alone earned the badge "Validated 3D".  Georeferencing
      *succeeding* and its error being *measured* are different facts -- the
      alignment RMSE went unmeasured for the whole history of this project
      while every run still advertised itself as validated.  "Validated" now
      requires a number.

    The capture-quality verdict outranks file existence in both directions: a
    rejected capture that still produced a .glb is rejected (the mesh is
    geometry fitted to parallax that was never there), and a capture the gate
    demoted to 2.5D is reported as 2.5D even if the mesh stage wrote a
    full-3D-looking file.

    ``run_name`` is accepted for call compatibility and deliberately unused.
    """
    stages = manifest.get("stages", {})
    failed = [k for k, v in stages.items() if v.get("status") == "failed"]
    accuracy = manifest.get("accuracy", {})
    verdict = (stages.get("pose", {}).get("metrics", {}) or {}).get("capture_verdict")
    mesh_type = (stages.get("mesh", {}).get("metrics", {}) or {}).get("mesh_type")

    georef_ok = (
        accuracy.get("crs") not in ("LOCAL_RELATIVE", "", None)
        and stages.get("georef", {}).get("status") == "ok"
        and accuracy.get("coordinate_mode") != "relative"
    )
    # None means "could not be measured"; a run may be correctly georeferenced
    # and still have no error figure, and that distinction is the whole
    # difference between "georeferenced" and "validated".
    rmse_measured = accuracy.get("alignment_rmse_m") is not None

    if stages.get("pose", {}).get("status") == "failed" or verdict == "reject":
        return "Capture rejected", "rejected"
    if not has_model:
        return ("Capture rejected", "rejected") if failed else ("Processing / Incomplete", "incomplete")
    if verdict == "terrain_2_5d" or mesh_type == "terrain_2.5d" or manifest.get("classification") == "terrain":
        return "Terrain-only (2.5D)", "terrain"
    if georef_ok and rmse_measured:
        return "Validated 3D", "validated"
    if georef_ok:
        return "Georeferenced (unvalidated)", "georeferenced"
    return "Relative/local", "relative"


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

def _run_mtime(p: Path) -> float:
    m = p / "manifest.json"
    if m.exists():
        try:
            return m.stat().st_mtime
        except OSError:
            pass
    return p.stat().st_mtime


@app.get("/api/runs")
def list_runs(all: bool = False) -> JSONResponse:
    runs_dir = _runs_dir()
    if not runs_dir.exists():
        return JSONResponse([])
    runs = []
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=_run_mtime, reverse=True)
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stages = m.get("stages", {})
        done = [k for k, v in stages.items() if v.get("status") == "ok"]
        failed = [k for k, v in stages.items() if v.get("status") == "failed"]
        has_model = (_run_export_dir(run_dir.name) / "model.glb").exists()

        badge, status_type = derive_status_tier(m, has_model=has_model, run_name=run_dir.name)

        # By default, do not clutter UI with failed runs unless all=True
        if status_type == "rejected" and not all:
            # Still show if it's the newest run so the user sees immediate rejection feedback
            if len(runs) > 0:
                continue

        runs.append({
            "run_id": run_dir.name,
            "created_utc": m.get("created_utc"),
            "gnss_mode": m.get("gnss", {}).get("mode", "NONE"),
            "stages_ok": done,
            "stages_failed": failed,
            "progress": f"{len(done)}/7",
            "has_model": has_model,
            "badge": badge,
            "status_type": status_type,
            # A run that produced two texture variants can be compared in the
            # viewer; one that did not must not offer a toggle that 404s.
            "has_alt_model": (_run_export_dir(run_dir.name) / "model_alt.glb").exists(),
        })
    return JSONResponse(runs)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    return JSONResponse(_run_manifest(run_id))


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str) -> JSONResponse:
    p = _run_export_dir(run_id) / "accuracy_report.json"
    if p.exists():
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
    # Fall back to manifest accuracy section
    m = _run_manifest(run_id)
    return JSONResponse(m.get("accuracy", {}))


@app.api_route("/api/runs/{run_id}/model.glb", methods=["GET", "HEAD"])
def get_model_glb(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "model.glb")),
        media_type="model/gltf-binary",
        filename="model.glb",
    )


@app.api_route("/api/runs/{run_id}/model_alt.glb", methods=["GET", "HEAD"])
def get_model_alt_glb(run_id: str) -> FileResponse:
    """The second texture variant, when stage 6 produced one.

    Same geometry as ``model.glb``, same viewer frame, different atlas: one
    textured with OpenMVS seam levelling on, one with it off.  404 when the run
    has only a single variant, which is the normal case for a scene where seam
    levelling worked.
    """
    return FileResponse(
        str(_export_file(run_id, "model_alt.glb")),
        media_type="model/gltf-binary",
        filename="model_alt.glb",
    )


@app.get("/api/runs/{run_id}/cloud.laz")
def get_cloud_laz(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "cloud.laz")),
        media_type="application/octet-stream",
        filename="cloud.laz",
    )


@app.get("/api/runs/{run_id}/dsm.tif")
def get_dsm(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "dsm.tif")),
        media_type="image/tiff",
        filename="dsm.tif",
    )


@app.get("/api/runs/{run_id}/orthomosaic.tif")
def get_ortho(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "orthomosaic.tif")),
        media_type="image/tiff",
        filename="orthomosaic.tif",
    )


@app.get("/api/runs/{run_id}/dtm.tif")
def get_dtm(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "dtm.tif")),
        media_type="image/tiff",
        filename="dtm.tif",
    )


@app.get("/api/runs/{run_id}/trajectory.kml")
def get_trajectory_kml(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "trajectory.kml")),
        media_type="application/vnd.google-earth.kml+xml",
        filename="trajectory.kml",
    )


@app.get("/api/runs/{run_id}/report.html")
def get_report_html(run_id: str) -> FileResponse:
    return FileResponse(
        str(_export_file(run_id, "report.html")),
        media_type="text/html",
        filename="report.html",
    )


# ---------------------------------------------------------------------------
# Measurement endpoint
# ---------------------------------------------------------------------------

class MeasureRequest(BaseModel):
    a: list[float]  # [x, y, z] in Three.js world space
    b: list[float]


class MeasureResponse(BaseModel):
    distance_m: float
    horizontal_m: float | None = None
    vertical_m: float | None = None
    bearing_deg: float | None = None
    is_metric: bool = True
    coordinate_frame: str = "LOCAL_RELATIVE"


@app.post("/api/runs/{run_id}/measure", response_model=MeasureResponse)
def measure(run_id: str, req: MeasureRequest) -> MeasureResponse:
    # Validate the run exists and check geodetic status
    manifest = _run_manifest(run_id)
    accuracy = manifest.get("accuracy", {})
    crs = accuracy.get("crs", "")
    georef_ok = (
        crs not in ("LOCAL_RELATIVE", "", None)
        and manifest.get("stages", {}).get("georef", {}).get("status") == "ok"
    )

    if len(req.a) != 3 or len(req.b) != 3:
        raise HTTPException(status_code=422, detail="a and b must each be [x, y, z]")

    ax, ay, az = req.a
    bx, by, bz = req.b
    dx, dy, dz = bx - ax, by - ay, bz - az

    # In Three.js viewer convention:
    # +Y is vertical up/down
    # X and Z form the horizontal ground plane
    dist_3d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    horiz_dist = math.sqrt(dx ** 2 + dz ** 2)
    vert_dist = abs(dy)

    if georef_ok:
        # Ground plane bearing in Three.js (dx is east, -dz is north)
        bearing = (math.degrees(math.atan2(dx, -dz)) + 360) % 360 if horiz_dist > 1e-4 else 0.0
        return MeasureResponse(
            distance_m=round(dist_3d, 4),
            horizontal_m=round(horiz_dist, 4),
            vertical_m=round(vert_dist, 4),
            bearing_deg=round(bearing, 2),
            is_metric=True,
            coordinate_frame=crs,
        )
    else:
        # Relative run: do not claim metric scale, vertical relief, or geographic bearing
        return MeasureResponse(
            distance_m=round(dist_3d, 4),
            horizontal_m=None,
            vertical_m=None,
            bearing_deg=None,
            is_metric=False,
            coordinate_frame="LOCAL_RELATIVE",
        )


# ---------------------------------------------------------------------------
# AI Copilot, Diagnostics & Transparency Endpoints
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str


@app.post("/api/runs/{run_id}/chat")
def run_chat(run_id: str, req: ChatRequest) -> dict[str, Any]:
    """Natural language 3D spatial query copilot for Web Studio."""
    from ..workspace import RunWorkspace
    from ..agent import handle_spatial_chat
    try:
        ws = RunWorkspace.open(_data_root, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return handle_spatial_chat(ws, req.query)


@app.get("/api/runs/{run_id}/intelligence")
def get_intelligence(run_id: str) -> dict[str, Any]:
    """Retrieve or generate multimodal tactical intelligence report."""
    from ..workspace import RunWorkspace
    from ..agent import generate_intelligence_report
    try:
        ws = RunWorkspace.open(_data_root, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    report = generate_intelligence_report(ws)
    return report.to_dict()


@app.get("/api/runs/{run_id}/diagnostics")
def get_diagnostics(run_id: str) -> dict[str, Any]:
    """Retrieve self-healing reconstruction diagnostics and parameter advice."""
    from ..workspace import RunWorkspace
    from ..agent import analyze_reconstruction_diagnostics
    try:
        ws = RunWorkspace.open(_data_root, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    report = analyze_reconstruction_diagnostics(ws)
    return report.to_dict()


@app.get("/api/runs/{run_id}/stages")
def get_stages_audit(run_id: str) -> dict[str, Any]:
    """Retrieve stage-by-stage execution metrics, Before/After data, and artifact paths."""
    from ..workspace import RunWorkspace
    try:
        ws = RunWorkspace.open(_data_root, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

    manifest = ws.manifest
    stages = manifest.get("stages", {})

    previews_dir = ws.masks_dir / "previews"
    preview_files = [p.name for p in sorted(previews_dir.glob("*.jpg"))] if previews_dir.exists() else []

    return {
        "run_id": run_id,
        "stages": stages,
        "has_model": (ws.export_dir / "model.glb").exists(),
        "has_masks": ws.masks_dir.exists(),
        "mask_previews": preview_files,
        "has_confidence": (ws.export_dir / "cloud.laz").exists(),
    }


@app.get("/api/runs/{run_id}/masks/preview/{filename}")
def get_mask_preview(run_id: str, filename: str) -> FileResponse:
    """Serve dynamic masking Before vs After overlay thumbnails."""
    from ..workspace import RunWorkspace
    try:
        ws = RunWorkspace.open(_data_root, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

    p = ws.masks_dir / "previews" / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"preview {filename!r} not found")
    return FileResponse(str(p), media_type="image/jpeg", filename=filename)


# ---------------------------------------------------------------------------
# Static file serving (viewer UI)
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():

    class _NoCacheStaticFiles(StaticFiles):
        """Serve the viewer shell with caching disabled.

        The browser aggressively caches ``index.html``, ``app.js`` and
        ``viewer.js``.  When the Three.js import map was moved from a CDN to
        the local ``vendor/`` tree, cached copies kept requesting the CDN,
        which fails offline -- so the viewer module never evaluated,
        ``window.load3DModel`` was never defined, and the UI sat on "No
        Project Selected" forever with a fully valid model on disk.

        Vendored library files keep their normal caching: they are large,
        immutable, and version-pinned.
        """

        _NO_CACHE_SUFFIXES = (".html", ".js", ".css", ".json")

        async def get_response(self, path: str, scope):  # type: ignore[override]
            response = await super().get_response(path, scope)
            if path.startswith("vendor/") or path.startswith("vendor\\"):
                return response
            if path == "." or path.endswith(self._NO_CACHE_SUFFIXES):
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.mount(
        "/", _NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static"
    )
else:
    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse({
            "message": "dronemap API is running. The viewer static files are not built yet.",
            "api_docs": "/docs",
        })


# ---------------------------------------------------------------------------
# Server entry point (called from CLI)
# ---------------------------------------------------------------------------

def serve(config: "Config", host: str = "127.0.0.1", port: int = 8000) -> None:
    global _data_root
    _data_root = Path(config.data_root)
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
