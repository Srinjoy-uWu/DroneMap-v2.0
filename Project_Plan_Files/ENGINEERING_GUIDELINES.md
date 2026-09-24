# Engineering Architecture & Developer Guidelines — DroneMap (SIH26158)

> **Engineering Standards & Architecture Specification**:
> Guidelines for developers, contributors, and evaluators working on **DroneMap**, an industrial UAV photogrammetry pipeline solving Smart India Hackathon problem statement **SIH26158**: *Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model*.
>
> All contributions must adhere strictly to the modular architecture, metric accuracy standards, and safety guardrails defined below.

---

## 1. Project Identity & Problem Statement

- **Problem Statement ID**: **SIH26158**
- **Title**: *Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model*
- **Primary Challenges Mandated by SIH26158**:
  1. Limited viewing angles from single-pass flight.
  2. UAV dynamics, motion blur, and video compression macroblocking.
  3. Dynamic moving objects (vehicles, pedestrians, animals).
  4. Variable illumination, transient shadows, and sun glare.
  5. GPS inaccuracies, sensor noise, and multi-path drift.
  6. Occluded ground regions (under canopies, behind buildings).
  7. Limited Ground Control Point (GCP) availability.
  8. Computational scalability (Edge RTX 3050 6GB $\to$ Cloud dual-T4/A100 accelerators).
  9. Near-real-time turnaround requirements.
  10. Verifiable metric accuracy and ASPRS/OGC GIS compliance.

---

## 2. Core Philosophy: Physics-First Augmented with AI

- **Epipolar Geometry is Ground Truth**: SfM bundle adjustment, multi-view ray intersections, and closed-form geodesy (WGS-84 $\leftrightarrow$ ECEF $\leftrightarrow$ ENU $\leftrightarrow$ UTM) represent physical truth.
- **Role of AI / Foundation Models**: Deep learning models **filter, guide, segment, and constrain** the pipeline; they **never invent or hallucinate unobserved geometry**.
- **No Generative Black-Box Meshes**: The 3D model is reconstructed from triangulated optical rays and watertight KDTree IDW terrain interpolation, not single-image generative diffusion.

---

## 3. Current System State & Verified Baselines

- **Test Suite**: **157 unit, integration, and hardening tests passing in ~4 seconds** (`pytest -q`).
- **Target Hardware Architecture**:
  - **Local Edge Mode**: Optimized for NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.44 GB VRAM), 16 GB RAM, Windows 10/11 or Linux.
  - **Cloud / Kaggle Accelerator Mode**: Unlocks dual NVIDIA T4 (32 GB total VRAM) or A100 GPUs via `configs/accelerated.yaml` for 4K video ingestion, large foundation models, 16K texture atlases, and 3D Gaussian Splatting (`splatfacto`).
- **Real Benchmark Figures (Recorded in `data/runs/`)**:
  - `test_real_drone_0904`: 4K real UAV orbit, 26/26 cameras registered (100%), 34,201 sparse points, 211,268 dense points, 10.1 cm/px GSD.
  - `test_orbit_georef`: 47/47 cameras registered (100%), 668,657 dense points, GPS alignment RMSE **3.91 m** using consumer drone GPS.
  - `synthetic_eval_run`: Blender ground-truth fixture, scale error $< 2.0\%$, alignment RMSE **2.14 m**.

---

## 4. Non-Negotiable Architectural Guardrails

When modifying or extending this codebase, you **MUST** uphold these rules:

1. **Truthful Georeferencing & Spatial Claims**:
   - If GPS alignment fails, is unaligned, or telemetry is absent, coordinate frame must remain `"LOCAL_RELATIVE"`.
   - **Never** assign a projected UTM / EPSG CRS code to an unaligned run.
   - Never fabricate zero alignment error.
2. **GPU Memory Lifecycle Safety**:
   - Every PyTorch inference stage (`stage2_masks.py`, `stage4_depth.py`, `stage4_semantics.py`) **must** be wrapped in a `try...finally` block that cleans up resources:
     ```python
     finally:
         if 'model' in locals() and model is not None:
             del model
         try:
             import torch
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
         except ImportError:
             pass
         import gc
         gc.collect()
     ```
3. **Escalation Frame Ceiling**:
   - In `stage3_pose.py`, exhaustive matching escalation is strictly capped at $N \le 150$ frames. When $N > 150$, bypass exhaustive matching and escalate directly to vocabulary tree matching to prevent $O(N^2)$ runaway latency.
4. **Air-Gapped Offline Operation**:
   - The Three.js viewer and all API assets in `src/dronemap/api/static/` are 100% self-contained and local. Never introduce external CDN script tags or live internet dependencies.
5. **watertight 2.5D Terrain Fallback**:
   - If OpenMVS dense meshing fails or produces degenerate shards, the system automatically falls back to `terrain.py` (KDTree IDW ground plane fitting) to produce a clean, watertight 2.5D DSM mesh.
6. **Semantic Problem Statement Compliance**:
   - Stage 4b output must roll up into the 4 mandatory SIH26158 classes: (i) terrain, (ii) buildings, (iii) roads & infrastructure, (iv) vegetation & obstacles (`04_semantics/sih26158_categories.json`).
7. **Zero Regressions**:
   - Every change must maintain **100% passing tests** (`pytest -q`).

---

## 5. Codebase Map & Entry Points

```text
SIH26158/
├── configs/
│   ├── default.yaml        # Authoritative base defaults
│   ├── accelerated.yaml    # Cloud / Kaggle dual-T4 / A100 accelerator profile
│   ├── lowvram.yaml        # Strict <4GB VRAM fallback
│   ├── stretch.yaml        # Local stretch features (depth, MASt3R)
│   └── smoke.yaml          # Fast smoke testing configuration
├── data/
│   ├── runs/               # Per-run execution workspaces
│   └── test_assets/        # Fixtures and drone flight clips
├── docs/                   # Authoritative engineering documentation
│   ├── ENGINEERING_GUIDELINES.md # Core engineering guidelines and standards
│   ├── AI_ROADMAP.md       # Incremental enhancement milestones & ablations
│   ├── AS_IS_ARCHITECTURE.md # Exhaustive codebase anatomy & algorithms
│   ├── BASELINE.md         # Measured run benchmarks & system capabilities
│   ├── CURRENT_DATA_FLOW.md# Stage I/O schemas & data flow
│   ├── REAL_PROCESS_AND_SAMPLES.md # Visual sample evidence
│   ├── RESEARCH.md         # Literature review & technology trade-offs
│   ├── SIH26158_MAPPING.md # Problem statement compliance matrix
│   └── V2_ARCHITECTURE.md  # Production dual-tier system architecture
├── notebooks/
│   ├── 3dgs_splatfacto_kaggle.ipynb # Kaggle 3DGS training notebook
│   └── sih26158_3dgs_kaggle.py     # Standalone Python script for Kaggle execution
├── src/dronemap/
│   ├── cli.py              # CLI entry point (Typer + Rich)
│   ├── config.py           # Pydantic configuration schemas
│   ├── pipeline.py         # Stage orchestrator & execution graph
│   ├── workspace.py        # RunWorkspace, StageRecord, manifests
│   ├── stage1_frames.py    # O(1) streaming 2-pass keyframe selector
│   ├── stage2_masks.py     # YOLOv8s-seg dynamic object exclusion
│   ├── stage3_pose.py      # COLMAP sequential SfM + escalation ladder
│   ├── stage3_pose_mast3r.py # MASt3R transformer SfM backend (stretch)
│   ├── stage3_georef.py    # Geodesy, ECEF, ENU, Sim(3) Umeyama alignment
│   ├── stage4_depth.py     # Monocular metric depth (Depth Anything / Metric3D)
│   ├── stage4_semantics.py # SegFormer + SIH26158 4-class rollup
│   ├── stage5_dense.py     # OpenMVS DensifyPointCloud dense stereo
│   ├── stage6_mesh.py      # OpenMVS ReconstructMesh + TextureMesh + retry
│   ├── terrain.py          # Watertight 2.5D DSM surface engine
│   ├── stage7_export.py    # GLB, LAZ, GeoTIFF DSM/DTM/Ortho, KML, HTML report
│   └── api/
│       ├── server.py       # FastAPI REST service & measurement endpoints
│       └── static/         # Three.js 3D measurement studio (air-gapped)
└── tests/
    ├── test_api_truthful.py
    ├── test_cleanup.py
    ├── test_config.py
    ├── test_export_coordinates.py
    ├── test_hardening.py   # 6 hardening regression tests
    ├── test_mesh_cleanup.py
    ├── test_pipeline.py
    ├── test_quality.py
    ├── test_stage1_frames.py
    ├── test_stage3_pose.py
    ├── test_telemetry.py
    ├── test_terrain.py
    ├── test_texture_coverage.py
    └── test_workspace.py
```

---

## 6. How to Run & Verify

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Run full regression test suite (157 tests)
pytest -q

# 3. System diagnosis
dronemap doctor

# 4. Run full pipeline on drone footage
dronemap run --video data/test_assets/orbit_flight.mp4 --telemetry data/test_assets/orbit_flight.srt --run-id flight_demo

# 5. Run with cloud accelerated profile (4K, 16K atlas, large models)
dronemap run --video data/test_assets/orbit_flight.mp4 --profile accelerated --run-id flight_cloud

# 6. Launch Web Measurement Studio
dronemap serve
# Open http://127.0.0.1:8000
```

---

## 7. Immediate Actionable Tasks for the Agent

When instructed to implement further upgrades, tackle these prioritized tasks:

### Task 1: LightGlue Neural Matcher Integration
- Implement `src/dronemap/matching_neural.py` integrating SuperPoint + LightGlue.
- Follow `hloc` conventions to inject keypoints and matches into COLMAP's SQLite database (`colmap.db`).
- Activate when consecutive frame matching yields $< 50$ SIFT inliers or for low-texture road/water boundaries.

### Task 2: 3D Regional Confidence Map
- Calculate per-point and per-vertex confidence score:
  - **Observed** ($\ge 3$ intersecting camera rays).
  - **Estimated** (Multi-view stereo interpolation).
  - **Inferred** (Terrain prior or monocular depth infill).
- Embed confidence values in `cloud.laz` (ExtraBytes or classification) and `model.glb` vertex colors/attributes.

### Task 3: Interactive Stage Drawer in Web Studio
- Extend `src/dronemap/api/static/index.html` and `viewer.js` with an expandable stage-by-stage inspection panel showing keyframes, YOLO masks, trajectory KML, depth maps, semantic distributions, and GIS rasters.

### Task 4: Near-Real-Time Progressive Streaming (`ChunkConfig`)
- Wire up `ChunkConfig` in `pipeline.py` to enable sliding-window reconstruction chunks, outputting a preliminary watertight 3D model within the first 60 seconds while dense MVS processes asynchronously.
