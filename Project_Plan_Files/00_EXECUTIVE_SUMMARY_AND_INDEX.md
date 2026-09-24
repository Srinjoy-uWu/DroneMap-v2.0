# Project Plan Files — Master Index & Executive Overview
## DroneMap v2.0 · Smart India Hackathon Problem Statement SIH26158

---

## 1. Project Overview & Mission

**DroneMap v2.0** is an industrial-grade, failsafe UAV photogrammetry and spatial intelligence pipeline designed to solve Smart India Hackathon problem statement **SIH26158**:
> **"Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model"**
> *Problem Owner: National Technical Research Organisation (NTRO)*

The mission of this project is to take a continuous single-pass flyover video from an uncrewed aerial vehicle (UAV)—along with accompanying flight telemetry (GPS / SRT / CSV)—and reliably reconstruct a georeferenced, metrically accurate, watertight 3D model and GIS deliverable suite without relying on ground control points (GCPs) or multi-pass cross-grid flight patterns.

---

## 2. Directory Structure of `Project_Plan_Files`

All planning, architectural specifications, empirical baselines, research surveys, and hackathon presentation assets are organized in this dedicated directory:

```text
Project_Plan_Files/
├── 00_EXECUTIVE_SUMMARY_AND_INDEX.md           <-- This master navigation guide
├── 01_COMPREHENSIVE_RESEARCH_AND_LITERATURE_SURVEY.md <-- Deep research: SfM, Depth, 3DGS, Geodesy
├── 02_SIH26158_FULL_SYSTEM_SPECIFICATION.md    <-- Problem mapping, requirements, deliverables, guardrails
├── 03_END_TO_END_IMPLEMENTATION_ROADMAP_V2.md  <-- Phased execution plan for DroneMap v2.0
├── 04_HACKATHON_JURY_DEFENSE_AND_EVALUATION.md <-- Jury demo script, scoring rubrics, defense arguments
├── 05_AI_AND_LLM_API_INTEGRATION_SPEC.md       <-- LLM/Multimodal APIs, Autonomous Agent, Chat-with-Model
├── 06_BEFORE_AFTER_AUDIT_AND_COMPLETE_MESH_SPEC.md <-- Visual before/after diagnostics & mesh perfection
│
├── ENGINEERING_GUIDELINES.md                   <-- Engineering standards, guidelines, entry points
├── AI_ROADMAP.md                               <-- Incremental AI enhancement milestones & ablation plan
├── AS_IS_ARCHITECTURE.md                       <-- Component-by-component anatomy of current codebase
├── BASELINE.md                                 <-- Empirical benchmark audit of real UAV runs
├── CURRENT_DATA_FLOW.md                        <-- Stage-by-stage I/O schemas and manifest definitions
├── REAL_PROCESS_AND_SAMPLES.md                 <-- Visual diagnostic proof (OpenCV, YOLO, COLMAP, OpenMVS)
├── RESEARCH.md                                 <-- Core technology trade-offs and latency budgets
├── SIH26158_MAPPING.md                         <-- NTRO challenge compliance matrix
├── V2_ARCHITECTURE.md                          <-- Production dual-tier edge/cloud architecture
│
├── images/                                     <-- Visual evidence diagrams and sample outputs
│   ├── 01_opencv_analysis.png
│   ├── 02_yolo_masking.png
│   ├── 03_colmap_sift_matching.png
│   ├── 04_colmap_sparse_reconstruction.png
│   └── 05_dense_and_mesh_quality.png
│
├── SIH26158_Research_Dossier.md                <-- Official hackathon research dossier (NTRO analysis)
├── SIH26158_Implementation_Plan_v4 (1).md      <-- Hackathon engineering plan (core vs stretch)
├── JURY_DEMO_SCRIPT.md                         <-- Live jury walkthrough and demo narrative
├── SIH26158_5Slide_Official_PPT.md             <-- 5-slide hackathon presentation deck
├── SIH26158_PPT_Presentation_Guide.md          <-- Complete presentation script and visual cues
├── IMPROVEMENT_PLAN_VERIFIED.md                <-- Verification logs and system hardening records
├── implementation_plan.md                      <-- Historical plan archive
└── walkthrough.md                              <-- Hardening and verification walkthrough log
```

---

## 3. The Core Challenge of SIH26158

Traditional photogrammetry tools (COLMAP, Pix4D, OpenDroneMap, Metashape) mandate **multi-pass grid flights with 70–85% front and side overlap**. When presented with a single-pass forward flight, traditional pipelines fail due to:
1. **Narrow baseline & weak parallax**: Triangulation error scales inversely with baseline; narrow baselines produce severe height/scale drift ("doming").
2. **One viewing direction**: Occluded building facades, terrain under canopies, and rear surfaces receive zero multi-view rays.
3. **Dynamic objects**: Moving cars, trucks, and pedestrians create ghost geometry, surface spikes, and distorted roads.
4. **Motion blur & compression artifacts**: High-frequency UAV vibrations and H.264 macroblocking destroy SIFT keypoint repeatability.
5. **No Ground Control Points (GCPs)**: Direct georeferencing must rely strictly on consumer GNSS telemetry, requiring robust outlier rejection and Sim(3) coordinate geodesy.

---

## 4. DroneMap's Hybrid Solution Philosophy

> **"Epipolar Geometry is Ground Truth. AI Filters, Guides, and Constrains — Never Hallucinates."**

- **Physics-First Core**: SfM bundle adjustment (COLMAP), closed-form geodesy (WGS-84 $\leftrightarrow$ ECEF $\leftrightarrow$ ENU $\leftrightarrow$ UTM), and Delaunay/KDTree interpolation form the unbreakable geometric foundation.
- **Targeted AI Intelligence**:
  - **Stage 1 (OpenCV + Multi-factor AI)**: $O(1)$ constant-RAM streaming keyframe selection based on Laplacian sharpness, exposure, contrast, and optical flow stride.
  - **Stage 2 (YOLOv8s-seg)**: Non-destructive binary masking of dynamic vehicles and pedestrians with velocity-adaptive dilation.
  - **Stage 3 (Dual-Engine SfM)**: Native GPU SIFT for rapid sequential matching + LightGlue neural matcher bridge for low-texture/oblique pairs.
  - **Stage 4a (Monocular Depth Priors)**: Depth Anything V2 / Metric3D provides occlusion filling and zero-shot scale cross-checking.
  - **Stage 4b (Aerial Semantics)**: SegFormer maps terrain into the 4 mandatory SIH26158 classes ((i) terrain, (ii) buildings, (iii) roads & infrastructure, (iv) vegetation & obstacles).
  - **Stage 5 & 6 (Watertight Meshing & 2.5D Fallback)**: OpenMVS dense MVS for volumetric scenes, automatically falling back to `terrain.py` (KDTree IDW ground fitting) for flat/nadir terrain to guarantee zero airborne spikes or hollow meshes.
  - **Stage 7 (Industrial GIS Deliverables)**: Full export of GLB (Y-up), ASPRS LAS 1.4 LAZ, DSM/DTM/Ortho GeoTIFFs, trajectory KML/JSON, and audit report.
  - **Web Measurement Studio**: 100% air-gapped, offline Three.js 3D viewport with real-world metric ruler, shading switchers, and pipeline inspection drawer.

---

## 5. Dual-Tier Deployment Profile

| Dimension | Local Edge Tier (Field Operation) | Cloud / Kaggle Accelerator Tier |
|---|---|---|
| **Target Hardware** | NVIDIA RTX 3050 6GB Laptop GPU, 16 GB RAM | Dual NVIDIA Tesla T4 (32 GB) or A100 |
| **Video Resolution** | 1080p / 1600 px downscale | Native 4K ($3840 \times 2160$) |
| **VRAM Ceiling** | Strict $< 6.44\text{ GB}$ (FP16, batch size 1) | Unconstrained (up to 32 GB) |
| **Mesh Texturing** | 8192×8192 Texture Atlas | 16384×16384 Texel Atlas |
| **Neural Rendering** | Watertight 2.5D / MVS mesh | 3D Gaussian Splatting (`splatfacto` / SuGaR) |
| **Turnaround Time** | Fast preview in ~60s; full in 15–40 min | High-throughput batch processing |
| **Connectivity** | 100% Offline & Air-gapped | Remote notebook execution via Kaggle |

---

## 6. Current Baseline Status

- **Test Suite**: **157 / 157 unit, integration, and hardening tests passing (100%) in ~26 seconds** on Python 3.11 (`pytest -q`).
- **Real UAV Flight Benchmark (`test_real_drone_0904`)**:
  - 4K UAV orbit (`0904.mp4`), 26/26 cameras registered (100%).
  - 34,201 sparse 3D points, 211,268 dense points, 10.1 cm/px GSD.
- **Georeferenced Flight Benchmark (`test_orbit_georef`)**:
  - 47/47 cameras registered (100%), 668,657 dense points.
  - GPS alignment RMSE: **3.91 m** using raw consumer drone telemetry without GCPs.
- **Watertight Mesh Guarantee**: Zero airborne floaters, zero inverted pits, zero dark/black textures (verified via UV texel luminance sampling).
