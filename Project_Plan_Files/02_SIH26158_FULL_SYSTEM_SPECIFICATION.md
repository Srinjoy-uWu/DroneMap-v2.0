# SIH26158 Full System Specification
## Industrial UAV Photogrammetry & Spatial Computing Architecture

---

## 1. NTRO Problem Statement & Challenge Mapping

Smart India Hackathon problem statement **SIH26158** states:
> **"Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model"**
> *Problem Owner: National Technical Research Organisation (NTRO)*

The table below defines how each mandated challenge is solved in DroneMap v2.0:

| Challenge ID | Challenge Description | Vulnerability in Standard Tools | DroneMap v2.0 Algorithmic Solution | Verification Metric |
|---|---|---|---|---|
| **CH-01** | Limited viewing angles (single-pass flight) | Classical SfM fails due to collinear camera positions and narrow parallax. | Sequential matching with quadratic stride + learned LightGlue matcher; monocular depth priors (Depth Anything V2) provide single-view geometric bounds. | 100% camera registration on single-pass orbital footage (`test_orbit_georef`). |
| **CH-02** | UAV motion blur & dynamics | High-frequency UAV vibrations blur frames, breaking SIFT corner detection. | O(1) streaming 2-pass frame analyzer evaluates Laplacian variance $\text{Var}(\nabla^2 I)$ in sliding windows, rejecting blurry clusters. | Median sharpness $\ge 1500$, zero blurred frames passed to SfM. |
| **CH-03** | Video compression macroblocking | H.264/H.265 DCT block edges create false corners in feature extraction. | Gradient entropy filtering suppresses block boundaries; bicubic downsampling to native GSD removes high-frequency ringing. | Sub-pixel feature reprojection error $< 0.8\text{ px}$. |
| **CH-04** | Variable illumination & sun glare | Brightness changes break photometric consistency across views. | CLAHE contrast equalization balances shadows and highlights; OpenMVS multi-view consistency filtering rejects glare artifacts. | Texture atlas black-fraction check ensures $0\%$ collapsed texels. |
| **CH-05** | Transient shadows | Moving cloud shadows cause contradictory multi-view matching. | Adaptive dynamic masking and edge-preserving normalization distinguish static structural edges from transient cast shadows. | Clean surface reconstruction in tree shadow regions. |
| **CH-06** | Dynamic moving objects (vehicles, humans, animals) | Moving objects create smeared road bumps and airborne ghost spikes. | YOLOv8s-seg on CUDA identifies dynamic objects; applies velocity-adaptive morphological dilation; exports non-destructive binary PNG masks to COLMAP. | Dense point cloud in `05_dense/` contains zero vehicle floaters on roadways. |
| **CH-07** | GPS inaccuracies & multi-path drift | Consumer GPS has 3–5m horizontal drift and poor vertical accuracy. | Robust Sim(3) Umeyama alignment with RANSAC outlier rejection; local origin shift prevents floating-point shader jitter. | Reported alignment RMSE **3.91 m** without ground survey targets. |
| **CH-08** | Sensor noise & rolling shutter | CMOS rolling shutter distorts vertical building edges during yaw. | Bundle adjustment optimizes radial distortion ($f, c_x, c_y, k_1$); lower-envelope PCA suppresses high-frequency sensor noise. | Ground plane RMS residual of **0.034 m** on real drone flights. |
| **CH-09** | Occluded regions (behind buildings / under trees) | Single-pass leaves shadowed ground voids without multi-view rays. | Watertight 2.5D Digital Surface Model KDTree IDW interpolation fills occluded voids; points tagged with 3-tier confidence (Observed / Estimated / Inferred). | 100% watertight DSM mesh without holes or spikes. |
| **CH-10** | Limited Ground Control Point (GCP) availability | Surveying physical GCPs is impossible in tactical or denied territory. | Direct georeferencing via telemetry parsing (DJI SRT / CSV) mapped to local projected UTM CRS (e.g. `EPSG:32643`). | Model automatically georeferenced in true-north ENU coordinates. |

---

## 2. End-to-End Pipeline Stages & Data Contracts

```text
[Input Video (.mp4/.mov)] + [Flight Telemetry (.srt/.csv)]
                           │
                           ▼
                  [Stage 1: Frames]
            O(1) Streaming Laplacian Filter
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
  01_frames/images/*.jpg              01_frames/keyframes.json
         │
         ▼
                  [Stage 2: Masks]
             YOLOv8s-seg + Adaptive Dilation
         │
         ▼
  02_masks/masks/*.png (255=dynamic, 0=static)
         │
         ├───────────────────────────────────┐
         ▼                                   ▼
  [Stage 4a: Depth]                  [Stage 4b: Semantics]
  Depth Anything V2 / Metric3D        SegFormer UAVid / LoveDA
  02b_depth/raw/*.png                 04_semantics/sih26158_categories.json
         │                                   │
         └─────────────────┬─────────────────┘
                           │
                           ▼
                  [Stage 3a: Pose]
             COLMAP SIFT + LightGlue Bridge
         │
         ├── 03_pose/sparse/0/ {cameras, images, points3D}.bin
         ▼
  03_pose/undistorted/ (Pinhole-rectified frames)
         │
         ▼
                  [Stage 3b: Georef]
             WGS-84 -> ECEF -> ENU -> UTM
             Sim(3) Umeyama Alignment
         │
         ▼
  03b_georef/sparse_enu/transform.json
         │
         ▼
           [Quality Gate & Routing Engine]
       Triangulation angle, parallax, planarity
         │
     ┌───┴───────────────────────────────┐
     ▼ (ACCEPT_3D)                       ▼ (TERRAIN_2_5D or Fallback)
[Stage 5: Dense MVS]               [terrain.py Engine]
 OpenMVS DensifyPointCloud          Lower-envelope PCA + IDW Grid
 05_dense/scene_dense.ply           Watertight TIN Surface Mesh
     │                                   │
     ▼                                   │
[Stage 6: Mesh & Texture]                │
 OpenMVS Reconstruct + Texture           │
 Atlas UV Luminance Check                │
 06_mesh/*_texture.obj                   │
     │                                   │
     └─────────────────┬─────────────────┘
                       │
                       ▼
              [Stage 7: Export]
        GLB, LAZ, DSM, DTM, Ortho, KML
                       │
                       ▼
         [Air-Gapped Web 3D Studio]
         FastAPI + Three.js Measurement
```

---

## 3. SIH26158 Semantic Classification Rollup

Stage 4b (`stage4_semantics.py`) ingests aerial foundation models (SegFormer-B0 fine-tuned on UAVid or LoveDA) and rolls up all segmented classes into the **4 mandatory problem statement categories**:

```json
{
  "terrain": {
    "target_name": "Terrain / Bare Earth",
    "description": "Ground, dirt, bare soil, sand, low vegetation, grass",
    "classes": ["clutter", "bare_soil", "grass"]
  },
  "buildings": {
    "target_name": "Buildings / Structures",
    "description": "Commercial buildings, residential roofs, industrial structures, walls",
    "classes": ["building", "roof"]
  },
  "roads_infrastructure": {
    "target_name": "Roads & Infrastructure",
    "description": "Roads, pavement, highways, tarmac, railways, bridges",
    "classes": ["road", "sidewalk"]
  },
  "vegetation_obstacles": {
    "target_name": "Vegetation & Obstacles",
    "description": "Trees, dense canopy, static obstacles, water bodies",
    "classes": ["tree", "vegetation", "water"]
  }
}
```

The rollup results are serialized to `04_semantics/sih26158_categories.json` and served to the Web Studio for instant distribution breakdown.

---

## 4. Standardized Deliverable Specifications

Every completed run produces a full suite of OGC/ASPRS-compliant spatial deliverables under `07_export/`:

| Deliverable | File Path | Format / Specification | Spatial Reference System | Primary Customer Use |
|---|---|---|---|---|
| **Textured 3D Mesh** | `07_export/model.glb` | glTF 2.0 Binary (Y-up orientation, embedded JPEG atlas) | Local Centroid Metric (meters) | Tactical 3D visualization, Three.js web studio, mission briefing. |
| **Dense Point Cloud** | `07_export/cloud.laz` | ASPRS LAS 1.4 compressed, RGB vertex color, confidence extra-bytes | Projected UTM (e.g. `EPSG:32643`) | Engineering volumetric measurement, CAD/GIS import. |
| **Digital Surface Model (DSM)** | `07_export/dsm.tif` | GeoTIFF (Single-band Float32, native GSD, meters) | Projected UTM | Line-of-sight analysis, flood modeling, obstacle clearance. |
| **Digital Terrain Model (DTM)** | `07_export/dtm.tif` | GeoTIFF (Bare-earth raster via morphological ground filter) | Projected UTM | Slope analysis, hydrological drainage, foundation planning. |
| **Orthomosaic** | `07_export/orthomosaic.tif` | GeoTIFF (3-band RGB, nadir projection, native GSD) | Projected UTM | 2D cartographic base-mapping, land-use change detection. |
| **Camera Trajectory** | `07_export/trajectory.kml` & `.json` | OGC KML 2.2 / GeoJSON with altitude coordinates | WGS-84 (`EPSG:4326`) | Google Earth trajectory playback, flight audit. |
| **Survey Accuracy Report** | `07_export/report.html` & `accuracy_report.json` | Self-contained HTML + machine-readable JSON | WGS-84 / UTM | Formal survey certification, jury compliance verification. |

---

## 5. Non-Negotiable Architectural Guardrails

To prevent regressions, catastrophic memory leaks, or false scientific claims, every modification must uphold these six rules:

1. **Truthful Georeferencing**:
   - If GPS alignment fails, is missing, or residual RMSE exceeds threshold, coordinate frame must remain `"LOCAL_RELATIVE"`.
   - Never assign a projected UTM / EPSG code to an unaligned run. Never claim 0.00 m alignment error.
2. **GPU Memory Lifecycle Safety**:
   - Every PyTorch inference stage (`stage2_masks.py`, `stage4_depth.py`, `stage4_semantics.py`) must be wrapped in `try...finally`:
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
   - In `stage3_pose.py`, exhaustive matching escalation is capped at $N \le 150$ frames. If $N > 150$, bypass exhaustive matching and escalate directly to vocabulary tree matching to prevent $O(N^2)$ runaway latency.
4. **Air-Gapped Offline Operation**:
   - All Three.js libraries and UI assets in `src/dronemap/api/static/` must be 100% vendored and self-contained. Never add CDN script tags or live internet calls.
5. **Watertight 2.5D Terrain Fallback**:
   - If OpenMVS dense meshing fails or capture quality is planar nadir, automatically route to `terrain.py` (KDTree IDW ground plane fitting) to produce a guaranteed watertight DSM mesh.
6. **Zero Regression Policy**:
   - All 157 existing tests in `tests/` must pass (`pytest -q`) on every code change.
