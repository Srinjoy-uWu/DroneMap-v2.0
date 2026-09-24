# Baseline Evaluation & System Capabilities — DroneMap (SIH26158)

This baseline document provides an empirical audit of the system's current performance, measured outputs, capabilities, and limitations prior to AI pipeline expansion.

---

## 1. Measured Performance & Benchmark Audit

The figures below represent **real, recorded runs** stored in this repository's `data/runs/` and `data/benchmark_summary.json`. No numbers are fabricated or assumed.

| Run Identifier | Video / Scene Type | GNSS Mode | Registered Frames / Total | Sparse 3D Points | Dense Point Cloud | Mesh Vertices / Faces | Ground Sampling Distance (GSD) | Georef Alignment RMSE | Total Runtime |
|---|---|---|---|---|---|---|---|---|---|
| **`test_real_drone_0904`** | Real 4K UAV Orbit (3840×2160, 7.1s) | `NONE` (Local) | 26 / 26 (100%) | 34,201 | 211,268 | 70,688 / 139,579 | 10.1 cm/px | N/A (Local Relative) | ~50 min (with OpenMVS dense) |
| **`test_orbit_georef`** | Georeferenced UAV Orbit (1600×900) | `STANDALONE` (SRT) | 47 / 47 (100%) | 23,280 | 668,657 | 46,500 / 92,917 | 23.3 cm/px | 3.91 m (GPS) | 48 min (with dense & georef) |
| **`synthetic_eval_run`** | Synthetic City Grid (Ground Truth) | `STANDALONE` (GPS) | 30 / 30 (100%) | 18,450 | 199,338 | 30,797 / 61,278 | 5.76 cm/px | 2.14 m (GPS) | 7.9 min |
| **`demo_aukerman_hd`** | Building Structure Video (HD) | `NONE` (Local) | 17 / 17 (100%) | 21,800 | 319,911 | 97,696 / 194,543 | 5.0 cm/px | N/A (Local Relative) | 14.1 min |
| **`demo_aukerman`** | Fast Smoke Test (10 frames) | `NONE` (Local) | 10 / 10 (100%) | 8,920 | 54,232 | 6,312 / 12,462 | 5.0 cm/px | N/A (Local Relative) | 1.0 min |

### Environment Specifications (Measured System)
- **Host Platform**: Windows 10 / 11 (build 10.0.26200)
- **Python Version**: 3.11.16
- **GPU Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.44 GB VRAM)
- **CUDA Runtime**: 12.8 (PyTorch 2.11.0+cu128)
- **Test Suite Execution**: 157 passed unit, integration & hardening tests in 3.94s (`python -m pytest -q`)

---

## 2. Current Capabilities

1. **Streaming Constant-Memory Video Ingestion**:
   - Streams 4K / 1080p drone videos without loading entire footage into RAM ($O(1)$ memory consumption).
   - Evaluates Laplacian sharpness in sliding temporal windows to eliminate motion-blurred clusters.
   - Adjusts keyframe stride using optical flow or drone velocity and flight trajectory turn rate.
2. **Dynamic Object Segmentation**:
   - Integrates YOLOv8s-seg on CUDA to generate binary dilation masks (vehicles, pedestrians, animals).
   - Masks feed directly into COLMAP's feature reader without modifying original JPEG frames on disk.
3. **Rigorous Camera Tracking & Escalation Ladder**:
   - Fast sequential matching with quadratic stride and periodic loop closure detection.
   - Escalation to exhaustive matching or FAISS vocabulary tree matching when camera registration drops below 55%.
   - Process failure diagnostics separating OOM/crashes from photographic rejection.
4. **Geodetic Coordinate Handling**:
   - Parses DJI SRT subtitles and generic CSV flight logs.
   - Converts WGS-84 to ECEF and local ENU with automatic UTM CRS resolution.
   - Center-shifts coordinate origin to preserve single-precision floating-point precision in 3D graphics shaders.
5. **Quality Gating & Geometric Routing**:
   - Measures multi-view triangulation angles, baseline/depth ratios, and depth dynamic range.
   - Automatically routes low-parallax or planar terrain scenes to a specialized 2.5D surface reconstruction engine (`terrain.py`).
6. **Robust 2.5D Terrain Reconstruction Engine**:
   - Lower-envelope ground plane PCA fitting with tilt rejection ($> 15^\circ$).
   - Watertight TIN mesh generated via KDTree Inverse Distance Weighting interpolation.
   - Texture atlas baking with zero degenerate floaters.
7. **Complete Deliverable Generation**:
   - Exports GLB (rotated to glTF standard $Y$-up), LAZ 1.4 point clouds, DSM, DTM, Orthomosaic GeoTIFFs, trajectory KML/JSON, and interactive HTML audit reports.
8. **Air-Gapped Web Viewport**:
   - 100% offline Three.js 3D viewport with interactive distance and elevation ruler, white/textured shading, and recency-based project navigation.

---

## 3. Current Limitations & Bottlenecks

1. **Uniform Downstream Processing of Keyframes**:
   - While Stage 1 filters for blur and stride, candidate keyframes are not yet ranked by *semantic richness*, *viewpoint entropy*, or *perceptual uniqueness*.
   - A straight hover or slow panning over uniform tarmac/grass still consumes downstream GPU compute.
2. **SIFT Vulnerability in Repetitive / Low-Texture Scenes**:
   - COLMAP SIFT features struggle on water surfaces, uniform sand, flat agricultural fields, or asphalt with low contrast.
   - While MASt3R is available as a stretch backend, it requires external repository cloning and has higher latency. A lightweight learned matcher (such as LightGlue) would dramatically improve registration on low-texture surfaces without breaking the COLMAP pipeline.
3. **Monocular Depth Disconnect**:
   - Stage 4a supports Depth Anything V2 and Metric3D v2, but their depth maps currently exist as standalone exports rather than being fused into sparse bundle adjustment or MVS depth filtering.
4. **Shadow and Lighting Transients**:
   - Dynamic masking handles physical objects (cars, people), but moving cloud shadows and specular water reflections can still deceive SIFT descriptors.
5. **Occlusion & Confidence Visibility**:
   - The 3D viewer displays the final reconstructed mesh, but does not visually convey regional confidence or highlight occluded zones where single-pass footage had insufficient viewing angles.
6. **OpenMVS Processing Duration**:
   - Multi-view stereo (`DensifyPointCloud` and `RefineMesh`) is by far the slowest component (~30 to 50 minutes on a 6GB laptop).

---

## 4. Current AI Usage Audit

| Stage | AI / ML Model | Framework | Purpose | Status in Codebase |
|---|---|---|---|---|
| **Stage 2 (Masks)** | YOLOv8s-seg (`yolov8s-seg.pt`) | `ultralytics` / PyTorch CUDA | Dynamic object instance segmentation | Active & production-ready |
| **Stage 3a (Pose)** | MASt3R ViT Backbone | Custom PyTorch | Transformer-based dense SfM (stretch) | Implemented; ~5.2 GB VRAM; recommended for Cloud / Kaggle tier (high risk on local 6 GB) |
| **Stage 4a (Depth)** | Depth Anything V2 (`Small-hf`) | Hugging Face `transformers` | Relative monocular depth estimation | Implemented; standalone output |
| **Stage 4a (Depth)** | Metric3D v2 (`vit_small`) | Hugging Face `transformers` | Universal zero-shot metric depth | Implemented; standalone output |
| **Stage 4b (Semantics)**| SegFormer (`segformer-b0`) | Hugging Face `transformers` | Aerial land-use segmentation | Active (UAVid / LoveDA aerial classes mapped to terrain, buildings, roads, vegetation) |

---

## 5. Architectural Guardrails: What Must NOT Be Changed

To preserve system reliability and engineering credibility:
1. **Never Replace Classical Photogrammetry with Hallucinatory Generative AI**:
   - SfM epipolar geometry, bundle adjustment, and multi-view triangulation represent physical optical truth. AI must augment (filter, guide, refine, match), never invent unobserved geometry.
2. **Preserve Closed-Form Geodesy & Coordinate Math**:
   - WGS-84, ECEF, ENU, and UTM transformations in `stage3_georef.py` and `stage7_export.py` must remain mathematically exact.
3. **Preserve Fallback Pathways**:
   - If an AI model fails (e.g. CUDA OOM or missing weights), the pipeline must seamlessly fall back to classical classical algorithms (e.g. SIFT matching, optical flow stride, classical 2.5D terrain).
4. **Preserve 100% Offline Air-Gapped Operation**:
   - No external API calls, cloud telemetry, or live CDN script downloads.
