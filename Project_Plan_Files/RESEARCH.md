# Research & Technology Evaluation — DroneMap (SIH26158)

This document provides a research-backed survey of computer vision, deep learning, and photogrammetric technologies evaluated for the DroneMap UAV reconstruction system.

---

## 1. Feature Extraction & Matching Architectures

In single-pass drone videography, forward camera motion, tilt angles, and variable ground textures create significant matching challenges.

| Method | Backbone / Architecture | Invariance & Robustness | Latency (per image pair) | VRAM Footprint | License | Verdict & Trade-offs |
|---|---|---|---|---|---|---|
| **SIFT (OpenCV / COLMAP)** | Difference-of-Gaussians + Gradient Histograms | Scale & rotation invariant; fails under extreme perspective tilt or low-texture tarmac | ~15 ms (GPU CUDA) | $< 500$ MB | BSD-3-Clause | **Primary Workhorse**: Fast, mathematically grounded, exact sub-pixel localization, zero training bias. |
| **SuperPoint + LightGlue (Lindenberger et al., 2023)** | Fully convolutional detector + Transformer graph matcher | Exceptional repeatability in low texture and large viewpoint changes | ~28 ms (PyTorch CUDA) | ~1.2 GB | LightGlue: Apache-2.0; SuperPoint weights: Magic Leap Research (Non-Commercial) | **Recommended Neural Matcher**: 4–5× faster than SuperGlue, memory-efficient on 6GB VRAM, highly robust on repetitive roofs and roads. Use [hloc](https://github.com/cvg/Hierarchical-Localization) schema for battle-tested colmap.db injection. |
| **LoFTR (Sun et al., 2021)** | Detector-free transformer with coarse-to-fine self/cross attention | Matches in textureless surfaces; no keypoint repeatability bottleneck | ~85 ms (PyTorch CUDA) | ~3.5 GB | Apache-2.0 | **High Accuracy Fallback**: High VRAM consumption; prone to memory bottlenecks on large image batches. |
| **MASt3R (Leroux et al., 2024)** | ViT backbone + cross-attention dense 3D point prediction | Jointly solves matching and dense 3D reconstruction without SIFT | ~350 ms (PyTorch CUDA) | ~5.2 GB | CC BY-NC-SA 4.0 | **Implemented Stretch Backend**: Excellent for extreme oblique/façade views; heavy memory footprint on 6GB laptops. Recommended for Cloud/Kaggle tier. |

### Architectural Conclusion:
Maintain GPU SIFT as the high-throughput sequential default, with SuperPoint + LightGlue available for difficult, low-texture, or high-bank-angle camera pairs. For injecting custom keypoints and matches into COLMAP's SQLite database, follow the production-proven conventions from [cvg/Hierarchical-Localization (hloc)](https://github.com/cvg/Hierarchical-Localization) to avoid database corruption or degraded bundle adjustment.

---

## 2. Monocular Depth Estimation Models

Monocular learned depth estimates scene depth from a single frame using deep geometric priors.

| Model | Architecture | Output Nature | Accuracy / RMSE | Hardware / Latency | License | Suitability for Drone Photogrammetry |
|---|---|---|---|---|---|---|
| **Depth Anything V2 (Yang et al., 2024)** | DINOv2 encoder + DPT decoder | Relative depth $d \in [0, 1]$ | High ordinal consistency; scale-free | Small: 25 ms, 0.8 GB VRAM; Base: 60 ms, 1.8 GB VRAM | Apache-2.0 | **Implemented**: Excellent edge crispness. Requires telemetry altitude scaling ($h / \cos\theta$) for metric conversion. |
| **Metric3D v2 (Yin et al., 2024)** | ViT-Small/Large with canonical camera transformation | Absolute metric depth (metres) | Direct zero-shot metric depth | Small: 65 ms, 1.4 GB VRAM | Apache-2.0 | **Implemented**: Universal metric prior; highly effective on oblique building views where nadir assumption fails. |
| **ZoeDepth (Bhat et al., 2023)** | Relative encoder + metric bin heads | Metric depth | Good indoor/outdoor metric scale | ~110 ms, 2.5 GB VRAM | MIT | Slower than Depth Anything V2; struggles with aerial viewpoints above 50m. |

### Architectural Conclusion:
Depth Anything V2 (Small) provides the optimal balance of inference speed and structural boundary accuracy on a 6GB laptop GPU. Monocular depth must be treated as a geometric prior and confidence mask, **never as ground-truth survey measurements**.

---

## 3. Dynamic Object & Transient Segmentation

The SIH26158 problem statement explicitly requires mitigating moving vehicles, pedestrians, and animals in drone footage.

| Model / Approach | Parameters | Dynamic Classes Covered | Latency | Mask Quality & Boundary Dilation | License | Operational Role |
|---|---|---|---|---|---|---|
| **YOLOv8s-seg (Ultralytics)** | 11.8M params | 14 dynamic aerial classes (vehicles, people, animals) | ~12 ms (CUDA) | Coarse polygonal boundary; requires $12$ px elliptical dilation to absorb motion blur | AGPL-3.0 | **Current Production Standard**: Fast, robust, operates in real-time during keyframe ingestion. |
| **Segment Anything 2 (SAM-2)** | 22.4M (Tiny) to 224M (Large) | Promptable zero-shot masks | ~90 ms (Tiny) | Sub-pixel accurate boundary segmentations | Apache-2.0 | High latency per frame; better suited for offline post-processing or interactive user annotation. |
| **SegFormer (Xie et al., 2021)** | 3.7M (B0) to 84M (B5) | Multi-class land-use (roads, buildings, trees) | ~20 ms (B0 CUDA) | Semantic classification of terrain | Apache-2.0 | **Supported in Stage 4b**: Requires domain-specific UAV checkpoints (UAVid/LoveDA); generic ADE20K weights fail on nadir views. |

---

## 4. 3D Representation Paradigm: MVS Meshes vs 3D Gaussian Splatting vs NeRF

| Metric / Requirement | Classical Multi-View Stereo (OpenMVS) | 2.5D Surface Mesh (`terrain.py`) | 3D Gaussian Splatting (3DGS) | Neural Radiance Fields (NeRF) |
|---|---|---|---|---|
| **Metric Accuracy** | High (derived from bundle adjustment rays) | High (KDTree IDW on registered points) | Moderate (visual radiance optimized, not metric coordinates) | Moderate (density volume optimized for photometric loss) |
| **GIS Deliverables (LAZ, DSM, DTM, Ortho)** | Native & direct from point cloud / mesh | Native & direct (exact raster elevation grid) | Requires rasterization / surface meshing approximation | Requires marching cubes / ray marching |
| **CAD / Engineering Use** | Direct OBJ / GLB polygon compatibility | Direct OBJ / GLB polygon compatibility | Radiance ellipsoids; incompatible with standard GIS/CAD tools | Implicit neural weights; incompatible with standard GIS/CAD tools |
| **Runtime on 6GB GPU** | ~30–50 min (laptop) | ~10–20 sec (pure Python / KDTree) | ~15–25 min training | ~45–90 min training |
| **Air-Gapped Web Viewport** | 100% standard WebGL (Three.js) | 100% standard WebGL (Three.js) | Heavy WebGL Gaussian shaders (high VRAM requirement on client) | Neural rendering requires dedicated client compute |

### Architectural Conclusion:
For an industrial GIS and defense photogrammetry solution addressing SIH26158, **classical MVS point clouds + watertight 2.5D/3D polygon meshes are indispensable**. 3DGS offers impressive visual novel-view synthesis, but cannot directly replace metrically validated DSM/DTM/LAZ GIS deliverables.

---

## 5. Hardware & Latency Budget (Target: RTX 3050 6GB Laptop GPU)

To guarantee that the entire pipeline runs reliably on consumer/laptop hardware without out-of-memory crashes:

| Sub-pipeline Component | Peak VRAM Allocation | Peak System RAM | Target Latency (per 30-keyframe run) | Memory Mitigation Strategy |
|---|---|---|---|---|
| **Video Decoding (Stage 1)** | 0 MB (CPU) | ~350 MB | ~30–50 sec | 2-pass streaming; zero raw frames cached in RAM |
| **Dynamic Masking (Stage 2)** | ~1.1 GB (CUDA) | ~400 MB | ~25–35 sec | Batch size = 1; immediate tensor release |
| **COLMAP SIFT Pose (Stage 3a)**| ~2.4 GB (CUDA) | ~1.5 GB | ~3–10 min | Sequential matching; sequential overlap = 10 |
| **Depth Anything V2 (Stage 4a)**| ~0.9 GB (CUDA) | ~600 MB | ~15–20 sec | Half-precision (FP16) inference |
| **OpenMVS Dense (Stage 5)** | ~3.8 GB (CUDA) | ~3.5 GB | ~15–25 min | Resolution downscale level = 1 or 2 |
| **Mesh Texturing (Stage 6)** | ~2.5 GB (CUDA) | ~4.0 GB | ~8–15 min | Texture atlas cap: 8192×8192 |
| **Terrain Mesh Engine (`terrain.py`)**| 0 MB (CPU) | ~800 MB | ~10–20 sec | Vectorized SciPy KD-Tree spatial indexing |

---

## 6. Online & Cloud Acceleration Architecture (Kaggle Dual-T4, Google Colab, Cloud GPUs)

While the pipeline is fully engineered to run locally on consumer laptop GPUs, **DroneMap natively supports high-throughput cloud and notebook accelerators**:

### A. Kaggle Dual-T4 GPU Environment (30 hr/week free)
- **Hardware Profile**: 2× NVIDIA Tesla T4 GPUs (32 GB total VRAM), 30 GB system RAM, 4 CPU cores.
- **Enabled Foundation Capabilities**:
  - **Native 4K Processing**: Frames processed at unconstrained $3840 \times 2160$ resolution (`max_long_edge: 3840`).
  - **Depth Anything V2 Large**: Full 335M parameter transformer backbone generating sub-centimeter relative depth boundaries.
  - **Metric3D-v2 ViT-Large**: Zero-shot universal metric depth estimation across high-altitude and oblique camera views.
  - **16K Texture Atlases**: OpenMVS texture projection configured for $16384 \times 16384$ texel maps, preserving microscopic surface details on roofs, brickwork, and road markings.
  - **3D Gaussian Splatting (3DGS)**: Integrated execution with `notebooks/3dgs_splatfacto_kaggle.ipynb` via Nerfstudio and `gsplat`, training 30,000-iteration radiance fields in ~25–35 minutes.

### B. Dual-Tier Workflow: Local Edge + Cloud Acceleration
1. **Local Run**: `dronemap run --video <video> --profile default` (instant local feedback, offline verification).
2. **Kaggle Run**: Run `notebooks/sih26158_3dgs_kaggle.py` on Kaggle with `configs/accelerated.yaml` to generate heavy neural point clouds and splats.
3. **Local Packaging**: Re-import high-resolution deliverables into `data/runs/<id>/` for standard GIS packaging (GLB, LAZ, GeoTIFFs, interactive viewer).

