# Unified Production Architecture — DroneMap (SIH26158)

## 1. Architectural Philosophy

DroneMap is designed as a unified, defense-grade UAV photogrammetry and spatial computing system. Rather than replacing proven projective geometry with speculative generative AI, DroneMap employs a **hybrid architecture**:

> **Classical projective photogrammetry forms the unbreakable geometric foundation (ray intersection, bundle adjustment, geodesy), while targeted AI models remove physical sensor noise, filter redundant or degraded frames, suppress dynamic artifacts, match difficult viewpoints, and quantify regional confidence.**

```
                      [ Single-Pass UAV Video + GNSS Telemetry ]
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 1: AI Frame Intelligence       │
                     │   - O(1) constant-RAM streaming        │
                     │   - Multi-factor quality scoring       │
                     │     (sharpness, exposure, contrast)    │
                     │   - Perceptual redundancy pruning      │
                     │   - Dynamic turn-rate bounded stride   │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 2: Transient & Dynamic Masking │
                     │   - YOLOv8s-seg on CUDA                │
                     │   - Moving vehicle/pedestrian masks    │
                     │   - Motion blur halo dilation          │
                     │   - Non-destructive PNG mask export    │
                     └───────────────────┬────────────────────┘
                                         │
                     ┌───────────────────┴────────────────────┐
                     ▼                                        ▼
       ┌───────────────────────────┐            ┌───────────────────────────┐
       │ Stage 4a: Depth Priors    │            │ Stage 4b: Aerial Semantics│
       │ - Depth Anything V2       │            │ - SegFormer UAVid         │
       │ - Metric3D v2             │            │ - Surface class breakdown │
       │ - Occlusion boundary map  │            │ - Land-use mask layers    │
       └─────────────┬─────────────┘            └─────────────┬─────────────┘
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 3a: Dual-Engine SfM            │
                     │   - Fast SIFT / Learned Neural Matcher │
                     │   - Epipolar geometric verification    │
                     │   - Bundle Adjustment & Sparse Cloud   │
                     │   - Robust failure diagnostics         │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 3b: Precision Georeferencing   │
                     │   - WGS-84 / ECEF / ENU transformations│
                     │   - GPS trajectory Sim(3) alignment    │
                     │   - Projected UTM CRS auto-resolution  │
                     │   - Centroid shift for FP32 stability  │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 4: Geometric Quality Gating    │
                     │   - Triangulation angle & parallax     │
                     │   - Baseline-to-depth dynamic ratio    │
                     │   - Planarity & relief ratio analysis  │
                     │   - Automated routing verdict          │
                     └───────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
      [ Volumetric 3D Pathway ]                 [ 2.5D Terrain Surface Pathway ]
      ┌───────────────────────────┐             ┌──────────────────────────────┐
      │ Stage 5: Dense MVS        │             │ dronemap.terrain Engine      │
      │ - Patch-match stereo      │             │ - Robust lower-envelope PCA  │
      │ - Normal vector estimation│             │ - Canonical horizontal frame │
      │ - RGB vertex colorization │             │ - KDTree IDW elevation grid  │
      └─────────────┬─────────────┘             │ - Watertight TIN surface mesh│
                    │                           └──────────────┬───────────────┘
      ┌─────────────┴─────────────┐                            │
      │ Stage 6: Surface & Texture│                            │
      │ - Poisson mesh extraction │                            │
      │ - Fragment cleaning       │                            │
      │ - Texture atlas projection│                            │
      │ - Atlas UV black recovery │                            │
      └─────────────┬─────────────┘                            │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Stage 7: Multi-Deliverable Export    │
                     │   - glTF/GLB Textured 3D Mesh (Y-up)   │
                     │   - ASPRS LAZ 1.4 Point Cloud (UTM)    │
                     │   - GeoTIFF Rasters (DSM, DTM, Ortho)  │
                     │   - Trajectory KML / GeoJSON           │
                     │   - Multi-Level Confidence Map         │
                     │   - Verifiable Survey Audit Report     │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Air-Gapped Industrial Web Studio     │
                     │   - 3D Viewport with Orbit & Clay Mode │
                     │   - Real-World 3D Measurement Ruler    │
                     │   - Full Pipeline Transparency Drawer  │
                     │   - Evidence-Led Project Scorecard     │
                     └────────────────────────────────────────┘
```

---

## 2. Detailed Technical Enhancements

### Component 1: Intelligent Multi-Factor Frame Selection
- **The Challenge**: Uniform FPS sampling extracts redundant frames during hover and captures motion-blurred frames during turns.
- **The Solution**: An upgraded streaming frame analyzer combining:
  1. **Sharpness Score ($S$)**: Normalized Laplacian variance $\sigma^2(\nabla^2 I)$.
  2. **Exposure Score ($E$)**: Deviation of image luma histogram from target mean ($[80, 180]$) and clipping penalty for over/under-exposure.
  3. **Contrast Score ($C$)**: Standard deviation of pixel intensities across gray channels.
  4. **Perceptual Uniqueness ($U$)**: Lightweight optical flow vector magnitude and structural difference against preceding keyframe.
- **Combined Quality Index**:
  $$Q_i = w_s \cdot S_i + w_e \cdot E_i + w_c \cdot C_i + w_u \cdot U_i$$
- **Downstream Impact**: Retains only informative, high-contrast, non-blurred frames. Reduces downstream COLMAP feature extraction time by 30–40% with zero loss of scene coverage.

### Component 2: Advanced Dynamic Object & Shadow Masking
- **The Challenge**: Moving cars and pedestrians create phantom geometric floaters; moving cloud shadows alter photometric appearance.
- **The Solution**:
  1. **Instance Segmentation**: YOLOv8s-seg detects dynamic objects with class-specific confidence thresholds.
  2. **Motion Halo Dilation**: Morphological dilation using an elliptical structuring element covers moving boundary blur.
  3. **Non-Destructive Integration**: Masks are written as separate binary PNGs and passed via COLMAP's `--ImageReader.mask_path`, ensuring source photography remains pure.

### Component 3: Dual-Engine Robust Feature Matching
- **The Challenge**: SIFT descriptors fail in uniform surfaces (smooth roofs, asphalt, sand, water bodies) and across steep oblique angles.
- **The Solution**:
  1. **Primary High-Speed Path**: Native GPU SIFT with sequential matching for normal consecutive frames.
  2. **Learned Matcher Assistance**: High-ambiguity or low-texture pairs utilize deep feature matching (such as LightGlue or MASt3R ViT) to recover cross-view correspondences where SIFT yields $< 50$ matches.
  3. **Escalation Ladder**: Automatically switches to exhaustive or vocabulary tree matching if registered cameras drop below 55%.

### Component 4: Metric Depth Prior Fusion & Occlusion Analysis
- **The Challenge**: Monocular cameras cannot observe surfaces behind obstacles, and single-pass flight provides limited oblique viewing angles.
- **The Solution**:
  1. **Depth Prior Integration**: Depth Anything V2 or Metric3D estimates single-image relative/metric depth.
  2. **Scale Calibration**: Relative depth scaled using drone telemetry altitude ($h / \cos\theta_{\text{gimbal}}$).
  3. **Confidence Classification**: Triangulated vertices are categorized into explicit confidence classes:
     - **Observed (High Confidence)**: Directly triangulated by 3 or more camera rays with triangulation angle $\ge 5^\circ$.
     - **Estimated (Medium Confidence)**: Triangulated with small parallax ($2^\circ \le \theta < 5^\circ$) or dense MVS interpolation.
     - **Weak / Inferred (Low Confidence)**: Inferred from planar IDW interpolation or monocular depth prior across occluded boundaries.

### Component 5: 2.5D Terrain Surface Mesh Engine
- **The Challenge**: Nadir drone surveys of flat ground or agricultural terrain often cause 3D volumetric Delaunay reconstruction (OpenMVS) to create hollow roofs or spikes due to lack of vertical parallax.
- **The Solution**:
  1. Automatically detected by `quality.py` (`capture_verdict == "terrain_2_5d"`).
  2. Robust ground plane fitted to the lower elevation envelope (2nd to 30th percentile).
  3. Continuous, watertight 2.5D Digital Surface Model constructed via KDTree IDW grid interpolation.
  4. Texture atlas baked onto UV coordinates, producing a clean, floater-free surface.

### Component 6: Multi-Deliverable Industrial GIS Export
- **Formats Generated**:
  - **GLB**: Compact glTF binary with embedded texture atlas, rotated to standard glTF $Y$-up.
  - **LAZ**: ASPRS LAS 1.4 compressed point cloud tagged with exact projected UTM CRS.
  - **DSM & DTM GeoTIFFs**: Surface and bare-earth digital elevation models with valid geospatial metadata.
  - **Orthomosaic GeoTIFF**: High-resolution nadir orthomosaic.
  - **KML & GeoJSON**: Flight trajectory for Google Earth / GIS validation.
  - **Survey Audit Report**: Verifiable HTML and JSON reports detailing exact RMSE, registered cameras, and GNSS status.

### Component 7: Dual-Tier Deployment Architecture (Local Edge & Cloud / Kaggle Accelerator)
- **Local Edge Tier (`configs/default.yaml` / `lowvram.yaml`)**:
  - 100% offline, air-gapped field deployment running on local hardware (e.g. laptop RTX 3050 6GB GPU or CPU fallback).
  - Fast turnaround for tactical field verification (~1–15 min).
- **Accelerated Cloud / Kaggle Tier (`configs/accelerated.yaml`)**:
  - High-throughput environment leveraging multi-GPU cloud environments (Kaggle dual T4, Google Colab, A100 clusters).
  - Enables native 4K processing (`3840×2160`), large foundation models (Depth Anything V2 Large, Metric3D ViT-Large, SAM-2), ultra-dense MVS point clouds, 16K texture atlases, and 3D Gaussian Splatting (`splatfacto`).
  - Seamless data bundling: Run on cloud, export bundles, and visualize in the DroneMap industrial viewer.

