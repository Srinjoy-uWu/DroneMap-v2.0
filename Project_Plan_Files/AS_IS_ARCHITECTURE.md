# As-Is Architecture — DroneMap (SIH26158)

## 1. Executive Summary

DroneMap is a production-grade, offline-capable photogrammetry pipeline built to solve Smart India Hackathon problem statement **SIH26158**: *Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model*.

The current codebase is implemented in Python 3.11 with typed configurations (`pydantic`), an explicit directed stage graph (`dronemap.pipeline`), standalone compiled vision binaries (COLMAP 4.1.1 with CUDA, OpenMVS 2.3+ x64, imageio-ffmpeg 7.1, Blender 5.1), and an air-gapped web studio backed by FastAPI and Three.js.

```
                         [ Single-Pass UAV Video ]
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Stage 1: Frames Extraction  │  (imageio-ffmpeg / OpenCV)
                      │  - 2-pass streaming O(1) RAM │
                      │  - Laplacian sharpness window│
                      │  - Optical flow & turn stride│
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Stage 2: Dynamic Masking    │  (YOLOv8s-seg CUDA)
                      │  - Moving vehicles & humans  │
                      │  - Dilated binary PNG masks  │
                      └──────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │ Stage 4a: Depth (Optional)│   │ Stage 4b: Semantics (Opt) │
       │ - Depth Anything V2       │   │ - SegFormer               │
       │ - Metric3D v2             │   │ - UAVid / LoveDA          │
       └─────────────┬─────────────┘   └─────────────┬─────────────┘
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Stage 3a: Camera Pose (SfM) │  (COLMAP / MASt3R)
                      │  - GPU SIFT + Mask exclusion │
                      │  - Sequential + Loop Match   │
                      │  - Bundle Adjustment Mapper  │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Stage 3b: Georeferencing    │  (COLMAP Aligner/Prior)
                      │  - DJI SRT / CSV parsing     │
                      │  - WGS-84 -> ECEF -> ENU     │
                      │  - UTM zone auto-projection  │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Quality Gating & Routing    │  (dronemap.quality)
                      │  - Triangulation & baseline  │
                      │  - Planarity & relief ratio  │
                      │  - ACCEPT_3D / TERRAIN / REJ │
                      └──────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       [ Route A: Volumetric 3D ]      [ Route B: 2.5D Terrain Surface ]
       ┌──────────────────────────┐    ┌───────────────────────────────┐
       │ Stage 5: Dense Cloud     │    │ dronemap.terrain Engine       │
       │ - OpenMVS Densify        │    │ - Robust ground plane PCA/fit │
       └────────────┬─────────────┘    │ - IDW KDTree elevation grid   │
                    │                  │ - Watertight TIN mesh + UV    │
       ┌────────────┴─────────────┐    └───────────────┬───────────────┘
       │ Stage 6: Mesh & Texture  │                    │
       │ - OpenMVS Reconstruct    │                    │
       │ - OpenMVS RefineMesh     │                    │
       │ - OpenMVS TextureMesh    │                    │
       │ - Atlas black check/retry│                    │
       └────────────┬─────────────┘                    │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Stage 7: Multi-Format Export│  (dronemap.stage7_export)
                      │  - Textured 3D Mesh (GLB)    │
                      │  - Coloured Point Cloud (LAZ)│
                      │  - GIS Rasters (DSM, DTM,    │
                      │    Orthomosaic GeoTIFFs)     │
                      │  - Trajectory (KML / JSON)   │
                      │  - Accuracy & Audit Report   │
                      └──────────────────────────────┘
```

---

## 2. Component-by-Component Specification

### Stage 1: Frame Extraction & Preprocessing (`stage1_frames.py`)
- **Module / Class / Function**: `dronemap.stage1_frames.run()`
- **Libraries & Tools**: `cv2` (OpenCV), `imageio-ffmpeg` (bundled static binary), `numpy`
- **Inputs**: Raw video file (`.mp4`, `.mov`, `.mkv`, etc.), optional telemetry (`.srt`, `.csv`).
- **Outputs**:
  - Keyframe images: `ws.images_dir/*.jpg` (`frame_NNNNNN.jpg`)
  - Keyframe index: `ws.frames_index` (`keyframes.json`)
  - Synchronized telemetry: `ws.telemetry_json` (`telemetry.json`)
- **Core Algorithms**:
  - **O(1) Memory Streaming 2-Pass Architecture**: Pass 1 probes video stream frame-by-frame without loading into RAM; computes Laplacian variance $\sigma^2(\nabla^2 I)$ and samples Lucas-Kanade optical flow. Pass 2 fast-forwards to target keyframe indices with `cv2.VideoCapture.grab()` and writes downscaled JPEGs directly to disk.
  - **Windowed Sharpness Filtering**: In sliding windows of `sharpness_window` frames, only the frame with peak sharpness is retained; blurry clusters are automatically rejected.
  - **Global Sharpness Cutoff**: Rejects frames whose sharpness drops below `min_sharpness_ratio` (default: 35%) of clip median.
  - **Baseline & Turn-Rate Stride Estimation**: Computes forward flight stride via GPS speed $v$ and footprint $0.8 \cdot h \cdot (1 - \text{overlap})$, or optical flow magnitude when GPS is absent. In curved/orbit trajectories, platform heading sweep $\theta$ is recovered from track chord ratios to prevent visual degradation.

### Stage 2: Dynamic Object Masking (`stage2_masks.py`)
- **Module / Class / Function**: `dronemap.stage2_masks.run()`
- **Libraries & Tools**: `ultralytics` (YOLOv8s-seg), `torch` (CUDA 12.8), `cv2`, `numpy`
- **Inputs**: Selected keyframes from Stage 1 (`ws.images_dir/*.jpg`).
- **Outputs**: Binary PNG masks in `ws.masks_dir/<stem>.png` (255 = dynamic/masked, 0 = valid static terrain).
- **Core Algorithms**:
  - Evaluates `yolov8s-seg.pt` on each keyframe.
  - Filters instance masks for dynamic categories: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `train`, `truck`, `boat`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`.
  - Applies morphological dilation (`dilate_px`, default: 12 px) using an elliptical structuring element to cover motion blur halos.
  - If a frame's masked area exceeds `max_masked_fraction` (default: 60%), the frame is dropped from `ws.frames_index`.
  - Passes masks directly to COLMAP `--ImageReader.mask_path` without mutating original images.

### Stage 3a: Camera Pose Estimation / SfM (`stage3_pose.py` & `stage3_pose_mast3r.py`)
- **Module / Class / Function**: `dronemap.stage3_pose.run()`, fallback to `stage3_pose_mast3r.run()`
- **Libraries & Tools**: COLMAP 4.1.1 (CUDA SIFT), SQLite3 (`colmap.db`), PyTorch (for MASt3R stretch)
- **Inputs**: Keyframes (`ws.images_dir`), dynamic masks (`ws.masks_dir`).
- **Outputs**:
  - COLMAP binary sparse reconstruction: `ws.sparse_dir/0/{cameras,images,points3D}.bin`
  - COLMAP text model: `ws.sparse_dir/0/txt/{cameras,images,points3D}.txt`
  - Undistorted pinhole camera frames: `ws.undistorted_dir/`
- **Core Algorithms**:
  - **Feature Extraction**: GPU SIFT extraction up to `max_num_features` (16,384 points/frame). Dynamic masks exclude transient objects before feature keypoints enter `colmap.db`.
  - **Sequential Matching**: Exploits continuous drone video temporal ordering (`sequential_overlap=10`), with quadratic stride and periodic loop closure detection (`loop_detection_period=10`).
  - **Escalation Ladder**: If registered frame fraction falls below `min_registered_fraction` (55%), automatically escalates to exhaustive matching and FAISS-based vocabulary tree retrieval matching.
  - **Process Failure Diagnostics**: Differentiates OS-level memory termination / crashes (exit codes $0xC0000005$, $0xC0000409$) from true photogrammetric failure to prevent misdirected hyperparameter advice.
  - **Undistortion**: Invokes `colmap image_undistorter` to prepare normalized pinhole projections for OpenMVS.

### Stage 3b: Metric Scale & Georeferencing (`stage3_georef.py`)
- **Module / Class / Function**: `dronemap.stage3_georef.run()`
- **Libraries & Tools**: COLMAP `pose_prior_mapper` / `model_aligner`, `pyproj`, `numpy`
- **Inputs**: Up-to-scale sparse model (`ws.sparse_dir/0/`), drone GPS fixes (`ws.telemetry_json`).
- **Outputs**:
  - Georeferenced sparse model: `ws.georef_sparse_dir/`
  - Control points file: `ws.georef_sparse_dir/ref_images.txt`
  - Coordinate transform metadata: `ws.georef_sparse_dir/transform.json`
- **Core Algorithms**:
  - Converts drone WGS-84 coordinates $(\phi, \lambda, h)$ to Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.
  - Estimates rigid Sim(3) similarity transformation $[s, R, t]$ via Umeyama alignment or non-linear pose prior bundle adjustment.
  - Re-anchors coordinate origin to the scene centroid (`model_offset_m`) to preserve single-precision floating-point precision in downstream OpenMVS shaders.
  - Projective CRS automatically resolved to local UTM zone (e.g., `EPSG:32643`).
  - Strict Quality Rule: If GPS alignment fails or telemetry is missing, flags status as `LOCAL_RELATIVE` and never assigns a false UTM datum.

### Quality Gate & Capture Diagnostics (`quality.py`)
- **Module / Class / Function**: `dronemap.quality.assess_pose()`, `assess_structure()`
- **Libraries & Tools**: `numpy`, `scipy.spatial`
- **Inputs**: Sparse geometry (camera poses, point tracks) and dense cloud geometry.
- **Outputs**: `Verdict` (`ACCEPT_3D`, `TERRAIN_2_5D`, `REJECT`), reason codes, warnings, and advice.
- **Metrics Evaluated**:
  - **Median Triangulation Angle**: Measured across all 3D points. Rejects if $< 2^\circ$; requires $> 5^\circ$ for volumetric 3D.
  - **Baseline-to-Depth Ratio**: $\frac{\text{baseline}}{\text{median depth}}$. Values $< 0.1$ indicate zero parallax (hover or straight line without angle change).
  - **Depth Dynamic Range**: $\frac{p_{95}(\text{depth})}{p_{5}(\text{depth})}$. Dynamic range $< 1.8\times$ indicates a single planar ground sheet without vertical building facades.
  - **Forward Motion Ratio**: Displacement along the optical axis. Values $> 0.85$ indicate degenerate zoom/forward flight where epipolar geometry collapses.
  - **Planarity**: Ratio of the smallest PCA eigenvalue $\lambda_3 / \lambda_1$. Below $0.06$, volumetric MVS produces collapsed sheets or spikes.

### Stage 5: Dense Reconstruction (`stage5_dense.py`)
- **Module / Class / Function**: `dronemap.stage5_dense.run()`
- **Libraries & Tools**: OpenMVS 2.3+ (`InterfaceCOLMAP`, `DensifyPointCloud`)
- **Inputs**: Undistorted images and camera poses (`ws.undistorted_dir`).
- **Outputs**: Dense colored point cloud (`ws.dense_dir/scene_dense.ply`), OpenMVS scene project (`scene_dense.mvs`).
- **Core Algorithms**:
  - Converts COLMAP cameras and image files to OpenMVS binary scene descriptor.
  - Computes patch-match multi-view stereo depth maps across adjacent camera clusters (`number_views=6`, `resolution_level=1` or `2`).
  - Fuses depth maps with minimum view consistency threshold (`number_views_fuse=3` or `2` for short sequences).
  - Estimates normal vectors and RGB vertex colors for every densified point.

### Stage 6: Mesh Reconstruction & Texturing (`stage6_mesh.py`)
- **Module / Class / Function**: `dronemap.stage6_mesh.run()`
- **Libraries & Tools**: OpenMVS (`ReconstructMesh`, `RefineMesh`, `TextureMesh`), `PIL`, `trimesh`
- **Inputs**: Dense point cloud (`ws.dense_dir/scene_dense.ply`, `scene_dense.mvs`).
- **Outputs**:
  - Cleaned watertight 3D mesh: `ws.mesh_dir/scene_dense_mesh_clean.ply`
  - Textured OBJ model: `ws.mesh_dir/*_texture.obj`, `*.mtl`, `*.jpg`
- **Core Algorithms**:
  - **Poisson-style Surface Extraction**: Invokes `ReconstructMesh` to build initial Delaunay surface.
  - **Connected Component Cleaning**: Removes disconnected floating fragments below `min_component_faces` (200 faces).
  - **Iterative Refinement**: Optional high-detail surface smoothing with `RefineMesh`.
  - **Photorealistic Texturing**: Projects source keyframes onto mesh geometry via `TextureMesh` at up to 8192×8192 atlas resolution.
  - **Atlas Black-Fraction Verification**: Directly inspects texels referenced by the mesh UVs (`vt`). If Poisson seam leveling collapses interior patch color ($> 20\%$ black texels), automatically re-textures with seam-leveling disabled to eliminate dark models.
  - **Automatic 2.5D Fallback**: If OpenMVS fails or capture quality verdict is `terrain_2_5d`, automatically falls back to `terrain.py`.

### Robust 2.5D Terrain Mesh Engine (`terrain.py`)
- **Module / Class / Function**: `dronemap.terrain.reconstruct_terrain_mesh()`
- **Libraries & Tools**: `numpy`, `scipy.spatial.cKDTree`, `trimesh`, `PIL`
- **Inputs**: Dense or sparse point cloud with vertex colors, optional camera optical axis / geodetic vertical prior.
- **Outputs**: Textured OBJ + MTL + atlas and PLY representing a clean, watertight Digital Surface Model TIN mesh.
- **Core Algorithms**:
  - **Lower-Envelope Ground Plane Fitting**: Fits ground plane to points between the 2nd and 30th elevation percentiles (true ground), avoiding building rooftops or vegetation canopy. Rejects fits tilted $> 15^\circ$ from vertical prior.
  - **Canonical Horizontal Alignment**: Orthonormally rotates cloud so ground normal aligns exactly with $+Z$, preserving exact metric scale.
  - **Vectorized IDW Surface Interpolation**: Computes high-density elevation grid ($250 \times 250$ up to $400 \times 400$) using KD-Tree Inverse Distance Weighting ($k=8$, $p=2$).
  - **Watertight Triangulated Irregular Network (TIN)**: Converts 2D elevation grid into dual-triangle cell topology with UV coordinates and baked composite texture atlas.

### Stage 7: Multi-Format Deliverables & Accuracy Audit (`stage7_export.py`)
- **Module / Class / Function**: `dronemap.stage7_export.run()`
- **Libraries & Tools**: `trimesh`, `pygltflib`, `laspy`, `rasterio`, `pyproj`, `numpy`
- **Inputs**: Reconstructed mesh (`06_mesh/`), dense cloud (`05_dense/`), coordinate transform (`03b_georef/transform.json`), accuracy manifest.
- **Outputs**:
  - **GLB**: Compact, viewer-ready glTF binary (`model.glb`) rotated into glTF standard $Y$-up orientation.
  - **LAZ**: ASPRS LAS 1.4 compressed point cloud (`cloud.laz`) tagged with projected UTM CRS.
  - **DSM**: Digital Surface Model GeoTIFF (`dsm.tif`) at native Ground Sampling Distance (GSD).
  - **DTM**: Digital Terrain Model GeoTIFF (`dtm.tif`) derived via morphological ground filtering.
  - **Orthomosaic**: High-resolution nadir orthomosaic GeoTIFF (`orthomosaic.tif`).
  - **Trajectory**: Google Earth KML (`trajectory.kml`) and GeoJSON (`trajectory.json`).
  - **Accuracy & Audit Reports**: Machine-readable JSON (`accuracy_report.json`) and styled self-contained HTML audit (`report.html`).

### Web Studio & Interactive 3D Viewport (`api/server.py` & `api/static/`)
- **Backend**: FastAPI running on `uvicorn`, background threading worker for asynchronous video processing, REST API (`/api/runs`, `/api/jobs/{id}`, `/api/runs/{id}/model.glb`, `/api/runs/{id}/measure`).
- **Frontend**: Air-gapped single-page application with vendored Three.js r168, OrbitControls, and GLTFLoader.
- **Key Features**:
  - Live progress terminal drawer streaming background photogrammetry stdout/stderr.
  - Recency-sorted sidebar projects list showing truthful badges: `Validated 3D`, `Relative/local`, `Terrain-only (2.5D)`, `Capture rejected`.
  - Interactive 3D measurement ruler calculating real-world Euclidean distances, horizontal distance, elevation delta, and compass bearings (metric for georeferenced runs, relative units for unaligned runs).
  - Shading switcher: Full textured photo-atlas or analytical white clay surface with computed vertex normals.

---

## 3. Dependency Inventory

The application dependencies are pinned in `pyproject.toml` and locked in `uv.lock`:

| Category | Dependency | Exact Version / Constraint | Purpose |
|---|---|---|---|
| **Python** | Python | `3.11.16` | Required base runtime (pinned for pycolmap/Open3D wheels) |
| **Numerics & Imaging** | `numpy` | `~2.2.6` | Vectorized matrix calculations, coordinate geodesy |
| | `scipy` | `>=1.11` | KDTree spatial indexing, morphological surface opening |
| | `opencv-python` | `>=4.10` | Video decoding, optical flow, CLAHE, Laplacian sharpness |
| | `pillow` | `>=10.0` | Texture atlas rendering and UV sampling verification |
| | `matplotlib` | `>=3.8` | Diagnostic plotting and curve evaluation |
| | `imageio-ffmpeg` | `>=0.5` | Bundled static FFmpeg binary |
| **CLI & Framework** | `typer` | `>=0.12` | Rich CLI application command dispatch |
| | `rich` | `>=13.0` | Terminal tables, progress rules, diagnostic cards |
| | `pydantic` | `>=2.7` | Typed settings and manifest schema serialization |
| | `pydantic-settings` | `>=2.3` | Hierarchical configuration overrides |
| | `pyyaml` | `>=6.0` | Profile parsing (`default.yaml`, `lowvram.yaml`, etc.) |
| **Geospatial** | `pyproj` | `>=3.6` | WGS-84, ECEF, UTM, and EPSG coordinate transformations |
| | `pymap3d` | `>=3.1` | Geodetic coordinate transforms |
| | `rasterio` | `>=1.3` | GeoTIFF reading, writing, and affine projection |
| | `laspy[lazrs]` | `>=2.5` | Compressed LAZ 1.4 point cloud writing |
| **3D & Meshing** | `trimesh` | `>=4.4` | OBJ/PLY parsing, mesh cleanup, normal generation |
| | `pygltflib` | `>=1.16` | glTF 2.0 / GLB file assembly |
| | `open3d` | `>=0.18` | Point cloud geometry analysis and PLY color handling |
| **Web Studio** | `fastapi` | `>=0.111` | REST API and static file serving |
| | `uvicorn[standard]`| `>=0.30` | Asynchronous ASGI HTTP server |
| | `python-multipart` | `>=0.0.9` | Video and telemetry file upload handling |
| **AI / ML (Optional)**| `torch` / `torchvision` | `2.11.0+cu128` | Deep learning execution with CUDA 12.8 acceleration |
| | `ultralytics` | `>=8.3` | YOLOv8s-seg dynamic object instance segmentation |
| | `transformers` | `>=4.44` | Hugging Face Depth Anything V2 & SegFormer backends |
| **Tools** | COLMAP | `4.1.1` (CUDA) | GPU SIFT, feature matching, bundle adjustment |
| | OpenMVS | `2.3.0` (x64) | Multi-view dense stereo, mesh reconstruction, texturing |
| | Blender | `5.1` | Synthetic ground-truth validation fixture rendering |
