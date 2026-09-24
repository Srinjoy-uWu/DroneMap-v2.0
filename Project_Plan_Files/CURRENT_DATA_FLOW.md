# Current Data Flow & Schema Specification — DroneMap (SIH26158)

This document tracks the precise physical and logical flow of data through the DroneMap pipeline as implemented in the current repository.

---

## 1. End-to-End Data Flow Pipeline

```text
[Input Video (.mp4/.mov)] + [Optional Telemetry (.srt/.csv)]
                           │
                           ▼
                  [stage1_frames.py]
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 01_frames/images/*.jpg              01_frames/keyframes.json
 (Selected sharp keyframes)          01_frames/telemetry.json
        │
        ▼
 [stage2_masks.py]
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
 02_masks/masks/*.png                Updated keyframes.json
 (255=dynamic, 0=static)             (Dropped frames excluded)
        │
        ▼
 [stage3_pose.py (COLMAP)]
        │
        ├── 03_pose/colmap.db (SIFT features & matches)
        ├── 03_pose/sparse/0/ {cameras, images, points3D}.{bin,txt}
        ▼
 03_pose/undistorted/
 (Pinhole-rectified frames & sparse cloud)
        │
        ▼
 [stage3_georef.py]
        │
        ├── 03b_georef/sparse_enu/ {cameras, images, points3D}.bin
        ├── 03b_georef/sparse_enu/ref_images.txt (GPS constraints)
        ▼
 03b_georef/sparse_enu/transform.json (Scale, R, t, CRS)
        │
        ▼
 [dronemap.quality (Geometric Gate)]
        │
        ├── Triangulation angle, parallax, baseline/depth, planarity
        ▼
    ┌───────────────────────────────────┐
    │ Verdict: ACCEPT_3D / TERRAIN_2_5D │
    └─┬───────────────────────────────┬─┘
      │ (ACCEPT_3D)                   │ (TERRAIN_2_5D or OpenMVS Fallback)
      ▼                               ▼
 [stage5_dense.py]               [terrain.py Engine]
      │                               │
      ├── 05_dense/scene.mvs          ├── Lower-envelope PCA ground fit
      ├── 05_dense/scene_dense.mvs    ├── Orthonormal rotation to +Z
      ▼                               ├── IDW elevation grid (250x250)
 05_dense/scene_dense.ply             ▼
 (Densified MVS point cloud)     06_mesh/scene_dense_mesh_terrain_texture.obj
      │                               │
      ▼                               │
 [stage6_mesh.py]                     │
      │                               │
      ├── 06_mesh/scene_dense_mesh.ply│
      ├── 06_mesh/scene_dense_mesh_clean.ply
      ├── 06_mesh/scene_dense_mesh_refine.ply
      ▼                               │
 06_mesh/*_texture.obj, *.mtl, *.jpg  │
      │                               │
      └───────────────┬───────────────┘
                      │
                      ▼
             [stage7_export.py]
                      │
                      ├── 07_export/model.glb (glTF 2.0 Y-up)
                      ├── 07_export/cloud.laz (ASPRS 1.4 UTM)
                      ├── 07_export/dsm.tif (Digital Surface Model GeoTIFF)
                      ├── 07_export/dtm.tif (Digital Terrain Model GeoTIFF)
                      ├── 07_export/orthomosaic.tif (Nadir Ortho GeoTIFF)
                      ├── 07_export/trajectory.{kml,json}
                      ├── 07_export/accuracy_report.json
                      ▼
             07_export/report.html (Self-contained audit)
```

---

## 2. Directory Layout for a Complete Run

Each execution is encapsulated in an isolated workspace under `data/runs/<run_id>/`:

```text
data/runs/<run_id>/
├── manifest.json                  # Canonical source of truth for the entire run
├── 01_frames/
│   ├── images/
│   │   ├── frame_000000.jpg       # Selected keyframes (JPEG, max_long_edge: 1920)
│   │   ├── frame_000030.jpg
│   │   └── ...
│   ├── keyframes.json             # Frame metadata (timestamp, sharpness, GPS)
│   └── telemetry.json             # Parsed GPS fixes from sidecar
├── 02_masks/
│   └── masks/
│       ├── frame_000000.png       # 8-bit single-channel binary mask (255=masked)
│       └── ...
├── 02b_depth/                     # Optional depth maps (if stage depth enabled)
│   ├── raw/
│   │   └── frame_000000_depth.png # 16-bit uint16 depth in millimetres
│   └── depth_index.json
├── 03_pose/
│   ├── colmap.db                  # SQLite database of keypoints and matches
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin        # Camera intrinsics (SIMPLE_RADIAL / PINHOLE)
│   │       ├── images.bin         # Registered poses (QW, QX, QY, QZ, TX, TY, TZ)
│   │       ├── points3D.bin       # Triangulated 3D points + track observations
│   │       └── txt/
│   │           ├── cameras.txt    # Human-readable export
│   │           ├── images.txt
│   │           └── points3D.txt
│   └── undistorted/
│       ├── images/                # Undistorted pinhole camera frames
│       └── sparse/                # Undistorted sparse geometry
├── 03b_georef/
│   └── sparse_enu/
│       ├── cameras.bin
│       ├── images.bin
│       ├── points3D.bin
│       ├── ref_images.txt         # GPS coordinates passed to COLMAP aligner
│       └── transform.json         # Coordinate transformation definition
├── 04_semantics/                  # Optional semantic label maps (SegFormer)
│   ├── labels/
│   ├── class_fractions.json
│   └── id2label.json
├── 05_dense/
│   ├── scene.mvs                  # OpenMVS project definition
│   ├── scene_dense.mvs            # OpenMVS dense project with depth maps
│   └── scene_dense.ply            # Coloured dense point cloud (binary PLY)
├── 06_mesh/
│   ├── scene_dense_mesh.ply       # Raw mesh from Delaunay / Poisson
│   ├── scene_dense_mesh_clean.ply # Largest coherent surface component
│   ├── scene_dense_mesh_refine.ply# Smoothed/refined surface
│   ├── scene_dense_mesh_refine_texture.obj # Wavefront OBJ
│   ├── scene_dense_mesh_refine_texture.mtl # Material file
│   └── scene_dense_mesh_refine_texture_material_00_map_Kd.jpg # Texture atlas
├── 07_export/
│   ├── model.glb                  # Binary glTF 2.0 (Three.js viewer deliverable)
│   ├── cloud.laz                  # ASPRS LAS 1.4 compressed point cloud
│   ├── dsm.tif                    # Digital Surface Model (GeoTIFF, metres)
│   ├── dtm.tif                    # Digital Terrain Model (GeoTIFF, bare-earth)
│   ├── orthomosaic.tif            # Nadir orthomosaic (RGB GeoTIFF)
│   ├── trajectory.kml             # Google Earth trajectory
│   ├── trajectory.json            # GeoJSON camera positions
│   ├── accuracy_report.json       # Machine-readable survey verification
│   └── report.html                # Standalone interactive audit report
└── logs/                          # Complete execution logs
    ├── colmap_feature_extractor.log
    ├── colmap_sequential_matcher.log
    ├── colmap_mapper.log
    ├── colmap_model_aligner.log
    ├── openmvs_interface.log
    ├── openmvs_densify.log
    ├── openmvs_reconstruct.log
    ├── openmvs_refine.log
    └── openmvs_texture.log
```

---

## 3. Data Schemas & Formats

### `keyframes.json` Schema
Generated by Stage 1 (`stage1_frames.py`), records per-frame analysis:
```json
[
  {
    "frame_idx": 30,
    "timestamp_s": 1.0,
    "path": "C:\\Users\\...\\01_frames\\images\\frame_000030.jpg",
    "sharpness": 1645.68,
    "selected": true,
    "lat": 28.58343,
    "lon": 77.21668,
    "alt_m": 276.5,
    "speed_mps": 4.8,
    "gimbal_pitch_deg": -90.0
  }
]
```

### `transform.json` Schema
Generated by Stage 3b (`stage3_georef.py`), records geodesy alignment:
```json
{
  "coordinate_frame": "ECEF",
  "crs": "EPSG:32643",
  "output_crs": "EPSG:32643",
  "georef_success": true,
  "scale": 1.042,
  "alignment_rmse_m": 3.9079,
  "datum": {
    "lat": 28.58343,
    "lon": 77.21668,
    "alt_m": 276.5
  },
  "model_offset_m": [1240257.0, 5466384.0, 3033573.0],
  "rotation_to_viewer": [
    [-0.9752, 0.2213, 0.0],
    [0.1943, 0.8564, 0.4784],
    [0.1059, 0.4666, -0.8781]
  ]
}
```

### `manifest.json` Top-Level Schema
Maintained by `dronemap.workspace.RunWorkspace` with atomic disk writes:
```json
{
  "schema_version": 1,
  "run_id": "test_orbit_georef",
  "created_utc": "2026-09-12T10:20:42+00:00",
  "host": {
    "platform": "Windows-10-10.0.26200-SP0",
    "python": "3.11.16"
  },
  "config": { ... },
  "input": {
    "video": "C:\\Users\\...\\video.mp4",
    "telemetry": "C:\\Users\\...\\telemetry.srt"
  },
  "tools": {
    "colmap": "4.1.1",
    "colmap_cuda": "True",
    "openmvs": "2.3.0",
    "ffmpeg": "7.1"
  },
  "gnss": {
    "mode": "STANDALONE",
    "source": "srt"
  },
  "stages": {
    "frames": { "status": "ok", "duration_s": 50.8, "metrics": { ... } },
    "masks": { "status": "ok", "duration_s": 28.4, "metrics": { ... } },
    "pose": { "status": "ok", "duration_s": 1723.3, "metrics": { ... } },
    "georef": { "status": "ok", "duration_s": 12.1, "metrics": { ... } },
    "dense": { "status": "ok", "duration_s": 1213.9, "metrics": { ... } },
    "mesh": { "status": "ok", "duration_s": 998.0, "metrics": { ... } },
    "export": { "status": "ok", "duration_s": 16.5, "metrics": { ... } }
  },
  "accuracy": {
    "run_id": "test_orbit_georef",
    "gnss_mode": "STANDALONE",
    "crs": "EPSG:32643",
    "alignment_rmse_m": 3.9079,
    "registered_fraction": 1.0,
    "n_sparse_points": 23280,
    "n_dense_points": 668657,
    "coordinate_mode": "georeferenced"
  }
}
```

---

## 4. Coordinate Transformation Lifecycle

| Stage | Data Representation | Coordinate Space | Units |
|---|---|---|---|
| **Stage 1 (Frames)** | Keyframe video timestamps | 1D time domain ($t$ seconds) | Seconds |
| **Stage 2 (Masks)** | 2D image coordinates | Pixel space $(u, v)$ | Pixels |
| **Stage 3a (SfM)** | Reconstructed sparse cameras & points | Arbitrary local COLMAP coordinate system | Arbitrary scale |
| **Stage 3b (Georef)** | Sim(3) aligned model | WGS-84 / ECEF ($X, Y, Z$) $\rightarrow$ Local ENU | Metres |
| **Stage 5 (Dense)** | OpenMVS multi-view dense points | Centred ECEF / ENU ($X - X_0, Y - Y_0, Z - Z_0$) | Metres |
| **Stage 6 (Mesh)** | Delaunay / TIN mesh vertices | Centred ECEF / ENU | Metres |
| **Stage 7 (Export GLB)** | 3D WebGL mesh | Orthonormally rotated to glTF $Y$-up $(x=\text{East}, y=\text{Up}, z=-\text{North})$ | Metres |
| **Stage 7 (Export LAZ / GIS)**| Point cloud & GeoTIFF rasters | Projected CRS (e.g. UTM Zone 43N, `EPSG:32643`) | Metres (Easting, Northing, Orthometric Height) |
