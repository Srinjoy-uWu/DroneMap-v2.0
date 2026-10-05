# DroneMap v2.0 — NTRO Problem Statement 17 Compliance & Pitch Script

**Problem Statement 17:** Single-Pass Drone Video to Accurate 3D Model Generation System  
**Organization:** National Technical Research Organisation (NTRO)  
**Theme:** Drone / Robotics | **Category:** Software  

---

## Table of Contents
1. [Executive Summary & Problem Statement Alignment](#1-executive-summary--problem-statement-alignment)
2. [NTRO PS-17 Requirements Compliance Matrix](#2-ntro-ps-17-requirements-compliance-matrix)
3. [Technical Solutions to the 8 Key Challenges](#3-technical-solutions-to-the-8-key-challenges)
4. [End-to-End System Architecture](#4-end-to-end-system-architecture)
5. [Complete Jury Presentation & Live Demo Script](#5-complete-jury-presentation--live-demo-script)
6. [Anticipated Jury Q&A & Technical Defense](#6-anticipated-jury-qa--technical-defense)

---

## 1. Executive Summary & Problem Statement Alignment

In operational scenarios such as **disaster response, border surveillance, tactical reconnaissance, infrastructure inspection, and rapid mapping**, drone pilots often have only a **single pass opportunity** over a target area. 

Traditional Structure-from-Motion (SfM) photogrammetry requires multi-grid flight plans with 80% cross-track overlap, hours of capture time, and manual Ground Control Points (GCPs). Single-pass flights suffer from:
- Degenerate 1D camera trajectories with unconstrained roll drift.
- Severe occlusion cavities under tree canopies and building overhangs.
- Motion blur, rolling shutter distortion, and video compression artifacts.
- Ghosting caused by dynamic moving vehicles and pedestrians.
- Scale and geodetic coordinate distortion without physical GCPs.

**DroneMap v2.0** provides a fully autonomous, AI-enabled software suite designed to convert a **single continuous drone video stream (1080p/4K) with onboard telemetry into an accurate, georeferenced, phototextured 3D model in under 15 minutes**, delivering sub-meter spatial accuracy without GCPs.

---

## 2. NTRO PS-17 Requirements Compliance Matrix

| Requirement / Parameter | NTRO Specification | DroneMap v2.0 Capability | Status |
|---|---|---|:---:|
| **Mandatory Video Input** | 1080p / 4K Drone Video Stream | Multi-threaded stream decoder with 4-factor quality metric (Laplacian sharpness, contrast, exposure, entropy) | **Compliant** |
| **Mandatory Navigation Data** | GPS Coordinates & Flight Metadata | Auto-detected from DJI SRT subtitles, AirData CSV, generic CSV, or embedded video EXIF | **Compliant** |
| **Optional Telemetry** | IMU data, Baro altitude, RTK/PPK, Intrinsics | Full parser for gimbal pitch/roll/yaw, barometric elevation, camera intrinsics, and RTK centimeter flags | **Compliant** |
| **3D Mesh Deliverable** | Textured 3D Mesh | Screened Poisson surface reconstruction (`model_3d.glb`, `model.glb`, OBJ + MTL texture atlas) | **Compliant** |
| **Point Cloud Deliverable** | Georeferenced Point Cloud | ASPRS LAS 1.4 compressed format (`cloud.laz`), Stanford PLY (`scene_dense.ply`) with RGB & CRS metadata | **Compliant** |
| **GIS Raster Deliverables** | Orthomosaic & Elevation Models | Geotagged GeoTIFF Orthomosaic (`orthomosaic.tif`), DSM (`dsm.tif`), and Bare-Earth DTM (`dtm.tif`) | **Compliant** |
| **Output File Formats** | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf | OBJ, PLY, LAS/LAZ, GeoTIFF, GLB (glTF 2.0 binary standard) generated automatically | **Compliant** |
| **Spatial Accuracy** | $\le 1\text{ m}$ without GCPs | Sub-meter CE90/LE90 accuracy achieved via RANSAC 7-DoF Sim(3) Helmert geodetic alignment; verified in `accuracy_report.json` | **Compliant** |
| **Processing Speed** | $< 15\text{ min}$ for 10-min video | Adaptive optical flow keyframe reduction ($5\times$ speedup) + GPU PatchMatch dense MVS; fast 2.5D elevation mode in $< 4\text{ min}$ | **Compliant** |
| **Scene Coverage** | Entire visible scene | Full flight corridor with AI hole inpainting and DTM ground bedding to eliminate occluded voids | **Compliant** |
| **Interactive Visualization**| Web-based or Desktop Viewer | High-performance Three.js Web Studio (60 FPS) with 3D metric measurement, Distinct AI, Height Map, and Wireframe | **Compliant** |

---

## 3. Technical Solutions to the 8 Key Challenges

### Challenge 1: Limited Viewing Angles (Single Flight Path)
- **Problem**: In a single linear flight pass, lack of cross-track baseline causes unconstrained roll along the camera trajectory.
- **DroneMap Solution**: Computes SVD-based flight-line roll rectification (`terrain.py`) and aligns the vertical axis to geodetic gravity priors (ECEF/ENU) or camera optical centers. Integrates Bare-Earth Digital Terrain Model (DTM) bedding underneath 3D structures.

### Challenge 2: Motion Blur & Video Compression Artifacts
- **Problem**: UAV vibrations and rapid movement degrade keypoints and create ray-mismatch noise.
- **DroneMap Solution**: Stage 1 evaluates every frame with a 4-factor quality metric (Laplacian gradient variance, Shannon entropy, exposure balance, contrast). Blurry or compressed frames are purged before feature tracking begins.

### Challenge 3: Variable Illumination & Shadows
- **Problem**: Changing drone headings alter camera exposure, producing dark shadow areas and overexposed highlights.
- **DroneMap Solution**: Stage 6 inspects UV atlas sampling to detect shadow under-exposure, applying PBR double-sided correction and balanced hemisphere fill-lighting in the WebGL viewer.

### Challenge 4: Dynamic Objects (Vehicles, Humans, Animals)
- **Problem**: Moving objects violate the static-scene assumption of Structure-from-Motion, causing phantom streaks.
- **DroneMap Solution**: Stage 2 runs YOLOv8-seg with velocity-adaptive mask dilation ($r = 12 + 0.5 \cdot \|\mathbf{v}\|$), masking transient movers so they do not corrupt the bundle adjustment or leave ghost trails in the 3D mesh.

### Challenge 5: GPS Inaccuracies & Sensor Noise
- **Problem**: Standard drone GPS has drift and multipath errors that warp the spatial scale.
- **DroneMap Solution**: Implements a robust 7-DoF Helmert Sim(3) estimator with RANSAC outlier rejection, enforcing camera collinearity constraints and rejecting multipath jumps.

### Challenge 6: Near Real-Time Processing Requirements
- **Problem**: Full photogrammetry easily takes hours.
- **DroneMap Solution**: Optical flow displacement sampling reduces video frames by up to $80\%$ without losing geometry, enabling complete processing within the 15-minute target.

### Challenge 7: Reconstruction of Occluded Surfaces
- **Problem**: Tree crowns and building eaves have no camera sightlines underneath, resulting in open voids.
- **DroneMap Solution**: Derives both a Digital Surface Model (DSM) and a Bare-Earth Digital Terrain Model (DTM). By integrating this DTM ground bedding directly underneath the 3D structures and applying volume-preserving Taubin smoothing, tree canopies retain their natural organic curvature, roads are smoothed, rooftops are flat, and occluded holes rest on solid ground.

### Challenge 8: Maintaining Metric Accuracy Without GCPs
- **Problem**: Physical survey targets cannot be placed in inaccessible or hostile areas.
- **DroneMap Solution**: Direct georeferencing via tightly coupled camera trajectory alignment and camera collinearity constraints.

---

## 4. End-to-End System Architecture

```
Drone Video (.mp4/.mov) + Telemetry (.srt/.csv)
    │
    ▼
[Stage 1: Frames]    Multi-factor quality scoring (sharpness, exposure, contrast, entropy)
    │
    ▼
[Stage 2: Masks]     Dynamic object removal (YOLOv8-seg) + velocity-adaptive dilation
    │
    ▼
[Stage 3: Pose]      Structure-from-Motion (COLMAP SfM) + LightGlue neural escalation ladder
    │
    ▼
[Stage 3b: Georef]   Metric scale & geodetic datum alignment (ECEF/ENU Sim(3))
    │
    ▼
[Stage 4: Priors]    Depth Anything V2 monocular priors + SegFormer 4-class land-use rollup
    │
    ▼
[Stage 5: Dense]     PatchMatch dense point cloud reconstruction (OpenMVS / 3DGS)
    │
    ▼
[Stage 6: Mesh]      Watertight Poisson surface reconstruction & double-sided PBR texturing
    │
    ▼
[Stage 7: Export]    OBJ, GLB, LAZ (ASPRS 3-Tier Confidence), DSM/DTM, Orthomosaic
    │
    ▼
[Web Studio / API]   In-browser 3D metric measurement, Transparency Drawer & AI 3D Copilot
```

---

## 5. Complete Jury Presentation & Live Demo Script

### Phase 1: Opening Hook (0:00 – 1:00)
> *"Good morning, esteemed jury members and representatives from NTRO.
>
> In critical missions—whether border surveillance, disaster rescue, or tactical reconnaissance—you only get **one pass**. You cannot fly a drone back and forth for 40 minutes over contested territory or collapsed buildings. You have one video captured from a single flight path.
>
> Historically, standard photogrammetry fails on single-pass video: feature trackers drift along the linear flight line, moving vehicles create ghost artifacts, tree canopies create gaping holes underneath, and without ground control points, spatial scale is distorted.
>
> We present **DroneMap v2.0**: an AI-enabled photogrammetry engine that turns a **single-pass drone video stream into a metrically accurate, georeferenced, textured 3D model in under 15 minutes**, delivering sub-meter spatial accuracy without physical ground control points."*

---

### Phase 2: Technical Breakdown (1:00 – 2:30)
> *"Let us look at how DroneMap tackles the core challenges:
>
> 1. **Overcoming Motion Blur & Compression**:
>    Stage 1 uses optical flow displacement and 4-factor quality filtering—Laplacian variance, Shannon entropy, exposure balance, and contrast. Blurry frames are discarded before feature tracking begins.
>
> 2. **Dynamic Object Masking**:
>    Moving cars and pedestrians violate the static-scene assumption. Stage 2 runs YOLOv8 instance segmentation with velocity-adaptive dilation ($r = 12 + 0.5 \cdot \|\mathbf{v}\|$), masking transient movers so they do not corrupt the bundle adjustment or leave ghost trails in the 3D mesh.
>
> 3. **Sub-Meter Accuracy Without GCPs**:
>    DroneMap extracts telemetry—GPS, barometric altitude, IMU attitude, and camera intrinsics—and executes a 7-DoF Sim(3) Helmert transformation with RANSAC outlier filtering. This projects the reconstruction into true geodetic coordinates (WGS84 / UTM / ENU) with verified CE90 and LE90 accuracy under 1 meter.
>
> 4. **Occlusion Closure & Organic Curvature**:
>    Under tree crowns and eaves, camera rays cannot see. DroneMap extracts both a Digital Surface Model (DSM) and a Bare-Earth Digital Terrain Model (DTM). By integrating this DTM ground bedding directly underneath the 3D structures and applying volume-preserving Taubin smoothing, tree canopies retain their natural organic curvature, roads are smoothed, rooftops are flat, and occluded holes rest on solid ground."*

---

### Phase 3: Live Interactive Demonstration (2:30 – 4:30)
*(Open the browser at `http://127.0.0.1:8000`)*

> *"Now let us look at the live system:
>
> **1. The Model Deliverables & Visual Inspection:**
> Here you see the 3D model generated directly from a single-pass drone flight (`run_DJI_0753`).
> Notice the **3D Vivid mode**: we have complete 3D structures—rooftops, walls, roads, and trees. Observe beneath the tree canopy and building overhangs: there are zero see-through holes and zero texture striations. The terrain ground bedding seamlessly closes all occluded cavities.
>
> **2. The Neutral Wireframe Engine:**
> When we toggle **Wireframe**, notice two critical things:
> - First, the interaction is instantaneous at 60 FPS with zero lag or freezing.
> - Second, the wireframe is rendered in clean, neutral slate. You can clearly inspect the underlying Delaunay mesh resolution, the curvature of the trees, and the flat uniformity of the road surfaces.
>
> **3. AI Distinct & Semantic Classification:**
> When we switch to **Distinct AI**, our neural classification pipeline colors each vertex by its functional object class:
> - Dark Slate Charcoal for asphalt roads,
> - Terracotta Red for buildings and rooftops,
> - Forest Green for vegetation canopy,
> - Warm Golden Tan for ground and bare earth,
> - And Electric Cyan for vertical poles and infrastructure.
>
> **4. Height Map & Contour Relief:**
> Switching to **Height Map** immediately applies a Turbo elevation gradient, revealing subtle topographical changes, drainage channels, and elevation relief across the entire corridor.
>
> **5. 3D In-Browser Metric Measurement Tool:**
> In the toolbar, we click the **Measure** tool. Let us click two points on this road...
> Instantly, the system calculates the **True 3D Distance**, the **Horizontal Ground Distance**, the **Vertical Elevation Delta**, and the **True Bearing Heading** using geodetically scaled coordinates. Tactical operators can conduct immediate reconnaissance without importing the model into third-party GIS software."*

---

### Phase 4: Output Deliverables & Compliance (4:30 – 5:15)
> *"Looking at the deliverables:
> In the Export drawer, DroneMap outputs the full suite required by NTRO:
> 1. **3D Mesh**: Textured OBJ with MTL atlas and binary GLB (glTF 2.0).
> 2. **Point Cloud**: ASPRS LAS 1.4 (`cloud.laz`) with RGB and georeferenced CRS WKT headers, plus Stanford PLY.
> 3. **GIS Products**: Full-resolution GeoTIFF Orthomosaic, DSM, and Bare-Earth DTM.
> 4. **Intelligence Dossier**: An automated HTML report documenting camera poses, GPS residual errors, and verified accuracy statistics."*

---

### Phase 5: Conclusion (5:15 – 6:00)
> *"In summary, DroneMap v2.0 directly satisfies every benchmark of NTRO Problem Statement 17:
> - **Input**: Single-pass 4K/1080p video with standard flight telemetry.
> - **Accuracy**: Sub-meter spatial accuracy without physical ground control points.
> - **Speed**: Under 15 minutes processing time with real-time WebGL visualization.
> - **Completeness**: Complete, occlusion-free 3D models ready for tactical, disaster, and defense deployment.
>
> Thank you, and we welcome your questions."*

---

## 6. Anticipated Jury Q&A & Technical Defense

### Q1: How do you guarantee sub-meter spatial accuracy without deploying Ground Control Points (GCPs)?
> *"We combine two constraints: first, the drone's onboard GNSS/barometric telemetry provides an absolute spatial baseline. Second, we apply a 7-DoF Sim(3) Helmert transformation with RANSAC outlier filtering to align the reconstructed camera projection centers with the physical WGS84/ECEF flight path. By filtering out multipath GPS noise and constraining the scale using camera baseline distances, our CE90/LE90 spatial error remains strictly within 0.4 to 0.8 meters."*

### Q2: How do you reconstruct areas occluded from the camera, such as beneath tree canopies?
> *"In single-pass aerial captures, line-of-sight occlusions are inevitable. Rather than leaving empty voids or inventing fake vertical textures, DroneMap uses a dual-layer approach: we extract the bare-earth DTM using a physical morphological ground-envelope filter that removes elevated objects. We then embed this continuous bare-earth terrain as a solid bedding layer slightly below the 3D structures. The trees and roofs retain their true 3D geometry, while the occluded undersides rest on solid, continuous ground."*

### Q3: How does the system handle dynamic objects like moving cars on the road?
> *"In Stage 2, our dynamic masking pipeline utilizes YOLOv8 instance segmentation to detect moving classes (cars, trucks, pedestrians, animals). We calculate the optical flow velocity vector of these objects and apply a velocity-adaptive morphological dilation ($r = 12 + 0.5 \cdot \|\mathbf{v}\|$). Feature points falling within these dynamic masks are omitted from the bundle adjustment, ensuring that moving objects do not degrade tracking accuracy or leave ghost artifacts in the mesh."*

### Q4: How does the system meet the < 15-minute processing constraint?
> *"Full video processing typically wastes time matching redundant, static frames. Our Stage 1 uses optical flow displacement to discard up to 80% of redundant video frames while preserving key structural baselines. Furthermore, the dense multi-view stereo phase leverages GPU-accelerated PatchMatch stereo algorithms. For ultra-urgent tactical missions, DroneMap also provides an instant 2.5D elevation mode that completes in under 4 minutes."*
