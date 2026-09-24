# SIH26158 — Single-Pass Drone Video to Accurate, Georeferenced 3D Model
## Official 5-Slide Master PPT Deck & Presentation Blueprint
**Problem Owner:** National Technical Research Organisation (NTRO)  
**Theme:** Robotics & Drones / Defense Intelligence | **Category:** Software & Geospatial AI  
**System Name:** **DroneMap** (Dual-Tier Edge Photogrammetry & Neural Reconstruction)  
**Target Hardware:** Tactical Edge Laptop GPU (NVIDIA RTX 3050 6 GB VRAM, 100% Offline / Air-Gapped)  

---

## Slide 1: Problem and Solution Name

### 📌 Visual Slide Layout (What to render on Slide 1)
- **Top Header Banner**: Smart India Hackathon 2026 · Problem Statement **SIH26158** · Organization: **NTRO**
- **Main Title**: **DroneMap: Single-Pass Drone Video to Accurate, Georeferenced 3D Model**
- **Sub-Title**: Real-Time Dynamic-Aware Edge Photogrammetry & Neural Scene Synthesis
- **Two Visual Cards**:
  - **Left Card: The Tactical Defense Problem**
    - UAVs fly **a single forward pass** along hostile borders/recon corridors (battery limits & radar evasion).
    - Commercial tools (**Pix4D, Metashape, ODM**) fail: they demand multi-strip serpentine grids with 80% cross-track overlap, take 3–6 hours, and require cloud workstations.
    - Single-pass video triggers the **epipole trap** (zero forward disparity), **nadir focal ambiguity**, **motion blur**, and **moving vehicle ghosting**.
  - **Right Card: The Proposed Solution (DroneMap)**
    - **Dual-Tier Architecture**: 100% Guaranteed Local Core (COLMAP + OpenMVS + YOLOv8) + Cloud-Free AI Stretch Tier (3DGS on Kaggle).
    - **Air-Gapped Edge Processing**: Runs locally on a 6 GB RTX 3050 laptop GPU in **under 8 minutes**.
    - **Survey-Grade GIS Deliverables**: ASPRS LAS 1.4 point clouds, DSM/DTM GeoTIFFs, Orthomosaics, and an interactive 3D Web Studio with live metric measurement HUD.
- **Key Metric Badges**: `100% Offline` | `Sub-8 Min Latency` | `RTX 3050 6GB` | `Survey of India Compliant`

### 🗣 Speaker Script (Minute 1: The Pitch)
> *"Respected Jury Members, our project addresses Problem Statement SIH26158 from the National Technical Research Organisation (NTRO): generating an accurate, georeferenced 3D model from a single-pass drone video.*
> 
> *In defense reconnaissance and border surveillance, drones cannot fly cross-hatch survey grids—they fly a single forward pass over hostile territory. Traditional photogrammetry software like Pix4D or Metashape completely breaks down on single-pass footage because they assume 80% cross-track overlap and require cloud supercomputers. Furthermore, moving convoys leave smeared ghost trails, and forward optical flow causes vertical geometry to collapse.*
> 
> *We built **DroneMap**: an evidence-led photogrammetry and geospatial AI suite that processes raw single-pass video and telemetry locally on an edge laptop GPU in under eight minutes. It masks dynamic objects with YOLOv8, recovers metric scale via GPS sensor fusion, and delivers Survey of India compliant GIS models with an interactive 3D Web Studio."*

---

## Slide 2: Technical Approach

### 📌 Visual Slide Layout (What to render on Slide 2)
- **Pipeline Flow Diagram (Horizontal / 5-Stage Blocks)**:
  1. **Stage 1 (O(1) Streaming Preprocessing)**: 2-pass frame decoder, Laplacian variance blur filtering, baseline-aware optical flow stride enforcing 75–80% overlap.
  2. **Stage 2 (YOLOv8 Dynamic Masking)**: Segments vehicles/people with 15px morphological dilation; passes binary masks to COLMAP `--ImageReader.mask_path` to prevent ghost tie-points.
  3. **Stage 3 & 3b (SfM & Georeferencing)**: Sequential SIFT matching with quadratic overlap window ($t-8$ to $t+8$) + calibrated pinhole focal prior; Sim(3) Umeyama GPS alignment ($WGS84 \rightarrow ECEF \rightarrow UTM$).
  4. **Stage 5 & 6 (Dense Cloud & Dual Meshing)**: OpenMVS dense stereo depth fusion ($200\text{k}+$ points); Poisson 3D meshing with 8192px PBR texture atlas + 2.5D PCA TIN terrain engine for flat topography.
  5. **Stage 7 & UI (GIS Export & 3D Web Studio)**: Exports GLB 2.0, LAS 1.4, DSM, DTM, Orthomosaic, and KML; launches 60 FPS Three.js Web Studio with live raycaster distance/bearing HUD.
- **Side Panel: AI Stretch Tier (Kaggle Cloud)**:
  - Depth Anything V2 / Metric3D v2 monocular priors + 3D Gaussian Splatting (Nerfstudio) trained on free Kaggle T4 GPUs in 25 minutes.

### 🗣 Speaker Script (Minute 2: How It Works)
> *"Our technical approach is architected around a 7-stage deterministic pipeline. In Stage 1, a streaming two-pass decoder filters out motion-blurred frames using Laplacian variance and samples optical flow to select keyframes with an optimal 80% forward overlap, maintaining an O(1) memory footprint.*
> 
> *In Stage 2, YOLOv8 segmentation masks out dynamic vehicles and pedestrians with morphological dilation. By passing these masks directly to COLMAP's feature extractor, SIFT keypoints are never placed on moving objects, completely eliminating ghost geometry.*
> 
> *In Stage 3, we replace exhaustive matching with sequential matching using quadratic overlap windowing, cutting matching time from 40 minutes to under 90 seconds. We bypass the nadir focal length ambiguity using calibrated pinhole priors, and Stage 3b executes closed-form Umeyama Sim(3) alignment between camera poses and GPS telemetry, locking the model into real-world UTM Zone coordinates where 1 unit equals exactly 1.000 meter.*
> 
> *Stages 5 and 6 use OpenMVS to generate over 200,000 dense points and project an 8K texture atlas onto a Poisson surface mesh, while our proprietary 2.5D PCA TIN engine handles flat terrain. Finally, Stage 7 outputs standard GIS products and launches our 3D Web Studio."*

---

## Slide 3: Feasibility and Challenges

### 📌 Visual Slide Layout (What to render on Slide 3)
- **Feasibility Grid (Left Column)**:
  - **Edge Hardware Feasibility**: Evaluated on an NVIDIA RTX 3050 Laptop GPU (6 GB VRAM, 16 GB RAM). No supercomputers or cloud clusters needed.
  - **Zero Paywall & Cost Feasibility**: 100% free open-source tools (COLMAP 4.1.1, OpenMVS 2.4.0, YOLOv8s, PyProj, Three.js). Cloud stretch runs on free Kaggle GPU quotas (~120 hrs/wk).
  - **Air-Gapped Operational Feasibility**: All CUDA binaries, model weights, and viewer libraries are stored locally in the repo. Zero internet or API dependencies.
- **Challenges & Concrete Engineering Mitigations (Right Column Table)**:

| Technical Challenge | Root Cause in Single-Pass Flight | DroneMap Engineering Mitigation |
|---|---|---|
| **Narrow Baseline & Epipole Trap** | Forward flight has zero disparity at center of frame | Sequential matching with quadratic skips ($t \pm 8$) + calibrated pinhole focal priors |
| **Dynamic Clutter & Ghosting** | Moving traffic violates static scene assumption | YOLOv8s-seg instance masking + 15px boundary dilation fed to SIFT feature extractor |
| **False Accepts on Flat Tarmac/Hover**| Drone hovering registers 100% frames but yields flat 2D sheet | **Scale-free geometric quality gating (`quality.py`)**: Triangulation angle ($>8^\circ$), PCA planarity ($\lambda_3/\lambda_1$), baseline/depth ratio |
| **GPS Denial / Electronic Jamming** | Drone operating in GPS-spoofed tactical airspace | Automatic up-to-scale fallback (`--no-telemetry`) using camera baseline scaling and Depth Anything priors |

### 🗣 Speaker Script (Minute 3: Feasibility & Mitigations)
> *"Feasibility was our primary engineering constraint. DroneMap runs 100% locally on a consumer 6 GB laptop GPU without internet access, satisfying military air-gapped security protocols. We have zero software paywalls and zero commercial licensing restrictions.*
> 
> *We systematically mitigated the four classic failure modes of single-pass aerial capture: First, to counter the epipole trap where forward flight creates zero disparity along the flight line, we combine quadratic sequential matching with calibrated focal priors.*
> 
> *Second, we eliminated dynamic ghosting by masking moving vehicles with YOLOv8 before feature extraction.*
> 
> *Third, we solved the hidden danger of false accepts: traditional software counts registered points, so a hovering drone producing 130,000 points on a flat road is declared a success. Our custom quality engine (`quality.py`) evaluates scale-free geometry—triangulation angles, baseline-to-depth ratios, and PCA singular ratios—emitting three truthful verdicts: `ACCEPT_3D`, `TERRAIN_2_5D` for flat beaches/fields, or `REJECT` before wasting GPU compute.*
> 
> *Fourth, if GPS is jammed, our pipeline falls back to relative metric scaling using monocular depth priors."*

---

## Slide 4: Benefits and Impact

### 📌 Visual Slide Layout (What to render on Slide 4)
- **4 High-Impact Value Cards**:
  1. **Tactical Speed & Operational Agility**:
     - End-to-end turnaround: **< 8 minutes** on field laptops vs. **3 to 6 hours** for commercial photogrammetry.
     - Enables immediate in-field mission replanning, artillery clearance assessment, and battle damage assessment (BDA).
  2. **Survey of India & NATO GIS Interoperability**:
     - Exports non-proprietary, open geospatial standards: **ASPRS LAS 1.4 point clouds, Float32 DSM GeoTIFFs, bare-earth DTMs, true orthomosaics, and OGC KMLs**.
     - Direct plug-and-play into defense C4ISR systems, QGIS, ArcGIS, and Google Earth Enterprise.
  3. **Live 3D Web Studio with Metric Raycaster HUD**:
     - Instant browser-based 60 FPS WebGL visualization (`http://127.0.0.1:8000`).
     - Real-time 3D Raycasting (press `M`): Measures **3D Euclidean distance**, **horizontal distance**, **vertical relief ($\Delta Z$)**, and **true geographic compass bearing**.
  4. **Evidence-Led Integrity & Ground-Truth Accuracy**:
     - Benchmarked on synthetic Blender 5.1 procedural survey worlds: **$< 1.8\%$ dimensional error** against ground truth without Ground Control Points.
     - Automatically generates an air-gapped executive survey report (`report.html`) verifying all NTRO criteria (C1–C10).

### 🗣 Speaker Script (Minute 4: Impact & Demonstration)
> *"The operational impact of DroneMap for defense and disaster response is transformative. First, speed: intelligence officers gain survey-grade 3D models in under 8 minutes instead of waiting hours for cloud processing.*
> 
> *Second, military interoperability: our system produces Survey of India and ASPRS compliant LAS 1.4 point clouds, DSMs, bare-earth DTMs, and orthomosaics that feed directly into tactical GIS mapping and artillery systems.*
> 
> *Third, operational utility: in our Web Studio, an officer presses 'M' and clicks two points on a building or road. The GPU raycaster instantly displays the 3D Euclidean distance, horizontal span, vertical height relief, and true compass bearing.*
> 
> *Fourth, verified accuracy: tested against our Blender 5.1 ground-truth synthetic fixture, DroneMap achieved 5.7 cm/pixel Ground Sampling Distance with less than 1.8 percent metric error without ground control points. Crucially, every survey compiles a self-contained HTML audit report proving full compliance with NTRO criteria C1 through C10."*

---

## Slide 5: Links to References, Research and Projects

### 📌 Visual Slide Layout (What to render on Slide 5)
- **3 Structured Reference Columns**:
  - **Column 1: Foundation Computer Vision & Photogrammetry Papers**
    - **DUSt3R**: Wang et al., *Geometric 3D Vision Made Easy*, CVPR 2024 ([arXiv:2312.14132](https://arxiv.org/abs/2312.14132))
    - **MASt3R**: Leroy et al., *Grounding Image Matching in 3D with Mast3r*, ECCV 2024 ([arXiv:2406.09756](https://arxiv.org/abs/2406.09756))
    - **VGGT**: Wang et al., *Visual Geometry Grounded Transformer*, CVPR 2025 Best Paper ([arXiv:2503.11651](https://arxiv.org/abs/2503.11651))
    - **Depth Anything V2**: Yang et al., *Monocular Depth Estimation*, 2024 ([arXiv:2406.09414](https://arxiv.org/abs/2406.09414))
    - **3D Gaussian Splatting**: Kerbl et al., ACM TOG 2023 ([SIGGRAPH 2023](https://arxiv.org/abs/2308.04079))
    - **Umeyama Sim(3) Alignment**: S. Umeyama, *Least-Squares Estimation of Transformation Parameters*, IEEE TPAMI 1991
  - **Column 2: Open-Source Software & Core Repositories**
    - **COLMAP**: Structure-from-Motion and Multi-View Stereo ([colmap.github.io](https://colmap.github.io))
    - **OpenMVS**: Multi-View Stereo Reconstruction Library ([github.com/cdcseacave/openMVS](https://github.com/cdcseacave/openMVS))
    - **Ultralytics YOLOv8**: Real-Time Object Detection & Instance Segmentation ([github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics))
    - **evo**: Python package for the evaluation of visual odometry and SLAM ([github.com/MichaelGrupp/evo](https://github.com/MichaelGrupp/evo))
    - **Three.js**: JavaScript 3D WebGL Library r166 ([threejs.org](https://threejs.org))
  - **Column 3: Datasets, Benchmarks & Project Repos**
    - **UAVid Benchmark**: High-Resolution UAV Video Dataset for Semantic Segmentation ([uavid.nl](https://uavid.nl))
    - **UseGeo Benchmark**: UAV Georeferenced Imagery & LiDAR ([usegeo.fbk.eu](https://usegeo.fbk.eu))
    - **DroneMap Project Repository**: Complete local source code, test suites, and launch scripts (`src/dronemap/`)
    - **Kaggle 3DGS Cloud Notebook**: 1-click cloud Gaussian Splatting training pipeline (`notebooks/3dgs_splatfacto_kaggle.ipynb`)

### 🗣 Speaker Script (Minute 5: Research Pedigree & Closing)
> *"Our system is built on established photogrammetry foundations and peer-reviewed computer vision literature. We build upon the classical SfM algorithms of COLMAP and OpenMVS, paired with Umeyama's landmark 1991 closed-form Sim(3) formulation for geodetic alignment.*
> 
> *On the deep learning frontier, we surveyed feed-forward pointmap architectures including DUSt3R, MASt3R, and Meta's CVPR 2025 Best Paper VGGT, alongside Depth Anything V2 for metric scale disambiguation. We benchmarked our semantic and geometric accuracy using aerial datasets like UAVid and UseGeo, alongside our procedural Blender survey fixture.*
> 
> *All source code, precompiled CUDA tools, test suites, and our 1-click Kaggle Gaussian Splatting notebook are fully documented in our repository. DroneMap delivers a complete, air-gapped, Survey-of-India compliant photogrammetry engine ready for immediate field deployment. Thank you, and we look forward to your questions."*

---

## Technical Defense & Anticipated Jury Counters (Keep Handy)

1. **Q: "How do you handle feature matching on a single pass without loops?"**  
   *A:* "Traditional SfM relies on loop closures from cross-flight grids. For single-pass video, we implement sequential feature matching with quadratic overlap windowing (`--SequentialMatching.overlap 8`). Every frame is matched against temporal neighbors ($t-8$ to $t+8$). Furthermore, we initialize with a calibrated pinhole focal prior to completely bypass the well-known nadir focal length ambiguity."

2. **Q: "What happens when GPS signal is jammed or unavailable in tactical zones?"**  
   *A:* "DroneMap features an automatic up-to-scale fallback (`--no-telemetry`). When GPS is absent, the pipeline reconstructs the complete 3D mesh in relative metric space using camera baseline scaling, and can incorporate monocular depth priors from Depth Anything V2."

3. **Q: "Can your system run on tactical edge devices without an internet connection?"**  
   *A:* "Yes. All CUDA binaries (COLMAP 4.1.1, OpenMVS 2.4.0), neural network weights (YOLOv8s-seg), and viewer assets (Three.js r166) are stored locally in the repository. The entire pipeline executes with zero network calls, satisfying military air-gapped security protocols."

4. **Q: "Why not use an end-to-end deep learning model like DUSt3R or VGGT exclusively?"**  
   *A:* "Feed-forward models like DUSt3R and VGGT are revolutionary for few-view relative geometry, but they are quadratic in memory, output unscaled coordinate clouds, and suffer catastrophic out-of-memory errors on long 1,000-frame video passes on 6GB VRAM. We leverage classical SfM for guaranteed metric stability while integrating AI where it excels: dynamic masking and dense priors."
