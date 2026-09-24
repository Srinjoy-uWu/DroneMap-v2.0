# DroneMap v2.0 (SIH26158)

**Single-Pass Drone Video → Georeferenced, Metrically Accurate 3D Model**  
Smart India Hackathon problem statement **SIH26158**: An end-to-end aerial photogrammetry suite that turns a single continuous drone flight video (with or without telemetry) into a georeferenced, metrically scaled, photo-textured 3D model, classified semantic layers, and survey GIS deliverables — complete with an interactive WebGL 3D measurement studio, autonomous AI copilot, and automated accuracy reporting.

---

## Architecture: Hybrid Dual-Engine (Offline Core + AI Booster)

Every stage features a **Guaranteed Core** implementation (deterministic, offline-ready, air-gapped) and an optional **AI Booster Layer** (multimodal LLMs, foundation vision models, self-healing diagnostics).

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

| Stage | Core (Offline Edge) | AI Booster Layer (Optional Keys) |
|---|---|---|
| **1 Preprocessing** | 4-factor scoring: sharpness, exposure, contrast, entropy | Keyframe cluster selection |
| **2 Dynamic Masking** | YOLOv8-seg + velocity dilation ($r = 12 + 0.5 \cdot \|\mathbf{v}\|$) | SAM 2 boundary refinement |
| **3 Pose & SfM** | COLMAP sequential SfM + GPS priors | LightGlue deep neural tie-point escalation |
| **3b Georeferencing** | Metric scale + ECEF/ENU geodetic alignment | RTK/PPK carrier phase verification |
| **4 Depth / Semantics** | Depth Anything V2 Small + SegFormer-B0 | Cloud Foundation APIs (Large) |
| **5 Dense Cloud** | OpenMVS `DensifyPointCloud` | 3D Gaussian Splatting (Kaggle) |
| **6 Mesh & Texturing** | Watertight screened Poisson + PBR atlas patch | Topology validation |
| **7 Deliverables & AI** | GLB, LAZ (Class 1-3 Confidence), DSM/DTM, Ortho | Multimodal Tactical Intelligence Dossier |
| **Web Studio** | Three.js 3D Viewer + Transparency Drawer | Natural Language 3D Spatial Copilot |

---

## 💻 Running Locally (Windows / Linux)

### 1. Prerequisites
- **OS**: Windows 10/11 or Ubuntu Linux (x64)
- **GPU**: NVIDIA GPU with CUDA support recommended (CPU mode supported)
- **Python**: Python 3.10 or 3.11 *(Note: Open3D and pycolmap require ≤ 3.11)*

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Srinjoy-uWu/DroneMap-v2.0.git
cd DroneMap-v2.0

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install DroneMap and dependencies
pip install -e .

# (Optional) Enable AI Booster Layer (Gemini, OpenAI, Anthropic):
cp .env.example .env
# Edit .env and insert your API key
```

*(Alternatively, if you use `uv`: `uv sync --extra ml --extra geo3d`)*

### 3. Bootstrap Tools (COLMAP, OpenMVS & Vocab Tree)
Run the automated bootstrap script to download and verify the external binaries into `tools/` (no admin rights or manual compilation needed):
```powershell
python scripts/bootstrap.py
```
Verify your environment:
```powershell
dronemap doctor
```

### 4. Launch the Web Studio
You can start the Web Studio with one click:
- **Windows**: Double-click `launch_studio.bat`, **OR**
- **Terminal**: Run `dronemap serve`

Open your browser at **`http://127.0.0.1:8000`**:
- Click **"+ Process Drone Video"** $\to$ choose your drone video (`.mp4` / `.mov`) $\to$ click **"Start Processing"**.
- View real-time stage progress in the stepper drawer.
- Once finished, view the photorealistic 3D model, switch between Textured/Clay modes, toggle seam-levelled texture variants, and measure 3D real-world distances with point-and-click accuracy.

### 5. CLI Usage
```powershell
# Run the full pipeline on a flight video:
dronemap run --video path/to/flight.mp4 --telemetry path/to/flight.srt --run-id demo01

# No telemetry available (relative scale mode):
dronemap run --video flight.mp4 --no-telemetry --run-id flight_relative

# Run individual stages:
dronemap frames --run-id demo01
dronemap pose   --run-id demo01
dronemap dense  --run-id demo01
dronemap mesh   --run-id demo01
dronemap export --run-id demo01

# Clean scratch, intermediate depth maps (.dmap), or old runs:
dronemap clean --all                       # Safe repo-wide cleanup
dronemap clean --intermediate              # Prune .dmap caches from runs
dronemap clean <run_id>                    # Delete a single specific run
```

---

## ☁ Running on Kaggle (Free Cloud GPU T4 × 2)

Kaggle provides **30 hours/week of free dual-GPU (NVIDIA T4 × 2)**. You can run DroneMap on Kaggle in two ways:

### Method A: Full Pipeline in a Kaggle Notebook
1. Open [Kaggle](https://www.kaggle.com/) $\to$ **New Notebook**.
2. In the right settings panel:
   - **Accelerator**: Select **GPU T4 × 2**
   - **Internet**: **ON**
3. In the first cell, install dependencies and clone DroneMap:
   ```bash
   # Install COLMAP and FFmpeg via apt
   !apt-get update -qq && apt-get install -y -qq colmap ffmpeg

   # Clone DroneMap repository
   !git clone https://github.com/Srinjoy-uWu/DroneMap.git
   %cd DroneMap

   # Install Python package
   !pip install -e . --quiet
   ```
4. Upload your drone video (or link a Kaggle Dataset) and run:
   ```bash
   !python -m dronemap.cli run --video /kaggle/input/your-dataset/flight.mp4 --profile lowvram
   ```
5. All outputs (`model.glb`, `cloud.laz`, GeoTIFFs, reports) are written to `data/runs/<run_id>/07_export/` ready for download from the notebook file browser.

---

### Method B: Hybrid 3D Gaussian Splatting (3DGS)
The repository contains ready-to-run 3DGS training notebooks under `notebooks/`:
- [`notebooks/3dgs_splatfacto_kaggle.ipynb`](notebooks/3dgs_splatfacto_kaggle.ipynb)
- [`notebooks/sih26158_3dgs_kaggle.py`](notebooks/sih26158_3dgs_kaggle.py)

#### Workflow:
1. **Locally**: Run stages 1–3 (`dronemap frames`, `dronemap pose`) to produce keyframe images and the sparse COLMAP model.
2. **Kaggle**: Upload `03_pose/` and `01_frames/images/` as a private Kaggle Dataset.
3. Open `notebooks/3dgs_splatfacto_kaggle.ipynb` on Kaggle with T4 × 2 GPU $\to$ trains 30,000 iterations in **~25–35 minutes** using `nerfstudio` and `gsplat`.
4. Download the trained `point_cloud.ply` and renders back to your machine.
5. **Locally**: Run `dronemap export --run-id <id>` to package the model into final deliverables.

---

## 🧪 Testing & Verification

The repository includes a comprehensive unit, integration, and hardening test suite (172 tests):
```powershell
# Run the complete test suite:
pytest -q
```
Tests cover:
- Truthful georeferencing status tiers and coordinate frames (LOCAL_RELATIVE vs UTM)
- 3D Regional Confidence classification (ASPRS LAZ Tiers: Observed, Estimated, Inferred)
- Stage-by-Stage Quality & Hardening Audit report verification
- Autonomous AI & LLM Copilot client, graceful degradation, and spatial chat
- Self-healing photogrammetry diagnostics and parameter recommendation
- Multi-factor frame scoring (sharpness, exposure, contrast, entropy)
- Ground plane PCA fitting with geodetic vertical priors
- UV-to-texel texture sampling and black-atlas detection
- Automatic mesh repair, non-manifold removal, and component filtering
- Clean command disk pruning safeguards
- Telemetry parsers (DJI SRT and generic CSV)
- Exhaustive matching escalation frame cap ($N \le 150$)
- Stage 4b SIH26158 4-class semantic rollup mapping
- Dynamic masking graceful degradation on all frames dropped
- PyTorch GPU memory cleanup and cache release

---

## 📐 Accuracy & Georeferencing Protocol

Absolute georeferencing accuracy is determined by the **GNSS regime**:

| GNSS Mode | Horizontal Accuracy | Vertical Accuracy | Application |
|---|---|---|---|
| **Standalone GNSS** (Consumer drones) | ~1–3 m | ~2–5 m | Site layout, visual inspection |
| **RTK / PPK** (Survey drones) | ~1–3 cm | ~3–8 cm | Cadastral survey, engineering |
| **No Telemetry** | Relative only | Relative only | Scaleless 3D geometry |

Every run records its geodetic coordinate frame in `manifest.json` and `accuracy_report.json`.

---

## 📂 Deliverables Generated per Run

Each run stores its deliverables in `data/runs/<run_id>/07_export/`:
- **`model.glb`**: Self-contained glTF binary with double-sided PBR materials, baked vertex normals, and 8K texture atlas.
- **`model_alt.glb`**: Alternate texture variant (retained when seam-leveling fallback occurs for direct comparison).
- **`cloud.laz`**: Georeferenced dense point cloud (ASPRS LAS 1.4).
- **`dsm.tif`**: Digital Surface Model (GeoTIFF, metric elevations).
- **`dtm.tif`**: Digital Terrain Model (bare-earth GeoTIFF).
- **`orthomosaic.tif`**: Orthorectified true-scale RGB aerial map.
- **`semantics.json` / `sih26158_categories.json`**: Semantic class distributions ((i) terrain, (ii) buildings, (iii) roads & infrastructure, (iv) vegetation & obstacles).
- **`accuracy_report.json` & `report.html`**: Quality metrics, flight geometry assessment, and CRS metadata.
- **`trajectory.kml` / `trajectory.json`**: Recovered 3D flight path.



