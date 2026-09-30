# DroneMap v2.0 — Test Datasets & Benchmark Catalog

This directory contains curated real-world aerial drone survey videos paired with synchronized DJI SRT (`.srt`) and CSV (`.csv`) flight telemetry for testing, benchmarking, and demonstrating **DroneMap v2.0**.

---

## 1. Local Ready-to-Run Drone Survey Datasets (`./samples/`)

All 6 datasets below can be downloaded/regenerated anytime via:
```powershell
.venv\Scripts\python.exe scripts\download_datasets.py
```

| Dataset ID | Video / Telemetry Files | Duration | Scene Type & Geometry | What It Tests Best |
| :--- | :--- | :---: | :--- | :--- |
| **1. Castle Ruins** | `drone_castle_survey.mp4`<br>`drone_castle_survey.srt` / `.csv` | 12s (1080p) | **Toolse Medieval Castle (Estonia)**<br>Stone walls, vertical ruins, coastal bluff | Full 3D OpenMVS surface meshing, oblique parallax, texture atlas seam-leveling |
| **2. Fortress Orbit** | `drone_fortress_orbit.mp4`<br>`drone_fortress_orbit.srt` / `.csv` | 14s (1080p) | **Haapsalu Episcopal Castle**<br>High fortress walls, cathedral tower, moat | Orbital camera trajectory, loop closure, multi-elevation 3D structure recovery |
| **3. Urban Bridge** | `drone_urban_bridge.mp4`<br>`drone_urban_bridge.srt` / `.csv` | 14s (1080p) | **Tartu Town Hall & Kaarsild Arch Bridge**<br>Urban plaza, river arch bridge, buildings | Urban façade geometry, bridge overhangs, YOLOv8 dynamic pedestrian/vehicle masking |
| **4. Botanical Garden** | `drone_botanical_garden.mp4`<br>`drone_botanical_garden.srt` / `.csv` | 12s (1080p) | **Univ. of Tartu Botanical Garden**<br>Glasshouses, dense tree canopy, pond | DSM vs. Bare-Earth DTM morphological filtering, vegetation vs. building separation |
| **5. City Blocks** | `drone_city_blocks.mp4`<br>`drone_city_blocks.srt` / `.csv` | 12s (1080p) | **Annelinn High-Rise District**<br>9-story residential blocks, road network | Urban canyon relief, road corridor trafficability, moving traffic removal |
| **6. Recreation Park** | `drone_flight_sample.mp4`<br>`drone_flight_sample.srt` / `.csv` | 20s (1080p) | **Germia Park Facility (Kosovo)**<br>Low-relief park, walkways, canopy | 2.5D Delaunay terrain surface meshing, geodetic ECEF/UTM alignment |

---

## 2. Quick Run Commands

### Run with GNSS Telemetry (Georeferenced UTM Metric Output)
```powershell
# 1. Medieval Castle Ruins (3D Structure)
.venv\Scripts\dronemap.exe run --video samples/drone_castle_survey.mp4 --telemetry samples/drone_castle_survey.srt

# 2. Haapsalu Fortress Orbit (3D Structure & Towers)
.venv\Scripts\dronemap.exe run --video samples/drone_fortress_orbit.mp4 --telemetry samples/drone_fortress_orbit.srt

# 3. Tartu Urban Plaza & Arch Bridge
.venv\Scripts\dronemap.exe run --video samples/drone_urban_bridge.mp4 --telemetry samples/drone_urban_bridge.srt

# 4. Botanical Garden (DSM / DTM Canopy Test)
.venv\Scripts\dronemap.exe run --video samples/drone_botanical_garden.mp4 --telemetry samples/drone_botanical_garden.srt

# 5. High-Rise City Blocks (Urban Corridor & Vehicle Masking)
.venv\Scripts\dronemap.exe run --video samples/drone_city_blocks.mp4 --telemetry samples/drone_city_blocks.srt
```

### Fast Iteration (Skip Slow `RefineMesh` Step)
To complete full 3D meshing in ~2–4 minutes instead of running iterative `RefineMesh`:
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_fortress_orbit.mp4 --telemetry samples/drone_fortress_orbit.srt --set mesh.refine=false
```

### Run Without Telemetry (Relative Scale Mode)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_urban_bridge.mp4 --no-telemetry
```

---

## 3. External Open-Source UAV Photogrammetry Benchmarks

If you want to test on large industry-standard aerial survey datasets, the following public repositories provide free drone flights, RTK/PPK logs, and ground-truth lidar/models:

1. **OpenDroneMap Official Benchmark Datasets (`ODM Data`)**
   - **URL**: https://github.com/OpenDroneMap/odm_data
   - **Scenes**: `odm_data_aukerman` (rural terrain), `odm_data_bellus` (urban town buildings), `odm_data_caliterra` (construction earthworks), `odm_data_waterbury` (stockpile volume measurement).
   - **How to convert image sequences to video for DroneMap**:
     ```powershell
     ffmpeg -framerate 5 -pattern_type glob -i "*.JPG" -c:v libx264 -pix_fmt yuv420p odm_survey.mp4
     ```

2. **Pix4D Official Example Datasets**
   - **URL**: https://support.pix4d.com/hc/en-us/articles/360000235126-Example-projects-real-photogrammetry-data
   - **Scenes**: Quarry stockpile (metric volume verification), Suburban housing, Cell tower inspection (3D orbital structure), Corridor road mapping.

3. **AgEagle / senseFly eBee UAV Datasets**
   - **URL**: https://ageagle.com/drone-datasets/
   - **Scenes**: High-precision RTK Quarry survey, Urban Village 3D mesh, Gravel Pit, Golf Course DEM/DSM.

4. **UAVid (High-Resolution UAV Semantic & Video Benchmark)**
   - **URL**: https://uavid.nl/
   - **Scenes**: 42 oblique 4K drone video sequences captured at 50m altitude with 45° camera pitch over urban streets, buildings, trees, and moving cars.

5. **Mill 19 / Mega-NeRF & UrbanScene3D (Large-Scale 3D Reconstruction)**
   - **URL**: https://github.com/cmusatyalab/mega-nerf
   - **Scenes**: Industrial building complex and rubble piles with high-density oblique drone passes and centimeter-accurate RTK poses.
