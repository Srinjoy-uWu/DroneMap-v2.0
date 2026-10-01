# DroneMap v2.0 — Test Datasets & Benchmark Catalog

This directory contains curated real-world aerial drone survey videos paired with synchronized DJI SRT (`.srt`) and CSV (`.csv`) flight telemetry for testing, benchmarking, and demonstrating **DroneMap v2.0**.

---

## 1. 360° Revolving & Orbit Drone Datasets (Revolving Around Objects & Buildings)

In aerial photogrammetry, **revolving / 360° Point of Interest (POI) orbital trajectories** are the gold standard for full 3D asset reconstruction:
- **All-Around Visibility**: Captures all 4 facades/sides of a structure rather than just one oblique face.
- **Wide Triangulation Angles**: Provides continuous $\Delta\theta \approx 5^\circ - 15^\circ$ baseline angles between consecutive frames, preventing planar collapse.
- **SfM Loop Closure**: Flying a complete loop allows COLMAP bundle adjustment to close the drift loop, yielding sub-centimeter registration.
- **Watertight 3D Meshes**: Enables OpenMVS to reconstruct watertight, convex 3D models with texture atlases spanning all azimuths.

The following datasets in `./samples/` specifically feature the drone **revolving around central buildings and vertical structures**:

| Dataset ID | Video & Telemetry Files | Revolving Object / Structure | Trajectory & Motion | Best 3D Photogrammetry Properties |
| :--- | :--- | :--- | :--- | :--- |
| **`bahai_temple`** | `drone_bahai_temple.mp4`<br>`.srt` / `.csv` | **Baha'i House of Worship (Illinois, USA)**<br>Intricate 9-sided white stone domed temple | **Circular 360° orbit** around the dome while maintaining camera lock on the temple apex | Perfect rotational symmetry, complex arched tracery, radial garden symmetry, dome curvature |
| **`ihasalu_lighthouse`** | `drone_ihasalu_lighthouse.mp4`<br>`.srt` / `.csv` | **Ihasalu Coastal Lighthouse (Estonia)**<br>Historic cylindrical masonry/concrete lighthouse tower | **Tight circular orbit** revolving around the vertical tower and coastal military structures | Single vertical cylinder benchmark, all-around azimuth coverage, sea datum leveling |
| **`fortress_orbit`** | `drone_fortress_orbit.mp4`<br>`.srt` / `.csv` | **Haapsalu Medieval Castle (Estonia)**<br>Stone fortress ramparts, cathedral, and watchtower | **Perimeter revolving orbit** around the castle courtyard and towers | High vertical stone walls, multi-elevation towers, inner courtyard vs outer moat |
| **`wind_turbine`** | `drone_wind_turbine.mp4`<br>`.srt` / `.csv` | **Kunda Industrial Wind Turbine (Estonia)**<br>80m slender turbine tower, nacelle, rotor blades | **Tight circular orbit** revolving around the turbine hub and nacelle | High-altitude multi-view triangulation of slender tall structures, rotor geometry |
| **`castle_survey`** | `drone_castle_survey.mp4`<br>`.srt` / `.csv` | **Toolse Medieval Castle (Estonia)**<br>Stone castle ruins on a coastal peninsula bluff | **Arc orbit & tracking gimbal** revolving around the coastal ruins while pitching down | 5-class scene segmentation, vertical stone wall prominence, bare-earth DTM stripping |

---

## 2. Complete Local Dataset Catalog (`./samples/`)

All datasets can be downloaded or updated anytime via:
```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --list        # List all datasets & installation status
.venv\Scripts\python.exe scripts\download_datasets.py              # Download/update all datasets
.venv\Scripts\python.exe scripts\download_datasets.py --only <id>  # Download a single dataset
```

| # | Dataset ID | Video / Telemetry Files | Duration / FPS | Scene Archetype & Location | What It Tests Best |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Sacred Domed Monument** | `drone_bahai_temple.mp4`<br>`.srt` / `.csv` | 14s / 30fps (1080p) | **Baha'i House of Worship (Illinois, USA)**<br>Intricate white dome, circular radial gardens | **360° orbital 3D structure recovery**, dome surface curvature, fine architectural lace masonry |
| **2** | **Cylindrical Lighthouse** | `drone_ihasalu_lighthouse.mp4`<br>`.srt` / `.csv` | 14s / 30fps (1080p) | **Ihasalu Coastal Lighthouse (Estonia)**<br>Cylindrical stone/concrete tower on bluff | **360° revolving orbit**, vertical cylindrical geometry, sea/land boundary elevation |
| **3** | **Wind Turbine Orbit** | `drone_wind_turbine.mp4`<br>`.srt` / `.csv` | 14s / 30fps (1080p) | **Kunda Industrial Wind Farm (Estonia)**<br>80m slender turbine tower, nacelle, blades | **Multi-angle revolving reconstruction**, slender vertical structure triangulation |
| **4** | **Fortress Orbit** | `drone_fortress_orbit.mp4`<br>`drone_fortress_orbit.srt` / `.csv` | 14s / 30fps (1080p) | **Haapsalu Episcopal Castle (Estonia)**<br>Stone ramparts, cathedral tower, moat | Orbital camera trajectory, loop closure, multi-elevation 3D structure recovery |
| **5** | **Castle Ruins** | `drone_castle_survey.mp4`<br>`drone_castle_survey.srt` / `.csv` | 12s / 30fps (1080p) | **Toolse Medieval Castle (Estonia)**<br>Stone ruins, vertical walls, coastal bluff | Full 3D OpenMVS surface meshing, 5-class scene segmentation (Sky/Water/Terrain/Vegetation/Castle) |
| **6** | **Maritime Shipyard** | `drone_loksa_shipyard.mp4`<br>`drone_loksa_shipyard.srt` / `.csv` | 14s / 30fps (1080p) | **Loksa Shipyard & Harbor (Gulf of Finland)**<br>Dry docks, gantry cranes, breakwater pier | Maritime port survey, water leveling at datum, industrial crane/shed verticality, breakwater geometry |
| **7** | **Highway Flyover** | `drone_chicago_interchange.mp4`<br>`drone_chicago_interchange.srt` / `.csv` | 14s / 30fps (1080p) | **Jane M. Byrne Interchange (Chicago, USA)**<br>Multi-tier curved concrete highway ramps | Multi-tier roadway bridge decks, road network trafficability, YOLO dynamic vehicle masking across moving traffic |
| **8** | **Crop Field Maze** | `drone_hemp_maze.mp4`<br>`drone_hemp_maze.srt` / `.csv` | 14s / 30fps (1080p) | **Kanepi Agricultural Maze (Estonia)**<br>Mowed labyrinth corridors, farm tracks | Precision farming survey, repetitive texture feature matching, orthomosaic stitching, crop canopy vs ground DTM |
| **9** | **Urban Arch Bridge** | `drone_urban_bridge.mp4`<br>`drone_urban_bridge.srt` / `.csv` | 14s / 30fps (1080p) | **Tartu Kaarsild Arch Bridge (Estonia)**<br>Town hall plaza, pedestrian arch bridge, river | Urban façade geometry, bridge overhangs, pedestrian/cyclist dynamic masking |
| **10** | **Wildlife Overpass** | `drone_highway_overpass.mp4`<br>`drone_highway_overpass.srt` / `.csv` | 14s / 30fps (1080p) | **Kolu Wildlife Overpass (Estonia)**<br>Green overpass bridge spanning dual-carriageway | Transportation corridor mapping, road lane surface planarity, bridge overpass geometry |
| **11** | **Botanical Garden** | `drone_botanical_garden.mp4`<br>`drone_botanical_garden.srt` / `.csv` | 12s / 30fps (1080p) | **Univ. of Tartu Botanical Garden (Estonia)**<br>Glasshouses, dense tree canopy, pond | DSM vs. Bare-Earth DTM morphological filtering, vegetation vs. building separation |
| **12** | **High-Rise City** | `drone_city_blocks.mp4`<br>`drone_city_blocks.srt` / `.csv` | 12s / 30fps (1080p) | **Annelinn High-Rise District (Estonia)**<br>9-story apartment blocks, parking lots | Urban canyon relief, road corridor trafficability, moving traffic removal |
| **13** | **Recreation Park** | `drone_flight_sample.mp4`<br>`drone_flight_sample.srt` / `.csv` | 20s / 30fps (1080p) | **Germia Park Facility (Kosovo)**<br>Low-relief park, walkways, pool basin | 2.5D Delaunay terrain surface meshing, geodetic ECEF/UTM alignment |
| **14** | **Alpine Viaduct** | `drone_landwasser_viaduct.mp4`<br>`drone_landwasser_viaduct.srt` / `.csv` | 14s / 30fps (1080p) | **Landwasser Alpine Viaduct (Switzerland)**<br>65m limestone railway bridge, canyon gorge | Extreme vertical relief (200m+ dynamic range), deep canyon ground envelope, curved stone pillar meshing |

---

## 3. Quick Run Commands for Revolving Datasets

### A. Baha'i House of Worship (Orbital Sacred Domed Architecture)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_bahai_temple.mp4 --telemetry samples/drone_bahai_temple.srt
```

### B. Ihasalu Coastal Lighthouse (360° Cylindrical Tower Orbit)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_ihasalu_lighthouse.mp4 --telemetry samples/drone_ihasalu_lighthouse.srt
```

### C. Haapsalu Medieval Castle (360° Fortress & Cathedral Orbit)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_fortress_orbit.mp4 --telemetry samples/drone_fortress_orbit.srt
```

### D. Industrial Wind Turbine (Continuous Nacelle & Rotor Orbit)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_wind_turbine.mp4 --telemetry samples/drone_wind_turbine.srt
```

### E. Fast Iteration Mode (Skip Heavy `RefineMesh` for ~2–3 Minute Results)
Add `--set mesh.refine=false` to skip the iterative OpenMVS `RefineMesh` stage while keeping full 3D point densification, reconstruction, and texturing:
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_ihasalu_lighthouse.mp4 --telemetry samples/drone_ihasalu_lighthouse.srt --set mesh.refine=false
```

---

## 4. External Open-Source UAV Photogrammetry Benchmarks

For testing on large multi-hectare industrial surveying projects with RTK ground control points (GCPs):

1. **OpenDroneMap Benchmark Datasets (`odm_data`)**
   - **URL**: https://github.com/OpenDroneMap/odm_data
   - **Key Sets**: `odm_data_caliterra` (open-pit earthworks & stockpiles), `odm_data_bellus` (urban village buildings), `odm_data_aukerman` (rural landscape).
   - Convert image sets to video:
     ```powershell
     ffmpeg -framerate 5 -pattern_type glob -i "*.JPG" -c:v libx264 -pix_fmt yuv420p odm_survey.mp4
     ```

2. **Pix4D Example Datasets**
   - **URL**: https://support.pix4d.com/hc/en-us/articles/360000235126-Example-projects-real-photogrammetry-data
   - **Key Sets**: Open quarry stockpile (metric volumetric calculation), suburban housing, industrial telecom tower.

3. **AgEagle / senseFly eBee UAV Datasets**
   - **URL**: https://ageagle.com/drone-datasets/
   - **Key Sets**: Quarry RTK survey, Urban Village 3D mesh, Gravel Pit earthworks.

4. **UAVid (High-Resolution UAV Semantic Benchmark)**
   - **URL**: https://uavid.nl/
   - **Key Sets**: 42 oblique 4K video sequences over urban corridors, buildings, trees, and moving vehicles.
