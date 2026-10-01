# DroneMap v2.0 — Test Datasets & Benchmark Catalog

This directory contains curated real-world aerial drone survey videos paired with synchronized DJI SRT (`.srt`) and CSV (`.csv`) flight telemetry for testing, benchmarking, and demonstrating **DroneMap v2.0**.

---

## 1. Local Ready-to-Run Drone Survey Datasets (`./samples/`)

All datasets can be downloaded or updated anytime via:
```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --list        # List all datasets & installation status
.venv\Scripts\python.exe scripts\download_datasets.py              # Download/update all datasets
.venv\Scripts\python.exe scripts\download_datasets.py --only <id>  # Download a single dataset
```

| # | Dataset ID | Video / Telemetry Files | Duration / FPS | Scene Archetype & Location | What It Tests Best |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Castle Ruins** | `drone_castle_survey.mp4`<br>`drone_castle_survey.srt` / `.csv` | 12s / 30fps (1080p) | **Toolse Medieval Castle (Estonia)**<br>Stone ruins, vertical walls, coastal bluff | Full 3D OpenMVS surface meshing, 5-class scene segmentation (Sky/Water/Terrain/Vegetation/Castle), horizon filtering |
| **2** | **Sacred Monument** | `drone_bahai_temple.mp4`<br>`drone_bahai_temple.srt` / `.csv` | 14s / 30fps (1080p) | **Baha'i House of Worship (Illinois, USA)**<br>Intricate white dome, circular radial gardens | 360-degree orbital 3D structure recovery, dome surface curvature, fine architectural lace masonry |
| **3** | **Maritime Shipyard** | `drone_loksa_shipyard.mp4`<br>`drone_loksa_shipyard.srt` / `.csv` | 14s / 30fps (1080p) | **Loksa Shipyard & Harbor (Gulf of Finland)**<br>Dry docks, gantry cranes, breakwater pier | Maritime port survey, water leveling at datum, industrial crane/shed verticality, breakwater geometry |
| **4** | **Highway Flyover** | `drone_chicago_interchange.mp4`<br>`drone_chicago_interchange.srt` / `.csv` | 14s / 30fps (1080p) | **Jane M. Byrne Interchange (Chicago, USA)**<br>Multi-tier curved concrete highway ramps | Multi-tier roadway bridge decks, road network trafficability, YOLO dynamic vehicle masking across moving traffic |
| **5** | **Crop Field Maze** | `drone_hemp_maze.mp4`<br>`drone_hemp_maze.srt` / `.csv` | 14s / 30fps (1080p) | **Kanepi Agricultural Maze (Estonia)**<br>Mowed labyrinth corridors, farm tracks | Precision farming survey, repetitive texture feature matching, orthomosaic stitching, crop canopy vs ground DTM |
| **6** | **Urban Arch Bridge** | `drone_urban_bridge.mp4`<br>`drone_urban_bridge.srt` / `.csv` | 14s / 30fps (1080p) | **Tartu Kaarsild Arch Bridge (Estonia)**<br>Town hall plaza, pedestrian arch bridge, river | Urban façade geometry, bridge overhangs, pedestrian/cyclist dynamic masking |
| **7** | **Fortress Orbit** | `drone_fortress_orbit.mp4`<br>`drone_fortress_orbit.srt` / `.csv` | 14s / 30fps (1080p) | **Haapsalu Episcopal Castle (Estonia)**<br>Stone ramparts, cathedral tower, moat | Orbital camera trajectory, loop closure, multi-elevation 3D structure recovery |
| **8** | **Wildlife Overpass** | `drone_highway_overpass.mp4`<br>`drone_highway_overpass.srt` / `.csv` | 14s / 30fps (1080p) | **Kolu Wildlife Overpass (Estonia)**<br>Green overpass bridge spanning dual-carriageway | Transportation corridor mapping, road lane surface planarity, bridge overpass geometry |
| **9** | **Botanical Garden** | `drone_botanical_garden.mp4`<br>`drone_botanical_garden.srt` / `.csv` | 12s / 30fps (1080p) | **Univ. of Tartu Botanical Garden (Estonia)**<br>Glasshouses, dense tree canopy, pond | DSM vs. Bare-Earth DTM morphological filtering, vegetation vs. building separation |
| **10** | **High-Rise City** | `drone_city_blocks.mp4`<br>`drone_city_blocks.srt` / `.csv` | 12s / 30fps (1080p) | **Annelinn High-Rise District (Estonia)**<br>9-story apartment blocks, parking lots | Urban canyon relief, road corridor trafficability, moving traffic removal |
| **11** | **Recreation Park** | `drone_flight_sample.mp4`<br>`drone_flight_sample.srt` / `.csv` | 20s / 30fps (1080p) | **Germia Park Facility (Kosovo)**<br>Low-relief park, walkways, pool basin | 2.5D Delaunay terrain surface meshing, geodetic ECEF/UTM alignment |
| **12** | **Alpine Viaduct** | `drone_landwasser_viaduct.mp4`<br>`drone_landwasser_viaduct.srt` / `.csv` | 14s / 30fps (1080p) | **Landwasser Alpine Viaduct (Switzerland)**<br>65m limestone railway bridge, canyon gorge | Extreme vertical relief (200m+ dynamic range), deep canyon ground envelope, curved stone pillar meshing |

---

## 2. Quick Run Commands

### Run with GNSS Telemetry (Georeferenced Metric Products in UTM CRS)
```powershell
# 1. Medieval Castle Ruins (3D Structures & 5-Class Segmentation)
.venv\Scripts\dronemap.exe run --video samples/drone_castle_survey.mp4 --telemetry samples/drone_castle_survey.srt

# 2. Baha'i House of Worship (Orbital Sacred Domed Architecture)
.venv\Scripts\dronemap.exe run --video samples/drone_bahai_temple.mp4 --telemetry samples/drone_bahai_temple.srt

# 3. Loksa Maritime Shipyard & Harbor Pier
.venv\Scripts\dronemap.exe run --video samples/drone_loksa_shipyard.mp4 --telemetry samples/drone_loksa_shipyard.srt

# 4. Chicago Multi-Tier Flyover Interchange (Expressway & Traffic)
.venv\Scripts\dronemap.exe run --video samples/drone_chicago_interchange.mp4 --telemetry samples/drone_chicago_interchange.srt

# 5. Agricultural Crop Maze & Farm Corridors (Precision Farming)
.venv\Scripts\dronemap.exe run --video samples/drone_hemp_maze.mp4 --telemetry samples/drone_hemp_maze.srt

# 6. Tartu Urban Plaza & Arch Bridge
.venv\Scripts\dronemap.exe run --video samples/drone_urban_bridge.mp4 --telemetry samples/drone_urban_bridge.srt
```

### Fast Iteration Mode (Skip Heavy `RefineMesh`)
To complete dense point clouds, 3D meshes, DSM/DTM rasters, and orthomosaics in **~2–3 minutes** without the slow iterative OpenMVS `RefineMesh` step:
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_bahai_temple.mp4 --telemetry samples/drone_bahai_temple.srt --set mesh.refine=false
```

### Run Without Telemetry (Relative Scale Mode)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_urban_bridge.mp4 --no-telemetry
```

---

## 3. External Open-Source UAV Photogrammetry Benchmarks

For testing on multi-gigabyte industrial surveying projects with RTK ground control points (GCPs):

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
