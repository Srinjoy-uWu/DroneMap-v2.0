# SIH26158 (NTRO) — 5-Minute Live Jury Demonstration Script

> **Rule for this script:** every number spoken aloud is either read live off the
> screen or cited below with the file it comes from. Nothing is asserted that the
> run does not measure. A demo that claims more than the run measured is one
> jury question away from collapsing — and the questions in §Q are exactly the
> ones that would do it.
>
> Figures below were measured on this machine (RTX 3050 6 GB) on the Blender
> fixtures. **Re-read them before presenting** — they change with settings:
>
> There are **two** fixture runs of the same orbit, and which one you demo changes
> the numbers you may speak. `fixture_orbit_v4` is the default-settings run;
> `fixture_orbit_hires` re-runs the *same imagery* with denser keyframe sampling
> and full-resolution depth. Quote one run's whole column, never a mix.
>
> | Quantity | `fixture_orbit_v4` (defaults) | `fixture_orbit_hires` | Where it comes from |
> |---|---|---|---|
> | Keyframes kept | 47 of 480 | **97** of 480 | `stages.frames.metrics.n_keyframes` |
> | GSD | 23.2 cm/px (**fails** C2) | **9.59 cm/px** (passes C2) | `stages.export.metrics.gsd_m` |
> | Dense points | 671,115 | **4,433,064** | `stages.export.metrics.n_export_points` |
> | Mesh faces | 201,238 | 176,688 | `stages.mesh.metrics.n_faces` |
> | Trajectory RMSE, nothing fitted | **1.646 m** | 2.244 m | `evaluation_scorecard.json` → `trajectory.rmse_georeferenced_m` |
> | Scale error, marker separations | **−0.429 %** | −0.848 % (4 markers) | → `reference_distance_summary.mean_signed_scale_error_pct` |
> | Georef alignment RMSE (pipeline) | 3.908 m | 3.819 m | `accuracy.alignment_rmse_m` |
> | Surface accuracy vs GT | 1.05 m mean | 1.308 m mean, 1.175 m median | → `surface.accuracy_georeferenced_m` |
> | GT area observed | 32.9 % | 32.1 % | → `surface.gt_area_observed_pct` |
> | Wall-clock | 23.4 min | 83.2 min | sum of `stages[*].duration_s` |
> | Texture atlas | 8192 px | 8192 px | `stages.mesh.metrics.texture_size` |
> | Dynamic-object masking | 0.0 % | 0.0 % | `stages.masks.metrics.mean_masked_fraction` |
>
> Also verified: 17-keyframe real-video demo `demo_aukerman_hd` completes in 3.5 min.
>
> **State the tradeoff if you demo the hires run.** Denser sampling bought GSD at
> a cost: trajectory RMSE rose 1.65 → 2.24 m and scale error roughly doubled to
> −0.85 %. Both remain inside the passing thresholds, and saying so is stronger
> than hoping nobody diffs the two scorecards. The measured building dimensions
> also carry a systematic sign: lengths read 0.45–1.44 m **short** while widths
> read slightly long, which is a bias to acknowledge, not noise to average away.

## Overview
- **Problem Statement**: SIH26158 (NTRO) — single-pass drone video to a georeferenced, metrically scaled, textured 3D model.
- **Hardware**: runs end-to-end on an edge laptop GPU (NVIDIA RTX 3050, 6 GB VRAM).
- **Core principle**: the reconstruction core is fully local and needs no network. The Gaussian-splatting tier in Minute 5 is explicitly **off-box and optional** — see the honesty note there.

---

## ⏱ Minute-by-Minute Live Demo Flow

### Minute 1: The Problem & Edge Constraint
- **Speaker Action**: Have the DroneMap Web Studio open on the laptop (`http://127.0.0.1:8000`).
- **Talking Points**:
  > *"Respected Jury Members, conventional photogrammetry pipelines are tuned for multi-strip serpentine grids with high cross-track overlap. A single forward pass gives far weaker geometry, and in tactical reconnaissance that is often all you get — flown by teams with no cloud access.*
  >
  > *We built **DroneMap**: it takes raw single-pass video plus telemetry, screens dynamic objects with YOLOv8-seg, and reconstructs a georeferenced textured 3D model locally. On this laptop the 47-keyframe survey fixture completes in 23 minutes and the 17-keyframe demo run in 3.5; pushed to sub-decimetre sampling the same orbit takes 83. No step of that requires a network."*

**Do not say "under 8 minutes."** Measured runtimes: **23.4 min** (47-keyframe orbit), **25.0 min** (corridor), **83.2 min** (97-keyframe hires orbit), **3.5 min** (17-keyframe real-video demo). Quote the figure for the run you actually demo.

If you demo the hires run, own the cost rather than eliding it: sub-decimetre GSD took 83 minutes, of which 60 was meshing and 20 dense fusion. The honest framing is that the edge box spans a *range* — minutes for a reconnaissance-grade product, over an hour for a survey-grade one — and the operator chooses. Claiming the 23-minute figure while showing the 9.6 cm/px result would be mixing columns.

---

### Minute 2: Interactive 3D Web Studio (Live Model Inspection)
- **Speaker Action**: In the left sidebar, click **`fixture_orbit_hires`** if you intend to claim sub-decimetre GSD, **`fixture_orbit_v4`** if you would rather lead with the tighter trajectory and dimension figures — or `demo_aukerman_hd` for the fast real-video run. Both orbit fixtures badge **Validated 3D**; pick one and stay in its column all the way through Minute 5.
  - Rotate the model with the WebGL orbit controls.
  - Toggle **🎨 Textured** → **⚪ White Clay**, then the **Wireframe** overlay.
- **Talking Points**:
  > *"This is the reconstructed survey scene. In textured mode OpenMVS projects an 8192-pixel texture atlas onto the geometry. Switching to White Clay exposes the bare surface — OpenMVS builds it by Delaunay tetrahedralisation with a graph-cut surface extraction, then refines it photometrically against the images."*

  Read the face count and component counts **off the run you opened**, because they differ:
  - `fixture_orbit_v4`: 201,238 faces; of 8 surface components it kept 1 and discarded 7 fragments.
  - `fixture_orbit_hires`: 176,688 faces; kept 1, discarded 1.

  The hires run has *fewer* faces despite 6.6× the dense points, and that is not a defect — photometric refinement subdivides and re-optimises, taking 738,381 raw faces to 176,692 at higher fidelity. If asked why denser input yields a smaller mesh, that is the answer: face count is not a quality metric.

Corrections against the earlier version of this script:
- The mesher is **not Poisson**. OpenMVS `ReconstructMesh` is Delaunay + graph-cut. (`stage6_mesh.py`'s own docstring says "Poisson-style" and is likewise inaccurate.)
- **Do not claim "no holes, no inverted normals, zero ghosting."** Nothing in the pipeline measures hole count or normal consistency. What *is* recorded is `components_kept: 1`, which you may cite.
- **Do not attribute clean geometry to YOLOv8 masking on the fixtures.** `mean_masked_fraction` is **0.0** on both Blender fixtures — the synthetic scenes contain no moving objects, so nothing was masked and masking cannot be the reason anything looks clean. If you want to demonstrate masking, use `demo_aukerman_hd`, where 2 frames were masked at 0.49 % mean coverage. A criterion that reports 0 % means *there was nothing to remove*, not that the detector is proven.

---

### Minute 3: Live 3D Metric Measurement (Proving Metric Scale)
- **Speaker Action**:
  - Click **📏 Measure** (or press `M`), then click two corners of Building Alpha.
  - Read the HUD live: 3D Euclidean distance, horizontal distance, vertical relief ΔZ, bearing.
- **Talking Points**:
  > *"This is a metric survey, not a visualisation. Building Alpha's ground truth is a 16 × 24 m footprint, and the harness measures it straight off the reconstruction with nothing fitted.*
  >
  > *The independent check is the four ground reference markers: all four recovered, and their reconstructed separations are short by under one percent — with no RTK and no manual control points. Trajectory error against ground truth is measured after geodetic frame changes alone — no scale fit, no yaw fit, no translation fit."*

  Fill in the numbers from the run you opened:

  | | `fixture_orbit_v4` | `fixture_orbit_hires` |
  |---|---|---|
  | Alpha long wall (GT 24 m) | 23.984 m (16 mm short) | 22.56 m (1.44 m short) |
  | Alpha width (GT 16 m) | 16.175 m | 16.19 m |
  | Marker scale error | −0.429 % | −0.848 % |
  | Trajectory RMSE, nothing fitted | 1.646 m | 2.244 m |

  **On the hires run, do not lead with Building Alpha's long wall.** 22.56 m against 24 m is a 1.4 m shortfall, and the dimension table shows the same sign on every structure — lengths short by 0.45–1.44 m, widths slightly long. That is a systematic bias, and the marker check is the honest headline there because it is an independent measurement over 100 m baselines rather than a single wall. On `fixture_orbit_v4` the 16 mm figure is real and you may lead with it.

Two precision points that the earlier script got wrong:
- **Do not say "1 unit equals exactly 1.000 metre."** The measured scale error is −0.429 %; the markers sit on a 100 m grid, so that is about 43 cm over an adjacent pair. Say the measured number — it is a good number, and it is defensible.
- **The model and the GIS rasters are in different frames, deliberately.** The delivered `.glb` is in a **local ENU frame, glTF Y-up**, with its origin offset recorded in `model_frame.json` beside the asset. The DSM/DTM/orthomosaic are in **EPSG:32643 (UTM 43N)**. Vertices are float32, and at ECEF magnitudes float32 spacing is 0.5 m — which is why the deliverable is shifted to a local origin rather than shipped in ECEF. If asked, that is the reason, and it is reversible from the recorded offset.

---

### Minute 4: Standard GIS Deliverables & The Quality Report
- **Speaker Action**:
  - Point to the sidebar **Export Deliverables** buttons.
  - Click **📑 View Full Quality Report (HTML)** → opens `report.html`.
  - Scroll to the KPI cards and the **NTRO Criteria Compliance Matrix (C1–C10)**.
- **Talking Points**:
  > *"Intelligence workflows cannot consume proprietary formats, so export emits ASPRS LAS/LAZ point clouds, a DSM and bare-earth DTM as GeoTIFFs, a true orthomosaic, and the flight path as KML.*
  >
  > *Every run generates this self-contained HTML report. It has zero remote asset references — the viewer's Three.js is vendored, and the report verifies that itself rather than asserting it.*
  >
  > *And this matrix reports four states, not two: pass, partial, fail, and not-attempted. On this run some criteria pass, one fails, and two were not attempted. I want to walk you through the failure, because a matrix that only ever showed green would tell you nothing."*

**This is the most important correction in the script.** The earlier version said the report *"verif[ies] that every single NTRO criterion from C1 to C10 is passed."* That was false and the matrix was hardcoded to ten green rows. It now derives every cell from a recorded measurement and prints the evidence in **every** state, including failure. Be ready to speak to:
- **C2 (sub-decimetre GSD)** — **which run you demo decides whether this passes, so say which one.** At default settings (`fixture_orbit_v4`) it fails honestly: 23.2 cm/px against a 10 cm/px bar. On `fixture_orbit_hires` it passes at **9.59 cm/px**, and the report prints "✓ Measured 9.6 cm/px (sub-decimetre)" from the measurement rather than from a hardcoded row.

  The cause of the original failure is worth explaining, because it is not coarse optics. GSD here is `sqrt(bbox_area / n_dense_points)` — dense-cloud spacing, not an image property — so it improves as `1/sqrt(point count)`. Two settings were discarding samples: keyframe selection kept 47 of 480 frames, and `dense.resolution_level: 1` halved each image in both axes. The fix needs **three** settings, not two:

  ```
  --set frames.sharpness_window=3 --set frames.target_overlap=0.90 --set dense.resolution_level=0
  ```

  `sharpness_window` is not optional there, and this is the interesting part if a judge asks: windowed best-frame selection already spaces candidates `window` frames apart, so the overlap stride can only thin them further. Requesting 90 % overlap computed a stride of 3, but a window of 8 held the real spacing at 8 — the request was silently overridden. Raising `target_overlap` alone provably does nothing. The pipeline now reports the overlap the spacing actually implies, so the override cannot pass unnoticed again.
- **Criteria marked "—" were not attempted**, which is a different claim from failing. Do not let them read as passes.

---

### Minute 5: Ground-Truth Validation (Blender) & the Optional 3DGS Tier
- **Speaker Action**:
  - Open the scorecard for **the run you demoed**: `data/runs/<run_id>/evaluation_scorecard.json`
    *(not `data/synthetic_flight/` — the scorecard is written per-run.)*
  - Optionally show `notebooks/3dgs_splatfacto_kaggle.ipynb`.
- **Talking Points**:
  > *"To validate geometry beyond visual inspection we generate a headless Blender procedural survey world whose structures have analytically exact dimensions, then score the reconstruction against them with a harness that is deliberately separate from the pipeline — the code that builds the model is not allowed to grade it.*
  >
  > *The harness reports accuracy and completeness separately and never averages them, and it quotes completeness only over the fraction of the ground-truth area this single orbit actually observed — about a third. A symmetric figure over the full 300 m ground truth would mostly be measuring where we did not fly."*

  Per-run figures — `fixture_orbit_v4`: surface accuracy 1.05 m mean, over 32.9 % observed. `fixture_orbit_hires`: 1.308 m mean and 1.175 m median, p90 2.401 m, over 32.1 % observed, completeness 1.384 m mean within the observed area.

  If a judge asks why the *denser* run scores slightly worse on accuracy, answer plainly: it does, by about 25 cm in the mean. Denser keyframe sampling bought a 2.4× improvement in GSD and cost some geometric accuracy. The scorecard also prints `chamfer_shape_only_m: 0.36`, and that number must be labelled — it is Sim(3)-fitted, so it describes *shape* after scale and pose are fitted away, and it is not an accuracy claim.

Honesty notes for this minute:
- **The 3DGS tier is off-box and optional.** It trains on Kaggle T4 GPUs — that is a cloud dependency and it contradicts the air-gap claim if you present it as part of the core. Present it as an optional enhancement that the offline core does not depend on, or omit it.
- **Do not claim monocular depth priors are in play.** `stage4_depth.py` has never executed in any recorded run (`depth: pending` in every manifest) and has known open defects. It is code on disk, not a demonstrated capability.
- **Drop "Survey of India compliant."** Nothing in the repository substantiates a Survey of India conformance claim. What is true and checkable: outputs carry a WGS84 datum and project to EPSG:32643.

---

## 🛡 Anticipated Jury Questions & Technical Counters

#### Q1: "How do you handle feature matching on a single pass without loops?"
> **Answer**: *"Sequential matching with quadratic overlap windowing — `--SequentialMatching.overlap 10` with `quadratic_overlap` enabled, so each frame matches its temporal neighbours and then progressively wider power-of-two offsets, which recovers some long-range constraints without a full exhaustive pass. Loop detection is configured but ran disabled on these fixtures because no vocabulary tree was present; the logs show `--SequentialMatching.loop_detection 0`."*

*Corrected from the earlier script, which said overlap 8 and "t−8 to t+8". The value is 10, and quadratic overlap means the window is not a flat ±N.*

#### Q2: "What is your camera model? Don't you hit focal-length ambiguity on nadir?"
> **Answer**: *"We use COLMAP's `SIMPLE_RADIAL` with `single_camera` enabled, and we do **not** pass a focal prior — COLMAP initialises focal length from EXIF where present and otherwise from its default heuristic. Focal/depth ambiguity on near-nadir capture is a real limitation we have not eliminated; what we do instead is *detect* it. The capture-quality gate measures median triangulation angle, baseline-to-depth ratio and per-view depth dynamic range, and downgrades a near-nadir strip to a 2.5D terrain product rather than shipping a confident-looking 3D model. The corridor fixture is exactly that case: it returns `terrain_2_5d`."*

*The earlier script claimed a "calibrated pinhole focal prior" that "completely bypass[es]" the ambiguity. No such prior is passed anywhere in the pipeline.*

#### Q3: "What happens when GPS is jammed or unavailable?"
> **Answer**: *"There is a `--no-telemetry` path that reconstructs in a relative metric frame. Be clear about what that costs: without telemetry the result is **up to scale and unreferenced**, so it is badged 'Relative/local' in the UI, and absolute-accuracy criteria are reported as not-attempted rather than passed. Recovering true scale then needs either a known baseline or ground control."*

#### Q4: "Can this run on tactical edge devices with no internet?"
> **Answer**: *"Yes, and the report checks it rather than claiming it. COLMAP 4.1.1 and OpenMVS 2.4.0 binaries, the YOLOv8s-seg weights, and the viewer's Three.js are all vendored in the repository. The export stage scans the shipped viewer for remote `src`/`href`/`@import`/module-import URLs and reports any it finds; on this build it reports none. If someone later adds a CDN `<script>`, that criterion turns red instead of the demo silently breaking on an airgapped host."*

#### Q5: "Your trajectory RMSE is 1.65 m but the pipeline reports 3.91 m. Which is it?"
> **Answer**: *"Both, and the difference is informative. 3.91 m is the pipeline's own alignment residual against the **noisy GNSS fixes** it was given (σ ≈ 1.8 m horizontal, 3.2 m vertical). 1.65 m is the error against **ground truth**. The reconstruction is more accurate than the GNSS it was aligned to, because bundle adjustment averages the fix noise down. We report the pessimistic number in the product and the true number in the harness."*

*Figures above are `fixture_orbit_v4`. On `fixture_orbit_hires` the same pair reads 3.82 m against GNSS and 2.24 m against ground truth — the relationship holds, the reconstruction still beats its own control.*

#### Q6: "You claim sub-decimetre GSD. Is that a real ground sampling distance?"
> **Answer**: *"Not in the optical sense, and we label it for what it is. The reported figure is `sqrt(bbox_area / n_dense_points)` — the mean spacing of the reconstructed dense cloud, which is a **point-density** measure, not an image-resolution one. On the hires run that is 9.59 cm. We report it because dense-cloud spacing is what bounds the achievable detail of the mesh and the derived rasters, but if the criterion is meant strictly as optical GSD then the honest answer is that we measure a related quantity and name it in the report. We would rather be precise about which number we computed than claim a stronger one."*

*This is the question most likely to expose an overclaim, so answer it before it is asked if C2 comes up.*
