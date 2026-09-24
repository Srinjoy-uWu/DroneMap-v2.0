# End-to-End Implementation Roadmap for DroneMap v2.0
## Engineering Strategy, Actionable Milestones, and Verification Protocols

---

## 1. Context & Workspace Strategy for DroneMap v2.0

The user's goal is:
> **"You need to make this project fully working, the problem statement is SIH26158, get all the context, add research and anything relevant which will improve the solution in that folder, and don't make any changes for now, plan first."**

### Current State Analysis:
- The workspace opened by the user is `c:\Users\srinj\Desktop\DroneMap - v2.0`.
- The existing codebase resides in `c:\Users\srinj\Desktop\SIH26158`, with all 157 regression tests passing cleanly.
- `DroneMap - v2.0` represents the next-generation iteration (v2.0) of the DroneMap platform.
- To make `DroneMap - v2.0` a fully working, self-contained, reproducible project, we must plan the exact code and asset migration from `SIH26158` into `DroneMap - v2.0` (including `src/`, `configs/`, `tests/`, `tools/`, `notebooks/`, `pyproject.toml`, and `.venv`), followed by implementing the prioritized v2.0 upgrades while upholding the strict zero-regression policy.

---

## 2. Prioritized Technical Upgrades for v2.0

### Upgrade 1: LightGlue Neural Matcher Bridge (`matching_neural.py`)
- **Problem**: SIFT feature matching struggles on low-texture surfaces common in drone flights: asphalt highways, uniform concrete roofs, water bodies, and uniform agricultural fields.
- **Solution**:
  - Implement `src/dronemap/matching_neural.py` integrating SuperPoint keypoint extraction + LightGlue graph transformer matching.
  - Follow production conventions from [cvg/Hierarchical-Localization (hloc)](https://github.com/cvg/Hierarchical-Localization) to write keypoints and two-view geometric matches directly into COLMAP's SQLite database (`colmap.db`).
  - Activate the neural matcher conditionally:
    1. During sequential matching when consecutive frame pairs yield $< 50$ SIFT inliers.
    2. During steep bank turns where viewing angle change exceeds $25^\circ$.
- **Impact**: Boosts camera registration on challenging single-pass sequences from $65\text{--}80\%$ to $100\%$ without altering COLMAP's downstream bundle adjustment or OpenMVS integration.

---

### Upgrade 2: 3D Regional Confidence Mapping
- **Problem**: Traditional photogrammetry produces a single point cloud or mesh without communicating the reliability of individual regions. An unobserved zone filled by interpolation looks identical to a surface measured by 10 camera rays.
- **Solution**:
  - Calculate a rigorous per-point and per-vertex confidence classification:
    - **Class 1 (Observed - High Confidence)**: Point triangulated by $\ge 3$ intersecting camera rays with median triangulation angle $\theta \ge 5^\circ$.
    - **Class 2 (Estimated - Medium Confidence)**: Point produced by dense multi-view stereo interpolation with verified multi-view photo-consistency ($2^\circ \le \theta < 5^\circ$).
    - **Class 3 (Inferred - Low Confidence)**: Point generated via 2.5D terrain IDW interpolation or monocular depth prior across an occluded void.
  - Embed confidence classifications directly into:
    1. `cloud.laz`: Encoded in ASPRS Point Data Record format (classification field or ExtraBytes scalar).
    2. `model.glb`: Stored as vertex color layers and custom vertex attributes accessible by WebGL shaders.
- **Impact**: Provides defense-grade honesty and traceability. Surveyors and tactical operators know exactly which surfaces are physically measured vs. interpolated.

---

### Upgrade 3: Interactive Stage Drawer in Web Measurement Studio
- **Problem**: The current web studio shows the final 3D model and basic download links, but hides intermediate pipeline intelligence (keyframes, dynamic masks, trajectory, depth maps, semantic classes).
- **Solution**:
  - Extend `src/dronemap/api/server.py` with REST endpoints:
    - `/api/runs/{id}/stages`: Stage-by-stage timings, metrics, and logs.
    - `/api/runs/{id}/keyframes`: Gallery of selected frames with sharpness, exposure, and mask badges.
    - `/api/runs/{id}/previews/{stem}`: Side-by-side original image vs. red dynamic mask overlay.
    - `/api/runs/{id}/semantics`: Interactive chart of SIH26158 category distributions.
  - Upgrade `viewer.js` and `index.html`:
    - Collapsible "Pipeline Inspection Drawer" on the right side of the viewport.
    - Shading Switcher: **Full Texture**, **Analytical Clay Surface**, and **3D Confidence Heatmap** (Green = Observed, Yellow = Estimated, Blue = Inferred).
    - 100% offline, air-gapped Three.js r168 implementation with zero external CDN dependencies.
- **Impact**: Unmatched presentation appeal for hackathon jury evaluation and industrial stakeholders. Proves every stage of the pipeline with visual transparency.

---

### Upgrade 4: Near-Real-Time Progressive Streaming (`ChunkConfig`)
- **Problem**: Waiting 30–45 minutes for full OpenMVS dense MVS before getting any 3D deliverable is unacceptable for tactical first-responders.
- **Solution**:
  - Wire up the existing `ChunkConfig` abstraction in `pipeline.py`:
    - As keyframes are extracted in Stage 1, partition the flight into temporal chunks of $M = 15\text{--}25$ frames with $30\%$ chunk overlap.
    - Run fast sparse SfM and immediate 2.5D terrain meshing on Chunk 1, outputting an initial georeferenced 3D model within the **first 60 seconds** of video ingestion.
    - As subsequent chunks complete, merge elevation grids and update the live Web Studio model progressively.
    - Run full OpenMVS dense stereo asynchronously in the background for analytical refinement.
- **Impact**: Solves the near-real-time requirement of SIH26158, delivering immediate situational awareness in < 60 seconds while preserving survey-grade accuracy in background refinement.

---

## 3. Phased Execution Plan (Once Approved)

### Phase 0: Workspace Setup & Clean Migration
1. Replicate the working codebase structure from `C:\Users\srinj\Desktop\SIH26158` into `DroneMap - v2.0`:
   - Copy `src/`, `configs/`, `tests/`, `tools/`, `notebooks/`, `pyproject.toml`, `.gitignore`, `yolov8s-seg.pt`.
   - Configure `.venv` virtual environment in `DroneMap - v2.0` with Python 3.11.
2. Execute baseline regression suite:
   ```powershell
   pytest -q
   ```
   **Pass Criteria**: All 157 tests pass with 0 failures in `DroneMap - v2.0`.

### Phase 1: Multi-Factor AI Frame Intelligence & Adaptive Masking
1. Upgrade `src/dronemap/stage1_frames.py`:
   - Implement multi-factor quality scoring: Sharpness ($S$), Exposure ($E$), Contrast ($C$), and Perceptual Uniqueness ($U$).
   - Record per-frame metrics into `keyframes.json`.
2. Upgrade `src/dronemap/stage2_masks.py`:
   - Implement optical-flow-driven velocity-adaptive dilation: $r_{\text{dilate}} = 12 + 0.5 \cdot \|\mathbf{v}\|$.
   - Export preview overlay images (`02_masks/previews/<stem>_overlay.jpg`).
3. Author unit tests in `tests/test_ai_frames.py`.

### Phase 2: Neural Matcher Bridge (SuperPoint + LightGlue)
1. Create `src/dronemap/matching_neural.py`:
   - Wrap LightGlue PyTorch model with automatic CUDA device detection.
   - Implement `hloc`-compliant SQLite writer for COLMAP `colmap.db`.
2. Update `src/dronemap/stage3_pose.py`:
   - Hook neural matcher into the escalation ladder when sequential SIFT yields $< 50$ matches.
3. Author unit tests in `tests/test_neural_matcher.py`.

### Phase 3: Monocular Depth Fusion & 3D Regional Confidence Mapping
1. Update `src/dronemap/stage4_depth.py`:
   - Implement telemetry altitude scaling: $Z_{\text{metric}} \approx h / \cos\theta$.
2. Update `src/dronemap/stage5_dense.py` and `stage7_export.py`:
   - Tag dense points and mesh vertices with confidence classes (Observed / Estimated / Inferred).
   - Write confidence classifications to `07_export/cloud.laz` and `model.glb`.
3. Author unit tests in `tests/test_confidence.py`.

### Phase 4: Air-Gapped Web Measurement Studio Enhancements
1. Update `src/dronemap/api/server.py`:
   - Add `/api/runs/{id}/stages`, `/api/runs/{id}/keyframes`, `/api/runs/{id}/previews/{stem}`.
2. Update `src/dronemap/api/static/viewer.js` & `index.html`:
   - Implement collapsible Pipeline Drawer.
   - Implement Confidence Heatmap shader.
3. Author integration tests in `tests/test_web_drawer.py`.

### Phase 5: Autonomous AI & LLM Copilot (`dronemap.agent`)
1. Create `src/dronemap/agent/` package:
   - `intelligence.py`: Multimodal tactical survey report generator using Gemini / OpenAI API with fallback to deterministic heuristic report.
   - `diagnostics.py`: Self-healing log analyzer that detects camera drops, yaw spikes, and suggests/applies parameter overrides.
   - `tools.py`: Spatial tool wrappers for 3D measurement, coordinates, and confidence queries.
2. Web Studio Natural Language Chat Drawer:
   - Add AI chat interface to `viewer.js` allowing interactive natural language queries ("What is the height of building X?", "Show dynamic mask areas").
3. Author unit tests in `tests/test_agent.py` asserting graceful offline degradation when API keys are absent.

### Phase 6: Near-Real-Time Progressive Streaming (`ChunkConfig`)
1. Implement chunked sliding-window execution in `src/dronemap/pipeline.py`.
2. Output fast 2.5D preview model in $< 60\text{ seconds}$ while background MVS processes.
3. Author regression test in `tests/test_streaming.py`.

### Phase 7: System Verification, Benchmarking, and Walkthrough
1. Run full regression test suite (`pytest -q`, targeting > 170 passed tests).
2. Execute end-to-end flight tests on real UAV video (`0904.mp4`, `flight.mp4`, `flight_aukerman.mp4`).
3. Generate comprehensive `walkthrough.md` with benchmark timings, GSD metrics, and visual screenshots.

---

## 4. Verification Protocol & Acceptance Criteria

| Stage / Component | Verification Command / Method | Strict Acceptance Threshold |
|---|---|---|
| **Regression Suite** | `pytest -q` | 100% tests pass (0 failures, 0 errors) in $< 30\text{ s}$. |
| **System Diagnostics** | `dronemap doctor` | All tools (COLMAP CUDA, OpenMVS, FFmpeg, PyTorch CUDA) verified. |
| **Real Drone Orbit** | `dronemap run --video data/test_assets/orbit_flight.mp4 --telemetry ...` | 100% cameras registered; watertight GLB mesh produced. |
| **Georef Accuracy** | `dronemap run --video ... --telemetry ...` | GPS alignment RMSE $< 5.0\text{ m}$; UTM CRS correctly assigned. |
| **Dynamic Masking** | Inspect `05_dense/scene_dense.ply` | Zero floating vehicles or pedestrian ghost artifacts. |
| **Texture Integrity** | `test_texture_coverage.py` | Texel black-fraction $< 5\%$; zero dark or striped models. |
| **Air-Gapped Studio** | Launch `dronemap serve` & open browser | 3D viewport renders GLB, metric ruler functions, 0 CDN network requests. |
