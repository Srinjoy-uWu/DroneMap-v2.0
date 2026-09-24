# Incremental AI Enhancement Roadmap — DroneMap (SIH26158)

This roadmap defines the disciplined, step-by-step upgrade strategy for integrating verified AI intelligence into the DroneMap pipeline while guaranteeing zero regression of existing functionality.

---

## 1. Upgrade Methodology: Test-Driven Incremental Evolution

Every improvement follows the strict engineering lifecycle:

```text
  [Existing Working Stage]
             │
             ▼
  [Benchmark & Record Baseline]
             │
             ▼
  [Implement Target Enhancement (Isolated Module)]
             │
             ▼
  [Automated Unit & Integration Test Suite]
             │
             ▼
  [Empirical Comparison Against Baseline (Runtime, Registration, Accuracy)]
             │
             ▼
  [Retain & Integrate IF Proven Beneficial / Revert IF Inferior]
```

---

## 2. Enhancement Milestones

### Milestone 1: Multi-Factor AI Frame Intelligence
- **Current State**: Stage 1 uses Laplacian sharpness in a window and optical flow magnitude for stride.
- **Target Enhancement**:
  - Implement a comprehensive multi-factor quality analyzer calculating:
    1. **Sharpness ($S$)**: Variance of Laplacian $\sigma^2(\nabla^2 I)$.
    2. **Exposure ($E$)**: Normalised histogram distribution penalizing under/over-exposed frames.
    3. **Contrast ($C$)**: Michelson / RMS contrast metric.
    4. **Perceptual Information Entropy ($H$)**: Shannon entropy of edge gradients.
    5. **Redundancy Distance ($D$)**: Lightweight structural difference against previously selected keyframe.
  - Generates detailed per-frame diagnostic records in `keyframes.json` with actual measurable figures.
- **Success Criteria**:
  - Reduces redundant keyframes by $\ge 25\%$ on hovering/panning footage.
  - Achieves $100\%$ camera registration in downstream SfM with fewer input images.
  - Complete execution within the existing $O(1)$ constant-memory streaming footprint.

### Milestone 2: Adaptive Dynamic Object & Shadow Masking
- **Current State**: Stage 2 evaluates YOLOv8s-seg on candidate keyframes with a fixed 12 px dilation kernel.
- **Target Enhancement**:
  - Implement velocity-adaptive dilation: scale dilation radius dynamically according to camera ground speed and optical flow magnitude at object boundaries.
  - Add transient shadow suppression heuristics to prevent dark moving cloud shadows from corrupting static ground features.
  - Export inspection artifacts: side-by-side visualization of original frame vs dynamic mask overlay for UI inspection.
- **Success Criteria**:
  - Eliminates dynamic vehicle/pedestrian floaters from point clouds.
  - Retains static parked cars and stationary ground infrastructure without false-positive over-masking.

### Milestone 3: Dual-Engine Robust Feature Matching
- **Current State**: Stage 3a relies exclusively on COLMAP GPU SIFT and sequential matching, with MASt3R as an external heavy backend.
- **Target Enhancement**:
  - Implement a hybrid matching bridge: GPU SIFT remains the default for standard consecutive frames; SuperPoint + LightGlue is invoked for low-contrast/low-texture frame pairs (asphalt, sand, water boundaries) or steep oblique transitions.
  - Preserves COLMAP's SQLite database format (`colmap.db`), ensuring seamless compatibility with COLMAP's bundle adjuster and mapper.
- **Success Criteria**:
  - Increases registered camera count on difficult / low-texture drone sequences where SIFT yields $< 50$ matches.
  - Maintains execution time within reasonable bounds on a 6GB VRAM budget.

### Milestone 4: Monocular Depth Fusion & Occlusion Confidence
- **Current State**: Stage 4a runs Depth Anything V2 / Metric3D, but output is stored solely as a secondary raster.
- **Target Enhancement**:
  - Fuse monocular depth priors with sparse SfM points to establish metric scale bounds and filter outlier floaters.
  - Generate a 3D **Regional Confidence Map**:
    - **Class 1 (Observed - High Confidence)**: Multi-ray optical triangulation ($\ge 3$ rays, angle $\ge 5^\circ$).
    - **Class 2 (Estimated - Medium Confidence)**: Dense multi-view stereo interpolation with verified photo-consistency.
    - **Class 3 (Inferred - Low Confidence)**: Terrain plane interpolation or monocular depth prior across occluded gaps.
  - Export confidence attributes embedded directly in the LAZ point cloud and GLB vertex attributes.
- **Success Criteria**:
  - Clear, honest distinction between directly measured physical geometry and interpolated/inferred surface geometry.

### Milestone 5: Full Pipeline Transparency & Verifiable Audit
- **Current State**: Web UI displays final 3D model and basic download links.
- **Target Enhancement**:
  - Provide an interactive stage-by-stage inspection drawer in the web studio:
    - Stage 1: Extracted keyframes with individual sharpness and exposure scores.
    - Stage 2: Dynamic object masks and dropped-frame reasons.
    - Stage 3: Camera trajectory with registered camera status.
    - Stage 4: Depth maps and confidence overlays.
    - Stage 5 & 6: 3D dense point cloud, watertight mesh, and texture atlas.
    - Stage 7: Survey audit report with alignment residuals, GSD, and spatial accuracy.
- **Success Criteria**:
  - Fully transparent, defensible presentation suitable for smart city planning, defense intelligence, and hackathon jury verification.

### Milestone 6: High-Throughput Cloud & Kaggle Acceleration (Completed)
- **Implemented State**:
  - Created `configs/accelerated.yaml` profile unlocking native 4K processing, large foundation models (Depth Anything V2 Large, Metric3D ViT-Large), dense MVS, and 16K texture atlases.
  - Provided Kaggle dual-T4 execution notebook (`notebooks/3dgs_splatfacto_kaggle.ipynb`) and `notebooks/sih26158_3dgs_kaggle.py` for 3D Gaussian Splatting (`splatfacto`).
  - Automated cloud bundle packaging and re-import into the local DroneMap GIS workspace.
- **Success Criteria**:
  - Enables zero-compromise ultra-high-resolution photogrammetry and radiance field training on free cloud GPUs (30 hr/week).

### Milestone 7: SIH26158 Hierarchical Semantic Classification Layer (Active)
- **Implemented State**:
  - Implemented `_map_to_sih26158_categories()` in `stage4_semantics.py` to aggregate raw SegFormer / UAVid classes into the 4 mandatory SIH26158 categories:
    1. **Terrain / Bare Earth**: Ground, dirt, bare soil, sand, low vegetation, grass.
    2. **Buildings / Structures**: Buildings, roofs, houses, walls, fences, constructions.
    3. **Roads & Infrastructure**: Roads, pavement, sidewalks, bridges, runways, tarmac, railways.
    4. **Vegetation & Obstacles**: Trees, dense canopy, clutter, vehicles, humans, water bodies.
  - Automatically auto-verifies domain if an aerial/drone checkpoint is specified in config (`uavid`, `loveda`, `isprs`, `potsdam`, `aerial`, `drone`).
  - Exports `04_semantics/sih26158_categories.json` alongside per-class `class_fractions.json` and label PNGs.
- **Success Criteria**:
  - 100% direct compliance with Smart India Hackathon SIH26158 problem statement deliverables.

---

## 3. Ablation Study Plan

To scientifically validate the exact contribution of each AI component, the following ablation matrix will be benchmarked on identical test footage:

| Configuration | Frame Selection | Dynamic Masking | Feature Matching | Depth Fusion | Quality Gating | Primary Evaluation Metrics |
|---|---|---|---|---|---|---|
| **A: Baseline** | Windowed Laplacian | Disabled | Native GPU SIFT | Disabled | Count-based | Baseline runtime, points, faces, registration % |
| **B: + Frame Intelligence** | Multi-Factor AI | Disabled | Native GPU SIFT | Disabled | Count-based | Keyframe reduction %, feature efficiency, runtime |
| **C: + Dynamic Masking** | Multi-Factor AI | YOLOv8s-seg + Adaptive Dilation | Native GPU SIFT | Disabled | Count-based | Dynamic artifact elimination, point cloud cleanliness |
| **D: + Learned Matching** | Multi-Factor AI | YOLOv8s-seg | SIFT + LightGlue Hybrid | Disabled | Count-based | Registration % on low-texture & high-angle turns |
| **E: + Depth & Confidence**| Multi-Factor AI | YOLOv8s-seg | SIFT + LightGlue Hybrid | Depth Anything V2 + Confidence | Full Geometric Gate | Confidence map completeness, vertical accuracy |
| **F: Full Unified Pipeline**| Multi-Factor AI | YOLOv8s-seg | SIFT + LightGlue Hybrid | Depth Anything V2 + Confidence | Full Geometric Gate + 2.5D Fallback | End-to-end runtime, GSD, mesh quality, survey compliance |
