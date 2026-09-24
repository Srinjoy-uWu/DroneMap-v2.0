# AI & LLM API Integration Specification
## Enhancing DroneMap v2.0 with Foundation Models, Autonomous Agents, and Cloud APIs

---

## 1. Executive Summary & Problem Statement Compliance

Smart India Hackathon problem statement **SIH26158** (*Single-Pass Drone Video to Georeferenced, Metrically Accurate 3D Model*, owned by NTRO) mandates strict metric accuracy, closed-form geodesy, and operational robustness.

When considering the integration of **Large Language Models (LLMs)**, **Multimodal Vision Models**, and **External AI APIs**, we must strictly distinguish between:
1. **Valid, High-Impact AI Enhancements**: Using AI to analyze, cross-validate, diagnose, summarize, query, and optimize photogrammetry without falsifying geometry.
2. **Disqualifying Anti-Patterns**: Using single-image generative diffusion APIs (such as Meshy or Tripo3D) that hallucinate fictitious unobserved geometry with zero metric scale and zero epipolar verification.

This specification defines the **Hybrid Dual-Engine Architecture**:
- **Offline Physics Core**: 100% functional without internet or API keys (using local CUDA models: YOLOv8s-seg, SIFT/LightGlue, Depth Anything V2 Small, SegFormer, OpenMVS, KDTree IDW).
- **AI & Cloud API Booster Layer**: Activated when API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `REPLICATE_API_KEY`) are present, unlocking autonomous geospatial intelligence, self-healing pipeline diagnostics, natural language 3D spatial querying, and remote cloud GPU acceleration.

---

## 2. Four High-Impact Solutions for AI & LLM APIs

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         DroneMap v2.0 Execution Core                        │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌──────────────────────────────┐              ┌────────────────────────────────┐
│   Local Air-Gapped Mode      │              │   AI & LLM API Booster Layer   │
│   (Zero Internet / No Keys)  │              │   (Optional API Keys Provided) │
├──────────────────────────────┤              ├────────────────────────────────┤
│ - YOLOv8s-seg on CUDA        │              │ 1. Autonomous Geospatial Agent │
│ - GPU SIFT / LightGlue       │              │    (Gemini 2.5 / GPT-4o /      │
│ - Depth Anything V2 (Small)  │              │    Claude 3.5 Sonnet)          │
│ - SegFormer-B0 4-class rollup│              │ 2. Self-Healing Photogrammetry │
│ - Sim(3) Geodesy             │              │    Diagnostician               │
│ - OpenMVS & 2.5D Terrain     │              │ 3. Natural Language 3D Chat    │
│ - Offline Three.js Viewport  │              │    in Web Measurement Studio   │
│                              │              │ 4. Cloud Foundation Vision API │
│                              │              │    (Depth Anything Large /     │
│                              │              │    Metric3D ViT-Large remote)  │
└──────────────────────────────┘              └────────────────────────────────┘
```

---

### Solution 1: Autonomous Geospatial Intelligence & Mission Report Agent
- **Target APIs**: Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-pro`), OpenAI API (`gpt-4o` / `gpt-4o-mini`), Anthropic API (`claude-3-5-sonnet`).
- **How It Works**:
  1. Once the photogrammetry pipeline completes, the agent collects:
     - High-resolution keyframes from `01_frames/images/`.
     - Drone flight telemetry (lat/lon coordinates, altitude profile, gimbal pitch, velocity).
     - Semantic land-use distribution from `04_semantics/sih26158_categories.json`.
     - Reconstruction metrics from `manifest.json` (camera registration rate, GSD, RMSE, terrain elevation relief).
  2. The agent executes a structured multimodal prompt with schema enforcement (`pydantic`):
     - **Strategic Terrain & Trafficability Assessment**: Analyzes ground condition, slope steepness, obstacle density for wheeled vs. tracked tactical vehicles.
     - **Critical Infrastructure Audit**: Identifies road pavement quality, bridge integrity, powerline corridors, and roof structural features.
     - **Vegetation Canopy & Blind Spot Advisory**: Evaluates tree cover density and explicitly highlights occluded terrain where ground truth cannot be verified from single-pass overhead view.
  3. **Deliverable**: Automatically compiles a formal defense-grade intelligence dossier exported as `07_export/geospatial_intelligence_report.pdf` and rendered into the interactive Web Studio.

---

### Solution 2: Self-Healing Photogrammetry Diagnostician (Autonomous Auto-Tuner)
- **The Problem**: In real drone missions, high winds, rapid turns, or featureless terrain (asphalt, sand) cause camera tracking to fail or bundle adjustment reprojection error to spike. Non-expert drone operators do not know how to tune hyper-parameters like SIFT octaves, peak thresholds, or matching overlap.
- **The Solution**:
  - An intelligent diagnostic loop `dronemap.diagnostics`:
    1. If Stage 3a registers $< 60\%$ of cameras or OpenMVS dense MVS fails, the diagnostician inspects the stage logs (`logs/colmap_mapper.log`, `logs/openmvs_densify.log`).
    2. Sends the mathematical summary to the LLM agent via API:
       ```json
       {
         "n_images": 35,
         "registered_cameras": 14,
         "average_triangulation_angle_deg": 3.1,
         "failed_pairs_cluster": [8, 9, 10, 11],
         "reason": "Low inlier count across rapid yaw maneuver (delta heading: 32 deg)"
       }
       ```
    3. The LLM returns an actionable parameter override dictionary:
       - Escalate to LightGlue neural matching on frame cluster 8–12.
       - Expand sequential overlap from 10 to 18 frames.
       - Lower SIFT peak threshold from 0.0066 to 0.0040.
    4. The pipeline automatically re-runs the affected stage with the optimized parameters, achieving 100% registration without human intervention.

---

### Solution 3: Natural Language 3D Spatial Querying & Measurement Assistant ("Chat with Your 3D Model")
- **The Problem**: Standard photogrammetry viewers only offer manual clicking tools. Hackathon juries and defense officers want instant situational answers.
- **The Solution**:
  - An interactive AI Chat Drawer integrated directly into the Three.js Web Measurement Studio (`api/static/`).
  - Powered by Function Calling / Tool Use via Gemini or OpenAI API:
    - User types: *"What is the height of the large building in the center of the scene?"*
    - The LLM calls backend endpoint: `POST /api/runs/{id}/query/height` with spatial coordinates.
    - Viewport automatically positions the camera, draws an interactive 3D dimension line in Three.js, and responds: *"The building at UTM (E: 721540m, N: 3163480m) has an estimated height of 14.8 meters above ground level."*
    - User types: *"Show me areas where moving vehicles were removed."*
    - The LLM toggles the dynamic mask overlay layer on the 3D model, visually highlighting the masked zones in red.

---

### Solution 4: Cloud Vision Foundation Model APIs (Offloading 6GB Edge VRAM)
- **Target APIs**: Replicate API, HuggingFace Inference Endpoints, or custom serverless GPU worker (RunPod / Kaggle API).
- **The Problem**: Foundation models like **Depth Anything V2 Large** (335M params) and **Metric3D ViT-Large** provide superior boundary crispness and sub-centimeter metric depth, but consume 8–14 GB VRAM, exceeding local RTX 3050 6GB capacity.
- **The Solution**:
  - If `REPLICATE_API_KEY` or `HF_API_KEY` is provided:
    - Stage 4a dispatches keyframe batches to the cloud foundation model API.
    - Receives full-resolution $3840 \times 2160$ metric depth maps and surface normals in $< 15\text{ seconds}$ without using local laptop VRAM.
  - If no API key is provided:
    - Pipeline automatically falls back to local `Depth Anything V2 Small` running on CUDA FP16 locally within 1.2 GB VRAM.

---

## 3. What Must NOT Be Done: The Generative 3D Fallacy

> [!CAUTION]
> **Strict Guardrail Against Hallucinatory Generative 3D APIs**:
> - Commercial generative APIs (e.g. Meshy, Tripo3D, CSM, Luma Genie, Stable Fast 3D) generate single-image 3D meshes using generative diffusion priors.
> - **Why they violate SIH26158**:
>   1. They invent geometry where none exists (hallucinating rear walls, windows, and ground relief).
>   2. They have **no epipolar mathematical constraint** to source video frames.
>   3. They lack **metric scale** (output is normalized to $[-1, 1]$ bounding box) and cannot be assigned real WGS-84/UTM georeferencing.
>   4. If presented to NTRO, the system would fail the $\le 2\%$ metric accuracy and ground-truth checkpoint criteria.
> - **Conclusion**: Generative diffusion meshes will **never** replace the photogrammetric reconstruction core in DroneMap. All 3D geometry is strictly derived from bundle adjustment rays and watertight KDTree IDW ground fitting.

---

## 4. Implementation Architecture: `dronemap.agent`

To keep the codebase modular, testable, and robust, all API and LLM capabilities will be encapsulated in a dedicated package:

```text
src/dronemap/agent/
├── __init__.py
├── config.py             # Agent settings (API keys, model selection: gemini-2.5-flash / gpt-4o)
├── intelligence.py       # Strategic terrain assessment & tactical report generation
├── diagnostics.py        # Self-healing SfM log analyzer and auto-tuner
├── tools.py              # Function calling tools (distance, height, semantics, confidence query)
└── client.py             # Resilient API client with exponential backoff & air-gapped fallback
```

### Resilient Fallback Pattern:
Every agent call is wrapped to guarantee zero-breakage:
```python
def generate_intelligence_report(workspace: RunWorkspace) -> IntelligenceReport:
    """Generates tactical report via LLM API if key is present; returns rule-based summary if offline."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("No LLM API key detected. Using offline deterministic intelligence engine.")
        return generate_offline_heuristic_report(workspace)
    
    try:
        return call_multimodal_llm_agent(workspace, api_key)
    except Exception as e:
        logger.warning(f"LLM API call failed ({e}). Falling back gracefully to offline report.")
        return generate_offline_heuristic_report(workspace)
```

---

## 5. Summary of Benefits for SIH26158 Hackathon

1. **Jury Wow Factor**: An interactive 3D Web Studio where judges can chat with the drone reconstruction in natural language and receive instant 3D dimensional measurements.
2. **Defensible Scientific Stance**: We maintain 100% physical truth for geometry, while leveraging state-of-the-art LLMs for what they do best: reasoning, synthesis, anomaly detection, and autonomous self-healing.
3. **True Dual-Mode Operation**: Fully functional on an air-gapped military field laptop without internet, but instantly supercharged when connected to cloud API services.
