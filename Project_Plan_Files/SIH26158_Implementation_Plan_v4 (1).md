# SIH26158 — Single-Pass Drone Video to Accurate 3D Model Generation System
## Project Implementation Plan (v4 — Guaranteed/Stretch Tiering, 3–4 Person Build, Zero Paywall)

**Organisation:** National Technical Research Organisation (NTRO)
**Theme:** Robotics and Drones | **Category:** Software
**Team:** 3–4 people, building primarily with AI coding assistance
**Local hardware:** RTX 3050 Laptop GPU, 6GB VRAM

---

## 0. Change Log (v3 → v4)

| # | Change |
|---|---|
| 1 | **Removed Colab Pro entirely** — it's paid, and you asked for zero paywall. Replaced with a multi-account free-Kaggle strategy |
| 2 | Added an explicit **Guaranteed Core vs. Stretch Backlog** split — this is the actual mechanism that prevents total failure, not a promise that nothing will ever break |
| 3 | Rescoped for a **3–4 person team** instead of 6 — task assignment consolidated, and the "build everything" ambition from the 6-person plan is deliberately trimmed down |
| 4 | Rechecked every added repo link this round (OpenMVS, evo) directly via search; full list of previously-verified links carried forward |
| 5 | Added **concrete, copy-pasteable CLI commands** for the guaranteed mesh fallback path (OpenMVS) — not just "use this tool," but the actual command sequence |
| 6 | Added a dedicated **paywall-check section** confirming every tool's cost and access requirements explicitly |
| 7 | Added modular stage-by-stage architecture specifications to facilitate rapid parallel development across the team |

---

## 1. Problem Recap

Build an AI system that takes a **single flyover drone video** and reconstructs a **georeferenced, metrically accurate, textured 3D model** — terrain, buildings, roads, vegetation — suitable for visualization, measurement, and analysis. Must handle motion blur, dynamic objects, occlusion, GPS noise, and ideally run near real-time.

---

## 2. How This Plan Avoids Total Failure

You can't eliminate bugs. You *can* eliminate the scenario where a bug in an ambitious component leaves you with nothing to show. Every stage below is split into:

- **Guaranteed Core** — classical, well-documented, boring-but-reliable tools. Build and fully validate this **first**, end-to-end, before touching anything advanced. This alone is a complete, working, judge-able submission.
- **Stretch** — the AI-forward differentiators (MASt3R/VGGT, 3D Gaussian Splatting, SuGaR). Attempt these only after the Guaranteed Core runs start-to-finish. If a stretch component breaks or doesn't finish in time, you fall back to the Core output for that stage — nothing else in the pipeline needs to change.

This is the actual "no single point of failure" design — not a claim that nothing will go wrong, but a structure where something going wrong in the ambitious 60% doesn't take down the other 40% you already know works.

With only 3–4 people, **do not attempt every stretch item** in section 6's tech stack. Treat it as a backlog you pull from in priority order once Core is done, not a checklist you're obligated to finish.

---

## 3. Compute Strategy (Zero Paywall)

| Tier | Covers | Where |
|---|---|---|
| **Local (RTX 3050, 6GB)** | Depth Anything V2 (small), Metric3D v2 (small), SAM 2 (tiny/small), YOLOv8-seg, SegFormer-B0/B1, Real-ESRGAN, COLMAP feature matching, MASt3R on small batches, OpenMVS dense reconstruction (CPU-heavy, not VRAM-heavy) | Every teammate's laptop |
| **Cloud (free)** | 3DGS/SuGaR training, VGGT/DROID-SLAM full-sequence inference | Kaggle Notebooks |

**Kaggle Notebooks** — free, no card required, ~30 GPU-hours/week per account (T4 or P100, 16GB VRAM), sessions up to ~9–12 hours. **Since each of your 3–4 teammates can create their own free Kaggle account, you collectively have ~90–120 GPU-hours/week** if you split heavy jobs across accounts — this is a legitimate way to multiply your free compute without paying for anything.

**Other free options, worth a 5-minute check:**
- **GitHub Student Developer Pack** (education.github.com/pack) — free for verified students, bundles free cloud credits from several providers
- Ask your department at IIIT Bhagalpur about spare GPU workstation time

**No paid fallback is included in this plan.** If Kaggle's combined team quota genuinely runs out before the deadline, scale down: smaller model variants, fewer frames, lower resolution — not a subscription.

---

## 4. Modular Development Strategy

To accelerate pipeline build and validation, each stage is decoupled with strict interface definitions:
- **Interface Decoupling:** Every stage reads from standardized previous stage outputs (`01_frames`, `02_masks`, `03_pose`, etc.) and writes verified metrics into `manifest.json`.
- **Parallel Workstreams:** Core classical reconstruction and AI booster layers can be developed and benchmarked in parallel.
- **Fail-Safe Operation:** If any optional stage is missing or skipped, the downstream stages continue without crashing.

---

## 5. Architecture — Guaranteed Core vs. Stretch

```
┌──────────────────────────┐
│  Drone Video Input          │  (1080p/4K + GPS + flight metadata)
└────────┬──────────────────┘
         ▼
┌──────────────────────────┐
│ Stage 1: Preprocessing      │  [CORE, LOCAL]
│ - Frame sampling             │
│ - Sharpness-score filter     │
└────────┬──────────────────┘
         ▼
┌──────────────────────────┐
│ Stage 2: Dynamic Masking     │  [CORE, LOCAL]
│ - YOLOv8-seg (fast, reliable) │
│ - [Stretch] SAM 2 for quality │
└────────┬──────────────────┘
         ▼
┌──────────────────────────────────────────┐
│ Stage 3: Pose + Geometry                    │
│ [CORE, LOCAL]  COLMAP — classical, always    │
│                 works, your safety net        │
│ [STRETCH, LOCAL small-batch / CLOUD full]     │
│                 MASt3R or VGGT — better fit    │
│                 for single-pass footage         │
│ [CORE, LOCAL]  evo — align trajectory to GPS   │
│                 (Umeyama) + accuracy metrics     │
└────────┬───────────────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ Stage 4a: Depth (parallel)          │  [STRETCH, LOCAL] Depth Anything V2 / Metric3D v2
│ Stage 4b: Segmentation (parallel)   │  [CORE, LOCAL] SegFormer — verify aerial-domain checkpoint
└────────┬───────────────┬───────────┘
         ▼               ▼
┌──────────────────────────────────┐
│ Stage 5: Reconstruction              │
│ [CORE, LOCAL]  OpenMVS dense point    │
│                 cloud from COLMAP       │
│ [STRETCH, CLOUD] 3D Gaussian Splatting  │
│                 (Nerfstudio splatfacto)  │
└────────┬──────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ Stage 6: Mesh Extraction             │
│ [CORE, LOCAL]  OpenMVS ReconstructMesh │
│                 + TextureMesh (see §7)  │
│ [STRETCH, CLOUD] SuGaR from splats      │
└────────┬──────────────────────────┘
         ▼
┌──────────────────────────────────┐
│ Stage 7: Output & Viewer            │  [CORE, LOCAL]
│ - Export OBJ/GLTF/PLY + GeoTIFF     │
│ - Three.js viewer, measurement tool  │
└──────────────────────────────────┘
```

**The Core-only path (Stages 1→2→3(COLMAP)→4b→5(OpenMVS)→6(OpenMVS)→7) is a complete working submission by itself.** Everything marked Stretch is an enhancement layered on top, never a dependency the Core path needs to function.

---

## 6. Tech Stack — Full List, Rechecked, With Paywall Status

| Stage | Tool | Link | Tier | Cost |
|---|---|---|---|---|
| Frame sampling/filtering | OpenCV / FFmpeg | opencv.org, ffmpeg.org | Core | Free |
| Dynamic masking (primary) | YOLOv8-seg | github.com/ultralytics/ultralytics | Core | Free (AGPL-3.0 — free for this use; commercial closed-source deployment would need a paid Ultralytics license, irrelevant for SIH) |
| Dynamic masking (quality stretch) | SAM 2 | github.com/facebookresearch/sam2 | Stretch | Free, no gating |
| Pose estimation (guaranteed) | COLMAP | github.com/colmap/colmap | Core | Free |
| Pose estimation (stretch, single-pass-optimized) | MASt3R | github.com/naver/mast3r | Stretch | Free (CC BY-NC-SA — non-commercial, fine for SIH) |
| Pose estimation (stretch, base model) | DUSt3R | github.com/naver/dust3r | Stretch | Free (non-commercial) |
| Pose estimation (stretch, multi-view) | VGGT | github.com/facebookresearch/vggt | Stretch | Free (non-commercial checkpoint; a commercial checkpoint exists but excludes military use — irrelevant for a hackathon demo) |
| Trajectory alignment + accuracy | evo | github.com/MichaelGrupp/evo | Core | Free |
| Depth estimation | Depth Anything V2 | github.com/DepthAnything/Depth-Anything-V2 | Stretch | Free |
| Metric depth | Metric3D v2 | github.com/YvanYin/Metric3D | Stretch | Free |
| Dense point cloud + guaranteed mesh | OpenMVS | github.com/cdcseacave/openMVS | Core | Free |
| Splatting (stretch reconstruction) | Nerfstudio (splatfacto) | github.com/nerfstudio-project/nerfstudio | Stretch | Free |
| Mesh from splats (stretch) | SuGaR | github.com/Anttwo/SuGaR | Stretch | Free |
| Semantic segmentation | SegFormer | huggingface.co/models?search=segformer | Core | Free — use a UAVid/LoveDA/ISPRS-tuned checkpoint, not a plain Cityscapes one (domain gap: those are street-level, not aerial) |
| Georeferencing export | pyproj, GDAL | pyproj4.github.io, gdal.org | Core | Free |
| Viewer | Three.js | threejs.org | Core | Free |
| Backend | FastAPI | fastapi.tiangolo.com | Core | Free |
| Frontend | React | react.dev | Core | Free |
| Compute | Kaggle Notebooks | kaggle.com | — | Free, ~30 GPU-hrs/week/account |

**Nothing in this list requires payment.** The two license notes above (YOLOv8 AGPL, MASt3R/VGGT non-commercial) don't block hackathon use — they only matter if this ever became a commercial product, which is worth one line in the pitch if asked, not a build blocker.

---

## 7. Guaranteed Mesh Path — Exact Commands

This is the concrete "it will work" path, so it's worth having it spelled out rather than left as "use OpenMVS." After COLMAP produces your sparse reconstruction and you convert it to `.mvs` format:

```bash
# Densify the sparse point cloud
DensifyPointCloud scene.mvs

# Build the mesh from the dense cloud
ReconstructMesh scene_dense.mvs -p scene_dense.ply

# Refine mesh geometry
RefineMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_refine.mvs

# Apply textures from the source images
TextureMesh scene_dense.mvs -m scene_dense_mesh_refine.ply -o scene_dense_mesh_refine_texture.mvs
```

This gives you a textured mesh guaranteed, independent of whether the splatting/SuGaR stretch path works. Assign this exact sequence to one teammate on Day 1 and have it running before anyone touches MASt3R or Nerfstudio.

---

## 8. Step-by-Step Timeline (3–4 Person, AI-Assisted)

### Day 1 — Core Path, Start to Finish
- Set up repo, pin dependency versions, everyone creates a Kaggle account
- Get test footage with 2–3 known-length reference objects in frame for ground-truth measurement
- Run the **entire Core path** (Stages 1→7, COLMAP + OpenMVS route) on test footage. By end of Day 1 you have a complete, if basic, working submission. This is non-negotiable before moving to stretch goals — it's what makes the rest of the week safe to spend on ambitious work

### Day 2 — Stretch: Pose/Geometry Upgrade
- Run MASt3R locally on small frame batches; validate against the COLMAP baseline from Day 1
- If it clearly outperforms COLMAP on your footage, move full-sequence runs to Kaggle
- Keep the COLMAP path in the repo and working — don't delete it

### Day 3 — Stretch: Reconstruction Quality + Accuracy Validation
- Run Nerfstudio splatfacto on Kaggle for the splat-based reconstruction
- Use evo to compute trajectory alignment accuracy (ATE/RPE) against GPS ground truth
- Verify your segmentation checkpoint is aerial-domain appropriate; swap or fine-tune on UAVid if not

### Day 4 — Integration & Mesh Upgrade
- If time allows, attempt SuGaR mesh extraction from splats on Kaggle; keep the OpenMVS mesh as the fallback regardless
- Finalize the web viewer and measurement tool against whichever mesh output is best by this point
- **Download every Kaggle-computed asset locally.** Don't depend on a live cloud session during judging

### Day 5 — Demo, Docs, Pitch
- Record a demo video as a hard fallback
- Package accuracy numbers (measured vs. reference objects), completeness, and processing time for both the Core and Stretch paths — showing both is a strength, not a weakness, since it demonstrates engineering judgment
- Rehearse Q&A: why MASt3R/VGGT for single-pass footage, how metric scale works without GCPs, what the Core-vs-Stretch fallback structure is (judges tend to respect visible risk management)

---

## 9. Evaluation / Success Metrics

- **Reconstruction completeness** — % of scene covered vs. gaps
- **Metric accuracy** — measured distance vs. known reference objects, cm/m error
- **Processing time** — report both Core (COLMAP/OpenMVS) and Stretch (MASt3R/3DGS) paths
- **Robustness** — before/after with dynamic-object masking on
- **Georeferencing accuracy** — via evo, vs. GPS ground truth
- **Visual quality** — texture fidelity, no ghosting from masked objects

---

## 10. Task Assignment

### If 4 people
| Member | Owns | Stages |
|---|---|---|
| A | Video pipeline & pose estimation | 1, 2, 3 (COLMAP + MASt3R) |
| B | Reconstruction & meshing | 5, 6 (OpenMVS core + splatting/SuGaR stretch) |
| C | Depth, segmentation & accuracy | 4a, 4b, evo validation |
| D | Output, demo, docs & pitch | 7, dataset curation, metrics packaging, PPT |

### If 3 people
| Member | Owns | Stages |
|---|---|---|
| A | Video pipeline, pose estimation & accuracy | 1, 2, 3, evo |
| B | Depth, segmentation, reconstruction & meshing | 4a, 4b, 5, 6 |
| C | Output, demo, docs & pitch | 7, dataset curation, metrics packaging, PPT |

With 3 people, be more conservative about which Stretch items you attempt — MASt3R upgrade and evo-based accuracy validation are the highest-value stretch goals for judging; splat-to-mesh (SuGaR) is the first thing to drop if time runs short, since the OpenMVS mesh already covers that deliverable.

---

## 11. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Any Stretch component fails or runs out of time | Core path already works end-to-end from Day 1 — nothing downstream depends on Stretch succeeding |
| Single-pass video gives insufficient overlap for COLMAP | MASt3R stretch path targets this directly; COLMAP fallback still produces *a* result even if lower quality |
| Segmentation doesn't generalize to aerial views | Verify checkpoint domain before Day 3; UAVid fine-tune as backup |
| 6GB local VRAM can't handle 3DGS/VGGT on real scenes | These run on Kaggle by default, not local |
| Kaggle's per-account quota runs out | Spread jobs across all 3–4 teammates' accounts; check GitHub Student Pack credits before considering anything paid |
| No real drone available | Phone/gimbal walkthrough or public UAV datasets; keep reference objects in frame regardless |
| Live demo depends on an unavailable cloud session | All heavy outputs downloaded and cached locally before judging; recorded video as hard fallback |

---

## 12. Datasets

- **UAVid** — aerial-view segmentation dataset; free, requires a quick registration form
- **LoveDA / ISPRS Potsdam / ISPRS Vaihingen** — additional free aerial-view segmentation datasets
- **SenseFly sample datasets** — free drone imagery samples
- **Self-recorded phone/gimbal footage** — fastest to iterate on; always include known-length reference objects

---

*Prepared as an internal reference document for Team [Your Team Name] — SIH 2026, Problem Statement SIH26158.*
