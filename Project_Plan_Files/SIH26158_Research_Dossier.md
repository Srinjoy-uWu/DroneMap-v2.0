# SIH26158 — Single-Pass Drone Video to Accurate 3D Model Generation System
### Research Dossier & Technical Approach
**Problem owner:** NTRO (National Technical Research Organisation)
**Category:** Software / Hardware · Computer Vision · Photogrammetry · Geospatial AI
**Prepared:** 2026-08-31

---

## 0. How to read this document

This dossier is organised so a team can go from *"what is even being asked"* to *"here is exactly what we will build and how we will be judged."* Sections 1–3 frame and decompose the problem. Sections 4–5 contain the two tables the problem statement asks us to add (**Desired Output** and **Evaluation Criteria**). Sections 6–14 are the deep technical research: the state of the art, a recommended architecture, and how each stated challenge is handled.

---

## 1. Executive summary

The task is to turn **one continuous drone video pass** (plus GPS and flight metadata) into a **georeferenced, metrically accurate, textured 3D model** of the scene — terrain, buildings, roads, vegetation and obstacles — fast enough to be useful in the field.

This is deliberately *harder* than normal drone mapping. Conventional photogrammetry (COLMAP, OpenDroneMap, Pix4D, Metashape, DJI Terra) is built for **many overlapping passes** — typically 70–80% front/side overlap flown in a grid — and it degrades badly when you give it a single strip of forward-moving footage with a **narrow baseline and one viewing direction**. The core engineering tension of SIH26158 is:

> **Recover dense, correctly-scaled 3D geometry from weak, single-direction parallax — without the redundancy that classical Structure-from-Motion relies on — and pin it to real-world coordinates without ground control points.**

The winning strategy is a **hybrid pipeline** that fuses three ingredients that have each matured dramatically in 2024–2026:

1. **Learning-based / feed-forward geometry** (DUSt3R → MASt3R → VGGT and streaming successors) that infer 3D structure and camera poses directly from images, even with little overlap and no known intrinsics — solving the *limited-viewing-angle* problem that breaks classical SfM.
2. **Monocular metric depth priors** (Metric3D v2, UniDepth, Depth Anything V2, Depth Pro) that supply *absolute-scale* per-pixel depth, filling occluded/low-parallax regions and anchoring metric accuracy.
3. **Metric georeferencing by sensor fusion** — aligning the reconstruction to **GPS/RTK + IMU + barometric altitude** via a similarity (Umeyama) transform or GNSS-aided bundle adjustment, so the model comes out in real-world coordinates (WGS84/UTM) with a stated accuracy — **without GCPs**.

For the *renderable, measurable* output we recommend a **3D Gaussian Splatting (3DGS)** representation with **mesh extraction (2DGS/GOF)** for measurement, and a classical **DSM/DTM + orthomosaic** export for GIS interoperability. Dynamic objects (vehicles, people, animals) are masked out with **SAM 2 + a video motion-segmentation stage** so they don't corrupt the static reconstruction.

*(Sections 6–8 justify each choice against the live state of the art; Sections 4–5 define what we deliver and how it is scored.)*

---

## 2. Problem decomposition

The single sentence in the brief unpacks into six coupled sub-problems:

| # | Sub-problem | What must be solved | Primary signal |
|---|-------------|---------------------|----------------|
| P1 | **Camera pose / trajectory** | Where was the camera for every frame? (6-DoF pose + intrinsics) | Video + IMU + GPS |
| P2 | **Dense geometry** | 3D position of every visible surface point | Video parallax + depth priors |
| P3 | **Metric scale + georeferencing** | Convert arbitrary-scale geometry to real metres and world coordinates | GPS/RTK, IMU, baro, intrinsics |
| P4 | **Surface + texture** | Convert points to a watertight, textured mesh / clean point cloud | Geometry + source frames |
| P5 | **Semantics** | Label terrain / façades / roofs / roads / vegetation / obstacles; remove dynamic objects | 2D+3D segmentation |
| P6 | **Speed** | Do the above in real-time or near-real-time on field-deployable hardware | System engineering |

A key insight: **P1–P3 are where single-pass makes life hard**, and where the newest research (feed-forward pointmap models + metric depth + GNSS fusion) gives us the biggest advantage over off-the-shelf photogrammetry. P4–P6 are comparatively well-served by existing tools.

---

## 3. Why single-pass is fundamentally hard (the constraints, made concrete)

Understanding *why* the obvious approach fails is what separates a winning entry from a naive one.

- **Weak parallax / limited viewing angles (Challenge i).** A single forward pass sees most surfaces from *one* direction over a *short* baseline. Triangulation error scales inversely with baseline; building sides facing away from the flight line and vertical façades under the flight path are barely seen or not seen at all. Classical multi-view stereo needs wide, multi-directional baselines it simply won't get here. → *Mitigation: feed-forward priors + monocular depth that "hallucinate" plausible geometry from learned priors, and flight-geometry-aware capture recommendations.*
- **Motion blur & compression artifacts (Challenge ii).** Rolling-shutter, motion blur and H.264/H.265 block artifacts destroy the sub-pixel feature matches SfM depends on. → *Mitigation: blur-aware frame selection, deblurring, learned matchers (feature-metric / dense) that tolerate degraded frames.*
- **Variable illumination & shadows (Challenge iii).** Exposure changes and moving shadows break the brightness-constancy assumption and create false geometry. → *Mitigation: appearance embeddings (NeRF-W / 3DGS appearance models), shadow-robust matching.*
- **Dynamic objects (Challenge iv).** Moving cars/people/animals violate the static-scene assumption and leave "ghost" geometry. → *Mitigation: video segmentation + motion masking; dynamic-aware reconstruction (MonST3R-style).*
- **GPS inaccuracy & sensor noise (Challenge v).** Consumer GNSS is metre-level and noisy; naive georeferencing inherits that error. → *Mitigation: robust GNSS-aided bundle adjustment / graph optimisation; RTK/PPK when available; outlier rejection.*
- **Near-real-time (Challenge vi).** Full offline photogrammetry takes hours. → *Mitigation: streaming/online reconstruction, incremental GS, GPU acceleration, tiered "live-preview vs refined" outputs.*
- **Occluded surfaces (Challenge vii).** Single pass = large occlusions. → *Mitigation: generative/priors-based completion, symmetry & planarity priors, explicit "confidence/holes" reporting.*
- **Metric accuracy without GCPs (Challenge viii).** No surveyed control points to lock accuracy. → *Mitigation: direct georeferencing from RTK/PPK + calibrated intrinsics + IMU, with an honest accuracy budget (Section 9).*

---

## 4. DESIRED OUTPUT

The system should emit a **layered, georeferenced product set** — a live preview during flight and refined deliverables after. All spatial products carry a defined CRS (e.g. **WGS84 / UTM zone**) and an accompanying **accuracy report**.

| # | Output product | Format(s) | Description | Primary use |
|---|----------------|-----------|-------------|-------------|
| 1 | **Textured 3D mesh** | OBJ / FBX / glTF / OSGB / 3D Tiles | Watertight-as-possible surface mesh with photo-texture, tiled for streaming | Visualization, digital twin, mission briefing |
| 2 | **Georeferenced point cloud** | LAS / LAZ / PLY / E57 | Dense, colorized, real-world-coordinate points with per-point confidence | Measurement, analysis, ingest into GIS/CAD |
| 3 | **Digital Surface Model (DSM)** | GeoTIFF (raster) | Elevation of top surface (incl. buildings, canopy) | Line-of-sight, flood, volumetrics |
| 4 | **Digital Terrain Model (DTM)** | GeoTIFF (raster) | Bare-earth elevation (buildings/veg removed) | Terrain analysis, slope, drainage |
| 5 | **Orthomosaic / true-ortho** | GeoTIFF / COG | Geometrically corrected top-down image mosaic | Mapping, overlay, change detection |
| 6 | **Semantic layers** | GeoJSON / vector + labeled point cloud/mesh | Per-class masks: terrain, façades, rooftops, roads/infrastructure, vegetation, obstacles; dynamic objects flagged & removed | Planning, inspection, threat/asset ID |
| 7 | **3D Gaussian Splatting scene** | `.ply` / `.splat` / `.ksplat` | Real-time photorealistic novel-view scene for immersive review | Rapid situational awareness, VR/AR |
| 8 | **Measurement outputs** | JSON / report / interactive | Distances, areas, volumes, heights, cross-sections with uncertainty | Damage assessment, inspection, construction |
| 9 | **Camera trajectory & poses** | JSON / KML / COLMAP / NVM | Recovered 6-DoF flight path + intrinsics | QA, re-processing, provenance |
| 10 | **Accuracy & provenance report** | PDF / JSON | CRS, GSD, reprojection error, GPS/RTK status, estimated horizontal/vertical accuracy, coverage & holes map, processing log | Trust, auditability, decision support |
| 11 | **Live in-flight preview** | On-screen 3D / web viewer | Progressive point cloud/splat as the drone flies (near-real-time) | Field situational awareness, re-fly decision |

**Design principle:** deliver *two tiers* — a **near-real-time "situational" tier** (progressive splat/point cloud, coarse but immediate) and a **refined "analytical" tier** (metric mesh, DSM/DTM, ortho, semantics) produced with a short delay. This directly answers both the "near-real-time situational awareness" and "measurement/analysis" goals.

---

## 5. EVALUATION CRITERIA

How NTRO / judges would (and should) score a solution. Weights are a suggested starting split for a hackathon rubric; the **bold** targets are realistic "good" thresholds for a single-pass system without GCPs.

| # | Criterion | How it is measured | Target / "good" value | Suggested weight |
|---|-----------|--------------------|-----------------------|------------------|
| C1 | **Metric / geometric accuracy** | RMSE of checkpoint coordinates & of known distances/heights vs. survey ground truth; Chamfer distance / F-score vs. reference mesh | **≤ 1–3× GSD** relative; **< 0.5 m** horiz. / **< 1 m** vert. with RTK; documented budget without RTK | 25% |
| C2 | **Georeferencing accuracy** | Absolute position error at independent checkpoints (horizontal & vertical) | **cm–dm** with RTK/PPK; **1–3 m** with standard GNSS | 10% |
| C3 | **Reconstruction completeness** | % of target area reconstructed; occlusion/hole coverage; point density (pts/m²) | High coverage; holes explicitly reported & minimized | 10% |
| C4 | **Reconstruction quality / fidelity** | Novel-view PSNR / SSIM / LPIPS; mesh smoothness & artifact rate; texture sharpness | Visually clean, low floaters; competitive PSNR/LPIPS | 10% |
| C5 | **Semantic accuracy** | Per-class IoU / mIoU for terrain, façade, roof, road/infra, vegetation, obstacle; dynamic-object removal precision/recall | **mIoU > 0.6–0.7**; high dynamic-removal recall | 10% |
| C6 | **Robustness** | Degradation under motion blur, illumination change, dynamic scenes, GPS noise (ablation tests) | Graceful degradation; no catastrophic failure | 10% |
| C7 | **Speed / latency** | Time-to-first-preview; end-to-end time per minute of video; real-time factor; on-device vs. cloud | Near-real-time preview (seconds–minutes); refined model in minutes | 10% |
| C8 | **Single-pass robustness** | Accuracy retained with single strip & low overlap vs. multi-pass baseline | Usable model from one pass; quantified gap to multi-pass | 5% |
| C9 | **Usability & deliverables** | Completeness of Section-4 outputs; measurement tools; standard formats; viewer UX | All key formats; interactive measure/annotate | 5% |
| C10 | **Scalability & deployment** | Area covered per unit time; memory scaling; field-hardware feasibility; robustness of pipeline | Scales to km²; runs on field GPU / edge+cloud | 5% |

**Evaluation methodology (recommended):** hold out **independent checkpoints** (not used in processing) for C1–C2; run **controlled ablations** for C6 (inject blur/illumination/dynamic objects/GPS noise); compare against a **multi-pass photogrammetry baseline** (ODM/Metashape) for C8; report **all** metrics with uncertainty. See Section 12 for the full protocol.

---

## 6. State of the art — reconstruction backbones

This section surveys the live 2024–2026 literature across six fronts: **6.1** feed-forward geometry (the game-changer for single-pass), **6.2** monocular metric depth, **6.3** camera pose & scale (SLAM/VIO + GNSS fusion), **6.4** neural rendering & mesh (3DGS/NeRF for aerial), **6.5** classical SfM/MVS baselines, **6.6** semantics & dynamic-object handling.

### 6.1 The feed-forward revolution (DUSt3R → VGGT → UAV-specific) — our reconstruction core

The single most important development for this problem: since late 2023, **learned networks now regress 3D geometry (pointmaps, depth, camera poses, even Gaussians) directly from images**, replacing or bootstrapping the fragile feature-matching + bundle-adjustment core of classical Structure-from-Motion. Because they lean on **learned priors**, they still produce plausible geometry where classical triangulation fails — exactly the low-overlap, single-direction, small-baseline regime of a single drone pass. This family is our recommended backbone.

**The lineage, and what each contributes:**

- **DUSt3R** (arXiv [2312.14132](https://arxiv.org/abs/2312.14132), [github](https://github.com/naver/dust3r)) — the origin. Regresses per-pixel 3D "pointmaps" for an image *pair* in a shared frame, no poses or intrinsics needed; handles >2 images by optimization-based global alignment. *Up-to-scale; static-only; O(N²) alignment doesn't scale to thousands of frames.*
- **MASt3R** (arXiv [2406.09756](https://arxiv.org/abs/2406.09756), [github](https://github.com/naver/mast3r)) — adds a dense matching head → far better, wide-baseline-robust correspondences (helps oblique aerial views). **MASt3R-SfM** (arXiv [2409.19152](https://arxiv.org/abs/2409.19152)) turns it into a full SfM pipeline whose retrieval step cuts complexity from **quadratic to linear** — a dependable, well-supported (NAVER) fallback. *Up-to-scale by default.*
- **Pow3R** (CVPR 2025, arXiv [2503.17316](https://arxiv.org/abs/2503.17316), [github](https://github.com/naver/pow3r)) — **the metric lever**: a DUSt3R-family model that *optionally ingests known intrinsics, relative pose, and sparse/dense depth* at inference. Feed it drone intrinsics + GPS/IMU-derived priors → **metric when metric priors are supplied.**
- **VGGT** (CVPR 2025 **Best Paper**, arXiv [2503.11651](https://arxiv.org/abs/2503.11651), [github](https://github.com/facebookresearch/vggt), Meta) — the current reference backbone. One Transformer infers **camera params + depth + pointmaps + 3D tracks** in a single pass, in **<1 s** for moderate view counts. *Best-in-class quality, but global attention → memory grows ~quadratically → OOM beyond ~a few hundred frames, and up-to-scale.* A full flight is thousands of frames, so VGGT **must be wrapped**:
  - **VGGT-Long** (ICRA 2026, arXiv [2507.16443](https://arxiv.org/abs/2507.16443), [github](https://github.com/DengKaiCQ/VGGT-Long)) — chunk → overlapping-align → lightweight loop-closure to reach **kilometre-scale** RGB video with no calibration/retraining. Directly targets the long-flight-line failure mode.
  - **VGGT-X** (arXiv [2509.25191](https://arxiv.org/abs/2509.25191)) — memory-efficient scaling to **1000+ images** + robust 3DGS head for dense novel-view synthesis.
  - **VGGT-Align** (arXiv 2608.15260) — kills **scale drift across chunks** via cross-chunk anchoring (reports ~-32% trajectory error).
- **Pi3 / π³** (arXiv [2507.13347](https://arxiv.org/abs/2507.13347), [github](https://github.com/yyfz/Pi3)) — permutation-**equivariant** successor (no fixed reference view → robust to input ordering), arguably more robust than VGGT. Same up-to-scale, memory-bound regime.
- **Streaming/online variants** — match "single continuous pass" literally:
  - **CUT3R** (CVPR 2025, arXiv [2501.12387](https://arxiv.org/abs/2501.12387), [github](https://github.com/CUT3R/CUT3R)) — stateful recurrent model, bounded state, emits per-pixel 3D per frame; **claims metric scale** *and* **tolerates dynamic content**. Caveat: trained indoor/driving → aerial metric reliability unproven.
  - **Spann3R** (arXiv [2408.16061](https://arxiv.org/abs/2408.16061), [github](https://github.com/HengyiWang/spann3r)) — DUSt3R + external spatial memory, incremental/real-time on ordered frames.
  - **StreamVGGT** (arXiv [2507.11539](https://arxiv.org/abs/2507.11539), [github](https://github.com/wzzheng/StreamVGGT)) — causal, KV-cached, VGGT-quality geometry online (low latency; KV cache still grows with length).
  - **Fast3R** (CVPR 2025, arXiv [2501.13928](https://arxiv.org/abs/2501.13928), [github](https://github.com/facebookresearch/fast3r)) — 1000+ images in one forward pass (VRAM-bound). **MV-DUSt3R+** (CVPR 2025, arXiv [2412.06974](https://arxiv.org/abs/2412.06974)) — sparse-view scene in ~2 s.
- **MapAnything** (3DV 2026, arXiv [2509.13414](https://arxiv.org/abs/2509.13414), Meta, [project](https://map-anything.github.io)) — **best "metric out-of-the-box" feed-forward model**: outputs a factored representation with an *explicit metric scale factor*, and **ingests optional intrinsics / poses / depth priors** to enforce a globally consistent metric frame in one pass. Chunk for thousands of frames.
- **Dynamic-native**: **MonST3R** (ICLR 2025, arXiv [2410.03825](https://arxiv.org/abs/2410.03825), [github](https://github.com/Junyi42/monst3r)) estimates geometry under motion; successor **PAGE-4D** (arXiv 2510.17568) extends VGGT to dynamic scenes — relevant to Challenge (iv).

**★ UAV-specific, and almost exactly our problem:**
- **GeoFF3D** — "Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping" (arXiv 2608.28288, [github](https://github.com/yanxian-ll/GeoFF3D)). Predicts poses + dense pointmaps **directly in a gravity-aligned, Z-up *metric* frame, anchored by georeferenced GPS camera translations** + optional priors; a chunked "SLRF" framework partitions images and aggregates hierarchically, and it explicitly **avoids the unstable full-Sim(3) alignment that near-collinear single-pass trajectories trigger**. Reconstructs **~2,000 images in ~5 min** with large gains over streaming/SLAM baselines on long UAV scenes. ⚠ *Very new (Aug 2026), single-group, not yet battle-tested — verify repo/checkpoints before committing.*
- **UAVFF3D** benchmark (arXiv 2605.17942, [github](https://github.com/yanxian-ll/UAVFF3D)) — 170k+ real + 370k+ synthetic UAV images targeting oblique views and height/FOV ambiguity. **Use this to validate any backbone on aerial data before trusting it.**

**Feed-forward Gaussian-splat predictors** (pixelSplat [2312.12337], MVSplat [2403.14627], Splatt3R [2408.13912], NoPoSplat [2410.24207]) predict 3D Gaussians for photorealistic rendering — useful as a *downstream appearance layer*, **not** the metric geometry core (2-view, up-to-scale, tuned to indoor/object data). Survey: arXiv [2507.14501](https://arxiv.org/abs/2507.14501).

**Comparison of candidate backbones (the axes that decide it):**

| Method | Metric scale? | Many-frame (flight) | Dynamic? | Poses needed? | Aerial fit | Maturity |
|--------|---------------|---------------------|----------|---------------|------------|----------|
| **GeoFF3D** | ✅ metric + geo (GPS-anchored) | ✅ chunked SLRF (~2k/5min) | ✖ | No (uses GPS) | ★★★ purpose-built | ⚠ brand-new |
| **VGGT + VGGT-Long/-X** | ✖ up-to-scale (anchor externally) | ✅ via chunk+loop-close | ✖ | No | ★★ strong general | ★★★ proven |
| **MapAnything** | ✅ metric (explicit scale) | ~ chunk for 1000s | ✖ | Optional | ★★★ ingests GPS/intrinsics | ★★ recent |
| **CUT3R** | ✅ metric (claimed) | ✅ streaming, bounded state | ✅ | No | ★★ (aerial unproven) | ★★ |
| **StreamVGGT** | ✖ up-to-scale | ~ KV cache grows | ~ | No | ★★ online | ★★ |
| **MASt3R-SfM** | ✖ up-to-scale | ✅ linear via retrieval | ✖ | No | ★★ robust matching | ★★★ |
| **Pow3R** | ✅ w/ priors | ~ pairwise core | ✖ | Optional | ★★ | ★★ |
| **Pi3 / Fast3R** | ✖ up-to-scale | Fast3R 1000+ (VRAM) | ✖ | No | ★★ | ★★ |

**Verdict (our backbone choice):**
1. **GeoFF3D is the closest turnkey match** — it solves metric+geo scale, many-frames, *and* near-collinear single-pass instability simultaneously. Adopt if the repo/checkpoints check out; otherwise mine its SLRF chunking idea.
2. **VGGT + VGGT-Long/-X + VGGT-Align is the most proven, lowest-risk route** — highest-quality general geometry, wrapped for long flights, georeferenced externally via GPS/IMU (Section 9). **This is our safe default.**
3. **MapAnything is the best native-metric option** that ingests our flight priors directly.
4. **CUT3R / StreamVGGT** if we need *live onboard* reconstruction during flight (situational tier).

**Design constraints this family forces on us:** (a) **chunk + align + loop-close** or **stream with bounded state** — global-attention models OOM past a few hundred frames; (b) most are **up-to-scale** → we *must* anchor metric/geo scale with GPS/IMU/intrinsics regardless (Section 9); (c) plan for **scale drift + loop closure** on long low-parallax lines; (d) backbones are trained largely on indoor/driving data → **aerial is out-of-distribution; validate on UAVFF3D first.**

---

### 6.2 Monocular metric depth priors

> **⚠ Load-bearing caveat:** at typical drone AGL (50–120 m, nadir), **monocular metric-depth models are the weak link, not the metric backbone.** Their "metric" prior is learned from *ground-level object sizes* (cars, rooms, people) and largely breaks down for high-altitude top-down imagery; metric error grows with altitude. **Use monocular depth for densification / occlusion-completion / cross-checking — never as the primary scale source at altitude.** The reliable metric signal is GNSS+IMU (§6.3, §9).

With that role in mind, the best models to supply a dense depth prior per frame:

| Model | Metric? | Needs intrinsics? | Aerial fit | Speed | Ref |
|-------|---------|-------------------|------------|-------|-----|
| **Metric3D v2** | ✅ (+ normals) | ✅ input intrinsics | Best-in-class outdoor generalization; feed drone calibration | Fast | TPAMI'24, [2404.15506](https://arxiv.org/abs/2404.15506), [code](https://github.com/YvanYin/Metric3D) |
| **UniDepthV2** | ✅ | ✖ (predicts camera) + **uncertainty** | Good outdoor; confidence useful for fusion weighting | Fast | [2502.20110](https://arxiv.org/abs/2502.20110), [code](https://github.com/lpiccinelli-eth/UniDepth) |
| **Depth Pro** (Apple) | ✅ + focal est. | ✖ | Sharp boundaries, strong outdoor, helps uncalibrated video | **2.25 MP / 0.3 s** | ICLR'25, [2410.02073](https://arxiv.org/abs/2410.02073), [code](https://github.com/apple/ml-depth-pro) |
| **Depth Anything V2** | relative; **metric fine-tunes** | per-domain | Robust, detailed; outdoor metric variant exists | **Very fast** | NeurIPS'24, [2406.09414](https://arxiv.org/abs/2406.09414), [code](https://github.com/DepthAnything/Depth-Anything-V2) |
| **MoGe-2** | ✅ **metric point maps** | ✖ | Point-map output convenient for direct 3D | Fast | [2507.02546](https://arxiv.org/abs/2507.02546) |
| **Video Depth Anything** | relative (metric downstream) | — | **Temporally consistent depth, 30 FPS** — avoids per-frame flicker on video | 30 FPS | [2501.12375](https://arxiv.org/abs/2501.12375) |
| *ZoeDepth / Marigold* | ✅ / ✖ relative | — | superseded / high-quality-relative-only | — | [2302.12288](https://arxiv.org/abs/2302.12288) / [2312.02145](https://arxiv.org/abs/2312.02145) |

**Recommendation:** **Metric3D v2** (fed our calibrated intrinsics) or **UniDepthV2/Depth Pro** (uncalibrated) for the depth prior, **MoGe-2** if we want metric point-maps directly, and **Video Depth Anything** for temporally stable video depth. Whatever we pick, **validate on nadir aerial data (UAVFF3D / UseGeo / Mid-Air) first** — none are trained primarily on high-AGL top-down imagery. **Prompt Depth Anything** ([2412.14015](https://arxiv.org/abs/2412.14015)) is relevant if the drone carries a low-res LiDAR to anchor scale.

---

### 6.3 Camera pose & scale: SLAM / VIO + GNSS fusion

This is where metric scale and georeferencing *actually* come from. **IMU → metric scale + attitude at high rate; GNSS → absolute geodetic position + datum; baro → vertical prior; intrinsics → removes focal ambiguity.**

| System | IMU | Metric? | GPS fusion | Robustness | RT? | Ref |
|--------|-----|---------|------------|------------|-----|-----|
| **ORB-SLAM3** | ✅ | ✅ w/ IMU | fork-only | features fail on blur/low-texture | ✅ CPU | [2007.11898](https://arxiv.org/abs/2007.11898), [code](https://github.com/UZ-SLAMLab/ORB_SLAM3) |
| **VINS-Fusion** | ✅ | ✅ | ✅ `global_fusion` GPS pose-graph | IMU bridges short blur | ✅ CPU | [code](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) |
| **OpenVINS** | ✅ | ✅ | research ext. | efficient MSCKF filter | ✅ >100 Hz | [rpng/open_vins](https://github.com/rpng/open_vins) |
| **DROID-SLAM** | ✖ | mono up-to-scale; metric w/ stereo | ✖ | **very high** (dense BA) | heavy GPU | [2108.10869](https://arxiv.org/abs/2108.10869), [code](https://github.com/princeton-vl/DROID-SLAM) |
| **DPVO / DPV-SLAM** | ✖ | up-to-scale | ✖ | **high** (learned, blur/dynamic-tolerant) | ✅ 1–4× RT, 1 GPU | [2208.04726](https://arxiv.org/abs/2208.04726) / [2408.01654](https://arxiv.org/abs/2408.01654) |
| **★ GVINS** | ✅ | ✅ | ✅ **tight, raw pseudorange+Doppler** → drift-free global metric | graceful through GNSS loss | ✅ | [2103.07899](https://arxiv.org/abs/2103.07899), [code](https://github.com/HKUST-Aerial-Robotics/GVINS) |
| **MASt3R-SLAM** | ✖ | up-to-scale | ✖ | dense, prior-based | ✅ real-time | [2412.12392](https://arxiv.org/abs/2412.12392) |

*(Also: **DM-VIO** [2201.04114](https://arxiv.org/abs/2201.04114) monocular-VIO SOTA; **maplab 2.0** for multi-session VI mapping with GPS constraints.)* IMU-based systems recover **true metric scale**; pure-monocular learned VO/SLAM (DPVO, DROID-mono, MASt3R-SLAM, VGGT) are **up-to-scale** and must be georeferenced externally.

**Three fusion strategies to get metric + georeferenced scale (increasing tightness):**
1. **Loose / post-hoc — Umeyama similarity alignment.** Reconstruct up-to-scale (VGGT/DROID/COLMAP), then fit a **7-DoF similarity transform** mapping recovered camera centres → GPS antenna positions (in local ENU/ECEF). COLMAP's `model_aligner` does exactly this via RANSAC (≥3 images, [FAQ](https://colmap.github.io/faq.html)). Cheap and robust, **but scale observability is poor on a straight flight line** (little spread).
2. **Tight — GNSS-aided bundle adjustment (recommended for accuracy without GCPs).** Insert **GPS position priors as weighted constraints in the BA cost** so georeferencing + self-calibration are jointly solved ("direct georeferencing"). COLMAP `pose_prior_mapper` (with `--prior_position_std_x/y/z`); production equivalents in Metashape/Pix4D/ODM/DJI Terra. **Must model camera↔GNSS-antenna lever arm and time-sync** (offset × ground speed = along-track bias).
3. **Tight VIO+GNSS (best for live / robust single-pass).** **GVINS** fuses raw GNSS + visual + inertial in a factor graph → globally consistent **drift-free metric** trajectory, degrading gracefully through partial GNSS loss. Lighter: VINS-Fusion `global_fusion` + PPK. Baro adds a vertical factor (helps the weak Z channel).

**Recommended pose/scale pipeline:** tight **VIO+GNSS (GVINS-style, or VINS-Fusion + PPK)** for real-time metric georeferenced poses → **GNSS-aided BA** (COLMAP `pose_prior_mapper`/Metashape) for a refined self-calibrated model → monocular metric depth used only to densify/complete/cross-check. (Full accuracy budget in §9.)

---

### 6.4 Neural rendering & mesh extraction — 3D Gaussian Splatting (3DGS) & NeRF for aerial

Once we have poses + geometry (from §6.1/§6.3), we need a **renderable, measurable** representation. **3D Gaussian Splatting has decisively overtaken NeRF** for this use case: NeRF aerial methods (Mega-NeRF [2112.10703], Block-NeRF [2202.05263], BungeeNeRF [2112.05504]) are the prior generation — slow to train, non-real-time to render, weak at clean mesh export — and are now only a baseline to beat. **One crucial nuance:** almost all high-quality 3DGS methods *render* in real time but *train offline* (minutes→hours). "Real-time reconstruction" only holds for the SLAM / on-the-fly family. This forces the **two-tier** design (§4).

**(a) Quality & mesh-extraction variants** (these turn splats into measurement-grade surfaces):

| Method | What it adds | Textured mesh? | Aerial value | Ref |
|--------|--------------|----------------|--------------|-----|
| **3DGS** (base) | Real-time photorealistic NVS from posed images | ✖ (floaters, no surface) | foundation | SIGGRAPH'23, [2308.04079](https://arxiv.org/abs/2308.04079), [code](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Mip-Splatting** | 3D+2D anti-alias filters for scale/altitude variation | ✖ (quality patch) | ★★★ (drones vary altitude/zoom) | CVPR'24, [2311.16493](https://arxiv.org/abs/2311.16493), [code](https://github.com/autonomousvision/mip-splatting) |
| **2DGS** | Planar disks → view-consistent geometry; TSDF→mesh | ✅ (bounded+unbounded) | ★★★ splat-to-mesh workhorse | SIGGRAPH'24, [2403.17888](https://arxiv.org/abs/2403.17888), [code](https://github.com/hbb1/2d-gaussian-splatting) |
| **GOF** | Opacity level-set + Marching Tetrahedra, **unbounded** | ✅ (unbounded) | ★★★ open outdoor scenes | SIGGRAPH Asia'24, [2404.10772](https://arxiv.org/abs/2404.10772), [code](https://github.com/autonomousvision/gaussian-opacity-fields) |
| **RaDe-GS** | Rasterized accurate depth+normal at ~3DGS speed | ✅ | ★★★ **best accuracy/speed** (Chamfer ≈ Neuralangelo on DTU) | TOG'24, [2406.01467](https://arxiv.org/abs/2406.01467), [code](https://github.com/HKUST-SAIL/RaDe-GS) |
| **SuGaR** | Surface-aligned Gaussians → Poisson mesh (editable) | ✅ (editable) | ★★ fast, lower fidelity | CVPR'24, [2311.12775](https://arxiv.org/abs/2311.12775), [code](https://github.com/Anttwo/SuGaR) |

**(b) Large-scale / aerial** — for km²-scale areas:
- **CityGaussianV2** (ICLR 2025, [2411.00771](https://arxiv.org/abs/2411.00771), [code](https://github.com/Linketic/CityGaussian)) — built on 2DGS → **geometrically accurate, efficient large-scale surface/mesh** (10× compression, ~25% faster, ~50% less memory). **Leading option for accurate mesh at city/aerial scale.**
- **VastGaussian** (CVPR 2024, [2402.17427](https://arxiv.org/abs/2402.17427)) — progressive partitioning + **decoupled appearance modeling** (handles exposure drift across a flight). NVS-only, no mesh, *no official code.*
- **Hierarchical-3DGS** (SIGGRAPH 2024, [2406.12080](https://arxiv.org/abs/2406.12080), [code](https://github.com/graphdeco-inria/hierarchical-3d-gaussians)) & **Octree-GS** (TPAMI 2025, [2403.17898](https://arxiv.org/abs/2403.17898), [code](https://github.com/city-super/Octree-GS)) — LoD structures for real-time rendering of very large captures; pair as the partitioning layer.

**(c) On-the-fly / SLAM-integrated (the live/near-real-time tier — pose-free):**
- **On-the-fly-NVS** (Inria, SIGGRAPH 2025, [2506.05558](https://arxiv.org/abs/2506.05558), [code](https://github.com/graphdeco-inria/on-the-fly-nvs)) — **standout**: jointly estimates poses *and* trains 3DGS immediately after capture, large wide-baseline scenes, ordered/unposed images → matches drone video. No metric scale by itself.
- **Gaussian On-the-Fly Splatting** ([2503.13086](https://arxiv.org/abs/2503.13086)) — progressive near-real-time GS with incremental SfM, no pre-COLMAP.
- **Photo-SLAM** (CVPR 2024, [2311.16728](https://arxiv.org/abs/2311.16728), [code](https://github.com/HuajianUP/Photo-SLAM)) — **runs on embedded GPU (Jetson)** → most onboard-feasible for a drone. **MonoGS** (CVPR 2024, [2312.06741](https://arxiv.org/abs/2312.06741), [code](https://github.com/muskie82/MonoGS)) — 3DGS-only SLAM, monocular. Both drift over long flights; medium scale. **RTG-SLAM** ([2404.19706](https://arxiv.org/abs/2404.19706)) only if an RGB-D sensor is carried.

**(d) Drone-specific 2025–2026 papers** (confirm strong momentum on *exactly* this problem):
- **DroneSplat** ([2503.16964](https://arxiv.org/abs/2503.16964), [project](https://bityia.github.io/DroneSplat/)) — robust 3DGS for **in-the-wild drone imagery**; removes dynamic distractors and **explicitly handles limited-view constraints**. ★ Read first — most directly targets our limited-baseline problem.
- **Large-scale Photorealistic Outdoor 3D from UAV using GS** (arXiv 2602.20342, 2026) — end-to-end **drone-video-stream → 3D**, low-latency. Closest to our exact system framing.
- **AnyCity / Feed-Forward GS from Sparse Aerial Views** (2605.19949), **AeroDGS** (CVPR 2026, 2602.22376, dynamic monocular UAV), **BlitzGS** (2605.13794, distributed city-scale). *(These 2026 IDs are very recent — verify before citing formally.)*

**(e) Mesh + texture extraction workflow for measurement:** render depth from the chosen GS model → **TSDF fusion (Open3D)** or Marching Tetrahedra → mesh → **texture-bake by projecting source frames** (photogrammetry-style UV atlas, e.g. `mvs-texturing`). GS stores per-Gaussian color, so a clean *UV-textured* mesh needs this separate baking step. Surface-first methods (2DGS/GOF/RaDe-GS/CityGaussianV2) are the measurement-grade routes.

**Verdict (rendering/mesh layer), two tracks:**
- **Measurement-grade mesh (accuracy priority, offline):** metric poses (§6.1/§6.3) → **RaDe-GS or 2DGS** for site-scale, **GOF** for unbounded, **CityGaussianV2 (+ Hierarchical/Octree LoD)** for large areas → TSDF/Marching-Tetrahedra mesh + texture-bake; add **Mip-Splatting** for altitude/scale variation.
- **Live/near-real-time (situational tier):** **On-the-fly-NVS** or **Photo-SLAM** (Jetson) to build a splat during flight, then run a mesh pass afterward.

**Limitations to design around:** every mesh-quality method **needs poses and is scale-ambiguous** → metric measurement impossible without GNSS/RTK poses or GCPs (§9); single-pass **weak parallax → floaters, background collapse** (DroneSplat mitigates); **real-time ⟷ quality is a hard trade-off today**; exposure/lighting drift needs appearance decoupling; textured mesh needs a separate baking stage.

---

### 6.5 Classical SfM / MVS baselines (and why single-pass breaks them)

These are the incumbent tools. We keep them as **optional high-accuracy refinement** and as the **baseline to beat** (Challenge C8) — not as the core engine.

| Tool | Open-source? | Direct georef (GPS/EXIF) | Role |
|------|--------------|--------------------------|------|
| **COLMAP** | ✅ BSD | ✅ `model_aligner` to EXIF GPS/RTK priors | Reference incremental SfM + MVS ([site](https://colmap.github.io), CVPR'16) |
| **GLOMAP** | ✅ BSD | ✅ via COLMAP DB/priors | **Global** SfM, orders-of-magnitude faster ([2407.20219](https://arxiv.org/abs/2407.20219), ECCV'24) |
| **OpenMVG + OpenMVS** | ✅ MPL2/AGPL | ✅ GPS priors | SfM + dense/mesh/texture |
| **OpenDroneMap / WebODM** | ✅ AGPL | ✅ EXIF GPS + GCP + RTK/PPK | Full open drone pipeline (on OpenSfM) |
| **Metashape / Pix4D / RealityCapture / DJI Terra** | ✖ commercial | ✅ GPS/RTK/PPK/GCP + self-calib | Survey-grade production references |

**Overlap they assume:** Pix4D recommends **≥75% front / ≥60% side**, rising to **85%/70%+** over vegetation/uniform terrain ([Pix4D support](https://support.pix4d.com/hc/en-us/articles/202557459)); USGS UAS guidance ~80%/60%. **A single forward pass simply cannot supply this.**

**Why single-pass low-overlap breaks classical SfM+MVS** — the five failure modes we must engineer around:
1. **Feature-matching collapse** — SfM needs each 3D point in *many* overlapping images; one strip gives mostly forward overlap and few viewpoints per point → too few tie points, failed registration, broken tracks.
2. **Weak intersection geometry → "doming"** — narrow, near-parallel viewing angles give poor triangulation angles and ill-conditioned bundle adjustment → systematic bowl/dome deformation and height/scale drift along the strip.
3. **MVS holes** — dense stereo needs multiple viewing directions; single-direction capture leaves occlusion gaps. **Building façades are essentially unrecoverable** without oblique/multi-directional views.
4. **Drift, no loop closure** — linear (non-loop) acquisition accumulates error; without GCPs the block is weakly constrained.
5. **Textureless/repetitive surfaces** (roads, roofs, water) starve the matcher further.

→ This failure analysis is precisely *why* the feed-forward backbone (§6.1) — which regresses geometry from learned priors instead of relying on redundant matching — is the right core, with classical MVS reserved for the rare frames where overlap is actually sufficient.

---

### 6.6 Semantics & dynamic-object handling

We must label the scene into the **five required classes** (terrain · façades/rooftops · roads/infrastructure · vegetation · obstacles) and **strip dynamic objects** (vehicles/humans/animals) so they don't corrupt the static model. Two complementary routes: label in 2D and lift to 3D, or segment the reconstruction directly.

**(a) Aerial 2D semantic segmentation** (per-frame, then fuse to 3D):
- **Mask2Former** ([2112.01527](https://arxiv.org/abs/2112.01527)), **OneFormer** ([2211.06220](https://arxiv.org/abs/2211.06220)), **SegFormer** ([2105.15203](https://arxiv.org/abs/2105.15203)) — universal/efficient seg transformers; **InternImage** ([2211.05778](https://arxiv.org/abs/2211.05778)) SOTA backbone for fine-tuning.
- **Remote-sensing foundation models** for label-scarce aerial data: SatMAE, Scale-MAE, SkySense, SAMRS, and **IBM–NASA Prithvi** ([HF](https://huggingface.co/ibm-nasa-geospatial)) — pretrain then fine-tune on **UAVid** (4K UAV video, 8 urban classes matching our label set — [uavid.nl](https://uavid.nl), [1810.10438](https://arxiv.org/abs/1810.10438)) / **ISPRS Vaihingen-Potsdam**.

**(b) 3D point-cloud / mesh semantic segmentation** (label the reconstruction directly):
- **Point Transformer V3 (PTv3)** — current SOTA, scales to outdoor ([2312.10035](https://arxiv.org/abs/2312.10035), [code](https://github.com/Pointcept/PointTransformerV3)); **Superpoint Transformer** — very efficient at large scale ([2306.08045](https://arxiv.org/abs/2306.08045), [code](https://github.com/drprojects/superpoint_transformer)); **RandLA-Net** (SensatUrban baseline), **KPConv**, **MinkowskiNet**. Unified framework: **Pointcept** ([code](https://github.com/Pointcept/Pointcept)).

**(c) SAM 2 — the linchpin for video masking** ([2408.00714](https://arxiv.org/abs/2408.00714), [code](https://github.com/facebookresearch/sam2)): promptable masks for images **and video, with memory to track/propagate masks across frames** — ideal for propagating per-class *and* dynamic-object masks through the drone video before lifting to 3D.

**(d) Dynamic-object removal** (vehicles/humans/animals → clean static reconstruction):
- **Semantic pre-masking** — detect movers (SAM2 + Mask2Former), exclude those pixels from reconstruction (native mask support in Metashape/RealityCapture and most GS trainers).
- **Distractor-robust neural reconstruction** — **RobustNeRF** ([2302.00833](https://arxiv.org/abs/2302.00833)), **NeRF On-the-go** ([2405.18715](https://arxiv.org/abs/2405.18715)), **SpotLessSplats** ([2406.20055](https://arxiv.org/abs/2406.20055)), **WildGaussians** ([2407.08447](https://arxiv.org/abs/2407.08447)), and drone-specific **DroneSplat** ([2503.16964](https://arxiv.org/abs/2503.16964)).
- **Dynamic-aware geometry** — **MonST3R** ([2410.03825](https://arxiv.org/abs/2410.03825)), **CUT3R** (§6.1, tolerates dynamics), **DAS3R** ([2412.19584](https://arxiv.org/abs/2412.19584)), **Shape-of-Motion** ([2407.13764](https://arxiv.org/abs/2407.13764)).

**Recommended semantics flow:** SAM 2 tracks movers + class regions across frames → mask dynamics out of reconstruction → after the static model is built, label it with PTv3 (3D) and/or fuse per-frame 2D seg → export per-class layers (§4 output #6).

---

## 7. Recommended system architecture

The design principle from §4 is a **two-tier system**: a **live situational tier** (onboard/edge, immediate, coarse) and a **refined analytical tier** (ground-station/cloud GPU, minutes-later, metric). Both share the same front-end.

```
                    ┌─────────────────────────────────────────────────────────────┐
   DRONE (in air)   │  Video 1080p/4K  +  GNSS(/RTK)  +  IMU  +  Baro  +  Intrinsics │
                    └───────────────┬─────────────────────────────┬────────────────┘
                                    │ (stream)                     │ (full log)
            ┌───────────────────────▼──────────┐      ┌────────────▼───────────────────────────┐
            │  LIVE / SITUATIONAL TIER (edge)   │      │  REFINED / ANALYTICAL TIER (GPU)         │
            │  onboard Jetson / ground laptop   │      │  ground station or cloud                 │
            ├───────────────────────────────────┤      ├──────────────────────────────────────────┤
            │ S1 Frame decode + time-sync        │      │ S1' Keyframe select (blur+baseline aware)│
            │ S2 Photo-SLAM / StreamVGGT / CUT3R │      │      deblur, exposure-normalize, undistort│
            │    → progressive pts/splat + pose  │      │ S2' SAM2 + Mask2Former → dynamic masking │
            │ S3 Live viewer (situational aware) │      │ S3' Tight VIO+GNSS (GVINS / VINS+PPK)    │
            │    "re-fly?" decision support      │      │      → metric georeferenced poses        │
            └───────────────────────────────────┘      │ S4' Feed-forward recon: VGGT(+VGGT-Long) │
                                                        │      / GeoFF3D / MapAnything (GPS+intrin. │
                                                        │      priors) → dense pointmaps + poses   │
                                                        │      + Metric3D-v2 depth densification   │
                                                        │ S5' Georeference/anchor: GNSS-aided BA / │
                                                        │      Umeyama → metric datum (WGS84/UTM)  │
                                                        │ S6' Surface+appearance: RaDe-GS/2DGS/GOF │
                                                        │      (CityGaussianV2 if large) → TSDF     │
                                                        │      mesh + texture-bake                  │
                                                        │ S7' Semantics: PTv3 on cloud + 2D-lift   │
                                                        │      → 5-class layers                    │
                                                        │ S8' Export + accuracy report + viewer    │
                                                        └──────────────────────────────────────────┘
                                                                            │
                    ┌───────────────────────────────────────────────────────▼─────────────────┐
   DELIVERABLES     │ textured mesh · georef point cloud · DSM · DTM · orthomosaic · 3DGS scene │
   (§4)             │ semantic layers · measurements · trajectory · accuracy & provenance report│
                    └──────────────────────────────────────────────────────────────────────────┘
```

**Per-stage tool choices (refined tier):**

| Stage | Job | Primary choice | Alternatives / notes |
|-------|-----|----------------|----------------------|
| S1' | Keyframe selection & cleanup | Blur-metric + baseline-aware sampling; deblur (Restormer/NAFNet); CLAHE exposure norm; lens undistort | FFmpeg decode; pyexiftool/mavlink for metadata |
| S2' | Dynamic masking | **SAM 2** (video mask propagation) + Mask2Former class masks | MonST3R/DAS3R if heavy motion |
| S3' | Metric georeferenced poses | **GVINS** (tight GNSS-VI) or **VINS-Fusion + PPK** | ORB-SLAM3-VI; COLMAP `pose_prior_mapper` for offline |
| S4' | Dense reconstruction | **VGGT + VGGT-Long** (proven) or **GeoFF3D** (UAV-native) or **MapAnything** (native metric) | chunk+loop-close mandatory; Metric3D-v2 densify |
| S5' | Georeference/anchor | **GNSS-aided BA** (COLMAP/Metashape) → Umeyama fallback | model lever-arm + time-sync |
| S6' | Mesh + texture | **RaDe-GS / 2DGS / GOF**; **CityGaussianV2** (large) → TSDF (Open3D) + `mvs-texturing` | Mip-Splatting anti-alias; DSM/DTM via CSF ground filter; ortho via mesh reprojection |
| S7' | Semantics | **PTv3** (3D) + fused 2D seg (Mask2Former/UAVid-tuned) | Superpoint Transformer for scale |
| S8' | Serve/visualize | **Potree** (point cloud), **CesiumJS / 3D Tiles** (mesh/geo), splat viewer | measurement tools; GeoTIFF/LAZ/glTF export |

---

## 8. Handling each key challenge (challenge → concrete technique)

| # | Challenge | Concrete mitigation in our pipeline |
|---|-----------|--------------------------------------|
| i | **Limited viewing angles** | Feed-forward priors (VGGT/GeoFF3D) that infer geometry from learned priors where triangulation fails; DroneSplat's limited-view handling; **capture guidance**: add slight gimbal obliquity / gentle yaw weave / cross-strip when mission allows (biggest single lever). |
| ii | **Motion blur & compression** | Blur-aware keyframe selection; deblurring (Restormer/NAFNet); learned dense matchers (MASt3R) tolerant to degraded frames; IMU bridges short blur gaps in VIO; DPVO's learned tracking is blur-robust. |
| iii | **Variable illumination/shadows** | Per-image **appearance embeddings** (VastGaussian-style decoupling, NeRF-W idea); exposure normalization (CLAHE); shadow-robust matching; WildGaussians for in-the-wild appearance. |
| iv | **Dynamic objects** | **SAM 2** track+mask movers across frames → exclude from recon; dynamic-aware geometry (MonST3R/CUT3R/DAS3R); distractor-robust GS (SpotLessSplats/DroneSplat). |
| v | **GPS inaccuracy & sensor noise** | Robust **GNSS-aided BA** with per-observation covariance + RANSAC; tight VIO+GNSS (GVINS) rejects outliers & bridges dropouts; baro vertical factor; RTK/PPK when available. |
| vi | **Near-real-time** | Two-tier design: streaming feed-forward (CUT3R/StreamVGGT) / Photo-SLAM on edge for live preview; GPU-accelerated refined tier; chunked processing pipelined with flight. |
| vii | **Occluded surfaces** | Monocular depth **completion** in unseen regions; planarity/symmetry priors for façades/roofs; generative splat priors; **explicit hole/confidence reporting** rather than silent fabrication. |
| viii | **Metric accuracy without GCPs** | Direct georeferencing from **RTK/PPK + IMU + calibrated intrinsics** via GNSS-aided BA; honest accuracy budget (§9); optional single GCP / vertical prior to kill doming. |

---

## 9. Metric accuracy & georeferencing strategy (no GCPs)

**The metric-signal hierarchy (most reliable first):** ① RTK/PPK GNSS antenna positions → absolute datum + scale; ② IMU inertial integration → local metric scale + attitude; ③ barometric altitude → vertical prior; ④ calibrated camera intrinsics → removes focal↔depth ambiguity; ⑤ *(last)* monocular metric depth → densification/cross-check only. **Do not let vision set the scale when GNSS/IMU can.**

**Expected absolute accuracy WITHOUT ground control points:**

| GNSS mode | Horizontal | Vertical | Verdict |
|-----------|-----------|----------|---------|
| **Standalone / consumer GNSS geotags** (typical non-RTK EXIF) | ~**1–5 m** | ~**several m** (worse) | Coarse mapping only; bowl/doming likely. **Ceiling set by GNSS, not vision.** |
| **RTK / PPK onboard** (corrected antenna) | ~**1–3 cm** (≈1–2× GSD) | ~**3–8 cm** | Rivals GCP-based accuracy; largely removes need for a GCP field. |
| **RTK/PPK + 1 GCP or vertical prior or oblique/cross lines** | ~1–3 cm | ~**2–4 cm** | Removes residual vertical systematic bias. |

*Figures are consensus values from the UAV-photogrammetry literature — Forlani et al., Remote Sensing 2018 ([10.3390/rs10020311](https://www.mdpi.com/2072-4292/10/2/311)); Štroner et al., Remote Sensing 2021 ([13/7/1336](https://www.mdpi.com/2072-4292/13/7/1336)); Sanz-Ablanedo et al. 2018 ([10/10/1606](https://www.mdpi.com/2072-4292/10/10/1606)). DJI Phantom 4 RTK module spec: 1 cm + 1 ppm horizontal / 1.5 cm + 1 ppm vertical.*

**Georeferencing method (recommended):** tight **VIO+GNSS (GVINS / VINS-Fusion + PPK)** for real-time metric poses → **GNSS-aided bundle adjustment** (COLMAP `pose_prior_mapper` with position covariances, or Metashape) for a refined, self-calibrated block → **Umeyama 7-DoF similarity** as a loose fallback/QA cross-check.

**Two failure modes to engineer against (both from the literature):**
1. **"Doming"/bowl vertical error** — from near-parallel nadir-only views + under-modeled lens distortion. *Fix:* robust in-BA self-calibration **plus** any of: slight obliquity, a cross/perpendicular strip, altitude variation, or a single vertical prior/GCP.
2. **Scale observability on a straight line** — a single collinear pass makes similarity scale poorly observed and full Sim(3) alignment ill-conditioned. *Fix:* IMU metric scale (doesn't need spread), GeoFF3D's collinearity-aware anchoring, and (if possible) a gentle non-collinear flight component.

**Also model:** camera↔GNSS-antenna **lever arm** and camera/GNSS **time synchronization** (offset × ground speed = along-track bias) — getting these wrong injects systematic error even with perfect RTK.

**Honest bottom line for judges:** *with RTK/PPK + IMU + calibrated intrinsics + tight fusion, expect ~1–3 cm horizontal / few-cm vertical with zero-to-one GCP; with only standalone GNSS, expect meter-level regardless of the vision method.* State which regime the demo is in.

---

## 10. Recommended tech stack

| Layer | Choice |
|-------|--------|
| **Core language** | Python 3.11 (orchestration/ML) + C++/CUDA for hot paths (SLAM, splat rasterization) |
| **DL framework** | PyTorch 2.x; CUDA 12.x; xFormers/FlashAttention for VGGT-class memory |
| **Feed-forward recon** | VGGT + VGGT-Long, GeoFF3D, MapAnything (per §6.1); DUSt3R/MASt3R utils |
| **Depth** | Metric3D v2 / UniDepthV2 / Depth Pro; Video Depth Anything for temporal |
| **Pose/VIO** | GVINS or VINS-Fusion (ROS 2); COLMAP/GLOMAP for offline BA |
| **Gaussian splatting / mesh** | gsplat / nerfstudio; 2DGS, RaDe-GS, GOF, CityGaussianV2; Open3D (TSDF), `mvs-texturing` |
| **Segmentation** | SAM 2, Mask2Former/OneFormer (MMSegmentation/Detectron2), Point Transformer V3 (Pointcept) |
| **Geospatial** | GDAL/rasterio (GeoTIFF/COG), PDAL/laspy (LAS/LAZ), pyproj (CRS/UTM), CloudCompare |
| **DSM/DTM** | CSF / cloth-simulation ground filter; point-cloud-to-raster |
| **Serving/viz** | Potree (point cloud), CesiumJS + 3D Tiles (geo mesh), web splat viewer, deck.gl |
| **Edge/live** | NVIDIA Jetson Orin (Photo-SLAM/StreamVGGT); GStreamer video ingest |
| **Refined compute** | RTX 4090 / A100-class GPU (ground station or cloud) |
| **Orchestration** | FastAPI service + task queue; Docker; optional ROS 2 for sensor sync |

---

## 11. Datasets for training & validation

| Purpose | Datasets |
|---------|----------|
| **Drone video + GPS/pose (most relevant)** | **UseGeo** (UAV images + poses + LiDAR + GT depth; [site](https://usegeo.fbk.eu)); **Mid-Air** (synthetic drone + GPS/IMU/depth/semantics); **TartanAir** (synthetic flights, GT poses, hard VO) |
| **Aerial feed-forward recon benchmark** | **UAVFF3D** (170k+ real / 370k+ synthetic UAV, oblique/height ambiguity — arXiv 2605.17942) |
| **MVS / recon benchmarks** | Tanks and Temples, ETH3D, DTU, **BlendedMVS** (incl. aerial), **WHU MVS/Stereo** (aerial) |
| **Large-scale aerial scenes** | **Mill19** (Building/Rubble), **UrbanScene3D**, **MatrixCity**, **GauU-Scene** |
| **Aerial 2D semantics** | **UAVid** (4K UAV video, 8 classes), **ISPRS Vaihingen/Potsdam** |
| **Aerial/urban 3D point-cloud semantics** | **SensatUrban** (~3B pts, 13 classes), **STPLS3D**, **Hessigheim 3D (H3D)**, **DALES** |
| **Synthetic dense supervision** | **Hypersim** (indoor only — domain gap to aerial; pretraining only) |

**Validation mapping:** geometry → UseGeo/WHU-MVS/ETH3D/T&T; large-scale aerial → Mill19/UrbanScene3D/MatrixCity; 2D semantics → UAVid/ISPRS; 3D semantics → SensatUrban/STPLS3D/H3D; georef accuracy → RTK/GCP checkpoints.

---

## 12. Evaluation & testing protocol (expands §5 into a runnable plan)

1. **Ground truth sources:** survey-grade GCP/checkpoints (RTK rover) for C1–C2; reference LiDAR or terrestrial-scan mesh for Chamfer/F-score (C3–C4); hand-labeled frames/clouds for C5.
2. **Held-out checkpoints:** reserve ≥5–10 checkpoints *not* used in processing; report horizontal/vertical RMSE (C1–C2).
3. **Geometry metrics:** Chamfer distance & F-score @ threshold vs reference; point density; completeness/coverage % + hole map (C3). Novel-view **PSNR/SSIM/LPIPS** on held-out frames (C4).
4. **Measurement accuracy:** compare known distances/heights/areas (building height, road width) vs tape/survey; report % error.
5. **Semantics:** per-class IoU + mIoU on the 5 classes; dynamic-removal precision/recall (C5).
6. **Robustness ablations (C6):** inject synthetic **motion blur**, **exposure shifts**, **added dynamic objects**, and **GPS noise** (e.g., ±3 m) → measure metric degradation. Report graceful vs catastrophic.
7. **Single-pass gap (C8):** run the *same* scene as (a) single pass and (b) multi-pass grid through ODM/Metashape; quantify the accuracy/completeness gap.
8. **Speed (C7):** time-to-first-preview; end-to-end minutes per minute of video; real-time factor; edge vs cloud.
9. **Report everything with uncertainty** and state the GNSS regime (standalone vs RTK/PPK).

---

## 13. Hackathon MVP scoping & phased build plan

**Phase 0 — Proof of concept (de-risk the core):** run **VGGT** on ~100 frames of an existing drone clip (or UAVFF3D/UseGeo) → up-to-scale point cloud in a viewer. Proves the feed-forward core works on aerial. *Fallback if VGGT OOMs: chunk with VGGT-Long or use CUT3R streaming.*

**Phase 1 — MVP (the demo spine):** frame sampling → VGGT(+VGGT-Long) recon → **Umeyama alignment to GPS** for metric scale → export **georeferenced point cloud (LAZ)** + a **2DGS/RaDe-GS textured mesh** → Potree/Cesium viewer with a **distance/height measurement** tool. This alone satisfies visualization + measurement + georeferencing.

**Phase 2 — Differentiators (win points):** add **SAM 2 dynamic-object removal** (dramatic visual before/after); **PTv3 semantic layers** (5 classes); **DSM/DTM + orthomosaic** export; **accuracy report** with checkpoint RMSE; **GNSS-aided BA** refinement.

**Phase 3 — Polish & story:** **live situational tier** (Photo-SLAM/StreamVGGT progressive preview) for the "single opportunity / near-real-time" narrative; robustness ablation slide (blur/GPS-noise); side-by-side vs ODM multi-pass baseline.

**What to demo vs. claim honestly:**
- ✅ *Demo:* end-to-end single clip → georeferenced measurable mesh + semantics + live preview.
- ✅ *Claim:* cm-level accuracy **only** if you actually have RTK/PPK data; otherwise state meter-level and show the *relative* measurement accuracy.
- ⚠ *Don't overclaim:* full façade reconstruction from pure nadir (physically impossible — show it as a known limitation + capture recommendation), or metric scale from monocular depth alone.

---

## 14. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VGGT/feed-forward **OOM** on long flights | High | High | Chunk + loop-close (VGGT-Long); streaming (CUT3R); cap frames per chunk |
| **Aerial domain gap** (models trained indoor/driving) | High | Med | Validate on UAVFF3D/UseGeo early; fine-tune if time; keep classical MVS fallback |
| **No RTK data** → only meter-level accuracy | Med | High | Be explicit about regime; emphasize *relative* accuracy; offer PPK workflow |
| **Façades unrecoverable** from nadir single pass | High | Med | Frame as physical limitation; capture-geometry recommendation (obliquity); depth completion + confidence map |
| **Dynamic objects** ghost the model | Med | Med | SAM 2 masking + dynamic-aware recon; show before/after |
| Very new UAV methods (**GeoFF3D**, 2026 papers) **unverified/no stable code** | Med | Med | Default to proven VGGT stack; treat GeoFF3D as upside, verify repo before relying |
| **Real-time vs quality** can't both be maxed | High | Low | Two-tier design makes this explicit, not a failure |
| Compute/hardware limits at demo | Med | Med | Pre-render heavy results; live tier on Jetson; cloud GPU fallback |

---

## 15. References (consolidated; grouped by topic)

> ⚠ **Verify the newest (2026) arXiv IDs before formal citation** — several were surfaced from very recent listings and could not be fully cross-checked here (flagged inline where relevant).

**Feed-forward reconstruction:** DUSt3R [2312.14132](https://arxiv.org/abs/2312.14132) · MASt3R [2406.09756](https://arxiv.org/abs/2406.09756) · MASt3R-SfM [2409.19152](https://arxiv.org/abs/2409.19152) · Pow3R [2503.17316](https://arxiv.org/abs/2503.17316) · VGGT [2503.11651](https://arxiv.org/abs/2503.11651) ([code](https://github.com/facebookresearch/vggt)) · VGGT-Long [2507.16443](https://arxiv.org/abs/2507.16443) · VGGT-X [2509.25191](https://arxiv.org/abs/2509.25191) · Pi3 [2507.13347](https://arxiv.org/abs/2507.13347) · CUT3R [2501.12387](https://arxiv.org/abs/2501.12387) · Spann3R [2408.16061](https://arxiv.org/abs/2408.16061) · StreamVGGT [2507.11539](https://arxiv.org/abs/2507.11539) · Fast3R [2501.13928](https://arxiv.org/abs/2501.13928) · MV-DUSt3R+ [2412.06974](https://arxiv.org/abs/2412.06974) · MapAnything [2509.13414](https://arxiv.org/abs/2509.13414) · MonST3R [2410.03825](https://arxiv.org/abs/2410.03825) · GeoFF3D (arXiv 2608.28288, [code](https://github.com/yanxian-ll/GeoFF3D)) · UAVFF3D (arXiv 2605.17942) · survey [2507.14501](https://arxiv.org/abs/2507.14501)

**Metric depth:** Metric3D v2 [2404.15506](https://arxiv.org/abs/2404.15506) · UniDepthV2 [2502.20110](https://arxiv.org/abs/2502.20110) · Depth Pro [2410.02073](https://arxiv.org/abs/2410.02073) · Depth Anything V2 [2406.09414](https://arxiv.org/abs/2406.09414) · MoGe-2 [2507.02546](https://arxiv.org/abs/2507.02546) · Video Depth Anything [2501.12375](https://arxiv.org/abs/2501.12375) · ZoeDepth [2302.12288](https://arxiv.org/abs/2302.12288) · Marigold [2312.02145](https://arxiv.org/abs/2312.02145)

**SLAM / VIO / GNSS fusion:** ORB-SLAM3 [2007.11898](https://arxiv.org/abs/2007.11898) · VINS-Fusion ([code](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)) · OpenVINS ([code](https://github.com/rpng/open_vins)) · DROID-SLAM [2108.10869](https://arxiv.org/abs/2108.10869) · DPVO [2208.04726](https://arxiv.org/abs/2208.04726) · DPV-SLAM [2408.01654](https://arxiv.org/abs/2408.01654) · GVINS [2103.07899](https://arxiv.org/abs/2103.07899) · DM-VIO [2201.04114](https://arxiv.org/abs/2201.04114) · MASt3R-SLAM [2412.12392](https://arxiv.org/abs/2412.12392)

**Gaussian splatting / NeRF / mesh:** 3DGS [2308.04079](https://arxiv.org/abs/2308.04079) · Mip-Splatting [2311.16493](https://arxiv.org/abs/2311.16493) · 2DGS [2403.17888](https://arxiv.org/abs/2403.17888) · GOF [2404.10772](https://arxiv.org/abs/2404.10772) · RaDe-GS [2406.01467](https://arxiv.org/abs/2406.01467) · SuGaR [2311.12775](https://arxiv.org/abs/2311.12775) · CityGaussianV2 [2411.00771](https://arxiv.org/abs/2411.00771) · VastGaussian [2402.17427](https://arxiv.org/abs/2402.17427) · Hierarchical-3DGS [2406.12080](https://arxiv.org/abs/2406.12080) · Octree-GS [2403.17898](https://arxiv.org/abs/2403.17898) · On-the-fly-NVS [2506.05558](https://arxiv.org/abs/2506.05558) · Photo-SLAM [2311.16728](https://arxiv.org/abs/2311.16728) · MonoGS [2312.06741](https://arxiv.org/abs/2312.06741) · DroneSplat [2503.16964](https://arxiv.org/abs/2503.16964) · Mega-NeRF [2112.10703](https://arxiv.org/abs/2112.10703)

**Classical SfM/MVS:** COLMAP ([site](https://colmap.github.io)) · GLOMAP [2407.20219](https://arxiv.org/abs/2407.20219) · OpenMVG/OpenMVS · OpenDroneMap ([site](https://www.opendronemap.org))

**Semantics / dynamic removal:** SAM 2 [2408.00714](https://arxiv.org/abs/2408.00714) · Mask2Former [2112.01527](https://arxiv.org/abs/2112.01527) · OneFormer [2211.06220](https://arxiv.org/abs/2211.06220) · SegFormer [2105.15203](https://arxiv.org/abs/2105.15203) · Point Transformer V3 [2312.10035](https://arxiv.org/abs/2312.10035) · Superpoint Transformer [2306.08045](https://arxiv.org/abs/2306.08045) · RobustNeRF [2302.00833](https://arxiv.org/abs/2302.00833) · SpotLessSplats [2406.20055](https://arxiv.org/abs/2406.20055) · WildGaussians [2407.08447](https://arxiv.org/abs/2407.08447) · DAS3R [2412.19584](https://arxiv.org/abs/2412.19584)

**Georeferencing accuracy studies:** Forlani et al. RS 2018 ([rs10020311](https://www.mdpi.com/2072-4292/10/2/311)) · Štroner et al. RS 2021 ([13/7/1336](https://www.mdpi.com/2072-4292/13/7/1336)) · Sanz-Ablanedo et al. RS 2018 ([10/10/1606](https://www.mdpi.com/2072-4292/10/10/1606))

**Datasets:** UAVid ([uavid.nl](https://uavid.nl)) · ISPRS Vaihingen/Potsdam · SensatUrban [2009.03137](https://arxiv.org/abs/2009.03137) · STPLS3D [2203.09065](https://arxiv.org/abs/2203.09065) · UseGeo ([site](https://usegeo.fbk.eu)) · Mid-Air · TartanAir · Mill19/Mega-NeRF · UrbanScene3D [2107.04286](https://arxiv.org/abs/2107.04286) · BlendedMVS [1911.10127](https://arxiv.org/abs/1911.10127) · Tanks and Temples · ETH3D · DTU

---

*End of dossier. This is a living document — the §6.1 UAV-specific methods (GeoFF3D) and the 2026 Gaussian-splatting papers are the fastest-moving frontier; re-check their repos/benchmarks before committing the architecture.*
