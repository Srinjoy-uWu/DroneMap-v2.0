# Improvement Plan — Verification & Amendments

Review of the five-phase plan against the audit findings, the measured state of
`data/runs`, NTRO's requirement, and the SIH26158 solution description.

**Verdict on the plan: correctly prioritised, with three substantive gaps that
would each have let a broken output through.** Phase 1 first is right. The gaps
are recorded below as amendments, with the measurements that justify them.

---

## Summary of the plan as submitted

| Phase | Goal | Assessment |
|---|---|---|
| 1 | Stop misleading outputs | **Correct and correctly first.** Under-specified: see A1–A3. |
| 2 | Make terrain output useful | **Correct, but must move ahead of Phase 3.** See A5. |
| 3 | Improve camera reconstruction | Correct, lower priority than stated. |
| 4 | Improve field capture | Correct and complete. Blocks Phase 5. |
| 5 | Judge-ready validation | Correct framing. Currently unachievable: see A6. |

---

## A1 — Phase 1 does not say *what* to measure, and counts cannot work

"Strengthen pose and dense-cloud failure gates" is the right instinct, but every
gate in the codebase before this work was a **count**: `min_registered_fraction`,
`min_sparse_points`, `min_dense_points`. Counts cannot distinguish a building
from a sheet of tarmac.

Measured proof, run `meow` (from `0904.mp4`, no telemetry):

| Gate | Threshold | Actual | Result |
|---|---|---|---|
| `min_registered_fraction` | 0.55 | **1.00** | pass |
| `min_sparse_points` | 20 | **15,636** | pass |
| `min_dense_points` | 5,000 | **129,750** | pass |

Every gate passed. The exported `model.glb` has PCA singular ratios
`[1, 0.427, 0.0379]` — a flat warped sheet of road surface. The system called it
a success and offered it for download as a 3D model.

**Amendment: gate on scale-free geometry, not counts.** Implemented in
`src/dronemap/quality.py`:

| Metric | Measures | `meow` |
|---|---|---|
| median per-point triangulation angle | can any point be intersected in 3D | 7.8° |
| max baseline ÷ median scene depth | did the platform translate enough | 0.345 |
| per-view depth dynamic range (p95/p5) | more than one depth slab | 1.62× |
| PCA planarity (3rd singular ratio) | surface, or volume | **0.0379** |
| forward-motion ratio | travel along the optical axis (epipole degeneracy) | 0.49 |
| path efficiency (baseline ÷ path length) | hover / out-and-back detection | 0.46 |

All are ratios or angles, so they behave identically on relative and
georeferenced models — which matters because most runs never reach georef.

---

## A2 — Flatness is ambiguous, and the plan's framing would reject valid terrain

This is the gap most likely to have caused real damage.

A flat result has **two** causes that count- or planarity-based gating cannot
separate:

- the reconstruction failed to recover depth (the `meow` carpet), **or**
- the site is genuinely flat (a beach, an airfield, a field).

Measured, from the existing corpus:

| Run | triangulation | baseline/depth | planarity | Correct verdict |
|---|---|---|---|---|
| `meow` | 7.8° | 0.345 | 0.0379 | ground surface only — buildings not recovered |
| `brighton_beach_survey` | **30.7°** | **1.785** | 0.0396 | **valid terrain product — a beach is flat** |

Near-identical planarity; opposite meanings. A planarity gate alone rejects the
beach survey, which is a perfectly good capture.

**Amendment: the disambiguator is the parallax evidence.** `assess_structure()`
takes the sparse geometry and applies:

- adequate parallax + flat result → the site really is flat → **TERRAIN_2_5D**, a
  legitimate deliverable
- marginal parallax + flat result → reconstruction failure → **REJECT**

Regression-tested both directions in `tests/test_quality.py`.

---

## A3 — "Strengthen the gates" misses false accepts entirely

Thresholds phrased as minimums only catch failures that make numbers *small*.
Numerical collapse makes them *large*.

Run `smoke03`: `depth_p50 = 1.66e-06`, so baseline ÷ depth = **10,089,473**. The
scene had collapsed onto the camera centres. Every minimum-threshold gate passed,
and it scored as the **best capture in the dataset** — the only ACCEPT_3D.

**Amendment: every ratio gate needs an upper sanity bound.** Added
`max_baseline_depth_ratio` (20.0) and `min_depth_over_baseline` (0.02). Real
aerial capture never flies a baseline many times the scene distance.

---

## A4 — "Test against the five existing failed videos" is the wrong test set

The already-failing videos are the easy case; they fail without help. The
dangerous case is a run that **succeeds and should not** — that is the entire
subject of the audit.

**Amendment: the acceptance criterion is "no false accepts across the whole run
corpus," not "the five known failures still fail."** Full sweep of all 29 runs
in `data/runs`:

```
ACCEPT_3D      0
TERRAIN_2_5D   8
REJECT        21
```

Zero ACCEPT_3D is the honest reading of this corpus: **no run in the project's
history has ever produced a validated 3D structural model.** Every previously
"successful" run was either a terrain surface or a failure.

The gate is proven to be capable of accepting good geometry — see
`test_well_flown_capture_is_accepted` and `test_merged_good_capture_is_accepted`.
It rejects everything here because everything here deserves it.

---

## A5 — Phase 2 is load-bearing and must precede Phase 3

Consequence of A2 and A4: once the system stops overclaiming, the honest verdict
for most single-pass near-nadir captures is **TERRAIN_2_5D**. Phase 2 is what
converts that verdict from "nothing to show" into an actual deliverable (DSM →
mesh → draped orthomosaic).

Without Phase 2, an honest system outputs almost nothing, which reads as a
regression rather than as the integrity fix it is.

**Amendment: reorder to 1 → 2 → 4 → 3 → 5.** Phase 4 (capture protocol) moves
ahead of Phase 3 because no amount of camera-prior work in Phase 3 rescues a
7-second hover; only different footage does.

`src/dronemap/terrain.py` already exists with `_fit_ground_plane()` and
`reconstruct_terrain_mesh()` and is the right foundation.

---

## A6 — Phase 5 cannot be completed without new footage

"One verified terrain demo and one verified structure demo" is the right target,
and "the system does not fabricate terrain when video evidence is insufficient"
is a genuinely strong safety story for NTRO — a system that refuses to guess is
more defensible than one that always emits a mesh.

But it is only credible alongside at least one accepted output, and there are
currently **zero** ACCEPT_3D runs. The blocker is input, not code: no video in
`data/runs` was flown to a geometry that supports 3D structure.

**Amendment: Phase 5 is gated on Phase 4 being executed in the field.** The
structure demo needs new footage meeting the Phase 4 spec — continuous lateral or
orbital pass, 70–80% forward overlap, 65–75° gimbal, 60–90 s minimum, two or
three known-size reference objects for scale validation without RTK/PPK.

---

## A7 — New finding: the viewer was dead, and the plan has no phase for it

Not covered anywhere in the five phases: **the UI could not display any model at
all.**

Root cause: `vendor/three/addons/loaders/GLTFLoader.js:68` imports
`toTrianglesDrawMode` from `../utils/BufferGeometryUtils.js`, which was never
vendored. That file 404s, so the ES module graph fails to resolve, `viewer.js`
never executes, `window.load3DModel` is never defined, and the viewport sits on
the neutral "No Project Selected" card forever — indistinguishable from an
unselected project.

This is the same assert-don't-measure pathology in the front end: the entire
engine sat inside a bare `if (canvas) { ... }` with no `catch` and no error
surface, so every failure mode rendered as the idle state.

Fixed: vendored the genuine upstream `BufferGeometryUtils.js` (three r166, 31,768
bytes, imports only `three`); wrapped the engine in a real `try`/`catch` that
reports to the overlay; added a classic-script boot watchdog in `index.html` that
surfaces resource 404s and reports if the module never signals readiness; replaced
the one-shot unguarded `resize()` with a zero-size guard plus `ResizeObserver`;
and disabled HTTP caching for `.html`/`.js`/`.css` (`_NoCacheStaticFiles`) so a
stale shell can never again point at an unreachable CDN. Verified in-browser:
`viewerReady: true`, `load3DModel: function`, model renders.

---

## A8 — New finding: run status is re-derived from the project name

`derive_status_tier()` in `src/dronemap/api/server.py` classifies a run as
terrain via:

```python
or "terrain" in run_name.lower()
```

A substring match on the user-supplied project name is not a measurement — naming
a run `terrain_test` changes its reported status. `manifest.json` also has **no
top-level `status` key**, so overall run outcome is nowhere recorded and must be
guessed on every API request.

**Amendment: record the verdict once, at the point of measurement, and read it
back.** The pose stage now writes `03_pose/capture_quality.json` and records
`capture_verdict` in the manifest metrics. `derive_status_tier` should read that
recorded verdict instead of re-deriving from filenames.

---

## NTRO / SIH26158 compliance

The amendments strengthen rather than weaken compliance:

- **Single-pass requirement** — unchanged. Nothing here asks for a second flight;
  A2/A5 make the single-pass *terrain* product a first-class deliverable rather
  than a failed structure product.
- **Accuracy claims** — the audit's core problem was unearned claims. Metric
  claims now require both georeferencing and an accepted verdict.
- **Compliance matrix (C1–C10)** — each row must be *measured*. `criterion()` in
  `stage7_export.py` is the right mechanism; C10 is still hardcoded.
- **"Does not fabricate"** — now a demonstrable property with recorded numbers
  behind each verdict, not a claim.

---

## Status

**Done and verified**

- `src/dronemap/quality.py` — measurement engine and three-way verdict
- `QualityConfig` in `config.py` — thresholds calibrated against real runs
- `tests/test_quality.py` — 21 tests; full suite 119 passed
- Pose-stage gate wired in `stage3_pose.py` (`_assess_capture_quality`), running
  before the undistort/dense/mesh budget is spent
- Viewer fixed end-to-end (A7), confirmed rendering in a real browser
- Full-corpus sweep of all 29 runs (A4)

**Outstanding**

- Structure gate at mesh/export; block invalid GLB/DSM/measurement export
- Surface verdict + reasons in the UI and `report.html`; top-level manifest
  `status`; fix `derive_status_tier` name-substring (A8)
- Phase 2 terrain product; Phases 3–5
- Audit findings still live: #6 (`stage4_depth` nadir scale ~100×), #2
  (alignment RMSE frame mismatch, `stage3_georef.py:372`), measure-axis
  convention, `scripts/evaluate_synthetic.py`, C10, `docs/JURY_DEMO_SCRIPT.md`
