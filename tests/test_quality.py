"""Tests for the capture-quality gate.

The gate's whole value is that it *discriminates*.  A gate that rejects
everything is as useless as one that accepts everything, so these tests pin
down both ends: a well-flown oblique orbit must reach ACCEPT_3D, and every
known failure mode must be caught and named.

The pose/structure thresholds are calibrated against real runs in
``data/runs``; the numbers quoted in the docstrings below are measured, not
invented, so a future threshold change that breaks a real capture fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dronemap.config import QualityConfig
from dronemap.quality import (
    SparseGeometry,
    StructureGeometry,
    Verdict,
    assess_pose,
    assess_structure,
    measure_structure,
)


@pytest.fixture
def cfg() -> QualityConfig:
    return QualityConfig()


def _good_sparse(**overrides) -> SparseGeometry:
    """Geometry of a properly flown oblique pass around a structure."""
    base = dict(
        n_registered=60,
        n_total=64,
        n_points3d=45_000,
        path_length=120.0,
        max_baseline=95.0,
        path_efficiency=0.79,
        forward_motion_ratio=0.35,
        depth_p5=40.0,
        depth_p50=110.0,
        depth_p95=180.0,
        depth_dynamic_range=4.5,
        baseline_depth_ratio=0.86,
        max_parallax_deg=46.0,
        median_tri_angle_deg=18.0,
        p90_tri_angle_deg=34.0,
        mean_track_length=7.4,
    )
    base.update(overrides)
    return SparseGeometry(**base)


def _good_structure(**overrides) -> StructureGeometry:
    """Shape of a cloud containing real above-ground structure."""
    base = dict(
        n_points=250_000,
        extent=(180.0, 150.0, 42.0),
        singular_ratios=(1.0, 0.72, 0.31),
        planarity=0.31,
        relief_p95=22.0,
        relief_ratio=0.18,
        above_ground_fraction=0.29,
    )
    base.update(overrides)
    return StructureGeometry(**base)


# ---------------------------------------------------------------------------
# The gate must be able to say yes
# ---------------------------------------------------------------------------


def test_well_flown_capture_is_accepted(cfg):
    assessment = assess_pose(_good_sparse(), cfg)
    assert assessment.verdict is Verdict.ACCEPT_3D
    assert assessment.accepted
    assert assessment.reasons == []


def test_good_structure_is_accepted(cfg):
    assessment = assess_structure(_good_structure(), cfg, sparse=_good_sparse())
    assert assessment.verdict is Verdict.ACCEPT_3D
    assert assessment.reasons == []


def test_merged_good_capture_is_accepted(cfg):
    sparse = _good_sparse()
    merged = assess_pose(sparse, cfg).merge(
        assess_structure(_good_structure(), cfg, sparse=sparse)
    )
    assert merged.verdict is Verdict.ACCEPT_3D
    assert merged.headline.startswith("Full 3D")


# ---------------------------------------------------------------------------
# Failure modes, each named
# ---------------------------------------------------------------------------


def test_hover_is_rejected_on_triangulation_angle(cfg):
    """A hovering platform cannot observe depth at any processing setting."""
    assessment = assess_pose(
        _good_sparse(median_tri_angle_deg=1.2, baseline_depth_ratio=0.04), cfg
    )
    assert assessment.verdict is Verdict.REJECT
    assert any("triangulation angle" in r for r in assessment.reasons)


def test_thin_baseline_demotes_to_terrain(cfg):
    """Marginal parallax supports a ground surface but not vertical structure."""
    assessment = assess_pose(
        _good_sparse(median_tri_angle_deg=4.0, baseline_depth_ratio=0.22), cfg
    )
    assert assessment.verdict is Verdict.TERRAIN_2_5D


def test_single_depth_slab_demotes_to_terrain(cfg):
    assessment = assess_pose(_good_sparse(depth_dynamic_range=1.1), cfg)
    assert assessment.verdict is Verdict.TERRAIN_2_5D
    assert any("single depth layer" in r for r in assessment.reasons)


def test_too_few_registered_images_is_rejected(cfg):
    assessment = assess_pose(_good_sparse(n_registered=6, n_total=6), cfg)
    assert assessment.verdict is Verdict.REJECT
    assert any("camera pose" in r for r in assessment.reasons)


def test_collapsed_reconstruction_is_rejected(cfg):
    """Regression: smoke03 had depth_p50=1.66e-06, so baseline/depth was 1e7.

    Every *minimum* threshold passed and the run scored as the best capture in
    the dataset.  The upper sanity bound is what catches it.
    """
    assessment = assess_pose(
        _good_sparse(
            depth_p5=1.0e-06,
            depth_p50=1.7e-06,
            max_baseline=16.8,
            baseline_depth_ratio=16.8 / 1.7e-06,
        ),
        cfg,
    )
    assert assessment.verdict is Verdict.REJECT
    assert any("degenerate" in r for r in assessment.reasons)


def test_forward_flight_warns_but_does_not_reject(cfg):
    """Flying down the optical axis is legitimate, just unreliable mid-frame."""
    assessment = assess_pose(_good_sparse(forward_motion_ratio=0.97), cfg)
    assert assessment.verdict is Verdict.ACCEPT_3D
    assert any("along the viewing direction" in w for w in assessment.warnings)


def test_low_path_efficiency_without_parallax_warns(cfg):
    """Regression: a 7 s clip logged 27 units of path but a 12 unit baseline."""
    assessment = assess_pose(
        _good_sparse(path_efficiency=0.20, median_tri_angle_deg=3.0,
                     baseline_depth_ratio=0.11),
        cfg,
    )
    assert any("retraced its own line" in w for w in assessment.warnings)


def test_low_path_efficiency_with_strong_parallax_is_read_as_an_orbit(cfg):
    """An orbit ends near where it started, and that is not a defect.

    Path efficiency is ambiguous the same way planarity is: a hover and an orbit
    both score low. The orbit fixture measures 15% efficiency with 18.2deg
    triangulation and reconstructs fully, so low efficiency plus strong parallax
    must not be reported as wasted flight.
    """
    assessment = assess_pose(
        _good_sparse(path_efficiency=0.15, median_tri_angle_deg=18.2,
                     baseline_depth_ratio=1.45),
        cfg,
    )
    assert assessment.verdict is Verdict.ACCEPT_3D
    assert not any("retraced its own line" in w for w in assessment.warnings)
    assert any("orbit around the subject" in r for r in assessment.reasons)


# ---------------------------------------------------------------------------
# The flat-sheet ambiguity: failure vs. a genuinely flat site
# ---------------------------------------------------------------------------


def test_flat_sheet_from_bad_parallax_is_rejected(cfg):
    """The 0904.mp4 carpet: planarity 0.038 off a near-hover capture."""
    sparse = _good_sparse(median_tri_angle_deg=2.5, baseline_depth_ratio=0.12)
    assessment = assess_structure(
        _good_structure(planarity=0.038, singular_ratios=(1.0, 0.43, 0.038)),
        cfg,
        sparse=sparse,
    )
    assert assessment.verdict is Verdict.REJECT
    assert any("flat sheet" in r for r in assessment.reasons)
    assert any("marginal" in r for r in assessment.reasons)


def test_flat_site_with_good_parallax_is_terrain_not_reject(cfg):
    """Regression: brighton_beach_survey measured 30.7deg / 1.79x / 0.040.

    A beach really is flat.  Rejecting it would be as wrong as accepting the
    hover carpet -- the parallax evidence is what separates the two.
    """
    sparse = _good_sparse(median_tri_angle_deg=30.7, baseline_depth_ratio=1.79)
    assessment = assess_structure(
        _good_structure(planarity=0.0396, singular_ratios=(1.0, 0.55, 0.0396)),
        cfg,
        sparse=sparse,
    )
    assert assessment.verdict is Verdict.TERRAIN_2_5D
    assert assessment.accepted
    assert any("genuinely flat" in r for r in assessment.reasons)


def test_flat_sheet_without_sparse_context_is_rejected(cfg):
    """With no parallax evidence to vouch for it, flatness is not believed."""
    assessment = assess_structure(_good_structure(planarity=0.03), cfg, sparse=None)
    assert assessment.verdict is Verdict.REJECT


def test_sparse_structure_is_rejected(cfg):
    assessment = assess_structure(_good_structure(n_points=800), cfg)
    assert assessment.verdict is Verdict.REJECT


# ---------------------------------------------------------------------------
# measure_structure on real geometry
# ---------------------------------------------------------------------------


def test_measure_structure_detects_a_plane():
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.uniform(-50, 50, 20_000),
            rng.uniform(-50, 50, 20_000),
            rng.normal(0, 0.05, 20_000),
        ]
    )
    geom = measure_structure(pts)
    assert geom.planarity < 0.01
    assert geom.relief_ratio < 0.01


def test_measure_structure_detects_buildings_on_ground():
    """Ground plane plus three boxes must not read as planar."""
    rng = np.random.default_rng(1)
    ground = np.column_stack(
        [
            rng.uniform(-50, 50, 20_000),
            rng.uniform(-50, 50, 20_000),
            rng.normal(0, 0.1, 20_000),
        ]
    )
    boxes = []
    for cx, cy in [(-25, -25), (0, 10), (30, -5)]:
        boxes.append(
            np.column_stack(
                [
                    rng.uniform(cx - 8, cx + 8, 8_000),
                    rng.uniform(cy - 8, cy + 8, 8_000),
                    rng.uniform(0, 20, 8_000),
                ]
            )
        )
    geom = measure_structure(np.vstack([ground, *boxes]))
    assert geom.planarity > 0.05
    assert geom.relief_ratio > 0.04
    assert geom.above_ground_fraction > 0.05


def test_measure_structure_is_rotation_invariant():
    """The pipeline's up-axis is not consistent across stages, so the shape
    metrics must not depend on which axis a stage calls "up"."""
    rng = np.random.default_rng(2)
    pts = np.column_stack(
        [
            rng.uniform(-40, 40, 15_000),
            rng.uniform(-40, 40, 15_000),
            rng.uniform(0, 12, 15_000),
        ]
    )
    theta = 0.9
    rot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    a = measure_structure(pts)
    b = measure_structure(pts @ rot.T)
    assert a.planarity == pytest.approx(b.planarity, rel=1e-6)
    assert a.relief_ratio == pytest.approx(b.relief_ratio, rel=1e-6)


def test_measure_structure_handles_degenerate_input():
    assert measure_structure(np.empty((0, 3))).n_points == 0
    assert measure_structure(np.zeros((4, 3))).n_points == 0
    assert measure_structure(np.full((100, 3), np.nan)).n_points == 0


# ---------------------------------------------------------------------------
# Verdict algebra
# ---------------------------------------------------------------------------


def test_worst_of_keeps_the_lower_verdict():
    assert Verdict.ACCEPT_3D.worst_of(Verdict.REJECT) is Verdict.REJECT
    assert Verdict.TERRAIN_2_5D.worst_of(Verdict.ACCEPT_3D) is Verdict.TERRAIN_2_5D
    assert Verdict.REJECT.worst_of(Verdict.REJECT) is Verdict.REJECT


def test_assessment_is_json_serialisable(cfg):
    import json

    payload = assess_pose(_good_sparse(), cfg).to_dict()
    assert json.loads(json.dumps(payload))["verdict"] == "accept_3d"


def test_non_finite_metrics_serialise_as_null(cfg):
    geom = _good_sparse(baseline_depth_ratio=float("nan"))
    assert geom.to_dict()["baseline_depth_ratio"] is None
