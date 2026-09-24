"""Regression tests for truthful coordinate-system labelling in Stage 7."""

from __future__ import annotations

import numpy as np

from dronemap.stage7_export import _guess_crs, _points_for_gis


def test_unaligned_run_is_explicitly_local_relative():
    assert _guess_crs({}) == "LOCAL_RELATIVE"


def test_local_points_are_not_changed():
    points = np.array([[1.0, 2.0, 3.0]])
    result = _points_for_gis(points, {})
    np.testing.assert_array_equal(result, points)


def test_ecef_points_convert_to_declared_projected_crs():
    # Approximate ECEF coordinate of the equator / Greenwich meridian.
    points = np.array([[6_378_137.0, 0.0, 0.0]])
    result = _points_for_gis(points, {
        "coordinate_frame": "ECEF",
        "output_crs": "EPSG:32631",
    })
    assert result.shape == (1, 3)
    assert np.isfinite(result).all()
    # At the central meridian of UTM zone 31N, Easting is roughly 166 km.
    assert 100_000 < result[0, 0] < 300_000
