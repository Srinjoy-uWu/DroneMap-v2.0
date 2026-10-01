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


def test_viewer_frame_transform_properties():
    from dronemap.stage7_export import _viewer_frame_transform

    # Case 1: unaligned / local relative
    rot_local, info_local = _viewer_frame_transform({"coordinate_frame": "LOCAL_RELATIVE"})
    assert rot_local is None
    assert not info_local["applied"]

    # Case 2: ECEF transform
    transform = {
        "coordinate_frame": "ECEF",
        "georef_success": True,
        "scene_centroid_wgs84": {"lat": 59.5, "lon": 25.1, "alt_m": 10.0},
        "model_offset_m": [1000.0, 2000.0, 3000.0],
    }
    rot_ecef, info_ecef = _viewer_frame_transform(transform)
    assert info_ecef["applied"]
    # Check that rotation is orthonormal and proper (det = +1)
    np.testing.assert_allclose(rot_ecef @ rot_ecef.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rot_ecef), 1.0, atol=1e-6)
