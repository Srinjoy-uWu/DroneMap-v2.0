"""Tests for 2.5D terrain surface mesh reconstruction."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import trimesh

from dronemap.terrain import (
    _rotation_aligning,
    fit_ground_plane,
    reconstruct_terrain_mesh,
)


def test_fit_ground_plane_recovers_a_horizontal_plane():
    # Points in a horizontal plane z = 5.0 with small noise
    rng = np.random.default_rng(42)
    x = rng.uniform(-10, 10, 200)
    y = rng.uniform(-10, 10, 200)
    z = 5.0 + rng.normal(0, 0.01, 200)
    pts = np.column_stack([x, y, z])

    plane = fit_ground_plane(pts)
    assert plane.center[2] == pytest.approx(5.0, abs=0.1)
    # The normal must point *up*, not merely lie on the vertical axis. The
    # previous assertion was ``abs(abs(normal[2]) - 1.0) < 0.05``, which passed
    # for a normal pointing straight down -- and a downward normal inverts the
    # whole surface, turning buildings into pits.
    assert plane.normal[2] == pytest.approx(1.0, abs=0.05)
    assert plane.tilt_deg < 1.0
    assert plane.source == "fitted"


def test_buildings_do_not_tip_the_ground_plane():
    """The regression this fix exists for.

    A cloud that is 40 % ground and 60 % vertical facade has its least-variance
    axis nowhere near vertical, so fitting PCA to *all* points returns a plane
    standing on its side. Measured on the real runs, that produced ground
    normals tilted 60-85 deg. Fitting the lower envelope must not.
    """
    rng = np.random.default_rng(0)
    ground = np.column_stack([
        rng.uniform(-40, 40, 800),
        rng.uniform(-6, 6, 800),      # narrow footprint, as a single pass gives
        rng.normal(0.0, 0.02, 800),
    ])
    # A tall wall along the strip: lots of variance in Z, little across it.
    wall = np.column_stack([
        rng.uniform(-40, 40, 1600),
        rng.normal(5.0, 0.05, 1600),
        rng.uniform(0.0, 30.0, 1600),
    ])
    pts = np.vstack([ground, wall])

    naive = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, 0]
    naive_tilt = np.degrees(np.arccos(abs(naive[2])))
    assert naive_tilt > 45.0, "fixture must actually exercise the failure"

    plane = fit_ground_plane(pts)
    assert plane.tilt_deg < 5.0
    assert plane.normal[2] > 0.99


def test_a_wildly_tilted_fit_is_rejected_in_favour_of_the_prior():
    """A wrong plane is worse than no plane, and must announce itself."""
    rng = np.random.default_rng(1)
    # A pure vertical sheet: the only plane here is a wall, not a ground.
    pts = np.column_stack([
        rng.uniform(-20, 20, 600),
        rng.normal(0.0, 0.01, 600),
        rng.uniform(0.0, 40.0, 600),
    ])
    plane = fit_ground_plane(pts, max_tilt_deg=15.0)
    assert plane.source == "fit_rejected"
    assert plane.normal == pytest.approx(np.array([0.0, 0.0, 1.0]))
    assert plane.tilt_deg == 0.0


def test_rotation_aligning_is_never_a_reflection():
    """Including the antiparallel case, which used to return ``-I``.

    ``-np.eye(3)`` has determinant -1: it mirrors the scene and flips every
    face winding, so normals point inward and the mesh renders inside-out.
    """
    cases = [
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.3, -0.4, 0.866]), np.array([0.0, 0.0, 1.0])),
    ]
    for src, dst in cases:
        src = src / np.linalg.norm(src)
        R = _rotation_aligning(src, dst)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
        assert R @ src == pytest.approx(dst, abs=1e-9)
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)


def test_georeferenced_prior_preserves_relief(tmp_path: Path):
    """Rotating onto a bogus plane collapses the scene's true height.

    ``fixture_corridor_v1`` had 228 m of Z span turn into 22 m of "elevation"
    after the old rotation. Relief in the aligned frame is now reported so the
    collapse is visible as a number, and with a correct plane it should track
    the cloud's own Z span.
    """
    rng = np.random.default_rng(7)
    ground = np.column_stack([
        rng.uniform(-50, 50, 1200),
        rng.uniform(-50, 50, 1200),
        rng.normal(0.0, 0.05, 1200),
    ])
    tower = np.column_stack([
        rng.uniform(-5, 5, 800),
        rng.uniform(-5, 5, 800),
        rng.uniform(0.0, 60.0, 800),
    ])
    pts = np.vstack([ground, tower])
    colors = rng.integers(50, 200, (len(pts), 3), dtype=np.uint8)

    out = tmp_path / "relief.obj"
    metrics = reconstruct_terrain_mesh(
        points=pts, colors=colors, output_obj=out,
        grid_dim=40, max_grid_dim=60,
        up_hint=np.array([0.0, 0.0, 1.0]),
    )
    assert metrics["ground_tilt_deg"] < 2.0
    assert metrics["relief_m"] == pytest.approx(metrics["cloud_z_span_m"], rel=0.05)


def test_reconstruct_terrain_mesh(tmp_path: Path):
    rng = np.random.default_rng(42)
    # Create a gentle rolling terrain
    x = rng.uniform(-20, 20, 500)
    y = rng.uniform(-20, 20, 500)
    z = 0.1 * (x**2 + y**2) / 20.0
    pts = np.column_stack([x, y, z])
    cols = rng.integers(50, 200, (500, 3), dtype=np.uint8)

    out_obj = tmp_path / "terrain.obj"
    out_ply = tmp_path / "terrain.ply"

    metrics = reconstruct_terrain_mesh(
        points=pts,
        colors=cols,
        output_obj=out_obj,
        output_ply=out_ply,
        grid_dim=50,
        max_grid_dim=100,
    )

    assert out_obj.exists()
    assert out_ply.exists()
    assert metrics["n_vertices"] > 0
    assert metrics["n_faces"] > 0
    assert metrics["ground_plane_source"] in {"fitted", "prior", "fit_rejected"}

    # Verify mesh can be loaded by trimesh
    mesh = trimesh.load(str(out_obj), process=False)
    assert len(mesh.vertices) == metrics["n_vertices"]
    assert len(mesh.faces) == metrics["n_faces"]


def _images_txt(directory: Path, quats: list[tuple[float, float, float, float]]) -> Path:
    """A COLMAP images.txt with the alternating POINTS2D lines, as COLMAP writes."""
    lines = ["# Image list"]
    for i, (qw, qx, qy, qz) in enumerate(quats, start=1):
        lines.append(f"{i} {qw} {qx} {qy} {qz} 0 0 10 1 frame_{i:06d}.jpg")
        lines.append("100.0 200.0 -1 150.0 250.0 -1")
    p = directory / "images.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_up_hint_from_nadir_cameras_is_minus_z_in_colmap():
    """COLMAP's camera looks down its own +Z, so a nadir flight gives up = -Z.

    Without telemetry, COLMAP's world frame has no relation to gravity, and
    defaulting the prior to +Z is a guess dressed as a datum. The poses carry
    the answer: identity rotation means the camera views along world +Z, so up
    is world -Z.
    """
    import tempfile

    from dronemap.terrain import up_hint_from_cameras

    with tempfile.TemporaryDirectory() as d:
        p = _images_txt(Path(d), [(1.0, 0.0, 0.0, 0.0)] * 5)
        up = up_hint_from_cameras(p)
    assert up == pytest.approx(np.array([0.0, 0.0, -1.0]), abs=1e-9)


def test_up_hint_is_none_when_views_cancel():
    """Cameras pointing every which way have no consistent down to extract.

    ``None`` must not collapse to "+Z": that is the assumption this function
    exists to avoid making.
    """
    import tempfile

    from dronemap.terrain import up_hint_from_cameras

    # Two pairs of opposed viewing directions: the mean axis is ~0.
    opposed = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)] * 3
    with tempfile.TemporaryDirectory() as d:
        p = _images_txt(Path(d), opposed)
        assert up_hint_from_cameras(p) is None


def test_up_hint_from_missing_poses_is_none():
    import tempfile

    from dronemap.terrain import up_hint_from_cameras

    with tempfile.TemporaryDirectory() as d:
        assert up_hint_from_cameras(Path(d) / "images.txt") is None


def test_georeferenced_ecef_prior_recovers_ground_plane():
    """At Delhi (lat ~28.58°, lon ~77.22°), true geodetic UP in ECEF is tilted

    61.4° away from ECEF +Z (the Earth's North Pole). A naive [0,0,1] prior
    falsely rejects the real ground plane as tilted >15°, whereas the true
    geodetic prior recovers the ground plane with tilt < 1°.
    """
    import math

    lat, lon = 28.5834, 77.2167
    phi, lam = math.radians(lat), math.radians(lon)
    ecef_up = np.array([
        math.cos(phi) * math.cos(lam),
        math.cos(phi) * math.sin(lam),
        math.sin(phi),
    ], dtype=np.float64)
    ecef_up /= np.linalg.norm(ecef_up)

    rng = np.random.default_rng(42)
    # Generate points in a plane perpendicular to ecef_up
    # Find two orthogonal tangent vectors
    ref = np.array([1.0, 0.0, 0.0]) if abs(ecef_up[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    tangent_x = np.cross(ecef_up, ref)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(ecef_up, tangent_x)

    u_coord = rng.uniform(-30, 30, 1000)
    v_coord = rng.uniform(-30, 30, 1000)
    noise = rng.normal(0, 0.02, 1000)

    ground_pts = (
        np.outer(u_coord, tangent_x)
        + np.outer(v_coord, tangent_y)
        + np.outer(noise, ecef_up)
    )

    # With naive [0,0,1] prior: angle to ecef_up is 61.4°, exceeding 15° limit -> rejected
    plane_bad = fit_ground_plane(ground_pts, up_hint=np.array([0.0, 0.0, 1.0]), max_tilt_deg=15.0)
    assert plane_bad.source == "fit_rejected"

    # With true ECEF geodetic UP prior: plane is fitted successfully
    plane_good = fit_ground_plane(ground_pts, up_hint=ecef_up, max_tilt_deg=15.0)
    assert plane_good.source == "fitted"
    assert plane_good.tilt_deg < 1.0
    assert np.dot(plane_good.normal, ecef_up) > 0.999


def test_terrain_mtl_full_diffuse_reflectance(tmp_path: Path):
    """Terrain mesh MTL must export with Kd 1.0 so glTF is not dimmed by 60%."""
    rng = np.random.default_rng(123)
    x = rng.uniform(-10, 10, 100)
    y = rng.uniform(-10, 10, 100)
    z = rng.uniform(0, 1, 100)
    pts = np.column_stack([x, y, z])
    cols = np.full((100, 3), 200, dtype=np.uint8)

    out_obj = tmp_path / "test_terrain.obj"
    reconstruct_terrain_mesh(
        points=pts,
        colors=cols,
        output_obj=out_obj,
        grid_dim=20,
        max_grid_dim=30,
    )

    mtl_file = tmp_path / "material.mtl"
    assert mtl_file.exists()
    mtl_text = mtl_file.read_text(encoding="utf-8")
    assert "Kd 1.00000000 1.00000000 1.00000000" in mtl_text


def test_terrain_boundary_support_masking(tmp_path: Path):
    """Empty space outside the survey footprint (e.g. L-shaped flight) must not be meshed with sagging skirts."""
    rng = np.random.default_rng(99)
    # L-shaped point cloud: arm 1 along X, arm 2 along Y, upper-right quadrant is completely empty
    arm1 = np.column_stack([rng.uniform(0, 40, 500), rng.uniform(0, 10, 500), rng.normal(5, 0.05, 500)])
    arm2 = np.column_stack([rng.uniform(0, 10, 500), rng.uniform(10, 40, 500), rng.normal(5, 0.05, 500)])
    pts = np.vstack([arm1, arm2])
    cols = np.full((len(pts), 3), 150, dtype=np.uint8)

    out_obj = tmp_path / "masked_terrain.obj"
    metrics = reconstruct_terrain_mesh(
        points=pts,
        colors=cols,
        output_obj=out_obj,
        grid_dim=40,
        max_grid_dim=50,
    )

    # In a naive regular 40x40 grid, total faces would be 2 * 39 * 39 = 3042.
    # The L-shape covers ~50% of the bounding box, so the empty quadrant faces must be pruned.
    assert metrics["n_faces"] < 1800
    assert metrics["n_vertices"] < 40 * 40

    # Ensure OBJ contains vertex normals (vn lines)
    obj_content = out_obj.read_text(encoding="utf-8")
    assert "vn " in obj_content


def test_terrain_texture_high_res(tmp_path: Path):
    """High-res texture atlas must be generated and properly dimensioned."""
    rng = np.random.default_rng(101)
    x = rng.uniform(-10, 10, 100)
    y = rng.uniform(-10, 10, 100)
    z = rng.uniform(0, 1, 100)
    pts = np.column_stack([x, y, z])
    cols = np.full((100, 3), 180, dtype=np.uint8)

    out_obj = tmp_path / "highres_terrain.obj"
    reconstruct_terrain_mesh(
        points=pts,
        colors=cols,
        output_obj=out_obj,
        grid_dim=25,
        max_grid_dim=30,
        texture_size=1024,
    )

    png_files = list(tmp_path.glob("*.png"))
    assert len(png_files) >= 1
    from PIL import Image
    with Image.open(png_files[0]) as img:
        assert img.size == (1024, 1024)

