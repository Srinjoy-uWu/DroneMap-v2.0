"""Tests for filtering floating reconstruction fragments before viewer export."""

from __future__ import annotations

import trimesh

from dronemap.stage6_mesh import _clean_mesh_for_viewing


def test_mesh_cleanup_keeps_main_surface_and_drops_tiny_fragment(tmp_path):
    main = trimesh.creation.icosphere(subdivisions=2, radius=5)
    fragment = trimesh.creation.icosphere(subdivisions=0, radius=0.1)
    fragment.apply_translation((100, 100, 100))
    source = tmp_path / "raw.ply"
    output = tmp_path / "clean.ply"
    trimesh.util.concatenate([main, fragment]).export(source)

    metrics = _clean_mesh_for_viewing(source, output, min_faces=40, max_components=3)

    cleaned = trimesh.load(output, force="mesh")
    assert output.exists()
    assert metrics["components_removed"] == 1
    assert len(cleaned.faces) == len(main.faces)


def test_fill_interior_boundary_holes_synthesizes_patch():
    from dronemap.mesh_completion import fill_interior_boundary_holes, repair_and_complete_openmvs_mesh
    import numpy as np

    # Create a 20x20 grid mesh
    n = 20
    x = np.linspace(-10, 10, n)
    y = np.linspace(-10, 10, n)
    gx, gy = np.meshgrid(x, y)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])

    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            # Cut a hole in the center: [-3, 3] x [-3, 3]
            cx, cy = gx[j, i], gy[j, i]
            if -3.5 <= cx <= 3.5 and -3.5 <= cy <= 3.5:
                continue
            v0 = j * n + i
            v1 = j * n + (i + 1)
            v2 = (j + 1) * n + (i + 1)
            v3 = (j + 1) * n + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    mesh_with_hole = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    initial_faces = len(mesh_with_hole.faces)

    res = repair_and_complete_openmvs_mesh(mesh_with_hole)
    assert res["holes_filled_faces"] > 0
    assert len(mesh_with_hole.faces) > initial_faces
