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
