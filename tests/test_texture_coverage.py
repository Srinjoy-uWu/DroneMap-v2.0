"""Tests for the texture-atlas coverage measurement in stage 6.

An atlas can be present, correctly referenced, full resolution, and still
deliver a black model.  Every check the pipeline had -- file exists, OBJ
parses, GLB validates, UVs in range -- passed on an atlas whose patch
interiors were zeroed by OpenMVS seam levelling.  These tests pin the one
measurement that caught it: sample the atlas *at the mesh's own UV
coordinates*, not over the whole image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dronemap.stage6_mesh import _atlas_for_obj, measure_atlas_coverage


def _write_textured_obj(
    directory: Path, atlas: np.ndarray, uvs: list[tuple[float, float]]
) -> Path:
    """A minimal OBJ + MTL + atlas triple, as OpenMVS emits."""
    obj = directory / "scene_texture.obj"
    tex = directory / "scene_texture_material_00_map_Kd.png"
    Image.fromarray(atlas.astype(np.uint8)).save(tex)
    (directory / "scene_texture.mtl").write_text(
        "newmtl material_00\nKd 1 1 1\nmap_Kd " + tex.name + "\n"
    )
    lines = ["mtllib scene_texture.mtl", "usemtl material_00"]
    lines += [f"v {i} 0 0" for i in range(len(uvs))]
    lines += [f"vt {u} {v}" for u, v in uvs]
    obj.write_text("\n".join(lines) + "\n")
    return obj


def test_a_lit_atlas_measures_as_lit(tmp_path: Path):
    atlas = np.full((64, 64, 3), 120, dtype=np.uint8)
    obj = _write_textured_obj(tmp_path, atlas, [(0.25, 0.25), (0.75, 0.75)])

    result = measure_atlas_coverage(obj)
    assert result["texture_uv_black_fraction"] == 0.0
    assert result["texture_uv_samples"] == 2
    assert result["texture_uv_mean_rgb"] == [120.0, 120.0, 120.0]


def test_the_whole_atlas_mean_cannot_substitute_for_uv_sampling(tmp_path: Path):
    """The exact shape of the shipped defect.

    The real atlas was 8192x8192, mean RGB ``[234, 121, 40]`` -- a confident,
    bright, entirely healthy-looking statistic -- because OpenMVS's orange
    empty-space fill covered 89 % of it.  The 12 % the mesh actually addressed
    was black.  A whole-image mean says "fine"; UV sampling says 98 % black.
    """
    atlas = np.zeros((100, 100, 3), dtype=np.uint8)
    atlas[:, :] = (255, 127, 39)   # OpenMVS --empty-color
    atlas[:12, :] = 0              # the region the UVs address, zeroed
    # v is measured from the *bottom*, so the top 12 rows are v in [0.88, 1.0].
    uvs = [(0.1 * i, 0.95) for i in range(10)]
    obj = _write_textured_obj(tmp_path, atlas, uvs)

    assert atlas.reshape(-1, 3).mean(axis=0)[0] > 200, "fixture must look healthy in bulk"

    result = measure_atlas_coverage(obj)
    assert result["texture_uv_black_fraction"] == 1.0
    assert result["texture_uv_mean_rgb"] == [0.0, 0.0, 0.0]


def test_v_is_measured_from_the_bottom(tmp_path: Path):
    """An off-by-a-flip here would report a healthy atlas as black, or worse."""
    atlas = np.zeros((100, 100, 3), dtype=np.uint8)
    atlas[:50, :] = 200   # top half lit, bottom half black
    obj = _write_textured_obj(tmp_path, atlas, [(0.5, 0.9), (0.5, 0.1)])

    result = measure_atlas_coverage(obj)
    # One sample in each half.
    assert result["texture_uv_black_fraction"] == 0.5


def test_a_missing_atlas_is_unmeasured_not_zero(tmp_path: Path):
    """``None`` means "could not measure"; 0.0 would mean "measured, perfect"."""
    obj = tmp_path / "bare.obj"
    obj.write_text("v 0 0 0\nvt 0.5 0.5\n")

    result = measure_atlas_coverage(obj)
    assert result["texture_uv_black_fraction"] is None
    assert result["texture_uv_samples"] is None


def test_an_obj_without_uvs_is_unmeasured(tmp_path: Path):
    atlas = np.full((16, 16, 3), 80, dtype=np.uint8)
    obj = _write_textured_obj(tmp_path, atlas, [])
    assert measure_atlas_coverage(obj)["texture_uv_black_fraction"] is None


def test_atlas_is_resolved_through_the_mtl_not_guessed(tmp_path: Path):
    """A second material or a renamed map must not silently measure nothing."""
    tex = tmp_path / "some_other_name.png"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(tex)
    (tmp_path / "m.mtl").write_text("newmtl material_00\nmap_Kd some_other_name.png\n")
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nvt 0.5 0.5\n")

    assert _atlas_for_obj(obj) == tex
    assert measure_atlas_coverage(obj)["texture_uv_black_fraction"] == 0.0
