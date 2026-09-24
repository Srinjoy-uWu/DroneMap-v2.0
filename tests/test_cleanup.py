"""Unit tests for dronemap.cleanup module."""

from pathlib import Path
from dronemap.cleanup import (
    clean_intermediate_depth_maps,
    clean_root_logs,
    clean_runs,
    clean_scratch,
    clean_unused_datasets,
)


def test_clean_root_logs(tmp_path: Path):
    log1 = tmp_path / "DensifyPointCloud-260901.log"
    log2 = tmp_path / "TextureMesh-260901.log"
    keep_file = tmp_path / "README.md"
    log1.write_text("log data")
    log2.write_text("log data")
    keep_file.write_text("keep me")

    deleted = clean_root_logs(tmp_path, dry_run=False)
    assert len(deleted) == 2
    assert not log1.exists()
    assert not log2.exists()
    assert keep_file.exists()


def test_clean_intermediate_depth_maps(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    dense_dir = run_dir / "05_dense"
    dense_dir.mkdir(parents=True)
    dense_ply = dense_dir / "scene_dense.ply"
    dense_ply.write_text("dense ply content")

    dmap1 = dense_dir / "depth0001.dmap"
    dmap2 = dense_dir / "depth0002.dmap.estimate"
    dmap1.write_bytes(b"depth content 1")
    dmap2.write_bytes(b"depth content 2")

    res = clean_intermediate_depth_maps(run_dir, dry_run=False)
    assert res["deleted_count"] == 2
    assert res["freed_bytes"] > 0
    assert dense_ply.exists()
    assert not dmap1.exists()
    assert not dmap2.exists()


def test_clean_intermediate_depth_maps_safeguard(tmp_path: Path):
    # Should NOT delete dmaps if scene_dense.ply doesn't exist
    run_dir = tmp_path / "run_test_incomplete"
    dense_dir = run_dir / "05_dense"
    dense_dir.mkdir(parents=True)
    dmap1 = dense_dir / "depth0001.dmap"
    dmap1.write_bytes(b"depth content")

    res = clean_intermediate_depth_maps(run_dir, dry_run=False)
    assert res["deleted_count"] == 0
    assert dmap1.exists()


def test_clean_scratch(tmp_path: Path):
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    tmp_inspect = repo_root / "tmp_inspect"
    tmp_inspect.mkdir(parents=True)
    (tmp_inspect / "test.png").write_text("scratch")

    data_tmp = data_root / "tmp"
    data_tmp.mkdir(parents=True)
    (data_tmp / "old.db").write_text("db")

    tools_downloads = repo_root / "tools" / "downloads"
    tools_downloads.mkdir(parents=True)
    (tools_downloads / "colmap.zip").write_text("zip")

    res = clean_scratch(repo_root, data_root, dry_run=False)
    assert res["total_freed_bytes"] > 0
    assert not tmp_inspect.exists()
    assert not data_tmp.exists()
    assert not (tools_downloads / "colmap.zip").exists()


def test_clean_unused_datasets(tmp_path: Path):
    data_root = tmp_path / "data"
    ds_keep = data_root / "synthetic_flight"
    ds_del = data_root / "synthetic_flight_closeup"
    ds_keep.mkdir(parents=True)
    ds_del.mkdir(parents=True)
    (ds_keep / "flight.mp4").write_text("video")
    (ds_del / "raw_frames.png").write_text("frames")

    res = clean_unused_datasets(data_root, keep_datasets=["synthetic_flight"], dry_run=False)
    assert len(res["removed"]) == 1
    assert res["removed"][0]["name"] == "synthetic_flight_closeup"
    assert ds_keep.exists()
    assert not ds_del.exists()


def test_clean_runs(tmp_path: Path):
    data_root = tmp_path / "data"
    runs_dir = data_root / "runs"
    keep_run = runs_dir / "fixture_orbit_hires"
    del_run = runs_dir / "old_run_01"
    keep_run.mkdir(parents=True)
    del_run.mkdir(parents=True)
    (keep_run / "manifest.json").write_text("{}")
    (del_run / "manifest.json").write_text("{}")

    res = clean_runs(data_root, keep_run_ids=["fixture_orbit_hires"], prune_depth_maps=False, dry_run=False)
    assert len(res["deleted_runs"]) == 1
    assert res["deleted_runs"][0]["run_id"] == "old_run_01"
    assert keep_run.exists()
    assert not del_run.exists()
