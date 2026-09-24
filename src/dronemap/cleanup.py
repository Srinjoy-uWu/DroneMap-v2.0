"""Automated disk cleanup and maintenance utility for DroneMap.

Provides safe pruning routines for:
1. Root OpenMVS/COLMAP log clutter.
2. Scratch directories (`tmp_inspect/`, `data/tmp/`, `data/uploads/`, `tools/downloads/*.zip`).
3. Intermediate OpenMVS depth map caches (`.dmap`) once dense cloud or mesh is built.
4. Abandoned/previous runs that lack final models.
5. Unused datasets, keeping only verified fixtures.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Sequence


def clean_root_logs(repo_root: Path, dry_run: bool = False) -> list[Path]:
    """Find and delete loose tool log files from the repo root."""
    patterns = [
        "DensifyPointCloud-*.log",
        "TextureMesh-*.log",
        "ReconstructMesh-*.log",
        "RefineMesh-*.log",
        "InterfaceCOLMAP-*.log",
        "*-260*.log",
        "server_output.log",
    ]
    deleted: list[Path] = []
    for pattern in patterns:
        for p in repo_root.glob(pattern):
            if p.is_file():
                deleted.append(p)
                if not dry_run:
                    try:
                        p.unlink()
                    except OSError:
                        pass
    return deleted


def clean_intermediate_depth_maps(
    run_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prune .dmap and .dmap.estimate files from 05_dense if cloud or mesh exists.

    OpenMVS DensifyPointCloud writes a depth map per image before fusing them
    into `scene_dense.ply`. Once fused, the depth maps are purely cache.
    """
    dense_dir = run_dir / "05_dense"
    dense_ply = dense_dir / "scene_dense.ply"
    model_glb = run_dir / "07_export" / "model.glb"

    if not dense_dir.exists() or (not dense_ply.exists() and not model_glb.exists()):
        return {"deleted_count": 0, "freed_bytes": 0, "pruned": False}

    count = 0
    freed = 0
    for ext in ("*.dmap", "*.dmap.estimate"):
        for f in dense_dir.glob(ext):
            if f.is_file():
                count += 1
                freed += f.stat().st_size
                if not dry_run:
                    try:
                        f.unlink()
                    except OSError:
                        pass

    return {
        "run_id": run_dir.name,
        "deleted_count": count,
        "freed_bytes": freed,
        "pruned": True,
    }


def clean_scratch(
    repo_root: Path,
    data_root: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove temporary scratch folders and downloaded installer zips."""
    freed = 0
    cleaned_items: dict[str, int] = {}

    # 1. tmp_inspect
    tmp_inspect = repo_root / "tmp_inspect"
    if tmp_inspect.exists():
        size = sum(f.stat().st_size for f in tmp_inspect.rglob("*") if f.is_file())
        cleaned_items["tmp_inspect"] = size
        freed += size
        if not dry_run:
            shutil.rmtree(tmp_inspect, ignore_errors=True)

    # 2. data/tmp
    data_tmp = data_root / "tmp"
    if data_tmp.exists():
        size = sum(f.stat().st_size for f in data_tmp.rglob("*") if f.is_file())
        cleaned_items["data_tmp"] = size
        freed += size
        if not dry_run:
            shutil.rmtree(data_tmp, ignore_errors=True)

    # 3. data/uploads
    data_uploads = data_root / "uploads"
    if data_uploads.exists():
        size = sum(f.stat().st_size for f in data_uploads.rglob("*") if f.is_file())
        cleaned_items["data_uploads"] = size
        freed += size
        if not dry_run:
            shutil.rmtree(data_uploads, ignore_errors=True)
            data_uploads.mkdir(parents=True, exist_ok=True)

    # 4. tools/downloads/*.zip
    tools_downloads = repo_root / "tools" / "downloads"
    if tools_downloads.exists():
        zip_size = 0
        for z in tools_downloads.glob("*.zip"):
            if z.is_file():
                zip_size += z.stat().st_size
                if not dry_run:
                    try:
                        z.unlink()
                    except OSError:
                        pass
        if zip_size:
            cleaned_items["tools_downloads_zips"] = zip_size
            freed += zip_size

    cleaned_items["total_freed_bytes"] = freed
    return cleaned_items


def clean_unused_datasets(
    data_root: Path,
    keep_datasets: Sequence[str] = ("synthetic_flight",),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove dataset folders that are not in `keep_datasets`."""
    keep_set = set(keep_datasets)
    removed: list[dict[str, Any]] = []
    total_freed = 0

    known_dataset_dirs = [
        "synthetic_flight_closeup",
        "synthetic_flight_corridor",
        "aukerman_imgs",
        "sample_datasets",
    ]

    for name in known_dataset_dirs:
        if name in keep_set:
            continue
        p = data_root / name
        if p.exists() and p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            removed.append({"name": name, "size_bytes": size})
            total_freed += size
            if not dry_run:
                shutil.rmtree(p, ignore_errors=True)

    return {"removed": removed, "total_freed_bytes": total_freed}


def clean_runs(
    data_root: Path,
    keep_run_ids: Sequence[str] = ("fixture_orbit_hires",),
    prune_depth_maps: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete all runs except `keep_run_ids`, and optionally prune .dmaps in kept runs."""
    runs_dir = data_root / "runs"
    if not runs_dir.exists():
        return {"deleted_runs": [], "total_freed_bytes": 0, "pruned_kept": []}

    keep_set = set(keep_run_ids)
    deleted_runs: list[dict[str, Any]] = []
    pruned_kept: list[dict[str, Any]] = []
    total_freed = 0

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue

        if run_path.name in keep_set:
            if prune_depth_maps:
                res = clean_intermediate_depth_maps(run_path, dry_run=dry_run)
                if res["deleted_count"] > 0:
                    pruned_kept.append(res)
                    total_freed += res["freed_bytes"]
            continue

        # This run is to be removed
        size = sum(f.stat().st_size for f in run_path.rglob("*") if f.is_file())
        deleted_runs.append({"run_id": run_path.name, "size_bytes": size})
        total_freed += size
        if not dry_run:
            shutil.rmtree(run_path, ignore_errors=True)

    return {
        "deleted_runs": deleted_runs,
        "pruned_kept": pruned_kept,
        "total_freed_bytes": total_freed,
    }
