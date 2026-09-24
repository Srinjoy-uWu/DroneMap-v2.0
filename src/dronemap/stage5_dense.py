"""Stage 5 — Dense point cloud with OpenMVS.

What this stage does
--------------------
1.  Converts the COLMAP undistorted output to OpenMVS scene format using
    ``InterfaceCOLMAP``.
2.  Runs ``DensifyPointCloud`` to compute per-view depth maps and fuse them
    into a dense, colorised point cloud (.ply).

The dense cloud is the input to Stage 6 (mesh extraction) and is also exported
directly as a LAZ file in Stage 7 for GIS use.

Why run inside ``ws.dense_dir`` rather than passing absolute paths
------------------------------------------------------------------
OpenMVS writes several auxiliary files (undistorted images cache, depth maps)
relative to the working directory.  Running in-place keeps these next to the
output so the Stage 6 mesh step can find them.

Outputs
-------
``ws.dense_dir/scene.mvs``         — OpenMVS project file
``ws.dense_dir/scene_dense.ply``   — dense coloured point cloud
``ws.dense_dir/scene_dense.mvs``   — OpenMVS project with embedded depth maps
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .tools import ToolRegistry
    from .workspace import RunWorkspace, _StageContext


def _count_ply_vertices(ply_path: Path) -> int | None:
    """Quick scan of PLY header for vertex count."""
    try:
        with ply_path.open("rb") as f:
            for line in f:
                line_s = line.decode("ascii", errors="replace").strip()
                if line_s.startswith("element vertex"):
                    return int(line_s.split()[-1])
                if line_s == "end_header":
                    break
    except Exception:
        pass
    return None


def _colmap_sparse_to_ply(ws: "RunWorkspace", out_ply: Path) -> Path:
    """Convert COLMAP points3D.txt → PLY as a dense stage fallback.

    Returns the output path (same as out_ply) whether or not conversion succeeded.
    """
    import struct

    # Find the largest sparse model
    txt_dirs = sorted(ws.sparse_dir.glob("*/txt"), reverse=True)
    if not txt_dirs:
        return out_ply  # nothing to convert

    pts_txt = txt_dirs[0] / "points3D.txt"
    if not pts_txt.exists():
        return out_ply

    points: list[tuple[float, float, float, int, int, int]] = []
    for line in pts_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append((x, y, z, r, g, b))
        except (ValueError, IndexError):
            continue

    if not points:
        return out_ply

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with out_ply.open("wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for x, y, z, r, g, b in points:
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))

    return out_ply




def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    openmvs = tools.openmvs
    if openmvs is None:
        raise RuntimeError("OpenMVS is required for stage 'dense'. Run `python scripts/bootstrap.py`.")

    undistorted_dir = ws.undistorted_dir
    if not undistorted_dir.exists() or not any(undistorted_dir.iterdir()):
        raise RuntimeError(
            f"Undistorted image directory is empty: {undistorted_dir}. "
            "Did stage 'pose' complete successfully?"
        )

    ws.dense_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.dense

    # ------------------------------------------------------------------
    # Step 1: InterfaceCOLMAP — convert COLMAP output to .mvs scene
    # ------------------------------------------------------------------
    scene_mvs = ws.dense_dir / "scene.mvs"
    ctx.note("running InterfaceCOLMAP")
    openmvs.run(
        "InterfaceCOLMAP",
        [
            "--working-folder", str(ws.dense_dir),
            "-i", str(undistorted_dir),
            "-o", str(scene_mvs),
        ],
        cwd=ws.dense_dir,
        log_path=ws.log_path("openmvs_interface"),
    )

    if not scene_mvs.exists():
        raise RuntimeError(
            "InterfaceCOLMAP did not produce scene.mvs. "
            "Check the log for errors: " + str(ws.log_path("openmvs_interface"))
        )

    # ------------------------------------------------------------------
    # Step 2: DensifyPointCloud
    # ------------------------------------------------------------------
    ctx.note("running DensifyPointCloud")
    # number-views-fuse: how many views must see a depth point to include it.
    # The config default is 3; for partially-registered (<50%) sequences lower
    # to 2 (minimum stereo) so we still get some density.
    n_registered = sum(1 for _ in (ws.undistorted_dir / "images").glob("*.jpg") if True)
    n_views_fuse = cfg.number_views_fuse
    if n_registered < 40:
        n_views_fuse = min(n_views_fuse, 2)
        ctx.note(f"low frame count ({n_registered}) — lowering number-views-fuse to {n_views_fuse}")

    openmvs.run(
        "DensifyPointCloud",
        [
            str(scene_mvs),
            "--resolution-level", str(cfg.resolution_level),
            "--max-resolution", str(cfg.max_resolution),
            "--number-views", str(cfg.number_views),
            "--number-views-fuse", str(n_views_fuse),
            "--estimate-colors", str(cfg.estimate_colors),
            "--estimate-normals", str(cfg.estimate_normals),
        ],
        cwd=ws.dense_dir,
        log_path=ws.log_path("openmvs_densify"),
    )

    # OpenMVS produces scene_dense.ply and scene_dense.mvs in the working dir
    dense_ply = ws.dense_dir / "scene_dense.ply"
    dense_mvs = ws.dense_dir / "scene_dense.mvs"

    if not dense_ply.exists():
        # Fallback: convert COLMAP sparse cloud (points3D.txt) to PLY.
        # This gives a usable (if sparse) point cloud for mesh and export stages
        # when OpenMVS paired 0 views (very low overlap video or partial registration).
        ctx.note(
            "DensifyPointCloud produced 0 points — falling back to COLMAP sparse cloud. "
            "The final model will be sparse. Use a higher-overlap video for dense results."
        )
        sparse_ply = _colmap_sparse_to_ply(ws, dense_ply)
        if not sparse_ply.exists():
            raise RuntimeError(
                "DensifyPointCloud produced no output and the COLMAP sparse cloud fallback "
                "also failed. Check the log: " + str(ws.log_path("openmvs_densify"))
            )

    n_points = _count_ply_vertices(dense_ply)
    if (n_points or 0) < cfg.min_dense_points:
        raise RuntimeError(
            f"Dense reconstruction produced only {n_points or 0} points (need at least "
            f"{cfg.min_dense_points} for a recognisable 3D scene). "
            "Do not export this run; capture slower footage with 70–80% overlap, "
            "visible ground/building texture, and less motion blur."
        )
    ctx.metric(n_dense_points=n_points or 0)
    ctx.output(dense_ply=str(dense_ply), dense_mvs=str(dense_mvs) if dense_mvs.exists() else "")
