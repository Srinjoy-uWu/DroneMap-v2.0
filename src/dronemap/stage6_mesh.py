"""Stage 6 — Mesh reconstruction, optional refinement, and texturing with OpenMVS.

What this stage does
--------------------
1.  ``ReconstructMesh`` — Poisson-style surface reconstruction from the dense
    point cloud.
2.  ``RefineMesh`` (optional, ``mesh.refine: true``) — iterative mesh
    refinement to recover fine detail.  This is the slowest step, easily 30–60
    minutes on a laptop; skip it when iterating.
3.  ``TextureMesh`` — projects the source photographs onto the mesh to produce
    a photo-realistic texture atlas.

The final OBJ + texture files are what Stage 7 converts to GLB, LAZ, and
raster products.

Outputs
-------
``ws.mesh_dir/scene_dense_mesh.ply``              — raw mesh
``ws.mesh_dir/scene_dense_mesh_refine.ply``       — refined mesh (if enabled)
``ws.mesh_dir/scene_dense_mesh_refine_texture.obj``  — textured mesh (OBJ)
``ws.mesh_dir/scene_dense_mesh_refine_texture.mtl``  — material library
``ws.mesh_dir/*.jpg`` / ``*.png``                 — texture atlas tiles
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .tools import ToolRegistry
    from .workspace import RunWorkspace, _StageContext


def measure_atlas_coverage(obj_path: Path) -> dict[str, float | int | None]:
    """Measure whether the texture atlas actually carries colour where the mesh reads it.

    An atlas can be present, correctly referenced, full-resolution, and still
    deliver a black model.  That is not a hypothetical: OpenMVS's seam-levelling
    solves a Poisson system per texture patch, and when the per-view colour
    estimates disagree strongly it converges to zero over the patch *interior*
    while leaving the dilated border padding untouched.  The atlas then looks
    populated by any whole-image statistic -- on the orbit fixture the mean was
    a confident-looking ``[234, 121, 40]`` -- because the orange background fill
    and the surviving borders dominate the pixel count.  Sampling *at the mesh's
    own UV coordinates* told the true story: 98.4 % of them landed on black.

    So this measures the only thing that matters for what the viewer shows: take
    each ``vt`` the OBJ declares, read the texel it points at, and report how
    many are black.  Whole-atlas means are not used and must not be substituted
    -- they are exactly the statistic that hid this defect.

    Returns ``black_fraction`` in ``[0, 1]``, the mean RGB *at the sampled
    texels*, and the sample count.  Values are ``None`` when the atlas or its
    UVs could not be read, which is a distinct outcome from "measured, and fine".
    """
    unmeasured: dict[str, float | int | None] = {
        "texture_uv_black_fraction": None,
        "texture_uv_mean_rgb": None,
        "texture_uv_samples": None,
    }
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return unmeasured

    # An 8192x8192 atlas is 67 Mpx and trips Pillow's decompression-bomb guard.
    Image.MAX_IMAGE_PIXELS = None

    atlas = _atlas_for_obj(obj_path)
    if atlas is None or not atlas.exists():
        return unmeasured

    uvs: list[tuple[float, float]] = []
    try:
        with obj_path.open("r", errors="replace") as fh:
            for line in fh:
                if line.startswith("vt "):
                    parts = line.split()
                    uvs.append((float(parts[1]), float(parts[2])))
    except Exception:
        return unmeasured
    if not uvs:
        return unmeasured

    try:
        with Image.open(atlas) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.int16)
    except Exception:
        return unmeasured

    height, width = pixels.shape[:2]
    uv = np.asarray(uvs, dtype=np.float64)
    # OBJ/glTF put v=0 at the *bottom* of the image; array row 0 is the top.
    col = np.clip((uv[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    row = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    sampled = pixels[row, col]

    # 20/255 is dark enough that no lit surface in a daylight capture reaches it,
    # and loose enough to absorb JPEG ringing around a genuinely zeroed patch.
    luma = sampled.mean(axis=1)
    return {
        "texture_uv_black_fraction": round(float((luma < 20).mean()), 4),
        "texture_uv_mean_rgb": [round(float(v), 1) for v in sampled.mean(axis=0)],
        "texture_uv_samples": int(len(uv)),
    }


def _atlas_for_obj(obj_path: Path) -> Path | None:
    """Resolve the ``map_Kd`` image the OBJ's material library points at.

    Guessing the filename from the OBJ stem would work for OpenMVS today and
    break silently the moment a variant emits a second material, so the MTL is
    read instead.
    """
    mtl = obj_path.with_suffix(".mtl")
    if mtl.exists():
        try:
            for line in mtl.read_text(errors="replace").splitlines():
                if line.strip().lower().startswith("map_kd"):
                    name = line.split(None, 1)[1].strip()
                    candidate = obj_path.parent / name
                    if candidate.exists():
                        return candidate
        except Exception:
            pass
    for pattern in ("*_map_Kd.jpg", "*_map_Kd.png"):
        found = sorted(obj_path.parent.glob(f"{obj_path.stem}{pattern}"))
        if found:
            return found[-1]
    return None


def _count_ply_elements(ply_path: Path) -> dict[str, int]:
    """Read vertex and face counts from PLY header."""
    counts: dict[str, int] = {}
    current_element: str | None = None
    try:
        with ply_path.open("rb") as f:
            for line in f:
                line_s = line.decode("ascii", errors="replace").strip()
                if line_s.startswith("element "):
                    parts = line_s.split()
                    current_element = parts[1]
                    counts[current_element] = int(parts[2])
                elif line_s == "end_header":
                    break
    except Exception:
        pass
    return counts


def _clean_mesh_for_viewing(source: Path, output: Path, min_faces: int, max_components: int) -> dict[str, int]:
    """Remove reconstruction debris while retaining meaningful scene surfaces.

    MVS often produces isolated sheets and tiny floating fragments.  They make
    a valid mesh look like visual noise in the viewer, even when the main
    building/terrain surface is good.  Cleaning happens before texturing so
    the exported GLB contains only the useful geometry.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to clean the reconstructed mesh") from exc

    loaded = trimesh.load(str(source), force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise RuntimeError("ReconstructMesh produced an empty or invalid surface")

    original_faces = len(loaded.faces)
    threshold = max(min_faces, int(original_faces * 0.005))
    components = loaded.split(only_watertight=False)
    keep = [part for part in components if len(part.faces) >= threshold]
    keep.sort(key=lambda part: len(part.faces), reverse=True)
    keep = keep[:max_components]
    if not keep:
        raise RuntimeError(
            "The reconstruction contains only tiny disconnected fragments; "
            "there is no coherent object to display."
        )

    cleaned = trimesh.util.concatenate(keep)
    cleaned.remove_unreferenced_vertices()
    if len(cleaned.faces) < min_faces:
        raise RuntimeError("Mesh cleanup left too little connected geometry for a useful 3D view")
    cleaned.export(str(output))
    return {
        "input_faces": original_faces,
        "output_faces": len(cleaned.faces),
        "components_kept": len(keep),
        "components_removed": max(0, len(components) - len(keep)),
    }


def _run_terrain_mesh(ws: "RunWorkspace", cfg: "MeshConfig", dense_ply: Path, ctx: "_StageContext") -> None:
    import numpy as np

    from .terrain import reconstruct_terrain_mesh, up_hint_from_cameras
    from .stage7_export import _load_dense_ply
    ctx.note("Running 2.5D terrain surface mesh reconstruction...")
    points, colors = _load_dense_ply(dense_ply)
    out_obj = ws.mesh_dir / "scene_dense_mesh_terrain_texture.obj"
    out_ply = ws.mesh_dir / "scene_dense_mesh_terrain.ply"

    # Determine vertical prior based on coordinate system:
    # In georeferenced runs, OpenMVS scene_dense.ply is in shifted-ECEF (EPSG:4978).
    # In ECEF, +Z is the Earth's polar axis (North Pole), NOT local vertical.
    # True geodetic UP at (lat, lon) is: (cos phi cos lam, cos phi sin lam, sin phi).
    transform = {}
    trans_p = ws.georef_sparse_dir / "transform.json"
    if trans_p.exists():
        try:
            import json
            transform = json.loads(trans_p.read_text(encoding="utf-8"))
        except Exception:
            pass

    accuracy = ws.manifest.get("accuracy") or {}
    is_georef = (
        transform.get("georef_success") is True
        or accuracy.get("crs") not in ("LOCAL_RELATIVE", "", None)
    )
    coord_frame = transform.get("coordinate_frame") or accuracy.get("coordinate_frame")
    centroid = transform.get("scene_centroid_wgs84") or {}
    lat = centroid.get("lat", accuracy.get("lat0"))
    lon = centroid.get("lon", accuracy.get("lon0"))

    if is_georef and coord_frame == "ECEF" and lat is not None and lon is not None:
        import math
        phi = math.radians(float(lat))
        lam = math.radians(float(lon))
        up_hint = np.array([
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        ], dtype=np.float64)
        up_hint /= max(float(np.linalg.norm(up_hint)), 1e-12)
        ctx.note(
            f"ground-plane prior: ECEF geodetic vertical at ({float(lat):.4f}°, {float(lon):.4f}°) "
            f"[{up_hint[0]:.3f}, {up_hint[1]:.3f}, {up_hint[2]:.3f}]"
        )
    elif is_georef and coord_frame in ("LOCAL_ENU", "ENU", "UTM"):
        up_hint = np.array([0.0, 0.0, 1.0])
        ctx.note("ground-plane prior: ENU/UTM local vertical (+Z), from georeferencing")
    else:
        # No telemetry means COLMAP's +Z is arbitrary, so recover the vertical
        # from where the cameras were pointing instead of assuming it.
        images_txt = None
        for cand in [
            ws.sparse_dir / "0" / "txt" / "images.txt",
            ws.sparse_dir / "txt" / "images.txt",
            ws.sparse_dir / "0" / "images.txt",
            ws.sparse_dir / "images.txt",
        ]:
            if cand.exists():
                images_txt = cand
                break
        if images_txt is None:
            txt_matches = list(ws.sparse_dir.rglob("images.txt"))
            images_txt = txt_matches[0] if txt_matches else (ws.sparse_dir / "0" / "images.txt")

        up_hint = up_hint_from_cameras(images_txt)
        if up_hint is not None:
            ctx.note(
                "ground-plane prior: mean camera optical axis "
                f"({np.round(up_hint, 3).tolist()}) — the cloud is not "
                "georeferenced, so this is an estimate of vertical, not a datum"
            )
        else:
            ctx.note(
                "no vertical prior available (not georeferenced, and the camera "
                "poses give no consistent down) — the 2.5D product's alignment "
                "is only as good as the plane fit"
            )

    metrics = reconstruct_terrain_mesh(
        points=points,
        colors=colors,
        output_obj=out_obj,
        output_ply=out_ply,
        grid_dim=cfg.terrain_grid_dim,
        max_grid_dim=cfg.terrain_grid_max,
        up_hint=up_hint,
        max_tilt_deg=cfg.terrain_max_tilt_deg if is_georef else max(cfg.terrain_max_tilt_deg, 60.0),
        texture_size=cfg.texture_size,
    )
    if metrics["ground_plane_source"] == "fit_rejected":
        ctx.note(
            f"ground-plane fit was rejected (tilt exceeded "
            f"{cfg.terrain_max_tilt_deg}°); the vertical prior was used instead"
        )
    else:
        ctx.note(
            f"ground plane {metrics['ground_plane_source']}: tilt "
            f"{metrics['ground_tilt_deg']}°, RMS residual "
            f"{metrics['ground_rms_residual_m']} m, relief {metrics['relief_m']} m"
        )

    ctx.metric(
        mesh_type="terrain_2.5d",
        n_vertices=metrics["n_vertices"],
        n_faces=metrics["n_faces"],
        texture_size=cfg.texture_size,
        refined=False,
        input_faces=metrics["n_faces"],
        output_faces=metrics["n_faces"],
        components_kept=1,
        components_removed=0,
        grid_nx=metrics["grid_nx"],
        grid_ny=metrics["grid_ny"],
        ground_tilt_deg=metrics["ground_tilt_deg"],
        ground_plane_source=metrics["ground_plane_source"],
        ground_rms_residual_m=metrics["ground_rms_residual_m"],
        ground_inlier_fraction=metrics["ground_inlier_fraction"],
        ground_normal=metrics.get("ground_normal"),
        relief_m=metrics["relief_m"],
        cloud_z_span_m=metrics["cloud_z_span_m"],
    )
    ctx.output(
        textured_obj=str(out_obj),
        mesh_ply=str(out_ply),
        ground_normal=metrics.get("ground_normal"),
    )


def run(ws: "RunWorkspace", config: "Config", tools: "ToolRegistry", ctx: "_StageContext") -> None:
    openmvs = tools.openmvs
    if openmvs is None:
        raise RuntimeError("OpenMVS is required for stage 'mesh'. Run `python scripts/bootstrap.py`.")

    cfg = config.mesh
    ws.mesh_dir.mkdir(parents=True, exist_ok=True)

    dense_ply = ws.dense_dir / "scene_dense.ply"
    if not dense_ply.exists():
        raise RuntimeError(f"Dense point cloud not found at {dense_ply}. Did stage 'dense' complete successfully?")

    # Explicit 2.5D terrain reconstruction
    if cfg.mode == "terrain_2.5d":
        _run_terrain_mesh(ws, cfg, dense_ply, ctx)
        return

    # Automatic routing from capture quality verdict
    pose_verdict = ws.stage("pose").metrics.get("capture_verdict")
    if cfg.mode == "auto" and config.quality.route_terrain_to_2_5d and pose_verdict == "terrain_2_5d":
        ctx.note("routing to 2.5D terrain mesh based on capture quality verdict (terrain_2_5d)")
        _run_terrain_mesh(ws, cfg, dense_ply, ctx)
        return

    # Locate the dense MVS project
    dense_mvs = ws.dense_dir / "scene_dense.mvs"
    if not dense_mvs.exists():
        if cfg.mode == "auto":
            ctx.note(f"scene_dense.mvs not found; falling back to 2.5D terrain reconstruction from {dense_ply.name}")
            _run_terrain_mesh(ws, cfg, dense_ply, ctx)
            return
        raise RuntimeError(
            f"scene_dense.mvs not found at {dense_mvs}. "
            "Did stage 'dense' complete successfully?"
        )

    # ------------------------------------------------------------------
    # Step 1 & 2: ReconstructMesh & RefineMesh (with step-level resume)
    # ------------------------------------------------------------------
    raw_mesh_ply = ws.mesh_dir / "scene_dense_mesh.ply"
    raw_mesh_mvs = ws.mesh_dir / "scene_dense_mesh.mvs"
    clean_raw_ply = ws.mesh_dir / "scene_dense_mesh_clean.ply"
    refined_ply = ws.mesh_dir / "scene_dense_mesh_refine.ply"
    refined_mvs = ws.mesh_dir / "scene_dense_mesh_refine.mvs"

    if clean_raw_ply.exists() and (refined_ply.exists() if cfg.refine else True):
        ctx.note("using existing reconstructed geometry on disk; skipping ReconstructMesh/RefineMesh")
        cleanup = {
            "input_faces": _count_ply_elements(clean_raw_ply).get("face", -1),
            "output_faces": _count_ply_elements(clean_raw_ply).get("face", -1),
            "components_kept": 1,
            "components_removed": 0,
        }
        best_ply = refined_ply if (cfg.refine and refined_ply.exists()) else clean_raw_ply
        best_mvs = refined_mvs if (cfg.refine and refined_ply.exists()) else raw_mesh_mvs
    else:
        ctx.note("running ReconstructMesh")
        n_dense_pts = _count_ply_elements(dense_ply).get("vertex", 0)
        if n_dense_pts < 50:
            raise RuntimeError(
                f"Dense point cloud has only {n_dense_pts} points, which is insufficient for 3D surface reconstruction. "
                "Reconstruction requires a video with camera parallax, texture, and visual overlap between consecutive frames."
            )

        try:
            openmvs.run(
                "ReconstructMesh",
                [
                    str(dense_mvs),
                    "--working-folder", str(ws.mesh_dir),
                    "-p", str(dense_ply),
                    "-o", str(raw_mesh_mvs),
                    "--min-point-distance", str(cfg.min_point_distance),
                    "--decimate", str(cfg.decimate),
                ],
                cwd=ws.mesh_dir,
                log_path=ws.log_path("openmvs_reconstruct"),
            )
        except Exception as exc:
            ctx.note(f"ReconstructMesh with -p failed ({exc}), retrying without point cloud override...")
            try:
                openmvs.run(
                    "ReconstructMesh",
                    [
                        str(dense_mvs),
                        "--working-folder", str(ws.mesh_dir),
                        "-o", str(raw_mesh_mvs),
                        "--decimate", str(cfg.decimate),
                    ],
                    cwd=ws.mesh_dir,
                    log_path=ws.log_path("openmvs_reconstruct"),
                )
            except Exception as inner_exc:
                if cfg.mode == "auto":
                    ctx.note(f"ReconstructMesh failed ({inner_exc}); recovering via 2.5D terrain reconstruction")
                    _run_terrain_mesh(ws, cfg, dense_ply, ctx)
                    return
                raise RuntimeError(
                    "3D mesh reconstruction could not be formed from the point cloud. "
                    "This typically happens when the scene lacks sufficient 3D depth variation or features."
                ) from exc

        if not raw_mesh_ply.exists():
            raise RuntimeError(
                "ReconstructMesh did not produce scene_dense_mesh.ply. "
                "Check log: " + str(ws.log_path("openmvs_reconstruct"))
            )

        try:
            cleanup = _clean_mesh_for_viewing(
                raw_mesh_ply, clean_raw_ply, cfg.min_component_faces, cfg.max_components
            )
        except Exception as exc:
            if cfg.mode == "auto":
                ctx.note(f"OpenMVS mesh cleanup failed ({exc}); recovering via 2.5D terrain reconstruction")
                _run_terrain_mesh(ws, cfg, dense_ply, ctx)
                return
            raise

        ctx.note(
            f"removed {cleanup['components_removed']} floating mesh fragment(s); "
            f"kept {cleanup['components_kept']} coherent component(s)"
        )
        best_ply = clean_raw_ply
        best_mvs = raw_mesh_mvs

        if cfg.refine:
            ctx.note("running RefineMesh (this is the slowest step — ~30-60 min on laptop)")
            try:
                openmvs.run(
                    "RefineMesh",
                    [
                        str(dense_mvs),
                        "-m", str(clean_raw_ply),
                        "-o", str(refined_mvs),
                        "--max-views", str(cfg.refine_max_views),
                    ],
                    cwd=ws.mesh_dir,
                    log_path=ws.log_path("openmvs_refine"),
                )
                if refined_ply.exists():
                    best_ply = refined_ply
                    best_mvs = refined_mvs
                else:
                    ctx.note("RefineMesh did not produce output — using unrefined mesh")
            except Exception as exc:
                ctx.note(f"RefineMesh failed ({exc}) — using unrefined mesh")
        else:
            ctx.note("RefineMesh skipped (mesh.refine: false)")

    # ------------------------------------------------------------------
    # Step 3: TextureMesh
    # ------------------------------------------------------------------
    ctx.note("running TextureMesh")
    texture_stem = best_ply.stem + "_texture"
    texture_mvs  = ws.mesh_dir / f"{texture_stem}.mvs"
    textured_obj = ws.mesh_dir / f"{texture_stem}.obj"

    def _texture(target: Path, out_mvs: Path, *, seam_leveling: bool) -> None:
        """Invoke TextureMesh, optionally with both seam-levelling passes off.

        ``--global-seam-leveling`` and ``--local-seam-leveling`` default to on
        and are the reason a textured mesh can come out black; see
        ``measure_atlas_coverage``.  They are only disabled on the retry, so a
        scene they handle correctly keeps the better-blended result.
        """
        args = [
            str(dense_mvs),
            "--working-folder", str(ws.dense_dir),
            "-m", str(target),
            "-o", str(out_mvs),
            "--export-type", "obj",
            "--texture-size", str(cfg.texture_size),
        ]
        if not seam_leveling:
            args += ["--global-seam-leveling", "0", "--local-seam-leveling", "0"]
        openmvs.run(
            "TextureMesh", args,
            cwd=ws.dense_dir,
            log_path=ws.log_path("openmvs_texture"),
        )

    try:
        _texture(best_ply, texture_mvs, seam_leveling=cfg.seam_leveling)
    except Exception as exc:
        fallback_target = None
        if raw_mesh_ply.exists() and raw_mesh_ply != best_ply:
            fallback_target = raw_mesh_ply
        elif clean_raw_ply.exists() and clean_raw_ply != best_ply:
            fallback_target = clean_raw_ply

        if fallback_target is not None:
            ctx.note(f"TextureMesh failed on {best_ply.name}; falling back to {fallback_target.name}")
            texture_stem = fallback_target.stem + "_texture"
            texture_mvs  = ws.mesh_dir / f"{texture_stem}.mvs"
            textured_obj = ws.mesh_dir / f"{texture_stem}.obj"
            try:
                _texture(fallback_target, texture_mvs, seam_leveling=cfg.seam_leveling)
                best_ply = fallback_target
            except Exception as inner_tex_exc:
                if cfg.mode == "auto":
                    ctx.note(f"TextureMesh failed ({inner_tex_exc}); recovering via 2.5D terrain reconstruction")
                    _run_terrain_mesh(ws, cfg, dense_ply, ctx)
                    return
                raise
        else:
            if cfg.mode == "auto":
                ctx.note(f"TextureMesh failed ({exc}); recovering via 2.5D terrain reconstruction")
                _run_terrain_mesh(ws, cfg, dense_ply, ctx)
                return
            raise exc

    if not textured_obj.exists():
        if cfg.mode == "auto":
            ctx.note("TextureMesh did not produce OBJ; recovering via 2.5D terrain reconstruction")
            _run_terrain_mesh(ws, cfg, dense_ply, ctx)
            return
        raise RuntimeError(
            f"TextureMesh did not produce {texture_stem}.obj. "
            "Check log: " + str(ws.log_path("openmvs_texture"))
        )

    # ------------------------------------------------------------------
    # Step 3b: verify the atlas carries colour, and re-texture if it does not
    # ------------------------------------------------------------------
    # TextureMesh exits 0 and writes a well-formed OBJ + MTL + atlas whether or
    # not the atlas contains anything.  Nothing downstream can tell the
    # difference -- the GLB validates, the material references a real texture,
    # the UVs are in range -- so the failure surfaced only as "the model is black
    # with coloured lines through it" in the viewer, which is what the border
    # padding of an interior-zeroed atlas looks like.  Measure it here, where the
    # inputs to a retry are still in hand.
    coverage = measure_atlas_coverage(textured_obj)
    black = coverage["texture_uv_black_fraction"]
    retexture: dict[str, object] = {"texture_seam_leveling": cfg.seam_leveling}

    # Both texture variants are kept when a retry happens, not just the winner.
    # The seam-levelled atlas is what produced "a black mesh with random
    # coloured lines through it", and the only way to show that those lines are
    # an artefact -- rather than something in the scene -- is to put the two
    # side by side.  ``alt_obj`` is the variant that was *not* adopted.
    alt_obj: Path | None = None
    alt_label: str | None = None

    if black is not None and black > cfg.max_texture_black_fraction and cfg.seam_leveling:
        ctx.note(
            f"texture atlas is black at {black:.1%} of the mesh's UV samples "
            f"(limit {cfg.max_texture_black_fraction:.0%}) — seam levelling "
            "collapsed the patch interiors; re-texturing with both levelling "
            "passes disabled"
        )
        retry_stem = texture_stem + "_nolevel"
        retry_mvs = ws.mesh_dir / f"{retry_stem}.mvs"
        retry_obj = ws.mesh_dir / f"{retry_stem}.obj"
        try:
            _texture(best_ply, retry_mvs, seam_leveling=False)
        except Exception as exc:
            ctx.note(f"unlevelled re-texture failed ({exc}); keeping the levelled atlas")
        else:
            retry_coverage = measure_atlas_coverage(retry_obj)
            retry_black = retry_coverage["texture_uv_black_fraction"]
            # Only adopt the retry if it is measurably better. A retry that is
            # just as black is not an improvement worth shipping, and silently
            # swapping it would hide that seam levelling was not the cause.
            if retry_obj.exists() and retry_black is not None and retry_black < black:
                ctx.note(
                    f"unlevelled atlas is black at {retry_black:.1%} of UV samples "
                    f"(was {black:.1%}); adopting it"
                )
                alt_obj = textured_obj
                alt_label = "seam_levelled"
                textured_obj = retry_obj
                coverage = retry_coverage
                retexture["texture_seam_leveling"] = False
                retexture["texture_retextured_without_seam_leveling"] = True
                retexture["texture_uv_black_fraction_levelled"] = black
            else:
                ctx.note(
                    "unlevelled atlas is no better; seam levelling was not the "
                    "cause and the black texture is a genuine texturing failure"
                )
                retexture["texture_retextured_without_seam_leveling"] = False
                if retry_obj.exists():
                    alt_obj = retry_obj
                    alt_label = "no_seam_levelling"
                    retexture["texture_uv_black_fraction_unlevelled"] = retry_black

    if black is not None and coverage["texture_uv_black_fraction"] is not None:
        final_black = coverage["texture_uv_black_fraction"]
        if final_black > cfg.max_texture_black_fraction:
            # Still recorded as a metric rather than raised: a geometrically
            # sound but poorly textured mesh is a partial result, and the report
            # is the right place to say so. The white-clay view remains usable.
            ctx.note(
                f"WARNING: exported texture is black at {final_black:.1%} of UV "
                "samples — the mesh will render dark in the viewer"
            )

    # Count mesh elements
    counts = _count_ply_elements(best_ply)
    if alt_obj is not None:
        retexture["texture_alt_variant"] = alt_label
        ctx.note(
            f"keeping the '{alt_label}' texture variant alongside the adopted one "
            "so the two can be compared in the viewer"
        )
    ctx.metric(
        mesh_type="openmvs_3d",
        n_vertices=counts.get("vertex", -1),
        n_faces=counts.get("face", -1),
        texture_size=cfg.texture_size,
        refined=cfg.refine,
        **coverage,
        **retexture,
        **cleanup,
    )
    outputs: dict[str, str] = {
        "textured_obj": str(textured_obj),
        "mesh_ply": str(best_ply),
    }
    if alt_obj is not None:
        outputs["textured_obj_alt"] = str(alt_obj)
        outputs["textured_obj_alt_label"] = alt_label or "alternate"
    ctx.output(**outputs)
