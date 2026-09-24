"""Generate a photogrammetry-grade synthetic drone fixture with ground truth (WP9).

Produces, per fixture:

    flight.mp4              encoded single-pass flight video
    frames/                 the rendered frames
    flight_telemetry.csv    GNSS-noised telemetry (DJI-style columns)
    gt_poses.json           exact camera centres and orientations
    gt_trajectory.txt       the same trajectory in TUM format
    gt_mesh.ply / .obj      the exact scene surface
    gt_dimensions.json      known object dimensions, centres and reference distances
    fixture.json            capture geometry, intrinsics and the expected verdict

Two capture geometries, both a single continuous pass:

    ``orbit``     descending spiral around a site. Wide triangulation angles and
                  oblique views, so facades and roofs are recoverable. This is
                  the fixture that can legitimately reach a 3D structure verdict.
    ``corridor``  straight near-nadir strip. Good parallax along track, but no
                  facade observations - the honest outcome is a 2.5D terrain
                  product.

Why the previous fixture reconstructed 3 of 30 images
-----------------------------------------------------
Every material tiled a small texture (grass 15x, road 10x, walls 4x). Tiling
makes a repeating signal, and a repeating signal makes SIFT descriptors
ambiguous: matches are plentiful but wrong, so the mapper sees thousands of
correspondences and can register almost nothing. Blender's default AgX view
transform then flattened what contrast was left.

This generator paints one non-repeating texture per surface at ground-sampling
resolution, and renders with the Standard view transform.

Usage
-----
    python scripts/make_synthetic_flight.py --mode orbit
    python scripts/make_synthetic_flight.py --mode corridor
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Windows install locations, newest first. Overridable with --blender.
BLENDER_CANDIDATES = [
    Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
]

# --------------------------------------------------------------------------
# Capture geometry presets
# --------------------------------------------------------------------------
# Frame count and fps are chosen together against how stage 1 actually thins a
# video, because a fixture that ignores that produces an unusable keyframe set:
#
#   * Stage 1 keeps the sharpest frame in each non-overlapping window of
#     ``sharpness_window`` (8), then applies a stride measured in *original*
#     frame indices. Whenever that stride is under 8 the windowing dominates, so
#     the retained keyframe count is n_frames // 8.
#   * COLMAP's incremental mapper wants well under ~15 deg of viewpoint change
#     between images it has to match. 480 frames over 1.6 revolutions gives 60
#     keyframes 9.6 deg apart - comfortable.
#   * fps is then set so the implied ground speed is one a real multirotor
#     flies; stage 1 derives its stride from telemetry speed, so an unrealistic
#     speed produces an unrealistic stride.
MODES = {
    "orbit": {
        "out": "data/synthetic_flight",
        "n_frames": 480,
        "fps": 6.0,
        "revolutions": 1.6,
        "radius_start": 88.0,
        "radius_end": 62.0,
        "alt_start": 76.0,
        "alt_end": 48.0,
        "look_at": (0.0, 0.0, 7.0),
        "expected_verdict": "accept_3d",
        "note": "descending spiral, oblique gimbal - facades and roofs observed",
    },
    "corridor": {
        "out": "data/synthetic_flight_corridor",
        "n_frames": 320,
        "fps": 4.4,
        "y_start": -110.0,
        "y_end": 110.0,
        "altitude": 72.0,
        "look_ahead": 18.0,
        "expected_verdict": "terrain_2_5d",
        "note": "straight near-nadir strip - roofs and ground only, no facades",
    },
    # A close, steeply oblique descending spiral.
    #
    # Read this together with the C2 (sub-decimetre GSD) finding, because the
    # obvious explanation for that failure was the wrong one. The reported GSD
    # is not an image property: stage 7 computes it as
    # ``sqrt(bbox_area / n_dense_points)``, i.e. the mean spacing of the dense
    # cloud. On the `orbit` fixture that was sqrt(36,122 / 671,115) = 23.2 cm.
    # So GSD improves with dense *point density*, and the two things that
    # actually set it were both throwing samples away:
    #
    #   * keyframe selection kept 47 of 480 decoded frames. The run's own note
    #     reads "stride from GPS speed: 9.5 m/s, alt: 62 m AGL -> 8" --
    #     `frames.target_overlap` of 0.75 advancing 25% of a 49.6 m assumed
    #     footprint. Raising overlap to 0.90 takes the stride to 3 (~160
    #     keyframes, 3.4x the pixels). 90% front overlap is the normal
    #     recommendation for 3D capture, not a thumb on the scale.
    #   * `dense.resolution_level` of 1 then halved each of those images in both
    #     axes, costing a further 4x.
    #
    # Those two are config, not geometry, and they apply to any dataset. What
    # *this* mode adds is the geometric part, and one trap is worth recording:
    # flying lower does not by itself shrink the reconstructed footprint. The
    # footprint is set by the *depression angle*, atan((z - 7) / r). Drop the
    # altitude while holding the radius and the view gets shallower, the frame's
    # top edge grazes further out (at 30 m altitude and 8.8 deg it reaches
    # ~194 m), and `bbox_area` grows -- worsening the very number being chased.
    # So radius is pulled in with altitude to hold ~38 deg depression, matching
    # the orbit fixture that is already known to register 47/47, while slant
    # range falls from 112->74 m to 72->51 m. Render sampling then lands at
    # 4.8->3.4 cm/px, so `ground_tex_px` is raised to 8192 (3.66 cm/texel):
    # rendering finer than the texture manufactures blur, which would show up as
    # a *reconstruction* deficiency it isn't.
    #
    # The path is deliberately the same *shape* as `orbit`, whose parallax is
    # measured and known good (20.6 deg median triangulation, 47/47 registered):
    # only the standoff changes, so a GSD result here cannot be confounded with
    # a change in capture geometry.
    #
    # Run it with the two config levers above:
    #   --set frames.target_overlap=0.90 --set dense.resolution_level=0
    "closeup": {
        "out": "data/synthetic_flight_closeup",
        "n_frames": 600,
        "fps": 6.0,
        "revolutions": 2.2,
        "radius_start": 56.0,
        "radius_end": 40.0,
        "alt_start": 52.0,
        "alt_end": 38.0,
        "look_at": (0.0, 0.0, 7.0),
        "width": 1920,
        "height": 1080,
        "ground_tex_px": 8192,
        "expected_verdict": "accept_3d",
        "note": "close descending spiral at 1920x1080 holding ~38 deg "
                "depression - sized to demonstrate sub-decimetre GSD; run with "
                "frames.target_overlap=0.90 and dense.resolution_level=0",
    },
}

GROUND_EXTENT = 300.0        # metres, square
GROUND_TEX_PX = 6144         # ~4.9 cm/px, matched to the flight's GSD
SURFACE_TEX_PX = 2048

SITE_NAME = "Synthetic_Survey_Range_v4"

# Reference markers: physical 2 x 2 m ground plates at exact coordinates. Their
# separations are the scale check that does not need RTK.
REFERENCE_MARKERS = {
    "REF_SW": (-50.0, -50.0),
    "REF_SE": (50.0, -50.0),
    "REF_NE": (50.0, 50.0),
    "REF_NW": (-50.0, 50.0),
}

# name, centre (x, y), footprint (x, y), height, roof slab thickness
BUILDINGS = [
    ("Building_Alpha", (24.0, 6.0), (16.0, 24.0), 8.0, 0.4),
    ("Building_Beta", (-26.0, -14.0), (12.0, 14.0), 6.0, 0.4),
    ("Tower_Gamma", (-18.0, 26.0), (8.0, 8.0), 18.0, 0.4),
    ("Hangar_Delta", (6.0, -34.0), (30.0, 12.0), 9.5, 0.5),
    ("Block_Epsilon", (34.0, -22.0), (10.0, 10.0), 12.0, 0.4),
]

# name, centre (x, y), cube edge - deliberately simple solids whose dimensions
# can be read straight off a reconstruction.
REFERENCE_SOLIDS = [
    ("Cube_4m", (14.0, 42.0), 4.0),
    ("Cube_6m", (-44.0, 4.0), 6.0),
]


def is_inside_blender() -> bool:
    try:
        import bpy  # noqa: F401
        return True
    except ImportError:
        return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(MODES), default="orbit")
    p.add_argument("--out", default="", help="output directory (default: per-mode)")
    p.add_argument("--frames", type=int, default=0, help="override frame count")
    p.add_argument("--fps", type=float, default=0.0, help="override video fps")
    p.add_argument("--width", type=int, default=0, help="override render width")
    p.add_argument("--height", type=int, default=0, help="override render height")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--blender", default="", help="path to blender.exe")
    p.add_argument("--skip-textures", action="store_true")
    p.add_argument("--skip-render", action="store_true", help="rebuild textures and metadata only")
    return p.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> dict:
    cfg = dict(MODES[args.mode])
    cfg["mode"] = args.mode
    cfg["out"] = str((REPO_ROOT / (args.out or cfg["out"])).resolve())
    if args.frames:
        cfg["n_frames"] = args.frames
    if args.fps:
        cfg["fps"] = args.fps
    # Render size resolves mode value -> CLI override -> global default. It used
    # to be an unconditional `cfg["width"] = args.width`, which silently
    # discarded any per-mode resolution: a mode asking for 1920x1080 still
    # rendered at the argparse default of 1600x900, and the only symptom was a
    # GSD number that would not improve.
    cfg["width"] = args.width or cfg.get("width", 1600)
    cfg["height"] = args.height or cfg.get("height", 900)
    cfg["seed"] = args.seed
    return cfg


# ==========================================================================
# Host pass - textures, then Blender, then video encode
# ==========================================================================

def build_textures(out_dir: Path, seed: int, ground_tex_px: int = GROUND_TEX_PX) -> None:
    """Paint one non-repeating texture per surface class.

    Every texture is applied exactly once across its surface, so nothing in the
    scene repeats. Detail is layered at three scales because SIFT needs
    structure at more than one octave: a smooth low-frequency field, discrete
    mid-scale features at unique positions, and fine grain.

    ``ground_tex_px`` is per-mode because it has to track the flight's render
    sampling. 6144 px over the 300 m ground is 4.9 cm/texel, which suits a
    ~60 m orbit; a closer flight samples the ground at ~3 cm/px, and rendering
    finer than the texture does not add detail, it adds interpolation blur --
    which would then show up as a *reconstruction* deficiency it isn't.
    """
    import cv2
    import numpy as np

    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def low_frequency_field(h: int, w: int, cells: int, amplitude: float) -> np.ndarray:
        """Smooth blotchy variation - breaks up flat colour without repeating."""
        coarse = rng.normal(0.0, amplitude, (cells, cells, 3)).astype(np.float32)
        return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    def add_grain(img: np.ndarray, sigma: float) -> None:
        """Fine noise, added in row bands so a 6144^2 texture stays in memory."""
        h = img.shape[0]
        band = max(1, 2 ** 22 // max(1, img.shape[1] * 3))
        for y0 in range(0, h, band):
            y1 = min(h, y0 + band)
            img[y0:y1] += rng.normal(0.0, sigma, img[y0:y1].shape).astype(np.float32)

    # ---------------- ground: terrain, roads, markings, reference plates ----
    n = ground_tex_px
    px_per_m = n / GROUND_EXTENT

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        """World metres -> texture pixel. Blender's V axis runs bottom-up."""
        col = int(round((x_m + GROUND_EXTENT / 2.0) * px_per_m))
        row = int(round((GROUND_EXTENT / 2.0 - y_m) * px_per_m))
        return col, row

    def m_to_px(d_m: float) -> int:
        return max(1, int(round(d_m * px_per_m)))

    ground = np.zeros((n, n, 3), np.float32)
    ground[:] = (58.0, 116.0, 74.0)                      # BGR grass base
    ground += low_frequency_field(n, n, 24, 16.0)

    # Mid-scale vegetation: unique positions, varied size and tone.
    for _ in range(9000):
        cx, cy = int(rng.integers(0, n)), int(rng.integers(0, n))
        r = int(rng.integers(m_to_px(0.25), m_to_px(2.2)))
        tone = float(rng.uniform(-45, 45))
        cv2.circle(ground, (cx, cy), r, (44 + tone, 96 + tone * 1.4, 58 + tone), -1)

    # Bare-earth patches, so the ground is not uniformly one material.
    for _ in range(70):
        cx, cy = int(rng.integers(0, n)), int(rng.integers(0, n))
        axes = (int(rng.integers(m_to_px(3), m_to_px(14))), int(rng.integers(m_to_px(3), m_to_px(14))))
        cv2.ellipse(ground, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360,
                    (78.0, 104.0, 128.0), -1)

    # Roads, drawn once at true scale into the ground texture. Baking them in
    # avoids the z-fighting and duplicate-geometry seams of a separate slab.
    def draw_road(p0: tuple[float, float], p1: tuple[float, float], width_m: float,
                  dashed: bool = True) -> None:
        a, b = to_px(*p0), to_px(*p1)
        cv2.line(ground, a, b, (52.0, 52.0, 55.0), m_to_px(width_m))
        cv2.line(ground, a, b, (44.0, 44.0, 47.0), m_to_px(width_m * 0.92))
        # Solid edge lines.
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length < 1e-6:
            return
        nx, ny = -(p1[1] - p0[1]) / length, (p1[0] - p0[0]) / length
        for side in (-1.0, 1.0):
            off = side * width_m * 0.42
            cv2.line(ground,
                     to_px(p0[0] + nx * off, p0[1] + ny * off),
                     to_px(p1[0] + nx * off, p1[1] + ny * off),
                     (48.0, 188.0, 208.0), m_to_px(0.20))
        if not dashed:
            return
        steps = max(1, int(length / 9.0))
        for k in range(steps):
            t0, t1 = k / steps, k / steps + 3.0 / max(length, 1.0)
            if t1 > 1.0:
                break
            cv2.line(ground,
                     to_px(p0[0] + (p1[0] - p0[0]) * t0, p0[1] + (p1[1] - p0[1]) * t0),
                     to_px(p0[0] + (p1[0] - p0[0]) * t1, p0[1] + (p1[1] - p0[1]) * t1),
                     (232.0, 232.0, 232.0), m_to_px(0.22))

    draw_road((0.0, -140.0), (0.0, 140.0), 11.0)
    draw_road((-140.0, -60.0), (140.0, -60.0), 9.0)
    draw_road((-60.0, 10.0), (-60.0, 120.0), 7.0, dashed=False)

    # Apron with parking bays - strong, uniquely-placed linear features. Kept
    # clear of REF_NE so the plate is not painted over.
    apron_a, apron_b = to_px(58.0, 18.0), to_px(98.0, 58.0)
    cv2.rectangle(ground, apron_a, apron_b, (72.0, 72.0, 76.0), -1)
    for k in range(13):
        y = 20.0 + k * 3.0
        cv2.line(ground, to_px(60.0, y), to_px(76.0, y), (226.0, 226.0, 226.0), m_to_px(0.18))

    # Reference plates: white 2 x 2 m with a dark cross, unmistakable in a mesh.
    for name, (mx, my) in REFERENCE_MARKERS.items():
        c = to_px(mx, my)
        half = m_to_px(1.0)
        cv2.rectangle(ground, (c[0] - half, c[1] - half), (c[0] + half, c[1] + half),
                      (238.0, 238.0, 240.0), -1)
        cv2.line(ground, (c[0] - half, c[1]), (c[0] + half, c[1]), (30.0, 30.0, 34.0), m_to_px(0.22))
        cv2.line(ground, (c[0], c[1] - half), (c[0], c[1] + half), (30.0, 30.0, 34.0), m_to_px(0.22))

    add_grain(ground, 6.0)
    cv2.imwrite(str(tex_dir / "tex_ground.png"), np.clip(ground, 0, 255).astype(np.uint8))

    # ---------------- facades: one distinct texture per building ------------
    wall_palettes = [
        (168.0, 172.0, 176.0), (150.0, 158.0, 172.0), (132.0, 148.0, 166.0),
        (176.0, 176.0, 168.0), (142.0, 152.0, 158.0),
    ]
    s = SURFACE_TEX_PX
    for idx, (bname, _c, _f, _h, _t) in enumerate(BUILDINGS):
        base = wall_palettes[idx % len(wall_palettes)]
        wall = np.zeros((s, s, 3), np.float32)
        wall[:] = base
        wall += low_frequency_field(s, s, 12, 10.0)

        # Window grid. Rows and columns are jittered per building, and each
        # pane gets its own tone, so no two facades correlate.
        rows = int(rng.integers(4, 7))
        cols = int(rng.integers(5, 9))
        pad = s // 18
        cw = (s - 2 * pad) // cols
        ch = (s - 2 * pad) // rows
        for r in range(rows):
            for c in range(cols):
                x0 = pad + c * cw + int(rng.integers(0, max(1, cw // 8)))
                y0 = pad + r * ch + int(rng.integers(0, max(1, ch // 8)))
                x1, y1 = x0 + int(cw * 0.62), y0 + int(ch * 0.58)
                cv2.rectangle(wall, (x0, y0), (x1, y1), (214.0, 216.0, 220.0), -1)
                tone = float(rng.uniform(-24, 24))
                cv2.rectangle(wall, (x0 + 5, y0 + 5), (x1 - 5, y1 - 5),
                              (46.0 + tone, 40.0 + tone, 34.0 + tone), -1)

        # Grunge: unique streaks that give the mapper something aperiodic.
        for _ in range(260):
            x0, y0 = int(rng.integers(0, s)), int(rng.integers(0, s))
            cv2.line(wall, (x0, y0),
                     (x0 + int(rng.integers(-70, 70)), y0 + int(rng.integers(-70, 70))),
                     tuple(float(v + rng.uniform(-34, 34)) for v in base),
                     int(rng.integers(1, 5)))
        add_grain(wall, 5.0)
        cv2.imwrite(str(tex_dir / f"tex_wall_{idx}.png"), np.clip(wall, 0, 255).astype(np.uint8))

        # Roof: tile courses plus rooftop plant, again unique per building.
        roof = np.zeros((s, s, 3), np.float32)
        roof[:] = (float(rng.uniform(48, 74)), float(rng.uniform(62, 94)), float(rng.uniform(150, 190)))
        roof += low_frequency_field(s, s, 16, 12.0)
        course = max(6, s // int(rng.integers(40, 70)))
        for y in range(0, s, course):
            cv2.line(roof, (0, y), (s, y), (28.0, 38.0, 96.0), 2)
        for _ in range(28):
            x0, y0 = int(rng.integers(0, s - 200)), int(rng.integers(0, s - 200))
            cv2.rectangle(roof, (x0, y0), (x0 + int(rng.integers(40, 170)), y0 + int(rng.integers(40, 170))),
                          (118.0, 122.0, 126.0), -1)
        add_grain(roof, 5.0)
        cv2.imwrite(str(tex_dir / f"tex_roof_{idx}.png"), np.clip(roof, 0, 255).astype(np.uint8))

    # ---------------- reference solids: high-contrast, non-repeating --------
    for idx, (sname, _c, _e) in enumerate(REFERENCE_SOLIDS):
        cube = np.zeros((s, s, 3), np.float32)
        cube[:] = (208.0, 210.0, 214.0)
        for _ in range(90):
            x0, y0 = int(rng.integers(0, s)), int(rng.integers(0, s))
            cv2.circle(cube, (x0, y0), int(rng.integers(12, 90)),
                       (float(rng.uniform(20, 90)), float(rng.uniform(20, 90)), float(rng.uniform(20, 90))), -1)
        add_grain(cube, 6.0)
        cv2.imwrite(str(tex_dir / f"tex_solid_{idx}.png"), np.clip(cube, 0, 255).astype(np.uint8))

    print(f"[OK] textures written to {tex_dir}")


def encode_video(out_dir: Path, fps: float) -> Path:
    import imageio_ffmpeg

    frames_dir = out_dir / "frames"
    video_out = out_dir / "flight.mp4"
    n = len(list(frames_dir.glob("frame_*.png")))
    if n == 0:
        raise RuntimeError(f"no rendered frames in {frames_dir}")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-framerate", f"{fps:.4f}",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-crf", "16",              # near-lossless; compression artefacts cost keypoints
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(video_out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[OK] encoded {n} frames at {fps:.2f} fps -> {video_out}")
    return video_out


def host_main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = resolve_settings(args)
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_textures:
        build_textures(out_dir, cfg["seed"], cfg.get("ground_tex_px", GROUND_TEX_PX))

    blender = Path(args.blender) if args.blender else next(
        (p for p in BLENDER_CANDIDATES if p.exists()), None
    )
    if blender is None or not blender.exists():
        print("[FAIL] Blender not found. Pass --blender <path to blender.exe>.")
        return 2

    (out_dir / "_settings.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if not args.skip_render:
        for stale in (out_dir / "frames").glob("frame_*.png"):
            stale.unlink()

    print(f"[*] {blender.name}: rendering {cfg['n_frames']} frames "
          f"({cfg['width']}x{cfg['height']}, mode={cfg['mode']})")
    res = subprocess.run([
        str(blender), "--background", "--factory-startup",
        "--python", str(Path(__file__).resolve()),
        "--", "--settings", str(out_dir / "_settings.json"),
    ] + (["--skip-render"] if args.skip_render else []))
    if res.returncode != 0:
        print(f"[FAIL] Blender exited {res.returncode}")
        return res.returncode

    if not args.skip_render:
        encode_video(out_dir, cfg["fps"])

    print(f"\n[OK] fixture ready: {out_dir}")
    print(f"     next: dronemap run --video {out_dir / 'flight.mp4'} "
          f"--telemetry {out_dir / 'flight_telemetry.csv'} --run-id <name>")
    return 0


# ==========================================================================
# Blender pass
# ==========================================================================

def blender_main() -> None:
    import bpy
    import mathutils

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--settings", required=True)
    p.add_argument("--skip-render", action="store_true")
    bargs = p.parse_args(argv)

    cfg = json.loads(Path(bargs.settings).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out"])
    tex_dir = out_dir / "textures"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- scene and render settings ----------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    engine_ids = [i.identifier for i in scene.render.bl_rna.properties["engine"].enum_items]
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if candidate in engine_ids:
            scene.render.engine = candidate
            break
    print(f"[*] render engine: {scene.render.engine}")

    scene.render.resolution_x = cfg["width"]
    scene.render.resolution_y = cfg["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # AgX / Filmic crush local contrast, which directly costs SIFT keypoints.
    # Photogrammetry wants the linear-to-sRGB transform and nothing else.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception as exc:            # pragma: no cover - Blender version drift
        print(f"[warn] could not force Standard view transform: {exc}")

    for attr, value in (("taa_render_samples", 32), ("use_gtao", True)):
        try:
            setattr(scene.eevee, attr, value)
        except Exception:
            pass

    world = bpy.data.worlds.new("DroneWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.68, 0.90, 1.0)
        bg.inputs["Strength"].default_value = 1.1
    scene.world = world

    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 3.4
    sun_data.color = (1.0, 0.97, 0.92)
    sun_data.angle = math.radians(2.5)          # soft-ish edges, still directional
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(42), math.radians(14), math.radians(38))
    scene.collection.objects.link(sun)

    # ---------------- materials --------------------------------------------
    def image_material(name: str, filename: str, projection: str = "FLAT"):
        """One image, mapped exactly once. No tiling, no scale multiplier."""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.72
        try:
            bsdf.inputs["Metallic"].default_value = 0.0
        except KeyError:
            pass
        tex = nodes.new("ShaderNodeTexImage")
        path = tex_dir / filename
        if not path.exists():
            raise RuntimeError(f"missing texture {path}; run without --skip-textures")
        tex.image = bpy.data.images.load(str(path))
        tex.extension = "EXTEND"
        tex.projection = projection
        if projection == "BOX":
            tex.projection_blend = 0.25
        coord = nodes.new("ShaderNodeTexCoord")
        links.new(coord.outputs["Generated"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    def flat_material(name: str, rgb: tuple[float, float, float]):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.8
        return mat

    mesh_objects = []

    def add_box(name: str, centre_xy, footprint, height: float, z0: float, material):
        bpy.ops.mesh.primitive_cube_add(size=1.0,
                                        location=(centre_xy[0], centre_xy[1], z0 + height / 2.0))
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (footprint[0], footprint[1], height)
        obj.data.materials.append(material)
        mesh_objects.append(obj)
        return obj

    # ---------------- ground ------------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=GROUND_EXTENT, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(image_material("MatGround", "tex_ground.png"))
    mesh_objects.append(ground)

    # ---------------- buildings --------------------------------------------
    gt_objects = []
    for idx, (bname, centre, footprint, height, slab) in enumerate(BUILDINGS):
        add_box(bname, centre, footprint, height, 0.0,
                image_material(f"MatWall{idx}", f"tex_wall_{idx}.png", projection="BOX"))
        add_box(f"Roof_{bname}", centre, (footprint[0] + 0.6, footprint[1] + 0.6), slab, height,
                image_material(f"MatRoof{idx}", f"tex_roof_{idx}.png", projection="BOX"))
        gt_objects.append({
            "name": bname,
            "width_x_m": footprint[0],
            "length_y_m": footprint[1],
            "height_z_m": height + slab,
            "centre_xyz_m": [centre[0], centre[1], (height + slab) / 2.0],
        })

    # ---------------- reference solids and plates ---------------------------
    for idx, (sname, centre, edge) in enumerate(REFERENCE_SOLIDS):
        add_box(sname, centre, (edge, edge), edge, 0.0,
                image_material(f"MatSolid{idx}", f"tex_solid_{idx}.png", projection="BOX"))
        gt_objects.append({
            "name": sname,
            "width_x_m": edge,
            "length_y_m": edge,
            "height_z_m": edge,
            "centre_xyz_m": [centre[0], centre[1], edge / 2.0],
        })

    # 12 cm plates: thin enough not to distort a DSM, thick enough to survive
    # meshing, and their painted crosses are already in the ground texture.
    plate_mat = flat_material("MatPlate", (0.88, 0.88, 0.90))
    for mname, (mx, my) in REFERENCE_MARKERS.items():
        add_box(mname, (mx, my), (2.0, 2.0), 0.12, 0.0, plate_mat)

    # ---------------- vegetation (occlusion realism) ------------------------
    import random
    rnd = random.Random(cfg["seed"] + 11)
    trunk_mat = flat_material("MatTrunk", (0.22, 0.15, 0.09))
    canopy_mat = flat_material("MatCanopy", (0.10, 0.26, 0.09))
    keep_clear = [(c[0], c[1], max(f) * 0.9 + 8.0) for _n, c, f, _h, _s in BUILDINGS]
    placed = 0
    while placed < 16:
        tx, ty = rnd.uniform(-90, 90), rnd.uniform(-90, 90)
        if any(math.hypot(tx - cx, ty - cy) < rad for cx, cy, rad in keep_clear):
            continue
        if abs(tx) < 9.0 or abs(ty + 60.0) < 8.0:       # keep the roads clear
            continue
        h = rnd.uniform(4.0, 8.5)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=h, location=(tx, ty, h / 2.0))
        trunk = bpy.context.active_object
        trunk.name = f"Trunk_{placed:02d}"
        trunk.data.materials.append(trunk_mat)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rnd.uniform(1.8, 3.2),
                                              location=(tx, ty, h + 1.2))
        canopy = bpy.context.active_object
        canopy.name = f"Canopy_{placed:02d}"
        canopy.data.materials.append(canopy_mat)
        mesh_objects.extend([trunk, canopy])
        placed += 1

    # ---------------- ground truth: dimensions and reference distances ------
    marker_names = list(REFERENCE_MARKERS)
    reference_distances = []
    for i in range(len(marker_names)):
        for j in range(i + 1, len(marker_names)):
            a, b = marker_names[i], marker_names[j]
            ax, ay = REFERENCE_MARKERS[a]
            bx, by = REFERENCE_MARKERS[b]
            reference_distances.append({
                "from": a, "to": b,
                "distance_m": round(math.hypot(bx - ax, by - ay), 4),
            })
    (out_dir / "gt_dimensions.json").write_text(json.dumps({
        "site_name": SITE_NAME,
        "ground_extent_m": GROUND_EXTENT,
        "ground_truth_objects": gt_objects,
        "reference_markers": {k: [v[0], v[1], 0.12] for k, v in REFERENCE_MARKERS.items()},
        "reference_distances": reference_distances,
    }, indent=2), encoding="utf-8")

    # ---------------- ground truth mesh ------------------------------------
    for o in bpy.data.objects:
        o.select_set(o in mesh_objects)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.wm.ply_export(filepath=str(out_dir / "gt_mesh.ply"), export_selected_objects=True)
    bpy.ops.wm.obj_export(filepath=str(out_dir / "gt_mesh.obj"), export_selected_objects=True)
    print("[OK] ground-truth mesh exported")

    # ---------------- camera and flight path -------------------------------
    cam_data = bpy.data.cameras.new("DroneCamera")
    cam_data.lens = 28.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("DroneCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    n_frames = int(cfg["n_frames"])
    fps = float(cfg["fps"])
    mode = cfg["mode"]

    def station(idx: int):
        """Camera position and look-at target for frame ``idx``."""
        t = idx / float(max(1, n_frames - 1))
        # Dispatch on the geometry the mode actually declares, not on its name.
        # `mode == "orbit"` meant a second spiral mode silently fell through to
        # the corridor branch and raised KeyError on "y_start" -- the same class
        # of bug as inferring a run's product type from its run_id.
        if "revolutions" in cfg:
            ang = t * cfg["revolutions"] * 2.0 * math.pi
            r = cfg["radius_start"] + t * (cfg["radius_end"] - cfg["radius_start"])
            z = cfg["alt_start"] + t * (cfg["alt_end"] - cfg["alt_start"])
            z += 1.6 * math.sin(t * 2.0 * math.pi * 5.0)   # gentle vertical ripple
            return (r * math.cos(ang), r * math.sin(ang), z), tuple(cfg["look_at"])
        y = cfg["y_start"] + t * (cfg["y_end"] - cfg["y_start"])
        x = 2.5 * math.sin(t * math.pi * 2.0)              # realistic lateral drift
        z = cfg["altitude"] + 1.2 * math.sin(t * math.pi * 3.0)
        return (x, y, z), (x * 0.4, y + cfg["look_ahead"], 0.0)

    ref_lat, ref_lon, ref_alt = 28.58330, 77.21670, 214.0
    lat_per_m = 1.0 / 111_139.0
    lon_per_m = 1.0 / (111_139.0 * math.cos(math.radians(ref_lat)))

    rnd_gnss = random.Random(cfg["seed"] + 4242)
    gt_poses: dict[str, dict] = {}
    tum_lines: list[str] = []
    telemetry = ["timestamp_s,latitude,longitude,abs_alt,rel_alt,speed,gimbal_pitch,accuracy_m"]

    stations = [station(i) for i in range(n_frames)]
    path_length = sum(
        math.dist(stations[i][0], stations[i + 1][0]) for i in range(n_frames - 1)
    )
    duration = (n_frames - 1) / fps if n_frames > 1 else 1.0
    print(f"[*] path length {path_length:.1f} m over {duration:.1f} s "
          f"-> {path_length / duration:.1f} m/s implied ground speed")

    for idx in range(n_frames):
        (x, y, z), target = stations[idx]
        cam.location = (x, y, z)
        direction = mathutils.Vector(target) - mathutils.Vector((x, y, z))
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        frame_name = f"frame_{idx:04d}.png"
        if not bargs.skip_render:
            scene.render.filepath = str(frames_dir / frame_name)
            bpy.ops.render.render(write_still=True)
            if idx % 25 == 0:
                print(f"    rendered {idx + 1}/{n_frames}")

        quat = cam.rotation_euler.to_quaternion()
        horiz = math.hypot(direction.x, direction.y)
        gimbal_pitch = -math.degrees(math.atan2(-direction.z, max(horiz, 1e-6)))
        gt_poses[frame_name] = {
            "x": x, "y": y, "z": z,
            "qw": quat.w, "qx": quat.x, "qy": quat.y, "qz": quat.z,
            "gimbal_pitch_deg": round(gimbal_pitch, 3),
            "alt_m": z,
        }

        ts = idx / fps
        tum_lines.append(
            f"{ts:.3f} {x:.4f} {y:.4f} {z:.4f} "
            f"{quat.x:.6f} {quat.y:.6f} {quat.z:.6f} {quat.w:.6f}"
        )

        # Standalone-GNSS noise, not RTK: 1.8 m horizontal, 3.2 m vertical 1-sigma.
        gx = rnd_gnss.gauss(0.0, 1.8)
        gy = rnd_gnss.gauss(0.0, 1.8)
        gz = rnd_gnss.gauss(0.0, 3.2)
        if idx == 0:
            speed = 0.0
        else:
            speed = math.dist(stations[idx][0], stations[idx - 1][0]) * fps
        telemetry.append(
            f"{ts:.3f},{ref_lat + (y + gy) * lat_per_m:.8f},"
            f"{ref_lon + (x + gx) * lon_per_m:.8f},"
            f"{ref_alt + z + gz:.2f},{z + gz:.2f},{speed:.2f},"
            f"{gimbal_pitch:.1f},2.50"
        )

    (out_dir / "gt_poses.json").write_text(json.dumps(gt_poses, indent=2), encoding="utf-8")
    (out_dir / "gt_trajectory.txt").write_text("\n".join(tum_lines) + "\n", encoding="utf-8")
    (out_dir / "flight_telemetry.csv").write_text("\n".join(telemetry) + "\n", encoding="utf-8")

    fx = cam_data.lens / cam_data.sensor_width * cfg["width"]
    centres = [s[0] for s in stations]
    max_baseline = max(
        math.dist(a, b) for i, a in enumerate(centres) for b in centres[i + 1:]
    ) if n_frames > 1 else 0.0
    (out_dir / "fixture.json").write_text(json.dumps({
        "site_name": SITE_NAME,
        "mode": mode,
        "note": cfg["note"],
        "expected_verdict": cfg["expected_verdict"],
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": round(duration, 2),
        "resolution": [cfg["width"], cfg["height"]],
        "intrinsics_pinhole": {
            "fx": round(fx, 3), "fy": round(fx, 3),
            "cx": cfg["width"] / 2.0, "cy": cfg["height"] / 2.0,
            "camera_params_simple_radial": f"{fx:.3f},{cfg['width'] / 2.0},{cfg['height'] / 2.0},0.0",
        },
        "capture_geometry": {
            "path_length_m": round(path_length, 2),
            "max_baseline_m": round(max_baseline, 2),
            "implied_ground_speed_mps": round(path_length / duration, 2),
            "altitude_range_m": [round(min(c[2] for c in centres), 2),
                                 round(max(c[2] for c in centres), 2)],
        },
        "gnss": {"model": "standalone", "sigma_horizontal_m": 1.8, "sigma_vertical_m": 3.2},
        "reference_datum": {"lat": ref_lat, "lon": ref_lon, "alt_m": ref_alt},
    }, indent=2), encoding="utf-8")
    print(f"[OK] ground truth written to {out_dir}")


if __name__ == "__main__":
    if is_inside_blender():
        blender_main()
    else:
        raise SystemExit(host_main(sys.argv[1:]))
