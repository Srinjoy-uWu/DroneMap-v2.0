"""Pipeline configuration.

Config is layered, lowest priority first:

1.  the defaults coded into the models below
2.  ``configs/default.yaml``
3.  a profile file (e.g. ``configs/lowvram.yaml``) passed via ``--profile``
4.  ``DRONEMAP_*`` environment variables
5.  explicit ``--set key=value`` overrides on the command line

Everything a stage needs must live here rather than being hardcoded, because the
whole point of the Core/Stretch split is being able to swap an implementation
without touching the orchestrator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class FrameConfig(BaseModel):
    """Stage 1 - decode, blur-filter and baseline-aware keyframe selection."""

    # OpenCV's Windows backend fails on some drone codecs (10-bit HEVC in
    # particular). "auto" tries OpenCV and falls back to piping from ffmpeg.
    decoder: Literal["auto", "opencv", "ffmpeg"] = "auto"

    # Downscale cap. 4K frames blow up COLMAP runtime and disk for little gain
    # at these baselines; 1920 keeps a full run inside ~10-15 GB.
    max_long_edge: int = 1920
    jpeg_quality: int = 95

    # Hard cap on keyframes. Guards both disk and COLMAP mapper runtime.
    max_keyframes: int = 600

    # Blur defence: score every frame, then keep only the sharpest frame within
    # each window of this many consecutive frames.
    sharpness_window: int = 8

    # Baseline-aware spacing. Consecutive video frames overlap ~99% and carry
    # almost no triangulation baseline, so we thin the stream until successive
    # keyframes have roughly this much forward overlap. Derived from telemetry
    # ground speed when available, else from median optical-flow magnitude.
    target_overlap: float = 0.75
    min_frame_stride: int = 2
    max_frame_stride: int = 120

    # Maximum course change tolerated between consecutive keyframes, degrees.
    # Incremental SfM matches reliably well under ~15 deg of viewpoint change;
    # 10 leaves margin. Only binds on curved flight (orbits, banked turns),
    # where a tracking gimbal hides the viewpoint change from image-space
    # overlap estimators.
    max_keyframe_turn_deg: float = 10.0

    # Reject frames whose sharpness is below this fraction of the clip median,
    # even if they win their window - a whole window can be blurred.
    min_sharpness_ratio: float = 0.35

    clahe: bool = False
    clahe_clip: float = 2.0

    @field_validator("target_overlap")
    @classmethod
    def _overlap_range(cls, v: float) -> float:
        if not 0.3 <= v <= 0.95:
            raise ValueError("target_overlap must be in [0.30, 0.95]")
        return v


class MaskConfig(BaseModel):
    """Stage 2 - dynamic object masking (Core: YOLOv8-seg)."""

    enabled: bool = True
    backend: Literal["yolo", "sam2", "none"] = "yolo"
    model: str = "yolov8s-seg.pt"
    conf: float = 0.25
    # Grow masks so the soft edge of a moving object is excluded too.
    dilate_px: int = 12
    device: str = "auto"
    # COCO class names treated as dynamic. Anything that can move under its own
    # power, plus boats/trains which are large enough to wreck a reconstruction.
    dynamic_classes: list[str] = Field(
        default_factory=lambda: [
            "person", "bicycle", "car", "motorcycle", "bus", "train",
            "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
        ]
    )
    # If masks would cover more than this fraction of a frame, drop the frame
    # instead - it has nothing useful left to contribute.
    max_masked_fraction: float = 0.6


class PoseConfig(BaseModel):
    """Stage 3a - structure-from-motion (Core: COLMAP; Stretch: MASt3R)."""

    # "colmap"  → Guaranteed Core path (always works, classical SfM)
    # "mast3r"  → Stretch path (better on single-pass, low-overlap footage)
    backend: Literal["colmap", "mast3r"] = "colmap"

    # ------ COLMAP settings (ignored when backend=mast3r) ------
    # SIMPLE_RADIAL (f, cx, cy, k1) enforces square pixels and robust radial distortion,
    # preventing the severe focal/distortion overfitting that OPENCV suffers on forward drone paths.
    camera_model: str = "SIMPLE_RADIAL"
    # Optional known intrinsics, e.g. "995.6,995.6,640,360" for a PINHOLE
    # camera. This removes focal-length ambiguity in low-parallax drone video.
    camera_params: str | None = None
    single_camera: bool = True
    use_gpu: bool = True
    max_num_features: int = 16384

    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    sequential_overlap: int = 10
    quadratic_overlap: bool = True
    loop_detection: bool = True
    loop_detection_period: int = 10

    # Relaxed from COLMAP's default 16.0 - a single forward pass simply does not
    # offer wide triangulation angles, and the default refuses to initialise.
    init_min_tri_angle: float = 4.0
    # Below this registered fraction the reconstruction is not usable and the
    # escalation ladder kicks in (see stage3_pose.py).
    min_registered_fraction: float = 0.55
    # A few accidental feature tracks can technically form a COLMAP model but
    # cannot describe a building, road, or terrain surface.  Stop before MVS
    # produces a visually misleading blob.
    min_sparse_points: int = 20
    escalate_on_failure: bool = True
    ba_refine_principal_point: bool = False

    # ------ MASt3R settings (ignored when backend=colmap) ------
    # Max images per MASt3R batch. 80-100 fits in 6 GB VRAM.
    mast3r_batch_size: int = 80
    # Number of global-alignment iterations.
    mast3r_niter: int = 300
    # Confidence threshold for filtering points (lower = more points, more noise).
    mast3r_min_conf: float = 3.0


class DepthConfig(BaseModel):
    """Stage 4a - monocular metric depth (Stretch).

    Depth maps are fused into the reconstruction as additional geometric
    constraints in Stage 3b (georef) and can be exported as dense products
    in Stage 7.  Two backends are supported:

    ``depth_anything``  — Depth Anything V2 (relative depth, scaled to metric
                          using telemetry altitude).  Fast, low VRAM.
    ``metric3d``        — Metric3D v2 (absolute metric depth from a single
                          image, no telemetry needed).  Higher quality but
                          slower and needs ~6 GB VRAM for the ViT-L variant.
    """

    enabled: bool = False
    backend: Literal["depth_anything", "metric3d"] = "depth_anything"

    # Depth Anything V2 model size. "small" runs in < 1 GB VRAM.
    depth_anything_model: Literal["small", "base", "large"] = "small"

    # Metric3D v2 backbone. "vit_small" fits in 6 GB VRAM.
    metric3d_model: Literal["vit_small", "vit_large"] = "vit_small"

    # Run depth on every Nth keyframe (depth is slow; we don't need every frame).
    stride: int = 5

    device: str = "auto"

    # Persist depth maps as 16-bit PNG (millimetres).  Large but lossless.
    save_depth_png: bool = True


class GeorefConfig(BaseModel):
    """Stage 3b - metric scale and georeferencing."""

    method: Literal["auto", "pose_prior", "aligner", "none"] = "auto"

    # 1-sigma GNSS position uncertainty in metres, used as the BA prior weight.
    # These are standalone-GNSS defaults; override to ~0.02/0.03 for RTK/PPK.
    gnss_std_xy: float = 2.5
    gnss_std_z: float = 5.0

    # RANSAC threshold for model_aligner, in metres.
    alignment_max_error: float = 3.0

    # Camera phase centre relative to the GNSS antenna, in the body frame
    # (metres, x=right y=forward z=up). Getting this wrong injects a constant
    # offset even with perfect RTK.
    lever_arm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Camera clock minus GNSS clock, seconds. offset x ground speed shows up as
    # an along-track bias, so it is worth measuring rather than assuming zero.
    time_offset_s: float = 0.0

    target_crs: str = "auto"  # "auto" picks the UTM zone from the first fix


class DenseConfig(BaseModel):
    """Stage 5 - OpenMVS dense point cloud."""

    # 0 = full resolution. 1 halves it. On a 6 GB / 16 GB laptop, 1 is the
    # sweet spot; densification is RAM-bound rather than VRAM-bound.
    resolution_level: int = 1
    max_resolution: int = 1920
    number_views: int = 6
    number_views_fuse: int = 3
    estimate_colors: int = 2
    estimate_normals: int = 2
    # Below this count OpenMVS results are usually sparse artifacts rather
    # than an interpretable scene.  Failing clearly is better than exporting a
    # misleading mesh.
    min_dense_points: int = 5000


class MeshConfig(BaseModel):
    """Stage 6 - Mesh reconstruction, refinement and texturing."""

    mode: Literal["auto", "openmvs", "terrain_2.5d"] = "auto"
    refine: bool = True
    min_point_distance: float = 2.5
    decimate: float = 1.0
    texture_size: int = 8192
    # RefineMesh is the slowest step by far; skip it when iterating.
    refine_max_views: int = 8
    # OpenMVS seam levelling blends per-view colour differences across patch
    # borders. It is on by default because it removes visible seams -- but it
    # solves a Poisson system per patch, and when the per-view estimates
    # disagree strongly it converges to zero over the patch interior, leaving
    # only the dilated border padding coloured. That is a black model with
    # coloured lines through it. Kept on, with a measured retry: see
    # ``max_texture_black_fraction``.
    seam_leveling: bool = True
    # If more than this fraction of the mesh's UV samples land on black texels,
    # re-texture with seam levelling off and keep whichever atlas is better.
    # 0.20 is well above what a legitimately shadowed capture produces (the
    # worst healthy fixture measured 0.10 whole-atlas, near 0 at UV samples) and
    # far below the 0.98 the collapse produces, so the two are not confusable.
    max_texture_black_fraction: float = 0.20
    # Remove floating shards left by multi-view reconstruction before texturing.
    min_component_faces: int = 200
    max_components: int = 12
    # 2.5D Terrain surface mesh settings
    terrain_grid_dim: int = 250
    terrain_grid_max: int = 400
    # Reject a ground-plane fit tilted further than this from the known
    # vertical. Measured on this repository's runs, the unconstrained PCA fit
    # returned 60-85 deg on five of six real scenes, which put the 2.5D product
    # on a plane with no relation to the ground.
    terrain_max_tilt_deg: float = 15.0


class SemanticConfig(BaseModel):
    """Stage 4b - semantic layers."""

    enabled: bool = False  # off until an aerial-domain checkpoint is verified
    model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    # A Cityscapes/ADE checkpoint is street-level and will NOT transfer to nadir
    # aerial imagery. Keep this False only when the checkpoint is UAVid /
    # LoveDA / ISPRS tuned, and record the check in the report.
    checkpoint_domain_verified: bool = False
    batch_size: int = 2


class QualityConfig(BaseModel):
    """Stage-gate thresholds for the capture-quality assessment.

    Every threshold here is a ratio or an angle, never a count, because counts
    cannot distinguish a building from a flat sheet of tarmac.  See
    ``dronemap/quality.py`` for the measurements these gate.

    Calibration note: the defaults are set from measured runs.  A 7 s hovering
    clip over an urban road produced median triangulation angle ~1.5deg,
    baseline/depth 0.29, depth dynamic range 1.7x, and a mesh with planarity
    0.038 -- it passed every count-based gate and exported a road carpet
    labelled as a 3D model.  These thresholds reject that capture.
    """

    enabled: bool = True
    # Refuse to run MVS when the pose-stage verdict is REJECT.  Turning this
    # off still records the verdict; it just does not stop the pipeline.
    block_on_reject: bool = True
    # Downgrade a TERRAIN_2_5D verdict to a 2.5D surface product instead of
    # letting ReconstructMesh emit a "3D model" that is really a sheet.
    route_terrain_to_2_5d: bool = True

    # ---- pose-stage gates (camera geometry) ----
    min_registered_images: int = 8
    min_registered_fraction: float = 0.55
    min_sparse_points: int = 500
    # Median per-point triangulation angle.  COLMAP will happily triangulate
    # at fractions of a degree; below ~2deg the depth error exceeds the
    # structure being measured.
    reject_tri_angle_deg: float = 2.0
    min_tri_angle_deg: float = 5.0
    # Max flight baseline over median scene depth.  Classic photogrammetric
    # guidance is a base/height ratio of ~0.3-0.5 for reliable height.
    reject_baseline_depth_ratio: float = 0.10
    min_baseline_depth_ratio: float = 0.30
    # Upper sanity bound.  Real aerial capture never flies a baseline many
    # times the scene distance; a huge ratio means the solution collapsed onto
    # the camera centres.  Without this, a degenerate model scores as the best
    # capture in the dataset because every minimum-threshold gate passes.
    max_baseline_depth_ratio: float = 20.0
    # Median scene depth must be at least this multiple of the baseline.
    min_depth_over_baseline: float = 0.02
    # p95/p5 of scene depth within a single view.  A nadir pass over flat
    # ground legitimately scores ~1.0, which is why falling below this
    # demotes to terrain rather than rejecting.
    min_depth_dynamic_range: float = 1.8
    # Fraction of camera travel along the optical axis; warn-only, because
    # forward flight is legitimate, it just makes the image centre unreliable.
    max_forward_motion_ratio: float = 0.85
    min_mean_track_length: float = 3.0
    # max_baseline / path_length.  A continuous one-directional pass scores
    # near 1.0; a hover or an out-and-back scores low despite a long path.
    min_path_efficiency: float = 0.60

    # ---- structure gates (reconstructed shape) ----
    min_structure_points: int = 5000
    # Third PCA singular ratio: cloud thickness over cloud width.
    reject_planarity: float = 0.06
    terrain_planarity: float = 0.12
    # Vertical relief over horizontal footprint.
    terrain_relief_ratio: float = 0.04
    min_above_ground_fraction: float = 0.02


class ExportConfig(BaseModel):
    """Stage 7 - deliverables."""

    # Ground sample distance for the raster products, metres/pixel.
    # "auto" derives it from the reconstructed point density.
    raster_gsd: float | Literal["auto"] = "auto"
    dsm: bool = True
    dtm: bool = True
    orthomosaic: bool = True
    point_cloud_laz: bool = True
    mesh_glb: bool = True
    # Voxel size for the exported cloud, metres. 0 disables downsampling.
    cloud_voxel_size: float = 0.0


class ChunkConfig(BaseModel):
    """Chunked processing.

    The Core path runs a single chunk over all keyframes. The chunking seam
    exists so the near-real-time progressive tier can be added later without
    restructuring the orchestrator.
    """

    enabled: bool = False
    size: int = 150
    overlap: int = 20


class Config(BaseModel):
    """Root configuration."""

    data_root: Path = REPO_ROOT / "data"
    tools_root: Path = REPO_ROOT / "tools"

    # Explicit binary paths. Leave empty to auto-discover in tools_root, then PATH.
    colmap_bin: Path | None = None
    openmvs_dir: Path | None = None
    ffmpeg_bin: Path | None = None
    blender_bin: Path | None = None

    frames: FrameConfig = Field(default_factory=FrameConfig)
    masks: MaskConfig = Field(default_factory=MaskConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    georef: GeorefConfig = Field(default_factory=GeorefConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    semantics: SemanticConfig = Field(default_factory=SemanticConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    chunks: ChunkConfig = Field(default_factory=ChunkConfig)

    # Delete large intermediates (undistorted images, dense .mvs) after export.
    purge_intermediates: bool = False
    # Refuse to start a run with less than this much free disk, in GB.
    min_free_disk_gb: float = 20.0


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    """Parse a ``--set key=value`` value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set '{dotted}': '{part}' is not a section")
    node[parts[-1]] = value


def _env_overrides() -> dict[str, Any]:
    """Read ``DRONEMAP_FRAMES__MAX_KEYFRAMES=400``-style variables."""
    out: dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith("DRONEMAP_"):
            continue
        dotted = name[len("DRONEMAP_"):].lower().replace("__", ".")
        _apply_dotted(out, dotted, _coerce(raw))
    return out


def load_config(
    profile: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Build a :class:`Config` from the layered sources described in the module docstring."""
    merged: dict[str, Any] = {}

    default_yaml = CONFIG_DIR / "default.yaml"
    if default_yaml.exists():
        merged = _deep_merge(merged, yaml.safe_load(default_yaml.read_text()) or {})

    if profile:
        path = Path(profile)
        if not path.exists():
            path = CONFIG_DIR / f"{profile}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"profile not found: {profile}")
        merged = _deep_merge(merged, yaml.safe_load(path.read_text()) or {})

    merged = _deep_merge(merged, _env_overrides())

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        _apply_dotted(merged, key.strip(), _coerce(raw.strip()))

    return Config.model_validate(merged)
