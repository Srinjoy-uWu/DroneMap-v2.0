"""Capture-quality measurement and reconstruction verdicts.

Every other gate in this pipeline counts things: how many cameras registered,
how many points survived densification.  Counts cannot tell a building apart
from a flat carpet of road tarmac -- a hovering drone happily produces 130k
dense points describing a single tilted plane, and every count-based gate
waves it through.

This module measures *geometry* instead:

  * triangulation angle  -- can any point actually be intersected in 3D?
  * baseline / depth     -- did the platform translate enough to see parallax?
  * depth dynamic range  -- was more than one depth slab reconstructed?
  * forward degeneracy   -- is motion along the optical axis (zero disparity
                            at the epipole, the classic drone fly-forward trap)?
  * planarity            -- is the result a surface or a volume?

and turns those numbers into one of three honest verdicts:

  ACCEPT_3D    full 3D structure is supported by the imagery
  TERRAIN_2_5D only a ground surface is supported; export a 2.5D DSM mesh
               and say so, rather than pretending the sheet is buildings
  REJECT       the video cannot support any metric product

The point of the REJECT verdict is that it fires *before* twenty minutes of
MVS, and that it is recorded in the manifest so the UI and the report can name
the reason instead of showing a green tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Verdict",
    "SparseGeometry",
    "StructureGeometry",
    "QualityAssessment",
    "measure_sparse_geometry",
    "measure_structure",
    "assess_pose",
    "assess_structure",
]


class Verdict(str, Enum):
    """What the imagery actually supports.  Ordered worst to best."""

    REJECT = "reject"
    TERRAIN_2_5D = "terrain_2_5d"
    ACCEPT_3D = "accept_3d"

    @property
    def rank(self) -> int:
        return {"reject": 0, "terrain_2_5d": 1, "accept_3d": 2}[self.value]

    def worst_of(self, other: "Verdict") -> "Verdict":
        return self if self.rank <= other.rank else other


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


@dataclass
class SparseGeometry:
    """Geometric facts about a sparse SfM reconstruction.

    All distances are in reconstruction units (metres only once georeferenced);
    every gate below is deliberately expressed as a *ratio* or an *angle* so it
    is scale-free and works identically on relative and georeferenced models.
    """

    n_registered: int = 0
    n_total: int = 0
    n_points3d: int = 0

    # Camera motion
    path_length: float = 0.0
    max_baseline: float = 0.0
    # max_baseline / path_length.  ~1.0 for a clean straight pass; low values
    # mean the platform wandered or came back on itself, i.e. it burned flight
    # time without buying any new viewpoint.  The hover signature.
    path_efficiency: float = 1.0
    # Largest camera displacement projected onto the mean optical axis, over
    # the total displacement.  ~1.0 means the platform flew straight down its
    # own line of sight: the degenerate fly-forward case.
    forward_motion_ratio: float = 0.0

    # Scene depth, measured as distance from each camera to the points it sees
    depth_p5: float = 0.0
    depth_p50: float = 0.0
    depth_p95: float = 0.0
    depth_dynamic_range: float = 1.0  # p95 / p5

    # Parallax
    baseline_depth_ratio: float = 0.0  # max_baseline / depth_p50
    max_parallax_deg: float = 0.0  # implied by baseline_depth_ratio
    median_tri_angle_deg: float = 0.0  # measured per-point, the honest number
    p90_tri_angle_deg: float = 0.0

    # Track quality
    mean_track_length: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: _clean(v) for k, v in asdict(self).items()}


@dataclass
class StructureGeometry:
    """Shape facts about a dense cloud or mesh: surface, or volume?"""

    n_points: int = 0
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # PCA singular values normalised so the first is 1.0.  The third value is
    # the thickness of the cloud perpendicular to its best-fit plane; below a
    # few percent the "3D model" is a sheet.
    singular_ratios: tuple[float, float, float] = (1.0, 0.0, 0.0)
    planarity: float = 0.0  # == singular_ratios[2]
    # Relief measured off the best-fit ground plane, over the horizontal
    # footprint.  Buildings give a fat tail here; tarmac does not.
    relief_p95: float = 0.0
    relief_ratio: float = 0.0  # relief_p95 / horizontal_extent
    # Fraction of points more than `off_plane_threshold` above the ground
    # plane -- i.e. how much of the cloud is actually above-ground structure.
    above_ground_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: _clean(v) for k, v in asdict(self).items()}
        d["extent"] = [_clean(x) for x in self.extent]
        d["singular_ratios"] = [_clean(x) for x in self.singular_ratios]
        return d


@dataclass
class QualityAssessment:
    """A verdict plus the human-readable reasons behind it."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verdict is not Verdict.REJECT

    @property
    def headline(self) -> str:
        return {
            Verdict.ACCEPT_3D: "Full 3D reconstruction supported",
            Verdict.TERRAIN_2_5D: "Terrain surface only (2.5D) — insufficient parallax for 3D structure",
            Verdict.REJECT: "Capture rejected — video cannot support a metric 3D product",
        }[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "headline": self.headline,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "advice": list(self.advice),
            "metrics": self.metrics,
        }

    def merge(self, other: "QualityAssessment") -> "QualityAssessment":
        """Combine two assessments, keeping the worse verdict."""
        return QualityAssessment(
            verdict=self.verdict.worst_of(other.verdict),
            reasons=self.reasons + other.reasons,
            warnings=self.warnings + other.warnings,
            advice=_dedupe(self.advice + other.advice),
            metrics={**self.metrics, **other.metrics},
        )


def _clean(value: Any) -> Any:
    """JSON-safe: numpy scalars to python, NaN/inf to None."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------
# Sparse-model measurement
# --------------------------------------------------------------------------


def measure_sparse_geometry(recon: Any, n_total: int | None = None) -> SparseGeometry:
    """Measure the geometry of a pycolmap ``Reconstruction``.

    Tolerant of pycolmap API drift: every accessor is probed, and a missing one
    leaves its metric at the neutral default rather than raising.  A quality
    gate that crashes is worse than no gate at all.
    """
    geom = SparseGeometry()

    images = list(getattr(recon, "images", {}).values())
    points = getattr(recon, "points3D", {})
    geom.n_registered = sum(1 for im in images if _is_registered(im))
    geom.n_total = int(n_total) if n_total else len(images)
    geom.n_points3d = len(points)

    if geom.n_registered < 2 or geom.n_points3d < 3:
        return geom

    centres = _camera_centres(images)
    if len(centres) >= 2:
        geom.path_length = float(
            np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))
        )
        geom.max_baseline = float(_max_pairwise_distance(centres))
        if geom.path_length > 1e-9:
            geom.path_efficiency = float(geom.max_baseline / geom.path_length)

    axes = _optical_axes(images)
    if len(centres) >= 2 and len(axes) >= 1:
        geom.forward_motion_ratio = _forward_motion_ratio(centres, axes)

    xyz = _point_coords(points)
    if xyz.size == 0:
        return geom

    depths, per_image_ratios = _per_image_depths(images, points)
    if depths.size == 0:
        # Orientation unavailable: fall back to euclidean range from the
        # centroid of the camera stations.  Weak, but not wrong.
        depths = np.linalg.norm(xyz - centres.mean(axis=0), axis=1)

    if depths.size >= 8:
        p5, p50, p95 = (float(v) for v in np.percentile(depths, [5, 50, 95]))
        geom.depth_p5, geom.depth_p50, geom.depth_p95 = p5, p50, p95
        # Prefer the median *per-view* stratification: it answers "does any one
        # image see more than a single depth layer", which is the structure
        # question.  Pooling all views instead conflates flight extent with
        # scene depth and reports 1.0x for perfectly good captures.
        if per_image_ratios:
            geom.depth_dynamic_range = float(np.median(per_image_ratios))
        elif p5 > 1e-9:
            geom.depth_dynamic_range = float(p95 / p5)
        if p50 > 1e-9:
            geom.baseline_depth_ratio = float(geom.max_baseline / p50)
            # Triangulating a point at median depth from the two extreme
            # camera stations subtends this angle.  It is the *best case*
            # the capture geometry can offer.
            geom.max_parallax_deg = math.degrees(
                2.0 * math.atan2(geom.max_baseline / 2.0, p50)
            )

    tri = _triangulation_angles(points, images, centres)
    if tri.size:
        geom.median_tri_angle_deg = float(np.median(tri))
        geom.p90_tri_angle_deg = float(np.percentile(tri, 90))

    lengths = [
        len(pt.track.elements)
        for pt in points.values()
        if getattr(getattr(pt, "track", None), "elements", None) is not None
    ]
    if lengths:
        geom.mean_track_length = float(np.mean(lengths))

    return geom


def _is_registered(image: Any) -> bool:
    for attr in ("has_pose", "registered"):
        val = getattr(image, attr, None)
        if val is None:
            continue
        return bool(val() if callable(val) else val)
    return True


def _camera_centre(image: Any) -> np.ndarray | None:
    """Camera centre in world coordinates, across pycolmap versions."""
    for attr in ("projection_center", "center"):
        fn = getattr(image, attr, None)
        if fn is None:
            continue
        try:
            return np.asarray(fn() if callable(fn) else fn, dtype=float).reshape(3)
        except Exception:
            continue
    cfw = _rigid_cam_from_world(image)
    if cfw is not None:
        try:
            inv = cfw.inverse() if callable(getattr(cfw, "inverse", None)) else None
            if inv is not None:
                return np.asarray(inv.translation, dtype=float).reshape(3)
        except Exception:
            pass
    return None


def _camera_centres(images: Sequence[Any]) -> np.ndarray:
    out = [c for c in (_camera_centre(im) for im in images if _is_registered(im)) if c is not None]
    return np.asarray(out, dtype=float) if out else np.empty((0, 3))


def _rigid_cam_from_world(image: Any) -> Any | None:
    """The world->camera rigid transform, whatever shape pycolmap exposes it in.

    In pycolmap 4.x ``cam_from_world`` is a *method*; in earlier builds it is a
    property.  Reading it without probing yields a bound-method object whose
    ``.rotation`` is None, which silently zeroes every orientation metric.
    """
    raw = getattr(image, "cam_from_world", None)
    if raw is None:
        return None
    if callable(raw) and not hasattr(raw, "rotation"):
        try:
            return raw()
        except Exception:
            return None
    return raw


def _optical_axis(image: Any) -> np.ndarray | None:
    """World-frame viewing direction (+Z of the camera frame)."""
    cfw = _rigid_cam_from_world(image)
    if cfw is None:
        return None
    try:
        rot = getattr(cfw, "rotation", None)
        if rot is None:
            return None
        mat = rot.matrix() if callable(getattr(rot, "matrix", None)) else np.asarray(rot)
        mat = np.asarray(mat, dtype=float).reshape(3, 3)
        # R maps world into camera, so the camera's +Z in world is R^T @ [0,0,1]
        axis = mat.T @ np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(axis)
        return axis / norm if norm > 1e-9 else None
    except Exception:
        return None


def _optical_axes(images: Sequence[Any]) -> np.ndarray:
    out = [a for a in (_optical_axis(im) for im in images if _is_registered(im)) if a is not None]
    return np.asarray(out, dtype=float) if out else np.empty((0, 3))


def _point_coords(points: Any) -> np.ndarray:
    out = []
    for pt in points.values():
        xyz = getattr(pt, "xyz", None)
        if xyz is None:
            continue
        try:
            out.append(np.asarray(xyz, dtype=float).reshape(3))
        except Exception:
            continue
    return np.asarray(out, dtype=float) if out else np.empty((0, 3))


def _max_pairwise_distance(centres: np.ndarray) -> float:
    """Exact for small camera counts, subsampled above that."""
    pts = centres
    if len(pts) > 400:
        idx = np.linspace(0, len(pts) - 1, 400).astype(int)
        pts = pts[idx]
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.max(np.linalg.norm(diff, axis=2)))


def _forward_motion_ratio(centres: np.ndarray, axes: np.ndarray) -> float:
    """How much of the camera travel was along the line of sight.

    1.0 = flew straight ahead down the optical axis (degenerate: points near
    the epipole never move, so depth is unobservable there).
    0.0 = pure sideways/orbital motion (ideal for triangulation).

    The dominant travel direction is taken from a PCA of the camera stations,
    not from first-minus-last: on a hover or an out-and-back pass the endpoints
    coincide and the naive difference vanishes, silently reporting the ideal
    value for the worst possible capture.
    """
    if len(centres) < 2 or axes.size == 0:
        return 0.0
    centred = centres - centres.mean(axis=0)
    spread = float(np.linalg.norm(centred))
    if spread < 1e-9:
        return 0.0
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    travel_dir = vh[0]

    mean_axis = axes.mean(axis=0)
    norm = float(np.linalg.norm(mean_axis))
    if norm < 1e-9:
        return 0.0
    return float(abs(np.dot(travel_dir, mean_axis / norm)))


def _per_image_depths(
    images: Sequence[Any], points: Any
) -> tuple[np.ndarray, list[float]]:
    """Depths of observed points, measured per image along its own axis.

    Measuring from a single averaged camera station along an averaged optical
    axis is meaningless for aerial capture: on any curved or multi-directional
    path the mean axis nearly cancels, and every scene reports one flat depth
    slab.  What actually matters is whether an *individual view* sees
    stratified depth -- ground at one range, rooftops at another.

    Returns (all pooled depths, per-image p95/p5 ratios).
    """
    pooled: list[np.ndarray] = []
    ratios: list[float] = []

    xyz_by_id: dict[int, np.ndarray] = {}
    for pid, pt in points.items():
        try:
            xyz_by_id[int(pid)] = np.asarray(pt.xyz, dtype=float).reshape(3)
        except Exception:
            continue
    if not xyz_by_id:
        return np.empty(0), ratios

    for im in images:
        if not _is_registered(im):
            continue
        centre = _camera_centre(im)
        axis = _optical_axis(im)
        if centre is None or axis is None:
            continue

        seen: list[np.ndarray] = []
        for p2d in getattr(im, "points2D", []) or []:
            if not getattr(p2d, "has_point3D", lambda: False)():
                continue
            xyz = xyz_by_id.get(int(p2d.point3D_id))
            if xyz is not None:
                seen.append(xyz)
        if len(seen) < 16:
            continue

        depths = (np.asarray(seen) - centre) @ axis
        depths = depths[depths > 1e-6]
        if depths.size < 16:
            continue
        pooled.append(depths)
        p5, p95 = np.percentile(depths, [5, 95])
        if p5 > 1e-9:
            ratios.append(float(p95 / p5))

    if not pooled:
        return np.empty(0), ratios
    return np.concatenate(pooled), ratios


def _triangulation_angles(
    points: Any, images: Sequence[Any], centres: np.ndarray, max_points: int = 5000
) -> np.ndarray:
    """Per-point max triangulation angle, in degrees.

    This is the number COLMAP itself uses to decide whether a point is
    trustworthy, and the one metric that separates "the drone moved" from
    "the drone hovered".
    """
    by_id: dict[int, np.ndarray] = {}
    for im in images:
        if not _is_registered(im):
            continue
        centre = _camera_centre(im)
        if centre is None:
            continue
        img_id = getattr(im, "image_id", None)
        if img_id is None:
            continue
        by_id[int(img_id)] = centre
    if len(by_id) < 2:
        return np.empty(0)

    items = list(points.values())
    if len(items) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(items), max_points, replace=False)
        items = [items[i] for i in idx]

    angles: list[float] = []
    for pt in items:
        track = getattr(pt, "track", None)
        elements = getattr(track, "elements", None)
        if not elements or len(elements) < 2:
            continue
        obs = []
        for el in elements:
            img_id = getattr(el, "image_id", None)
            if img_id is None:
                continue
            centre = by_id.get(int(img_id))
            if centre is not None:
                obs.append(centre)
        if len(obs) < 2:
            continue
        try:
            xyz = np.asarray(pt.xyz, dtype=float).reshape(3)
        except Exception:
            continue
        rays = np.asarray(obs) - xyz
        norms = np.linalg.norm(rays, axis=1)
        keep = norms > 1e-9
        if keep.sum() < 2:
            continue
        rays = rays[keep] / norms[keep, None]
        # Widest angle between any pair of viewing rays to this point
        cos = np.clip(rays @ rays.T, -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(np.min(cos)))))

    return np.asarray(angles) if angles else np.empty(0)


# --------------------------------------------------------------------------
# Structure measurement
# --------------------------------------------------------------------------


def measure_structure(
    points: np.ndarray, off_plane_threshold: float | None = None
) -> StructureGeometry:
    """Measure whether a point set is a surface or a volume.

    ``points`` is an (N, 3) array from the dense cloud or mesh vertices, in any
    coordinate frame -- the metrics are rotation invariant, which matters
    because the pipeline's up-axis is not consistent across stages.
    """
    geom = StructureGeometry()
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 16:
        return geom

    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < 16:
        return geom
    if len(finite) > 200000:
        idx = np.random.default_rng(0).choice(len(finite), 200000, replace=False)
        finite = finite[idx]

    geom.n_points = int(len(finite))
    geom.extent = tuple(float(v) for v in (finite.max(axis=0) - finite.min(axis=0)))

    centred = finite - finite.mean(axis=0)
    try:
        sv = np.linalg.svd(centred, compute_uv=False)
    except np.linalg.LinAlgError:
        return geom
    if sv[0] < 1e-12:
        return geom
    ratios = sv / sv[0]
    geom.singular_ratios = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    geom.planarity = float(ratios[2])

    # Relief above the best-fit plane.  The plane normal is the third right
    # singular vector; signed distance along it is the local height field.
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[2]
    except np.linalg.LinAlgError:
        return geom
    height = centred @ normal
    # Orient so structure is positive: most of a scene is ground, so the
    # ground plane sits near the *median*, and buildings form the upper tail.
    if abs(float(np.percentile(height, 95) - np.median(height))) < abs(
        float(np.median(height) - np.percentile(height, 5))
    ):
        height = -height
    relief = height - float(np.median(height))
    geom.relief_p95 = float(np.percentile(relief, 95))

    # Footprint measured along the dominant in-plane axis, so the ratio below
    # stays rotation invariant regardless of which axis the stage calls "up".
    horizontal = max(geom.extent)
    footprint = float(np.percentile(np.abs(centred @ vh[0]), 95) * 2.0)
    if footprint > 1e-9:
        geom.relief_ratio = float(geom.relief_p95 / footprint)
    elif horizontal > 1e-9:
        geom.relief_ratio = float(geom.relief_p95 / horizontal)

    threshold = (
        off_plane_threshold
        if off_plane_threshold is not None
        else max(0.05 * max(footprint, 1e-9), 1e-6)
    )
    geom.above_ground_fraction = float(np.mean(relief > threshold))
    return geom


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def assess_pose(geom: SparseGeometry, cfg: Any) -> QualityAssessment:
    """Decide, from camera geometry alone, what this capture can support.

    Runs immediately after SfM -- before twenty minutes of MVS -- so a hover
    or a straight fly-through is named as such while the user is still
    watching the progress bar.

    ``cfg`` is a ``QualityConfig``; passed loosely so this module stays
    importable without the config package.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    advice: list[str] = []
    verdict = Verdict.ACCEPT_3D

    def demote(to: Verdict, why: str, tip: str | None = None) -> None:
        nonlocal verdict
        verdict = verdict.worst_of(to)
        reasons.append(why)
        if tip:
            advice.append(tip)

    # --- hard rejects: no usable reconstruction at all -------------------
    if geom.n_registered < cfg.min_registered_images:
        demote(
            Verdict.REJECT,
            f"Only {geom.n_registered} camera pose(s) recovered; "
            f"at least {cfg.min_registered_images} are required.",
            "Fly a continuous pass with 70-80% forward overlap and avoid "
            "motion blur; re-shoot at a lower ground speed.",
        )
        return QualityAssessment(verdict, reasons, warnings, advice, geom.to_dict())

    if geom.n_total and geom.n_registered / geom.n_total < cfg.min_registered_fraction:
        frac = geom.n_registered / geom.n_total
        demote(
            Verdict.REJECT,
            f"Only {frac:.0%} of keyframes registered "
            f"(minimum {cfg.min_registered_fraction:.0%}); the flight breaks "
            "into disconnected segments.",
            "Keep the camera pointed consistently and avoid yaw spins mid-pass.",
        )

    if geom.n_points3d < cfg.min_sparse_points:
        demote(
            Verdict.REJECT,
            f"Only {geom.n_points3d} sparse 3D points triangulated "
            f"(minimum {cfg.min_sparse_points}).",
            "Ensure the ground has visible texture; featureless surfaces "
            "(water, fresh tarmac, sand) cannot be matched.",
        )

    # --- numerical degeneracy -------------------------------------------
    # A collapsed reconstruction puts the scene essentially at the camera
    # centres, so scene depth underflows and every *ratio* gate that looks for
    # a minimum sails through: baseline/depth becomes astronomically large
    # rather than small.  Guard the upper end explicitly, or a degenerate
    # model reads as the best capture in the dataset.
    if geom.depth_p50 > 0 and geom.max_baseline > 0:
        if geom.baseline_depth_ratio > cfg.max_baseline_depth_ratio:
            demote(
                Verdict.REJECT,
                f"Reconstruction is numerically degenerate: the flight baseline "
                f"is {geom.baseline_depth_ratio:.3g}x the scene depth, which is "
                "physically impossible for aerial imagery. The solution has "
                "collapsed onto the camera centres.",
                "Provide known camera intrinsics or telemetry to constrain the "
                "solution, and ensure the footage is not a static shot.",
            )
        elif geom.depth_p50 < cfg.min_depth_over_baseline * geom.max_baseline:
            demote(
                Verdict.REJECT,
                f"Reconstruction is numerically degenerate: median scene depth "
                f"({geom.depth_p50:.3g}) is negligible against the camera "
                f"baseline ({geom.max_baseline:.3g}).",
            )

    # --- parallax: the hover test ---------------------------------------
    if geom.median_tri_angle_deg and geom.median_tri_angle_deg < cfg.reject_tri_angle_deg:
        demote(
            Verdict.REJECT,
            f"Median triangulation angle is {geom.median_tri_angle_deg:.1f}deg "
            f"(minimum {cfg.reject_tri_angle_deg:.1f}deg). The platform barely "
            "translated relative to scene distance, so depth is not observable.",
            "Fly a continuous lateral or orbital pass; a hover or a very short "
            "clip cannot produce 3D structure at any processing setting.",
        )
    elif geom.median_tri_angle_deg and geom.median_tri_angle_deg < cfg.min_tri_angle_deg:
        demote(
            Verdict.TERRAIN_2_5D,
            f"Median triangulation angle is only "
            f"{geom.median_tri_angle_deg:.1f}deg (want "
            f">={cfg.min_tri_angle_deg:.1f}deg for 3D structure); a ground "
            "surface can be estimated but vertical structure cannot.",
            "Increase the flight baseline, or lower the altitude so the same "
            "travel yields more parallax.",
        )

    if geom.baseline_depth_ratio and geom.baseline_depth_ratio < cfg.reject_baseline_depth_ratio:
        demote(
            Verdict.REJECT,
            f"Flight baseline is only {geom.baseline_depth_ratio:.3f}x the "
            f"scene depth (minimum {cfg.reject_baseline_depth_ratio:.2f}x).",
            "Either fly further along the pass or descend; scene distance and "
            "travel distance must be comparable.",
        )
    elif geom.baseline_depth_ratio and geom.baseline_depth_ratio < cfg.min_baseline_depth_ratio:
        demote(
            Verdict.TERRAIN_2_5D,
            f"Flight baseline is {geom.baseline_depth_ratio:.3f}x the scene "
            f"depth (want >={cfg.min_baseline_depth_ratio:.2f}x for reliable 3D).",
        )

    # --- depth stratification: is there anything but one slab? ----------
    if (
        geom.depth_dynamic_range
        and geom.depth_dynamic_range < cfg.min_depth_dynamic_range
    ):
        demote(
            Verdict.TERRAIN_2_5D,
            f"Reconstructed depth spans only "
            f"{geom.depth_dynamic_range:.2f}x (p95/p5); the scene resolved as a "
            "single depth layer, which is a ground surface, not buildings.",
            "For structures use a 65-75deg gimbal so facades and ground occupy "
            "clearly different depths.",
        )

    # --- forward-motion degeneracy --------------------------------------
    if geom.forward_motion_ratio > cfg.max_forward_motion_ratio:
        warnings.append(
            f"{geom.forward_motion_ratio:.0%} of the camera travel was along "
            "the viewing direction. Points near the image centre have almost "
            "no disparity in this geometry, so the middle of the model is the "
            "least reliable part."
        )
        advice.append(
            "Angle the gimbal 65-75deg (obliquely forward-down) rather than "
            "straight along the flight direction."
        )

    if geom.mean_track_length and geom.mean_track_length < cfg.min_mean_track_length:
        warnings.append(
            f"Features are seen in only {geom.mean_track_length:.1f} images on "
            "average; overlap is marginal."
        )
        advice.append("Increase forward overlap to 70-80% (slower flight or higher frame rate).")

    if geom.path_efficiency < cfg.min_path_efficiency:
        # Path efficiency is ambiguous in exactly the way planarity is, and the
        # disambiguator is again the parallax evidence. A hover and an orbit both
        # score low - the platform ends up near where it started - but they are
        # opposite captures: an orbit around a subject is the *best* geometry for
        # 3D structure, because every facade gets observed from a wide arc.
        # Judge them by whether the parallax actually materialised.
        orbital = (
            geom.median_tri_angle_deg >= cfg.min_tri_angle_deg
            and geom.baseline_depth_ratio >= cfg.min_baseline_depth_ratio
        )
        if orbital:
            reasons.append(
                f"Flight path is closed rather than linear ({geom.path_efficiency:.0%} "
                f"efficiency) but parallax is strong ({geom.median_tri_angle_deg:.1f}deg "
                f"triangulation, {geom.baseline_depth_ratio:.2f}x baseline/depth) - "
                "an orbit around the subject, which is good geometry for structure."
            )
        else:
            warnings.append(
                f"The platform covered {geom.path_length:.1f} units of flight path "
                f"but its two most-separated stations are only "
                f"{geom.max_baseline:.1f} apart ({geom.path_efficiency:.0%} "
                f"efficiency), and the parallax to show for it is weak "
                f"({geom.median_tri_angle_deg:.1f}deg triangulation) - it hovered "
                "or retraced its own line instead of building a baseline."
            )
            advice.append(
                "Either fly one continuous pass in a single direction, or orbit "
                "the subject at a wide radius - but do not hover or retrace the "
                "same line."
            )

    return QualityAssessment(verdict, reasons, warnings, advice, geom.to_dict())


def assess_structure(
    geom: StructureGeometry, cfg: Any, sparse: SparseGeometry | None = None
) -> QualityAssessment:
    """Decide, from the reconstructed shape, whether it is a 3D model.

    This is the gate that catches the flat-carpet failure: a hovering capture
    can pass every count-based check and still yield a mesh whose third
    principal axis is 4% of its first.

    Crucially, flatness on its own is ambiguous.  A beach survey flown with
    excellent parallax also produces a near-planar cloud -- because a beach is
    flat.  Rejecting that would be as wrong as accepting the carpet.  The
    disambiguator is the capture geometry: pass ``sparse`` and a flat result
    backed by adequate parallax becomes a legitimate 2.5D terrain product,
    while a flat result from marginal parallax is a reconstruction failure.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    advice: list[str] = []
    verdict = Verdict.ACCEPT_3D

    if geom.n_points < cfg.min_structure_points:
        return QualityAssessment(
            Verdict.REJECT,
            [
                f"Only {geom.n_points} points in the reconstructed surface "
                f"(minimum {cfg.min_structure_points})."
            ],
            warnings,
            ["Re-shoot with more overlap and visible ground texture."],
            geom.to_dict(),
        )

    # Was the capture geometry good enough that a flat answer can be believed?
    parallax_trustworthy = sparse is not None and (
        sparse.median_tri_angle_deg >= cfg.min_tri_angle_deg
        and sparse.baseline_depth_ratio >= cfg.min_baseline_depth_ratio
    )

    if geom.planarity < cfg.reject_planarity:
        if parallax_trustworthy:
            verdict = Verdict.TERRAIN_2_5D
            reasons.append(
                f"The scene is genuinely flat: surface thickness is "
                f"{geom.planarity:.1%} of its width, measured from a capture "
                f"with adequate parallax ({sparse.median_tri_angle_deg:.1f}deg "
                f"triangulation, {sparse.baseline_depth_ratio:.2f}x "
                "baseline/depth). Exported as a 2.5D terrain product, which is "
                "the correct product for this site."
            )
        else:
            verdict = Verdict.REJECT
            detail = ""
            if sparse is not None:
                detail = (
                    f" Capture geometry was marginal "
                    f"({sparse.median_tri_angle_deg:.1f}deg triangulation, "
                    f"{sparse.baseline_depth_ratio:.2f}x baseline/depth), so "
                    "the flatness is a reconstruction failure, not a flat site."
                )
            reasons.append(
                f"The reconstruction is a flat sheet: its thickness is "
                f"{geom.planarity:.1%} of its width (minimum "
                f"{cfg.reject_planarity:.1%}). No 3D structure was "
                f"recovered.{detail}"
            )
            advice.append(
                "This capture geometry cannot yield buildings. Fly a lateral or "
                "orbital pass at 65-75deg gimbal with 70-80% overlap."
            )
    elif geom.planarity < cfg.terrain_planarity:
        verdict = Verdict.TERRAIN_2_5D
        reasons.append(
            f"Surface thickness is {geom.planarity:.1%} of its width - this is "
            "a terrain surface, not volumetric structure. Exported as a 2.5D "
            "terrain product."
        )

    if verdict is Verdict.ACCEPT_3D and geom.relief_ratio < cfg.terrain_relief_ratio:
        verdict = Verdict.TERRAIN_2_5D
        reasons.append(
            f"Vertical relief is {geom.relief_ratio:.1%} of the horizontal "
            "footprint; too low to contain resolvable buildings."
        )

    if geom.above_ground_fraction < cfg.min_above_ground_fraction:
        warnings.append(
            f"Only {geom.above_ground_fraction:.1%} of points sit above the "
            "ground plane; above-ground detail is sparse."
        )

    return QualityAssessment(verdict, reasons, warnings, advice, geom.to_dict())
