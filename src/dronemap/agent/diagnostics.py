"""Self-Healing Photogrammetry Diagnostician & Auto-Tuning Agent."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .client import LLMClient

if TYPE_CHECKING:
    from ..workspace import RunWorkspace

logger = logging.getLogger("dronemap.agent.diagnostics")


@dataclass
class DiagnosticAction:
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str


@dataclass
class SelfHealingReport:
    run_id: str
    overall_health: str  # "HEALTHY" | "DEGRADED" | "CRITICAL"
    detected_bottlenecks: list[str]
    tuning_recommendations: list[DiagnosticAction]
    auto_repair_eligible: bool
    diagnostic_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_reconstruction_diagnostics(
    ws: "RunWorkspace",
    client: LLMClient | None = None,
) -> SelfHealingReport:
    """Analyze stage execution logs and metrics to pinpoint reconstruction issues and recommend parameter tuning."""
    client = client or LLMClient()
    manifest = ws.manifest
    stages = manifest.get("stages", {})

    bottlenecks: list[str] = []
    actions: list[DiagnosticAction] = []

    # 1. Inspect Stage 1: Frame decimation & blur
    frames = stages.get("frames", {})
    if frames.get("status") == "ok":
        fm = frames.get("metrics", {})
        n_dec = fm.get("n_frames_decoded", 0)
        n_kf = fm.get("n_keyframes", 0)
        if n_dec > 0 and n_kf < 10:
            bottlenecks.append(f"Excessive keyframe decimation: only {n_kf} of {n_dec} frames retained.")
            actions.append(DiagnosticAction(
                parameter="min_sharpness_percentile",
                current_value=0.25,
                recommended_value=0.15,
                reason="Capture has significant motion blur; relaxing sharpness threshold preserves stereo baseline."
            ))

    # 2. Inspect Stage 2: Dynamic object masks
    masks = stages.get("masks", {})
    if masks.get("status") == "ok":
        mm = masks.get("metrics", {})
        mean_mask = mm.get("mean_masked_fraction", 0.0)
        dropped = mm.get("n_dropped_frames", 0)
        if mean_mask > 0.35:
            bottlenecks.append(f"Heavy transient occlusion: {mean_mask * 100:.1f}% average area masked.")
        if dropped > 5:
            bottlenecks.append(f"{dropped} keyframes dropped due to extreme motion occlusion.")

    # 3. Inspect Stage 3: Camera pose & SfM
    pose = stages.get("pose", {})
    reg_frac = pose.get("metrics", {}).get("registered_fraction", 0.0) if pose.get("status") == "ok" else 0.0
    reproj = pose.get("metrics", {}).get("mean_reproj_error")

    if pose.get("status") == "failed":
        bottlenecks.append("Stage 3 Pose failed: COLMAP mapper could not initialize or register camera graph.")
        actions.append(DiagnosticAction(
            parameter="neural_matching",
            current_value=False,
            recommended_value=True,
            reason="Low visual texture; escalate to LightGlue neural feature matching."
        ))
        actions.append(DiagnosticAction(
            parameter="camera_model",
            current_value="SIMPLE_RADIAL",
            recommended_value="OPENCV",
            reason="Enable distortion parameters to allow convergence under non-calibrated drone optics."
        ))
    elif reg_frac < 0.80:
        bottlenecks.append(f"Incomplete camera registration: {reg_frac * 100:.1f}% keyframes registered (< 80% bar).")
        actions.append(DiagnosticAction(
            parameter="neural_matching",
            current_value=False,
            recommended_value=True,
            reason="Low inlier count across sharp maneuvers; LightGlue escalation restores lost camera links."
        ))
        actions.append(DiagnosticAction(
            parameter="matching_overlap",
            current_value=10,
            recommended_value=16,
            reason="Increase sequential neighborhood to bridge larger viewpoint baselines."
        ))

    if reproj is not None and reproj > 1.2:
        bottlenecks.append(f"High reprojection error: {reproj:.2f} px (> 1.0 px desired threshold).")
        actions.append(DiagnosticAction(
            parameter="bundle_adjustment_loss",
            current_value="Trivial",
            recommended_value="Cauchy",
            reason="Use robust Cauchy loss to suppress outlier tie-points in bundle adjustment."
        ))

    # Determine health
    if pose.get("status") == "failed" or reg_frac < 0.50:
        health = "CRITICAL"
    elif bottlenecks:
        health = "DEGRADED"
    else:
        health = "HEALTHY"

    summary = (
        f"Pipeline diagnostic completed for run {ws.run_id}. System status: {health}. "
        f"{len(bottlenecks)} bottleneck(s) identified, {len(actions)} parameter adaptation(s) suggested."
    )

    report = SelfHealingReport(
        run_id=ws.run_id,
        overall_health=health,
        detected_bottlenecks=bottlenecks,
        tuning_recommendations=actions,
        auto_repair_eligible=bool(actions),
        diagnostic_summary=summary,
    )

    # Save to logs
    try:
        log_dir = ws.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "diagnostics_report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not persist diagnostics report: %s", exc)

    return report
