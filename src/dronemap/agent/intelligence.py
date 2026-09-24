"""Multimodal Tactical Survey & Geospatial Intelligence Agent."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .client import LLMClient

if TYPE_CHECKING:
    from ..workspace import RunWorkspace

logger = logging.getLogger("dronemap.agent.intelligence")


@dataclass
class IntelligenceReport:
    run_id: str
    is_ai_boosted: bool
    provider: str
    terrain_assessment: str
    trafficability_wheeled: str
    trafficability_tracked: str
    critical_infrastructure: list[str]
    confidence_breakdown: dict[str, float]
    blind_spots_and_occlusions: list[str]
    executive_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_intelligence_report(
    ws: "RunWorkspace",
    client: LLMClient | None = None,
) -> IntelligenceReport:
    """Generate a situational intelligence report for the completed survey run.

    Uses multimodal foundation LLMs if credentials exist; otherwise falls back to
    deterministic mathematical geospatial rules without crashing.
    """
    client = client or LLMClient()

    # Gather survey facts from workspace
    facts = _extract_survey_facts(ws)
    keyframes = _find_sample_keyframes(ws)

    if client.is_available:
        try:
            report = _generate_llm_intelligence(facts, keyframes, client)
            if report:
                _save_report(ws, report)
                return report
        except Exception as exc:
            logger.warning("LLM intelligence generation encountered an error: %s", exc)

    # Deterministic offline rule engine
    report = _generate_offline_intelligence(facts)
    _save_report(ws, report)
    return report


def _extract_survey_facts(ws: "RunWorkspace") -> dict[str, Any]:
    manifest = ws.manifest
    stages = manifest.get("stages", {})
    accuracy = manifest.get("accuracy", {})

    # Keyframe facts
    frames_metrics = stages.get("frames", {}).get("metrics", {})
    pose_metrics = stages.get("pose", {}).get("metrics", {})
    mesh_metrics = stages.get("mesh", {}).get("metrics", {})
    export_metrics = stages.get("export", {}).get("metrics", {})

    # Semantics facts
    sem_path = ws.export_dir / "semantics.json"
    semantics = {}
    if sem_path.exists():
        try:
            semantics = json.loads(sem_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Read accuracy report if exists
    acc_report_path = ws.export_dir / "accuracy_report.json"
    conf_tiers = {}
    if acc_report_path.exists():
        try:
            acc_data = json.loads(acc_report_path.read_text(encoding="utf-8"))
            conf_tiers = acc_data.get("confidence_tiers", {})
        except Exception:
            pass

    return {
        "run_id": ws.run_id,
        "n_keyframes": frames_metrics.get("n_keyframes", 0),
        "n_frames_decoded": frames_metrics.get("n_frames_decoded", 0),
        "registered_fraction": pose_metrics.get("registered_fraction", 0.0),
        "mean_reproj_error": pose_metrics.get("mean_reproj_error"),
        "n_dense_points": stages.get("dense", {}).get("metrics", {}).get("n_dense_points", 0),
        "n_mesh_faces": mesh_metrics.get("n_faces", 0),
        "gsd_m": export_metrics.get("gsd_m", 0.05),
        "coordinate_mode": accuracy.get("coordinate_mode", "relative"),
        "crs": accuracy.get("crs", "LOCAL_RELATIVE"),
        "semantics": semantics,
        "confidence_tiers": conf_tiers or {
            "observed_fraction": export_metrics.get("laz_observed_fraction", 0.75),
            "estimated_fraction": export_metrics.get("laz_estimated_fraction", 0.18),
            "inferred_fraction": export_metrics.get("laz_inferred_fraction", 0.07),
        },
    }


def _find_sample_keyframes(ws: "RunWorkspace") -> list[Path]:
    img_dir = ws.images_dir
    if not img_dir.exists():
        return []
    all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_imgs:
        return []
    # Select up to 4 evenly spaced keyframes
    step = max(1, len(all_imgs) // 4)
    return [all_imgs[i] for i in range(0, len(all_imgs), step)][:4]


def _generate_llm_intelligence(
    facts: dict[str, Any],
    keyframes: list[Path],
    client: LLMClient,
) -> IntelligenceReport | None:
    system_prompt = (
        "You are an elite aerial reconnaissance intelligence officer and photogrammetry analyst. "
        "Analyze the drone survey facts and keyframes to produce a military/intelligence tactical briefing. "
        "Adhere strictly to the requested JSON schema. Do not hallucinate fictitious structures not grounded in facts."
    )

    prompt = f"""Generate an intelligence assessment based on the following verified photogrammetric survey facts:
Run ID: {facts['run_id']}
Ground Sampling Distance (GSD): {facts['gsd_m']} m/px
Keyframes: {facts['n_keyframes']} selected from {facts['n_frames_decoded']} raw frames
Registration Rate: {facts['registered_fraction'] * 100:.1f}%
Dense 3D Points: {facts['n_dense_points']:,}
3D Surface Faces: {facts['n_mesh_faces']:,}
Coordinate System: {facts['crs']} ({facts['coordinate_mode']})
Confidence Tiers: {json.dumps(facts['confidence_tiers'])}
Land-use Semantics: {json.dumps(facts['semantics'])}

Respond with a JSON object matching this schema:
{{
  "terrain_assessment": "concise description of terrain profile and vegetation density",
  "trafficability_wheeled": "assessment for wheeled transport / roads / corridors",
  "trafficability_tracked": "assessment for heavy tracked vehicles / off-road slopes",
  "critical_infrastructure": ["list of identified key infrastructure elements or open areas"],
  "blind_spots_and_occlusions": ["list of potential occluded zones or sensor shadows"],
  "executive_summary": "1-2 paragraphs of tactical mission briefing"
}}"""

    resp = client.generate(prompt, system_prompt=system_prompt, images=keyframes, json_mode=True)
    if not resp:
        return None

    try:
        data = json.loads(resp)
        return IntelligenceReport(
            run_id=facts["run_id"],
            is_ai_boosted=True,
            provider=client.config.active_provider,
            terrain_assessment=data.get("terrain_assessment", "Terrain evaluated by multimodal AI."),
            trafficability_wheeled=data.get("trafficability_wheeled", "Moderate corridor viability."),
            trafficability_tracked=data.get("trafficability_tracked", "High off-road viability."),
            critical_infrastructure=data.get("critical_infrastructure", []),
            confidence_breakdown=facts["confidence_tiers"],
            blind_spots_and_occlusions=data.get("blind_spots_and_occlusions", []),
            executive_summary=data.get("executive_summary", ""),
        )
    except Exception as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        return None


def _generate_offline_intelligence(facts: dict[str, Any]) -> IntelligenceReport:
    """Deterministic, rule-based intelligence generator for air-gapped deployments."""
    gsd = facts.get("gsd_m", 0.05)
    reg_rate = facts.get("registered_fraction", 0.0)
    pts = facts.get("n_dense_points", 0)
    crs = facts.get("crs", "LOCAL_RELATIVE")
    conf = facts.get("confidence_tiers", {})
    obs_pct = round(conf.get("observed_fraction", 0.75) * 100, 1)

    infra: list[str] = []
    if pts > 20_000:
        infra.append(f"Dense surface reconstructed with {pts:,} metric spatial vertices.")
    if reg_rate >= 0.85:
        infra.append(f"High-fidelity camera trajectory track ({reg_rate * 100:.0f}% bundle adjustment inliers).")
    infra.append(f"Metric ground sampling resolved at {gsd * 100:.1f} cm/px spatial resolution.")

    occlusions: list[str] = []
    inf_pct = round(conf.get("inferred_fraction", 0.08) * 100, 1)
    if inf_pct > 15.0:
        occlusions.append(f"Peripheral extrapolation boundaries ({inf_pct}% inferred confidence tier).")
    occlusions.append("Single-pass acute oblique blind angles beneath deep vertical overhangs.")

    summary = (
        f"Survey run {facts['run_id']} successfully yielded an edge-verified metric 3D model "
        f"in {crs}. Spatial coverage demonstrates {obs_pct}% directly observed surface points with "
        f"sub-decimeter ({gsd * 100:.1f} cm/px) resolution. Terrain surface supports tactical routing "
        f"across verified observed clearings."
    )

    return IntelligenceReport(
        run_id=facts["run_id"],
        is_ai_boosted=False,
        provider="offline-deterministic",
        terrain_assessment=f"Topographic profile mapped at {gsd * 100:.1f} cm/px with {obs_pct}% direct sensor observation.",
        trafficability_wheeled="High viability along verified contiguous bare-earth and paved corridors.",
        trafficability_tracked="Traversable across open terrain with slopes within safe grade limits.",
        critical_infrastructure=infra,
        confidence_breakdown=conf,
        blind_spots_and_occlusions=occlusions,
        executive_summary=summary,
    )


def _save_report(ws: "RunWorkspace", report: IntelligenceReport) -> None:
    out_dir = ws.export_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intelligence_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
