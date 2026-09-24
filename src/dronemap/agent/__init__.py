"""DroneMap Autonomous AI & LLM Booster Layer.

Provides multi-provider foundation model integration (Gemini, OpenAI, Anthropic),
tactical mission intelligence generation, self-healing SfM diagnostics,
and natural language 3D spatial query copilot for Web Measurement Studio.
"""

from .client import LLMClient
from .config import AgentConfig
from .diagnostics import DiagnosticAction, SelfHealingReport, analyze_reconstruction_diagnostics
from .intelligence import IntelligenceReport, generate_intelligence_report
from .tools import (
    calculate_3d_distance,
    explain_pipeline_stage,
    handle_spatial_chat,
    query_regional_confidence,
)

__all__ = [
    "AgentConfig",
    "DiagnosticAction",
    "IntelligenceReport",
    "LLMClient",
    "SelfHealingReport",
    "analyze_reconstruction_diagnostics",
    "calculate_3d_distance",
    "explain_pipeline_stage",
    "generate_intelligence_report",
    "handle_spatial_chat",
    "query_regional_confidence",
]
