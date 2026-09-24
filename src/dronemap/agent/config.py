"""Agent configuration and credentials management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass


@dataclass
class AgentConfig:
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    preferred_provider: str = field(
        default_factory=lambda: os.environ.get("DRONEMAP_LLM_PROVIDER", "auto").lower()
    )
    model_name: str | None = field(default_factory=lambda: os.environ.get("DRONEMAP_LLM_MODEL"))
    timeout_seconds: float = 30.0
    temperature: float = 0.2

    @property
    def has_api_keys(self) -> bool:
        """Return True if at least one LLM API key is present."""
        return bool(self.gemini_api_key or self.openai_api_key or self.anthropic_api_key)

    @property
    def active_provider(self) -> str:
        """Determine which provider to use based on available keys and preference."""
        if not self.has_api_keys or self.preferred_provider == "offline":
            return "offline"

        if self.preferred_provider in ("gemini", "google") and self.gemini_api_key:
            return "gemini"
        if self.preferred_provider == "openai" and self.openai_api_key:
            return "openai"
        if self.preferred_provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"

        # Auto-detection priority: Gemini -> OpenAI -> Anthropic
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return "offline"

    @property
    def default_model(self) -> str:
        if self.model_name:
            return self.model_name
        provider = self.active_provider
        if provider == "gemini":
            return "gemini-2.5-flash"
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
        return "offline-heuristic"
