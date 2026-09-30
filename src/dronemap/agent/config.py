"""Agent configuration for Local Ollama SLM and Offline Deterministic Engine."""

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
    """Configuration for DroneMap's 100% local/offline Intelligence & Copilot layer.

    Operates with zero external cloud API keys:
    - ``local``: Uses a local OpenAI/Ollama-compatible SLM server on localhost:11434
      (automatically launching the built-in Ollama SLM daemon if native ollama.exe is not running)
    - ``offline``: Uses the deterministic photogrammetric rule engine
    """

    local_endpoint: str = field(
        default_factory=lambda: os.environ.get("DRONEMAP_LOCAL_LLM_URL", "http://localhost:11434/v1")
    )
    preferred_provider: str = field(
        default_factory=lambda: os.environ.get("DRONEMAP_LLM_PROVIDER", "auto").lower()
    )
    model_name: str | None = field(default_factory=lambda: os.environ.get("DRONEMAP_LOCAL_MODEL"))
    timeout_seconds: float = 15.0
    temperature: float = 0.2
    auto_start_daemon: bool = True

    def _is_local_endpoint_alive(self) -> bool:
        """Fast socket probe to check if a local SLM server (Ollama) is listening on localhost:11434."""
        import socket
        from urllib.parse import urlparse
        try:
            parsed = urlparse(self.local_endpoint)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.25)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
            if self.auto_start_daemon and host in ("127.0.0.1", "localhost") and port == 11434:
                from .ollama_daemon import ensure_ollama_daemon
                return ensure_ollama_daemon(port=11434)
            return False
        except Exception:
            return False

    @property
    def active_provider(self) -> str:
        """Resolve active local/offline provider (`local` or `offline`)."""
        pref = self.preferred_provider
        if pref in ("offline", "none", "heuristic", "deterministic"):
            return "offline"
        if pref in ("local", "ollama", "slm"):
            return "local"

        if self._is_local_endpoint_alive():
            return "local"
        return "offline"

    @property
    def default_model(self) -> str:
        if self.active_provider == "offline":
            return "offline-deterministic"
        if self.model_name:
            return self.model_name
        return os.environ.get("DRONEMAP_LOCAL_MODEL", "llama3.2:1b")
