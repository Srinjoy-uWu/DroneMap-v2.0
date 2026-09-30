"""Local SLM (Ollama / OpenAI-compatible localhost) client with deterministic offline fallback."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from .config import AgentConfig

logger = logging.getLogger("dronemap.agent")


class LLMClient:
    """Client for interacting with a local Small Language Model (Ollama / LM Studio / vLLM).

    Requires zero external cloud API keys. When no local SLM server is running on
    localhost, ``is_available`` returns False and ``generate()`` returns None so
    the calling subsystem immediately engages the deterministic offline rule engine.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    @property
    def is_available(self) -> bool:
        if self.config.active_provider == "offline":
            return False
        return self.config._is_local_endpoint_alive()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[Path] | None = None,
        json_mode: bool = False,
    ) -> str | None:
        """Send prompt to local SLM endpoint and return response text, or None to trigger offline engine."""
        if self.config.active_provider == "offline":
            return None

        try:
            return self._call_local(prompt, system_prompt, images, json_mode)
        except Exception as exc:
            logger.debug("Local SLM call failed (%s); engaging offline deterministic engine.", exc)
            return None

    def _call_local(
        self,
        prompt: str,
        system_prompt: str | None,
        images: list[Path] | None,
        json_mode: bool,
    ) -> str | None:
        import httpx

        url = f"{self.config.local_endpoint.rstrip('/')}/chat/completions"
        model = self.config.default_model or "llama3.2:1b"

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_path in images[:2]:
                if img_path.exists():
                    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        return None
