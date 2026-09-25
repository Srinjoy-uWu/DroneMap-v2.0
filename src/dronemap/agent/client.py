"""Resilient multi-provider LLM API client with offline fallback."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from .config import AgentConfig

logger = logging.getLogger("dronemap.agent")


class LLMClient:
    """Client for interacting with LLM providers (Gemini, OpenAI, Anthropic).

    If API keys are missing or calls fail (due to rate limits, network outage,
    invalid tokens), all calls gracefully degrade and return None so the calling
    subsystem can immediately engage deterministic offline heuristics.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    @property
    def is_available(self) -> bool:
        return self.config.active_provider != "offline"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[Path] | None = None,
        json_mode: bool = False,
    ) -> str | None:
        """Send prompt to active provider and return response text, or None on failure."""
        provider = self.config.active_provider
        if provider == "offline":
            logger.debug("LLMClient is in offline mode; skipping API dispatch.")
            return None

        try:
            if provider == "gemini":
                return self._call_gemini(prompt, system_prompt, images, json_mode)
            elif provider == "openai":
                return self._call_openai(prompt, system_prompt, images, json_mode)
            elif provider == "anthropic":
                return self._call_anthropic(prompt, system_prompt, images, json_mode)
            else:
                return None
        except Exception as exc:
            logger.warning("LLM API call to %s failed (%s); degrading to offline engine.", provider, exc)
            return None

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str | None,
        images: list[Path] | None,
        json_mode: bool,
    ) -> str | None:
        import httpx

        api_key = self.config.gemini_api_key
        primary_model = self.config.default_model or "gemini-flash-lite-latest"
        candidate_models = [primary_model]
        for fallback in ("gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-flash-latest"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        parts: list[dict[str, Any]] = []

        # Attach multimodal image frames if provided
        if images:
            for img_path in images[:4]:  # limit to top 4 frames
                if img_path.exists():
                    data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    parts.append({"inline_data": {"mime_type": mime, "data": data}})

        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.config.temperature,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if not content_parts:
                            continue
                        return content_parts[0].get("text", "")
                    elif resp.status_code in (429, 503):
                        logger.warning(
                            "Gemini model %s returned HTTP %d (temporary load spike); cascading to fallback candidate.",
                            model,
                            resp.status_code,
                        )
                        continue
                    else:
                        logger.warning("Gemini API error %d on %s: %s", resp.status_code, model, resp.text[:200])
                        if resp.status_code in (400, 401, 403):
                            return None
            except Exception as exc:
                logger.warning("Exception querying Gemini model %s: %s", model, exc)
                continue

        return None

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        images: list[Path] | None,
        json_mode: bool,
    ) -> str | None:
        import httpx

        api_key = self.config.openai_api_key
        model = self.config.default_model or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, Any]] | str
        if images:
            content_list: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_path in images[:4]:
                if img_path.exists():
                    data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    })
            user_content = content_list
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.warning("OpenAI API error %d: %s", resp.status_code, resp.text[:200])
                return None
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("content", "")

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        images: list[Path] | None,
        json_mode: bool,
    ) -> str | None:
        import httpx

        api_key = self.config.anthropic_api_key
        model = self.config.default_model or "claude-3-5-sonnet-20241022"
        url = "https://api.anthropic.com/v1/messages"

        user_content: list[dict[str, Any]] = []
        if images:
            for img_path in images[:4]:
                if img_path.exists():
                    data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": data,
                        },
                    })
        user_content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.config.temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.warning("Anthropic API error %d: %s", resp.status_code, resp.text[:200])
                return None
            data = resp.json()
            content = data.get("content", [])
            if not content:
                return None
            return content[0].get("text", "")
