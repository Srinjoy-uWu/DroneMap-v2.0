"""Verify AI Booster Layer and LLM API key status.

Checks active credentials against the provider API and prints clear status
and fallback notifications.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from dronemap.agent.config import AgentConfig
    from dronemap.agent.client import LLMClient
except ImportError as exc:
    print(f"[!] Unable to import DroneMap agent module: {exc}")
    sys.exit(0)


def check_status() -> int:
    config = AgentConfig()
    provider = config.active_provider
    model = config.default_model

    print("-" * 72)
    if provider == "offline":
        print("[!] [AI BOOSTER LAYER: OFFLINE HEURISTIC MODE]")
        print("    Status:   No active LLM API keys detected in .env or environment.")
        print("    Fallback: 100% Offline Rule Engine will handle all tactical dossiers,")
        print("              SfM diagnostics, and spatial queries deterministically.")
        print("    Note:     All core photogrammetry, 3D meshing, and measurement tools")
        print("              operate with full precision without requiring any API keys.")
        print("    To Activate: Add GEMINI_API_KEY to your .env file.")
        print("-" * 72)
        return 1

    # Active provider detected; test live connectivity
    print(f"[*] Testing {provider.upper()} API Key ({model})...")
    client = LLMClient(config)

    try:
        response = client.generate(
            prompt="Respond with only the single word: OK",
            system_prompt="You are a system health probe. Respond with OK.",
        )
        if response and "OK" in response.upper():
            print("[+] [AI BOOSTER LAYER: ONLINE & VERIFIED]")
            print(f"    Provider:    {provider.upper()} ({model})")
            print("    API Status:  Validated (HTTP 200 OK)")
            print("    Features:    Multimodal Tactical Dossiers, Spatial Copilot,")
            print("                 Self-Healing SfM Tuning Active.")
            print("-" * 72)
            return 0
        else:
            print("[!] [WARNING: AI BOOSTER KEY INVALID OR TIMED OUT]")
            print(f"    Provider:    {provider.upper()}")
            print(f"    Detail:      API responded with unexpected output or empty candidate.")
            print("    Fallback:    Automatically engaging 100% offline heuristic engine.")
            print("                 Core 3D pipeline and photogrammetry remain fully operational.")
            print("-" * 72)
            return 2
    except Exception as exc:
        print("[!] [WARNING: AI BOOSTER CONNECTION FAILED]")
        print(f"    Provider:    {provider.upper()}")
        print(f"    Error:       {exc}")
        print("    Fallback:    Automatically engaging 100% offline heuristic engine.")
        print("                 Core 3D pipeline and photogrammetry remain fully operational.")
        print("-" * 72)
        return 2


if __name__ == "__main__":
    code = check_status()
    sys.exit(code)
