"""Keep unit tests away from the real user's keys, settings and network.

Every test gets its own empty config folder (so ~/.config/nepse-quant-terminal
credentials.json / ai.json are never read or written) and no API keys in the
environment, so nothing can reach nepsequant.com, OpenAI or Anthropic.
"""

from __future__ import annotations

import sys

import pytest

_SECRET_ENV = (
    "NEPSE_QUANT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_BASE_URL",
    "OLLAMA_HOST",
    "NEPSE_AGENT_BACKEND",
    "NEPSE_AGENT_MODEL",
    "NQT_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_user_config(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    briefing = sys.modules.get("backend.agents.nepse_briefing")
    if briefing is not None:
        briefing._CACHE.clear()
    analyst = sys.modules.get("backend.agents.agent_analyst")
    if analyst is not None:
        analyst._BASE_CONTEXT_CACHE.clear()
        analyst._MACRO_CACHE.clear()
    yield
