"""Where the AI analyst's provider choice and API keys live on this computer.

    ~/.config/nepse-quant-terminal/ai.json     (folder 0700, file 0600)

Same folder rule as ``apps/cli/connect.py``: ``$XDG_CONFIG_HOME`` wins over
``~/.config``. The file holds the provider picked in ``./nqt-connect ai setup``,
the model remembered for each provider, and the OpenAI / Anthropic keys.
Environment variables ``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` always win
over the file. Keys are only ever sent to the provider they belong to, and are
never printed, logged or written to chat history.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

CLOUD_PROVIDERS = ("openai", "anthropic")
PROVIDERS = ("ollama", "openai", "anthropic")
ENV_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# What a pasted secret looks like: OpenAI (sk-..., sk-proj-...), Anthropic
# (sk-ant-...) and NEPSE Quant (nqk_...) keys.
_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}|nqk_[A-Za-z0-9_\-]{12,})")


def ai_config_path() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / "nepse-quant-terminal" / "ai.json"


def load_ai_settings() -> dict[str, Any]:
    try:
        data = json.loads(ai_config_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_ai_settings(data: dict[str, Any]) -> Path:
    """Write the settings with owner-only permissions (folder 0700, file 0600)."""
    path = ai_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, stat.S_IRWXU)
    except OSError:
        pass
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def clear_ai_settings() -> bool:
    """Delete the settings file (keys included). True when there was one."""
    path = ai_config_path()
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def get_api_key(provider: str) -> tuple[str, str]:
    """(key, where it came from: "env" | "file" | "") for openai / anthropic."""
    clean = str(provider or "").strip().lower()
    env_name = ENV_KEYS.get(clean)
    if not env_name:
        return "", ""
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env, "env"
    saved = str(dict(load_ai_settings().get("keys") or {}).get(clean) or "").strip()
    return (saved, "file") if saved else ("", "")


def update_ai_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    key: str | None = None,
    model_provider: str | None = None,
) -> Path:
    """Merge one change into the file. ``model`` is remembered for ``model_provider``
    (default: ``provider``); ``key`` is saved for ``provider``."""
    data = load_ai_settings()
    if provider:
        data["provider"] = provider
    target = model_provider or provider
    if model is not None and target:
        models = dict(data.get("models") or {})
        if model.strip():
            models[target] = model.strip()
        else:
            models.pop(target, None)
        data["models"] = models
    if key is not None and provider:
        keys = dict(data.get("keys") or {})
        if key.strip():
            keys[provider] = key.strip()
        else:
            keys.pop(provider, None)
        data["keys"] = keys
    return save_ai_settings(data)


def remembered_model(provider: str) -> str:
    return str(dict(load_ai_settings().get("models") or {}).get(str(provider or "").lower()) or "").strip()


def setting(name: str, default: str = "") -> str:
    """A plain string setting from the file, e.g. ``openai_base_url`` or ``ollama_host``."""
    return str(load_ai_settings().get(name) or default).strip()


def looks_like_api_key(text: str) -> bool:
    """True when the text contains something shaped like an API key."""
    return bool(_KEY_PATTERN.search(str(text or "")))


def redact_keys(text: str) -> str:
    return _KEY_PATTERN.sub("[key removed]", str(text or ""))


KEY_IN_CHAT_MESSAGE = (
    "That looks like an API key, so it was not sent or saved (chat history is stored on disk). "
    "Save keys with ./nqt-connect ai setup in a terminal instead."
)
