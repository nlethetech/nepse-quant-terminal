from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.quant_pro.paths import ensure_dir, get_project_root, get_runtime_dir, migrate_legacy_path

RUNTIME_DIR = ensure_dir(get_runtime_dir(__file__))
AGENTS_RUNTIME_DIR = ensure_dir(RUNTIME_DIR / "agents")
PROJECT_ROOT = get_project_root(__file__)
ACTIVE_AGENT_FILE = migrate_legacy_path(
    AGENTS_RUNTIME_DIR / "active_agent.json",
    [PROJECT_ROOT / "active_agent.json"],
)

DEFAULT_GEMMA4_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
EXPERIMENTAL_GEMMA4_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
DEFAULT_AGENT_PRESET = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

AGENT_BACKEND_PRESETS: dict[str, dict[str, Any]] = {
    "ollama": {
        "backend": "ollama",
        "label": "Ollama (local)",
        "provider_label": "ollama",
        "source_label": "local_ollama",
        "model": DEFAULT_OLLAMA_MODEL,
        "ollama_host": DEFAULT_OLLAMA_HOST,
        "fallback_backend": "",
        "trust_remote_code": False,
        "description": "Default local agent backend served by Ollama.",
    },
    "gemma4_mlx": {
        "backend": "gemma4_mlx",
        "label": "Gemma 4 MLX",
        "provider_label": "gemma4_mlx",
        "source_label": "local_gemma4_mlx",
        "model": DEFAULT_GEMMA4_MODEL,
        "fallback_backend": "",
        "trust_remote_code": False,
        "description": "Optional local Gemma 4 on MLX for Apple Silicon.",
    },
    "gemma4_experimental": {
        "backend": "gemma4_mlx",
        "label": "Gemma 4 MLX Experimental",
        "provider_label": "gemma4_mlx",
        "source_label": "local_gemma4_mlx",
        "model": EXPERIMENTAL_GEMMA4_MODEL,
        "fallback_backend": "",
        "trust_remote_code": False,
        "description": "Local Gemma 4 experimental MLX checkpoint.",
    },
    "claude": {
        "backend": "claude",
        "label": "Claude CLI",
        "provider_label": "claude",
        "source_label": "local_claude",
        "model": "",
        "fallback_backend": "",
        "trust_remote_code": False,
        "description": "Claude CLI backend for built-in analysis and local chat.",
    },
}


def list_agent_backends() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, preset in AGENT_BACKEND_PRESETS.items():
        rows.append(
            {
                "id": key,
                "backend": str(preset.get("backend") or ""),
                "label": str(preset.get("label") or key),
                "provider_label": str(preset.get("provider_label") or ""),
                "source_label": str(preset.get("source_label") or ""),
                "model": str(preset.get("model") or ""),
                "fallback_backend": str(preset.get("fallback_backend") or ""),
                "description": str(preset.get("description") or ""),
            }
        )
    return rows


def _default_active_agent_config() -> dict[str, Any]:
    payload = dict(AGENT_BACKEND_PRESETS[DEFAULT_AGENT_PRESET])
    payload["selected_preset"] = DEFAULT_AGENT_PRESET
    payload["updated_at"] = time.time()
    return payload


def _normalize_active_agent_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    selected_preset = str(payload.get("selected_preset") or payload.get("backend") or DEFAULT_AGENT_PRESET).strip().lower()
    base = dict(AGENT_BACKEND_PRESETS.get(selected_preset) or _default_active_agent_config())

    backend = str(payload.get("backend") or base.get("backend") or DEFAULT_AGENT_PRESET).strip().lower()
    normalized = {
        "selected_preset": selected_preset if selected_preset in AGENT_BACKEND_PRESETS else backend,
        "backend": backend,
        "label": str(payload.get("label") or base.get("label") or backend),
        "provider_label": str(payload.get("provider_label") or base.get("provider_label") or backend),
        "source_label": str(payload.get("source_label") or base.get("source_label") or backend),
        "model": str(payload.get("model") or base.get("model") or "").strip(),
        "fallback_backend": str(payload.get("fallback_backend") or base.get("fallback_backend") or "").strip().lower(),
        "trust_remote_code": bool(payload.get("trust_remote_code") if payload.get("trust_remote_code") is not None else base.get("trust_remote_code")),
        "description": str(payload.get("description") or base.get("description") or ""),
        "updated_at": float(payload.get("updated_at") or time.time()),
    }
    return normalized


def load_active_agent_config() -> dict[str, Any]:
    if not ACTIVE_AGENT_FILE.exists():
        return _default_active_agent_config()
    try:
        payload = json.loads(ACTIVE_AGENT_FILE.read_text())
    except Exception:
        return _default_active_agent_config()
    return _normalize_active_agent_config(payload if isinstance(payload, dict) else {})


def save_active_agent_config(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_active_agent_config(config)
    normalized["updated_at"] = time.time()
    ACTIVE_AGENT_FILE.write_text(json.dumps(normalized, indent=2, default=str))
    return normalized


def set_active_agent(
    backend: str,
    *,
    model: str | None = None,
    provider_label: str | None = None,
    source_label: str | None = None,
    fallback_backend: str | None = None,
    trust_remote_code: bool | None = None,
) -> dict[str, Any]:
    clean = str(backend or "").strip().lower()
    if not clean:
        raise ValueError("backend is required")

    preset = dict(AGENT_BACKEND_PRESETS.get(clean) or {})
    if not preset and clean not in {preset["backend"] for preset in AGENT_BACKEND_PRESETS.values()}:
        raise ValueError(f"Unknown agent backend or preset: {backend}")

    payload = _normalize_active_agent_config(
        {
            "selected_preset": clean if clean in AGENT_BACKEND_PRESETS else "",
            "backend": str(preset.get("backend") or clean),
            "label": str(preset.get("label") or clean),
            "provider_label": provider_label if provider_label is not None else preset.get("provider_label"),
            "source_label": source_label if source_label is not None else preset.get("source_label"),
            "model": model if model is not None else preset.get("model"),
            "fallback_backend": fallback_backend if fallback_backend is not None else preset.get("fallback_backend"),
            "trust_remote_code": trust_remote_code if trust_remote_code is not None else preset.get("trust_remote_code"),
            "description": str(preset.get("description") or ""),
        }
    )
    return save_active_agent_config(payload)
