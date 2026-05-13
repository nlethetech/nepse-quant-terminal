from __future__ import annotations

import json

from backend.agents.runtime_config import load_active_agent_config, save_active_agent_config, set_active_agent


def test_active_agent_config_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime_config.ACTIVE_AGENT_FILE",
        tmp_path / "active_agent.json",
        raising=False,
    )

    cfg = load_active_agent_config()

    assert cfg["selected_preset"] == "ollama"
    assert cfg["backend"] == "ollama"
    assert cfg["provider_label"] == "ollama"
    assert cfg["model"] == "llama3"
    assert (tmp_path / "active_agent.json").exists()
    saved = json.loads((tmp_path / "active_agent.json").read_text())
    assert saved["backend"] == "ollama"


def test_set_active_agent_persists_selected_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime_config.ACTIVE_AGENT_FILE",
        tmp_path / "active_agent.json",
        raising=False,
    )

    saved = set_active_agent("claude")
    restored = load_active_agent_config()

    assert saved["backend"] == "claude"
    assert restored["backend"] == "claude"
    assert restored["provider_label"] == "claude"


def test_save_active_agent_config_allows_model_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime_config.ACTIVE_AGENT_FILE",
        tmp_path / "active_agent.json",
        raising=False,
    )

    payload = save_active_agent_config(
        {
            "selected_preset": "gemma4_mlx",
            "backend": "gemma4_mlx",
            "model": "custom/model",
            "provider_label": "gemma4_mlx",
            "source_label": "local_gemma4_mlx",
        }
    )

    assert payload["model"] == "custom/model"


def test_active_agent_config_repairs_invalid_json(tmp_path, monkeypatch):
    target = tmp_path / "active_agent.json"
    target.write_text("{bad json")
    monkeypatch.setattr(
        "backend.agents.runtime_config.ACTIVE_AGENT_FILE",
        target,
        raising=False,
    )

    cfg = load_active_agent_config()

    assert cfg["backend"] == "ollama"
    assert json.loads(target.read_text())["backend"] == "ollama"
