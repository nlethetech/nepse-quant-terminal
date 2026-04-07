from __future__ import annotations

from backend.agents.runtime_config import load_active_agent_config, save_active_agent_config, set_active_agent


def test_active_agent_config_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime_config.ACTIVE_AGENT_FILE",
        tmp_path / "active_agent.json",
        raising=False,
    )

    cfg = load_active_agent_config()

    assert cfg["backend"] == "gemma4_mlx"
    assert cfg["provider_label"] == "gemma4_mlx"


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
