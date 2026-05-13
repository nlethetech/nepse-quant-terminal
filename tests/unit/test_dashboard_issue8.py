from __future__ import annotations

import json

from apps.tui import dashboard_tui as tui
from backend.agents import runtime_config


def test_watchlist_load_dedupes_and_repairs_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "watchlist.json"
    target.write_text(
        json.dumps(
            [
                "nabil",
                {"kind": "stock", "symbol": "NABIL"},
                {"kind": "forex", "key": "forex:USD", "label": "USD"},
                {"kind": "forex", "key": "forex:USD", "label": "USD"},
            ]
        )
    )
    monkeypatch.setattr(tui, "WATCHLIST_FILE", target)

    rows = tui._load_watchlist()

    assert [row["key"] for row in rows] == ["stock:NABIL", "forex:USD"]
    repaired = json.loads(target.read_text())
    assert [row["key"] for row in repaired] == ["stock:NABIL", "forex:USD"]


def test_agent_chat_command_can_switch_ollama_model(tmp_path, monkeypatch):
    target = tmp_path / "active_agent.json"
    monkeypatch.setattr(runtime_config, "ACTIVE_AGENT_FILE", target)
    monkeypatch.setattr(tui, "ACTIVE_AGENT_FILE", target)
    notes: list[str] = []
    statuses: list[str] = []
    populated: list[bool] = []

    app = tui.NepseDashboard.__new__(tui.NepseDashboard)
    app._append_chat_note = notes.append
    app._set_status = statuses.append
    app._populate_agent_tab = lambda: populated.append(True)

    handled = tui.NepseDashboard._handle_agent_chat_command(app, "/agent ollama gemma4:e2b")

    cfg = runtime_config.load_active_agent_config()
    assert handled
    assert cfg["backend"] == "ollama"
    assert cfg["model"] == "gemma4:e2b"
    assert notes and "gemma4:e2b" in notes[-1]
    assert statuses and "Agent switched" in statuses[-1]
    assert populated == [True]
