"""Linked to nepsequant.com, the AI analyst reviews the assigned website
strategy's picks and the shared paper book — never the local signal engine."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from apps.tui import dashboard_tui
from apps.tui.io import website_strategies as ws
from backend.agents import agent_analyst


class _Book:
    def portfolio_frame(self):
        return pd.DataFrame([{"Entry_Date": "2026-11-10", "Symbol": "NABIL", "Quantity": 10, "Buy_Price": 500.0,
                              "Total_Cost_Basis": 5025.0, "Last_LTP": 510.0, "Signal_Type": ""}])


def _strategies(assigned=True):
    s = ws.WebsiteStrategies("nqk_x", request=lambda *a, **k: {})
    s.strategies = [{"id": "rs:value", "kind": "rules", "name": "Value banks"}]
    s.loaded = True
    s.assigned = "rs:value" if assigned else None
    s._picks["rs:value"] = ws.Picks(strategy_id="rs:value", as_of="2026-11-13", fetched_at=1e12, rows=[
        {"rank": 1, "symbol": "SBL", "reason": "P/E 9.8"}, {"rank": 2, "symbol": "HBL", "reason": "P/E 12"}])
    return s


@pytest.fixture
def app(tmp_path, monkeypatch):
    from backend.quant_pro import paths

    monkeypatch.setattr(paths, "get_runtime_dir", lambda anchor=None: tmp_path)
    monkeypatch.setattr(agent_analyst, "AGENT_SIGNAL_SNAPSHOT_FILE", tmp_path / "snap.json")
    for name in ("NEPSE_ACTIVE_ACCOUNT_ID", "NEPSE_ACTIVE_STRATEGY_ID", "NEPSE_ACTIVE_PORTFOLIO_FILE",
                 "NEPSE_ACTIVE_ACCOUNT_DIR", "NEPSE_ACTIVE_ACCOUNT_NAME", "NEPSE_ACTIVE_STRATEGY_NAME",
                 "NEPSE_AGENT_SIGNALS_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    a = dashboard_tui.NepseDashboard.__new__(dashboard_tui.NepseDashboard)
    a._website_book = _Book()
    return a


def test_agent_reviews_the_assigned_website_strategy_and_shared_book(app):
    app._website_strategies = _strategies()
    app._sync_agent_account_context_env()
    snap = agent_analyst._matching_signal_snapshot()
    assert [s["symbol"] for s in snap["signals"]] == ["SBL", "HBL"]
    assert snap["strategy_name"] == "Value banks" and "Value banks" in snap["signals"][0]["reasoning"]
    rows, _meta = agent_analyst._load_active_portfolio()
    assert rows == [{"symbol": "NABIL", "qty": 10, "entry": 500.0}]  # the shared book, not a local file
    assert os.environ["NEPSE_AGENT_SIGNALS_SOURCE"] == "website"


def test_no_assigned_strategy_means_no_candidates_and_a_clear_reason(app, monkeypatch):
    app._website_strategies = _strategies(assigned=False)
    app._agent_analysis_running = False
    app._sync_agent_account_context_env()
    assert agent_analyst._matching_signal_snapshot() == {}
    assert "ASSIGN" in app._agent_empty_picks_reason({"timestamp": 1, "shortlist": []})


def test_reasons_for_an_empty_picks_table(app):
    app._website_strategies = _strategies()
    app._agent_analysis_running = False
    assert app._agent_empty_picks_reason({"raw_response": "ERROR: Ollama has no models installed."}).startswith("Ollama")
    assert "picked nothing" in app._agent_empty_picks_reason({"timestamp": 1, "shortlist": []})
    app._agent_analysis_running = True
    assert app._agent_empty_picks_reason({}).startswith("Analysing")
