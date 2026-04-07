"""Unit tests for local TMS audit and validation helpers."""

from __future__ import annotations

from types import SimpleNamespace

from backend.quant_pro.tms_audit import (
    load_latest_tms_snapshot,
    load_execution_intent,
    list_executed_trade_events,
    save_execution_intent,
    save_execution_result,
    save_tms_snapshot,
)
from backend.quant_pro.tms_executor import LocalTMSExecutionService, TMSBrowserExecutor, _extract_watchlist_items
from backend.quant_pro.tms_models import (
    ExecutionAction,
    ExecutionIntent,
    ExecutionResult,
    ExecutionSource,
    ExecutionStatus,
    FillState,
)
from backend.quant_pro.tms_session import TMSSettings


def test_audit_round_trip_and_executed_events(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_LIVE_AUDIT_DB_FILE", str(tmp_path / "live_audit.db"))

    intent = ExecutionIntent(
        action=ExecutionAction.BUY,
        symbol="NABIL",
        quantity=10,
        limit_price=500.0,
        source=ExecutionSource.OWNER_MANUAL,
        requires_confirmation=True,
        status=ExecutionStatus.PENDING_CONFIRMATION,
    )
    save_execution_intent(intent)

    restored = load_execution_intent(intent.intent_id)
    assert restored is not None
    assert restored.symbol == "NABIL"
    assert restored.status == ExecutionStatus.PENDING_CONFIRMATION

    result = ExecutionResult(
        intent_id=intent.intent_id,
        status=ExecutionStatus.FILLED,
        submitted=True,
        fill_state=FillState.FILLED,
        broker_order_ref="BRK-1",
        observed_price=501.0,
        observed_qty=10,
        completed_at="2026-03-26T12:30:00+00:00",
        status_text="Filled",
    )
    save_execution_result(intent, result, source="test")

    events = list_executed_trade_events(limit=5)
    assert events
    assert events[0]["intent_id"] == intent.intent_id
    assert events[0]["broker_order_ref"] == "BRK-1"


def test_modify_validation_allows_price_only_change(monkeypatch, tmp_path):
    settings = TMSSettings(
        enabled=True,
        mode="live",
        profile_dir=tmp_path / "profile",
        screenshot_dir=tmp_path / "shots",
    )
    settings.profile_dir.mkdir(parents=True, exist_ok=True)
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    service = LocalTMSExecutionService(settings, snapshot_provider=lambda: {"cash": 1_000_000.0, "positions": {}, "max_positions": 10})

    monkeypatch.setattr(service, "session_status", lambda force=False: SimpleNamespace(ready=True, login_required=False))
    monkeypatch.setattr("backend.trading.live_trader.is_market_open", lambda: True)
    monkeypatch.setattr("backend.quant_pro.tms_executor.fetch_latest_ltp", lambda symbol: 500.0)
    monkeypatch.setattr("backend.quant_pro.tms_executor.count_intents_for_day", lambda day_str: 0)
    monkeypatch.setattr("backend.quant_pro.tms_executor.find_recent_open_intent", lambda symbol, within_seconds=None: None)

    intent = ExecutionIntent(
        action=ExecutionAction.MODIFY,
        symbol="NABIL",
        quantity=0,
        limit_price=505.0,
        source=ExecutionSource.OWNER_MANUAL,
        target_order_ref="ORD-1",
        requires_confirmation=True,
    )

    ok, detail = service.validate_intent(intent)

    assert ok is True
    assert detail == "ok"


def test_tms_snapshot_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_LIVE_AUDIT_DB_FILE", str(tmp_path / "live_audit.db"))

    save_tms_snapshot(
        "tms_funds",
        {
            "snapshot_time_utc": "2026-03-26T12:00:00+00:00",
            "collateral_available": 123.45,
            "fund_transfer_amount": 50.0,
        },
    )

    restored = load_latest_tms_snapshot("tms_funds")

    assert restored is not None
    assert restored["collateral_available"] == 123.45
    assert restored["snapshot_type"] == "tms_funds"


def test_extract_watchlist_items_filters_and_normalizes_rows():
    rows = [
        {"symbol": "NABIL", "ltp": "520.5", "change_pct": "+1.2", "volume": "12,345"},
        {"scrip": "SHIVM", "last_traded_price": "501.0", "change": "-0.6", "qty": "98"},
        {"symbol": "TOTAL", "ltp": "0"},
        {"symbol": "nabil", "ltp": "521.0"},
        {"script": "bad symbol", "price": "123"},
    ]

    items = _extract_watchlist_items(rows)

    assert [item["symbol"] for item in items] == ["NABIL", "SHIVM"]
    assert items[0]["ltp"] == 520.5
    assert items[0]["change_pct"] == 1.2
    assert items[0]["volume"] == 12345
    assert items[1]["ltp"] == 501.0
    assert items[1]["change_pct"] == -0.6


def test_headed_watchlist_reads_reuse_browser_session(tmp_path, monkeypatch):
    settings = TMSSettings(
        enabled=True,
        mode="live",
        headless=False,
        profile_dir=tmp_path / "profile",
        screenshot_dir=tmp_path / "shots",
    )
    settings.profile_dir.mkdir(parents=True, exist_ok=True)
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    executor = TMSBrowserExecutor(settings)

    launches = []
    contexts = []

    class FakePlaywright:
        def stop(self):
            return None

    class FakeContext:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_page = SimpleNamespace(url="https://tms19.nepsetms.com.np/tms/client/dashboard", is_closed=lambda: False)

    def fake_launch_context(*, headless=None):
        launches.append(bool(headless))
        ctx = FakeContext()
        contexts.append(ctx)
        return FakePlaywright(), ctx

    monkeypatch.setattr(executor, "_launch_context", fake_launch_context)
    monkeypatch.setattr(executor, "_open_page", lambda context, preferred_page=None: preferred_page or fake_page)
    monkeypatch.setattr(executor, "_first_locator", lambda page, key: None)
    monkeypatch.setattr(executor, "_fetch_watchlist_snapshot", lambda page: {"symbols": ["RURU"], "count": 1})

    first = executor.fetch_watchlist_snapshot()
    second = executor.fetch_watchlist_snapshot()

    assert first["symbols"] == ["RURU"]
    assert second["symbols"] == ["RURU"]
    assert launches == [False]
    assert contexts and contexts[0].closed is False

    executor.close()

    assert contexts[0].closed is True


def test_watchlist_editor_descriptor_finds_global_modal(tmp_path):
    settings = TMSSettings(
        enabled=True,
        mode="live",
        profile_dir=tmp_path / "profile",
        screenshot_dir=tmp_path / "shots",
    )
    settings.profile_dir.mkdir(parents=True, exist_ok=True)
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    executor = TMSBrowserExecutor(settings)

    class FakePage:
        def evaluate(self, script):
            return {
                "trigger_id": "edit-watchlist",
                "target": "#watchlist-modal",
                "modal_id": "watchlist-modal",
            }

    descriptor = executor._watchlist_editor_descriptor(FakePage())

    assert descriptor == {
        "trigger_id": "edit-watchlist",
        "target": "#watchlist-modal",
        "modal_id": "watchlist-modal",
    }


def test_watchlist_table_rows_parse_member_market_watch_rows(tmp_path, monkeypatch):
    settings = TMSSettings(
        enabled=True,
        mode="live",
        profile_dir=tmp_path / "profile",
        screenshot_dir=tmp_path / "shots",
    )
    settings.profile_dir.mkdir(parents=True, exist_ok=True)
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    executor = TMSBrowserExecutor(settings)

    class FakePage:
        def evaluate(self, script):
            return [
                {"cells": ["Particulars", "Remarks"], "tooltip": ""},
                {
                    "cells": ["RURU", "636", "636", "636", "636", "636", "0.00", "0.00"],
                    "tooltip": "Ru Ru Jalbidhyut",
                },
                {
                    "cells": ["CIT", "1758", "1758", "1758", "1758", "1758", "0.00", "0.00"],
                    "tooltip": "",
                },
            ]

    monkeypatch.setattr(executor, "_open_member_market_watch", lambda page: None)

    rows = executor._watchlist_table_rows(FakePage())

    assert [row["symbol"] for row in rows] == ["RURU", "CIT"]
    assert rows[0]["ltp"] == 636.0
    assert rows[0]["high"] == 636.0
    assert rows[0]["change_pct"] == 0.0
