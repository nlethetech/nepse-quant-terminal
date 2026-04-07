"""Unit tests for dual-bot Telegram reporting surfaces."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

import backend.quant_pro.telegram_alerts as alerts
import backend.quant_pro.telegram_bot as bot
import backend.quant_pro.reporting as reporting
from backend.quant_pro.tms_models import ExecutionAction, ExecutionIntent, ExecutionSource, ExecutionStatus


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyTarget:
    def __init__(self):
        self.messages = []

    async def reply_text(self, text, **kwargs):
        self.messages.append(("text", text, kwargs))

    async def reply_photo(self, photo=None, caption=None, **kwargs):
        self.messages.append(("photo", caption, kwargs))


class DummyBot:
    def __init__(self, status="member"):
        self.status = status

    async def get_chat_member(self, chat_id, user_id):
        return SimpleNamespace(status=self.status)


def _dummy_trader():
    trader = SimpleNamespace(
        _state_lock=DummyLock(),
        last_refresh=None,
        last_price_source_label="Nepalstock",
        last_price_source_detail="primary",
        last_price_snapshot_time_utc="2026-03-26T12:00:00+00:00",
        capital=1_000_000.0,
        cash=400_000.0,
        positions={},
        trade_log_file="paper_trade_log.csv",
        nav_log_file="paper_nav_log.csv",
        regime="bull",
        signals_today=[],
        live_execution_enabled=False,
    )
    trader.get_tms_snapshot = lambda kind, force=False, max_age_secs=180: {}
    trader.get_tms_health_summary = lambda force=False: {"enabled": True, "ready": True, "login_required": False, "selector_health_pct": 100.0, "selector_health": {}, "last_sync_utc": "2026-03-26T12:00:00+00:00"}
    return trader


def _dummy_live_trader():
    intent = ExecutionIntent(
        action=ExecutionAction.BUY,
        symbol="NABIL",
        quantity=10,
        limit_price=500.0,
        source=ExecutionSource.OWNER_MANUAL,
        requires_confirmation=True,
        status=ExecutionStatus.PENDING_CONFIRMATION,
    )
    trader = _dummy_trader()
    trader.live_execution_enabled = True
    trader.create_live_owner_buy_intent = lambda symbol, shares, limit_price: (True, "pending_confirmation", intent)
    trader._format_live_receipt_html = lambda live_intent, result=None: f"<b>LIVE EXECUTION</b>\n<code>{live_intent.symbol}</code>"
    return trader, intent


def _fake_owner_report():
    return {
        "generated_at_nst": "2026-03-26T18:00:00+05:45",
        "summary": {
            "capital": 1_000_000.0,
            "cash": 400_000.0,
            "nav": 1_050_000.0,
            "total_return_pct": 5.0,
            "realized_pnl": 30_823.0,
            "realized_return_pct": 3.23,
            "open_positions": 3,
        },
        "portfolio": {
            "holdings_ranked": [
                {
                    "symbol": "BBB",
                    "unrealized_pnl": 20_000.0,
                    "contribution_vs_nepse_pts": 1.5,
                    "active_vs_sector_pct": 2.1,
                    "holding_days": 10,
                    "mark_source": "nepalstock",
                },
                {
                    "symbol": "AAA",
                    "unrealized_pnl": 11_000.0,
                    "contribution_vs_nepse_pts": 0.6,
                    "active_vs_sector_pct": 0.8,
                    "holding_days": 7,
                    "mark_source": "sqlite_cache",
                },
                {
                    "symbol": "CCC",
                    "unrealized_pnl": -5_000.0,
                    "contribution_vs_nepse_pts": -0.4,
                    "active_vs_sector_pct": -1.2,
                    "holding_days": 3,
                    "mark_source": "merolagani",
                },
            ],
            "recent_trades": [
                {"date": "2026-03-26", "action": "BUY", "symbol": "BBB", "shares": 100, "price": 500.0},
            ],
        },
        "alpha": {
            "performance": {"deployed_return_pct": 12.75, "realized_pnl": 30_823.0, "realized_return_pct": 3.23},
            "global_benchmark": {"return_pct": 10.39},
            "attribution_stack": {
                "gross_alpha_pnl": 24500.0,
                "stock_selection_alpha_pnl": 22000.0,
                "sector_allocation_alpha_pnl": 4000.0,
                "timing_alpha_pnl": 1500.0,
                "cash_drag_pnl": -3000.0,
                "turnover_drag_pnl": -900.0,
                "fee_drag_pnl": -700.0,
                "net_alpha_pnl": 22900.0,
                "gross_alpha_return_pct": 2.9,
                "realized_alpha_pnl": 11000.0,
                "unrealized_alpha_pnl": 11900.0,
                "turnover_ratio_pct": 18.0,
            },
            "positions": [
                {
                    "symbol": "BBB",
                    "contribution_vs_nepse_pts": 1.5,
                    "active_vs_sector_pct": 2.1,
                    "active_vs_nepse_pct": 3.0,
                },
                {
                    "symbol": "AAA",
                    "contribution_vs_nepse_pts": 0.6,
                    "active_vs_sector_pct": 0.8,
                    "active_vs_nepse_pct": 1.2,
                },
            ],
        },
        "health": {
            "data_quality": {
                "freeze_alpha": False,
                "confidence_score": 92,
                "stale_symbols": ["CCC"],
                "alerts": ["mixed-source marks detected"],
                "common_benchmark_date": "2026-03-26",
                "primary_symbols": ["BBB"],
                "fallback_symbols": ["CCC"],
                "cache_symbols": ["AAA"],
            }
        },
        "daily": {
            "date": "2026-03-26",
            "nav": 1_050_000.0,
            "day_pnl": -8_624.0,
            "day_return_pct": -0.82,
            "invested_pct": 61.9,
            "benchmark_move_pct": 1.25,
            "confidence_score": 92,
            "last_refresh_nst": "2026-03-26 11:29:35",
            "trades_today": [{"action": "BUY", "symbol": "BBB", "shares": 100, "price": 500.0}],
            "biggest_contributor": {"symbol": "BBB", "contribution_vs_nepse_pts": 1.5},
            "biggest_detractor": {"symbol": "CCC", "contribution_vs_nepse_pts": -0.4},
        },
    }


def test_viewer_application_registers_only_public_handlers():
    app = bot._build_application("123:ABC", role="viewer", allowed_chat_ids={1})
    handlers = app.handlers[0]
    commands = {
        next(iter(handler.commands))
        for handler in handlers
        if getattr(handler, "commands", None)
    }
    callback_patterns = {getattr(handler, "pattern", None) for handler in handlers if handler.__class__.__name__ == "CallbackQueryHandler"}

    assert commands == {"start", "help", "portfolio", "performance", "daily", "trades"}
    assert not {"buy", "sell", "alpha", "risk", "health", "refresh", "summary"} & commands
    assert any(getattr(pattern, "pattern", "") == "^viewer_" for pattern in callback_patterns if pattern is not None)
    assert not any(getattr(pattern, "pattern", "").startswith("^buy_") for pattern in callback_patterns if pattern is not None)


def test_owner_application_registers_private_handlers():
    app = bot._build_application("123:ABC", role="owner", allowed_chat_ids={1})
    handlers = app.handlers[0]
    commands = {
        next(iter(handler.commands))
        for handler in handlers
        if getattr(handler, "commands", None)
    }
    callback_patterns = [getattr(handler, "pattern", None) for handler in handlers if handler.__class__.__name__ == "CallbackQueryHandler"]

    assert {"buy", "sell", "health", "daily", "attribution", "risk", "alpha", "cancel", "modify", "tms_status", "tms_account", "tms_funds", "tms_holdings", "tms_orders", "tms_trades", "tms_health", "orders_live", "positions_live", "kill_live", "resume_live", "reconcile_live", "mode_live"} <= commands
    assert any(getattr(pattern, "pattern", "") == "^buy_(confirm|cancel)$" for pattern in callback_patterns if pattern is not None)
    assert any(getattr(pattern, "pattern", "") == "^live_(confirm|cancel)_.+$" for pattern in callback_patterns if pattern is not None)


def test_viewer_report_respects_privacy_toggles(monkeypatch):
    monkeypatch.setenv("NEPSE_VIEWER_SHOW_HOLDINGS", "false")
    monkeypatch.setenv("NEPSE_VIEWER_SHOW_TRADES", "false")
    monkeypatch.setenv("NEPSE_VIEWER_SHOW_BENCHMARK", "false")
    monkeypatch.setattr(reporting, "build_owner_report", lambda trader: _fake_owner_report())

    report = reporting.build_viewer_report(_dummy_trader())

    assert report is not None
    assert report["summary"]["benchmark_return_pct"] is None
    assert report["portfolio"]["holdings"] == []
    assert report["portfolio"]["recent_trades"] == []
    assert report["daily"]["trades_today"] == []
    assert report["daily"]["biggest_contributor"] is None


def test_viewer_messages_separate_total_deployed_and_today(monkeypatch):
    monkeypatch.setattr(reporting, "build_owner_report", lambda trader: _fake_owner_report())
    monkeypatch.setattr(bot, "_format_refresh_meta", lambda trader: ["Marks: 2026-04-02 11:29:35 NST (1m ago)"])
    trader = _dummy_trader()

    portfolio = bot._build_viewer_portfolio_message(trader)
    performance = bot._build_viewer_performance_message(trader)

    assert "Total Ret" in portfolio
    assert "Realized:" in portfolio
    assert "Deployed" in portfolio
    assert "Today" in portfolio
    assert "Marks:" in portfolio
    assert "Total Return" in performance
    assert "Deployed" in performance
    assert "Today" in performance


def test_owner_messages_render_health_daily_and_attribution(monkeypatch):
    monkeypatch.setattr(reporting, "build_owner_report", lambda trader: _fake_owner_report())
    trader = _dummy_trader()

    health = bot._build_health_message(trader)
    daily = bot._build_owner_daily_message(trader)
    attribution = bot._build_attribution_message(trader)

    assert "Confidence" in health and "Alpha Frozen" in health
    assert "mixed-source marks detected" in health
    assert "TRADES TODAY" in daily and "BIGGEST CONTRIBUTOR" in daily
    assert "Portfolio Move" in daily and "Last Snapshot" in daily
    assert "ATTRIBUTION" in attribution
    assert "Gross Alpha" in attribution and "Net Alpha" in attribution


def test_alpha_message_is_summary_first(monkeypatch):
    trader = _dummy_trader()

    monkeypatch.setattr(
        "backend.trading.live_trader.compute_portfolio_intelligence",
        lambda *args, **kwargs: {
            "performance": {
                "deployed_return_pct": 12.75,
                "open_return_pct": -1.44,
                "realized_pnl": 30823.0,
                "open_unrealized_pnl": -12354.0,
            },
            "global_benchmark": {"return_pct": 10.39},
            "data_quality": {
                "freeze_alpha": False,
                "confidence_score": 92,
                "alerts": ["mixed-source marks detected"],
                "common_benchmark_date": "2026-04-02",
            },
            "attribution_stack": {
                "gross_alpha_pnl": 24500.0,
                "stock_selection_alpha_pnl": 22000.0,
                "sector_allocation_alpha_pnl": 4000.0,
                "timing_alpha_pnl": 1500.0,
                "cash_drag_pnl": -3000.0,
                "alpha_cost_drag_pnl": -1600.0,
                "net_alpha_pnl": 22900.0,
                "gross_alpha_return_pct": 2.9,
            },
            "positions": [
                {"symbol": "BBB", "contribution_vs_nepse_pts": 1.5, "active_vs_nepse_pct": 3.0},
                {"symbol": "AAA", "contribution_vs_nepse_pts": 0.6, "active_vs_nepse_pct": 1.2},
            ],
        },
    )
    monkeypatch.setattr(bot, "_format_refresh_meta", lambda trader: ["Marks: 2026-04-02 11:29:35 NST (1m ago)"])

    text = bot._build_alpha_message(trader)

    assert "SUMMARY" in text
    assert "PROFIT" in text
    assert "ACTIVE NAMES" in text
    assert "ATTRIBUTION STACK" not in text
    assert "Fees + Turnover" in text


def test_alpha_dashboard_image_renders(monkeypatch):
    trader = _dummy_trader()

    monkeypatch.setattr(
        "backend.trading.live_trader.compute_portfolio_intelligence",
        lambda *args, **kwargs: {
            "performance": {
                "deployed_return_pct": 12.75,
                "open_return_pct": -1.44,
                "realized_pnl": 30823.0,
                "open_unrealized_pnl": -12354.0,
            },
            "global_benchmark": {"return_pct": 10.39},
            "data_quality": {
                "freeze_alpha": False,
                "confidence_score": 92,
                "alerts": [],
                "common_benchmark_date": "2026-04-02",
            },
            "attribution_stack": {
                "net_alpha_pnl": 22900.0,
                "gross_alpha_return_pct": 2.9,
                "turnover_drag_pnl": -900.0,
                "fee_drag_pnl": -700.0,
            },
            "positions": [
                {"symbol": "BBB", "contribution_vs_nepse_pts": 1.5, "active_vs_nepse_pct": 3.0, "active_vs_sector_pct": 2.1},
                {"symbol": "AAA", "contribution_vs_nepse_pts": 0.6, "active_vs_nepse_pct": 1.2, "active_vs_sector_pct": 0.8},
            ],
        },
    )
    monkeypatch.setattr(
        "backend.trading.live_trader.compute_deployed_nav_chart_data",
        lambda *args, **kwargs: {
            "sleeve_index": [100.0, 104.0, 107.5],
            "benchmark_index": [100.0, 101.5, 103.0],
            "dates": [
                pd.Timestamp("2026-02-09"),
                pd.Timestamp("2026-03-26"),
                pd.Timestamp("2026-04-02"),
            ],
        },
    )
    monkeypatch.setattr(bot, "_format_refresh_meta", lambda trader: ["Marks: 2026-04-02 11:29:35 NST (1m ago)"])

    image = bot._render_alpha_dashboard_image(trader)

    assert image is not None
    assert image.name == "alpha_dashboard.png"


def test_owner_portfolio_is_ranked_by_contribution(monkeypatch):
    monkeypatch.setattr(reporting, "build_owner_report", lambda trader: _fake_owner_report())
    monkeypatch.setattr(bot, "_format_refresh_meta", lambda trader: ["Marks: 2026-04-02 11:29:35 NST (1m ago)"])
    trader = _dummy_trader()
    bot._trader = trader
    target = DummyTarget()

    asyncio.run(bot._send_portfolio(target, role="owner"))

    assert target.messages
    text = target.messages[0][1]
    assert text.index("BBB") < text.index("AAA") < text.index("CCC")
    assert "Realized:" in text
    assert "Marks:" in text
    assert "Active:" not in text
    assert "SQLite cache" in text
    assert "stale" in text


def test_tms_messages_render(monkeypatch):
    trader = _dummy_trader()
    snapshots = {
        "account": {
            "trade_summary": {"total_turnover": 0.0, "traded_shares": 0, "transactions": 0, "scrips_traded": 0, "buy_count": 0, "sell_count": 0},
            "collateral_summary": {"collateral_amount": 0.55, "collateral_utilized": 0.0, "collateral_available": 0.55, "payable_amount": 0.0, "receivable_amount": 0.0},
            "dp_holding_summary": {"holdings_count": 1, "total_amount_cp": 393.5, "last_sync": "2025-05-23 21:24:00"},
        },
        "funds": {
            "collateral_total": 0.55,
            "collateral_utilized": 0.0,
            "collateral_available": 0.55,
            "fund_transfer_amount": 0.55,
            "max_refund_allowed": 0.0,
            "pending_refund_request": 0.0,
            "available_trading_limit": 0.55,
            "utilized_trading_limit": 0.0,
            "payable_amount": 0.0,
            "receivable_amount": 0.0,
            "recent_transactions": [{"date": "2026-03-26", "particular": "COLLATERAL DEPOSIT VIA CIPS", "amount": "0.55"}],
            "snapshot_time_utc": "2026-03-26T12:00:00+00:00",
        },
        "holdings": {
            "count": 1,
            "total_amount_cp": 393.5,
            "total_amount_ltp": 393.5,
            "last_sync": "2025-05-23 21:24:00",
            "items": [{"symbol": "HLI", "tms_balance": 1, "ltp": 393.5, "value_as_of_ltp": 393.5}],
        },
        "orders_daily": {"row_count": 0, "records": [], "snapshot_time_utc": "2026-03-26T12:00:00+00:00"},
        "orders_historic": {"row_count": 0, "records": []},
        "trades_daily": {"row_count": 1, "records": [{"symbol": "HLI", "buy_sell": "BUY", "qty": "1", "price": "393.5"}], "snapshot_time_utc": "2026-03-26T12:00:00+00:00"},
        "trades_historic": {"row_count": 1, "records": [{"symbol": "HLI", "date": "2026-03-26", "qty": "1"}]},
    }
    trader.get_tms_snapshot = lambda kind, force=False, max_age_secs=180: snapshots.get(kind, {})
    trader.get_tms_health_summary = lambda force=False: {
        "enabled": True,
        "ready": True,
        "login_required": False,
        "selector_health_pct": 100.0,
        "selector_health": {"dashboard": True, "funds": True},
        "last_sync_utc": "2026-03-26T12:00:00+00:00",
        "detail": "ok",
    }

    account = bot._build_tms_account_message(trader)
    funds = bot._build_tms_funds_message(trader)
    holdings = bot._build_tms_holdings_message(trader)
    trades = bot._build_tms_trades_message(trader)
    health = bot._build_tms_health_message(trader)

    assert "TMS ACCOUNT" in account and "DP HOLDING" in account
    assert "COLLATERAL" in funds and "REFUND / LOAD" in funds
    assert "HLI" in holdings
    assert "DAILY TRADE BOOK" in trades and "BUY" in trades
    assert "Selector Health" in health and "PAGE CHECKS" in health


def test_viewer_daily_summary_routes_public_when_enabled(monkeypatch):
    sent = {"owner": [], "viewer": []}
    monkeypatch.setattr(alerts, "send_alert", lambda message: sent["owner"].append(message))
    monkeypatch.setattr(alerts, "send_public_alert", lambda message: sent["viewer"].append(message))
    monkeypatch.setattr(alerts, "VIEWER_DAILY_POSTS", True)

    alerts.send_daily_summary(
        portfolio_value=1_050_000.0,
        day_pnl=12_500.0,
        open_positions=4,
        signals_generated=5,
        capital=1_000_000.0,
        since_start_return=0.1275,
        portfolio_day_return=0.0125,
        benchmark_day_return=0.0080,
        benchmark_return=0.1039,
        alpha_points=2.35,
        realized_pnl=18250.0,
    )

    assert sent["owner"]
    assert sent["viewer"]
    assert "Since Start" in sent["owner"][0]
    assert "NEPSE Today" in sent["owner"][0]
    assert "Realized" in sent["owner"][0]
    assert "+18,250" in sent["owner"][0]
    assert "Since Start" in sent["viewer"][0]
    assert "NEPSE Today" in sent["viewer"][0]
    assert "Today" in sent["viewer"][0]
    assert "NEPSE All" in sent["viewer"][0]
    assert "Alpha" in sent["viewer"][0]
    assert "+12.75%" in sent["viewer"][0]


def test_viewer_report_prefers_live_executed_trades(monkeypatch):
    monkeypatch.setattr(reporting, "build_owner_report", lambda trader: _fake_owner_report())
    monkeypatch.setattr(
        reporting,
        "list_executed_trade_events",
        lambda limit=8: [
            {
                "completed_at": "2026-03-26T12:00:00+00:00",
                "action": "buy",
                "symbol": "LIVE",
                "quantity": 25,
                "limit_price": 410.0,
                "broker_order_ref": "ABC123",
                "status_text": "Filled",
            }
        ],
    )

    report = reporting.build_viewer_report(_dummy_trader())

    assert report is not None
    assert report["portfolio"]["recent_trades"][0]["symbol"] == "LIVE"


def test_live_buy_entry_returns_confirmation_preview(monkeypatch):
    trader, intent = _dummy_live_trader()
    target = DummyTarget()
    update = SimpleNamespace(message=target)
    context = SimpleNamespace(args=["NABIL", "10", "500"], user_data={}, application=None)

    bot._trader = trader
    monkeypatch.setattr(bot, "load_execution_intent", lambda intent_id: intent)

    result = asyncio.run(bot.buy_entry.__wrapped__(update, context))

    assert result == bot.ConversationHandler.END
    assert target.messages
    kind, text, kwargs = target.messages[0]
    assert kind == "text"
    assert "LIVE EXECUTION" in text
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data.startswith("live_confirm_")


def test_start_bot_threads_only_starts_configured_roles(monkeypatch):
    created = []

    class FakeThread:
        def __init__(self, target=None, args=None, kwargs=None, name=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.name = name
            self.daemon = daemon
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(bot.threading, "Thread", FakeThread)
    trader = _dummy_trader()

    threads = bot.start_bot_threads(
        trader,
        owner_token="owner-token",
        owner_chat_id="123",
        viewer_token=None,
        viewer_chat_ids=None,
    )
    assert set(threads) == {"owner"}
    assert created[0].started is True

    created.clear()
    threads = bot.start_bot_threads(
        trader,
        owner_token="owner-token",
        owner_chat_id="123",
        viewer_token="viewer-token",
        viewer_chat_ids="456,789",
        viewer_startup_chat_id="-10011",
    )
    assert set(threads) == {"owner", "viewer"}
    assert all(thread.started for thread in created)


def test_viewer_authorizes_from_channel_membership():
    app = SimpleNamespace(
        bot_data={"role": "viewer", "allowed_chat_ids": set(), "viewer_channel_id": -1003894501692},
        bot=DummyBot(status="member"),
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=7265587869),
        effective_user=SimpleNamespace(id=7265587869),
    )
    context = SimpleNamespace(application=app)

    allowed = asyncio.run(bot._authorize_viewer_member(update, context))

    assert allowed is True
    assert 7265587869 in app.bot_data["allowed_chat_ids"]
