"""Unit tests for live trader accounting and benchmark helpers."""

from __future__ import annotations

import threading

import pandas as pd
import pytest

from backend.trading import live_trader
from backend.trading.live_trader import (
    LiveTrader,
    Position,
    calculate_cash_from_trade_log,
    compute_portfolio_intelligence,
    compute_portfolio_vs_nepse,
    compute_risk_snapshot,
    compute_sector_attribution,
    compute_strategy_attribution,
    estimate_execution_price,
    load_portfolio,
    resolve_daily_start_nav,
    save_portfolio,
)


def test_calculate_cash_from_trade_log(tmp_path):
    trade_log = tmp_path / "paper_trade_log.csv"
    trade_log.write_text(
        "\n".join(
            [
                "Date,Action,Symbol,Shares,Price,Fees,Reason,PnL,PnL_Pct",
                "2026-02-09,BUY,AAA,100,10,1,buy,0,0",
                "2026-02-10,SELL,AAA,100,12,1,sell,198,0.18",
                "2026-02-11,BUY,BBB,50,20,1,buy,0,0",
            ]
        ),
        encoding="utf-8",
    )

    cash = calculate_cash_from_trade_log(1000.0, str(trade_log))
    assert cash == 197.0


def test_compute_portfolio_vs_nepse(monkeypatch, tmp_path):
    trade_log = tmp_path / "paper_trade_log.csv"
    trade_log.write_text(
        "\n".join(
            [
                "Date,Action,Symbol,Shares,Price,Fees,Reason,PnL,PnL_Pct",
                "2026-02-09,BUY,AAA,100,10,1,buy,0,0",
                "2026-02-10,SELL,AAA,100,12,1,sell,198,0.18",
                "2026-02-11,BUY,BBB,50,20,1,buy,0,0",
            ]
        ),
        encoding="utf-8",
    )

    positions = {
        "BBB": Position(
            symbol="BBB",
            shares=50,
            entry_price=20.0,
            entry_date="2026-02-11",
            buy_fees=1.0,
            signal_type="manual",
            high_watermark=23.0,
            last_ltp=24.0,
        )
    }

    benchmark_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-02-09", "2026-03-26"]),
            "Open": [100.0, 109.0],
            "High": [100.0, 110.0],
            "Low": [100.0, 109.0],
            "Close": [100.0, 110.0],
            "Volume": [1.0, 1.0],
        }
    )

    monkeypatch.setattr("backend.trading.live_trader.ensure_benchmark_history", lambda *args, **kwargs: benchmark_df)

    comparison = compute_portfolio_vs_nepse(1000.0, positions, str(trade_log))

    assert comparison is not None
    assert round(comparison["performance"]["deployed_return_pct"], 4) == 39.6603
    assert round(comparison["benchmark"]["return_pct"], 4) == 10.0
    assert round(comparison["alpha_pct"], 4) == 29.6603


def test_portfolio_round_trip_preserves_mark_source(tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    positions = {
        "AAA": Position(
            symbol="AAA",
            shares=10,
            entry_price=100.0,
            entry_date="2026-02-09",
            buy_fees=5.0,
            signal_type="fundamental",
            high_watermark=111.0,
            last_ltp=110.0,
            last_ltp_source="merolagani",
        )
    }

    save_portfolio(positions, str(portfolio_path))
    restored = load_portfolio(str(portfolio_path))

    assert restored["AAA"].last_ltp == 110.0
    assert restored["AAA"].last_ltp_source == "merolagani"


def test_resolve_daily_start_nav_uses_prior_nav(monkeypatch, tmp_path):
    nav_log = tmp_path / "paper_nav_log.csv"
    nav_log.write_text(
        "\n".join(
            [
                "Date,Cash,Positions_Value,NAV,Num_Positions",
                "2026-03-27,400000,600000,1000000,4",
                "2026-03-28,410000,610000,1020000,4",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeNow:
        @staticmethod
        def date():
            from datetime import date
            return date(2026, 3, 29)

    monkeypatch.setattr("backend.trading.live_trader.now_nst", lambda: _FakeNow())

    baseline = resolve_daily_start_nav(str(nav_log), fallback_nav=999999.0)

    assert baseline == 1_020_000.0


def test_strategy_attribution_and_risk_snapshot(tmp_path):
    trade_log = tmp_path / "paper_trade_log.csv"
    trade_log.write_text(
        "\n".join(
            [
                "Date,Action,Symbol,Shares,Price,Fees,Reason,PnL,PnL_Pct",
                "2026-02-09,BUY,AAA,100,10,1,fundamental,0,0",
                "2026-02-10,SELL,AAA,100,12,1,take_profit,198,0.18",
                "2026-02-11,BUY,BBB,50,20,1,fundamental,0,0",
                "2026-02-12,BUY,CCC,20,30,1,manual,0,0",
            ]
        ),
        encoding="utf-8",
    )
    nav_log = tmp_path / "paper_nav_log.csv"
    nav_log.write_text(
        "\n".join(
            [
                "Date,Cash,Positions_Value,NAV,Num_Positions",
                "2026-02-10,1200,0,1200,0",
                "2026-02-12,500,1820,2320,2",
            ]
        ),
        encoding="utf-8",
    )

    positions = {
        "BBB": Position(
            symbol="BBB",
            shares=50,
            entry_price=20.0,
            entry_date="2026-02-11",
            buy_fees=1.0,
            signal_type="fundamental",
            high_watermark=25.0,
            last_ltp=24.0,
        ),
        "CCC": Position(
            symbol="CCC",
            shares=20,
            entry_price=30.0,
            entry_date="2026-02-12",
            buy_fees=1.0,
            signal_type="manual",
            high_watermark=31.0,
            last_ltp=28.0,
        ),
    }

    attribution = compute_strategy_attribution(positions, str(trade_log))

    fundamental = next(row for row in attribution if row["strategy"] == "fundamental")
    manual = next(row for row in attribution if row["strategy"] == "manual")
    assert round(fundamental["realized_pnl"], 2) == 198.0
    assert round(fundamental["unrealized_pnl"], 2) == 199.0
    assert round(manual["unrealized_pnl"], 2) == -41.0

    risk = compute_risk_snapshot(2000.0, 500.0, positions, str(nav_log))
    assert risk["top_positions"][0]["symbol"] == "BBB"
    assert round(risk["drawdown_pct"], 2) == -2.59


def test_sector_attribution_includes_sector_benchmark(monkeypatch):
    monkeypatch.setattr("backend.trading.live_trader.get_symbol_sector", lambda symbol: "Finance")
    monkeypatch.setattr(
        "backend.trading.live_trader.compute_benchmark_return",
        lambda benchmark, start_date, end_date=None: {
            "benchmark": benchmark,
            "return_pct": 5.0,
            "base_date": start_date,
            "base_close": 100.0,
            "latest_date": start_date,
            "latest_close": 105.0,
        },
    )

    positions = {
        "AAA": Position(
            symbol="AAA",
            shares=10,
            entry_price=100.0,
            entry_date="2026-02-09",
            buy_fees=0.0,
            signal_type="fundamental",
            high_watermark=110.0,
            last_ltp=110.0,
        )
    }

    rows = compute_sector_attribution(positions)

    assert rows[0]["sector"] == "Finance"
    assert rows[0]["benchmark"]["benchmark"] == "FINANCE"
    assert round(rows[0]["alpha_pct"], 2) == 5.0


def test_compute_portfolio_intelligence_builds_quality_and_attribution(monkeypatch, tmp_path):
    trade_log = tmp_path / "paper_trade_log.csv"
    trade_log.write_text(
        "\n".join(
            [
                "Date,Action,Symbol,Shares,Price,Fees,Reason,PnL,PnL_Pct",
                "2026-02-09,BUY,AAA,100,10,1,fundamental,0,0",
                "2026-02-10,SELL,AAA,100,12,1,take_profit,198,0.18",
                "2026-02-11,BUY,BBB,50,20,1,fundamental,0,0",
            ]
        ),
        encoding="utf-8",
    )
    nav_log = tmp_path / "paper_nav_log.csv"
    nav_log.write_text(
        "\n".join(
            [
                "Date,Cash,Positions_Value,NAV,Num_Positions",
                "2026-02-10,1198,0,1198,0",
                "2026-02-11,197,1200,1397,1",
            ]
        ),
        encoding="utf-8",
    )

    positions = {
        "BBB": Position(
            symbol="BBB",
            shares=50,
            entry_price=20.0,
            entry_date="2026-02-11",
            buy_fees=1.0,
            signal_type="fundamental",
            high_watermark=24.0,
            last_ltp=24.0,
            last_ltp_source="nepalstock",
            last_ltp_time_utc="2026-03-26T00:00:00+00:00",
        )
    }

    benchmark_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-02-09", "2026-02-11", "2026-03-26"]),
            "Open": [100.0, 101.0, 110.0],
            "High": [100.0, 101.0, 110.0],
            "Low": [100.0, 101.0, 110.0],
            "Close": [100.0, 101.0, 110.0],
            "Volume": [1.0, 1.0, 1.0],
            "Fetched_At": ["2026-03-26T00:00:00+00:00"] * 3,
        }
    )

    monkeypatch.setattr("backend.trading.live_trader.ensure_benchmark_history", lambda *args, **kwargs: benchmark_df)
    monkeypatch.setattr("backend.trading.live_trader.get_symbol_sector", lambda symbol: "Unknown")

    intelligence = compute_portfolio_intelligence(
        1000.0,
        197.0,
        positions,
        str(trade_log),
        str(nav_log),
    )

    assert intelligence is not None
    assert intelligence["data_quality"]["confidence_score"] > 0
    assert "stock_selection_alpha_pnl" in intelligence["attribution_stack"]
    assert intelligence["performance"]["realized_pnl"] == 198.0
    assert intelligence["attribution_stack"]["realized_alpha_pnl"] == pytest.approx(200.0)
    assert intelligence["attribution_stack"]["gross_alpha_return_pct"] > 0
    assert intelligence["positions"][0]["symbol"] == "BBB"


def test_ensure_benchmark_history_caches_no_data(monkeypatch):
    live_trader._BENCHMARK_NO_DATA_CACHE.clear()
    calls = {"fetch": 0}

    monkeypatch.setattr("backend.trading.live_trader.load_benchmark_history", lambda *args, **kwargs: pd.DataFrame())

    def _fetch(*args, **kwargs):
        calls["fetch"] += 1
        return pd.DataFrame()

    monkeypatch.setattr("backend.trading.live_trader.fetch_ohlcv_chunk", _fetch)

    start = pd.Timestamp("2026-03-01").date()
    end = pd.Timestamp("2026-04-02").date()

    first = live_trader.ensure_benchmark_history("MANUFACTURING", start, end)
    second = live_trader.ensure_benchmark_history("MANUFACTURING", start, end)

    assert first.empty
    assert second.empty
    assert calls["fetch"] == 1


def test_execute_buy_signals_backfills_past_sector_blocked_names(monkeypatch, tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    trade_log_path = tmp_path / "paper_trade_log.csv"
    state_path = tmp_path / "paper_state.json"

    held = {
        "MFIL": Position("MFIL", 178, 800.0, "2026-02-09", 516.28, "fundamental", 870.0, last_ltp=822.0),
        "SBL": Position("SBL", 375, 380.0, "2026-02-09", 516.625, "fundamental", 415.1, last_ltp=399.6),
        "AKJCL": Position("AKJCL", 393, 363.0, "2026-02-09", 517.17355, "fundamental", 419.8, last_ltp=399.2),
        "UMHL": Position("UMHL", 225, 633.2, "2026-03-25", 610.5517, "fundamental", 640.0, last_ltp=617.0),
    }

    class DummyTrader:
        def __init__(self):
            self._state_lock = threading.RLock()
            self.max_positions = 7
            self.positions = dict(held)
            self.capital = 1_000_000.0
            self.cash = 456_870.5
            self.live_execution_enabled = False
            self.dry_run = False
            self.portfolio_file = str(portfolio_path)
            self.trade_log_file = str(trade_log_path)
            self.last_price_snapshot_time_utc = "2026-03-30T09:10:29+00:00"
            self.activity = []

        def calculate_nav(self):
            return self.cash + sum(pos.market_value for pos in self.positions.values())

        def _persist_runtime_state(self):
            state_path.write_text("{}", encoding="utf-8")

        def _log_activity(self, message):
            self.activity.append(message)

        def _send_telegram(self, *args, **kwargs):
            return None

        def _rank_entry_signals(self, signals, *, as_of_date=None):
            return list(signals)

    trader = DummyTrader()

    monkeypatch.setattr(
        "backend.trading.live_trader.fetch_prices_for_symbols",
        lambda symbols: {"API": 357.6, "GLH": 286.0, "SHPC": 541.0, "SMH": 621.0},
    )
    monkeypatch.setattr("backend.trading.live_trader.get_latest_ltps_context", lambda: {"source_map": {}, "timestamp_map": {}})
    monkeypatch.setattr(
        "backend.trading.live_trader.get_symbol_sector",
        lambda symbol: {
            "MFIL": "Finance",
            "SBL": "Commercial Banks",
            "AKJCL": "Hydropower",
            "UMHL": "Hydropower",
            "API": "Hydropower",
            "GLH": "Hydropower",
            "SHPC": "Hydropower",
            "SMH": None,
        }.get(symbol),
    )

    signals = [
        {"symbol": "API", "signal_type": "fundamental", "score": 0.25},
        {"symbol": "GLH", "signal_type": "fundamental", "score": 0.24},
        {"symbol": "SHPC", "signal_type": "fundamental", "score": 0.23},
        {"symbol": "SMH", "signal_type": "fundamental", "score": 0.09},
    ]

    LiveTrader._execute_buy_signals(trader, signals)

    assert "SMH" in trader.positions
    assert trader.positions["SMH"].shares > 0
    assert any("BUY SMH" in entry for entry in trader.activity)


def test_execute_buy_signals_uses_estimated_fill_price(monkeypatch, tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    trade_log_path = tmp_path / "paper_trade_log.csv"
    state_path = tmp_path / "paper_state.json"

    class DummyTrader:
        def __init__(self):
            self._state_lock = threading.RLock()
            self.max_positions = 5
            self.positions = {}
            self.capital = 100_000.0
            self.cash = 100_000.0
            self.live_execution_enabled = False
            self.dry_run = False
            self.portfolio_file = str(portfolio_path)
            self.trade_log_file = str(trade_log_path)
            self.last_price_snapshot_time_utc = "2026-03-31T04:00:00+00:00"
            self.activity = []

        def calculate_nav(self):
            return self.cash + sum(pos.market_value for pos in self.positions.values())

        def _persist_runtime_state(self):
            state_path.write_text("{}", encoding="utf-8")

        def _log_activity(self, message):
            self.activity.append(message)

        def _send_telegram(self, *args, **kwargs):
            return None

        def _rank_entry_signals(self, signals, *, as_of_date=None):
            return list(signals)

    trader = DummyTrader()

    monkeypatch.setattr("backend.trading.live_trader.fetch_prices_for_symbols", lambda symbols: {"AAA": 100.0})
    monkeypatch.setattr(
        "backend.trading.live_trader.get_latest_ltps_context",
        lambda: {"source_map": {"AAA": "test_feed"}, "timestamp_map": {"AAA": "2026-03-31T04:00:00+00:00"}},
    )
    monkeypatch.setattr("backend.trading.live_trader.get_symbol_sector", lambda symbol: None)

    LiveTrader._execute_buy_signals(
        trader,
        [{"symbol": "AAA", "signal_type": "fundamental", "score": 0.42}],
    )

    pos = trader.positions["AAA"]
    expected_fill = estimate_execution_price(100.0, is_buy=True)

    assert pos.entry_price == expected_fill
    assert pos.last_ltp == 100.0
    assert pos.entry_price > pos.last_ltp

    trade_log = pd.read_csv(trade_log_path)
    assert trade_log.loc[0, "Price"] == expected_fill


def test_check_and_execute_exits_uses_estimated_sell_price(monkeypatch, tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    trade_log_path = tmp_path / "paper_trade_log.csv"
    state_path = tmp_path / "paper_state.json"

    class DummyTrader:
        def __init__(self):
            self._state_lock = threading.RLock()
            self.holding_days = 40
            self.positions = {
                "AAA": Position(
                    symbol="AAA",
                    shares=100,
                    entry_price=100.0,
                    entry_date="2026-03-01",
                    buy_fees=25.0,
                    signal_type="fundamental",
                    high_watermark=125.0,
                    last_ltp=120.0,
                )
            }
            self.cash = 5_000.0
            self.live_execution_enabled = False
            self.dry_run = False
            self.trade_log_file = str(trade_log_path)
            self.portfolio_file = str(portfolio_path)
            self.activity = []
            self.consecutive_losses = 0

        def _persist_runtime_state(self):
            state_path.write_text("{}", encoding="utf-8")

        def _log_activity(self, message):
            self.activity.append(message)

        def _send_telegram(self, *args, **kwargs):
            return None

    trader = DummyTrader()

    monkeypatch.setattr("backend.trading.live_trader.check_exits", lambda positions, holding_days: [("AAA", "holding_period")])

    LiveTrader.check_and_execute_exits(trader)

    expected_fill = estimate_execution_price(120.0, is_buy=False)

    assert "AAA" not in trader.positions
    trade_log = pd.read_csv(trade_log_path)
    assert trade_log.loc[0, "Price"] == expected_fill
    assert trade_log.loc[0, "Price"] < 120.0


def test_execute_manual_buy_uses_estimated_fill_price(monkeypatch, tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    trade_log_path = tmp_path / "paper_trade_log.csv"
    state_path = tmp_path / "paper_state.json"

    class DummyTrader:
        def __init__(self):
            self.positions = {}
            self.max_positions = 5
            self.cash = 100_000.0
            self.dry_run = False
            self.portfolio_file = str(portfolio_path)
            self.trade_log_file = str(trade_log_path)
            self.last_price_snapshot_time_utc = "2026-03-31T04:00:00+00:00"
            self.activity = []

        def _persist_runtime_state(self):
            state_path.write_text("{}", encoding="utf-8")

        def _log_activity(self, message):
            self.activity.append(message)

    trader = DummyTrader()

    monkeypatch.setattr(
        "backend.trading.live_trader.get_latest_ltps_context",
        lambda: {"source_map": {"AAA": "manual_test"}, "timestamp_map": {"AAA": "2026-03-31T04:00:00+00:00"}},
    )

    success, _ = LiveTrader.execute_manual_buy(trader, "AAA", 100, 100.0)

    expected_fill = estimate_execution_price(100.0, is_buy=True)

    assert success is True
    assert trader.positions["AAA"].entry_price == expected_fill
    assert trader.positions["AAA"].last_ltp == 100.0
    trade_log = pd.read_csv(trade_log_path)
    assert trade_log.loc[0, "Price"] == expected_fill


def test_execute_manual_sell_uses_estimated_fill_price(tmp_path):
    portfolio_path = tmp_path / "paper_portfolio.csv"
    trade_log_path = tmp_path / "paper_trade_log.csv"
    state_path = tmp_path / "paper_state.json"

    class DummyTrader:
        def __init__(self):
            self.positions = {
                "AAA": Position(
                    symbol="AAA",
                    shares=100,
                    entry_price=100.0,
                    entry_date="2026-03-01",
                    buy_fees=25.0,
                    signal_type="manual",
                    high_watermark=120.0,
                    last_ltp=115.0,
                )
            }
            self.cash = 10_000.0
            self.dry_run = False
            self.portfolio_file = str(portfolio_path)
            self.trade_log_file = str(trade_log_path)
            self.activity = []

        def _persist_runtime_state(self):
            state_path.write_text("{}", encoding="utf-8")

        def _log_activity(self, message):
            self.activity.append(message)

    trader = DummyTrader()

    success, _ = LiveTrader.execute_manual_sell(trader, "AAA")

    expected_fill = estimate_execution_price(115.0, is_buy=False)

    assert success is True
    assert "AAA" not in trader.positions
    trade_log = pd.read_csv(trade_log_path)
    assert trade_log.loc[0, "Price"] == expected_fill
