from __future__ import annotations

import json
import sys
from datetime import datetime

import pandas as pd

from backend.backtesting.simple_backtest import BacktestResult, Trade, build_close_pivot, run_backtest
from backend.quant_pro.alpha_practical import AlphaSignal, SignalType
from backend.trading import strategy_registry


def test_build_close_pivot_tolerates_duplicate_symbol_dates():
    prices = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-01"), "symbol": "AAA", "close": 100.0},
            {"date": pd.Timestamp("2026-01-01"), "symbol": "AAA", "close": 101.0},
            {"date": pd.Timestamp("2026-01-02"), "symbol": "AAA", "close": 103.0},
        ]
    )

    pivot = build_close_pivot(prices)

    assert float(pivot.loc[pd.Timestamp("2026-01-01"), "AAA"]) == 101.0
    assert float(pivot.loc[pd.Timestamp("2026-01-02"), "AAA"]) == 103.0


def test_run_strategy_backtest_does_not_require_private_temp_runner(monkeypatch, tmp_path):
    sys.modules.pop("run_live_trader_temp_forward_experiments", None)

    result = BacktestResult(
        trades=[
            Trade(
                symbol="AAA",
                entry_date=datetime(2026, 1, 2),
                entry_price=100.0,
                shares=10,
                position_value=1000.0,
                buy_fees=1.0,
                signal_date=datetime(2026, 1, 1),
                exit_date=datetime(2026, 1, 5),
                exit_price=110.0,
                sell_fees=1.0,
                signal_type="quality",
            )
        ],
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 5),
        holding_period=3,
        initial_capital=1000.0,
        daily_nav=[
            (datetime(2026, 1, 1), 1000.0),
            (datetime(2026, 1, 2), 1020.0),
            (datetime(2026, 1, 5), 1100.0),
        ],
    )

    progress_events = []

    def fake_run_backtest(**kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            callback({"phase": "running", "progress_pct": 42, "message": "2/5 days"})
        return result

    monkeypatch.setattr("backend.backtesting.simple_backtest.run_backtest", fake_run_backtest)
    monkeypatch.setattr(strategy_registry, "_nepse_return", lambda *_args: {"return_pct": 5.0})
    monkeypatch.setattr(strategy_registry, "BACKTEST_RESULTS_DIR", tmp_path)

    payload = strategy_registry.run_strategy_backtest(
        {
            "id": "demo",
            "name": "Demo",
            "description": "Demo strategy",
            "runner_mode": "standard",
            "execution_mode": "paper_runtime",
            "config": {"holding_days": 3, "max_positions": 1},
            "ranking_overlay": {"mode": "baseline"},
        },
        start_date="2026-01-01",
        end_date="2026-01-05",
        capital=1000.0,
        progress_callback=progress_events.append,
    )

    assert progress_events == [{"phase": "running", "progress_pct": 42, "message": "2/5 days"}]
    assert payload["summary"]["total_return_pct"] == 10.0
    assert payload["summary"]["sharpe_ratio"] == payload["summary"]["sharpe"]
    assert payload["summary"]["trade_count"] == payload["summary"]["total_trades"]
    assert payload["summary"]["vs_nepse_pct_points"] == 5.0
    assert payload["summary"]["daily_nav"][-1] == ["2026-01-05", 1100.0]
    assert (tmp_path / "demo_latest.json").exists()

    latest = strategy_registry.load_strategy_backtest_result("demo")
    assert latest is not None
    metrics = strategy_registry.comparison_metrics_for_strategy("demo")
    assert metrics is not None
    assert metrics["total_return_pct"] == 10.0
    assert metrics["vs_nepse_pct_points"] == 5.0
    assert metrics["trade_count"] == metrics["total_trades"]
    assert metrics["sharpe_ratio"] == metrics["sharpe"]


def test_load_strategy_chart_result_uses_comparison_snapshot_when_latest_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(strategy_registry, "BACKTEST_RESULTS_DIR", tmp_path)
    comparison_path = tmp_path / "registry_strategies_vs_nepse_latest.json"
    monkeypatch.setattr(strategy_registry, "COMPARISON_LATEST_JSON", comparison_path)
    comparison_path.write_text(
        json.dumps(
            {
                "window": {"start": "2026-01-01", "end": "2026-01-08", "capital": 1000.0},
                "nepse": {"return_pct": 2.0},
                "generated_at": "2026-01-08T12:00:00",
                "strategies": [
                    {
                        "id": "demo",
                        "name": "Demo",
                        "summary": {
                            "total_return_pct": 5.0,
                            "daily_nav": [
                                ["2026-01-01", 1000.0],
                                ["2026-01-02", 1010.0],
                                ["2026-01-05", 1020.0],
                                ["2026-01-06", 1030.0],
                                ["2026-01-07", 1040.0],
                                ["2026-01-08", 1050.0],
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = strategy_registry.load_strategy_chart_result("demo")

    assert payload is not None
    assert payload["strategy"]["id"] == "demo"
    assert payload["window"]["start"] == "2026-01-01"
    assert payload["nepse"]["return_pct"] == 2.0
    assert payload["summary"]["daily_nav"][-1] == ["2026-01-08", 1050.0]


def test_load_strategy_chart_result_ignores_metric_only_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(strategy_registry, "BACKTEST_RESULTS_DIR", tmp_path)
    comparison_path = tmp_path / "registry_strategies_vs_nepse_latest.json"
    monkeypatch.setattr(strategy_registry, "COMPARISON_LATEST_JSON", comparison_path)
    comparison_path.write_text(
        json.dumps(
            {
                "window": {"start": "2026-01-01", "end": "2026-01-08"},
                "strategies": [
                    {"id": "demo", "name": "Demo", "summary": {"total_return_pct": 5.0}},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert strategy_registry.load_strategy_chart_result("demo") is None


def test_run_backtest_broker_exit_uses_public_db_path(monkeypatch, tmp_path):
    db_path = tmp_path / "market.db"
    conn = __import__("sqlite3").connect(db_path)
    conn.execute(
        "CREATE TABLE stock_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("backend.backtesting.simple_backtest.get_db_path", lambda: db_path)

    result = run_backtest(
        start_date="2026-01-01",
        end_date="2026-01-05",
        signal_types=["volume"],
        use_broker_exit=True,
    )

    assert result.trades == []
    assert result.daily_nav == []


def test_regime_adaptive_hold_uses_current_regime_and_trade_hold(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    rows = []
    for symbol in ["AAA", *[f"S{i:02d}" for i in range(25)]]:
        for idx, date in enumerate(dates):
            price = 100.0 + idx
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000.0,
                }
            )
    prices = pd.DataFrame(rows)

    monkeypatch.setattr("backend.backtesting.simple_backtest.load_all_prices", lambda _conn: prices)
    monkeypatch.setattr("backend.backtesting.simple_backtest.compute_market_regime", lambda *_args, **_kwargs: "bear")
    monkeypatch.setattr("backend.backtesting.simple_backtest.get_symbol_sector", lambda _symbol: None)

    def _signals(_prices_df, current_date, **_kwargs):
        if pd.Timestamp(current_date) == dates[0]:
            return [
                AlphaSignal(
                    symbol="AAA",
                    signal_type=SignalType.LIQUIDITY,
                    direction=1,
                    strength=1.0,
                    confidence=1.0,
                    reasoning="test signal",
                )
            ]
        return []

    monkeypatch.setattr("backend.backtesting.simple_backtest.generate_volume_breakout_signals_at_date", _signals)

    result = run_backtest(
        start_date="2026-01-01",
        end_date="2026-01-25",
        holding_days=12,
        max_positions=1,
        signal_types=["volume"],
        initial_capital=1_000_000,
        rebalance_frequency=99,
        use_regime_filter=True,
        regime_max_positions={"bear": 1},
        regime_adaptive_hold=True,
        regime_hold_days={"bear": 3, "neutral": 6, "bull": 12},
        use_trailing_stop=False,
        stop_loss_pct=0.99,
    )

    assert result.completed_trades
    assert result.completed_trades[0].max_holding_days == 3
    assert result.completed_trades[0].exit_reason == "holding_period"
