from datetime import datetime

import pandas as pd

from backend.trading.live_trader import PORTFOLIO_COLS, TRADE_LOG_COLS, Position
from backend.trading.paper_execution import PaperExecutionService
from backend.trading.tui_trading_engine import TUITradingEngine


def _fixed_nst():
    return datetime(2026, 4, 9, 11, 5, 0)


def test_paper_execution_submits_and_fills_buy_in_canonical_account_book(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.trading.paper_execution.current_nepal_datetime", _fixed_nst)
    service = PaperExecutionService(
        "account_1",
        account_dir=tmp_path,
        initial_capital=1_000_000,
        max_positions=5,
        max_order_notional=500_000,
    )

    submitted = service.submit_order(
        "account_1",
        "BUY",
        "NABIL",
        100,
        100.0,
        "strategy_paper",
        "volume",
        strategy_id="s1",
        run_id="run-1",
    )
    matched = service.match_open_orders(
        "account_1",
        {"NABIL": {"ltp": 100.0, "source": "test_quote", "age_seconds": 0}},
    )

    assert submitted.ok
    assert len(matched.filled_orders) == 1
    assert (tmp_path / "paper_portfolio.csv").exists()
    assert (tmp_path / "paper_trade_log.csv").exists()
    assert (tmp_path / "tui_paper_order_history.json").exists()

    portfolio = pd.read_csv(tmp_path / "paper_portfolio.csv")
    trades = pd.read_csv(tmp_path / "paper_trade_log.csv")
    state = service.get_account_execution_state("account_1")

    assert portfolio.loc[0, "Symbol"] == "NABIL"
    assert trades.loc[0, "Action"] == "BUY"
    assert trades.loc[0, "Reason"] == "volume"
    assert state["cash"] < 1_000_000
    assert matched.filled_orders[0].quote_source == "test_quote"


def test_paper_execution_rejects_duplicate_open_order_visibly(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.trading.paper_execution.current_nepal_datetime", _fixed_nst)
    service = PaperExecutionService("account_1", account_dir=tmp_path, max_order_notional=500_000)

    first = service.submit_order("account_1", "BUY", "NABIL", 100, 100.0, "strategy_paper", "volume")
    duplicate = service.submit_order("account_1", "BUY", "NABIL", 100, 100.0, "strategy_paper", "volume")
    state = service.get_account_execution_state("account_1")

    assert first.ok
    assert not duplicate.ok
    assert duplicate.risk_result["reason"] == "duplicate_open_order"
    assert any(row["status"] == "REJECTED" for row in state["order_history"])


def test_paper_execution_migrates_legacy_tui_trade_log_without_duplicates(tmp_path):
    pd.DataFrame(columns=PORTFOLIO_COLS).to_csv(tmp_path / "paper_portfolio.csv", index=False)
    legacy_trade = pd.DataFrame(
        [
            {
                "Date": "2026-04-09",
                "Action": "BUY",
                "Symbol": "NABIL",
                "Shares": 100,
                "Price": 100.0,
                "Fees": 10.0,
                "Reason": "legacy",
                "PnL": 0.0,
                "PnL_Pct": 0.0,
            }
        ],
        columns=TRADE_LOG_COLS,
    )
    legacy_trade.to_csv(tmp_path / "tui_paper_trade_log.csv", index=False)

    PaperExecutionService("account_1", account_dir=tmp_path)
    PaperExecutionService("account_1", account_dir=tmp_path)

    trades = pd.read_csv(tmp_path / "paper_trade_log.csv")
    assert len(trades) == 1
    assert trades.loc[0, "Reason"] == "legacy"


def test_tui_trading_engine_autopilot_writes_canonical_account_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.trading.paper_execution.current_nepal_datetime", _fixed_nst)
    monkeypatch.setattr("backend.trading.tui_trading_engine.now_nst", _fixed_nst)
    monkeypatch.setattr("backend.trading.tui_trading_engine.fetch_prices_for_symbols", lambda symbols: {sym: 100.0 for sym in symbols})

    events = []
    engine = TUITradingEngine(
        capital=1_000_000,
        signal_types=["volume"],
        max_positions=5,
        holding_days=40,
        sector_limit=0.35,
        portfolio_file=tmp_path / "paper_portfolio.csv",
        trade_log_file=tmp_path / "paper_trade_log.csv",
        nav_log_file=tmp_path / "paper_nav_log.csv",
        state_file=tmp_path / "paper_state.json",
        account_id="account_1",
        strategy_id="s1",
        strategy_config={"id": "s1"},
        on_activity=events.append,
    )

    engine._active_run_id = "run-1"
    engine._execute_buys([{"symbol": "NABIL", "signal_type": "volume", "agent_reason": "", "score": 0.9}])

    portfolio = pd.read_csv(tmp_path / "paper_portfolio.csv")
    trades = pd.read_csv(tmp_path / "paper_trade_log.csv")
    assert portfolio.loc[0, "Symbol"] == "NABIL"
    assert trades.loc[0, "Reason"] == "volume"
    assert (tmp_path / "strategy_runs" / "run-1.json").exists()
    assert any("BUY NABIL" in event for event in events)


def test_tui_trading_engine_passes_risk_thresholds_to_exit_check(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.trading.tui_trading_engine.fetch_prices_for_symbols", lambda symbols: {"NABIL": 91.0})
    seen = {}

    def fake_check_exits(positions, holding_days, stop_loss_pct, trailing_stop_pct):
        seen["holding_days"] = holding_days
        seen["stop_loss_pct"] = stop_loss_pct
        seen["trailing_stop_pct"] = trailing_stop_pct
        return []

    monkeypatch.setattr("backend.trading.tui_trading_engine.check_exits", fake_check_exits)
    engine = TUITradingEngine(
        capital=1_000_000,
        holding_days=12,
        stop_loss_pct=0.06,
        trailing_stop_pct=0.11,
        portfolio_file=tmp_path / "paper_portfolio.csv",
        trade_log_file=tmp_path / "paper_trade_log.csv",
        nav_log_file=tmp_path / "paper_nav_log.csv",
        state_file=tmp_path / "paper_state.json",
        account_id="account_1",
    )
    engine.positions = {
        "NABIL": Position(
            symbol="NABIL",
            shares=10,
            entry_price=100.0,
            entry_date="2026-04-01",
            buy_fees=0.0,
            signal_type="volume",
            high_watermark=105.0,
            last_ltp=100.0,
        )
    }

    engine._refresh_and_check_exits()

    assert seen == {
        "holding_days": 12,
        "stop_loss_pct": 0.06,
        "trailing_stop_pct": 0.11,
    }
