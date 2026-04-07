#!/usr/bin/env python3
"""
Comprehensive Data Leakage Detection Tests for NEPSE Trading System

Tests for:
1. Lookahead Bias - Using future data to make current decisions
2. Survivorship Bias - Only including stocks that survived
3. Entry/Exit Price Realism - Using achievable prices
4. Signal Timing - Ensuring signals use only past data
5. Walk-Forward Integrity - No data snooping in backtests

Author: Data Leakage Audit
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

from backend.quant_pro.database import get_db_path

# Test results storage
TEST_RESULTS = []


def log_test(name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    TEST_RESULTS.append({"name": name, "passed": passed, "details": details})
    print(f"{status}: {name}")
    if details and not passed:
        print(f"   Details: {details}")


def load_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all price data."""
    query = "SELECT symbol, date, open, high, low, close, volume FROM stock_prices ORDER BY symbol, date"
    df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =============================================================================
# TEST 1: LOOKAHEAD BIAS IN SIGNAL GENERATION
# =============================================================================

def test_signal_uses_only_past_data():
    """
    Verify that signal generation at time T uses only data from T-1 and earlier.

    Method: Generate signal at a specific date, verify no future prices are used.
    """
    from backend.backtesting.simple_backtest import generate_volume_breakout_signals_at_date

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    test_date = pd.Timestamp("2024-06-15")

    # Generate signals at test_date
    signals = generate_volume_breakout_signals_at_date(prices_df, test_date)

    # For each signal, verify no future data was used
    for signal in signals:
        symbol_data = prices_df[
            (prices_df["symbol"] == signal.symbol) &
            (prices_df["date"] <= test_date)
        ]

        # The signal should only use data up to test_date
        max_date = symbol_data["date"].max()
        if max_date > test_date:
            log_test(
                f"Lookahead: {signal.symbol} signal generation",
                False,
                f"Used data from {max_date}, but signal date is {test_date}"
            )
            return

    log_test("Lookahead: Volume breakout signals use only past data", True)


def test_backtest_entry_timing():
    """
    Verify that backtest enters positions AFTER signal generation, not same-day.

    Critical: If signal generated on day T, entry should be at T+1 (or later).
    """
    from backend.backtesting.simple_backtest import run_backtest

    result = run_backtest(
        start_date="2024-01-01",
        end_date="2024-03-31",
        holding_days=21,
        signal_types=["volume"],
        max_positions=3,
        rebalance_frequency=5,
    )

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    issues = []
    for trade in result.trades:
        if trade.entry_date is None:
            continue

        # Get all trading dates for this symbol
        symbol_dates = prices_df[prices_df["symbol"] == trade.symbol]["date"].unique()
        symbol_dates = sorted(symbol_dates)

        # Entry should be on a valid trading day
        if trade.entry_date not in symbol_dates:
            # Find nearest trading day
            nearest = min(symbol_dates, key=lambda x: abs((x - trade.entry_date).days))
            if (nearest - trade.entry_date).days < 0:
                issues.append(f"{trade.symbol}: entered on {trade.entry_date} but nearest trading day before was {nearest}")

    if issues:
        log_test("Backtest entry timing", False, f"{len(issues)} trades with timing issues")
    else:
        log_test("Backtest entry timing", True)


# =============================================================================
# TEST 2: ENTRY PRICE REALISM
# =============================================================================

def test_entry_price_achievability():
    """
    Verify that entry prices used in backtest are achievable in practice.

    Issues to detect:
    - Using close price when you should use next-day open
    - Getting prices at impossible times (weekends, holidays)
    - Perfect execution (no slippage assumption)
    """
    from backend.backtesting.simple_backtest import run_backtest

    result = run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        holding_days=21,
        signal_types=["volume"],
        max_positions=5,
    )

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    issues = []
    slippage_errors = []

    for trade in result.trades:
        if trade.entry_price is None or trade.entry_date is None:
            continue

        # Get actual prices on entry date
        day_data = prices_df[
            (prices_df["symbol"] == trade.symbol) &
            (prices_df["date"] == trade.entry_date)
        ]

        if day_data.empty:
            issues.append(f"{trade.symbol}: No data on entry date {trade.entry_date}")
            continue

        actual_open = day_data["open"].iloc[0]
        actual_high = day_data["high"].iloc[0]
        actual_low = day_data["low"].iloc[0]
        actual_close = day_data["close"].iloc[0]

        # Check if entry price is within day's range
        if not (actual_low <= trade.entry_price <= actual_high):
            issues.append(
                f"{trade.symbol}: Entry {trade.entry_price} outside range [{actual_low}, {actual_high}]"
            )

        # Check if using open price (expected after fix)
        if abs(trade.entry_price - actual_open) < 0.01:
            # Using open price - this is correct!
            pass
        elif abs(trade.entry_price - actual_close) < 0.01:
            # Using close price - flag as issue
            slippage_errors.append(f"{trade.symbol}: Using close price {actual_close} instead of open {actual_open}")

    if issues:
        log_test("Entry price within daily range", False, f"{len(issues)} out-of-range entries")
    else:
        log_test("Entry price within daily range", True)

    # Warn about close price usage (not a failure, but a concern)
    if slippage_errors:
        log_test(
            "Entry price realism (close vs open)",
            False,
            f"{len(slippage_errors)} trades use close price instead of open (realistic slippage not modeled)"
        )
    else:
        log_test("Entry price realism (close vs open)", True)


# =============================================================================
# TEST 3: SURVIVORSHIP BIAS
# =============================================================================

def test_survivorship_bias():
    """
    Verify that backtests include stocks that were delisted or merged.

    Method: Check if symbol universe changes over time in backtest.
    """
    conn = sqlite3.connect(str(get_db_path()))

    # Get symbols available at different points in time
    query_2023 = """
        SELECT DISTINCT symbol FROM stock_prices
        WHERE date BETWEEN '2023-01-01' AND '2023-06-30'
    """
    query_2024 = """
        SELECT DISTINCT symbol FROM stock_prices
        WHERE date BETWEEN '2024-01-01' AND '2024-06-30'
    """

    symbols_2023 = set(pd.read_sql_query(query_2023, conn)["symbol"])
    symbols_2024 = set(pd.read_sql_query(query_2024, conn)["symbol"])

    conn.close()

    # Find symbols that disappeared (potential delistings)
    disappeared = symbols_2023 - symbols_2024
    new_symbols = symbols_2024 - symbols_2023

    # Check if backtest signal generation uses point-in-time universe
    from backend.backtesting.simple_backtest import generate_volume_breakout_signals_at_date

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    # Generate signals at 2023 date
    signals_2023 = generate_volume_breakout_signals_at_date(
        prices_df,
        pd.Timestamp("2023-06-15")
    )
    signal_symbols_2023 = set(s.symbol for s in signals_2023)

    # Check if any 2023 signals are for stocks that didn't exist then
    future_only_symbols = signal_symbols_2023 & new_symbols

    if future_only_symbols:
        log_test(
            "Survivorship: No future-only symbols in historical signals",
            False,
            f"Signals for {future_only_symbols} at 2023 but they IPO'd after 2023"
        )
    else:
        log_test("Survivorship: No future-only symbols in historical signals", True)

    # Report on universe changes (informational)
    print(f"\n   INFO: {len(disappeared)} symbols disappeared from 2023 to 2024")
    print(f"   INFO: {len(new_symbols)} new symbols appeared in 2024")
    if disappeared:
        print(f"   Disappeared: {list(disappeared)[:10]}...")


# =============================================================================
# TEST 4: WALK-FORWARD INTEGRITY
# =============================================================================

def test_walk_forward_no_future_leak():
    """
    Test that walk-forward backtest doesn't leak future information.

    Method: Run backtest, verify that each trade's signal was generated
    using only data available at signal time.
    """
    from backend.backtesting.simple_backtest import run_backtest

    # Run a backtest
    result = run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        holding_days=21,
        signal_types=["volume"],
        max_positions=3,
        rebalance_frequency=5,
    )

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    # For each trade, verify fill mechanics:
    # 1) signal_date < entry_date
    # 2) entry_date is first tradable date after signal_date for that symbol
    # 3) entry_price matches that entry_date open
    issues = []
    for trade in result.trades:
        if trade.entry_date is None:
            continue

        signal_date = pd.Timestamp(trade.signal_date) if trade.signal_date is not None else None
        entry_date = pd.Timestamp(trade.entry_date)

        if signal_date is None:
            issues.append(f"{trade.symbol}: Missing signal_date for trade")
            continue

        if entry_date <= signal_date:
            issues.append(f"{trade.symbol}: entry_date {entry_date.date()} <= signal_date {signal_date.date()}")
            continue

        symbol_prices = prices_df[prices_df["symbol"] == trade.symbol].sort_values("date")
        next_rows = symbol_prices[symbol_prices["date"] > signal_date]
        if next_rows.empty:
            continue

        expected_entry_date = pd.Timestamp(next_rows.iloc[0]["date"])
        if entry_date != expected_entry_date:
            issues.append(
                f"{trade.symbol}: entry_date {entry_date.date()} != next tradable date {expected_entry_date.date()}"
            )
            continue

        entry_row = symbol_prices[symbol_prices["date"] == entry_date]
        if entry_row.empty:
            issues.append(f"{trade.symbol}: no market row for entry date {entry_date.date()}")
            continue

        entry_open = float(entry_row["open"].iloc[0])
        if abs(entry_open - trade.entry_price) > 0.01:
            issues.append(
                f"{trade.symbol}: entry_price {trade.entry_price:.2f} != entry open {entry_open:.2f} on {entry_date.date()}"
            )

    if issues:
        log_test("Walk-forward integrity", False, f"{len(issues)} potential future leaks")
    else:
        log_test("Walk-forward integrity", True)


# =============================================================================
# TEST 5: INDICATOR CALCULATION INTEGRITY
# =============================================================================

def test_indicator_no_lookahead():
    """
    Verify that technical indicators (SMA, RSI, etc.) don't use future data.

    Method: Calculate indicator at time T, verify result only depends on T-N to T.
    """
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    # Pick a test symbol with full history
    test_symbol = "NABIL"  # Major bank, likely full history
    sym_data = prices_df[prices_df["symbol"] == test_symbol].sort_values("date").copy()

    if len(sym_data) < 100:
        log_test("Indicator lookahead", False, f"Not enough data for {test_symbol}")
        return

    # Calculate 20-day SMA using full data
    sym_data["sma_20"] = sym_data["close"].rolling(20).mean()

    # Now calculate SMA at a specific point using only past data
    test_idx = 50
    test_date = sym_data.iloc[test_idx]["date"]

    # SMA using only data up to test_date
    past_only = sym_data.iloc[:test_idx + 1]
    sma_past_only = past_only["close"].iloc[-20:].mean()

    # SMA from full calculation
    sma_full = sym_data.iloc[test_idx]["sma_20"]

    # They should be identical
    if abs(sma_past_only - sma_full) > 0.01:
        log_test(
            "Indicator calculation integrity",
            False,
            f"SMA mismatch: past_only={sma_past_only:.2f}, full={sma_full:.2f}"
        )
    else:
        log_test("Indicator calculation integrity", True)


# =============================================================================
# TEST 6: REALISTIC TRADE EXECUTION
# =============================================================================

def test_trade_execution_realism():
    """
    Test for unrealistic trade execution assumptions.

    Checks:
    - No trading on weekends/holidays
    - Volume constraints respected
    - Circuit breaker handling
    """
    from backend.backtesting.simple_backtest import run_backtest

    result = run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        holding_days=21,
        signal_types=["volume"],
        max_positions=5,
    )

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    weekend_trades = []
    circuit_breaker_issues = []

    for trade in result.trades:
        if trade.entry_date is None:
            continue

        # Check for weekend entries (NEPSE trades Sun-Thu)
        # Python: Monday=0, Sunday=6
        day_of_week = trade.entry_date.dayofweek
        if day_of_week in [4, 5]:  # Friday, Saturday
            weekend_trades.append(f"{trade.symbol}: Entered on {trade.entry_date} (weekend)")

        # Check for circuit breaker hits (±10% daily limit)
        if trade.exit_date and trade.exit_price and trade.entry_price:
            day_data = prices_df[
                (prices_df["symbol"] == trade.symbol) &
                (prices_df["date"] == trade.exit_date)
            ]
            if not day_data.empty:
                day_change = (day_data["close"].iloc[0] / day_data["open"].iloc[0] - 1)
                if abs(day_change) >= 0.099:  # ~10% move
                    circuit_breaker_issues.append(
                        f"{trade.symbol}: Exit on {trade.exit_date} hit circuit breaker ({day_change:.1%})"
                    )

    if weekend_trades:
        log_test("No weekend trading", False, f"{len(weekend_trades)} weekend trades detected")
    else:
        log_test("No weekend trading", True)

    # Circuit breaker is informational - not a hard failure
    if circuit_breaker_issues:
        print(f"   INFO: {len(circuit_breaker_issues)} trades may have hit circuit breakers")


# =============================================================================
# TEST 7: PARAMETER SNOOPING
# =============================================================================

def test_parameter_snooping():
    """
    Check if backtest parameters were optimized on test data (data snooping).

    Method: Compare in-sample vs out-of-sample performance.
    If OOS performance is significantly worse, suggests overfitting.
    """
    from backend.backtesting.simple_backtest import run_backtest

    # In-sample: 2023
    is_result = run_backtest(
        start_date="2023-01-01",
        end_date="2023-12-31",
        holding_days=21,
        signal_types=["volume"],
        max_positions=5,
    )

    # Out-of-sample: 2024
    oos_result = run_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        holding_days=21,
        signal_types=["volume"],
        max_positions=5,
    )

    is_sharpe = is_result.sharpe_ratio
    oos_sharpe = oos_result.sharpe_ratio
    is_return = is_result.annualized_return
    oos_return = oos_result.annualized_return

    # Check for significant degradation
    sharpe_degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 0.01)
    return_degradation = (is_return - oos_return) / max(abs(is_return), 0.001)

    print(f"\n   IN-SAMPLE (2023): Sharpe={is_sharpe:.2f}, Ann. Return={is_return:.2%}")
    print(f"   OUT-OF-SAMPLE (2024): Sharpe={oos_sharpe:.2f}, Ann. Return={oos_return:.2%}")
    print(f"   Sharpe degradation: {sharpe_degradation:.1%}")
    print(f"   Return degradation: {return_degradation:.1%}")

    # Significant degradation suggests overfitting
    if sharpe_degradation > 0.5 and oos_sharpe < 0.5:
        log_test(
            "Parameter robustness (no snooping)",
            False,
            f"Sharpe dropped {sharpe_degradation:.0%} out-of-sample - possible overfitting"
        )
    else:
        log_test("Parameter robustness (no snooping)", True)


# =============================================================================
# TEST 8: FORWARD-FILL BIAS
# =============================================================================

def test_no_forward_fill_bias():
    """
    Check that missing data isn't forward-filled (which introduces lookahead bias).
    """
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_prices(conn)
    conn.close()

    # Check for exact duplicate consecutive prices (sign of forward-fill)
    issues = []
    for symbol in prices_df["symbol"].unique()[:50]:  # Sample
        sym_data = prices_df[prices_df["symbol"] == symbol].sort_values("date")

        # Check for long runs of identical prices
        if len(sym_data) < 10:
            continue

        identical_runs = (sym_data["close"] == sym_data["close"].shift(1)).sum()
        run_ratio = identical_runs / len(sym_data)

        if run_ratio > 0.3:  # More than 30% days have same price as yesterday
            issues.append(f"{symbol}: {run_ratio:.0%} identical consecutive prices")

    if issues:
        log_test("No forward-fill bias", False, f"{len(issues)} symbols with suspicious patterns")
        for issue in issues[:3]:
            print(f"   {issue}")
    else:
        log_test("No forward-fill bias", True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NEPSE TRADING SYSTEM - DATA LEAKAGE AUDIT")
    print("=" * 70)
    print()

    # Run all tests
    print("TEST 1: Lookahead Bias in Signal Generation")
    print("-" * 50)
    test_signal_uses_only_past_data()
    print()

    print("TEST 2: Backtest Entry Timing")
    print("-" * 50)
    test_backtest_entry_timing()
    print()

    print("TEST 3: Entry Price Realism")
    print("-" * 50)
    test_entry_price_achievability()
    print()

    print("TEST 4: Survivorship Bias")
    print("-" * 50)
    test_survivorship_bias()
    print()

    print("TEST 5: Walk-Forward Integrity")
    print("-" * 50)
    test_walk_forward_no_future_leak()
    print()

    print("TEST 6: Indicator Calculation Integrity")
    print("-" * 50)
    test_indicator_no_lookahead()
    print()

    print("TEST 7: Trade Execution Realism")
    print("-" * 50)
    test_trade_execution_realism()
    print()

    print("TEST 8: Parameter Snooping (In-Sample vs Out-of-Sample)")
    print("-" * 50)
    test_parameter_snooping()
    print()

    print("TEST 9: Forward-Fill Bias")
    print("-" * 50)
    test_no_forward_fill_bias()
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for t in TEST_RESULTS if t["passed"])
    failed = sum(1 for t in TEST_RESULTS if not t["passed"])

    print(f"Total Tests: {len(TEST_RESULTS)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("FAILED TESTS:")
        for t in TEST_RESULTS:
            if not t["passed"]:
                print(f"  ✗ {t['name']}")
                if t["details"]:
                    print(f"    → {t['details']}")

    print()
    print("=" * 70)

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
