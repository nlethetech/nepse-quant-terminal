"""
NEPSE-specific regime stress tests.

Tests the strategy across known NEPSE market regimes:
- Bull (COVID recovery)
- Crash (post-peak)
- Sideways/recovery
- Recent period

Also analyzes circuit breaker impact and T+2 settlement lag.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Known NEPSE market regime periods
NEPSE_REGIMES: Dict[str, tuple] = {
    "Bull (COVID Recovery)": ("2020-07-01", "2021-08-15"),
    "Crash (Post-Peak)":     ("2021-08-16", "2022-07-31"),
    "Sideways/Recovery":     ("2023-01-01", "2024-06-30"),
    "Recent":                ("2024-07-01", "2025-12-31"),
}


@dataclass
class RegimeResult:
    """Performance in a single regime."""
    name: str
    start_date: str
    end_date: str
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    annualized_return: float


def regime_stress_test(
    regimes: Optional[Dict[str, tuple]] = None,
    max_dd_limit: float = -0.40,
    **backtest_kwargs,
) -> dict:
    """
    Run backtest separately on each NEPSE regime period.

    Parameters
    ----------
    regimes : {name: (start, end)} — defaults to NEPSE_REGIMES
    max_dd_limit : Fail threshold for max drawdown (default -40%)
    **backtest_kwargs : Passed to run_backtest()

    Returns
    -------
    Dict with per-regime results and pass/fail status
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import run_backtest

    if regimes is None:
        regimes = NEPSE_REGIMES

    results: List[RegimeResult] = []
    all_pass = True

    for name, (start, end) in regimes.items():
        logger.info(f"Regime stress: {name} ({start} to {end})")
        kwargs = {**backtest_kwargs}
        kwargs["start_date"] = start
        kwargs["end_date"] = end

        try:
            result = run_backtest(**kwargs)
            rr = RegimeResult(
                name=name,
                start_date=start,
                end_date=end,
                sharpe=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                n_trades=result.total_trades,
                win_rate=result.win_rate,
                annualized_return=result.annualized_return,
            )
            results.append(rr)

            if result.max_drawdown < max_dd_limit:
                logger.warning(
                    f"  FAIL: {name} max_dd={result.max_drawdown:.2%} < limit {max_dd_limit:.2%}"
                )
                all_pass = False
            else:
                logger.info(
                    f"  OK: Sharpe={result.sharpe_ratio:.3f}, "
                    f"Return={result.total_return:.2%}, MaxDD={result.max_drawdown:.2%}"
                )
        except Exception as e:
            logger.error(f"  ERROR in {name}: {e}")
            results.append(RegimeResult(
                name=name, start_date=start, end_date=end,
                sharpe=0.0, total_return=0.0, max_drawdown=-1.0,
                n_trades=0, win_rate=0.0, annualized_return=0.0,
            ))
            all_pass = False

    return {
        "regime_results": results,
        "all_pass": all_pass,
        "worst_regime": min(results, key=lambda r: r.max_drawdown).name if results else "N/A",
        "worst_dd": min(r.max_drawdown for r in results) if results else 0.0,
    }


def circuit_breaker_analysis(trades: list) -> dict:
    """
    Analyze impact of circuit breakers on trade execution.

    Examines trades where exit price was at or near circuit breaker limits,
    indicating the stop loss couldn't fully execute.

    Parameters
    ----------
    trades : List of Trade objects from a backtest

    Returns
    -------
    Dict with circuit breaker impact stats
    """
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "cb_affected_count": 0,
            "cb_affected_pct": 0.0,
            "avg_slippage_from_stop": 0.0,
        }

    cb_affected = 0
    slippage_from_stop = []

    for trade in trades:
        if trade.exit_price is None or trade.entry_price is None:
            continue

        # Check if exit was near a circuit breaker limit (-10% from some reference)
        # Heuristic: if the trade was a stop loss AND the loss is approximately 10%
        if trade.exit_reason == "stop_loss":
            actual_loss = (trade.exit_price - trade.entry_price) / trade.entry_price
            # If the actual loss is very close to -10% (circuit breaker),
            # the stop likely couldn't execute at its target
            if actual_loss <= -0.095:  # Near -10% circuit breaker
                cb_affected += 1
                # Expected stop was at stop_loss_pct, actual was worse
                expected_loss = -0.08  # Default stop loss
                slippage = actual_loss - expected_loss
                slippage_from_stop.append(slippage)

    avg_slippage = float(np.mean(slippage_from_stop)) if slippage_from_stop else 0.0

    return {
        "total_trades": total,
        "cb_affected_count": cb_affected,
        "cb_affected_pct": cb_affected / total * 100 if total > 0 else 0.0,
        "avg_slippage_from_stop": avg_slippage,
    }


def settlement_lag_analysis(
    daily_nav: list,
    n_settlement_days: int = 2,
) -> dict:
    """
    Model T+2 settlement lag impact on capital efficiency.

    In NEPSE, sell proceeds aren't available for 2 trading days.
    This estimates the capital efficiency loss from settlement delays.

    Parameters
    ----------
    daily_nav : List of (date, nav) tuples from backtest
    n_settlement_days : Settlement delay in trading days (T+2)

    Returns
    -------
    Dict with settlement lag impact metrics
    """
    if len(daily_nav) < n_settlement_days + 2:
        return {
            "capital_efficiency_loss_pct": 0.0,
            "max_cash_lockup_pct": 0.0,
            "avg_utilization": 1.0,
        }

    # Estimate: on average, sell proceeds representing X% of NAV are
    # locked up for n_settlement_days. With 5 positions and ~monthly turnover,
    # approximately 1/5 of portfolio turns over per holding period.
    # Rough estimate: locked capital ≈ (positions_turning_over / total_positions)
    #                 * (settlement_days / rebalance_frequency) * NAV
    # For typical config: ~20% of NAV locked ~10% of the time → ~2% efficiency loss

    navs = np.array([nav for _, nav in daily_nav])
    daily_returns = np.diff(navs) / navs[:-1]
    total_return = navs[-1] / navs[0] - 1 if navs[0] > 0 else 0.0

    # Model: shift negative returns (sell days) forward by settlement delay
    # Capital that would have been reinvested immediately is delayed
    n = len(daily_returns)
    delayed_returns = daily_returns.copy()
    for i in range(n):
        if daily_returns[i] > 0.01:  # Large positive day after a sell
            # Check if there was a sell n_settlement_days ago
            look_back = max(0, i - n_settlement_days)
            if any(daily_returns[look_back:i] < -0.005):
                # Reduce the positive capture by settlement delay factor
                delayed_returns[i] *= 0.98  # ~2% reduction

    delayed_total = float(np.prod(1 + delayed_returns) - 1)
    efficiency_loss = total_return - delayed_total if total_return > 0 else 0.0

    return {
        "capital_efficiency_loss_pct": efficiency_loss * 100,
        "original_return": total_return,
        "delayed_return": delayed_total,
        "avg_utilization": 1.0 - efficiency_loss / max(total_return, 0.001),
    }
