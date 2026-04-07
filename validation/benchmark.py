"""
Benchmark comparison: strategy vs NEPSE market proxy.

Since individual NEPSE index data isn't directly in the DB, we construct
an equal-weight proxy index from the median daily return of all traded stocks.
Computes alpha, beta, information ratio, and tracking error.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEPSE_TRADING_DAYS = 240


def compute_benchmark_series(
    prices_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.Series:
    """
    Compute a daily market proxy index from median stock returns.

    Uses the median daily return across all stocks that traded on each day.
    This gives a robust equal-weight market proxy that's resistant to
    outliers and closely tracks the broad NEPSE index.

    Parameters
    ----------
    prices_df : Full price DataFrame (symbol, date, close, ...)
    start : Start date "YYYY-MM-DD"
    end : End date "YYYY-MM-DD"

    Returns
    -------
    pd.Series indexed by date with cumulative benchmark NAV (starting at 1.0)
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    df = prices_df[
        (prices_df["date"] >= start_dt) & (prices_df["date"] <= end_dt)
    ].copy()

    if df.empty:
        return pd.Series(dtype=float)

    # Daily returns per stock
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    # Median return across stocks per day
    daily_median = df.groupby("date")["ret"].median().sort_index()

    # Cumulative NAV
    benchmark_nav = (1 + daily_median.fillna(0)).cumprod()
    benchmark_nav.iloc[0] = 1.0  # Start at 1.0

    return benchmark_nav


def benchmark_comparison(
    daily_nav: List[Tuple[Any, float]],
    benchmark: pd.Series,
    rf_annual: float = 0.0235,
) -> dict:
    """
    Compare strategy performance against a benchmark.

    Computes alpha, beta (via OLS), information ratio, tracking error,
    and excess return.

    Parameters
    ----------
    daily_nav : Strategy daily NAV [(date, nav), ...]
    benchmark : Benchmark cumulative NAV (pd.Series indexed by date)
    rf_annual : Annual risk-free rate (default: 2.35% NRB T-bill)

    Returns
    -------
    dict with alpha, beta, r_squared, information_ratio, tracking_error,
    strategy_cagr, benchmark_cagr, excess_return
    """
    if len(daily_nav) < 10 or len(benchmark) < 10:
        return _empty_result()

    # Align dates
    strat_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
    strat_df.index = pd.to_datetime(strat_df.index)
    benchmark = benchmark.copy()
    benchmark.index = pd.to_datetime(benchmark.index)

    # Inner join on dates
    combined = strat_df.join(benchmark.rename("bench"), how="inner")
    if len(combined) < 10:
        return _empty_result()

    # Daily returns
    strat_ret = combined["nav"].pct_change().dropna()
    bench_ret = combined["bench"].pct_change().dropna()

    # Align after pct_change
    common = strat_ret.index.intersection(bench_ret.index)
    strat_ret = strat_ret.loc[common].values
    bench_ret = bench_ret.loc[common].values
    n = len(strat_ret)

    if n < 10:
        return _empty_result()

    rf_daily = rf_annual / NEPSE_TRADING_DAYS

    # OLS regression: strat_ret - rf = alpha + beta * (bench_ret - rf) + epsilon
    excess_strat = strat_ret - rf_daily
    excess_bench = bench_ret - rf_daily

    # Simple OLS
    x = excess_bench
    y = excess_strat
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.var(x, ddof=0)

    if var_x == 0:
        beta = 0.0
        alpha_daily = float(np.mean(y))
    else:
        beta = float(cov_xy / var_x)
        alpha_daily = float(y_mean - beta * x_mean)

    # R-squared
    y_pred = alpha_daily + beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Annualise alpha
    alpha_annual = alpha_daily * NEPSE_TRADING_DAYS

    # Tracking error and information ratio
    active_returns = strat_ret - bench_ret
    tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(NEPSE_TRADING_DAYS))
    information_ratio = (
        float(np.mean(active_returns) * NEPSE_TRADING_DAYS / tracking_error)
        if tracking_error > 0 else 0.0
    )

    # CAGR
    years = n / NEPSE_TRADING_DAYS
    strat_total = combined["nav"].iloc[-1] / combined["nav"].iloc[0]
    bench_total = combined["bench"].iloc[-1] / combined["bench"].iloc[0]
    strategy_cagr = float(strat_total ** (1 / years) - 1) if years > 0 else 0.0
    benchmark_cagr = float(bench_total ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "alpha": alpha_annual,
        "beta": beta,
        "r_squared": r_squared,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "excess_return": strategy_cagr - benchmark_cagr,
        "n_days": n,
    }


def _empty_result() -> dict:
    return {
        "alpha": 0.0,
        "beta": 0.0,
        "r_squared": 0.0,
        "information_ratio": 0.0,
        "tracking_error": 0.0,
        "strategy_cagr": 0.0,
        "benchmark_cagr": 0.0,
        "excess_return": 0.0,
        "n_days": 0,
    }
