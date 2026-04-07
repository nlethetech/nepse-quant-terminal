"""
Slippage and liquidity model for NEPSE stocks.

Models realistic execution costs for illiquid NEPSE stocks based on
order size relative to average daily volume. Provides:
- Per-trade slippage estimation
- Post-hoc slippage adjustment to backtest results
- Liquidity filter for the tradeable universe
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Slippage model parameters ───────────────────────────────────────────
# Calibrated for NEPSE's retail-dominated, relatively illiquid market
LIQUID_THRESHOLD_NPR = 500_000    # Daily turnover threshold for "liquid"
BASE_SPREAD_LIQUID = 0.003        # 0.3% base spread for liquid stocks
BASE_SPREAD_ILLIQUID = 0.008      # 0.8% base spread for illiquid stocks
IMPACT_FACTOR_LIQUID = 0.005      # 0.5% impact factor for liquid stocks
IMPACT_FACTOR_ILLIQUID = 0.015    # 1.5% impact factor for illiquid stocks

NEPSE_TRADING_DAYS = 240


def estimate_slippage(
    shares: int,
    price: float,
    avg_daily_volume: float,
    avg_daily_turnover: float,
    side: str = "buy",
) -> float:
    """
    Estimate slippage for a single trade.

    Model: slippage_pct = base_spread + impact_factor * (shares / avg_volume)^0.5

    Parameters
    ----------
    shares : Number of shares in the order
    price : Expected execution price
    avg_daily_volume : Average daily volume (shares) for this symbol
    avg_daily_turnover : Average daily turnover (NPR) for this symbol
    side : "buy" or "sell"

    Returns
    -------
    Estimated slippage as a decimal (e.g., 0.005 = 0.5%)
    """
    if avg_daily_volume <= 0 or shares <= 0:
        return 0.0

    is_liquid = avg_daily_turnover >= LIQUID_THRESHOLD_NPR
    base = BASE_SPREAD_LIQUID if is_liquid else BASE_SPREAD_ILLIQUID
    impact = IMPACT_FACTOR_LIQUID if is_liquid else IMPACT_FACTOR_ILLIQUID

    # Market impact: square root model
    participation_rate = shares / avg_daily_volume
    market_impact = impact * (participation_rate ** 0.5)

    slippage = base + market_impact

    # Sell side tends to have slightly higher slippage (urgency premium)
    if side == "sell":
        slippage *= 1.1

    return slippage


def compute_volume_stats(
    prices_df: pd.DataFrame,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Compute average daily volume and turnover per symbol.

    Parameters
    ----------
    prices_df : Full price data
    lookback_days : Number of trading days to average over

    Returns
    -------
    DataFrame with columns: symbol, avg_volume, avg_turnover
    """
    df = prices_df.copy()
    df["turnover"] = df["close"] * df["volume"]

    # Get the last `lookback_days` trading days for each symbol
    latest = df.groupby("symbol").tail(lookback_days)
    stats = latest.groupby("symbol").agg(
        avg_volume=("volume", "mean"),
        avg_turnover=("turnover", "mean"),
    ).reset_index()

    return stats


def run_backtest_with_slippage(
    trades: list,
    prices_df: pd.DataFrame,
    initial_capital: float = 1_000_000,
    daily_nav: Optional[List[Tuple[Any, float]]] = None,
    lookback_days: int = 60,
) -> dict:
    """
    Post-hoc slippage adjustment to backtest trades.

    Recomputes P&L after applying slippage to entry and exit prices.

    Parameters
    ----------
    trades : List of Trade objects from a backtest
    prices_df : Full price data for volume computation
    initial_capital : Starting capital
    daily_nav : Original daily NAV (for comparison)
    lookback_days : Days to average volume over

    Returns
    -------
    Dict with:
        original_sharpe, adjusted_sharpe, sharpe_impact,
        original_return, adjusted_return, return_impact,
        total_slippage_cost, avg_slippage_per_trade,
        n_trades_affected
    """
    vol_stats = compute_volume_stats(prices_df, lookback_days)
    vol_lookup = {
        row["symbol"]: (row["avg_volume"], row["avg_turnover"])
        for _, row in vol_stats.iterrows()
    }

    # Compute original metrics from daily NAV
    if daily_nav and len(daily_nav) >= 2:
        navs = np.array([nav for _, nav in daily_nav])
        daily_rets = np.diff(navs) / navs[:-1]
        orig_sharpe = (
            float(np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(NEPSE_TRADING_DAYS))
            if np.std(daily_rets) > 0 else 0.0
        )
        orig_return = float(navs[-1] / navs[0] - 1)
    else:
        orig_sharpe = 0.0
        orig_return = 0.0

    total_slippage_cost = 0.0
    adjusted_pnls = []
    original_pnls = []
    n_affected = 0

    for trade in trades:
        if trade.exit_price is None or trade.net_return is None:
            continue

        avg_vol, avg_to = vol_lookup.get(trade.symbol, (0, 0))

        # Buy slippage: pay more
        buy_slip = estimate_slippage(
            trade.shares, trade.entry_price, avg_vol, avg_to, "buy"
        )
        adj_entry = trade.entry_price * (1 + buy_slip)

        # Sell slippage: receive less
        sell_slip = estimate_slippage(
            trade.shares, trade.exit_price, avg_vol, avg_to, "sell"
        )
        adj_exit = trade.exit_price * (1 - sell_slip)

        orig_pnl = (trade.exit_price - trade.entry_price) * trade.shares
        adj_pnl = (adj_exit - adj_entry) * trade.shares
        slip_cost = orig_pnl - adj_pnl

        original_pnls.append(orig_pnl)
        adjusted_pnls.append(adj_pnl)
        total_slippage_cost += slip_cost

        if buy_slip > 0 or sell_slip > 0:
            n_affected += 1

    # Recompute equity curve with slippage adjustments
    n_trades = len(adjusted_pnls)
    if n_trades > 0 and initial_capital > 0:
        adj_returns = np.array(adjusted_pnls) / initial_capital * len(adjusted_pnls)
        orig_returns = np.array(original_pnls) / initial_capital * len(original_pnls)

        adj_total_return = orig_return - total_slippage_cost / initial_capital

        # Rough Sharpe adjustment
        adj_sharpe = orig_sharpe * (1 + adj_total_return) / (1 + orig_return) if orig_return > -1 else 0.0
    else:
        adj_total_return = orig_return
        adj_sharpe = orig_sharpe

    return {
        "original_sharpe": orig_sharpe,
        "adjusted_sharpe": adj_sharpe,
        "sharpe_impact": adj_sharpe - orig_sharpe,
        "original_return": orig_return,
        "adjusted_return": adj_total_return,
        "return_impact": adj_total_return - orig_return,
        "total_slippage_cost": total_slippage_cost,
        "avg_slippage_per_trade": total_slippage_cost / n_trades if n_trades > 0 else 0.0,
        "n_trades": n_trades,
        "n_trades_affected": n_affected,
    }


def liquidity_filter(
    prices_df: pd.DataFrame,
    position_size: float,
    max_volume_pct: float = 0.10,
    lookback_days: int = 60,
) -> List[str]:
    """
    Return symbols that pass the liquidity filter.

    A stock passes if the position size is less than max_volume_pct
    of its average daily turnover.

    Parameters
    ----------
    prices_df : Full price data
    position_size : Target position size in NPR
    max_volume_pct : Maximum fraction of avg daily volume (default 10%)
    lookback_days : Days to average volume over

    Returns
    -------
    List of symbols that pass the filter
    """
    vol_stats = compute_volume_stats(prices_df, lookback_days)

    passed = []
    for _, row in vol_stats.iterrows():
        if row["avg_turnover"] > 0:
            pct_of_daily = position_size / row["avg_turnover"]
            if pct_of_daily <= max_volume_pct:
                passed.append(row["symbol"])

    logger.info(
        f"Liquidity filter: {len(passed)}/{len(vol_stats)} symbols pass "
        f"(position_size={position_size:,.0f}, max_vol_pct={max_volume_pct:.0%})"
    )
    return passed
