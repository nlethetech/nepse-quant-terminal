"""
Fast random entry baseline using pre-computed numpy matrices.

Replaces per-row DataFrame lookups with O(1) numpy indexing.
Pre-computes: price matrices, regime array, circuit-breaker caps, universe mask.

Typical speedup: ~200-1800x per-sim vs the original DataFrame-based version.
10K simulations on 8 cores: ~10-30 minutes (vs ~365 hours original).

Usage:
    python -m validation.random_baseline_fast --sims 100    # quick verify
    python -m validation.random_baseline_fast --sims 10000  # full run
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

NEPSE_TRADING_DAYS = 240
CIRCUIT_BREAKER_PCT = 0.10


# ═══════════════════════════════════════════════════════════════════════════
# Fee model — simplified for speed (matches TransactionCostModel.total_fees)
# ═══════════════════════════════════════════════════════════════════════════

# Broker tiers: (upper_bound, rate_or_flat, is_flat)
_BROKER_TIERS = [
    (2_500,       10.0,   True),
    (50_000,      0.0036, False),
    (500_000,     0.0033, False),
    (2_000_000,   0.0031, False),
    (10_000_000,  0.0027, False),
    (np.inf,      0.0024, False),
]
_SEBON_FEE_PCT = 0.00015
_NEPSE_FEE_MULT = 0.20  # 20% of broker commission
_DP_CHARGE = 25.0
_DP_NAME_TRANSFER = 5.0


def _fast_total_fees(amount: float, is_sell: bool = False) -> float:
    """Total single-leg fees. Matches TransactionCostModel.total_fees exactly."""
    # Broker commission (tiered)
    commission = 0.0
    for upper, rate, is_flat in _BROKER_TIERS:
        if amount <= upper:
            commission = rate if is_flat else amount * rate
            break

    sebon = amount * _SEBON_FEE_PCT
    nepse_fee = commission * _NEPSE_FEE_MULT
    total = commission + sebon + nepse_fee + _DP_CHARGE
    if is_sell:
        total += _DP_NAME_TRANSFER
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Pre-computed data structures (built once per worker)
# ═══════════════════════════════════════════════════════════════════════════

class PriceMatrices:
    """
    Pre-computed numpy price matrices for O(1) lookups.

    Attributes:
        open_matrix:   (n_dates, n_symbols) — open prices, NaN where missing
        close_matrix:  (n_dates, n_symbols) — close prices, NaN where missing
        prev_close:    (n_dates, n_symbols) — previous day's close (shifted)
        volume_matrix: (n_dates, n_symbols) — volumes
        dates:         sorted array of dates
        symbols:       sorted array of symbols
        date_to_idx:   dict[pd.Timestamp, int]
        sym_to_idx:    dict[str, int]
        open_capped:   (n_dates, n_symbols) — open prices capped at ±10% of prev_close
        close_capped:  (n_dates, n_symbols) — close prices capped at ±10% of prev_close
        regime_array:  (n_dates,) — 'bull', 'neutral', or 'bear' per date
        universe_by_date: list of arrays — tradeable symbol indices per date
    """

    def __init__(self, prices_df: pd.DataFrame = None, start_date: str = None, end_date: str = None):
        """Use _build_price_matrices() instead — this is a placeholder for __new__ usage."""
        if prices_df is not None:
            raise RuntimeError(
                "Use _build_price_matrices(prices_df, start_date, end_date) instead of "
                "PriceMatrices(). Direct construction is not supported."
            )

    def _cap_prices(self, price_matrix: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        """Apply ±10% circuit breaker caps."""
        capped = price_matrix.copy()
        valid = ~np.isnan(prev_close) & (prev_close > 0) & ~np.isnan(price_matrix)
        upper = prev_close * (1 + CIRCUIT_BREAKER_PCT)
        lower = prev_close * (1 - CIRCUIT_BREAKER_PCT)
        capped[valid] = np.clip(price_matrix[valid], lower[valid], upper[valid])
        return capped

def _build_price_matrices(prices_df: pd.DataFrame, start_date: str, end_date: str) -> PriceMatrices:
    """Build PriceMatrices using fully vectorized pandas/numpy operations."""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    pm = PriceMatrices.__new__(PriceMatrices)

    # All symbols from full history (needed for consistent indexing)
    pm.symbols = np.sort(prices_df["symbol"].unique())
    pm.sym_to_idx = {s: i for i, s in enumerate(pm.symbols)}
    pm.n_symbols = len(pm.symbols)

    # Build full close matrix using pandas pivot (fast!)
    # This covers ALL dates so we can compute prev_close and regime
    all_dates = np.sort(prices_df["date"].unique())
    all_date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    n_all_dates = len(all_dates)

    # Vectorized pivot via numpy fancy indexing
    prices_df_indexed = prices_df.copy()
    prices_df_indexed["_d_idx"] = prices_df_indexed["date"].map(all_date_to_idx)
    prices_df_indexed["_s_idx"] = prices_df_indexed["symbol"].map(pm.sym_to_idx)
    prices_df_indexed = prices_df_indexed.dropna(subset=["_d_idx", "_s_idx"])
    d_all = prices_df_indexed["_d_idx"].astype(int).values
    s_all = prices_df_indexed["_s_idx"].astype(int).values

    full_close = np.full((n_all_dates, pm.n_symbols), np.nan)
    full_close[d_all, s_all] = prices_df_indexed["close"].values

    full_open = np.full((n_all_dates, pm.n_symbols), np.nan)
    full_open[d_all, s_all] = prices_df_indexed["open"].values

    full_volume = np.full((n_all_dates, pm.n_symbols), np.nan)
    full_volume[d_all, s_all] = prices_df_indexed["volume"].values

    # Determine date range indices
    range_mask = (all_dates >= start_dt) & (all_dates <= end_dt)
    range_indices = np.where(range_mask)[0]

    pm.dates = all_dates[range_indices]
    pm.n_dates = len(pm.dates)
    pm.date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(pm.dates)}

    # Slice to range
    pm.open_matrix = full_open[range_indices]
    pm.close_matrix = full_close[range_indices]
    pm.volume_matrix = full_volume[range_indices]

    # Prev close: forward-fill close matrix per symbol, then take the row before each date.
    # Forward-fill ensures we get the last known close even if a symbol didn't trade yesterday.
    full_close_ffill = pd.DataFrame(full_close).ffill().values
    pm.prev_close = np.full((pm.n_dates, pm.n_symbols), np.nan)
    for i, full_idx in enumerate(range_indices):
        if full_idx > 0:
            pm.prev_close[i] = full_close_ffill[full_idx - 1]

    # Circuit-breaker capped prices
    pm.open_capped = pm._cap_prices(pm.open_matrix, pm.prev_close)
    pm.close_capped = pm._cap_prices(pm.close_matrix, pm.prev_close)

    # Pre-compute regimes using the full_close matrix (already built)
    pm.regime_array = np.empty(pm.n_dates, dtype=object)
    lookback = 60
    for i, full_idx in enumerate(range_indices):
        if full_idx < lookback:
            pm.regime_array[i] = "neutral"
            continue
        first_prices = full_close[full_idx - lookback]
        last_prices = full_close[full_idx]
        valid = ~np.isnan(first_prices) & ~np.isnan(last_prices) & (first_prices > 0)
        if valid.sum() < 20:
            pm.regime_array[i] = "neutral"
            continue
        returns = last_prices[valid] / first_prices[valid] - 1
        median_ret = float(np.median(returns))
        if median_ret < -0.05:
            pm.regime_array[i] = "bear"
        elif median_ret < 0.02:
            pm.regime_array[i] = "neutral"
        else:
            pm.regime_array[i] = "bull"

    # Universe per date
    pm.universe_by_date = []
    for d_idx in range(pm.n_dates):
        valid = np.where(~np.isnan(pm.close_matrix[d_idx]))[0]
        pm.universe_by_date.append(valid)

    return pm


# ═══════════════════════════════════════════════════════════════════════════
# Fast single-simulation engine
# ═══════════════════════════════════════════════════════════════════════════

def _execute_single_sim_fast(
    sim_idx: int,
    rng_seed: int,
    pm: PriceMatrices,
    holding_days: int,
    max_positions: int,
    initial_capital: float,
    rebalance_frequency: int,
    use_trailing_stop: bool,
    trailing_stop_pct: float,
    stop_loss_pct: float,
    use_regime_filter: bool,
    regime_max_positions: Optional[Dict[str, int]] = None,
) -> float:
    """
    Run a single random-entry simulation using pre-computed numpy matrices.
    Returns Sharpe ratio.
    """
    rng = np.random.default_rng(rng_seed + sim_idx + 1)

    n_dates = pm.n_dates
    cash = initial_capital

    # Position tracking arrays (max_positions slots)
    pos_symbol_idx = np.full(max_positions, -1, dtype=np.int32)  # -1 = empty
    pos_entry_price = np.zeros(max_positions)
    pos_shares = np.zeros(max_positions, dtype=np.int32)
    pos_max_price = np.zeros(max_positions)
    pos_entry_day = np.zeros(max_positions, dtype=np.int32)
    pos_buy_fees = np.zeros(max_positions)

    daily_nav = np.zeros(n_dates)
    pending_symbols = []  # Symbol indices queued for next-day entry
    pending_signal_day = -1
    last_signal_day_idx = -rebalance_frequency  # Force first signal

    for i in range(n_dates):
        # ── STEP 1: Execute pending entries at today's open ──
        if pending_symbols:
            for s_idx in pending_symbols:
                # Find empty slot
                n_active = int(np.sum(pos_symbol_idx >= 0))
                if n_active >= max_positions:
                    break

                open_price = pm.open_capped[i, s_idx]
                if np.isnan(open_price) or open_price <= 0:
                    continue

                # Check not already held
                if s_idx in pos_symbol_idx:
                    continue

                # Position sizing
                per_position = initial_capital / max_positions
                available = min(per_position, cash * 0.95)
                if available < 10000:
                    continue

                shares = int(available / open_price)
                if shares < 10:
                    continue

                amount = shares * open_price
                buy_fees = _fast_total_fees(amount, is_sell=False)
                total_cost = amount + buy_fees
                if total_cost > cash:
                    shares = int((cash - buy_fees) / open_price)
                    if shares < 10:
                        continue
                    amount = shares * open_price
                    buy_fees = _fast_total_fees(amount, is_sell=False)
                    total_cost = amount + buy_fees

                cash -= total_cost

                # Find empty slot
                slot = -1
                for s in range(max_positions):
                    if pos_symbol_idx[s] < 0:
                        slot = s
                        break
                if slot < 0:
                    break

                pos_symbol_idx[slot] = s_idx
                pos_entry_price[slot] = open_price
                pos_shares[slot] = shares
                pos_max_price[slot] = open_price
                pos_entry_day[slot] = i
                pos_buy_fees[slot] = buy_fees

            pending_symbols = []

        # ── STEP 2: Check exits ──
        for slot in range(max_positions):
            s_idx = pos_symbol_idx[slot]
            if s_idx < 0:
                continue

            current_price = pm.open_capped[i, s_idx]
            if np.isnan(current_price):
                continue

            if current_price > pos_max_price[slot]:
                pos_max_price[slot] = current_price

            trading_days_held = i - pos_entry_day[slot]
            exit_reason = None

            # Stop loss
            if current_price < pos_entry_price[slot] * (1 - stop_loss_pct):
                exit_reason = "stop_loss"

            # Trailing stop
            if exit_reason is None and use_trailing_stop and pos_max_price[slot] > pos_entry_price[slot]:
                trailing_stop_price = pos_max_price[slot] * (1 - trailing_stop_pct)
                if current_price < trailing_stop_price:
                    exit_reason = "trailing_stop"

            # Holding period
            if exit_reason is None and trading_days_held >= holding_days:
                exit_reason = "holding_period"
                close_price = pm.close_capped[i, s_idx]
                if not np.isnan(close_price):
                    current_price = close_price

            if exit_reason:
                sell_amount = pos_shares[slot] * current_price
                sell_fees = _fast_total_fees(sell_amount, is_sell=True)
                cash += sell_amount - sell_fees
                pos_symbol_idx[slot] = -1  # Clear slot

        # ── STEP 3: Record NAV ──
        nav = cash
        for slot in range(max_positions):
            s_idx = pos_symbol_idx[slot]
            if s_idx < 0:
                continue
            close_price = pm.close_matrix[i, s_idx]
            if np.isnan(close_price):
                close_price = pos_entry_price[slot]
            nav += pos_shares[slot] * close_price
        daily_nav[i] = nav

        # ── STEP 4: Random signal generation ──
        days_since_signal = i - last_signal_day_idx
        # Use calendar-day-like check: rebalance_frequency is in calendar days
        # but we approximate by checking index difference
        if days_since_signal >= rebalance_frequency:
            remaining = n_dates - i - 1
            if remaining < holding_days:
                continue

            # Regime filter: graduated positions or binary skip
            regime = pm.regime_array[i]
            if use_regime_filter and regime == "bear" and regime_max_positions is None:
                # Legacy behavior: skip all entries in bear markets
                last_signal_day_idx = i
                continue

            # Graduated regime position limits
            effective_max = max_positions
            if regime_max_positions is not None and regime in regime_max_positions:
                effective_max = regime_max_positions[regime]

            # Pick random symbols from today's universe
            universe = pm.universe_by_date[i]
            if len(universe) == 0:
                last_signal_day_idx = i
                continue

            n_active = int(np.sum(pos_symbol_idx >= 0))
            slots = effective_max - n_active
            if slots <= 0:
                last_signal_day_idx = i
                continue

            # Exclude already-held symbols
            held_set = set(pos_symbol_idx[pos_symbol_idx >= 0].tolist())
            available = np.array([s for s in universe if s not in held_set])
            if len(available) == 0:
                last_signal_day_idx = i
                continue

            n_picks = min(slots, len(available))
            picks = rng.choice(available, size=n_picks, replace=False)
            pending_symbols = picks.tolist()

            last_signal_day_idx = i

    # ── Close remaining positions at last available price ──
    for slot in range(max_positions):
        s_idx = pos_symbol_idx[slot]
        if s_idx < 0:
            continue
        # Find last valid close
        exit_price = np.nan
        for d in range(n_dates - 1, -1, -1):
            p = pm.close_matrix[d, s_idx]
            if not np.isnan(p):
                exit_price = p
                break
        if not np.isnan(exit_price):
            sell_amount = pos_shares[slot] * exit_price
            sell_fees = _fast_total_fees(sell_amount, is_sell=True)
            cash += sell_amount - sell_fees
        pos_symbol_idx[slot] = -1

    # ── Compute Sharpe from daily NAV ──
    if n_dates < 31:
        return 0.0
    returns = np.diff(daily_nav) / daily_nav[:-1]
    # Filter out any inf/nan from division issues
    returns = returns[np.isfinite(returns)]
    if len(returns) < 30:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(NEPSE_TRADING_DAYS))


# ═══════════════════════════════════════════════════════════════════════════
# Beta-hedged Alpha Sharpe (Phase 4A)
# ═══════════════════════════════════════════════════════════════════════════

def compute_alpha_sharpe(
    strategy_nav: np.ndarray,
    benchmark_nav: np.ndarray,
) -> float:
    """
    Compute alpha Sharpe — residual returns after removing market beta.

    alpha_returns = strategy_ret - beta * benchmark_ret
    alpha_sharpe = mean(alpha_returns) / std(alpha_returns) * sqrt(240)
    """
    if len(strategy_nav) < 31 or len(benchmark_nav) < 31:
        return 0.0

    # Align lengths
    n = min(len(strategy_nav), len(benchmark_nav))
    strat_ret = np.diff(strategy_nav[:n]) / strategy_nav[:n - 1]
    bench_ret = np.diff(benchmark_nav[:n]) / benchmark_nav[:n - 1]

    # Remove NaN/inf
    valid = np.isfinite(strat_ret) & np.isfinite(bench_ret)
    strat_ret = strat_ret[valid]
    bench_ret = bench_ret[valid]

    if len(strat_ret) < 30:
        return 0.0

    # Beta via OLS
    var_bench = np.var(bench_ret, ddof=0)
    if var_bench == 0:
        return 0.0
    cov = np.mean((strat_ret - np.mean(strat_ret)) * (bench_ret - np.mean(bench_ret)))
    beta = cov / var_bench

    # Alpha returns
    alpha_returns = strat_ret - beta * bench_ret
    std_alpha = np.std(alpha_returns, ddof=1)
    if std_alpha == 0:
        return 0.0

    return float(np.mean(alpha_returns) / std_alpha * np.sqrt(NEPSE_TRADING_DAYS))


# ═══════════════════════════════════════════════════════════════════════════
# Q1 vs Q5 Spread Test (Phase 4B)
# ═══════════════════════════════════════════════════════════════════════════

def compute_q1_q5_spread(
    prices_df: pd.DataFrame,
    signal_func,
    start_date: str,
    end_date: str,
    holding_days: int = 40,
    rebalance_frequency: int = 21,
) -> dict:
    """
    Test cross-sectional signal quality: do top-ranked stocks outperform bottom-ranked?

    On each signal date, rank the universe by the signal function.
    Q1 = top 20%, Q5 = bottom 20%.
    Compute forward holding-period returns for each quintile.
    T-test: mean(Q1 returns) - mean(Q5 returns).

    Parameters
    ----------
    prices_df : Full price DataFrame
    signal_func : callable(prices_df, date) -> dict[symbol, float] mapping symbols to scores
    start_date, end_date : Date range
    holding_days : Forward return holding period
    rebalance_frequency : Days between signal evaluations

    Returns
    -------
    dict with q1_mean, q5_mean, spread, t_stat, p_value, n_dates
    """
    from scipy import stats as sp_stats

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    all_dates = sorted(prices_df["date"].unique())
    trading_dates = [d for d in all_dates if start_dt <= d <= end_dt]

    # Build close lookup for forward returns
    close_by_sym = {}
    for sym, grp in prices_df.groupby("symbol"):
        grp = grp.sort_values("date")
        close_by_sym[sym] = dict(zip(grp["date"].values, grp["close"].values))

    q1_returns = []
    q5_returns = []
    n_signal_dates = 0

    for i in range(0, len(trading_dates) - holding_days, rebalance_frequency):
        signal_date = trading_dates[i]
        exit_date_idx = min(i + holding_days, len(trading_dates) - 1)
        exit_date = trading_dates[exit_date_idx]

        # Get signal scores
        scores = signal_func(prices_df, signal_date)
        if len(scores) < 10:
            continue

        # Rank into quintiles
        sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        n = len(sorted_syms)
        q_size = max(1, n // 5)

        q1_syms = sorted_syms[:q_size]
        q5_syms = sorted_syms[-q_size:]

        # Forward returns
        for sym in q1_syms:
            entry_close = close_by_sym.get(sym, {}).get(signal_date)
            exit_close = close_by_sym.get(sym, {}).get(exit_date)
            if entry_close and exit_close and entry_close > 0:
                q1_returns.append(exit_close / entry_close - 1)

        for sym in q5_syms:
            entry_close = close_by_sym.get(sym, {}).get(signal_date)
            exit_close = close_by_sym.get(sym, {}).get(exit_date)
            if entry_close and exit_close and entry_close > 0:
                q5_returns.append(exit_close / entry_close - 1)

        n_signal_dates += 1

    q1_returns = np.array(q1_returns)
    q5_returns = np.array(q5_returns)

    if len(q1_returns) < 5 or len(q5_returns) < 5:
        return {
            "q1_mean": 0.0, "q5_mean": 0.0, "spread": 0.0,
            "t_stat": 0.0, "p_value": 1.0, "n_signal_dates": n_signal_dates,
            "n_q1": len(q1_returns), "n_q5": len(q5_returns),
            "passes": False,
        }

    q1_mean = float(np.mean(q1_returns))
    q5_mean = float(np.mean(q5_returns))
    spread = q1_mean - q5_mean

    # Welch's t-test
    t_stat, p_value = sp_stats.ttest_ind(q1_returns, q5_returns, equal_var=False)

    return {
        "q1_mean": q1_mean,
        "q5_mean": q5_mean,
        "spread": spread,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_signal_dates": n_signal_dates,
        "n_q1": len(q1_returns),
        "n_q5": len(q5_returns),
        "passes": p_value < 0.05 and spread > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Worker batch function (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def _run_fast_batch(args):
    """
    Top-level batch worker for multiprocessing.
    Loads data once, builds PriceMatrices, runs a range of simulations.
    Returns list of (sharpe, daily_nav_array) tuples.
    """
    (sim_start, sim_end, rng_seed, start_date, end_date,
     holding_days, max_positions, initial_capital, rebalance_frequency,
     use_trailing_stop, trailing_stop_pct, stop_loss_pct,
     use_regime_filter, return_navs, regime_max_positions) = args

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger("random_baseline_fast.worker")

    from backend.backtesting.simple_backtest import load_all_prices
    from backend.quant_pro.database import get_db_path

    # Load data once per worker
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)
    conn.close()

    _logger.info(f"Worker building price matrices for sims {sim_start}-{sim_end - 1}...")
    pm = _build_price_matrices(prices_df, start_date, end_date)
    _logger.info(f"Worker matrices ready: {pm.n_dates} dates x {pm.n_symbols} symbols")

    results = []
    n_sims = sim_end - sim_start
    for sim in range(sim_start, sim_end):
        if (sim - sim_start) % 100 == 0:
            _logger.info(f"  Worker sim {sim - sim_start}/{n_sims}")

        sharpe = _execute_single_sim_fast(
            sim_idx=sim,
            rng_seed=rng_seed,
            pm=pm,
            holding_days=holding_days,
            max_positions=max_positions,
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct,
            use_regime_filter=use_regime_filter,
            regime_max_positions=regime_max_positions,
        )
        results.append(sharpe)

    _logger.info(
        f"Worker done: sims {sim_start}-{sim_end - 1}, "
        f"mean Sharpe={np.mean(results):.3f}"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def random_entry_baseline_fast(
    n_simulations: int = 10000,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    rng_seed: int = 42,
    max_workers: int = 0,  # 0 = auto (cpu_count)
    compute_alpha: bool = True,
    **backtest_kwargs,
) -> dict:
    """
    Compare actual strategy Sharpe against random entry simulations (fast version).

    Uses pre-computed numpy matrices for O(1) price lookups.
    Supports multiprocessing for parallel simulation.

    Parameters
    ----------
    n_simulations : Number of random baseline runs (default 10K)
    start_date : Backtest start date
    end_date : Backtest end date
    rng_seed : Random seed for reproducibility
    max_workers : Number of parallel workers (0 = auto)
    compute_alpha : Whether to compute beta-hedged alpha Sharpe
    **backtest_kwargs : Passed to run_backtest()

    Returns
    -------
    Dict with actual_sharpe, random_sharpes, percentile_rank, p_value,
    alpha_sharpe (if compute_alpha), etc.
    """
    from backend.backtesting.simple_backtest import run_backtest, load_all_prices
    from backend.quant_pro.database import get_db_path

    if max_workers <= 0:
        max_workers = max(1, os.cpu_count() or 1)

    # 1. Run actual strategy
    kwargs = {**backtest_kwargs, "start_date": start_date, "end_date": end_date}
    actual_result = run_backtest(**kwargs)
    actual_sharpe = actual_result.sharpe_ratio
    logger.info(f"Actual strategy Sharpe: {actual_sharpe:.3f}")

    # Extract params
    holding_days = kwargs.get("holding_days", 40)
    max_positions = kwargs.get("max_positions", 5)
    initial_capital = kwargs.get("initial_capital", 1_000_000)
    rebalance_frequency = kwargs.get("rebalance_frequency", 5)
    use_trailing_stop = kwargs.get("use_trailing_stop", True)
    trailing_stop_pct = kwargs.get("trailing_stop_pct", 0.10)
    stop_loss_pct = kwargs.get("stop_loss_pct", 0.08)
    use_regime_filter = kwargs.get("use_regime_filter", True)
    regime_max_positions = kwargs.get("regime_max_positions", None)

    # 2. Compute benchmark for alpha Sharpe
    actual_alpha_sharpe = None
    if compute_alpha:
        try:
            from validation.benchmark import compute_benchmark_series
            conn = sqlite3.connect(str(get_db_path()))
            prices_df = load_all_prices(conn)
            conn.close()

            benchmark = compute_benchmark_series(prices_df, start_date, end_date)
            bench_nav = benchmark.values

            strat_nav = np.array([nav for _, nav in actual_result.daily_nav])

            actual_alpha_sharpe = compute_alpha_sharpe(strat_nav, bench_nav)
            logger.info(f"Actual alpha Sharpe (beta-hedged): {actual_alpha_sharpe:.3f}")
        except Exception as e:
            logger.warning(f"Could not compute alpha Sharpe: {e}")

    # 3. Run random simulations in parallel
    chunk_size = (n_simulations + max_workers - 1) // max_workers
    batches = []
    for i in range(max_workers):
        batch_start = i * chunk_size
        batch_end = min((i + 1) * chunk_size, n_simulations)
        if batch_start >= n_simulations:
            break
        batches.append((
            batch_start, batch_end, rng_seed, start_date, end_date,
            holding_days, max_positions, initial_capital, rebalance_frequency,
            use_trailing_stop, trailing_stop_pct, stop_loss_pct,
            use_regime_filter, False,  # return_navs
            regime_max_positions,
        ))

    logger.info(
        f"Dispatching {n_simulations} sims across {len(batches)} workers "
        f"(~{chunk_size} sims each)"
    )

    t0 = time.time()
    if max_workers == 1:
        # Sequential — useful for debugging
        all_sharpes = _run_fast_batch(batches[0])
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            batch_results = list(pool.map(_run_fast_batch, batches))
        all_sharpes = []
        for batch in batch_results:
            all_sharpes.extend(batch)

    elapsed = time.time() - t0
    logger.info(f"Simulations complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    random_sharpes = np.array(all_sharpes[:n_simulations])

    # 4. Statistics
    percentile_rank = float(np.mean(random_sharpes < actual_sharpe) * 100)
    p_value = float(np.mean(random_sharpes >= actual_sharpe))

    result = {
        "actual_sharpe": actual_sharpe,
        "random_sharpes": random_sharpes,
        "percentile_rank": percentile_rank,
        "p_value": p_value,
        "mean_random": float(np.mean(random_sharpes)),
        "std_random": float(np.std(random_sharpes)),
        "median_random": float(np.median(random_sharpes)),
        "n_simulations": n_simulations,
        "passes": percentile_rank >= 95.0,
        "elapsed_seconds": elapsed,
    }

    if actual_alpha_sharpe is not None:
        result["alpha_sharpe"] = actual_alpha_sharpe

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fast random baseline (numpy-optimized)"
    )
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--workers", type=int, default=0, help="Workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    t_start = time.time()

    result = random_entry_baseline_fast(
        n_simulations=args.sims,
        start_date=args.start,
        end_date=args.end,
        rng_seed=args.seed,
        max_workers=args.workers,
        compute_alpha=True,
        holding_days=40,
        max_positions=5,
        signal_types=["volume", "quality", "low_vol"],
        initial_capital=1_000_000,
        rebalance_frequency=5,
        use_trailing_stop=True,
        trailing_stop_pct=0.10,
        stop_loss_pct=0.08,
        use_regime_filter=True,
        sector_limit=0.35,
    )

    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print(f"FAST RANDOM BASELINE RESULTS ({args.sims} sims)")
    print("=" * 60)
    print(f'Actual Strategy Sharpe: {result["actual_sharpe"]:.3f}')
    print(f'Random Mean Sharpe:     {result["mean_random"]:.3f}')
    print(f'Random Std:             {result["std_random"]:.3f}')
    print(f'Random Median:          {result["median_random"]:.3f}')
    print(f'Percentile Rank:        {result["percentile_rank"]:.1f}%')
    print(f'p-value:                {result["p_value"]:.4f}')
    print(f'PASS (p<0.05):          {result["p_value"] < 0.05}')
    if "alpha_sharpe" in result:
        print(f'Alpha Sharpe (hedged):  {result["alpha_sharpe"]:.3f}')
    print()
    rs = result["random_sharpes"]
    print(f"Random Sharpe range: [{np.min(rs):.3f}, {np.max(rs):.3f}]")
    print(f'Random Sharpes > Actual: {np.sum(rs >= result["actual_sharpe"])}/{len(rs)}')
    print(f"Elapsed: {elapsed / 3600:.2f} hours ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
