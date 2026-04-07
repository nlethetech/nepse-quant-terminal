"""
Random entry baseline test.

Proves the strategy's signal has genuine edge by comparing against random
stock picks using identical exit rules, position sizing, and risk management.

Supports multiprocessing: each worker loads its own price data and runs a
batch of simulations independently.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _execute_single_sim(
    sim_idx: int,
    rng_seed: int,
    trading_dates: list,
    prices_df: pd.DataFrame,
    universe_by_date: dict,
    close_lookup: dict,
    holding_days: int,
    max_positions: int,
    initial_capital: float,
    rebalance_frequency: int,
    use_trailing_stop: bool,
    trailing_stop_pct: float,
    stop_loss_pct: float,
    use_regime_filter: bool,
    end_dt: pd.Timestamp,
) -> float:
    """
    Run a single random-entry simulation and return its Sharpe ratio.

    Uses a deterministic per-simulation seed: rng_seed + sim_idx + 1.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import (
        NepseFees, Trade, BacktestResult, compute_market_regime,
        apply_circuit_breaker, get_price_at_date, get_prev_close,
        get_price_on_or_before_date,
    )

    rng = np.random.default_rng(rng_seed + sim_idx + 1)

    cash = initial_capital
    current_positions: Dict[str, Trade] = {}
    completed_trades: List[Trade] = []
    pending_entries: List[dict] = []
    daily_nav = []
    last_signal_date = None

    for i, current_date in enumerate(trading_dates):
        current_date = pd.Timestamp(current_date)

        # Execute pending entries
        for entry_info in pending_entries:
            symbol = entry_info["symbol"]
            if symbol in current_positions or len(current_positions) >= max_positions:
                continue

            open_price = get_price_at_date(
                prices_df, symbol, current_date, look_forward_days=0, use_open=True
            )
            if open_price is None or open_price <= 0:
                continue

            prev_close = get_prev_close(prices_df, symbol, current_date)
            if prev_close is not None:
                open_price = apply_circuit_breaker(open_price, prev_close)

            per_position = initial_capital / max_positions
            available = min(per_position, cash * 0.95)
            if available < 10000:
                continue

            shares = int(available / open_price)
            if shares < 10:
                continue

            buy_fees = NepseFees.total_fees(shares, open_price)
            total_cost = shares * open_price + buy_fees
            if total_cost > cash:
                shares = int((cash - buy_fees) / open_price)
                if shares < 10:
                    continue
                buy_fees = NepseFees.total_fees(shares, open_price)
                total_cost = shares * open_price + buy_fees

            cash -= total_cost
            trade = Trade(
                symbol=symbol,
                signal_date=entry_info.get("signal_date"),
                entry_date=current_date,
                entry_price=open_price,
                shares=shares,
                position_value=shares * open_price,
                buy_fees=buy_fees,
                signal_type="random",
                direction=1,
                max_price=open_price,
                entry_trading_idx=i,
            )
            current_positions[symbol] = trade

        pending_entries = []

        # Check exits (identical to real strategy)
        to_close = []
        for symbol, trade in current_positions.items():
            current_price = get_price_at_date(
                prices_df, symbol, current_date, look_forward_days=0, use_open=True
            )
            if current_price is None:
                continue

            prev_close = get_prev_close(prices_df, symbol, current_date)
            if prev_close is not None:
                current_price = apply_circuit_breaker(current_price, prev_close)

            if current_price > trade.max_price:
                trade.max_price = current_price

            trading_days_held = i - trade.entry_trading_idx
            exit_reason = None

            if current_price < trade.entry_price * (1 - stop_loss_pct):
                exit_reason = "stop_loss"

            if exit_reason is None and use_trailing_stop and trade.max_price > trade.entry_price:
                trailing_stop_price = trade.max_price * (1 - trailing_stop_pct)
                if current_price < trailing_stop_price:
                    exit_reason = "trailing_stop"

            if exit_reason is None and trading_days_held >= holding_days:
                exit_reason = "holding_period"
                close_price = get_price_at_date(
                    prices_df, symbol, current_date, look_forward_days=0, use_open=False
                )
                if close_price is not None:
                    if prev_close is not None:
                        close_price = apply_circuit_breaker(close_price, prev_close)
                    current_price = close_price

            if exit_reason:
                sell_fees = NepseFees.total_fees(trade.shares, current_price)
                cash += trade.shares * current_price - sell_fees
                trade.exit_date = current_date
                trade.exit_price = current_price
                trade.sell_fees = sell_fees
                trade.exit_reason = exit_reason
                completed_trades.append(trade)
                to_close.append(symbol)

        for symbol in to_close:
            del current_positions[symbol]

        # NAV
        nav = cash
        for sym, trade in current_positions.items():
            cp = close_lookup.get(sym, {}).get(current_date, trade.entry_price)
            nav += trade.shares * cp
        daily_nav.append((current_date, nav))

        # Random signal generation (same frequency and count as real strategy)
        if last_signal_date is None or (current_date - last_signal_date).days >= rebalance_frequency:
            remaining = len(trading_dates) - i - 1
            if remaining < holding_days:
                continue

            if use_regime_filter:
                regime = compute_market_regime(prices_df, current_date)
                if regime == "bear":
                    last_signal_date = current_date
                    continue

            # Pick random symbols
            universe = universe_by_date.get(current_date, [])
            if universe:
                slots = max_positions - len(current_positions)
                already_held = set(current_positions.keys())
                available_symbols = [s for s in universe if s not in already_held]
                n_picks = min(slots, len(available_symbols))
                if n_picks > 0:
                    picks = rng.choice(available_symbols, size=n_picks, replace=False)
                    for sym in picks:
                        pending_entries.append({
                            "symbol": sym,
                            "signal_type": "random",
                            "signal_date": current_date,
                        })

            last_signal_date = current_date

    # Close remaining positions
    for symbol, trade in current_positions.items():
        exit_fill = get_price_on_or_before_date(prices_df, symbol, end_dt)
        if exit_fill:
            exit_date, exit_price = exit_fill
            sell_fees = NepseFees.total_fees(trade.shares, exit_price)
            trade.exit_date = exit_date
            trade.exit_price = exit_price
            trade.sell_fees = sell_fees
            trade.exit_reason = "end_of_backtest"
            cash += trade.shares * exit_price - sell_fees
            completed_trades.append(trade)

    # Compute Sharpe for this random run
    start_dt = pd.Timestamp(trading_dates[0]) if trading_dates else end_dt
    sim_result = BacktestResult(
        trades=completed_trades,
        start_date=start_dt,
        end_date=end_dt,
        holding_period=holding_days,
        initial_capital=initial_capital,
        daily_nav=daily_nav,
    )
    return sim_result.sharpe_ratio


def _run_random_batch(args):
    """
    Top-level batch worker for multiprocessing (must be picklable).

    Loads price data once, then runs a range of simulations.
    Returns list of Sharpe ratios.
    """
    (sim_start, sim_end, rng_seed, start_date, end_date,
     holding_days, max_positions, initial_capital, rebalance_frequency,
     use_trailing_stop, trailing_stop_pct, stop_loss_pct,
     use_regime_filter) = args

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger("random_baseline.worker")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import load_all_prices, build_close_lookup
    from backend.quant_pro.database import get_db_path

    # Load data once per worker
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)
    conn.close()

    all_dates = sorted(prices_df["date"].unique())
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    trading_dates = [d for d in all_dates if start_dt <= d <= end_dt]

    universe_by_date = {}
    for date in trading_dates:
        day_data = prices_df[prices_df["date"] == date]
        symbols = day_data["symbol"].unique().tolist()
        universe_by_date[pd.Timestamp(date)] = symbols

    close_lookup = build_close_lookup(prices_df)

    _logger.info(f"Worker starting sims {sim_start}-{sim_end-1} ({sim_end - sim_start} sims)")

    sharpes = []
    for sim in range(sim_start, sim_end):
        if (sim - sim_start) % 10 == 0:
            _logger.info(f"  Worker sim {sim - sim_start}/{sim_end - sim_start}")

        sharpe = _execute_single_sim(
            sim_idx=sim,
            rng_seed=rng_seed,
            trading_dates=trading_dates,
            prices_df=prices_df,
            universe_by_date=universe_by_date,
            close_lookup=close_lookup,
            holding_days=holding_days,
            max_positions=max_positions,
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct,
            use_regime_filter=use_regime_filter,
            end_dt=end_dt,
        )
        sharpes.append(sharpe)

    _logger.info(f"Worker done: sims {sim_start}-{sim_end-1}, mean Sharpe={np.mean(sharpes):.3f}")
    return sharpes


def random_entry_baseline(
    n_simulations: int = 1000,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    rng_seed: int = 42,
    max_workers: int = 1,
    **backtest_kwargs,
) -> dict:
    """
    Compare actual strategy Sharpe against random entry simulations.

    For each simulation, replaces signal generation with random stock picks
    (same number of picks per rebalance, random symbols from the tradeable
    universe). Uses identical exit rules, position sizing, and risk management.

    Parameters
    ----------
    n_simulations : Number of random baseline runs
    start_date : Backtest start date
    end_date : Backtest end date
    rng_seed : Random seed for reproducibility
    max_workers : Number of parallel workers (1 = sequential)
    **backtest_kwargs : Passed to run_backtest()

    Returns
    -------
    Dict with:
        actual_sharpe, random_sharpes (array), percentile_rank,
        p_value, mean_random, std_random, passes (bool)
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import run_backtest

    # 1. Run actual strategy
    kwargs = {**backtest_kwargs, "start_date": start_date, "end_date": end_date}
    actual_result = run_backtest(**kwargs)
    actual_sharpe = actual_result.sharpe_ratio
    logger.info(f"Actual strategy Sharpe: {actual_sharpe:.3f}")

    # Extract backtest params for workers
    holding_days = kwargs.get("holding_days", 40)
    max_positions = kwargs.get("max_positions", 5)
    initial_capital = kwargs.get("initial_capital", 1_000_000)
    rebalance_frequency = kwargs.get("rebalance_frequency", 5)
    use_trailing_stop = kwargs.get("use_trailing_stop", True)
    trailing_stop_pct = kwargs.get("trailing_stop_pct", 0.10)
    stop_loss_pct = kwargs.get("stop_loss_pct", 0.08)
    use_regime_filter = kwargs.get("use_regime_filter", True)

    # 2. Run random simulations (parallel or sequential)
    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        # Split simulations into chunks for workers
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
                use_regime_filter,
            ))

        logger.info(
            f"Dispatching {n_simulations} sims across {len(batches)} workers "
            f"(~{chunk_size} sims each)"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            batch_results = list(pool.map(_run_random_batch, batches))

        # Flatten results
        random_sharpes = []
        for batch in batch_results:
            random_sharpes.extend(batch)
    else:
        # Sequential: load data once in main process
        from backend.backtesting.simple_backtest import load_all_prices, build_close_lookup
        from backend.quant_pro.database import get_db_path

        conn = sqlite3.connect(str(get_db_path()))
        prices_df = load_all_prices(conn)
        conn.close()

        all_dates = sorted(prices_df["date"].unique())
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        trading_dates = [d for d in all_dates if start_dt <= d <= end_dt]

        universe_by_date = {}
        for date in trading_dates:
            day_data = prices_df[prices_df["date"] == date]
            symbols = day_data["symbol"].unique().tolist()
            universe_by_date[pd.Timestamp(date)] = symbols

        close_lookup = build_close_lookup(prices_df)

        random_sharpes = []
        for sim in range(n_simulations):
            if sim % 100 == 0:
                logger.info(f"Random baseline simulation {sim}/{n_simulations}")

            sharpe = _execute_single_sim(
                sim_idx=sim,
                rng_seed=rng_seed,
                trading_dates=trading_dates,
                prices_df=prices_df,
                universe_by_date=universe_by_date,
                close_lookup=close_lookup,
                holding_days=holding_days,
                max_positions=max_positions,
                initial_capital=initial_capital,
                rebalance_frequency=rebalance_frequency,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_pct=trailing_stop_pct,
                stop_loss_pct=stop_loss_pct,
                use_regime_filter=use_regime_filter,
                end_dt=end_dt,
            )
            random_sharpes.append(sharpe)

    random_sharpes = np.array(random_sharpes)

    # Statistics
    percentile_rank = float(np.mean(random_sharpes < actual_sharpe) * 100)
    p_value = float(np.mean(random_sharpes >= actual_sharpe))

    return {
        "actual_sharpe": actual_sharpe,
        "random_sharpes": random_sharpes,
        "percentile_rank": percentile_rank,
        "p_value": p_value,
        "mean_random": float(np.mean(random_sharpes)),
        "std_random": float(np.std(random_sharpes)),
        "median_random": float(np.median(random_sharpes)),
        "n_simulations": n_simulations,
        "passes": percentile_rank >= 95.0,
    }
