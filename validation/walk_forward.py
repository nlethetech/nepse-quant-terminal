"""
Walk-forward cross-validation for backtest robustness.

Splits the data into overlapping train/test windows, runs the backtest
on each test window (using the train window for regime context),
then stitches the out-of-sample equity curves together.

Supports multiprocessing: each fold runs independently.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEPSE_TRADING_DAYS = 240


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_id: int
    start_date: str
    end_date: str
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    annualized_return: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    fold_metrics: List[FoldMetrics]
    oos_equity_curve: List[Tuple[Any, float]]
    aggregate: dict  # mean, std, min, max of Sharpe across folds
    oos_sharpe: float
    oos_total_return: float
    passes: bool  # True if mean OOS Sharpe > threshold


def _run_wf_fold(args):
    """
    Top-level worker function for multiprocessing (must be picklable).

    Runs a single walk-forward fold backtest.
    Returns dict with fold metrics and daily NAV, or error.
    """
    fold_id, test_start_str, test_end_str, backtest_kwargs = args

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger("walk_forward.worker")

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import run_backtest

    kwargs = {**backtest_kwargs}
    kwargs["start_date"] = test_start_str
    kwargs["end_date"] = test_end_str
    if "initial_capital" not in backtest_kwargs:
        kwargs["initial_capital"] = 1_000_000

    try:
        result = run_backtest(**kwargs)
        _logger.info(
            f"Fold {fold_id}: Sharpe={result.sharpe_ratio:.3f}, "
            f"Return={result.total_return:.2%}, Trades={result.total_trades}"
        )
        return {
            "fold_id": fold_id,
            "start_date": test_start_str,
            "end_date": test_end_str,
            "sharpe": result.sharpe_ratio,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "n_trades": result.total_trades,
            "win_rate": result.win_rate,
            "annualized_return": result.annualized_return,
            "daily_nav": result.daily_nav,
        }
    except Exception as e:
        _logger.warning(f"Fold {fold_id} failed: {e}")
        return {"fold_id": fold_id, "error": str(e)}


def walk_forward_validation(
    train_window_days: int = 504,
    test_window_days: int = 63,
    step_days: int = 21,
    min_sharpe: float = 0.8,
    max_workers: int = 1,
    **backtest_kwargs,
) -> WalkForwardResult:
    """
    Walk-forward cross-validation.

    Slides a train/test window forward through the full date range,
    running the backtest on each test window. The train window provides
    context (regime detection uses data before the test period).

    Parameters
    ----------
    train_window_days : Training window in trading days (~2 years at 504)
    test_window_days : Test window in trading days (~3 months at 63)
    step_days : Step size in trading days (~1 month at 21)
    min_sharpe : Minimum mean OOS Sharpe to pass
    max_workers : Number of parallel workers (1 = sequential)
    **backtest_kwargs : Passed to run_backtest()

    Returns
    -------
    WalkForwardResult with per-fold metrics and stitched OOS equity curve
    """
    # Defer import to avoid circular dependency
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import run_backtest, load_all_prices
    from backend.quant_pro.database import get_db_path

    # Load date range
    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)
    conn.close()

    all_dates = sorted(prices_df["date"].unique())
    total_days = len(all_dates)
    logger.info(
        f"Walk-forward: {total_days} trading days, "
        f"train={train_window_days}, test={test_window_days}, step={step_days}"
    )

    # Generate fold definitions
    fold_defs = []
    fold_id = 0
    start_idx = train_window_days
    while start_idx + test_window_days <= total_days:
        test_start = all_dates[start_idx]
        test_end_idx = min(start_idx + test_window_days - 1, total_days - 1)
        test_end = all_dates[test_end_idx]

        test_start_str = pd.Timestamp(test_start).strftime("%Y-%m-%d")
        test_end_str = pd.Timestamp(test_end).strftime("%Y-%m-%d")

        fold_defs.append((fold_id, test_start_str, test_end_str, backtest_kwargs))

        start_idx += step_days
        fold_id += 1

    logger.info(f"Generated {len(fold_defs)} folds")

    # Execute folds (parallel or sequential)
    if max_workers > 1 and len(fold_defs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Dispatching {len(fold_defs)} folds across {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            fold_results_raw = list(pool.map(_run_wf_fold, fold_defs))
    else:
        fold_results_raw = []
        for fold_def in fold_defs:
            fold_results_raw.append(_run_wf_fold(fold_def))

    # Process results (sort by fold_id to maintain order)
    fold_results_raw.sort(key=lambda x: x["fold_id"])

    fold_metrics: List[FoldMetrics] = []
    all_oos_nav: List[Tuple[Any, float]] = []

    for raw in fold_results_raw:
        if "error" in raw:
            logger.warning(f"Fold {raw['fold_id']} failed: {raw['error']}")
            continue

        fm = FoldMetrics(
            fold_id=raw["fold_id"],
            start_date=raw["start_date"],
            end_date=raw["end_date"],
            sharpe=raw["sharpe"],
            total_return=raw["total_return"],
            max_drawdown=raw["max_drawdown"],
            n_trades=raw["n_trades"],
            win_rate=raw["win_rate"],
            annualized_return=raw["annualized_return"],
        )
        fold_metrics.append(fm)

        # Collect OOS equity curve (normalised to 1.0 at fold start)
        daily_nav = raw.get("daily_nav", [])
        if daily_nav:
            first_nav = daily_nav[0][1]
            if first_nav > 0:
                for date, nav in daily_nav:
                    all_oos_nav.append((date, nav / first_nav))

    # Aggregate
    if fold_metrics:
        sharpes = [f.sharpe for f in fold_metrics]
        aggregate = {
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "n_folds": len(fold_metrics),
            "n_positive_sharpe": sum(1 for s in sharpes if s > 0),
            "mean_return": float(np.mean([f.total_return for f in fold_metrics])),
            "mean_max_dd": float(np.mean([f.max_drawdown for f in fold_metrics])),
            "total_trades": sum(f.n_trades for f in fold_metrics),
        }
    else:
        aggregate = {
            "mean_sharpe": 0.0, "std_sharpe": 0.0, "min_sharpe": 0.0,
            "max_sharpe": 0.0, "median_sharpe": 0.0, "n_folds": 0,
            "n_positive_sharpe": 0, "mean_return": 0.0, "mean_max_dd": 0.0,
            "total_trades": 0,
        }

    # Stitch OOS equity curve
    oos_equity = _stitch_oos_equity(all_oos_nav)

    # OOS metrics from stitched curve
    oos_sharpe = _compute_oos_sharpe(oos_equity)
    oos_total_return = (
        (oos_equity[-1][1] / oos_equity[0][1] - 1)
        if len(oos_equity) >= 2 and oos_equity[0][1] > 0
        else 0.0
    )

    return WalkForwardResult(
        fold_metrics=fold_metrics,
        oos_equity_curve=oos_equity,
        aggregate=aggregate,
        oos_sharpe=oos_sharpe,
        oos_total_return=oos_total_return,
        passes=aggregate["mean_sharpe"] >= min_sharpe,
    )


def _stitch_oos_equity(
    all_oos_nav: List[Tuple[Any, float]],
) -> List[Tuple[Any, float]]:
    """
    Stitch normalised per-fold equity curves into a single OOS curve.

    When folds overlap, takes the average NAV for overlapping dates.
    """
    if not all_oos_nav:
        return []

    # Group by date, take mean of normalised NAVs
    by_date: Dict[Any, List[float]] = {}
    for date, nav in all_oos_nav:
        by_date.setdefault(date, []).append(nav)

    stitched = sorted(
        [(date, float(np.mean(navs))) for date, navs in by_date.items()],
        key=lambda x: x[0],
    )

    # Chain into cumulative equity (start at 1.0)
    if not stitched:
        return []

    result = [(stitched[0][0], 1.0)]
    for i in range(1, len(stitched)):
        prev_raw = stitched[i - 1][1]
        curr_raw = stitched[i][1]
        if prev_raw > 0:
            daily_ret = curr_raw / prev_raw - 1
        else:
            daily_ret = 0.0
        result.append((stitched[i][0], result[-1][1] * (1 + daily_ret)))

    return result


def _compute_oos_sharpe(equity: List[Tuple[Any, float]]) -> float:
    """Compute annualised Sharpe from stitched OOS equity curve."""
    if len(equity) < 2:
        return 0.0
    navs = np.array([nav for _, nav in equity])
    daily_returns = np.diff(navs) / navs[:-1]
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        return 0.0
    return float(
        np.mean(daily_returns)
        / np.std(daily_returns, ddof=1)
        * np.sqrt(NEPSE_TRADING_DAYS)
    )
