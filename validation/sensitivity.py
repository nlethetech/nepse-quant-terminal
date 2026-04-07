"""
Parameter sensitivity analysis.

Varies each tunable parameter across a range of values while holding
others constant, then checks for smooth Sharpe surfaces. Flags parameters
where a +-10% change causes a Sharpe drop > 0.3 (cliff effect = overfitting).

Supports multiprocessing: each (param, value) backtest runs independently.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default parameter ranges to sweep
DEFAULT_PARAM_RANGES: Dict[str, list] = {
    "holding_days": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80],
    "trailing_stop_pct": [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],
    "stop_loss_pct": [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20],
    "max_positions": [3, 4, 5, 6, 7, 8, 10],
    "rebalance_frequency": [1, 3, 5, 7, 10, 15, 20],
}

# Maximum Sharpe drop for adjacent values before flagging
SMOOTHNESS_THRESHOLD = 0.4


@dataclass
class ParamSensitivity:
    """Sensitivity results for a single parameter."""
    param_name: str
    values: list
    sharpes: List[float]
    returns: List[float]
    max_dds: List[float]
    is_smooth: bool
    max_adjacent_drop: float
    best_value: Any
    best_sharpe: float


def _run_sensitivity_point(args):
    """
    Top-level worker function for multiprocessing (must be picklable).

    Runs a single backtest with one parameter overridden.
    Returns (param_name, val, sharpe, total_return, max_drawdown).
    """
    base_params, param_name, val, start_date, end_date = args

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtesting.simple_backtest import run_backtest

    kwargs = {**base_params}
    kwargs["start_date"] = start_date
    kwargs["end_date"] = end_date
    kwargs[param_name] = val

    try:
        result = run_backtest(**kwargs)
        return (param_name, val, result.sharpe_ratio, result.total_return, result.max_drawdown)
    except Exception as e:
        return (param_name, val, float("nan"), float("nan"), float("nan"))


def parameter_sensitivity(
    base_params: Optional[dict] = None,
    param_ranges: Optional[Dict[str, list]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    smoothness_threshold: float = SMOOTHNESS_THRESHOLD,
    max_workers: int = 1,
) -> Dict[str, ParamSensitivity]:
    """
    For each tunable parameter, vary across values while holding others constant.

    Parameters
    ----------
    base_params : Default backtest params (if None, uses system defaults)
    param_ranges : {param_name: [values]} to sweep (if None, uses defaults)
    start_date : Backtest start date
    end_date : Backtest end date
    smoothness_threshold : Max Sharpe drop between adjacent values before flagging
    max_workers : Number of parallel workers (1 = sequential)

    Returns
    -------
    Dict mapping param_name to ParamSensitivity results
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if base_params is None:
        base_params = {
            "holding_days": 40,
            "max_positions": 5,
            "signal_types": ["volume", "quality", "low_vol"],
            "initial_capital": 1_000_000,
            "rebalance_frequency": 5,
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.10,
            "stop_loss_pct": 0.08,
            "use_regime_filter": True,
            "sector_limit": 0.35,
        }

    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES

    # Flatten all (param, value) pairs into tasks
    all_tasks = []
    for param_name, values in param_ranges.items():
        for val in values:
            all_tasks.append((base_params.copy(), param_name, val, start_date, end_date))

    total_runs = len(all_tasks)
    logger.info(f"Sensitivity analysis: {total_runs} backtest runs across {len(param_ranges)} params")

    # Execute (parallel or sequential)
    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        logger.info(f"Dispatching {total_runs} runs across {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            all_results = list(pool.map(_run_sensitivity_point, all_tasks))
    else:
        from backend.backtesting.simple_backtest import run_backtest

        all_results = []
        for i, task in enumerate(all_tasks):
            _, param_name, val, _, _ = task
            logger.info(f"  [{i+1}/{total_runs}] {param_name}={val}")
            all_results.append(_run_sensitivity_point(task))

    # Reassemble results by param_name
    results_by_param: Dict[str, List] = {}
    for param_name, val, sharpe, total_return, max_dd in all_results:
        results_by_param.setdefault(param_name, []).append((val, sharpe, total_return, max_dd))

    results: Dict[str, ParamSensitivity] = {}

    for param_name, values in param_ranges.items():
        # Sort results by the original value order
        value_to_result = {
            val: (sharpe, ret, dd)
            for val, sharpe, ret, dd in results_by_param.get(param_name, [])
        }

        sharpes = [value_to_result.get(v, (float("nan"), float("nan"), float("nan")))[0] for v in values]
        returns = [value_to_result.get(v, (float("nan"), float("nan"), float("nan")))[1] for v in values]
        max_dds = [value_to_result.get(v, (float("nan"), float("nan"), float("nan")))[2] for v in values]

        for v, s in zip(values, sharpes):
            logger.info(f"  {param_name}={v}: Sharpe={s:.3f}")

        # Check smoothness: max drop between adjacent values
        is_smooth, max_drop = _check_smoothness(sharpes, smoothness_threshold)

        # Best value
        valid_sharpes = [
            (v, s) for v, s in zip(values, sharpes) if not np.isnan(s)
        ]
        if valid_sharpes:
            best_value, best_sharpe = max(valid_sharpes, key=lambda x: x[1])
        else:
            best_value, best_sharpe = values[0], 0.0

        results[param_name] = ParamSensitivity(
            param_name=param_name,
            values=values,
            sharpes=sharpes,
            returns=returns,
            max_dds=max_dds,
            is_smooth=is_smooth,
            max_adjacent_drop=max_drop,
            best_value=best_value,
            best_sharpe=best_sharpe,
        )

    return results


def _check_smoothness(
    sharpes: List[float], threshold: float
) -> tuple:
    """
    Check if the Sharpe surface is smooth (no cliffs).

    Returns (is_smooth, max_adjacent_drop).
    """
    valid = [s for s in sharpes if not np.isnan(s)]
    if len(valid) < 2:
        return True, 0.0

    max_drop = 0.0
    for i in range(len(sharpes) - 1):
        s1, s2 = sharpes[i], sharpes[i + 1]
        if np.isnan(s1) or np.isnan(s2):
            continue
        drop = abs(s1 - s2)
        if drop > max_drop:
            max_drop = drop

    return max_drop <= threshold, max_drop


def sensitivity_summary(results: Dict[str, ParamSensitivity]) -> dict:
    """
    Produce a summary dict for reporting.

    Returns
    -------
    {
        all_smooth: bool,
        flagged_params: list of param names with cliffs,
        param_summaries: {param: {best_value, best_sharpe, smooth, max_drop}}
    }
    """
    flagged = [
        name for name, r in results.items() if not r.is_smooth
    ]
    return {
        "all_smooth": len(flagged) == 0,
        "flagged_params": flagged,
        "param_summaries": {
            name: {
                "best_value": r.best_value,
                "best_sharpe": r.best_sharpe,
                "is_smooth": r.is_smooth,
                "max_adjacent_drop": r.max_adjacent_drop,
            }
            for name, r in results.items()
        },
    }
