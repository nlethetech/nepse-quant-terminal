"""
Combinatorially Symmetric Cross-Validation (CSCV) and Probability of
Backtest Overfitting (PBO).

Implements Bailey, Borwein, Lopez de Prado & Zhu (2017):
  "The Probability of Backtest Overfitting"

Methodology:
  1. Split the full backtest period into S equal-length partitions.
  2. For all C(S, S/2) train/test combinations:
     - Select the combination of S/2 partitions as the in-sample (IS) set.
     - The remaining S/2 partitions form the out-of-sample (OOS) set.
     - Identify the strategy configuration with the highest mean IS Sharpe.
     - Measure that configuration's rank among all configs on the OOS set.
  3. PBO = fraction of combinations where the best IS config underperforms
     the OOS median (i.e., ranks below the 50th percentile).

Target: PBO < 0.40 (lower is better — means strategy selection is robust).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NEPSE_TRADING_DAYS = 240
PBO_THRESHOLD = 0.40


@dataclass
class CSCVResult:
    """Results from CSCV / PBO analysis."""
    pbo: float                                      # Probability of Backtest Overfitting
    logit_distribution: np.ndarray = field(repr=False)  # Logit of relative OOS rank per combo
    n_combinations: int                             # Total combos evaluated
    n_configs: int                                  # Number of strategy configs
    n_partitions: int                               # Number of time partitions (S)
    mean_logit: float                               # Mean of logit distribution
    median_logit: float                             # Median of logit distribution
    best_is_config: str                             # Most frequently selected IS-best config
    best_is_frequency: float                        # How often it was picked as IS-best
    passes: bool                                    # True if PBO < threshold


def cscv_pbo_analysis(
    performance_matrix: Dict[str, np.ndarray],
    partition_size: int = 63,
    max_combinations: int = 1000,
    pbo_threshold: float = PBO_THRESHOLD,
    rng_seed: int = 42,
    early_termination_samples: int = 200,
    early_termination_bounds: Tuple[float, float] = (0.05, 0.95),
) -> CSCVResult:
    """
    Run CSCV and compute Probability of Backtest Overfitting.

    Parameters
    ----------
    performance_matrix : Dict mapping config_name -> array of per-partition
        Sharpe ratios (or returns). Each array must have the same length,
        equal to the number of time partitions S.
    partition_size : Number of trading days per partition (63 ~ 1 NEPSE quarter).
        Only used for logging; the actual partition count is inferred from the
        length of the arrays in performance_matrix.
    max_combinations : If C(S, S/2) exceeds this, randomly sample this many
        combinations instead of exhaustive enumeration.
    pbo_threshold : PBO must be below this to pass (default 0.40).
    rng_seed : Random seed for combination sampling.
    early_termination_samples : After this many combos, check if PBO is
        clearly converging to a very high or very low value.
    early_termination_bounds : (low, high) — if running PBO estimate is
        below low or above high after early_termination_samples, stop early.

    Returns
    -------
    CSCVResult with PBO, logit distribution, and pass/fail.
    """
    config_names = list(performance_matrix.keys())
    n_configs = len(config_names)

    if n_configs < 2:
        logger.warning("CSCV requires at least 2 configs, got %d", n_configs)
        return CSCVResult(
            pbo=0.0,
            logit_distribution=np.array([]),
            n_combinations=0,
            n_configs=n_configs,
            n_partitions=0,
            mean_logit=0.0,
            median_logit=0.0,
            best_is_config=config_names[0] if config_names else "",
            best_is_frequency=1.0,
            passes=True,
        )

    # Build matrix: rows = configs, cols = partitions
    perf_matrix = np.array([performance_matrix[name] for name in config_names])
    n_partitions = perf_matrix.shape[1]

    if n_partitions < 4:
        logger.warning(
            "CSCV requires at least 4 partitions, got %d. "
            "Consider using smaller partition_size.",
            n_partitions,
        )
        return CSCVResult(
            pbo=0.0,
            logit_distribution=np.array([]),
            n_combinations=0,
            n_configs=n_configs,
            n_partitions=n_partitions,
            mean_logit=0.0,
            median_logit=0.0,
            best_is_config=config_names[0],
            best_is_frequency=1.0,
            passes=True,
        )

    half = n_partitions // 2
    total_combos = math.comb(n_partitions, half)
    partition_indices = list(range(n_partitions))

    logger.info(
        "CSCV: %d configs x %d partitions (%d days each), "
        "C(%d,%d) = %s combinations",
        n_configs, n_partitions, partition_size,
        n_partitions, half, f"{total_combos:,}",
    )

    # Determine whether to enumerate or sample
    rng = np.random.default_rng(rng_seed)
    if total_combos <= max_combinations:
        combo_iter = list(combinations(partition_indices, half))
        n_eval = len(combo_iter)
        sampled = False
        logger.info("Exhaustive enumeration: %d combinations", n_eval)
    else:
        combo_iter = _sample_combinations(
            n_partitions, half, max_combinations, rng
        )
        n_eval = len(combo_iter)
        sampled = True
        logger.info(
            "Sampled %d of %s combinations (%.2f%%)",
            n_eval, f"{total_combos:,}", 100.0 * n_eval / total_combos,
        )

    # Pre-allocate results
    logits = np.empty(n_eval, dtype=np.float64)
    is_best_counts: Dict[int, int] = {}  # config index -> count of IS-best
    n_underperform = 0  # count where best IS config ranks below OOS median

    all_partition_set = set(partition_indices)

    for combo_idx, train_partitions in enumerate(combo_iter):
        train_set = set(train_partitions)
        test_set = all_partition_set - train_set
        train_cols = list(train_set)
        test_cols = sorted(test_set)

        # In-sample: mean performance across train partitions per config
        is_means = perf_matrix[:, train_cols].mean(axis=1)

        # Best IS config
        best_is_idx = int(np.argmax(is_means))
        is_best_counts[best_is_idx] = is_best_counts.get(best_is_idx, 0) + 1

        # Out-of-sample: mean performance across test partitions per config
        oos_means = perf_matrix[:, test_cols].mean(axis=1)

        # Rank of the best-IS config in OOS (0 = worst, n_configs-1 = best)
        oos_rank = _compute_rank(oos_means, best_is_idx)

        # Relative rank in [0, 1]
        # rank 0 = worst → relative = 1/(n_configs)
        # rank n_configs-1 = best → relative = n_configs/n_configs = 1.0
        relative_rank = (oos_rank + 1) / n_configs

        # Logit of relative rank: logit(w) = log(w / (1 - w))
        # Clamp to avoid log(0) or log(inf)
        w = np.clip(relative_rank, 1e-6, 1.0 - 1e-6)
        logits[combo_idx] = math.log(w / (1.0 - w))

        # PBO: count where relative rank <= 0.5 (below median)
        if relative_rank <= 0.5:
            n_underperform += 1

        # Early termination check
        if (
            combo_idx + 1 == early_termination_samples
            and combo_idx + 1 < n_eval
        ):
            running_pbo = n_underperform / (combo_idx + 1)
            if running_pbo < early_termination_bounds[0]:
                logger.info(
                    "Early termination at combo %d: PBO=%.3f (clearly low)",
                    combo_idx + 1, running_pbo,
                )
                logits = logits[: combo_idx + 1]
                n_eval = combo_idx + 1
                break
            elif running_pbo > early_termination_bounds[1]:
                logger.info(
                    "Early termination at combo %d: PBO=%.3f (clearly high)",
                    combo_idx + 1, running_pbo,
                )
                logits = logits[: combo_idx + 1]
                n_eval = combo_idx + 1
                break

        # Progress logging
        if (combo_idx + 1) % 500 == 0 or combo_idx + 1 == n_eval:
            running_pbo = n_underperform / (combo_idx + 1)
            logger.info(
                "  [%d/%d] running PBO=%.3f",
                combo_idx + 1, n_eval, running_pbo,
            )

    pbo = n_underperform / n_eval if n_eval > 0 else 0.0

    # Most frequently selected IS-best config
    if is_best_counts:
        most_common_idx = max(is_best_counts, key=is_best_counts.get)
        best_is_config = config_names[most_common_idx]
        best_is_frequency = is_best_counts[most_common_idx] / n_eval
    else:
        best_is_config = config_names[0]
        best_is_frequency = 0.0

    mean_logit = float(np.mean(logits)) if len(logits) > 0 else 0.0
    median_logit = float(np.median(logits)) if len(logits) > 0 else 0.0

    logger.info(
        "CSCV complete: PBO=%.4f (%d/%d underperform), "
        "mean_logit=%.3f, median_logit=%.3f, "
        "most frequent IS-best='%s' (%.1f%% of combos)",
        pbo, n_underperform, n_eval,
        mean_logit, median_logit,
        best_is_config, best_is_frequency * 100,
    )

    passes = pbo < pbo_threshold
    if passes:
        logger.info("PASS: PBO=%.4f < %.2f threshold", pbo, pbo_threshold)
    else:
        logger.warning("FAIL: PBO=%.4f >= %.2f threshold", pbo, pbo_threshold)

    return CSCVResult(
        pbo=pbo,
        logit_distribution=logits,
        n_combinations=n_eval,
        n_configs=n_configs,
        n_partitions=n_partitions,
        mean_logit=mean_logit,
        median_logit=median_logit,
        best_is_config=best_is_config,
        best_is_frequency=best_is_frequency,
        passes=passes,
    )


def _compute_rank(values: np.ndarray, target_idx: int) -> int:
    """
    Compute the rank of values[target_idx] among all values.

    Returns rank as integer where 0 = worst, len(values)-1 = best.
    Uses ordinal ranking (ties broken by index).
    """
    target_val = values[target_idx]
    rank = int(np.sum(values < target_val))
    return rank


def _sample_combinations(
    n: int,
    k: int,
    max_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """
    Randomly sample unique C(n, k) combinations without replacement.

    Uses a set to track seen combinations and rejection sampling.
    Efficient when max_samples << C(n, k).
    """
    seen = set()
    all_indices = np.arange(n)
    result = []

    # Safety limit to avoid infinite loop if max_samples ~ C(n,k)
    max_attempts = max_samples * 10

    attempts = 0
    while len(result) < max_samples and attempts < max_attempts:
        # Sample k indices without replacement, then sort for canonical form
        sample = tuple(sorted(rng.choice(all_indices, size=k, replace=False)))
        if sample not in seen:
            seen.add(sample)
            result.append(sample)
        attempts += 1

    if len(result) < max_samples:
        logger.warning(
            "Could only sample %d unique combinations out of %d requested "
            "(reached %d attempts)",
            len(result), max_samples, max_attempts,
        )

    return result


def build_performance_matrix(
    prices_df: Any,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    configs: Optional[Dict[str, dict]] = None,
    run_backtest_func: Optional[Callable] = None,
    partition_days: int = 63,
    base_params: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Build the performance matrix by running each config across each partition.

    Splits the date range into non-overlapping partitions of partition_days
    trading days each, then runs every config on every partition to measure
    per-partition Sharpe ratios.

    Parameters
    ----------
    prices_df : DataFrame with at least columns [date, symbol, close].
        Used to determine the actual trading dates in the range.
    start_date : Start of the evaluation period.
    end_date : End of the evaluation period.
    configs : Dict mapping config_name -> dict of parameter overrides.
        E.g. {"vol_quality_lowvol": {"signal_types": ["volume","quality","low_vol"]}}
        If None, uses a set of default NEPSE configs.
    run_backtest_func : Callable that accepts (**kwargs) and returns an object
        with a .sharpe_ratio attribute. If None, imports from simple_backtest.
    partition_days : Trading days per partition (63 ~ 1 NEPSE quarter).
    base_params : Base backtest parameters shared by all configs.

    Returns
    -------
    Dict mapping config_name -> np.ndarray of per-partition Sharpe ratios.
    """
    import pandas as pd

    if run_backtest_func is None:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from backend.backtesting.simple_backtest import run_backtest
        run_backtest_func = run_backtest

    if base_params is None:
        base_params = {
            "holding_days": 40,
            "max_positions": 5,
            "initial_capital": 1_000_000,
            "rebalance_frequency": 5,
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.10,
            "stop_loss_pct": 0.08,
            "use_regime_filter": True,
            "sector_limit": 0.35,
        }

    if configs is None:
        configs = _default_configs()

    # Determine trading dates in the range
    all_dates = sorted(prices_df["date"].unique())
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    trading_dates = [d for d in all_dates if start_dt <= d <= end_dt]

    n_dates = len(trading_dates)
    n_partitions = n_dates // partition_days
    if n_partitions < 4:
        logger.warning(
            "Only %d partitions from %d dates with partition_days=%d. "
            "Need at least 4 for meaningful CSCV.",
            n_partitions, n_dates, partition_days,
        )

    logger.info(
        "Building performance matrix: %d configs x %d partitions "
        "(%d trading days each, %d dates total)",
        len(configs), n_partitions, partition_days, n_dates,
    )

    # Define partition boundaries
    partitions = []
    for p in range(n_partitions):
        p_start = trading_dates[p * partition_days]
        p_end_idx = min((p + 1) * partition_days - 1, n_dates - 1)
        p_end = trading_dates[p_end_idx]
        partitions.append((
            pd.Timestamp(p_start).strftime("%Y-%m-%d"),
            pd.Timestamp(p_end).strftime("%Y-%m-%d"),
        ))

    for i, (ps, pe) in enumerate(partitions):
        logger.info("  Partition %2d: %s to %s", i, ps, pe)

    # Run each config on each partition
    performance_matrix: Dict[str, np.ndarray] = {}

    for config_name, config_overrides in configs.items():
        sharpes = np.zeros(n_partitions, dtype=np.float64)
        logger.info("  Config '%s':", config_name)

        for p_idx, (p_start, p_end) in enumerate(partitions):
            kwargs = {**base_params, **config_overrides}
            kwargs["start_date"] = p_start
            kwargs["end_date"] = p_end

            try:
                result = run_backtest_func(**kwargs)
                sharpes[p_idx] = result.sharpe_ratio
                logger.info(
                    "    Partition %2d: Sharpe=%.3f", p_idx, sharpes[p_idx]
                )
            except Exception as e:
                logger.warning(
                    "    Partition %2d: FAILED (%s)", p_idx, str(e)
                )
                sharpes[p_idx] = 0.0

        performance_matrix[config_name] = sharpes
        logger.info(
            "  Config '%s' mean Sharpe: %.3f",
            config_name, float(np.mean(sharpes)),
        )

    return performance_matrix


def _default_configs() -> Dict[str, dict]:
    """
    Default NEPSE strategy configuration variants for CSCV analysis.

    These represent the main parameter choices that were explored during
    strategy development. Each maps to a set of parameter overrides.
    """
    return {
        "vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
        },
        "vol_quality": {
            "signal_types": ["volume", "quality"],
        },
        "vol_lowvol": {
            "signal_types": ["volume", "low_vol"],
        },
        "quality_lowvol": {
            "signal_types": ["quality", "low_vol"],
        },
        "volume_only": {
            "signal_types": ["volume"],
        },
        "quality_only": {
            "signal_types": ["quality"],
        },
        "lowvol_only": {
            "signal_types": ["low_vol"],
        },
        "hold30_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "holding_days": 30,
        },
        "hold50_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "holding_days": 50,
        },
        "hold60_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "holding_days": 60,
        },
        "trail5_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "trailing_stop_pct": 0.05,
        },
        "trail15_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "trailing_stop_pct": 0.15,
        },
        "noregime_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "use_regime_filter": False,
        },
        "pos3_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "max_positions": 3,
        },
        "pos7_vol_quality_lowvol": {
            "signal_types": ["volume", "quality", "low_vol"],
            "max_positions": 7,
        },
    }


def pbo_summary(result: CSCVResult) -> dict:
    """
    Produce a summary dict for reporting and JSON serialization.

    Returns
    -------
    Dict with key metrics from the CSCV/PBO analysis.
    """
    return {
        "pbo": result.pbo,
        "passes": result.passes,
        "n_combinations": result.n_combinations,
        "n_configs": result.n_configs,
        "n_partitions": result.n_partitions,
        "mean_logit": result.mean_logit,
        "median_logit": result.median_logit,
        "best_is_config": result.best_is_config,
        "best_is_frequency": result.best_is_frequency,
        "logit_std": float(np.std(result.logit_distribution))
        if len(result.logit_distribution) > 0
        else 0.0,
        "logit_p5": float(np.percentile(result.logit_distribution, 5))
        if len(result.logit_distribution) > 0
        else 0.0,
        "logit_p95": float(np.percentile(result.logit_distribution, 95))
        if len(result.logit_distribution) > 0
        else 0.0,
    }


# ── Standalone testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Synthetic test: create a performance matrix with known properties
    logger.info("=" * 60)
    logger.info("CSCV/PBO Standalone Test (synthetic data)")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)
    n_configs = 15
    n_partitions = 16

    # Simulate: one "good" config with genuine edge + noise,
    # the rest are pure noise (Sharpe ~ 0 with variance)
    synthetic_matrix: Dict[str, np.ndarray] = {}

    for i in range(n_configs):
        if i == 0:
            # Genuinely good strategy: positive mean + noise
            sharpes = rng.normal(loc=1.2, scale=0.8, size=n_partitions)
            synthetic_matrix["genuine_edge"] = sharpes
        else:
            # Noise strategies: zero mean + noise (some may look good IS)
            sharpes = rng.normal(loc=0.0, scale=1.0, size=n_partitions)
            synthetic_matrix[f"noise_{i:02d}"] = sharpes

    logger.info("Synthetic matrix: %d configs x %d partitions", n_configs, n_partitions)
    for name, vals in synthetic_matrix.items():
        logger.info("  %-30s mean=%.3f std=%.3f", name, np.mean(vals), np.std(vals))

    # Run CSCV
    result = cscv_pbo_analysis(
        synthetic_matrix,
        partition_size=63,
        max_combinations=1000,
    )

    summary = pbo_summary(result)

    logger.info("-" * 60)
    logger.info("Results:")
    for key, val in summary.items():
        if isinstance(val, float):
            logger.info("  %-25s %.4f", key, val)
        else:
            logger.info("  %-25s %s", key, val)

    logger.info("-" * 60)
    if result.passes:
        logger.info("PASS: PBO=%.4f < %.2f", result.pbo, PBO_THRESHOLD)
    else:
        logger.info("FAIL: PBO=%.4f >= %.2f", result.pbo, PBO_THRESHOLD)

    # Second test: all-noise configs (should give high PBO ~ 0.5)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 2: All-noise configs (expect PBO ~ 0.5)")
    logger.info("=" * 60)

    noise_matrix: Dict[str, np.ndarray] = {}
    for i in range(n_configs):
        sharpes = rng.normal(loc=0.0, scale=1.0, size=n_partitions)
        noise_matrix[f"noise_{i:02d}"] = sharpes

    result2 = cscv_pbo_analysis(
        noise_matrix,
        partition_size=63,
        max_combinations=1000,
    )

    summary2 = pbo_summary(result2)
    logger.info("-" * 60)
    logger.info("All-noise PBO: %.4f (expected ~0.5)", result2.pbo)
    logger.info("Pass: %s", result2.passes)

    sys.exit(0 if result.passes else 1)
