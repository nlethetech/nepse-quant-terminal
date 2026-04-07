#!/usr/bin/env python3
"""
Run random baseline with fast numpy-optimized engine.

Default: 10K simulations using all CPU cores.
Falls back to original slow version with --legacy flag.
"""
import sys
import os
import logging
import time
from backend.quant_pro.paths import get_project_root

PROJECT = str(get_project_root(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import numpy as np

BACKTEST_KWARGS = dict(
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

ENHANCED_KWARGS = dict(
    holding_days=40,
    max_positions=5,
    signal_types=["volume", "quality", "low_vol", "mean_reversion", "xsec_momentum"],
    initial_capital=1_000_000,
    rebalance_frequency=5,
    use_trailing_stop=True,
    trailing_stop_pct=0.10,
    stop_loss_pct=0.08,
    use_regime_filter=True,
    sector_limit=0.35,
    regime_max_positions={"bull": 5, "neutral": 4, "bear": 2},
)


def run_fast(n_sims=10000):
    """Run using fast numpy-optimized baseline with multiprocessing."""
    from validation.random_baseline_fast import random_entry_baseline_fast

    return random_entry_baseline_fast(
        n_simulations=n_sims,
        start_date="2020-01-01",
        end_date="2025-12-31",
        max_workers=0,  # auto = cpu_count
        compute_alpha=True,
        **BACKTEST_KWARGS,
    )


def run_legacy(n_sims=1000):
    """Run using original DataFrame-based baseline (slow)."""
    from validation.random_baseline import random_entry_baseline

    return random_entry_baseline(
        n_simulations=n_sims,
        start_date="2020-01-01",
        end_date="2025-12-31",
        max_workers=1,
        **BACKTEST_KWARGS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random baseline runner")
    parser.add_argument("--sims", type=int, default=10000, help="Number of sims (default: 10000)")
    parser.add_argument("--legacy", action="store_true", help="Use slow DataFrame-based version")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced all-weather config")
    args = parser.parse_args()

    if args.enhanced:
        BACKTEST_KWARGS.update(ENHANCED_KWARGS)

    t0 = time.time()

    if args.legacy:
        result = run_legacy(args.sims)
    else:
        result = run_fast(args.sims)

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"RANDOM BASELINE RESULTS ({args.sims} sims, {'legacy' if args.legacy else 'fast'})")
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
    print(f"Elapsed: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
