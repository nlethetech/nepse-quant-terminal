#!/usr/bin/env python3
"""Run random baseline with 25 sims (standalone)."""
import sys
import os
import logging
from backend.quant_pro.paths import get_project_root

PROJECT = str(get_project_root(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import numpy as np
from validation.random_baseline import random_entry_baseline

result = random_entry_baseline(
    n_simulations=25,
    start_date="2020-01-01",
    end_date="2025-12-31",
    max_workers=1,
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

print()
print("=" * 60)
print("RANDOM BASELINE RESULTS (25 sims)")
print("=" * 60)
print(f'Actual Strategy Sharpe: {result["actual_sharpe"]:.3f}')
print(f'Random Mean Sharpe:     {result["mean_random"]:.3f}')
print(f'Random Std:             {result["std_random"]:.3f}')
print(f'Random Median:          {result["median_random"]:.3f}')
print(f'Percentile Rank:        {result["percentile_rank"]:.1f}%')
print(f'p-value:                {result["p_value"]:.4f}')
print(f'PASS:                   {result["passes"]}')
print()
rs = result["random_sharpes"]
print(f"Random Sharpe range: [{np.min(rs):.3f}, {np.max(rs):.3f}]")
print(f'Random Sharpes > Actual: {np.sum(rs >= result["actual_sharpe"])}/{len(rs)}')
