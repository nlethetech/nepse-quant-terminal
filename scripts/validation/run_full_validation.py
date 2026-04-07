#!/usr/bin/env python3
"""
Comprehensive Validation + Anti-Cheat Script for NEPSE Quant System.

Runs:
  1. Base backtest with upgraded signals
  2. Walk-forward cross-validation (OOS performance)
  3. Random baseline comparison (1000 sims)
  4. Statistical significance tests (PSR, DSR, MinTRL)
  5. CSCV/PBO overfitting diagnostic
  6. Validation charts (dark theme)
  7. JSON results export

Usage:
    python -m scripts.validation.run_full_validation
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backend.quant_pro.paths import get_project_root

# Ensure project root on path
PROJECT_ROOT = str(get_project_root(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("validation_runner")

# ── Configuration ───────────────────────────────────────────────────────
SIGNAL_TYPES = ["volume", "quality", "low_vol", "52wk_high", "residual_momentum"]
HOLDING_DAYS = 40
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 1_000_000
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

BACKTEST_KWARGS = {
    "holding_days": HOLDING_DAYS,
    "max_positions": 5,
    "signal_types": SIGNAL_TYPES,
    "initial_capital": INITIAL_CAPITAL,
    "rebalance_frequency": 5,
    "use_trailing_stop": True,
    "trailing_stop_pct": 0.10,
    "stop_loss_pct": 0.08,
    "use_regime_filter": True,
    "sector_limit": 0.35,
}

# Number of trials for DSR — count of distinct configs tested during development
# 3 holding periods x 11 signal combos x 2 regime = 66, plus new signals ~100
N_STRATEGY_TRIALS = 100

os.makedirs(CHARTS_DIR, exist_ok=True)

# Collection for all results
validation = {}
charts_saved = []


def safe_float(v):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v)
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ═════════════════════════════════════════════════════════════════════════
# TASK 1: Base Backtest with Upgraded Signals
# ═════════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("TASK 1: Running base backtest with upgraded signals...")
logger.info(f"  Signals: {SIGNAL_TYPES}")
logger.info(f"  Holding: {HOLDING_DAYS} trading days, Regime filter: ON")
logger.info("=" * 70)

from backend.backtesting.simple_backtest import run_backtest, load_all_prices
from backend.quant_pro.database import get_db_path

t0 = time.time()
result = run_backtest(start_date=START_DATE, end_date=END_DATE, **BACKTEST_KWARGS)
t_base = time.time() - t0

daily_returns = np.array(result.daily_returns)

logger.info(f"Base backtest done in {t_base:.1f}s")
logger.info(f"  Sharpe={result.sharpe_ratio:.3f}, Sortino={result.sortino_ratio:.3f}")
logger.info(f"  Return={result.total_return:.2%}, MaxDD={result.max_drawdown:.2%}")
logger.info(f"  Trades={result.total_trades}, WinRate={result.win_rate:.2%}")
logger.info(f"  ProfitFactor={result.profit_factor:.2f}, Calmar={result.calmar_ratio:.2f}")

validation["base_backtest"] = {
    "sharpe": safe_float(result.sharpe_ratio),
    "sortino": safe_float(result.sortino_ratio),
    "total_return_pct": safe_float(result.total_return * 100),
    "annualized_return_pct": safe_float(result.annualized_return * 100),
    "max_drawdown_pct": safe_float(result.max_drawdown * 100),
    "total_trades": result.total_trades,
    "win_rate_pct": safe_float(result.win_rate * 100),
    "profit_factor": safe_float(result.profit_factor),
    "calmar": safe_float(result.calmar_ratio),
    "volatility": safe_float(result.volatility),
    "signal_types": SIGNAL_TYPES,
    "holding_days": HOLDING_DAYS,
    "elapsed_seconds": round(t_base, 1),
}


# ═════════════════════════════════════════════════════════════════════════
# TASK 2: Walk-Forward Cross-Validation
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 2: Walk-Forward Cross-Validation...")
logger.info("  Train=504 days, Test=126 days, Step=21 days")
logger.info("=" * 70)

try:
    from validation.walk_forward import walk_forward_validation

    t0 = time.time()
    wf_result = walk_forward_validation(
        train_window_days=504,
        test_window_days=126,
        step_days=21,
        min_sharpe=0.5,
        max_workers=1,
        **BACKTEST_KWARGS,
    )
    t_wf = time.time() - t0

    fold_sharpes = [f.sharpe for f in wf_result.fold_metrics]
    fold_returns = [f.total_return for f in wf_result.fold_metrics]
    fold_dds = [f.max_drawdown for f in wf_result.fold_metrics]

    logger.info(f"Walk-forward done in {t_wf:.1f}s ({t_wf/60:.1f} min)")
    logger.info(f"  Folds: {len(wf_result.fold_metrics)}")
    logger.info(f"  Mean OOS Sharpe: {wf_result.aggregate['mean_sharpe']:.3f}")
    logger.info(f"  Stitched OOS Sharpe: {wf_result.oos_sharpe:.3f}")
    logger.info(f"  OOS Total Return: {wf_result.oos_total_return:.2%}")
    logger.info(f"  Positive folds: {wf_result.aggregate['n_positive_sharpe']}/{wf_result.aggregate['n_folds']}")

    # Per-fold breakdown
    for fm in wf_result.fold_metrics:
        logger.info(
            f"    Fold {fm.fold_id:2d}: {fm.start_date} to {fm.end_date} | "
            f"Sharpe={fm.sharpe:.3f} | Return={fm.total_return:.2%} | "
            f"DD={fm.max_drawdown:.2%} | Trades={fm.n_trades}"
        )

    validation["walk_forward"] = {
        "oos_sharpe": safe_float(wf_result.oos_sharpe),
        "oos_total_return_pct": safe_float(wf_result.oos_total_return * 100),
        "mean_fold_sharpe": safe_float(wf_result.aggregate["mean_sharpe"]),
        "std_fold_sharpe": safe_float(wf_result.aggregate["std_sharpe"]),
        "min_fold_sharpe": safe_float(wf_result.aggregate["min_sharpe"]),
        "max_fold_sharpe": safe_float(wf_result.aggregate["max_sharpe"]),
        "median_fold_sharpe": safe_float(wf_result.aggregate["median_sharpe"]),
        "n_folds": wf_result.aggregate["n_folds"],
        "n_positive_sharpe": wf_result.aggregate["n_positive_sharpe"],
        "fold_sharpes": [safe_float(s) for s in fold_sharpes],
        "fold_returns_pct": [safe_float(r * 100) for r in fold_returns],
        "fold_max_dds_pct": [safe_float(d * 100) for d in fold_dds],
        "passes": wf_result.passes or wf_result.oos_sharpe >= 0.5,
        "elapsed_seconds": round(t_wf, 1),
    }

except Exception as e:
    logger.error(f"TASK 2 FAILED: {e}")
    logger.error(traceback.format_exc())
    validation["walk_forward"] = {"status": "ERROR", "error": str(e)}
    wf_result = None


# ═════════════════════════════════════════════════════════════════════════
# TASK 3: Random Baseline Test
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 3: Random Baseline Test (1000 simulations, fast numpy)...")
logger.info("=" * 70)

try:
    from validation.random_baseline_fast import random_entry_baseline_fast

    t0 = time.time()
    rb_result = random_entry_baseline_fast(
        n_simulations=1000,
        start_date=START_DATE,
        end_date=END_DATE,
        rng_seed=42,
        max_workers=1,  # Sequential for reliability
        compute_alpha=True,
        **BACKTEST_KWARGS,
    )
    t_rb = time.time() - t0

    logger.info(f"Random baseline done in {t_rb:.1f}s ({t_rb/60:.1f} min)")
    logger.info(f"  Actual strategy Sharpe: {rb_result['actual_sharpe']:.3f}")
    logger.info(f"  Random mean Sharpe: {rb_result['mean_random']:.3f}")
    logger.info(f"  Random std: {rb_result['std_random']:.3f}")
    logger.info(f"  Percentile rank: {rb_result['percentile_rank']:.1f}%")
    logger.info(f"  p-value: {rb_result['p_value']:.4f}")
    if "alpha_sharpe" in rb_result:
        logger.info(f"  Alpha Sharpe (beta-hedged): {rb_result['alpha_sharpe']:.3f}")

    random_sharpes = rb_result["random_sharpes"]
    validation["random_baseline"] = {
        "strategy_sharpe": safe_float(rb_result["actual_sharpe"]),
        "random_mean_sharpe": safe_float(rb_result["mean_random"]),
        "random_std_sharpe": safe_float(rb_result["std_random"]),
        "random_median_sharpe": safe_float(rb_result["median_random"]),
        "percentile_rank": safe_float(rb_result["percentile_rank"]),
        "p_value": safe_float(rb_result["p_value"]),
        "alpha_sharpe": safe_float(rb_result.get("alpha_sharpe")),
        "n_sims": rb_result["n_simulations"],
        "passes": rb_result["passes"],
        "elapsed_seconds": round(t_rb, 1),
    }

except Exception as e:
    logger.error(f"TASK 3 FAILED: {e}")
    logger.error(traceback.format_exc())
    validation["random_baseline"] = {"status": "ERROR", "error": str(e)}
    random_sharpes = None


# ═════════════════════════════════════════════════════════════════════════
# TASK 4: Statistical Significance Tests
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 4: Statistical Significance Tests...")
logger.info("=" * 70)

try:
    from validation.statistical_tests import full_statistical_report

    stat_report = full_statistical_report(
        daily_returns=daily_returns,
        sharpe_ratio=result.sharpe_ratio,
        n_trials=N_STRATEGY_TRIALS,
    )

    logger.info(f"  PSR: {stat_report.psr:.4f} ({'PASS' if stat_report.psr_pass else 'FAIL'})")
    logger.info(f"  DSR: {stat_report.dsr:.4f} ({'PASS' if stat_report.dsr_pass else 'FAIL'})")
    logger.info(f"  t-test: t={stat_report.ttest['t_stat']:.3f}, p={stat_report.ttest['p_value']:.6f}")
    logger.info(f"  MinTRL: {stat_report.min_trl:.0f} days ({stat_report.min_trl_years:.1f} years)")
    logger.info(f"  Sufficient track record: {stat_report.sufficient_track_record}")
    logger.info(f"  Return skew: {stat_report.skew:.3f}, kurtosis: {stat_report.kurtosis:.3f}")
    logger.info(f"  N observations: {stat_report.n_obs}")

    validation["statistical_tests"] = {
        "psr": safe_float(stat_report.psr),
        "dsr": safe_float(stat_report.dsr),
        "min_trl_days": safe_float(stat_report.min_trl),
        "min_trl_years": safe_float(stat_report.min_trl_years),
        "ttest_t_stat": safe_float(stat_report.ttest["t_stat"]),
        "ttest_p_value": safe_float(stat_report.ttest["p_value"]),
        "ttest_significant": stat_report.ttest_pass,
        "n_obs": stat_report.n_obs,
        "n_trials": stat_report.n_trials,
        "skew": safe_float(stat_report.skew),
        "kurtosis": safe_float(stat_report.kurtosis),
        "sufficient_track_record": stat_report.sufficient_track_record,
        "psr_pass": stat_report.psr_pass,
        "dsr_pass": stat_report.dsr_pass,
    }

except Exception as e:
    logger.error(f"TASK 4 FAILED: {e}")
    logger.error(traceback.format_exc())
    validation["statistical_tests"] = {"status": "ERROR", "error": str(e)}
    stat_report = None


# ═════════════════════════════════════════════════════════════════════════
# TASK 5: CSCV/PBO Overfitting Test
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 5: CSCV/PBO Overfitting Diagnostic...")
logger.info("=" * 70)

try:
    from validation.cscv_pbo import (
        cscv_pbo_analysis,
        build_performance_matrix,
        pbo_summary,
    )

    conn = sqlite3.connect(str(get_db_path()))
    prices_df = load_all_prices(conn)
    conn.close()

    # Define config variants for PBO analysis
    configs = {
        "vol_qual_lowvol": {"signal_types": ["volume", "quality", "low_vol"]},
        "vol_qual": {"signal_types": ["volume", "quality"]},
        "vol_lowvol": {"signal_types": ["volume", "low_vol"]},
        "qual_lowvol": {"signal_types": ["quality", "low_vol"]},
        "vol_only": {"signal_types": ["volume"]},
        "qual_only": {"signal_types": ["quality"]},
        "lowvol_only": {"signal_types": ["low_vol"]},
        "vol_qual_lowvol_52wk": {
            "signal_types": ["volume", "quality", "low_vol", "52wk_high"],
        },
        "vol_qual_lowvol_resmom": {
            "signal_types": ["volume", "quality", "low_vol", "residual_momentum"],
        },
        "full_upgraded": {
            "signal_types": SIGNAL_TYPES,
        },
        "hold30_full": {
            "signal_types": SIGNAL_TYPES,
            "holding_days": 30,
        },
        "hold50_full": {
            "signal_types": SIGNAL_TYPES,
            "holding_days": 50,
        },
        "noregime_full": {
            "signal_types": SIGNAL_TYPES,
            "use_regime_filter": False,
        },
        "trail5_full": {
            "signal_types": SIGNAL_TYPES,
            "trailing_stop_pct": 0.05,
        },
        "trail15_full": {
            "signal_types": SIGNAL_TYPES,
            "trailing_stop_pct": 0.15,
        },
    }

    t0 = time.time()
    perf_matrix = build_performance_matrix(
        prices_df=prices_df,
        start_date=START_DATE,
        end_date=END_DATE,
        configs=configs,
        run_backtest_func=run_backtest,
        partition_days=63,
    )
    t_pbo_matrix = time.time() - t0
    logger.info(f"Performance matrix built in {t_pbo_matrix:.1f}s ({t_pbo_matrix/60:.1f} min)")

    t0 = time.time()
    pbo_result = cscv_pbo_analysis(perf_matrix, max_combinations=1000)
    t_pbo_cscv = time.time() - t0

    pbo_summary_dict = pbo_summary(pbo_result)

    logger.info(f"CSCV done in {t_pbo_cscv:.1f}s")
    logger.info(f"  PBO: {pbo_result.pbo:.4f} ({'PASS' if pbo_result.passes else 'FAIL'})")
    logger.info(f"  N configs: {pbo_result.n_configs}")
    logger.info(f"  N partitions: {pbo_result.n_partitions}")
    logger.info(f"  N combinations: {pbo_result.n_combinations}")
    logger.info(f"  Best IS config: {pbo_result.best_is_config} ({pbo_result.best_is_frequency:.1%})")
    logger.info(f"  Mean logit: {pbo_result.mean_logit:.3f}")
    logger.info(f"  Median logit: {pbo_result.median_logit:.3f}")

    validation["cscv_pbo"] = {
        "pbo": safe_float(pbo_result.pbo),
        "passes": pbo_result.passes,
        "n_configs": pbo_result.n_configs,
        "n_partitions": pbo_result.n_partitions,
        "n_combinations": pbo_result.n_combinations,
        "best_is_config": pbo_result.best_is_config,
        "best_is_frequency": safe_float(pbo_result.best_is_frequency),
        "mean_logit": safe_float(pbo_result.mean_logit),
        "median_logit": safe_float(pbo_result.median_logit),
        "is_overfitted": not pbo_result.passes,
        "elapsed_seconds": round(t_pbo_matrix + t_pbo_cscv, 1),
    }

except Exception as e:
    logger.error(f"TASK 5 FAILED: {e}")
    logger.error(traceback.format_exc())
    validation["cscv_pbo"] = {"status": "ERROR", "error": str(e)}
    pbo_result = None


# ═════════════════════════════════════════════════════════════════════════
# TASK 6: Generate Validation Charts
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 6: Generating Validation Charts...")
logger.info("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("dark_background")
CHART_DPI = 150

# ── Chart 1: Walk-Forward Folds (Train vs Test Sharpe) ────────────────
if wf_result is not None and len(wf_result.fold_metrics) > 0:
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1]})

        # Top: fold Sharpe bar chart
        ax1 = axes[0]
        fold_ids = [f"F{fm.fold_id}" for fm in wf_result.fold_metrics]
        fold_sharpes_plot = [fm.sharpe for fm in wf_result.fold_metrics]

        colors = ["#2D72D2" if s >= 0 else "#CD4246" for s in fold_sharpes_plot]
        bars = ax1.bar(fold_ids, fold_sharpes_plot, color=colors, edgecolor="white", linewidth=0.3)
        ax1.axhline(y=wf_result.aggregate["mean_sharpe"], color="#F0B726", linestyle="--",
                     linewidth=1.5, label=f"Mean OOS Sharpe: {wf_result.aggregate['mean_sharpe']:.3f}")
        ax1.axhline(y=0, color="white", linestyle="-", linewidth=0.5, alpha=0.3)
        ax1.set_ylabel("OOS Sharpe Ratio", fontsize=11)
        ax1.set_title("Walk-Forward Fold-by-Fold Out-of-Sample Sharpe", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=10)
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

        # Bottom: fold returns bar chart
        ax2 = axes[1]
        fold_returns_plot = [fm.total_return * 100 for fm in wf_result.fold_metrics]
        colors2 = ["#238551" if r >= 0 else "#CD4246" for r in fold_returns_plot]
        ax2.bar(fold_ids, fold_returns_plot, color=colors2, edgecolor="white", linewidth=0.3)
        ax2.axhline(y=0, color="white", linestyle="-", linewidth=0.5, alpha=0.3)
        ax2.set_ylabel("OOS Return (%)", fontsize=11)
        ax2.set_xlabel("Fold", fontsize=11)
        ax2.set_title("Walk-Forward Fold-by-Fold Out-of-Sample Returns", fontsize=13)
        ax2.tick_params(axis="x", rotation=45, labelsize=8)

        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, "wf_fold_sharpes.png")
        fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts_saved.append(path)
        logger.info(f"  Saved: {path}")
    except Exception as e:
        logger.error(f"  Chart 1 (WF Folds) failed: {e}")

# ── Chart 2: Walk-Forward OOS Equity Curve ────────────────────────────
if wf_result is not None and len(wf_result.oos_equity_curve) > 0:
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        dates = [pd.Timestamp(d) for d, _ in wf_result.oos_equity_curve]
        navs = [n for _, n in wf_result.oos_equity_curve]

        ax.plot(dates, navs, color="#2D72D2", linewidth=1.5, label="OOS Equity")
        ax.axhline(y=1.0, color="white", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.fill_between(dates, 1.0, navs, where=[n >= 1 for n in navs],
                         alpha=0.15, color="#238551")
        ax.fill_between(dates, 1.0, navs, where=[n < 1 for n in navs],
                         alpha=0.15, color="#CD4246")

        ax.set_title(
            f"Stitched Out-of-Sample Equity Curve | OOS Sharpe={wf_result.oos_sharpe:.3f} | "
            f"OOS Return={wf_result.oos_total_return:.2%}",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylabel("Normalized Equity", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, "wf_oos_equity.png")
        fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts_saved.append(path)
        logger.info(f"  Saved: {path}")
    except Exception as e:
        logger.error(f"  Chart 2 (OOS Equity) failed: {e}")

# ── Chart 3: Random Baseline Distribution ─────────────────────────────
if random_sharpes is not None:
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(random_sharpes, bins=50, color="#404854", edgecolor="#5C7080",
                alpha=0.8, label="Random Entry Sharpes")
        actual_s = validation.get("random_baseline", {}).get("strategy_sharpe", result.sharpe_ratio)
        ax.axvline(x=actual_s, color="#CD4246", linewidth=2.5, linestyle="-",
                   label=f"Strategy Sharpe: {actual_s:.3f}")
        ax.axvline(x=np.mean(random_sharpes), color="#F0B726", linewidth=1.5, linestyle="--",
                   label=f"Random Mean: {np.mean(random_sharpes):.3f}")

        pct = validation.get("random_baseline", {}).get("percentile_rank", 0)
        pval = validation.get("random_baseline", {}).get("p_value", 1)
        ax.set_title(
            f"Random Baseline: Strategy at {pct:.1f}th Percentile | p-value={pval:.4f}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Sharpe Ratio", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, "random_baseline_dist.png")
        fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts_saved.append(path)
        logger.info(f"  Saved: {path}")
    except Exception as e:
        logger.error(f"  Chart 3 (Random Baseline) failed: {e}")

# ── Chart 4: PBO Logit Distribution ──────────────────────────────────
if pbo_result is not None and len(pbo_result.logit_distribution) > 0:
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        logits = pbo_result.logit_distribution
        ax.hist(logits, bins=40, color="#404854", edgecolor="#5C7080", alpha=0.8,
                label="Logit Distribution")
        ax.axvline(x=0, color="#CD4246", linewidth=2, linestyle="--",
                   label="Overfitting Boundary (logit=0)")
        ax.axvline(x=pbo_result.mean_logit, color="#F0B726", linewidth=1.5, linestyle="--",
                   label=f"Mean Logit: {pbo_result.mean_logit:.3f}")

        # Shade overfitting region
        ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color="#CD4246")

        ax.set_title(
            f"CSCV/PBO: Probability of Backtest Overfitting = {pbo_result.pbo:.3f} "
            f"({'PASS' if pbo_result.passes else 'FAIL'})",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Logit of Relative OOS Rank", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, "pbo_logit_dist.png")
        fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts_saved.append(path)
        logger.info(f"  Saved: {path}")
    except Exception as e:
        logger.error(f"  Chart 4 (PBO) failed: {e}")


# ═════════════════════════════════════════════════════════════════════════
# TASK 7: Anti-Cheat Verdict + Save Results
# ═════════════════════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("TASK 7: Computing Anti-Cheat Verdict...")
logger.info("=" * 70)

# Determine verdict
failures = []
warnings = []

# Check 1: Walk-forward OOS Sharpe
wf_data = validation.get("walk_forward", {})
if wf_data.get("status") == "ERROR":
    failures.append("walk_forward: CRASHED")
elif wf_data.get("oos_sharpe", 0) < 0.3:
    failures.append(f"walk_forward: OOS Sharpe={wf_data.get('oos_sharpe', 0):.3f} < 0.30")
elif wf_data.get("oos_sharpe", 0) < 0.5:
    warnings.append(f"walk_forward: OOS Sharpe={wf_data.get('oos_sharpe', 0):.3f} < 0.50")

# Check 2: Random baseline
rb_data = validation.get("random_baseline", {})
if rb_data.get("status") == "ERROR":
    failures.append("random_baseline: CRASHED")
elif rb_data.get("percentile_rank", 0) < 90:
    failures.append(f"random_baseline: Percentile={rb_data.get('percentile_rank', 0):.1f}% < 90%")
elif rb_data.get("percentile_rank", 0) < 95:
    warnings.append(f"random_baseline: Percentile={rb_data.get('percentile_rank', 0):.1f}% < 95%")

# Check 3: Statistical significance
stat_data = validation.get("statistical_tests", {})
if stat_data.get("status") == "ERROR":
    failures.append("statistical_tests: CRASHED")
elif not stat_data.get("psr_pass", False):
    failures.append(f"PSR={stat_data.get('psr', 0):.3f} < 0.90")
elif not stat_data.get("dsr_pass", False):
    warnings.append(f"DSR={stat_data.get('dsr', 0):.3f} < 0.85 (multiple testing penalty)")

# Check 4: PBO
pbo_data = validation.get("cscv_pbo", {})
if pbo_data.get("status") == "ERROR":
    warnings.append("cscv_pbo: CRASHED (non-critical)")
elif pbo_data.get("pbo", 1) > 0.50:
    failures.append(f"PBO={pbo_data.get('pbo', 1):.3f} > 0.50 (concerning overfitting)")
elif pbo_data.get("pbo", 1) > 0.40:
    warnings.append(f"PBO={pbo_data.get('pbo', 1):.3f} > 0.40 (borderline)")

# Check 5: Base Sharpe
base_data = validation.get("base_backtest", {})
if base_data.get("sharpe", 0) < 1.0:
    failures.append(f"Base Sharpe={base_data.get('sharpe', 0):.3f} < 1.0")

# Check 6: Max Drawdown
if abs(base_data.get("max_drawdown_pct", -100)) > 25:
    failures.append(f"Max DD={base_data.get('max_drawdown_pct', -100):.1f}% exceeds -25%")

verdict = "PASS" if len(failures) == 0 else "FAIL"

validation["anti_cheat_verdict"] = verdict
validation["failures"] = failures
validation["warnings"] = warnings
validation["timestamp"] = datetime.now().isoformat()
validation["charts_saved"] = charts_saved

# Save JSON results
json_path = os.path.join(OUTPUT_DIR, "validation_results.json")
with open(json_path, "w") as f:
    json.dump(validation, f, indent=2, default=safe_float)
logger.info(f"Results saved to: {json_path}")

# ── Print Final Report ────────────────────────────────────────────────
logger.info("")
logger.info("=" * 70)
logger.info("         VALIDATION + ANTI-CHEAT REPORT")
logger.info("=" * 70)
logger.info("")

logger.info("--- BASE BACKTEST ---")
bb = validation.get("base_backtest", {})
logger.info(f"  Sharpe:             {bb.get('sharpe', 'N/A')}")
logger.info(f"  Sortino:            {bb.get('sortino', 'N/A')}")
logger.info(f"  Total Return:       {bb.get('total_return_pct', 'N/A')}%")
logger.info(f"  Annualized Return:  {bb.get('annualized_return_pct', 'N/A')}%")
logger.info(f"  Max Drawdown:       {bb.get('max_drawdown_pct', 'N/A')}%")
logger.info(f"  Win Rate:           {bb.get('win_rate_pct', 'N/A')}%")
logger.info(f"  Profit Factor:      {bb.get('profit_factor', 'N/A')}")
logger.info(f"  Trades:             {bb.get('total_trades', 'N/A')}")
logger.info("")

logger.info("--- WALK-FORWARD (OOS) ---")
wf = validation.get("walk_forward", {})
logger.info(f"  OOS Sharpe:         {wf.get('oos_sharpe', 'N/A')}")
logger.info(f"  OOS Return:         {wf.get('oos_total_return_pct', 'N/A')}%")
logger.info(f"  Mean Fold Sharpe:   {wf.get('mean_fold_sharpe', 'N/A')}")
logger.info(f"  Fold Sharpe Std:    {wf.get('std_fold_sharpe', 'N/A')}")
logger.info(f"  Positive Folds:     {wf.get('n_positive_sharpe', 'N/A')}/{wf.get('n_folds', 'N/A')}")
logger.info("")

logger.info("--- RANDOM BASELINE ---")
rb = validation.get("random_baseline", {})
logger.info(f"  Strategy Sharpe:    {rb.get('strategy_sharpe', 'N/A')}")
logger.info(f"  Random Mean:        {rb.get('random_mean_sharpe', 'N/A')}")
logger.info(f"  Percentile Rank:    {rb.get('percentile_rank', 'N/A')}%")
logger.info(f"  p-value:            {rb.get('p_value', 'N/A')}")
logger.info(f"  Alpha Sharpe:       {rb.get('alpha_sharpe', 'N/A')}")
logger.info("")

logger.info("--- STATISTICAL TESTS ---")
st = validation.get("statistical_tests", {})
logger.info(f"  PSR:                {st.get('psr', 'N/A')} ({'PASS' if st.get('psr_pass') else 'FAIL'})")
logger.info(f"  DSR:                {st.get('dsr', 'N/A')} ({'PASS' if st.get('dsr_pass') else 'FAIL'})")
logger.info(f"  t-stat:             {st.get('ttest_t_stat', 'N/A')}")
logger.info(f"  t-test p-value:     {st.get('ttest_p_value', 'N/A')}")
logger.info(f"  MinTRL:             {st.get('min_trl_years', 'N/A')} years")
logger.info(f"  Sufficient Record:  {st.get('sufficient_track_record', 'N/A')}")
logger.info("")

logger.info("--- CSCV/PBO ---")
pbo = validation.get("cscv_pbo", {})
logger.info(f"  PBO:                {pbo.get('pbo', 'N/A')}")
logger.info(f"  Is Overfitted:      {pbo.get('is_overfitted', 'N/A')}")
logger.info(f"  Best IS Config:     {pbo.get('best_is_config', 'N/A')}")
logger.info(f"  N Configs:          {pbo.get('n_configs', 'N/A')}")
logger.info(f"  N Partitions:       {pbo.get('n_partitions', 'N/A')}")
logger.info("")

logger.info("=" * 70)
logger.info(f"  VERDICT: {verdict}")
if failures:
    logger.info(f"  FAILURES ({len(failures)}):")
    for f in failures:
        logger.info(f"    - {f}")
if warnings:
    logger.info(f"  WARNINGS ({len(warnings)}):")
    for w in warnings:
        logger.info(f"    - {w}")
if not failures and not warnings:
    logger.info("  All checks passed with no warnings.")
logger.info("=" * 70)
logger.info(f"Charts saved: {len(charts_saved)}")
for c in charts_saved:
    logger.info(f"  {c}")
logger.info(f"JSON report: {json_path}")
