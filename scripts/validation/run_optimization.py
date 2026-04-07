#!/usr/bin/env python3
"""
Master optimization script for NEPSE trading system.

Tests multiple configurations through full backtest + walk-forward validation
to find the optimal config that achieves a GO verdict.

Configs tested:
  A: Graduated regime positions (vol+qual+lowvol, bull:5/neutral:4/bear:1)
  B: +Mean reversion (vol+qual+lowvol+mean_reversion, bull:5/neutral:4/bear:2)
  C: +Mean reversion +XSec momentum (bull:5/neutral:4/bear:2)
  D: Conservative (vol+qual+lowvol+mean_reversion, bull:5/neutral:3/bear:1)

Usage:
    python -m scripts.validation.run_optimization
    python -m scripts.validation.run_optimization --fast          # Skip walk-forward, just backtest
    python -m scripts.validation.run_optimization --configs A B   # Test specific configs
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from backend.quant_pro.paths import get_project_root

# Ensure project root on path
PROJECT_ROOT = str(get_project_root(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration definitions
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "A": {
        "name": "Graduated (baseline signals)",
        "signal_types": ["volume", "quality", "low_vol"],
        "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 1},
        "holding_days": 40,
        "max_positions": 5,
        "initial_capital": 1_000_000,
        "rebalance_frequency": 5,
        "use_trailing_stop": True,
        "trailing_stop_pct": 0.10,
        "stop_loss_pct": 0.08,
        "use_regime_filter": True,
        "sector_limit": 0.35,
    },
    "B": {
        "name": "+Mean Reversion",
        "signal_types": ["volume", "quality", "low_vol", "mean_reversion"],
        "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
        "holding_days": 40,
        "max_positions": 5,
        "initial_capital": 1_000_000,
        "rebalance_frequency": 5,
        "use_trailing_stop": True,
        "trailing_stop_pct": 0.10,
        "stop_loss_pct": 0.08,
        "use_regime_filter": True,
        "sector_limit": 0.35,
    },
    "C": {
        "name": "+Mean Reversion +XSec Momentum",
        "signal_types": ["volume", "quality", "low_vol", "mean_reversion", "xsec_momentum"],
        "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
        "holding_days": 40,
        "max_positions": 5,
        "initial_capital": 1_000_000,
        "rebalance_frequency": 5,
        "use_trailing_stop": True,
        "trailing_stop_pct": 0.10,
        "stop_loss_pct": 0.08,
        "use_regime_filter": True,
        "sector_limit": 0.35,
    },
    "D": {
        "name": "Conservative (+MeanRev, tighter limits)",
        "signal_types": ["volume", "quality", "low_vol", "mean_reversion"],
        "regime_max_positions": {"bull": 5, "neutral": 3, "bear": 1},
        "holding_days": 40,
        "max_positions": 5,
        "initial_capital": 1_000_000,
        "rebalance_frequency": 5,
        "use_trailing_stop": True,
        "trailing_stop_pct": 0.10,
        "stop_loss_pct": 0.08,
        "use_regime_filter": True,
        "sector_limit": 0.35,
    },
}

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"


def run_single_config(
    config_id: str,
    config: dict,
    run_wf: bool = True,
    wf_step_days: int = 42,
) -> dict:
    """Run backtest + optional walk-forward for a single config."""
    from backend.backtesting.simple_backtest import run_backtest
    from validation.walk_forward import walk_forward_validation

    name = config.pop("name", config_id)
    logger.info("=" * 70)
    logger.info(f"CONFIG {config_id}: {name}")
    logger.info(f"  Signals: {config['signal_types']}")
    logger.info(f"  Regime positions: {config.get('regime_max_positions', 'N/A')}")
    logger.info("=" * 70)

    result_data = {
        "config_id": config_id,
        "name": name,
        "config": {k: v for k, v in config.items()},
    }

    # Phase 1: Full backtest
    t0 = time.time()
    bt_result = run_backtest(start_date=START_DATE, end_date=END_DATE, **config)
    bt_elapsed = time.time() - t0

    result_data["backtest"] = {
        "sharpe": bt_result.sharpe_ratio,
        "sortino": bt_result.sortino_ratio,
        "total_return": bt_result.total_return,
        "annualized_return": bt_result.annualized_return,
        "max_drawdown": bt_result.max_drawdown,
        "total_trades": bt_result.total_trades,
        "win_rate": bt_result.win_rate,
        "profit_factor": bt_result.profit_factor if bt_result.profit_factor != float("inf") else 99.9,
        "calmar": bt_result.calmar_ratio,
        "elapsed_seconds": bt_elapsed,
    }

    logger.info(f"  Backtest: Sharpe={bt_result.sharpe_ratio:.3f}, "
                f"Return={bt_result.total_return:.2%}, "
                f"MaxDD={bt_result.max_drawdown:.2%}, "
                f"Trades={bt_result.total_trades} "
                f"({bt_elapsed:.0f}s)")

    # Phase 2: Walk-forward
    if run_wf:
        t0 = time.time()
        try:
            wf_result = walk_forward_validation(
                train_window_days=504,
                test_window_days=126,
                step_days=wf_step_days,
                min_sharpe=0.5,
                max_workers=1,
                **config,
            )
            wf_elapsed = time.time() - t0

            result_data["walk_forward"] = {
                "mean_sharpe": wf_result.aggregate["mean_sharpe"],
                "std_sharpe": wf_result.aggregate["std_sharpe"],
                "oos_sharpe": wf_result.oos_sharpe,
                "oos_total_return": wf_result.oos_total_return,
                "n_folds": wf_result.aggregate["n_folds"],
                "n_positive": wf_result.aggregate["n_positive_sharpe"],
                "passes": wf_result.passes or wf_result.oos_sharpe >= 0.5,
                "fold_sharpes": [f.sharpe for f in wf_result.fold_metrics],
                "elapsed_seconds": wf_elapsed,
            }

            logger.info(f"  Walk-Forward: Mean OOS Sharpe={wf_result.aggregate['mean_sharpe']:.3f}, "
                        f"Stitched OOS Sharpe={wf_result.oos_sharpe:.3f}, "
                        f"Folds={wf_result.aggregate['n_folds']}, "
                        f"Positive={wf_result.aggregate['n_positive_sharpe']}/{wf_result.aggregate['n_folds']} "
                        f"({'PASS' if result_data['walk_forward']['passes'] else 'FAIL'}) "
                        f"({wf_elapsed:.0f}s)")
        except Exception as e:
            logger.error(f"  Walk-Forward FAILED: {e}")
            result_data["walk_forward"] = {"error": str(e), "passes": False}
    else:
        result_data["walk_forward"] = {"skipped": True}

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Trading System — Configuration Optimization"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGS.keys()),
        help="Config IDs to test (default: all)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip walk-forward (backtest only)"
    )
    parser.add_argument(
        "--output", default="reports/optimization",
        help="Output directory"
    )
    parser.add_argument(
        "--wf-step", type=int, default=42,
        help="Walk-forward step size in days (42=fast, 21=full)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    all_results = []

    for config_id in args.configs:
        if config_id not in CONFIGS:
            logger.error(f"Unknown config: {config_id}. Available: {list(CONFIGS.keys())}")
            continue

        config = {**CONFIGS[config_id]}  # Deep copy
        try:
            result = run_single_config(
                config_id, config, run_wf=not args.fast, wf_step_days=args.wf_step
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Config {config_id} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "config_id": config_id,
                "error": str(e),
            })

    # Rank by walk-forward OOS Sharpe (or backtest Sharpe if WF skipped)
    def sort_key(r):
        wf = r.get("walk_forward", {})
        if wf.get("oos_sharpe") is not None:
            return wf["oos_sharpe"]
        bt = r.get("backtest", {})
        return bt.get("sharpe", -999)

    all_results.sort(key=sort_key, reverse=True)

    # Summary
    elapsed_total = time.time() - t_total
    print()
    print("=" * 80)
    print("OPTIMIZATION RESULTS (ranked by Walk-Forward OOS Sharpe)")
    print("=" * 80)
    print(f"{'Rank':<5} {'ID':<4} {'Name':<40} {'BT Sharpe':>10} {'WF OOS Sharpe':>14} {'WF Pass':>8}")
    print("-" * 80)
    for rank, r in enumerate(all_results, 1):
        config_id = r.get("config_id", "?")
        name = r.get("name", "?")[:38]
        bt_sharpe = r.get("backtest", {}).get("sharpe", float("nan"))
        wf_sharpe = r.get("walk_forward", {}).get("oos_sharpe", float("nan"))
        wf_pass = r.get("walk_forward", {}).get("passes", False)
        print(f"{rank:<5} {config_id:<4} {name:<40} {bt_sharpe:>10.3f} {wf_sharpe:>14.3f} {'PASS' if wf_pass else 'FAIL':>8}")
    print("-" * 80)
    print(f"Total elapsed: {elapsed_total / 3600:.2f} hours ({elapsed_total / 60:.1f} min)")
    print()

    winner = all_results[0] if all_results else None
    if winner:
        print(f"WINNER: Config {winner['config_id']} — {winner.get('name', '?')}")
        wf = winner.get("walk_forward", {})
        bt = winner.get("backtest", {})
        print(f"  Backtest Sharpe: {bt.get('sharpe', 0):.3f}")
        print(f"  WF OOS Sharpe:   {wf.get('oos_sharpe', 0):.3f}")
        print(f"  WF Passes:       {wf.get('passes', False)}")
        cfg = winner.get("config", {})
        signals = cfg.get("signal_types", [])
        regime_pos = cfg.get("regime_max_positions", {})
        print(f"\n  Recommended validation command:")
        print(f"  python -m validation.run_all --fast --fast-baseline \\")
        print(f"    --signals {' '.join(signals)} \\")
        rp_str = ",".join(f"{k}:{v}" for k, v in regime_pos.items())
        print(f'    --regime-positions "{rp_str}" \\')
        print(f"    --output reports/validation_final")

    # Save results
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    output_path = output_dir / "optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "results": make_serializable(all_results),
                "winner": winner["config_id"] if winner else None,
                "elapsed_hours": elapsed_total / 3600,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
