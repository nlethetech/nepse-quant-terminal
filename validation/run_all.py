"""
Master validation runner.

Single entry point that runs all validation phases and produces
a go/no-go recommendation with JSON + PDF reports.

Usage:
    cd /path/to/Nepse_quant_clean
    python -m validation.run_all
    python -m validation.run_all --fast          # Quick mode (fewer sims)
    python -m validation.run_all --output reports/validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from validation.transaction_costs import TransactionCostModel
from validation.statistical_tests import full_statistical_report, NEPSE_TRADING_DAYS
from validation.monte_carlo import monte_carlo_trade_resample, block_bootstrap_ci
from validation.kill_switch import KillSwitch
from validation.walk_forward import walk_forward_validation
from validation.benchmark import compute_benchmark_series, benchmark_comparison
from validation.sensitivity import parameter_sensitivity, sensitivity_summary
from validation.random_baseline import random_entry_baseline
from validation.random_baseline_fast import random_entry_baseline_fast
from validation.regime_stress import (
    regime_stress_test, circuit_breaker_analysis, settlement_lag_analysis,
)
from validation.slippage import run_backtest_with_slippage
from validation.report_generator import (
    save_json_report, print_console_report, generate_plots, generate_pdf_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Go/No-Go Criteria ────────────────────────────────────────────────────
GO_CRITERIA = {
    "min_sharpe": 1.0,
    "min_psr": 0.90,
    "min_dsr": 0.85,
    "max_drawdown": -0.25,          # Must be less negative than this
    "random_baseline_pct": 95.0,
    "wf_mean_sharpe": 0.45,
    "mc_sharpe_ci_lower": 0.0,
    "regime_max_dd": -0.40,
    "sensitivity_smooth": True,
    "slip_min_adj_sharpe": 0.8,     # Min Sharpe after slippage
    "slip_max_sharpe_impact": -0.5, # Max allowed Sharpe degradation from slippage
}


def _phase_error(tests: dict, name: str, error: Exception):
    """Record a phase error in the tests dict."""
    logger.error(f"  PHASE ERROR in {name}: {error}")
    logger.error(traceback.format_exc())
    tests[name] = {
        "status": "ERROR",
        "summary": f"Phase crashed: {error}",
        "details": {"error": str(error), "traceback": traceback.format_exc()},
    }


def run_full_validation(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    output_dir: str = "reports/validation",
    fast_mode: bool = False,
    max_workers: int = 1,
    fast_baseline: bool = False,
    **backtest_kwargs,
) -> dict:
    """
    Run all validation phases.

    Parameters
    ----------
    start_date : Backtest start date
    end_date : Backtest end date
    output_dir : Directory for reports and plots
    fast_mode : If True, reduce simulation counts for quick testing
    max_workers : Number of parallel workers for heavy phases (1 = sequential)
    **backtest_kwargs : Passed to run_backtest()

    Returns
    -------
    Dict with go_nogo, reason, and per-test details
    """
    from backend.backtesting.simple_backtest import run_backtest, load_all_prices
    from backend.quant_pro.database import get_db_path

    t_start = time.time()
    tests: Dict[str, dict] = {}

    # Default backtest params
    if not backtest_kwargs:
        backtest_kwargs = {
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

    # Simulation counts
    n_random_sims = 100 if fast_mode else 1000
    n_mc_sims = 1000 if fast_mode else 10_000
    n_bootstrap = 1000 if fast_mode else 10_000

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Base Backtest (REQUIRED — others depend on this)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1: Running base backtest...")
    logger.info("=" * 60)

    result = run_backtest(start_date=start_date, end_date=end_date, **backtest_kwargs)

    tests["base_backtest"] = {
        "status": "PASS" if result.sharpe_ratio >= GO_CRITERIA["min_sharpe"] else "FAIL",
        "summary": (
            f"Sharpe={result.sharpe_ratio:.3f}, "
            f"Return={result.total_return:.2%}, "
            f"MaxDD={result.max_drawdown:.2%}, "
            f"Trades={result.total_trades}"
        ),
        "details": {
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "calmar": result.calmar_ratio,
        },
    }
    logger.info(f"  Base backtest: {tests['base_backtest']['summary']}")

    daily_returns = np.array(result.daily_returns)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Transaction Cost Verification
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2: Verifying transaction cost model...")
    logger.info("=" * 60)

    try:
        rt = TransactionCostModel.round_trip_cost(
            shares=100, entry_price=1000.0, exit_price=1100.0, holding_days=40
        )
        cost_model_ok = (
            rt.buy.commission > 0
            and rt.sell.commission > 0
            and rt.cgt > 0
            and rt.total_cost > 0
            and 0.005 <= rt.cost_pct <= 0.05  # Round-trip between 0.5% and 5%
            and rt.buy.nepse_fee > 0          # NEPSE fee included
            and rt.sell.dp_name_transfer > 0  # DP name transfer on sell
        )
        tests["transaction_costs"] = {
            "status": "PASS" if cost_model_ok else "FAIL",
            "summary": (
                f"Round-trip cost={rt.cost_pct:.4%}, "
                f"CGT={rt.cgt:,.0f}, Total={rt.total_cost:,.0f}"
            ),
            "details": {
                "cost_pct": rt.cost_pct,
                "buy_fees": rt.buy.total,
                "sell_fees": rt.sell.total,
                "cgt": rt.cgt,
                "total_cost": rt.total_cost,
            },
        }
        logger.info(f"  Transaction costs: {tests['transaction_costs']['summary']}")
    except Exception as e:
        _phase_error(tests, "transaction_costs", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Statistical Significance
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 3: Statistical significance tests...")
    logger.info("=" * 60)

    try:
        # n_trials = actual number of strategy variants tested in parameter sweep
        # 3 holding periods x 11 signal combos x 2 regime settings = 66
        n_strategy_trials = 66
        stat_report = full_statistical_report(
            daily_returns=daily_returns,
            sharpe_ratio=result.sharpe_ratio,
            n_trials=n_strategy_trials,
        )

        psr_pass = stat_report.psr >= GO_CRITERIA["min_psr"]
        dsr_pass = stat_report.dsr >= GO_CRITERIA["min_dsr"]
        tests["statistical_significance"] = {
            "status": "PASS" if (psr_pass and dsr_pass and stat_report.ttest_pass) else "FAIL",
            "summary": (
                f"PSR={stat_report.psr:.3f} ({'PASS' if psr_pass else 'FAIL'}), "
                f"DSR={stat_report.dsr:.3f} ({'PASS' if dsr_pass else 'FAIL'}), "
                f"t-test p={stat_report.ttest['p_value']:.4f}, "
                f"MinTRL={stat_report.min_trl_years:.1f}yr"
            ),
            "details": {
                "psr": stat_report.psr,
                "dsr": stat_report.dsr,
                "min_trl_days": stat_report.min_trl,
                "min_trl_years": stat_report.min_trl_years,
                "ttest_t_stat": stat_report.ttest["t_stat"],
                "ttest_p_value": stat_report.ttest["p_value"],
                "ttest_significant": stat_report.ttest_pass,
                "skew": stat_report.skew,
                "kurtosis": stat_report.kurtosis,
                "n_obs": stat_report.n_obs,
                "n_trials": stat_report.n_trials,
                "sufficient_track_record": stat_report.sufficient_track_record,
            },
        }
        logger.info(f"  Statistical: {tests['statistical_significance']['summary']}")
    except Exception as e:
        _phase_error(tests, "statistical_significance", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward Cross-Validation
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 4: Walk-forward cross-validation...")
    logger.info("=" * 60)

    try:
        wf_kwargs = {k: v for k, v in backtest_kwargs.items()}
        wf_result = walk_forward_validation(
            train_window_days=504,
            test_window_days=126,
            step_days=42 if fast_mode else 21,
            min_sharpe=GO_CRITERIA["wf_mean_sharpe"],
            max_workers=max_workers,
            **wf_kwargs,
        )

        # Pass if EITHER mean fold Sharpe OR stitched OOS Sharpe meets threshold
        wf_passes = (
            wf_result.passes
            or wf_result.oos_sharpe >= GO_CRITERIA["wf_mean_sharpe"]
        )
        tests["walk_forward"] = {
            "status": "PASS" if wf_passes else "FAIL",
            "summary": (
                f"Mean OOS Sharpe={wf_result.aggregate['mean_sharpe']:.3f}, "
                f"Stitched OOS Sharpe={wf_result.oos_sharpe:.3f}, "
                f"Std={wf_result.aggregate['std_sharpe']:.3f}, "
                f"Folds={wf_result.aggregate['n_folds']}, "
                f"Positive={wf_result.aggregate['n_positive_sharpe']}/{wf_result.aggregate['n_folds']}"
            ),
            "details": {
                **wf_result.aggregate,
                "fold_metrics": [
                    {
                        "fold_id": f.fold_id,
                        "start_date": f.start_date,
                        "end_date": f.end_date,
                        "sharpe": f.sharpe,
                        "total_return": f.total_return,
                        "max_drawdown": f.max_drawdown,
                        "n_trades": f.n_trades,
                    }
                    for f in wf_result.fold_metrics
                ],
                "oos_sharpe": wf_result.oos_sharpe,
                "oos_total_return": wf_result.oos_total_return,
            },
        }
        logger.info(f"  Walk-forward: {tests['walk_forward']['summary']}")
    except Exception as e:
        _phase_error(tests, "walk_forward", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5: Benchmark Comparison
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 5: Benchmark comparison...")
    logger.info("=" * 60)

    prices_df = None
    try:
        conn = sqlite3.connect(str(get_db_path()))
        prices_df = load_all_prices(conn)
        conn.close()

        benchmark = compute_benchmark_series(prices_df, start_date, end_date)
        bench_result = benchmark_comparison(result.daily_nav, benchmark)

        tests["benchmark"] = {
            "status": "PASS" if bench_result["excess_return"] > 0 else "FAIL",
            "summary": (
                f"Alpha={bench_result['alpha']:.3f}, "
                f"Beta={bench_result['beta']:.2f}, "
                f"IR={bench_result['information_ratio']:.3f}, "
                f"Excess={bench_result['excess_return']:.2%}"
            ),
            "details": bench_result,
        }
        logger.info(f"  Benchmark: {tests['benchmark']['summary']}")
    except Exception as e:
        _phase_error(tests, "benchmark", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 6: Monte Carlo
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 6: Monte Carlo simulation...")
    logger.info("=" * 60)

    try:
        trade_returns = [
            t.net_return for t in result.completed_trades if t.net_return is not None
        ]
        mc_result = monte_carlo_trade_resample(
            trade_returns,
            initial_capital=backtest_kwargs.get("initial_capital", 1_000_000),
            n_simulations=n_mc_sims,
        )
        bs_result = block_bootstrap_ci(
            daily_returns, n_bootstrap=n_bootstrap, block_size=21
        )

        mc_pass = mc_result.sharpe_ci[0] > GO_CRITERIA["mc_sharpe_ci_lower"]
        tests["monte_carlo"] = {
            "status": "PASS" if mc_pass else "FAIL",
            "summary": (
                f"Sharpe CI=[{mc_result.sharpe_ci[0]:.3f}, {mc_result.sharpe_ci[1]:.3f}], "
                f"P(ruin)={mc_result.prob_ruin:.2%}, "
                f"P(loss)={mc_result.prob_loss:.2%}, "
                f"Bootstrap Sharpe CI=[{bs_result.sharpe_ci[0]:.3f}, {bs_result.sharpe_ci[1]:.3f}]"
            ),
            "details": {
                "terminal_wealth_pcts": mc_result.terminal_wealth_pcts,
                "max_dd_pcts": mc_result.max_dd_pcts,
                "sharpe_ci": mc_result.sharpe_ci,
                "prob_ruin": mc_result.prob_ruin,
                "prob_loss": mc_result.prob_loss,
                "n_simulations": mc_result.n_simulations,
                "n_trades": mc_result.n_trades,
                "bootstrap_sharpe_ci": bs_result.sharpe_ci,
                "bootstrap_cagr_ci": bs_result.cagr_ci,
                "all_terminal_wealth": mc_result.all_terminal_wealth,
                "all_max_dd": mc_result.all_max_dd,
            },
        }
        logger.info(f"  Monte Carlo: {tests['monte_carlo']['summary']}")
    except Exception as e:
        _phase_error(tests, "monte_carlo", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 7: Regime Stress Tests
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 7: Regime stress tests...")
    logger.info("=" * 60)

    try:
        regime_result = regime_stress_test(**backtest_kwargs)
        cb_analysis = circuit_breaker_analysis(result.completed_trades)
        settle_analysis = settlement_lag_analysis(result.daily_nav)

        tests["regime_stress"] = {
            "status": "PASS" if regime_result["all_pass"] else "FAIL",
            "summary": (
                f"Worst regime: {regime_result['worst_regime']} "
                f"(DD={regime_result['worst_dd']:.2%}), "
                f"CB affected: {cb_analysis['cb_affected_count']} trades, "
                f"Settlement loss: {settle_analysis['capital_efficiency_loss_pct']:.2f}%"
            ),
            "details": {
                "regimes": [
                    {
                        "name": r.name,
                        "sharpe": r.sharpe,
                        "return": r.total_return,
                        "max_dd": r.max_drawdown,
                        "trades": r.n_trades,
                    }
                    for r in regime_result["regime_results"]
                ],
                "circuit_breaker": cb_analysis,
                "settlement_lag": settle_analysis,
            },
        }
        logger.info(f"  Regime stress: {tests['regime_stress']['summary']}")
    except Exception as e:
        _phase_error(tests, "regime_stress", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 8: Sensitivity Analysis
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 8: Parameter sensitivity analysis...")
    logger.info("=" * 60)

    try:
        if fast_mode:
            param_ranges = {
                "holding_days": [20, 30, 40, 50, 60],
                "trailing_stop_pct": [0.05, 0.08, 0.10, 0.15],
                "stop_loss_pct": [0.05, 0.08, 0.10, 0.15],
            }
        else:
            param_ranges = None

        sens_results = parameter_sensitivity(
            start_date=start_date,
            end_date=end_date,
            param_ranges=param_ranges,
            max_workers=max_workers,
        )
        sens_summary_dict = sensitivity_summary(sens_results)

        all_smooth = sens_summary_dict["all_smooth"]
        tests["sensitivity"] = {
            "status": "PASS" if all_smooth else "FAIL",
            "summary": (
                f"{'All smooth' if all_smooth else 'Cliffs detected in: ' + ', '.join(sens_summary_dict['flagged_params'])}"
            ),
            "details": {
                **sens_summary_dict,
                "param_summaries": {
                    name: {
                        **info,
                        "values": sens_results[name].values if name in sens_results else [],
                        "sharpes": sens_results[name].sharpes if name in sens_results else [],
                    }
                    for name, info in sens_summary_dict["param_summaries"].items()
                },
            },
        }
        logger.info(f"  Sensitivity: {tests['sensitivity']['summary']}")
    except Exception as e:
        _phase_error(tests, "sensitivity", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 9: Random Entry Baseline
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"PHASE 9: Random entry baseline ({n_random_sims} simulations)...")
    logger.info("=" * 60)

    try:
        if fast_baseline:
            n_random_sims_actual = 10000 if not fast_mode else 500
            rb_result = random_entry_baseline_fast(
                n_simulations=n_random_sims_actual,
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers if max_workers > 1 else 0,
                compute_alpha=True,
                **backtest_kwargs,
            )
        else:
            rb_result = random_entry_baseline(
                n_simulations=n_random_sims,
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers,
                **backtest_kwargs,
            )

        alpha_sharpe = rb_result.get("alpha_sharpe")
        alpha_str = f", Alpha Sharpe={alpha_sharpe:.3f}" if alpha_sharpe is not None else ""
        tests["random_baseline"] = {
            "status": "PASS" if rb_result["passes"] else "FAIL",
            "summary": (
                f"Actual Sharpe={rb_result['actual_sharpe']:.3f}, "
                f"Percentile={rb_result['percentile_rank']:.1f}%, "
                f"p-value={rb_result['p_value']:.4f}, "
                f"Random mean={rb_result['mean_random']:.3f}"
                f"{alpha_str}"
            ),
            "details": {
                "actual_sharpe": rb_result["actual_sharpe"],
                "percentile_rank": rb_result["percentile_rank"],
                "p_value": rb_result["p_value"],
                "mean_random": rb_result["mean_random"],
                "std_random": rb_result["std_random"],
                "n_simulations": rb_result["n_simulations"],
                "random_sharpes": rb_result["random_sharpes"],
                "alpha_sharpe": alpha_sharpe,
            },
        }
        logger.info(f"  Random baseline: {tests['random_baseline']['summary']}")
    except Exception as e:
        _phase_error(tests, "random_baseline", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 10: Slippage Impact
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 10: Slippage impact analysis...")
    logger.info("=" * 60)

    try:
        if prices_df is None:
            conn = sqlite3.connect(str(get_db_path()))
            prices_df = load_all_prices(conn)
            conn.close()

        slip_result = run_backtest_with_slippage(
            trades=result.completed_trades,
            prices_df=prices_df,
            initial_capital=backtest_kwargs.get("initial_capital", 1_000_000),
            daily_nav=result.daily_nav,
        )

        slip_adjusted_sharpe = slip_result.get(
            "adjusted_sharpe",
            result.sharpe_ratio + slip_result["sharpe_impact"],
        )
        slip_pass = (
            slip_adjusted_sharpe >= GO_CRITERIA["slip_min_adj_sharpe"]
            and slip_result["sharpe_impact"] > GO_CRITERIA["slip_max_sharpe_impact"]
        )
        tests["slippage"] = {
            "status": "PASS" if slip_pass else "FAIL",
            "summary": (
                f"Sharpe impact={slip_result['sharpe_impact']:.3f}, "
                f"Return impact={slip_result['return_impact']:.2%}, "
                f"Total slippage={slip_result['total_slippage_cost']:,.0f}, "
                f"Adj Sharpe={slip_adjusted_sharpe:.3f}"
            ),
            "details": slip_result,
        }
        logger.info(f"  Slippage: {tests['slippage']['summary']}")
    except Exception as e:
        _phase_error(tests, "slippage", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 11: CSCV / Probability of Backtest Overfitting
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 11: CSCV / Probability of Backtest Overfitting...")
    logger.info("=" * 60)

    try:
        from validation.cscv_pbo import cscv_pbo_analysis, build_performance_matrix, pbo_summary

        if prices_df is None:
            conn = sqlite3.connect(str(get_db_path()))
            prices_df = load_all_prices(conn)
            conn.close()

        # Build performance matrix from parameter sweep configs
        from backend.backtesting.simple_backtest import run_backtest as _run_bt
        configs = {}
        _hold_periods = [20, 40, 60]
        _sig_combos = [
            ["volume"], ["quality"], ["low_vol"],
            ["volume", "quality"], ["quality", "low_vol"],
            ["volume", "quality", "low_vol"],
            ["volume", "quality", "low_vol", "xsec_momentum"],
        ]
        for hold in _hold_periods:
            for sigs in _sig_combos:
                for regime in [True, False]:
                    name = f"{'+'.join(sigs)}_h{hold}_r{int(regime)}"
                    configs[name] = {
                        "holding_days": hold,
                        "signal_types": sigs,
                        "use_regime_filter": regime,
                        "max_positions": 5,
                        "use_trailing_stop": True,
                        "trailing_stop_pct": 0.10,
                        "stop_loss_pct": 0.08,
                        "sector_limit": 0.35,
                    }

        perf_matrix = build_performance_matrix(
            prices_df=prices_df,
            start_date=start_date,
            end_date=end_date,
            configs=configs,
            run_backtest_func=_run_bt,
            partition_days=63,
        )

        pbo_result = cscv_pbo_analysis(perf_matrix, max_combinations=1000)
        pbo_pass = pbo_result.passes

        tests["cscv_pbo"] = {
            "status": "PASS" if pbo_pass else "FAIL",
            "summary": (
                f"PBO={pbo_result.pbo:.3f} "
                f"({'PASS' if pbo_pass else 'FAIL'}, threshold=0.40), "
                f"n_configs={pbo_result.n_configs}, "
                f"n_combos={pbo_result.n_combinations}"
            ),
            "details": pbo_summary(pbo_result),
        }
        logger.info(f"  CSCV/PBO: {tests['cscv_pbo']['summary']}")
    except Exception as e:
        _phase_error(tests, "cscv_pbo", e)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 12: Max Drawdown Check
    # ═══════════════════════════════════════════════════════════════════
    dd_pass = result.max_drawdown > GO_CRITERIA["max_drawdown"]
    tests["max_drawdown"] = {
        "status": "PASS" if dd_pass else "FAIL",
        "summary": (
            f"Max DD={result.max_drawdown:.2%} "
            f"(limit: {GO_CRITERIA['max_drawdown']:.2%})"
        ),
        "details": {"max_drawdown": result.max_drawdown},
    }

    # ═══════════════════════════════════════════════════════════════════
    # Go/No-Go Decision
    # ═══════════════════════════════════════════════════════════════════
    critical_tests = [
        "base_backtest",
        "transaction_costs",
        "statistical_significance",
        "walk_forward",
        "monte_carlo",
        "regime_stress",
        "sensitivity",
        "random_baseline",
        "slippage",
        "max_drawdown",
    ]
    failures = [
        name for name in critical_tests
        if tests.get(name, {}).get("status") in ("FAIL", "ERROR")
    ]

    go_nogo = "GO" if len(failures) == 0 else "NO-GO"
    reason = (
        "All critical tests passed."
        if go_nogo == "GO"
        else f"Failed tests: {', '.join(failures)}"
    )

    elapsed = time.time() - t_start
    full_results = {
        "go_nogo": go_nogo,
        "reason": reason,
        "tests": tests,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "backtest_params": backtest_kwargs,
            "go_criteria": GO_CRITERIA,
            "fast_mode": fast_mode,
        },
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        # Raw data for report plots (not serialized to JSON)
        "_daily_nav": result.daily_nav,
        "_trades": result.completed_trades,
    }

    # ═══════════════════════════════════════════════════════════════════
    # Generate Reports (always runs, even if phases failed)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Generating reports...")
    logger.info("=" * 60)

    try:
        json_path = save_json_report(full_results, output_dir)
        logger.info(f"  JSON: {json_path}")
    except Exception as e:
        logger.error(f"  JSON report failed: {e}")
        json_path = None

    print_console_report(full_results)

    try:
        plot_paths = generate_plots(full_results, output_dir)
        logger.info(f"  Plots: {len(plot_paths)} files")
    except Exception as e:
        logger.error(f"  Plot generation failed: {e}")
        plot_paths = []

    try:
        pdf_path = generate_pdf_report(full_results, output_dir)
        if pdf_path:
            logger.info(f"  PDF:  {pdf_path}")
    except Exception as e:
        logger.error(f"  PDF report failed: {e}")
        pdf_path = None

    logger.info(f"Validation complete in {elapsed:.0f}s")
    logger.info(f"  Verdict: {go_nogo}")

    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Trading System — Production Readiness Validation"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="reports/validation", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer simulations)")
    parser.add_argument("--holding-days", type=int, default=40, help="Holding period")
    parser.add_argument("--max-positions", type=int, default=5, help="Max positions")
    parser.add_argument(
        "--signals", nargs="+",
        default=["volume", "quality", "low_vol"],
        help="Signal types"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for heavy phases (default: 1)"
    )
    parser.add_argument(
        "--fast-baseline", action="store_true",
        help="Use fast numpy-based random baseline (enables 10K sims with multiprocessing)"
    )
    parser.add_argument(
        "--regime-positions", type=str, default=None,
        help='Graduated regime position limits, e.g. "bull:5,neutral:4,bear:1"'
    )
    parser.add_argument(
        "--bear-threshold", type=float, default=-0.05,
        help="Bear regime threshold (default: -0.05, tighter: -0.03)"
    )
    args = parser.parse_args()

    # Parse regime position limits
    regime_max_positions = None
    if args.regime_positions:
        regime_max_positions = {}
        for pair in args.regime_positions.split(","):
            regime, count = pair.strip().split(":")
            regime_max_positions[regime.strip()] = int(count.strip())

    results = run_full_validation(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        fast_mode=args.fast,
        max_workers=args.workers,
        fast_baseline=args.fast_baseline,
        holding_days=args.holding_days,
        max_positions=args.max_positions,
        signal_types=args.signals,
        initial_capital=1_000_000,
        rebalance_frequency=5,
        use_trailing_stop=True,
        trailing_stop_pct=0.10,
        stop_loss_pct=0.08,
        use_regime_filter=True,
        sector_limit=0.35,
        regime_max_positions=regime_max_positions,
        bear_threshold=args.bear_threshold,
    )

    sys.exit(0 if results["go_nogo"] == "GO" else 1)


if __name__ == "__main__":
    main()
