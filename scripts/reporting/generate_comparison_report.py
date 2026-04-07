#!/usr/bin/env python3
"""
Generate Baseline vs Upgraded Comparison Report.

Reads validation results from both runs and produces:
  - Side-by-side metrics comparison (console + JSON)
  - Comparison charts (PNG)
  - Summary PDF

Usage:
    python3 generate_comparison_report.py

Expects:
    reports/validation_baseline/validation_results.json
    reports/validation_upgraded/validation_results.json
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from backend.quant_pro.paths import get_project_root

PROJECT_ROOT = str(get_project_root(__file__))
BASELINE_DIR = os.path.join(PROJECT_ROOT, "reports", "validation_baseline")
UPGRADED_DIR = os.path.join(PROJECT_ROOT, "reports", "validation_upgraded")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "comparison")

# Dark theme
DARK_BG = "#0D1117"
CARD_BG = "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"
ACCENT = "#58A6FF"
GREEN = "#3FB950"
RED = "#F85149"
GOLD = "#D29922"
PURPLE = "#BC8CFF"
CYAN = "#39D2C0"
ORANGE = "#F0883E"
BASELINE_COLOR = "#4FC3F7"
UPGRADED_COLOR = "#66BB6A"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.fontsize": 9,
})


def load_results(path):
    """Load validation_results.json from a directory."""
    json_path = os.path.join(path, "validation_results.json")
    if not os.path.exists(json_path):
        print(f"  WARNING: {json_path} not found")
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_key_metrics(results):
    """Pull out the most important metrics for comparison."""
    if results is None:
        return {}

    base = results.get("tests", {}).get("base_backtest", {}).get("details", {})
    stats_sig = results.get("tests", {}).get("statistical_significance", {}).get("details", {})
    wf = results.get("tests", {}).get("walk_forward", {}).get("details", {})
    mc = results.get("tests", {}).get("monte_carlo", {}).get("details", {})
    rb = results.get("tests", {}).get("random_baseline", {}).get("details", {})
    slip = results.get("tests", {}).get("slippage", {}).get("details", {})
    cscv = results.get("tests", {}).get("cscv_pbo", {}).get("details", {})

    return {
        "go_nogo": results.get("go_nogo", "N/A"),
        "sharpe": base.get("sharpe", 0),
        "sortino": base.get("sortino", 0),
        "total_return": base.get("total_return", 0),
        "annualized_return": base.get("annualized_return", 0),
        "max_drawdown": base.get("max_drawdown", 0),
        "total_trades": base.get("total_trades", 0),
        "win_rate": base.get("win_rate", 0),
        "profit_factor": base.get("profit_factor", 0),
        "calmar": base.get("calmar", 0),
        "psr": stats_sig.get("psr", 0),
        "dsr": stats_sig.get("dsr", 0),
        "ttest_p": stats_sig.get("ttest_p_value", 1),
        "wf_mean_sharpe": wf.get("mean_sharpe", 0),
        "wf_stitched_sharpe": wf.get("stitched_oos_sharpe", 0),
        "wf_positive_folds": wf.get("n_positive_sharpe", 0),
        "wf_total_folds": wf.get("n_folds", 0),
        "mc_sharpe_ci_lower": mc.get("sharpe_ci_lower", 0) if mc else 0,
        "mc_sharpe_ci_upper": mc.get("sharpe_ci_upper", 0) if mc else 0,
        "random_pctile": rb.get("percentile_rank", 0) if rb else 0,
        "random_beats_pct": rb.get("beats_random_pct", 0) if rb else 0,
        "slip_adj_sharpe": slip.get("adjusted_sharpe", 0) if slip else 0,
        "pbo": cscv.get("pbo_probability", 0) if cscv else 0,
    }


def print_comparison_table(base_m, up_m):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 78)
    print("  BASELINE vs UPGRADED — PERFORMANCE COMPARISON")
    print("=" * 78)

    rows = [
        ("Verdict", "go_nogo", "", False),
        ("Sharpe Ratio", "sharpe", ".3f", True),
        ("Sortino Ratio", "sortino", ".3f", True),
        ("Total Return", "total_return", ".2%", True),
        ("Annualized Return", "annualized_return", ".2%", True),
        ("Max Drawdown", "max_drawdown", ".2%", False),  # Less negative = better
        ("Total Trades", "total_trades", "d", True),
        ("Win Rate", "win_rate", ".1%", True),
        ("Profit Factor", "profit_factor", ".2f", True),
        ("Calmar Ratio", "calmar", ".2f", True),
        ("PSR", "psr", ".4f", True),
        ("DSR", "dsr", ".4f", True),
        ("t-test p-value", "ttest_p", ".4f", False),  # lower = better
        ("WF Mean OOS Sharpe", "wf_mean_sharpe", ".3f", True),
        ("WF Stitched OOS Sharpe", "wf_stitched_sharpe", ".3f", True),
        ("WF Positive Folds", "wf_positive_folds", "d", True),
        ("MC Sharpe CI Lower", "mc_sharpe_ci_lower", ".3f", True),
        ("Random Baseline %ile", "random_pctile", ".1f", True),
        ("Slippage-Adj Sharpe", "slip_adj_sharpe", ".3f", True),
        ("PBO (lower=better)", "pbo", ".3f", False),
    ]

    print(f"  {'Metric':<28} {'Baseline':>14} {'Upgraded':>14} {'Delta':>10} {'Better?':>8}")
    print("  " + "-" * 74)

    for label, key, fmt, higher_better in rows:
        bv = base_m.get(key, "N/A")
        uv = up_m.get(key, "N/A")

        if fmt == "" or isinstance(bv, str) or isinstance(uv, str):
            print(f"  {label:<28} {str(bv):>14} {str(uv):>14} {'':>10} {'':>8}")
            continue

        b_str = f"{bv:{fmt}}"
        u_str = f"{uv:{fmt}}"

        if fmt.endswith("%"):
            delta = uv - bv
            d_str = f"{delta:+{fmt}}"
        elif fmt.endswith("f"):
            delta = uv - bv
            d_str = f"{delta:+.3f}"
        else:
            delta = uv - bv
            d_str = f"{delta:+.0f}"

        if higher_better:
            better = "YES" if uv > bv else ("SAME" if uv == bv else "no")
        else:
            better = "YES" if uv < bv else ("SAME" if uv == bv else "no")

        print(f"  {label:<28} {b_str:>14} {u_str:>14} {d_str:>10} {better:>8}")

    print("=" * 78)


def chart_metrics_comparison(base_m, up_m, output_dir):
    """Generate comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Key performance metrics side-by-side
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("BASELINE vs UPGRADED: KEY METRICS COMPARISON",
                 fontsize=18, fontweight="bold", color=ACCENT, y=0.98)

    metrics_to_plot = [
        ("Sharpe Ratio", "sharpe", False),
        ("Sortino Ratio", "sortino", False),
        ("Total Return (%)", "total_return", True),
        ("Max Drawdown (%)", "max_drawdown", True),
        ("Win Rate (%)", "win_rate", True),
        ("Profit Factor", "profit_factor", False),
    ]

    for idx, (label, key, as_pct) in enumerate(metrics_to_plot):
        ax = axes[idx // 3][idx % 3]
        bv = base_m.get(key, 0)
        uv = up_m.get(key, 0)

        if as_pct:
            bv *= 100
            uv *= 100

        x = [0, 1]
        bars = ax.bar(x, [bv, uv], color=[BASELINE_COLOR, UPGRADED_COLOR],
                       width=0.6, edgecolor=GRID_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Upgraded"])
        ax.set_title(label, fontsize=12, fontweight="bold")

        for bar, val in zip(bars, [bv, uv]):
            va = "bottom" if val >= 0 else "top"
            offset = 0.02 * max(abs(bv), abs(uv)) if max(abs(bv), abs(uv)) > 0 else 0.01
            y_pos = val + offset if val >= 0 else val - offset
            fmt_str = f"{val:.2f}" if not as_pct else f"{val:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, fmt_str,
                    ha="center", va=va, fontsize=11, fontweight="bold", color=TEXT_COLOR)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "01_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_metrics_comparison.png")

    # Chart 2: Walk-forward and anti-cheat comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("WALK-FORWARD & ANTI-CHEAT VALIDATION COMPARISON",
                 fontsize=16, fontweight="bold", color=ACCENT, y=1.02)

    # WF Sharpe
    ax = axes[0]
    bv = base_m.get("wf_mean_sharpe", 0)
    uv = up_m.get("wf_mean_sharpe", 0)
    bars = ax.bar([0, 1], [bv, uv], color=[BASELINE_COLOR, UPGRADED_COLOR], width=0.6)
    ax.axhline(y=0.5, color=GOLD, linestyle="--", alpha=0.6, label="GO threshold (0.5)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Upgraded"])
    ax.set_title("Walk-Forward Mean OOS Sharpe", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, [bv, uv]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # PSR
    ax = axes[1]
    bv = base_m.get("psr", 0)
    uv = up_m.get("psr", 0)
    bars = ax.bar([0, 1], [bv, uv], color=[BASELINE_COLOR, UPGRADED_COLOR], width=0.6)
    ax.axhline(y=0.90, color=GOLD, linestyle="--", alpha=0.6, label="GO threshold (0.90)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Upgraded"])
    ax.set_title("Probabilistic Sharpe Ratio (PSR)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, [bv, uv]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Random baseline percentile
    ax = axes[2]
    bv = base_m.get("random_pctile", 0)
    uv = up_m.get("random_pctile", 0)
    bars = ax.bar([0, 1], [bv, uv], color=[BASELINE_COLOR, UPGRADED_COLOR], width=0.6)
    ax.axhline(y=95, color=GOLD, linestyle="--", alpha=0.6, label="GO threshold (95th)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Upgraded"])
    ax.set_title("Random Baseline Percentile", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, [bv, uv]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}th",
                ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_anticheat_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_anticheat_comparison.png")

    # Chart 3: Go/No-Go dashboard
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("PRODUCTION READINESS — GO / NO-GO DASHBOARD",
                 fontsize=18, fontweight="bold", color=ACCENT, y=0.98)

    criteria = [
        ("Sharpe > 1.0", "sharpe", 1.0, True),
        ("PSR > 0.90", "psr", 0.90, True),
        ("DSR > 0.85", "dsr", 0.85, True),
        ("Max DD > -25%", "max_drawdown", -0.25, False),
        ("WF Sharpe > 0.5", "wf_mean_sharpe", 0.5, True),
        ("Random > 95th", "random_pctile", 95, True),
        ("Slip Sharpe > 0.8", "slip_adj_sharpe", 0.8, True),
    ]

    y_positions = list(range(len(criteria)))

    for i, (label, key, threshold, higher) in enumerate(criteria):
        bv = base_m.get(key, 0)
        uv = up_m.get(key, 0)

        if higher:
            b_pass = bv >= threshold
            u_pass = uv >= threshold
        else:
            b_pass = bv >= threshold  # For max_drawdown, -20% > -25%
            u_pass = uv >= threshold

        # Baseline indicator
        ax.scatter(0.3, i, s=300, color=GREEN if b_pass else RED,
                  marker="o", zorder=5, edgecolors=GRID_COLOR)
        ax.text(0.3, i - 0.3, f"{'PASS' if b_pass else 'FAIL'}",
                ha="center", fontsize=7, color=GREEN if b_pass else RED)

        # Upgraded indicator
        ax.scatter(0.7, i, s=300, color=GREEN if u_pass else RED,
                  marker="o", zorder=5, edgecolors=GRID_COLOR)
        ax.text(0.7, i - 0.3, f"{'PASS' if u_pass else 'FAIL'}",
                ha="center", fontsize=7, color=GREEN if u_pass else RED)

        # Label
        ax.text(-0.05, i, label, ha="right", va="center", fontsize=11, color=TEXT_COLOR)

    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.8, len(criteria) - 0.2)
    ax.set_xticks([0.3, 0.7])
    ax.set_xticklabels(["Baseline\n(Vol+Qual+LowVol)", "Upgraded\n(+52WK+IMOM)"],
                       fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.set_title("")

    # Verdicts
    b_verdict = base_m.get("go_nogo", "N/A")
    u_verdict = up_m.get("go_nogo", "N/A")
    b_color = GREEN if b_verdict == "GO" else RED
    u_color = GREEN if u_verdict == "GO" else RED

    ax.text(0.3, len(criteria) - 0.5, f"Verdict: {b_verdict}",
            ha="center", fontsize=14, fontweight="bold", color=b_color)
    ax.text(0.7, len(criteria) - 0.5, f"Verdict: {u_verdict}",
            ha="center", fontsize=14, fontweight="bold", color=u_color)

    fig.tight_layout(rect=[0.15, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "03_go_nogo_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_go_nogo_dashboard.png")


def generate_pdf(base_m, up_m, base_results, up_results, output_dir):
    """Generate comparison PDF report."""
    pdf_path = os.path.join(output_dir, "NEPSE_Quant_Comparison_Report.pdf")

    with PdfPages(pdf_path) as pdf:
        # Page 1: Cover
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.72, "NEPSE QUANTITATIVE\nTRADING SYSTEM", fontsize=36,
                 fontweight="bold", ha="center", va="center", color=TEXT_COLOR,
                 linespacing=1.4)
        fig.text(0.5, 0.55, "Baseline vs Upgraded — Validation Comparison Report",
                 fontsize=16, ha="center", color=ACCENT, style="italic")

        line_ax = fig.add_axes([0.2, 0.48, 0.6, 0.002])
        line_ax.set_facecolor(ACCENT)
        line_ax.set_xticks([])
        line_ax.set_yticks([])

        fig.text(0.5, 0.40,
                 "Baseline: Volume + Quality + Low Volatility\n"
                 "Upgraded: + 52-Week High Anchoring + Residual Momentum (IMOM)",
                 fontsize=12, ha="center", color="#8B949E", linespacing=1.6)

        b_sharpe = base_m.get("sharpe", 0)
        u_sharpe = up_m.get("sharpe", 0)
        b_ret = base_m.get("total_return", 0)
        u_ret = up_m.get("total_return", 0)

        metrics_text = (
            f"Baseline Sharpe: {b_sharpe:.3f}   |   Upgraded Sharpe: {u_sharpe:.3f}\n"
            f"Baseline Return: {b_ret:.1%}   |   Upgraded Return: {u_ret:.1%}"
        )
        fig.text(0.5, 0.28, metrics_text, fontsize=11, ha="center",
                 color=GOLD, family="monospace", linespacing=1.8)

        fig.text(0.5, 0.12, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=10, ha="center", color="#8B949E")

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Metrics Comparison Table
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "PERFORMANCE METRICS COMPARISON", fontsize=20,
                 fontweight="bold", ha="center", color=ACCENT)

        rows = [
            ("Sharpe Ratio", f"{base_m.get('sharpe', 0):.3f}", f"{up_m.get('sharpe', 0):.3f}"),
            ("Sortino Ratio", f"{base_m.get('sortino', 0):.3f}", f"{up_m.get('sortino', 0):.3f}"),
            ("Total Return", f"{base_m.get('total_return', 0):.1%}", f"{up_m.get('total_return', 0):.1%}"),
            ("CAGR", f"{base_m.get('annualized_return', 0):.1%}", f"{up_m.get('annualized_return', 0):.1%}"),
            ("Max Drawdown", f"{base_m.get('max_drawdown', 0):.1%}", f"{up_m.get('max_drawdown', 0):.1%}"),
            ("Win Rate", f"{base_m.get('win_rate', 0):.1%}", f"{up_m.get('win_rate', 0):.1%}"),
            ("Profit Factor", f"{base_m.get('profit_factor', 0):.2f}", f"{up_m.get('profit_factor', 0):.2f}"),
            ("Calmar Ratio", f"{base_m.get('calmar', 0):.2f}", f"{up_m.get('calmar', 0):.2f}"),
            ("Total Trades", f"{base_m.get('total_trades', 0)}", f"{up_m.get('total_trades', 0)}"),
            ("", "", ""),
            ("PSR", f"{base_m.get('psr', 0):.4f}", f"{up_m.get('psr', 0):.4f}"),
            ("DSR", f"{base_m.get('dsr', 0):.4f}", f"{up_m.get('dsr', 0):.4f}"),
            ("t-test p-value", f"{base_m.get('ttest_p', 1):.4f}", f"{up_m.get('ttest_p', 1):.4f}"),
            ("", "", ""),
            ("WF Mean OOS Sharpe", f"{base_m.get('wf_mean_sharpe', 0):.3f}", f"{up_m.get('wf_mean_sharpe', 0):.3f}"),
            ("WF Stitched OOS Sharpe", f"{base_m.get('wf_stitched_sharpe', 0):.3f}", f"{up_m.get('wf_stitched_sharpe', 0):.3f}"),
            ("WF Positive Folds", f"{base_m.get('wf_positive_folds', 0)}/{base_m.get('wf_total_folds', 0)}", f"{up_m.get('wf_positive_folds', 0)}/{up_m.get('wf_total_folds', 0)}"),
            ("", "", ""),
            ("Random Baseline %ile", f"{base_m.get('random_pctile', 0):.1f}", f"{up_m.get('random_pctile', 0):.1f}"),
            ("Slippage-Adj Sharpe", f"{base_m.get('slip_adj_sharpe', 0):.3f}", f"{up_m.get('slip_adj_sharpe', 0):.3f}"),
            ("PBO Probability", f"{base_m.get('pbo', 0):.3f}", f"{up_m.get('pbo', 0):.3f}"),
        ]

        # Headers
        fig.text(0.20, 0.88, "Metric", fontsize=12, fontweight="bold", color=GOLD)
        fig.text(0.52, 0.88, "Baseline", fontsize=12, fontweight="bold", color=BASELINE_COLOR)
        fig.text(0.72, 0.88, "Upgraded", fontsize=12, fontweight="bold", color=UPGRADED_COLOR)

        y = 0.85
        for label, bv, uv in rows:
            if label == "":
                y -= 0.015
                continue
            fig.text(0.10, y, label, fontsize=10, color=TEXT_COLOR)
            fig.text(0.55, y, bv, fontsize=10, color=BASELINE_COLOR, ha="center", family="monospace")
            fig.text(0.75, y, uv, fontsize=10, color=UPGRADED_COLOR, ha="center", family="monospace")
            y -= 0.032

        # Verdicts
        y -= 0.03
        fig.text(0.10, y, "VERDICT", fontsize=14, fontweight="bold", color=GOLD)
        bv = base_m.get("go_nogo", "N/A")
        uv = up_m.get("go_nogo", "N/A")
        fig.text(0.55, y, bv, fontsize=14, fontweight="bold",
                 color=GREEN if bv == "GO" else RED, ha="center")
        fig.text(0.75, y, uv, fontsize=14, fontweight="bold",
                 color=GREEN if uv == "GO" else RED, ha="center")

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Test-by-test results
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "TEST-BY-TEST VALIDATION RESULTS", fontsize=20,
                 fontweight="bold", ha="center", color=ACCENT)

        test_names = [
            "base_backtest", "transaction_costs", "statistical_significance",
            "walk_forward", "monte_carlo", "random_baseline", "regime_stress",
            "sensitivity", "slippage", "cscv_pbo",
        ]

        y = 0.88
        fig.text(0.10, y, "Test", fontsize=11, fontweight="bold", color=GOLD)
        fig.text(0.45, y, "Baseline", fontsize=11, fontweight="bold", color=BASELINE_COLOR)
        fig.text(0.75, y, "Upgraded", fontsize=11, fontweight="bold", color=UPGRADED_COLOR)
        y -= 0.01

        for test_name in test_names:
            y -= 0.035
            b_test = base_results.get("tests", {}).get(test_name, {}) if base_results else {}
            u_test = up_results.get("tests", {}).get(test_name, {}) if up_results else {}

            b_status = b_test.get("status", "N/A")
            u_status = u_test.get("status", "N/A")
            b_summary = b_test.get("summary", "")[:60]
            u_summary = u_test.get("summary", "")[:60]

            # Test name
            display_name = test_name.replace("_", " ").title()
            fig.text(0.10, y, display_name, fontsize=10, color=TEXT_COLOR)

            # Baseline status + summary
            b_color = GREEN if b_status == "PASS" else (RED if b_status == "FAIL" else GOLD)
            fig.text(0.40, y, b_status, fontsize=10, fontweight="bold", color=b_color)
            fig.text(0.40, y - 0.018, b_summary, fontsize=7, color="#8B949E")

            # Upgraded status + summary
            u_color = GREEN if u_status == "PASS" else (RED if u_status == "FAIL" else GOLD)
            fig.text(0.70, y, u_status, fontsize=10, fontweight="bold", color=u_color)
            fig.text(0.70, y - 0.018, u_summary, fontsize=7, color="#8B949E")

            y -= 0.025

        # System info
        fig.text(0.5, 0.05,
                 "Baseline: volume + quality + low_vol  |  "
                 "Upgraded: + 52wk_high + residual_momentum\n"
                 f"Period: 2020-01-01 to 2026-01-31  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=8, ha="center", color="#484F58", linespacing=1.5)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: New models documentation
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.95, "UPGRADED SYSTEM — NEW MODELS DOCUMENTATION", fontsize=20,
                 fontweight="bold", ha="center", color=ACCENT)

        models_text = """TIER 1 — CORE SIGNALS (Price-Based, No External Data)

52-Week High Anchoring (George & Hwang, 2004)
  Retail traders anchor to 52-week highs prominently displayed on ShareSansar/MeroLagani.
  Signal: Proximity to 252-day rolling high > 80%. Closer proximity = stronger signal.
  Rationale: Stocks near highs tend to break through due to anchoring bias.

Residual Momentum / IMOM (Blitz et al., 2011)
  OLS regression strips market beta, ranks stocks by cumulative residual return.
  Signal: Top 20% by residual score over 126-day formation, 21-day skip period.
  Rationale: Idiosyncratic momentum avoids crash risk of raw momentum.

Cross-Sectional Momentum (6m-1m)
  Ranks stocks by 6-month return minus most recent 1-month return.
  Signal: Top quintile gets buy signals. Short-term reversal protection built in.

Accumulation (OBV ROC + Chaikin Money Flow)
  Detects institutional accumulation via On-Balance Volume trend + money flow analysis.
  Signal: Both OBV breakout AND positive CMF confirm smart money buying.

TIER 2 — REGIME DETECTION

3-State Hidden Markov Model (hmmlearn GaussianHMM)
  Models market as Bull/Neutral/Bear via Gaussian emission distributions.
  252-day lookback, retrained every 5 trading days, 10 random restarts.

Bayesian Online Changepoint Detection (Adams & MacKay, 2007)
  Real-time changepoint detection using Normal-Gamma conjugate prior.
  Hazard rate 1/200, triggers when cumulative P(r<5) > 0.5.

Calendar Effects (NRB Working Paper backed)
  Wednesday/Thursday boost (+3-5%), Sunday penalty (-3%), Dashain rally (+8%).
  21-day pre-Dashain and Tihar festival effects with empirical backing.

TIER 3 — PORTFOLIO CONSTRUCTION

Hierarchical Risk Parity (Lopez de Prado, 2016)
  Clustering-based allocation using skfolio. CVaR risk measure.
  Replaces naive equal-weight when sufficient position diversification exists.

CVaR Optimization (Rockafellar & Uryasev, 2000)
  Minimizes Conditional Value at Risk at 5th percentile.
  CLARABEL solver, max 30% per position, long-only constraint.

TIER 4 — BEHAVIORAL + MICROSTRUCTURE

Capital Gains Overhang (Grinblatt & Han, 2005)
  260-day VWAP reference price. CGO threshold 0.15 with volume confirmation.
  Disposition effect creates predictable momentum in NEPSE retail market.

Sector Lead-Lag (8 NEPSE sector pairs)
  Banking→Microfinance, HydroPower→Insurance, etc. Median stock return proxy.
  1.5 sigma threshold for significant lead signal.

OU Pairs Trading (28 cointegrated pairs)
  Ornstein-Uhlenbeck mean-reversion on spread Z-scores. Entry at Z>2.0.
  Long-only pairs (buy lagging stock). Half-life filter < 60 days.

TIER 5 — ALTERNATIVE DATA

Open-Meteo Rainfall → Hydropower (3 river basins)
  Koshi, Gandaki, Karnali basin rainfall anomaly maps to hydro ticker performance.

NRB Remittance Regime Classifier
  Strong/Normal/Weak remittance inflow → banking sector signal multiplier.

Post-Earnings Announcement Drift (PEAD)
  ShareSansar-scraped quarterly earnings. SUE = standardized unexpected earnings.
  Time-decaying signal with 40-day drift window.

TIER 6 — ADAPTIVE ML

First-Order MAML (FOMAML)
  Pre-trained on Indian market (50 Nifty stocks), adapted to NEPSE with few-shot.
  10-dim feature → 32→32→3 MLP. Regime classification without Hessian computation."""

        fig.text(0.05, 0.90, models_text, fontsize=7.2, color=TEXT_COLOR,
                 family="monospace", va="top", linespacing=1.25)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Saved: {pdf_path}")
    return pdf_path


def save_comparison_json(base_m, up_m, output_dir):
    """Save comparison as JSON."""
    comparison = {
        "generated": datetime.now().isoformat(),
        "baseline": base_m,
        "upgraded": up_m,
        "improvements": {},
    }

    for key in base_m:
        if isinstance(base_m[key], (int, float)) and isinstance(up_m.get(key), (int, float)):
            delta = up_m[key] - base_m[key]
            comparison["improvements"][key] = {
                "delta": round(delta, 6),
                "pct_change": round(delta / abs(base_m[key]) * 100, 2) if base_m[key] != 0 else None,
            }

    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Saved: {json_path}")


def main():
    print("=" * 60)
    print("  NEPSE QUANT — BASELINE vs UPGRADED COMPARISON REPORT")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load results
    print("\nLoading validation results...")
    base_results = load_results(BASELINE_DIR)
    up_results = load_results(UPGRADED_DIR)

    if base_results is None and up_results is None:
        print("ERROR: No validation results found. Run validation first:")
        print("  python3 -m validation.run_all --fast --signals volume quality low_vol --output reports/validation_baseline")
        print("  python3 -m validation.run_all --fast --signals volume quality low_vol 52wk_high residual_momentum --output reports/validation_upgraded")
        sys.exit(1)

    base_m = extract_key_metrics(base_results)
    up_m = extract_key_metrics(up_results)

    # Console comparison
    if base_m and up_m:
        print_comparison_table(base_m, up_m)
    elif base_m:
        print("\nOnly baseline results available:")
        for k, v in base_m.items():
            print(f"  {k}: {v}")
    elif up_m:
        print("\nOnly upgraded results available:")
        for k, v in up_m.items():
            print(f"  {k}: {v}")

    # Generate charts
    print("\nGenerating comparison charts...")
    if base_m and up_m:
        chart_metrics_comparison(base_m, up_m, OUTPUT_DIR)

    # Generate PDF
    print("\nGenerating comparison PDF...")
    if base_m and up_m:
        generate_pdf(base_m, up_m, base_results, up_results, OUTPUT_DIR)

    # Save JSON
    print("\nSaving comparison JSON...")
    save_comparison_json(base_m or {}, up_m or {}, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  COMPARISON REPORT COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
