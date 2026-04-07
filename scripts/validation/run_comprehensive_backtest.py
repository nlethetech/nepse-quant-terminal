#!/usr/bin/env python3
"""
Comprehensive Backtest & Charting Script
=========================================
Runs baseline vs upgraded signal configurations, generates
publication-quality dark-theme charts and a metrics JSON report.

Output:
    reports/
    +-- charts/
    |   +-- equity_curve_comparison.png
    |   +-- drawdown_chart.png
    |   +-- monthly_returns_heatmap.png
    |   +-- signal_contribution.png
    |   +-- rolling_sharpe.png
    |   +-- annual_returns.png
    |   +-- risk_return_scatter.png
    +-- backtest_metrics.json
    +-- equity_curves.csv
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from backend.quant_pro.paths import get_project_root

# Ensure project root is on the path
PROJECT_ROOT = str(get_project_root(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.backtesting.simple_backtest import run_backtest, BacktestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
START_DATE = "2020-01-01"
END_DATE = "2026-01-31"
INITIAL_CAPITAL = 1_000_000
HOLDING_DAYS = 40
CHARTS_DIR = os.path.join(PROJECT_ROOT, "reports", "charts")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Color palette for charts (dark theme)
COLORS = {
    "baseline": "#4FC3F7",   # Light blue
    "upgraded": "#66BB6A",   # Green
    "nepse": "#FF7043",      # Orange
    "grid": "#333333",
    "text": "#E0E0E0",
    "accent1": "#AB47BC",    # Purple
    "accent2": "#FFA726",    # Amber
    "accent3": "#EF5350",    # Red
    "accent4": "#26C6DA",    # Cyan
    "positive": "#66BB6A",
    "negative": "#EF5350",
}

# Signal combos to test
SIGNAL_CONFIGS = {
    "baseline": {
        "label": "Baseline (Vol+Quality+LowVol)",
        "signal_types": ["volume", "quality", "low_vol"],
    },
    "upgraded_full": {
        "label": "Upgraded (Vol+Qual+LowVol+52WK+IMOM+Disp)",
        "signal_types": ["volume", "quality", "low_vol", "52wk_high", "residual_momentum", "disposition"],
    },
    "upgraded_v2": {
        "label": "V2 (Vol+Qual+LowVol+52WK+IMOM)",
        "signal_types": ["volume", "quality", "low_vol", "52wk_high", "residual_momentum"],
    },
    "upgraded_v3": {
        "label": "V3 (Vol+Qual+52WK+IMOM+Disp+LeadLag)",
        "signal_types": ["volume", "quality", "52wk_high", "residual_momentum", "disposition", "lead_lag"],
    },
    "upgraded_v4": {
        "label": "V4 (Vol+Qual+LowVol+52WK+IMOM+Pairs)",
        "signal_types": ["volume", "quality", "low_vol", "52wk_high", "residual_momentum", "pairs_trade"],
    },
}

# ===========================================================================
# HELPERS
# ===========================================================================

def style_dark():
    """Apply consistent dark style to matplotlib."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#111418",
        "axes.facecolor": "#1C2127",
        "axes.edgecolor": "#404854",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "legend.facecolor": "#252A31",
        "legend.edgecolor": "#404854",
    })


def safe_run_backtest(name, config):
    """Run a backtest config, returning result or None on error."""
    logger.info(f"Running backtest: {name} ({config['label']})")
    t0 = time.time()
    try:
        result = run_backtest(
            start_date=START_DATE,
            end_date=END_DATE,
            holding_days=HOLDING_DAYS,
            max_positions=5,
            signal_types=config["signal_types"],
            initial_capital=INITIAL_CAPITAL,
            rebalance_frequency=5,
            use_trailing_stop=True,
            trailing_stop_pct=0.10,
            stop_loss_pct=0.08,
            use_regime_filter=True,
            sector_limit=0.35,
        )
        elapsed = time.time() - t0
        logger.info(f"  {name}: Sharpe={result.sharpe_ratio:.3f}, "
                     f"Return={result.total_return:.2%}, "
                     f"MaxDD={result.max_drawdown:.2%}, "
                     f"Trades={result.total_trades} ({elapsed:.0f}s)")
        return result
    except Exception as e:
        logger.error(f"  FAILED {name}: {e}")
        traceback.print_exc()
        return None


def extract_metrics(result: BacktestResult) -> dict:
    """Extract key metrics from a BacktestResult."""
    if result is None:
        return {}
    return {
        "total_return_pct": round(result.total_return * 100, 2),
        "annual_return_pct": round(result.annualized_return * 100, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "sortino_ratio": round(result.sortino_ratio, 3),
        "max_drawdown_pct": round(result.max_drawdown * 100, 2),
        "calmar_ratio": round(result.calmar_ratio, 3),
        "win_rate_pct": round(result.win_rate * 100, 1),
        "profit_factor": round(min(result.profit_factor, 99.9), 2),
        "total_trades": result.total_trades,
        "avg_trade_return_pct": round(
            float(np.mean([t.net_return for t in result.completed_trades if t.net_return is not None])) * 100, 2
        ) if result.completed_trades else 0,
        "avg_holding_days": round(result.avg_holding_days, 1),
        "max_consecutive_losses": result.max_consecutive_losses,
        "total_pnl": round(result.total_pnl, 0),
        "total_fees_paid": round(result.total_fees_paid, 0),
        "volatility_pct": round(result.volatility * 100, 2),
        "final_nav": round(result.daily_nav[-1][1], 0) if result.daily_nav else INITIAL_CAPITAL,
    }


def get_nav_df(result: BacktestResult) -> pd.DataFrame:
    """Convert daily_nav to a DataFrame."""
    if result is None or not result.daily_nav:
        return pd.DataFrame(columns=["date", "nav"])
    df = pd.DataFrame(result.daily_nav, columns=["date", "nav"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def get_drawdown_series(nav_df: pd.DataFrame) -> pd.Series:
    """Compute drawdown series from NAV DataFrame."""
    running_max = nav_df["nav"].cummax()
    return (nav_df["nav"] - running_max) / running_max


def get_rolling_sharpe(nav_df: pd.DataFrame, window: int = 240) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    returns = nav_df["nav"].pct_change().dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std * np.sqrt(240)).dropna()


# ===========================================================================
# TASK 1 & 2: Run Backtests
# ===========================================================================

def run_all_backtests():
    """Run baseline and all upgraded configs."""
    results = {}
    for name, config in SIGNAL_CONFIGS.items():
        results[name] = safe_run_backtest(name, config)
    return results


# ===========================================================================
# TASK 3: Generate Charts
# ===========================================================================

def load_nepse_index(start_date, end_date):
    """Load NEPSE index from database for benchmark overlay."""
    import sqlite3
    from backend.quant_pro.database import get_db_path
    conn = sqlite3.connect(str(get_db_path()))
    df = pd.read_sql(
        "SELECT date, close FROM stock_prices WHERE symbol='NEPSE' ORDER BY date",
        conn,
    )
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.set_index("date")
    return df


def chart_equity_curves(results, nepse_df):
    """Chart 1: Equity curve comparison."""
    style_dark()
    fig, ax = plt.subplots(figsize=(14, 7))

    baseline_nav = get_nav_df(results.get("baseline"))
    # Find the best upgraded result by Sharpe
    best_name = None
    best_sharpe = -999
    for name, res in results.items():
        if name == "baseline" or res is None:
            continue
        if res.sharpe_ratio > best_sharpe:
            best_sharpe = res.sharpe_ratio
            best_name = name

    upgraded_nav = get_nav_df(results.get(best_name)) if best_name else pd.DataFrame()

    if not baseline_nav.empty:
        ax.plot(baseline_nav.index, baseline_nav["nav"],
                color=COLORS["baseline"], linewidth=2, label="Baseline (Vol+Qual+LowVol)", alpha=0.9)

    if not upgraded_nav.empty:
        label = SIGNAL_CONFIGS[best_name]["label"]
        ax.plot(upgraded_nav.index, upgraded_nav["nav"],
                color=COLORS["upgraded"], linewidth=2, label=f"Best Upgraded: {label}", alpha=0.9)

    # NEPSE index normalized to same starting value
    if nepse_df is not None and not nepse_df.empty:
        nepse_normalized = nepse_df["close"] / nepse_df["close"].iloc[0] * INITIAL_CAPITAL
        ax.plot(nepse_normalized.index, nepse_normalized.values,
                color=COLORS["nepse"], linewidth=1.5, linestyle="--", label="NEPSE Index (normalized)", alpha=0.7)

    ax.set_title("NEPSE Quant System: Baseline vs Upgraded", fontsize=16, fontweight="bold")
    ax.set_ylabel("Portfolio Value (NPR)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"NPR {x:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # Add initial capital line
    ax.axhline(y=INITIAL_CAPITAL, color="#666666", linestyle=":", linewidth=1, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "equity_curve_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")
    return best_name


def chart_drawdowns(results, best_name):
    """Chart 2: Drawdown comparison."""
    style_dark()
    fig, ax = plt.subplots(figsize=(14, 5))

    baseline_nav = get_nav_df(results.get("baseline"))
    upgraded_nav = get_nav_df(results.get(best_name)) if best_name else pd.DataFrame()

    if not baseline_nav.empty:
        dd_baseline = get_drawdown_series(baseline_nav)
        ax.fill_between(dd_baseline.index, dd_baseline.values * 100, 0,
                        color=COLORS["baseline"], alpha=0.3, label="Baseline DD")
        ax.plot(dd_baseline.index, dd_baseline.values * 100,
                color=COLORS["baseline"], linewidth=1, alpha=0.8)

    if not upgraded_nav.empty:
        dd_upgraded = get_drawdown_series(upgraded_nav)
        ax.fill_between(dd_upgraded.index, dd_upgraded.values * 100, 0,
                        color=COLORS["upgraded"], alpha=0.3, label="Upgraded DD")
        ax.plot(dd_upgraded.index, dd_upgraded.values * 100,
                color=COLORS["upgraded"], linewidth=1, alpha=0.8)

    ax.set_title("Drawdown Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "drawdown_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def chart_monthly_returns_heatmap(results, best_name):
    """Chart 3: Monthly returns heatmap for the best upgraded strategy."""
    style_dark()

    # Use the best upgraded result
    target = results.get(best_name) if best_name else results.get("baseline")
    if target is None:
        logger.warning("  Skipping monthly returns heatmap - no result available")
        return

    monthly = target.monthly_returns()
    if monthly.empty:
        logger.warning("  Skipping monthly returns heatmap - no monthly data")
        return

    # Build year x month matrix
    monthly_df = monthly.to_frame(name="return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    # Pivot to year x month
    pivot = monthly_df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Custom colormap: red -> white -> green
    from matplotlib.colors import LinearSegmentedColormap
    colors_div = ["#EF5350", "#1C2127", "#66BB6A"]
    cmap = LinearSegmentedColormap.from_list("rg", colors_div, N=256)

    # Find the max absolute value for symmetric colorbar
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].min()),
               abs(pivot.values[~np.isnan(pivot.values)].max())) if not pivot.empty else 0.1

    im = ax.imshow(pivot.values * 100, cmap=cmap, aspect="auto", vmin=-vmax*100, vmax=vmax*100)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells with return values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "#111418" if abs(val) > vmax * 0.5 else COLORS["text"]
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Monthly Return (%)", fontsize=11)

    label = SIGNAL_CONFIGS.get(best_name, {}).get("label", "Strategy")
    ax.set_title(f"Monthly Returns Heatmap - {label}", fontsize=16, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "monthly_returns_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def chart_signal_contribution(results, best_name):
    """Chart 4: Signal contribution analysis."""
    style_dark()

    # Use the best upgraded result
    target = results.get(best_name) if best_name else results.get("baseline")
    if target is None:
        logger.warning("  Skipping signal contribution chart - no result available")
        return

    sig_breakdown = target.by_signal_type()
    if not sig_breakdown:
        logger.warning("  Skipping signal contribution chart - no signal breakdown")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sig_names = list(sig_breakdown.keys())
    counts = [sig_breakdown[s]["count"] for s in sig_names]
    win_rates = [sig_breakdown[s]["win_rate"] * 100 for s in sig_names]
    avg_rets = [sig_breakdown[s]["avg_return"] * 100 for s in sig_names]
    total_pnls = [sig_breakdown[s]["total_pnl"] for s in sig_names]

    # Clean up signal type names for display
    display_names = [s.replace("_", " ").title() for s in sig_names]

    palette = [COLORS["baseline"], COLORS["upgraded"], COLORS["accent1"],
               COLORS["accent2"], COLORS["accent4"], COLORS["nepse"],
               COLORS["accent3"], "#8D6E63", "#78909C"]

    # Chart 4a: Trade count by signal type
    bars = axes[0].barh(display_names, counts, color=palette[:len(sig_names)], alpha=0.85)
    axes[0].set_xlabel("Number of Trades")
    axes[0].set_title("Trade Count by Signal", fontweight="bold")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     str(count), va="center", fontsize=9, color=COLORS["text"])

    # Chart 4b: Win rate by signal type
    bar_colors = [COLORS["positive"] if wr >= 50 else COLORS["negative"] for wr in win_rates]
    bars = axes[1].barh(display_names, win_rates, color=bar_colors, alpha=0.85)
    axes[1].axvline(x=50, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("Win Rate (%)")
    axes[1].set_title("Win Rate by Signal", fontweight="bold")
    for bar, wr in zip(bars, win_rates):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{wr:.1f}%", va="center", fontsize=9, color=COLORS["text"])

    # Chart 4c: Total P&L by signal type
    pnl_colors = [COLORS["positive"] if p > 0 else COLORS["negative"] for p in total_pnls]
    bars = axes[2].barh(display_names, [p/1000 for p in total_pnls], color=pnl_colors, alpha=0.85)
    axes[2].axvline(x=0, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    axes[2].set_xlabel("Total P&L (NPR '000)")
    axes[2].set_title("P&L by Signal", fontweight="bold")
    for bar, pnl in zip(bars, total_pnls):
        axes[2].text(bar.get_width() + 0.5 if pnl >= 0 else bar.get_width() - 0.5,
                     bar.get_y() + bar.get_height()/2,
                     f"NPR {pnl:,.0f}", va="center", fontsize=8, color=COLORS["text"],
                     ha="left" if pnl >= 0 else "right")

    fig.suptitle("Signal Contribution Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "signal_contribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def chart_rolling_sharpe(results, best_name):
    """Chart 5: Rolling Sharpe ratio comparison."""
    style_dark()
    fig, ax = plt.subplots(figsize=(14, 6))

    baseline_nav = get_nav_df(results.get("baseline"))
    upgraded_nav = get_nav_df(results.get(best_name)) if best_name else pd.DataFrame()

    if not baseline_nav.empty:
        rs_baseline = get_rolling_sharpe(baseline_nav, window=240)
        if not rs_baseline.empty:
            ax.plot(rs_baseline.index, rs_baseline.values,
                    color=COLORS["baseline"], linewidth=1.5, label="Baseline (rolling 240d)", alpha=0.9)

    if not upgraded_nav.empty:
        rs_upgraded = get_rolling_sharpe(upgraded_nav, window=240)
        if not rs_upgraded.empty:
            ax.plot(rs_upgraded.index, rs_upgraded.values,
                    color=COLORS["upgraded"], linewidth=1.5, label="Upgraded (rolling 240d)", alpha=0.9)

    # Reference lines
    ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color=COLORS["accent2"], linestyle="--", linewidth=1, alpha=0.5, label="Sharpe = 1.0")
    ax.axhline(y=2.0, color=COLORS["accent1"], linestyle="--", linewidth=1, alpha=0.4, label="Sharpe = 2.0")

    ax.set_title("Rolling 240-Day Sharpe Ratio", fontsize=16, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "rolling_sharpe.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def chart_annual_returns(results, best_name):
    """Chart 6: Annual returns bar chart."""
    style_dark()

    baseline = results.get("baseline")
    upgraded = results.get(best_name) if best_name else None

    fig, ax = plt.subplots(figsize=(12, 6))

    def compute_annual_returns(result):
        if result is None or not result.daily_nav:
            return {}
        nav_df = pd.DataFrame(result.daily_nav, columns=["date", "nav"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")
        yearly = nav_df["nav"].resample("YE").last()
        yearly_start = nav_df["nav"].resample("YS").first()
        annual = {}
        for year_end, year_start in zip(yearly.items(), yearly_start.items()):
            yr = year_end[0].year
            if year_start[1] > 0:
                annual[yr] = (year_end[1] / year_start[1] - 1) * 100
        return annual

    baseline_annual = compute_annual_returns(baseline)
    upgraded_annual = compute_annual_returns(upgraded)

    all_years = sorted(set(list(baseline_annual.keys()) + list(upgraded_annual.keys())))
    if not all_years:
        logger.warning("  Skipping annual returns chart - no data")
        plt.close(fig)
        return

    x = np.arange(len(all_years))
    width = 0.35

    baseline_vals = [baseline_annual.get(y, 0) for y in all_years]
    upgraded_vals = [upgraded_annual.get(y, 0) for y in all_years]

    bars1 = ax.bar(x - width/2, baseline_vals, width,
                   label="Baseline", color=COLORS["baseline"], alpha=0.85)
    if upgraded:
        bars2 = ax.bar(x + width/2, upgraded_vals, width,
                       label="Best Upgraded", color=COLORS["upgraded"], alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=COLORS["text"])
    if upgraded:
        for bar, val in zip(bars2, upgraded_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    ax.set_title("Annual Returns Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1, alpha=0.5)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "annual_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def chart_risk_return_scatter(results, nepse_df):
    """Chart 7: Risk-return scatter plot with Sharpe contour lines."""
    style_dark()
    fig, ax = plt.subplots(figsize=(10, 8))

    points = []
    for name, result in results.items():
        if result is None:
            continue
        ann_ret = result.annualized_return * 100
        max_dd = abs(result.max_drawdown) * 100
        sharpe = result.sharpe_ratio
        label = SIGNAL_CONFIGS[name]["label"]
        points.append((ann_ret, max_dd, sharpe, label, name))

    # NEPSE index stats
    if nepse_df is not None and not nepse_df.empty:
        nepse_ret = nepse_df["close"].pct_change().dropna()
        nepse_ann_ret = float(nepse_ret.mean() * 240) * 100
        nepse_cummax = nepse_df["close"].cummax()
        nepse_dd = ((nepse_df["close"] - nepse_cummax) / nepse_cummax).min()
        nepse_max_dd = abs(float(nepse_dd)) * 100
        nepse_vol = float(nepse_ret.std() * np.sqrt(240))
        nepse_sharpe = float(nepse_ret.mean() / nepse_ret.std() * np.sqrt(240)) if nepse_ret.std() > 0 else 0
        points.append((nepse_ann_ret, nepse_max_dd, nepse_sharpe, "NEPSE Index", "nepse"))

    # Plot Sharpe contour lines (Sharpe = Return / Volatility; approximate with DD)
    dd_range = np.linspace(1, 50, 100)
    for s in [0.5, 1.0, 1.5, 2.0]:
        ret_line = s * dd_range * 0.5  # Rough approximation: volatility ~ DD * 0.5
        ax.plot(dd_range, ret_line, color="#404854", linestyle=":", linewidth=0.8, alpha=0.5)
        # Label at the right end
        if ret_line[-1] < 80:
            ax.text(dd_range[-1] + 0.5, ret_line[-1], f"SR~{s}",
                    color="#606060", fontsize=8, va="center")

    # Plot points
    color_map = {
        "baseline": COLORS["baseline"],
        "nepse": COLORS["nepse"],
    }
    default_colors = [COLORS["upgraded"], COLORS["accent1"], COLORS["accent2"], COLORS["accent4"]]
    color_idx = 0

    for ann_ret, max_dd, sharpe, label, name in points:
        if name in color_map:
            c = color_map[name]
        else:
            c = default_colors[color_idx % len(default_colors)]
            color_idx += 1

        ax.scatter(max_dd, ann_ret, s=150, c=c, edgecolors="white", linewidths=1.5,
                   zorder=5, alpha=0.9)
        ax.annotate(f"{label}\nSR={sharpe:.2f}",
                    (max_dd, ann_ret),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=8, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, alpha=0.5))

    ax.set_title("Risk-Return Scatter (All Configurations)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "risk_return_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ===========================================================================
# TASK 4: Save Metrics
# ===========================================================================

def save_metrics(results, best_name):
    """Save comprehensive metrics JSON."""
    baseline_metrics = extract_metrics(results.get("baseline"))
    upgraded_metrics = extract_metrics(results.get(best_name)) if best_name else {}

    # Compute improvement
    improvement = {}
    if baseline_metrics and upgraded_metrics:
        improvement = {
            "sharpe_delta": round(upgraded_metrics.get("sharpe_ratio", 0) -
                                  baseline_metrics.get("sharpe_ratio", 0), 3),
            "return_delta_pct": round(upgraded_metrics.get("total_return_pct", 0) -
                                     baseline_metrics.get("total_return_pct", 0), 2),
            "max_dd_improvement_pct": round(upgraded_metrics.get("max_drawdown_pct", 0) -
                                           baseline_metrics.get("max_drawdown_pct", 0), 2),
            "sortino_delta": round(upgraded_metrics.get("sortino_ratio", 0) -
                                   baseline_metrics.get("sortino_ratio", 0), 3),
            "win_rate_delta_pct": round(upgraded_metrics.get("win_rate_pct", 0) -
                                       baseline_metrics.get("win_rate_pct", 0), 1),
        }

    # All configs comparison
    all_configs = {}
    for name, result in results.items():
        if result is not None:
            all_configs[name] = {
                "label": SIGNAL_CONFIGS[name]["label"],
                "signals": SIGNAL_CONFIGS[name]["signal_types"],
                "metrics": extract_metrics(result),
            }

    metrics = {
        "baseline": baseline_metrics,
        "best_upgraded": {
            "name": best_name,
            "label": SIGNAL_CONFIGS.get(best_name, {}).get("label", "N/A"),
            "signals": SIGNAL_CONFIGS.get(best_name, {}).get("signal_types", []),
            "metrics": upgraded_metrics,
        },
        "improvement": improvement,
        "all_configurations": all_configs,
        "config": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "holding_days": HOLDING_DAYS,
        },
    }

    path = os.path.join(REPORTS_DIR, "backtest_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  Saved: {path}")


def save_equity_curves(results):
    """Save equity curves to CSV."""
    dfs = []
    for name, result in results.items():
        if result is None or not result.daily_nav:
            continue
        nav_df = pd.DataFrame(result.daily_nav, columns=["date", f"nav_{name}"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        dfs.append(nav_df.set_index("date"))

    if not dfs:
        logger.warning("  No equity curves to save")
        return

    combined = pd.concat(dfs, axis=1)
    path = os.path.join(REPORTS_DIR, "equity_curves.csv")
    combined.to_csv(path)
    logger.info(f"  Saved: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_global = time.time()

    logger.info("=" * 70)
    logger.info("NEPSE QUANT SYSTEM - COMPREHENSIVE BACKTEST & CHARTING")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Capital: NPR {INITIAL_CAPITAL:,.0f}")
    logger.info("=" * 70)

    # TASK 1 & 2: Run backtests
    logger.info("\n>>> PHASE 1/2: Running all backtest configurations...")
    results = run_all_backtests()

    # Determine best upgraded config
    best_name = None
    best_sharpe = -999
    for name, res in results.items():
        if name == "baseline" or res is None:
            continue
        if res.sharpe_ratio > best_sharpe:
            best_sharpe = res.sharpe_ratio
            best_name = name

    # Print comparison table
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 90)
    header = f"{'Config':<50} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>8}"
    logger.info(header)
    logger.info("-" * 90)
    for name, result in results.items():
        if result is None:
            logger.info(f"{'  ' + SIGNAL_CONFIGS[name]['label']:<50} {'FAILED':>8}")
            continue
        marker = " ***" if name == best_name else ""
        logger.info(
            f"  {SIGNAL_CONFIGS[name]['label']:<48} "
            f"{result.sharpe_ratio:>8.3f} "
            f"{result.total_return:>9.2%} "
            f"{result.max_drawdown:>9.2%} "
            f"{result.total_trades:>8d} "
            f"{result.win_rate:>7.1%}"
            f"{marker}"
        )
    logger.info("=" * 90)

    # TASK 3: Generate charts
    logger.info("\n>>> PHASE 3: Generating publication-quality charts...")
    nepse_df = load_nepse_index(START_DATE, END_DATE)

    best_from_chart = chart_equity_curves(results, nepse_df)
    chart_drawdowns(results, best_name)
    chart_monthly_returns_heatmap(results, best_name)
    chart_signal_contribution(results, best_name)
    chart_rolling_sharpe(results, best_name)
    chart_annual_returns(results, best_name)
    chart_risk_return_scatter(results, nepse_df)

    # TASK 4: Save metrics
    logger.info("\n>>> PHASE 4: Saving metrics and equity curves...")
    save_metrics(results, best_name)
    save_equity_curves(results)

    # Print final summary
    elapsed = time.time() - t_global
    logger.info(f"\nTotal elapsed time: {elapsed:.0f}s")
    logger.info(f"Best upgraded config: {best_name} (Sharpe={best_sharpe:.3f})")

    if results.get("baseline") and best_name and results.get(best_name):
        bl = results["baseline"]
        up = results[best_name]
        logger.info(f"\n  BASELINE:  Sharpe={bl.sharpe_ratio:.3f}, Return={bl.total_return:.2%}, MaxDD={bl.max_drawdown:.2%}")
        logger.info(f"  UPGRADED:  Sharpe={up.sharpe_ratio:.3f}, Return={up.total_return:.2%}, MaxDD={up.max_drawdown:.2%}")
        logger.info(f"  DELTA:     Sharpe={up.sharpe_ratio - bl.sharpe_ratio:+.3f}, "
                     f"Return={up.total_return - bl.total_return:+.2%}, "
                     f"MaxDD={up.max_drawdown - bl.max_drawdown:+.2%}")

    # Print full summary of best result
    if best_name and results.get(best_name):
        print("\n" + results[best_name].summary())

    logger.info("\nDone! Charts saved to: reports/charts/")
    logger.info("Metrics saved to: reports/backtest_metrics.json")
    logger.info("Equity curves saved to: reports/equity_curves.csv")


if __name__ == "__main__":
    main()
