"""
Institutional-grade validation report generator.

Produces:
- Console summary with PASS/FAIL for each test
- JSON results file
- Multi-page PDF with professional dark-theme plots
  (equity curve, drawdown, rolling Sharpe, Monte Carlo distributions,
   sensitivity heatmaps, walk-forward folds, regime analysis, etc.)
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Palantir-inspired dark theme ─────────────────────────────────────────
BG       = "#0B0E11"
CARD     = "#151A20"
SURFACE  = "#1C2127"
BORDER   = "#2F343C"
TEXT     = "#F6F7F9"
TEXT2    = "#ABB3BF"
TEXT3    = "#738091"
ACCENT   = "#2D72D2"
GREEN    = "#238551"
RED      = "#CD4246"
AMBER    = "#D1980B"
ORANGE   = "#C87619"
CYAN     = "#17B8A6"
PURPLE   = "#7961DB"


def ensure_output_dir(output_dir: str) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(exist_ok=True)
    return base


def save_json_report(results: dict, output_dir: str) -> str:
    base = ensure_output_dir(output_dir)
    path = base / "validation_results.json"
    clean = _make_serializable(results)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info(f"JSON report saved to {path}")
    return str(path)


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "__dataclass_fields__"):
        return _make_serializable(
            {k: getattr(obj, k) for k in obj.__dataclass_fields__
             if not k.startswith("all_")}
        )
    return obj


def print_console_report(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  NEPSE QUANTITATIVE TRADING SYSTEM")
    print("  Production Readiness Validation Report")
    print("=" * 72)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)

    tests = results.get("tests", {})
    for test_name, test_data in tests.items():
        status = test_data.get("status", "UNKNOWN")
        icon = "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "SKIP"
        details = test_data.get("summary", "")
        print(f"\n  [{icon}] {test_name}")
        if details:
            print(f"         {details}")

    print("\n" + "=" * 72)
    go_nogo = results.get("go_nogo", "UNKNOWN")
    if go_nogo == "GO":
        print("  VERDICT:  GO — Strategy meets all production readiness criteria.")
    else:
        reason = results.get("reason", "One or more tests failed.")
        print(f"  VERDICT:  NO-GO — {reason}")
    elapsed = results.get("elapsed_seconds", 0)
    print(f"  Elapsed:  {elapsed:.0f}s")
    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════

def _style_ax(ax, fig, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply institutional dark theme to axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT3, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)


def _add_watermark(fig):
    """Add subtle watermark."""
    fig.text(0.99, 0.01, "NEPSE Quant | Confidential",
             fontsize=6, color=TEXT3, alpha=0.4,
             ha="right", va="bottom", fontfamily="monospace")


# ═══════════════════════════════════════════════════════════════════════════
# Individual plot generators
# ═══════════════════════════════════════════════════════════════════════════

def _plot_equity_and_drawdown(results: dict, plots_dir: Path) -> Optional[str]:
    """P1: Equity curve + underwater drawdown chart."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    base = results.get("tests", {}).get("base_backtest", {}).get("details", {})
    daily_nav = results.get("_daily_nav", [])
    if not daily_nav or len(daily_nav) < 10:
        return None

    dates = [d for d, _ in daily_nav]
    navs = np.array([n for _, n in daily_nav])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.08})

    # Equity curve
    _style_ax(ax1, fig, title="Portfolio Equity Curve", ylabel="NAV (NPR)")
    ax1.fill_between(range(len(navs)), navs, navs.min() * 0.95,
                     color=ACCENT, alpha=0.08)
    ax1.plot(range(len(navs)), navs, color=ACCENT, linewidth=1.2)

    # Mark peak and trough
    peak_idx = np.argmax(navs)
    ax1.plot(peak_idx, navs[peak_idx], "^", color=GREEN, markersize=8, zorder=5)
    ax1.annotate(f"Peak: NPR {navs[peak_idx]:,.0f}",
                 (peak_idx, navs[peak_idx]), textcoords="offset points",
                 xytext=(10, 10), fontsize=7, color=GREEN)

    # Initial capital line
    ax1.axhline(navs[0], color=TEXT3, linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
    ax1.set_xticklabels([])

    # Drawdown
    peak_nav = np.maximum.accumulate(navs)
    dd = (navs - peak_nav) / peak_nav
    _style_ax(ax2, fig, xlabel="Trading Days", ylabel="Drawdown")
    ax2.fill_between(range(len(dd)), dd, 0, color=RED, alpha=0.3)
    ax2.plot(range(len(dd)), dd, color=RED, linewidth=0.8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Max drawdown annotation
    max_dd_idx = np.argmin(dd)
    ax2.annotate(f"Max DD: {dd[max_dd_idx]:.1%}",
                 (max_dd_idx, dd[max_dd_idx]), textcoords="offset points",
                 xytext=(10, -10), fontsize=7, color=RED)

    _add_watermark(fig)
    path = plots_dir / "01_equity_drawdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_rolling_sharpe(results: dict, plots_dir: Path) -> Optional[str]:
    """P2: 63-day rolling Sharpe ratio."""
    import matplotlib.pyplot as plt

    daily_nav = results.get("_daily_nav", [])
    if not daily_nav or len(daily_nav) < 100:
        return None

    navs = np.array([n for _, n in daily_nav])
    rets = np.diff(navs) / navs[:-1]

    window = 63  # ~3 months
    if len(rets) < window + 10:
        return None

    rolling_sharpe = []
    for i in range(window, len(rets)):
        chunk = rets[i - window:i]
        std = np.std(chunk, ddof=1)
        if std > 0:
            rolling_sharpe.append(np.mean(chunk) / std * np.sqrt(240))
        else:
            rolling_sharpe.append(0.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    _style_ax(ax, fig, title="63-Day Rolling Sharpe Ratio",
              xlabel="Trading Days", ylabel="Sharpe Ratio")

    x = range(window, window + len(rolling_sharpe))
    rs = np.array(rolling_sharpe)

    ax.fill_between(x, rs, 0, where=(rs > 0), color=GREEN, alpha=0.15)
    ax.fill_between(x, rs, 0, where=(rs <= 0), color=RED, alpha=0.15)
    ax.plot(x, rs, color=ACCENT, linewidth=1.0)
    ax.axhline(0, color=TEXT3, linewidth=0.5)
    ax.axhline(1.0, color=GREEN, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1.0, color=RED, linewidth=0.5, linestyle="--", alpha=0.5)

    # Annotate mean
    mean_sr = np.mean(rs)
    ax.axhline(mean_sr, color=AMBER, linewidth=0.8, linestyle="-.")
    ax.text(len(rs) * 0.02 + window, mean_sr + 0.15,
            f"Mean: {mean_sr:.2f}", fontsize=7, color=AMBER)

    _add_watermark(fig)
    path = plots_dir / "02_rolling_sharpe.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_monthly_returns(results: dict, plots_dir: Path) -> Optional[str]:
    """P3: Monthly returns heatmap-style bar chart."""
    import matplotlib.pyplot as plt
    import pandas as pd

    daily_nav = results.get("_daily_nav", [])
    if not daily_nav or len(daily_nav) < 30:
        return None

    df = pd.DataFrame(daily_nav, columns=["date", "nav"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    monthly = df["nav"].resample("ME").last().ffill().pct_change().dropna()

    if len(monthly) < 3:
        return None

    fig, ax = plt.subplots(figsize=(12, 4.5))
    _style_ax(ax, fig, title="Monthly Returns",
              xlabel="Month", ylabel="Return")

    colors = [GREEN if v >= 0 else RED for v in monthly.values]
    labels = [d.strftime("%b'%y") for d in monthly.index]
    bars = ax.bar(range(len(monthly)), monthly.values * 100, color=colors, alpha=0.8,
                  width=0.7, edgecolor=BORDER, linewidth=0.3)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(labels, rotation=60, fontsize=6)
    ax.axhline(0, color=TEXT3, linewidth=0.5)

    # Annotate key stats
    pos = sum(1 for v in monthly.values if v > 0)
    neg = len(monthly) - pos
    ax.text(0.98, 0.95,
            f"Positive: {pos}/{len(monthly)} ({pos/len(monthly):.0%})\n"
            f"Best:   {monthly.max()*100:+.1f}%\n"
            f"Worst:  {monthly.min()*100:+.1f}%\n"
            f"Mean:   {monthly.mean()*100:+.1f}%",
            transform=ax.transAxes, fontsize=7, color=TEXT2,
            va="top", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=BORDER, alpha=0.9))

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    _add_watermark(fig)
    path = plots_dir / "03_monthly_returns.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_walk_forward(results: dict, plots_dir: Path) -> Optional[str]:
    """P4: Walk-forward fold Sharpe bars + cumulative OOS equity."""
    import matplotlib.pyplot as plt

    wf = results.get("tests", {}).get("walk_forward", {}).get("details", {})
    fold_metrics = wf.get("fold_metrics", [])
    if not fold_metrics:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left: OOS Sharpe by fold
    _style_ax(ax1, fig, title="Walk-Forward OOS Sharpe by Fold",
              xlabel="Fold", ylabel="Sharpe Ratio")
    sharpes = [f.get("sharpe", 0) if isinstance(f, dict) else f.sharpe
               for f in fold_metrics]
    colors = [GREEN if s > 0 else RED for s in sharpes]
    ax1.bar(range(len(sharpes)), sharpes, color=colors, alpha=0.85,
            edgecolor=BORDER, linewidth=0.3, width=0.7)
    ax1.axhline(0, color=TEXT3, linewidth=0.5)
    ax1.axhline(0.8, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(len(sharpes) + 0.3, 0.8, "Threshold", fontsize=7, color=AMBER, va="center")

    mean_s = np.mean(sharpes)
    ax1.axhline(mean_s, color=CYAN, linewidth=0.8, linestyle="-.")
    ax1.text(len(sharpes) + 0.3, mean_s, f"Mean: {mean_s:.2f}",
             fontsize=7, color=CYAN, va="center")

    # Right: summary stats table
    _style_ax(ax2, fig, title="OOS Summary Statistics")
    ax2.axis("off")

    stats = [
        ("Folds", f"{len(fold_metrics)}"),
        ("Mean Sharpe", f"{np.mean(sharpes):.3f}"),
        ("Median Sharpe", f"{np.median(sharpes):.3f}"),
        ("Std Sharpe", f"{np.std(sharpes):.3f}"),
        ("Min Sharpe", f"{np.min(sharpes):.3f}"),
        ("Max Sharpe", f"{np.max(sharpes):.3f}"),
        ("% Positive", f"{sum(1 for s in sharpes if s > 0)/len(sharpes):.0%}"),
        ("Mean Return", f"{wf.get('mean_return', 0):.2%}"),
        ("Mean Max DD", f"{wf.get('mean_max_dd', 0):.2%}"),
    ]
    y = 0.92
    for label, value in stats:
        ax2.text(0.1, y, label, fontsize=9, color=TEXT2, transform=ax2.transAxes)
        ax2.text(0.85, y, value, fontsize=9, color=TEXT, fontweight="bold",
                 ha="right", transform=ax2.transAxes, fontfamily="monospace")
        y -= 0.10

    _add_watermark(fig)
    path = plots_dir / "04_walk_forward.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_monte_carlo(results: dict, plots_dir: Path) -> Optional[str]:
    """P5: Monte Carlo terminal wealth + max drawdown distributions."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    mc = results.get("tests", {}).get("monte_carlo", {}).get("details", {})
    all_tw = np.array(mc.get("all_terminal_wealth", []))
    all_dd = np.array(mc.get("all_max_dd", []))
    if len(all_tw) < 10:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Terminal wealth
    _style_ax(ax1, fig, title="Monte Carlo: Terminal Wealth Distribution",
              xlabel="Terminal Wealth (NPR)", ylabel="Density")
    ax1.hist(all_tw, bins=80, density=True, color=ACCENT, alpha=0.7,
             edgecolor=BG, linewidth=0.2)
    init_cap = 1_000_000
    ax1.axvline(init_cap, color=RED, linewidth=1.5, linestyle="-", label="Initial Capital")
    median_tw = np.median(all_tw)
    ax1.axvline(median_tw, color=GREEN, linewidth=1.2, linestyle="--",
                label=f"Median: {median_tw/1e6:.2f}M")
    pct5 = np.percentile(all_tw, 5)
    ax1.axvline(pct5, color=AMBER, linewidth=1, linestyle=":",
                label=f"5th pct: {pct5/1e6:.2f}M")
    ax1.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT2)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    # Prob annotations
    prob_loss = float(np.mean(all_tw < init_cap))
    prob_ruin = float(np.mean(all_tw < init_cap * 0.5))
    ax1.text(0.97, 0.95,
             f"P(loss):  {prob_loss:.1%}\n"
             f"P(ruin):  {prob_ruin:.1%}\n"
             f"N={len(all_tw):,}",
             transform=ax1.transAxes, fontsize=7, color=TEXT2,
             va="top", ha="right", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                       edgecolor=BORDER, alpha=0.9))

    # Max drawdown
    _style_ax(ax2, fig, title="Monte Carlo: Max Drawdown Distribution",
              xlabel="Max Drawdown", ylabel="Density")
    ax2.hist(all_dd, bins=60, density=True, color=RED, alpha=0.6,
             edgecolor=BG, linewidth=0.2)
    med_dd = np.median(all_dd)
    ax2.axvline(med_dd, color=AMBER, linewidth=1.2, linestyle="--",
                label=f"Median: {med_dd:.1%}")
    pct95_dd = np.percentile(all_dd, 5)  # 5th pct of DD = worst
    ax2.axvline(pct95_dd, color=RED, linewidth=1.2, linestyle=":",
                label=f"95th worst: {pct95_dd:.1%}")
    ax2.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT2)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    _add_watermark(fig)
    path = plots_dir / "05_monte_carlo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_sensitivity(results: dict, plots_dir: Path) -> Optional[str]:
    """P6: Parameter sensitivity grid — 2x3 subplots."""
    import matplotlib.pyplot as plt

    sens = results.get("tests", {}).get("sensitivity", {}).get("details", {})
    param_summaries = sens.get("param_summaries", {})
    if not param_summaries:
        return None

    params = list(param_summaries.items())
    n = len(params)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, summary) in enumerate(params):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        _style_ax(ax, fig, title=name, ylabel="Sharpe")

        values = summary.get("values", [])
        sharpes = summary.get("sharpes", [])
        if not values or not sharpes:
            continue

        valid = [(v, s) for v, s in zip(values, sharpes)
                 if s is not None and not (isinstance(s, float) and np.isnan(s))]
        if not valid:
            continue

        vs, ss = zip(*valid)
        is_smooth = summary.get("is_smooth", True)
        line_color = GREEN if is_smooth else RED

        ax.fill_between(vs, ss, min(ss) - 0.1, color=line_color, alpha=0.08)
        ax.plot(vs, ss, "o-", color=line_color, linewidth=1.5, markersize=4,
                markeredgecolor=BG, markeredgewidth=0.5)

        best_v = summary.get("best_value")
        best_s = summary.get("best_sharpe", 0)
        if best_v is not None:
            ax.plot(best_v, best_s, "*", color=AMBER, markersize=12, zorder=5)

        if not is_smooth:
            ax.text(0.5, 0.02, "CLIFF DETECTED", transform=ax.transAxes,
                    fontsize=8, color=RED, fontweight="bold",
                    ha="center", va="bottom")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Parameter Sensitivity Analysis", color=TEXT, fontsize=13,
                 fontweight="bold", y=1.02)
    _add_watermark(fig)
    path = plots_dir / "06_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_random_baseline(results: dict, plots_dir: Path) -> Optional[str]:
    """P7: Strategy vs random baseline distribution."""
    import matplotlib.pyplot as plt

    rb = results.get("tests", {}).get("random_baseline", {}).get("details", {})
    random_sharpes = np.array(rb.get("random_sharpes", []))
    actual = rb.get("actual_sharpe")
    if len(random_sharpes) < 10 or actual is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, fig, title="Signal Edge: Strategy vs Random Entry Baseline",
              xlabel="Sharpe Ratio", ylabel="Density")

    ax.hist(random_sharpes, bins=50, density=True, color=TEXT3, alpha=0.5,
            edgecolor=BG, linewidth=0.2, label="Random entries")
    ax.axvline(actual, color=ACCENT, linewidth=2.5,
               label=f"Actual strategy: {actual:.2f}")
    pct95 = np.percentile(random_sharpes, 95)
    ax.axvline(pct95, color=AMBER, linewidth=1, linestyle="--",
               label=f"95th percentile: {pct95:.2f}")

    # Shade the area beyond actual
    x_fill = np.linspace(actual, random_sharpes.max() + 0.5, 100)
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(random_sharpes)
        ax.fill_between(x_fill, kde(x_fill), alpha=0.3, color=ACCENT)
    except Exception:
        pass

    rank = rb.get("percentile_rank", 0)
    pval = rb.get("p_value", 1)
    ax.text(0.97, 0.95,
            f"Percentile rank: {rank:.1f}%\n"
            f"p-value: {pval:.4f}\n"
            f"Random mean: {rb.get('mean_random', 0):.3f}\n"
            f"Random std:  {rb.get('std_random', 0):.3f}\n"
            f"N={rb.get('n_simulations', 0):,}",
            transform=ax.transAxes, fontsize=8, color=TEXT2,
            va="top", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE,
                      edgecolor=BORDER, alpha=0.9))

    ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT2)
    _add_watermark(fig)
    path = plots_dir / "07_random_baseline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_regime_performance(results: dict, plots_dir: Path) -> Optional[str]:
    """P8: Regime-by-regime performance comparison."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    regime_data = results.get("tests", {}).get("regime_stress", {}).get("details", {})
    regimes = regime_data.get("regimes", [])
    if not regimes:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = [r["name"] for r in regimes]
    sharpes = [r.get("sharpe", 0) for r in regimes]
    max_dds = [r.get("max_dd", 0) for r in regimes]
    returns = [r.get("return", 0) for r in regimes]

    # Sharpe by regime
    _style_ax(ax1, fig, title="Sharpe Ratio by Market Regime", ylabel="Sharpe")
    colors = [GREEN if s > 0 else RED for s in sharpes]
    bars = ax1.barh(range(len(names)), sharpes, color=colors, alpha=0.8,
                    edgecolor=BORDER, linewidth=0.3, height=0.6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8, color=TEXT2)
    ax1.axvline(0, color=TEXT3, linewidth=0.5)

    for i, (s, r) in enumerate(zip(sharpes, returns)):
        ax1.text(max(s, 0) + 0.05, i, f"SR={s:.2f}, Ret={r:.0%}",
                 fontsize=7, color=TEXT2, va="center")

    # Max drawdown by regime
    _style_ax(ax2, fig, title="Max Drawdown by Market Regime", ylabel="Max DD")
    bars2 = ax2.barh(range(len(names)), [d * 100 for d in max_dds],
                     color=[RED if d < -0.2 else AMBER if d < -0.1 else GREEN for d in max_dds],
                     alpha=0.8, edgecolor=BORDER, linewidth=0.3, height=0.6)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8, color=TEXT2)
    ax2.axvline(-40, color=RED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.text(-40, len(names) - 0.2, "Limit: -40%", fontsize=7, color=RED)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    _add_watermark(fig)
    path = plots_dir / "08_regime_performance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_risk_metrics_dashboard(results: dict, plots_dir: Path) -> Optional[str]:
    """P9: Key risk metrics dashboard — single overview page."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    base = results.get("tests", {}).get("base_backtest", {}).get("details", {})
    stat = results.get("tests", {}).get("statistical_significance", {}).get("details", {})
    mc = results.get("tests", {}).get("monte_carlo", {}).get("details", {})
    bench = results.get("tests", {}).get("benchmark", {}).get("details", {})

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    ax.text(0.5, 0.97, "Risk & Performance Dashboard", fontsize=16,
            fontweight="bold", color=TEXT, ha="center", va="top",
            transform=ax.transAxes)

    # Metric cards — 4 columns, 4 rows
    metrics = [
        # Row 1: Returns
        ("Sharpe Ratio", f"{base.get('sharpe', 0):.3f}", ACCENT),
        ("Sortino Ratio", f"{base.get('sortino', 0):.3f}", ACCENT),
        ("CAGR", f"{base.get('annualized_return', 0):.1%}", GREEN),
        ("Total Return", f"{base.get('total_return', 0):.1%}", GREEN),
        # Row 2: Risk
        ("Max Drawdown", f"{base.get('max_drawdown', 0):.1%}", RED),
        ("Calmar Ratio", f"{base.get('calmar', 0):.2f}", AMBER),
        ("Win Rate", f"{base.get('win_rate', 0):.1%}", ACCENT),
        ("Profit Factor", f"{base.get('profit_factor', 0):.2f}", ACCENT),
        # Row 3: Statistical
        ("PSR", f"{stat.get('psr', 0):.3f}", GREEN if stat.get('psr', 0) >= 0.9 else RED),
        ("DSR", f"{stat.get('dsr', 0):.3f}", GREEN if stat.get('dsr', 0) >= 0.85 else RED),
        ("t-stat", f"{stat.get('ttest_t_stat', 0):.2f}", ACCENT),
        ("MinTRL (yr)", f"{stat.get('min_trl_years', 0):.1f}", AMBER),
        # Row 4: Benchmark & MC
        ("Alpha", f"{bench.get('alpha', 0):.3f}", GREEN if bench.get('alpha', 0) > 0 else RED),
        ("Beta", f"{bench.get('beta', 0):.2f}", ACCENT),
        ("MC P(ruin)", f"{mc.get('prob_ruin', 0):.1%}", GREEN if mc.get('prob_ruin', 0) < 0.05 else RED),
        ("Total Trades", f"{base.get('total_trades', 0):,}", TEXT2),
    ]

    cols, rows_grid = 4, 4
    card_w, card_h = 0.22, 0.16
    x_start, y_start = 0.03, 0.78

    for idx, (label, value, color) in enumerate(metrics):
        r, c = divmod(idx, cols)
        x = x_start + c * 0.245
        y = y_start - r * 0.20

        # Card background
        rect = FancyBboxPatch((x, y), card_w, card_h, transform=ax.transAxes,
                               boxstyle="round,pad=0.01", facecolor=CARD,
                               edgecolor=BORDER, linewidth=0.5)
        ax.add_patch(rect)

        # Value
        ax.text(x + card_w / 2, y + card_h * 0.62, value, fontsize=16,
                fontweight="bold", color=color, ha="center", va="center",
                transform=ax.transAxes, fontfamily="monospace")
        # Label
        ax.text(x + card_w / 2, y + card_h * 0.22, label, fontsize=8,
                color=TEXT3, ha="center", va="center", transform=ax.transAxes)

    _add_watermark(fig)
    path = plots_dir / "09_risk_dashboard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def _plot_trade_analysis(results: dict, plots_dir: Path) -> Optional[str]:
    """P10: Trade-level analysis — return distribution + exit reasons."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    trades = results.get("_trades", [])
    if not trades:
        return None

    returns = [t.net_return for t in trades if t.net_return is not None]
    exit_reasons = {}
    for t in trades:
        if t.exit_reason:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    if not returns:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Return distribution
    _style_ax(ax1, fig, title="Per-Trade Net Return Distribution",
              xlabel="Net Return", ylabel="Count")
    ret_arr = np.array(returns) * 100
    bins = np.linspace(max(ret_arr.min(), -50), min(ret_arr.max(), 50), 50)
    n, edges, patches = ax1.hist(ret_arr, bins=bins, edgecolor=BG, linewidth=0.3)
    for patch, left_edge in zip(patches, edges):
        patch.set_facecolor(GREEN if left_edge >= 0 else RED)
        patch.set_alpha(0.7)
    ax1.axvline(0, color=TEXT3, linewidth=0.8)
    ax1.axvline(np.mean(ret_arr), color=AMBER, linewidth=1, linestyle="--",
                label=f"Mean: {np.mean(ret_arr):.1f}%")
    ax1.axvline(np.median(ret_arr), color=CYAN, linewidth=1, linestyle=":",
                label=f"Median: {np.median(ret_arr):.1f}%")
    ax1.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT2)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%"))

    # Exit reasons pie
    _style_ax(ax2, fig, title="Exit Reasons")
    ax2.axis("equal")
    if exit_reasons:
        labels = list(exit_reasons.keys())
        sizes = list(exit_reasons.values())
        pie_colors = [AMBER, CYAN, RED, GREEN, PURPLE, ACCENT][:len(labels)]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=None, autopct="%1.0f%%",
            colors=pie_colors, pctdistance=0.75,
            textprops={"fontsize": 7, "color": TEXT},
            wedgeprops={"edgecolor": BG, "linewidth": 0.5},
        )
        ax2.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, sizes)],
                   loc="lower center", fontsize=7,
                   facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT2)

    _add_watermark(fig)
    path = plots_dir / "10_trade_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Main generators
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: dict, output_dir: str) -> List[str]:
    """Generate all PNG plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        })
    except ImportError:
        logger.warning("matplotlib not available")
        return []

    base = ensure_output_dir(output_dir)
    plots_dir = base / "plots"
    saved = []

    generators = [
        _plot_equity_and_drawdown,
        _plot_rolling_sharpe,
        _plot_monthly_returns,
        _plot_walk_forward,
        _plot_monte_carlo,
        _plot_sensitivity,
        _plot_random_baseline,
        _plot_regime_performance,
        _plot_risk_metrics_dashboard,
        _plot_trade_analysis,
    ]

    for gen in generators:
        try:
            path = gen(results, plots_dir)
            if path:
                saved.append(path)
                logger.info(f"  Generated: {Path(path).name}")
        except Exception as e:
            logger.warning(f"  Failed {gen.__name__}: {e}")

    logger.info(f"Generated {len(saved)} plots in {plots_dir}")
    return saved


def generate_pdf_report(results: dict, output_dir: str) -> Optional[str]:
    """
    Generate institutional-grade multi-page PDF report.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logger.warning("matplotlib not available — skipping PDF")
        return None

    base = ensure_output_dir(output_dir)
    pdf_path = base / "validation_report.pdf"

    try:
        with PdfPages(str(pdf_path)) as pdf:
            # ── Page 1: Title ────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            fig.patch.set_facecolor(BG)

            ax.text(0.5, 0.75, "NEPSE QUANTITATIVE TRADING SYSTEM",
                    fontsize=24, fontweight="bold", ha="center", va="center",
                    color=TEXT, transform=ax.transAxes, fontfamily="sans-serif")
            ax.text(0.5, 0.65, "Production Readiness Validation Report",
                    fontsize=16, ha="center", va="center", color=TEXT2,
                    transform=ax.transAxes)
            ax.text(0.5, 0.55, datetime.now().strftime("%B %d, %Y  |  %H:%M"),
                    fontsize=12, ha="center", va="center", color=TEXT3,
                    transform=ax.transAxes)

            verdict = results.get("go_nogo", "UNKNOWN")
            v_color = GREEN if verdict == "GO" else RED
            # Verdict box
            rect = FancyBboxPatch((0.25, 0.32), 0.5, 0.12, transform=ax.transAxes,
                                   boxstyle="round,pad=0.02", facecolor=CARD,
                                   edgecolor=v_color, linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, 0.38, f"VERDICT:  {verdict}",
                    fontsize=22, fontweight="bold", ha="center", va="center",
                    color=v_color, transform=ax.transAxes, fontfamily="monospace")

            elapsed = results.get("elapsed_seconds", 0)
            config = results.get("config", {})
            bp = config.get("backtest_params", {})
            ax.text(0.5, 0.20,
                    f"Period: {config.get('start_date', '?')} to {config.get('end_date', '?')}\n"
                    f"Signals: {', '.join(bp.get('signal_types', []))}\n"
                    f"Holding: {bp.get('holding_days', '?')} trading days  |  "
                    f"Max Positions: {bp.get('max_positions', '?')}\n"
                    f"Validation runtime: {elapsed:.0f}s",
                    fontsize=10, ha="center", va="center", color=TEXT3,
                    transform=ax.transAxes, linespacing=1.8)

            _add_watermark(fig)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

            # ── Page 2: Test Results Summary ─────────────────────────
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            fig.patch.set_facecolor(BG)

            ax.text(0.05, 0.95, "Validation Test Results",
                    fontsize=16, fontweight="bold", color=TEXT,
                    transform=ax.transAxes)

            y = 0.88
            tests = results.get("tests", {})
            for name, data in tests.items():
                status = data.get("status", "UNKNOWN")
                summary = data.get("summary", "")
                s_color = GREEN if status == "PASS" else RED if status == "FAIL" else TEXT3

                # Status badge
                rect = FancyBboxPatch((0.05, y - 0.012), 0.06, 0.024,
                                       transform=ax.transAxes,
                                       boxstyle="round,pad=0.005",
                                       facecolor=s_color, edgecolor="none", alpha=0.15)
                ax.add_patch(rect)
                ax.text(0.08, y, status, fontsize=8, fontweight="bold",
                        color=s_color, ha="center", va="center",
                        transform=ax.transAxes, fontfamily="monospace")

                # Test name
                display_name = name.replace("_", " ").title()
                ax.text(0.13, y, display_name, fontsize=10, color=TEXT,
                        va="center", transform=ax.transAxes)

                # Summary
                if summary:
                    # Truncate long summaries
                    disp = summary if len(summary) < 85 else summary[:82] + "..."
                    ax.text(0.13, y - 0.025, disp, fontsize=7, color=TEXT3,
                            va="center", transform=ax.transAxes, fontfamily="monospace")

                y -= 0.065
                if y < 0.03:
                    break

            _add_watermark(fig)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

            # ── Pages 3+: Embed all generated plots ──────────────────
            plots_dir = base / "plots"
            if plots_dir.exists():
                for png in sorted(plots_dir.glob("*.png")):
                    try:
                        img = plt.imread(str(png))
                        h, w = img.shape[:2]
                        aspect = w / h
                        fig_w = 11
                        fig_h = fig_w / aspect
                        fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 8.5)))
                        ax.imshow(img)
                        ax.axis("off")
                        fig.patch.set_facecolor(BG)
                        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        pdf.savefig(fig, facecolor=fig.get_facecolor())
                        plt.close(fig)
                    except Exception:
                        pass

        logger.info(f"PDF report saved to {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None
