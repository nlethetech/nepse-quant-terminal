"""
Quick chart generator — strategy NAV vs NEPSE benchmark.

Produces a single professional PNG without running the full validation suite.
Called after every TUI strategy backtest. Auto-opens in system viewer.

Layout
------
Left col (3/4 width):
  - Equity curve with real calendar date x-axis and month boundaries
  - Monthly return bar chart (strategy green/red bars, NEPSE grey dots)
Right col (1/4 width):
  - Key metrics + rolling 3-month performance table
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ── Palantir dark theme (matches report_generator.py) ────────────────────────
BG      = "#0B0E11"
CARD    = "#151A20"
SURFACE = "#1C2127"
BORDER  = "#2F343C"
TEXT    = "#F6F7F9"
TEXT2   = "#ABB3BF"
TEXT3   = "#738091"
ACCENT  = "#2D72D2"    # strategy line
NEPSE_C = "#4A5568"    # NEPSE benchmark
GREEN   = "#238551"
RED     = "#CD4246"
AMBER   = "#D1980B"
GOLD    = "#D1980B"


_QUICK_CHARTS_DIR = Path(__file__).parent.parent / "reports" / "quick_charts"
_MONTH_YEAR_LABEL = "%b '%y"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_nepse_daily(db_path: str, start_date: str, end_date: str) -> list[tuple]:
    """Return [(date_str, close), ...] for the NEPSE index price series."""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.execute(
            "SELECT date, close FROM stock_prices "
            "WHERE symbol='NEPSE' AND date BETWEEN ? AND ? ORDER BY date",
            (start_date, end_date),
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _align_to_dates(
    target_dates: list[str],
    source_rows: list[tuple],
    fallback_end_pct: float = 0.0,
) -> np.ndarray:
    """
    Forward-fill ``source_rows`` ([(date_str, value), ...]) onto ``target_dates``.
    Falls back to a linear series using ``fallback_end_pct`` if no rows available.
    """
    if not source_rows:
        n = len(target_dates)
        return np.linspace(100.0, 100.0 * (1 + fallback_end_pct / 100), n)

    src_map: dict[str, float] = {str(d)[:10]: float(v) for d, v in source_rows}
    last = None
    aligned = []
    for d in target_dates:
        v = src_map.get(d, last)
        aligned.append(v if v is not None else np.nan)
        if v is not None:
            last = v

    arr = np.array(aligned, dtype=float)

    # Fill leading NaN with first valid
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) == 0:
        n = len(target_dates)
        return np.linspace(100.0, 100.0 * (1 + fallback_end_pct / 100), n)
    arr[:valid_idx[0]] = arr[valid_idx[0]]

    # Forward-fill remaining NaN
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]

    return arr / arr[0] * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def generate_quick_chart(
    result: dict,
    *,
    strategy_name: str,
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    auto_open: bool = True,
) -> Optional[str]:
    """
    Generate a strategy-vs-NEPSE chart PNG from a run_strategy_backtest result.

    Parameters
    ----------
    result        : Dict returned by strategy_registry.run_strategy_backtest()
    strategy_name : Human-readable name (title + filename slug)
    start_date    : Backtest start  "YYYY-MM-DD"
    end_date      : Backtest end    "YYYY-MM-DD"
    db_path       : Path to SQLite DB (for live NEPSE daily prices)
    output_dir    : Override default directory (reports/quick_charts/)
    auto_open     : Open PNG in system viewer after saving

    Returns
    -------
    Absolute path to saved PNG, or None if generation failed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker
        import pandas as pd
    except ImportError:
        return None

    # ── Extract strategy daily NAV ────────────────────────────────────────────
    summary   = result.get("summary") or {}
    daily_nav = summary.get("daily_nav") or result.get("_daily_nav") or []
    if len(daily_nav) < 5:
        return None

    raw_dates  = [str(d)[:10] for d, _ in daily_nav]
    raw_vals   = np.array([float(v) for _, v in daily_nav])
    strat_norm = raw_vals / raw_vals[0] * 100.0

    # Convert to pandas DatetimeIndex for date-aware operations
    pd_dates = pd.to_datetime(raw_dates)

    # ── NEPSE daily series ────────────────────────────────────────────────────
    nepse_ret_pct = float((result.get("nepse") or {}).get("return_pct", 0) or 0)
    if db_path:
        nepse_rows = _fetch_nepse_daily(db_path, start_date, end_date)
    else:
        nepse_rows = []
    nepse_norm = _align_to_dates(raw_dates, nepse_rows, fallback_end_pct=nepse_ret_pct)

    # ── Monthly returns (for bar chart) ───────────────────────────────────────
    strat_s  = pd.Series(strat_norm,  index=pd_dates)
    nepse_s  = pd.Series(nepse_norm,  index=pd_dates)

    # Last value of each month → pct change
    strat_monthly = strat_s.resample("ME").last().pct_change().dropna() * 100
    nepse_monthly = nepse_s.resample("ME").last().pct_change().dropna() * 100

    # Align on common months
    common_months = strat_monthly.index.intersection(nepse_monthly.index)
    strat_monthly = strat_monthly.loc[common_months]
    nepse_monthly = nepse_monthly.loc[common_months]

    # ── Drawdown ──────────────────────────────────────────────────────────────
    peak_nav = np.maximum.accumulate(strat_norm)
    drawdown = (strat_norm - peak_nav) / peak_nav * 100.0

    # ── Key metrics ───────────────────────────────────────────────────────────
    strat_ret = float(summary.get("total_return_pct", 0) or 0)
    sharpe    = float(summary.get("sharpe_ratio", summary.get("sharpe")) or 0)
    max_dd    = float(summary.get("max_drawdown_pct", 0) or 0)
    trades    = int(summary.get("trade_count", summary.get("total_trades")) or 0)
    win_rate  = float(summary.get("win_rate_pct", 0) or 0)
    alpha     = strat_ret - nepse_ret_pct

    # ── Output path ───────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else _QUICK_CHARTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = strategy_name.lower().replace(" ", "_")
    slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)[:28]
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"{slug}_{ts}.png"

    # ─────────────────────────────────────────────────────────────────────────
    # Build figure
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=BG)

    # 3 rows × 2 cols
    # Row 0 = equity (large)
    # Row 1 = monthly bar chart
    # Row 2 = drawdown (thin)
    # Col 0 = charts, Col 1 = stats panel
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[4.0, 1.5, 1.0],
        width_ratios=[3.2, 1.0],
        hspace=0.06,
        wspace=0.03,
        left=0.05,
        right=0.975,
        top=0.88,
        bottom=0.06,
    )
    ax_eq    = fig.add_subplot(gs[0, 0])
    ax_month = fig.add_subplot(gs[1, 0])
    ax_dd    = fig.add_subplot(gs[2, 0])
    ax_stat  = fig.add_subplot(gs[:, 1])

    # ── Shared style ─────────────────────────────────────────────────────────
    def _style(ax, show_date_axis: bool = False):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT3, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
            sp.set_linewidth(0.5)
        ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.4)
        if show_date_axis:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter(_MONTH_YEAR_LABEL))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right",
                     fontsize=7)
        else:
            ax.tick_params(labelbottom=False)

    for ax in (ax_eq, ax_month, ax_dd, ax_stat):
        _style(ax)
    _style(ax_dd, show_date_axis=True)   # only bottom chart gets date labels
    ax_stat.axis("off")

    # ── Month boundary vertical lines ─────────────────────────────────────────
    def _add_month_lines(ax):
        month_starts = pd.date_range(start=pd_dates[0], end=pd_dates[-1], freq="MS")
        for ms in month_starts:
            ax.axvline(ms, color=BORDER, linewidth=0.4, linestyle=":", alpha=0.6, zorder=1)

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.05, 0.935, strategy_name,
             fontsize=14, fontweight="bold", color=TEXT,
             fontfamily="monospace", va="top")
    fig.text(0.05, 0.905,
             f"{start_date}  →  {end_date}   ·   {len(raw_dates)} trading days   ·   "
             f"{len(strat_monthly)} months",
             fontsize=8.5, color=TEXT3, va="top")

    # Verdict pill (top right)
    ret_col  = GREEN if strat_ret >= 0 else RED
    alp_sign = "+" if alpha >= 0 else ""
    alp_col  = GREEN if alpha >= 0 else RED
    fig.text(
        0.96, 0.935,
        f"  {strat_ret:+.2f}%   α {alp_sign}{alpha:.1f}pp   Sharpe {sharpe:.2f}  ",
        fontsize=9, color=ret_col, va="top", ha="right",
        fontfamily="monospace",
        bbox=dict(facecolor=SURFACE, edgecolor=BORDER,
                  boxstyle="round,pad=0.3", linewidth=0.8),
    )

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax_eq.fill_between(pd_dates, strat_norm, 100, color=ACCENT, alpha=0.07)
    ax_eq.plot(pd_dates, strat_norm, color=ACCENT, linewidth=1.6,
               label=strategy_name, zorder=4)
    ax_eq.plot(pd_dates, nepse_norm, color=NEPSE_C, linewidth=1.0,
               linestyle="--", alpha=0.8, label="NEPSE Index", zorder=3)
    ax_eq.axhline(100, color=TEXT3, linewidth=0.5, linestyle=":", alpha=0.35)
    _add_month_lines(ax_eq)

    # Shade months where strategy underperforms NEPSE (very subtle)
    for i in range(len(strat_monthly)):
        month_end = strat_monthly.index[i]
        month_start = month_end - pd.offsets.MonthBegin(1)
        if strat_monthly.iloc[i] < 0:
            ax_eq.axvspan(month_start, month_end,
                          color=RED, alpha=0.04, zorder=0)

    # Peak marker
    pk = int(np.argmax(strat_norm))
    ax_eq.plot(pd_dates[pk], strat_norm[pk], marker="^",
               color=GREEN, markersize=7, zorder=6, clip_on=False)

    ax_eq.set_ylabel("NAV  (100 = start)", fontsize=8, color=TEXT2, labelpad=6)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax_eq.tick_params(labelbottom=False)

    leg = ax_eq.legend(loc="upper left", fontsize=7.5, framealpha=0.0,
                       handlelength=1.4, handletextpad=0.5)
    for text, col in zip(leg.get_texts(), [ACCENT, NEPSE_C]):
        text.set_color(col)

    # ── Monthly returns bar chart ─────────────────────────────────────────────
    ax_month.axhline(0, color=TEXT3, linewidth=0.5, alpha=0.6)
    _add_month_lines(ax_month)

    bar_width_days = 18   # slightly less than a month so bars don't overlap

    for dt, ret in strat_monthly.items():
        col   = GREEN if ret >= 0 else RED
        alpha = min(0.85, abs(ret) / 8 + 0.25)   # stronger colour for bigger moves
        ax_month.bar(dt, ret, width=bar_width_days, color=col, alpha=alpha,
                     align="center", zorder=3)
        # Label bars with >2% moves
        if abs(ret) >= 2.0:
            va   = "bottom" if ret >= 0 else "top"
            yoff = 0.15 if ret >= 0 else -0.15
            ax_month.text(dt, ret + yoff, f"{ret:+.1f}%",
                          fontsize=5.5, color=col, ha="center", va=va, zorder=4)

    # NEPSE monthly dots
    for dt, ret in nepse_monthly.items():
        ax_month.plot(dt, ret, marker="o", markersize=4,
                      color=NEPSE_C, alpha=0.7, zorder=5)

    ax_month.set_ylabel("Monthly %", fontsize=7.5, color=TEXT2, labelpad=6)
    ax_month.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax_month.tick_params(labelbottom=False)

    # Small legend for monthly chart
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    month_legend = [
        Patch(facecolor=GREEN, alpha=0.6, label="Strategy +"),
        Patch(facecolor=RED,   alpha=0.6, label="Strategy −"),
        Line2D([0], [0], marker="o", color=NEPSE_C, linestyle="None",
               markersize=4, label="NEPSE"),
    ]
    ax_month.legend(handles=month_legend, loc="upper left", fontsize=6.5,
                    framealpha=0.0, ncol=3, handlelength=1.0)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax_dd.fill_between(pd_dates, drawdown, 0, color=RED, alpha=0.22)
    ax_dd.plot(pd_dates, drawdown, color=RED, linewidth=0.8)
    ax_dd.axhline(0, color=TEXT3, linewidth=0.4, alpha=0.5)
    _add_month_lines(ax_dd)

    mdd_i = int(np.argmin(drawdown))
    if drawdown[mdd_i] < -0.5:
        ax_dd.annotate(
            f"  {drawdown[mdd_i]:.1f}%",
            xy=(pd_dates[mdd_i], drawdown[mdd_i]),
            xytext=(pd_dates[mdd_i], drawdown[mdd_i] * 0.55),
            fontsize=6.5, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.5),
        )

    ax_dd.set_ylabel("Drawdown", fontsize=7.5, color=TEXT2, labelpad=6)
    ax_dd.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Shared date axis on bottom chart
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter(_MONTH_YEAR_LABEL))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Stats panel ───────────────────────────────────────────────────────────
    ax_stat.axis("off")

    def _row(y: float, label: str, value: str, color: str = TEXT2, bold: bool = False):
        ax_stat.text(0.07, y, label, transform=ax_stat.transAxes,
                     fontsize=7.5, color=TEXT3, va="center",
                     fontfamily="monospace")
        ax_stat.text(0.97, y, value, transform=ax_stat.transAxes,
                     fontsize=8.5 if bold else 7.5, color=color,
                     va="center", ha="right",
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace")

    def _sep(y: float):
        ax_stat.plot([0.05, 0.95], [y, y], color=BORDER, linewidth=0.7,
                     transform=ax_stat.transAxes, clip_on=False)

    sha_c = GREEN if sharpe >= 1.0 else (AMBER if sharpe >= 0.5 else RED)
    mdd_c = GREEN if max_dd > -10 else (AMBER if max_dd > -20 else RED)
    wr_c  = GREEN if win_rate >= 55 else (AMBER if win_rate >= 45 else RED)

    ax_stat.text(0.5, 0.975, "PERFORMANCE",
                 transform=ax_stat.transAxes, fontsize=7, color=TEXT3,
                 ha="center", va="top", fontfamily="monospace", fontweight="bold")
    _sep(0.956)
    _row(0.920, "RETURN",   f"{strat_ret:+.2f}%",  GREEN if strat_ret >= 0 else RED, True)
    _row(0.873, "vs NEPSE", f"{nepse_ret_pct:+.2f}%", TEXT3)
    _row(0.826, "ALPHA",    f"{'+' if alpha>=0 else ''}{alpha:.2f}pp",
         GREEN if alpha >= 0 else RED, True)
    _sep(0.793)
    _row(0.756, "SHARPE",   f"{sharpe:.3f}",        sha_c,  True)
    _row(0.709, "MAX DD",   f"{max_dd:.1f}%",       mdd_c)
    _sep(0.676)
    _row(0.639, "TRADES",   f"{trades}",             TEXT2)
    _row(0.592, "WIN RATE", f"{win_rate:.1f}%",     wr_c)
    _sep(0.559)
    _row(0.522, "MONTHS",   f"{len(strat_monthly)}", TEXT3)

    # Monthly heat table (last 6 months max)
    n_show = min(len(strat_monthly), 8)
    if n_show:
        ax_stat.text(0.5, 0.500, "MONTHLY",
                     transform=ax_stat.transAxes, fontsize=7, color=TEXT3,
                     ha="center", va="top", fontfamily="monospace", fontweight="bold")
        _sep(0.483)
        row_y   = 0.460
        row_gap = 0.048
        recent  = strat_monthly.iloc[-n_show:]
        nepse_r = nepse_monthly.reindex(recent.index)
        for dt, ret in recent.items():
            nr   = nepse_r.get(dt, float("nan"))
            col  = GREEN if ret >= 0 else RED
            ncol = GREEN if (not np.isnan(nr) and nr >= 0) else RED
            nr_s = f"{nr:+.1f}%" if not np.isnan(nr) else "  —  "
            ax_stat.text(0.07, row_y, dt.strftime("%b '%y"),
                         transform=ax_stat.transAxes,
                         fontsize=6.5, color=TEXT3, va="center",
                         fontfamily="monospace")
            ax_stat.text(0.60, row_y, f"{ret:+.1f}%",
                         transform=ax_stat.transAxes,
                         fontsize=6.5, color=col, va="center", ha="right",
                         fontfamily="monospace")
            ax_stat.text(0.97, row_y, nr_s,
                         transform=ax_stat.transAxes,
                         fontsize=6.5, color=ncol, va="center", ha="right",
                         fontfamily="monospace")
            row_y -= row_gap

    # Column headers for monthly table
    ax_stat.text(0.60, 0.500 - 0.01, "STRAT",
                 transform=ax_stat.transAxes, fontsize=5.5, color=TEXT3,
                 ha="right", va="center", fontfamily="monospace")
    ax_stat.text(0.97, 0.500 - 0.01, "NEPSE",
                 transform=ax_stat.transAxes, fontsize=5.5, color=TEXT3,
                 ha="right", va="center", fontfamily="monospace")

    # Watermark
    fig.text(0.975, 0.012, "NEPSE QUANT  ·  NOT FINANCIAL ADVICE",
             fontsize=5.5, color=BORDER, ha="right", va="bottom",
             fontfamily="monospace")

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)

    if auto_open:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(out_path)])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(out_path)])
        except Exception:
            pass

    return str(out_path)
