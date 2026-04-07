#!/usr/bin/env python3
"""Generate cleaner postable charts for the latest backtest run."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import get_project_root


ROOT = get_project_root(__file__)
NAV_CSV = ROOT / "backtest_nav.csv"
TRADES_CSV = ROOT / "backtest_trades.csv"
OUTPUT_DIR = ROOT / "reports" / "postable_upgraded_20260330"
RECENT_START = pd.Timestamp("2025-01-01")
RECENT_END = pd.Timestamp("2025-12-31")

BG = "#0B1020"
PANEL = "#121A2B"
GRID = "#2A3548"
TEXT = "#E7EDF7"
MUTED = "#97A6BC"
STRATEGY = "#56D39B"
BENCHMARK = "#D9A441"
NEGATIVE = "#D96B6B"
POSITIVE = "#37C58A"
ACCENT = "#7BA7FF"
CYAN = "#3BC8E8"
CARD = "#0F172A"
CARD_BORDER = "#334155"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.titlecolor": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.18,
        "font.family": ["Avenir Next", "Helvetica Neue", "Avenir", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.labelsize": 10.5,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "savefig.facecolor": BG,
    })


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="-", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)


def style_panel(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)


def add_metric_card(fig: plt.Figure, x: float, y: float, w: float, h: float, label: str, value: str, accent: str) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.72, label, color=MUTED, fontsize=9.5, ha="left", va="center", transform=ax.transAxes)
    ax.text(0.06, 0.30, value, color=accent, fontsize=17, fontweight="bold", ha="left", va="center", transform=ax.transAxes)


def add_comparison_table(ax: plt.Axes, rows: list[tuple[str, str, str]]) -> None:
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(CARD_BORDER)
        spine.set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.07, 0.90, "Metric Snapshot", fontsize=13, fontweight="bold", color=TEXT, ha="left", transform=ax.transAxes)
    ax.text(0.58, 0.80, "Strategy", fontsize=9.5, color=STRATEGY, fontweight="bold", ha="right", transform=ax.transAxes)
    ax.text(0.90, 0.80, "NEPSE", fontsize=9.5, color=BENCHMARK, fontweight="bold", ha="right", transform=ax.transAxes)
    y = 0.68
    for label, strategy_value, benchmark_value in rows:
        ax.text(0.07, y, label, fontsize=9.5, color=MUTED, ha="left", transform=ax.transAxes)
        ax.text(0.58, y, strategy_value, fontsize=10.5, color=TEXT, ha="right", transform=ax.transAxes)
        ax.text(0.90, y, benchmark_value, fontsize=10.5, color=TEXT, ha="right", transform=ax.transAxes)
        ax.axhline(y - 0.07, xmin=0.07, xmax=0.93, color=GRID, linewidth=0.8)
        y -= 0.14


def load_nav_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    nav = pd.read_csv(NAV_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    trades = pd.read_csv(TRADES_CSV, parse_dates=["signal_date", "entry_date", "exit_date"])
    nav["returns"] = nav["nav"].pct_change()
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    trades["year"] = trades["entry_date"].dt.year
    return nav, trades


def load_nepse_index(nav: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(str(get_db_path()))
    idx = pd.read_sql(
        "SELECT date, close FROM stock_prices WHERE symbol = 'NEPSE' ORDER BY date",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    if idx.empty:
        raise RuntimeError("NEPSE index series not found in stock_prices")

    idx = idx.sort_values("date").set_index("date")
    idx = idx.loc[: nav["date"].max()]
    aligned = idx.reindex(nav["date"]).ffill().reset_index()
    aligned.columns = ["date", "close"]
    aligned = aligned.dropna().reset_index(drop=True)

    merged = nav[["date", "nav"]].merge(aligned, on="date", how="inner")
    merged["benchmark_nav"] = merged["close"] / merged["close"].iloc[0] * merged["nav"].iloc[0]
    merged["benchmark_drawdown"] = merged["benchmark_nav"] / merged["benchmark_nav"].cummax() - 1.0
    merged["benchmark_returns"] = merged["benchmark_nav"].pct_change()
    return merged


def compute_summary(nav: pd.DataFrame, merged: pd.DataFrame) -> dict[str, float]:
    years = (nav["date"].iloc[-1] - nav["date"].iloc[0]).days / 365.25
    strat_total = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0
    bench_total = merged["benchmark_nav"].iloc[-1] / merged["benchmark_nav"].iloc[0] - 1.0
    strat_cagr = (1.0 + strat_total) ** (1.0 / years) - 1.0
    bench_cagr = (1.0 + bench_total) ** (1.0 / years) - 1.0
    strat_vol = nav["returns"].dropna().std(ddof=1) * np.sqrt(234)
    bench_vol = merged["benchmark_returns"].dropna().std(ddof=1) * np.sqrt(234)
    strat_sharpe = nav["returns"].dropna().mean() / nav["returns"].dropna().std(ddof=1) * np.sqrt(234)
    bench_sharpe = merged["benchmark_returns"].dropna().mean() / merged["benchmark_returns"].dropna().std(ddof=1) * np.sqrt(234)
    return {
        "strategy_total": float(strat_total),
        "benchmark_total": float(bench_total),
        "strategy_cagr": float(strat_cagr),
        "benchmark_cagr": float(bench_cagr),
        "strategy_vol": float(strat_vol),
        "benchmark_vol": float(bench_vol),
        "strategy_sharpe": float(strat_sharpe),
        "benchmark_sharpe": float(bench_sharpe),
        "strategy_dd": float(nav["drawdown"].min()),
        "benchmark_dd": float(merged["benchmark_drawdown"].min()),
    }


def filter_window(
    nav: pd.DataFrame,
    merged: pd.DataFrame,
    trades: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nav_view = nav[(nav["date"] >= start_date) & (nav["date"] <= end_date)].copy()
    merged_view = merged[(merged["date"] >= start_date) & (merged["date"] <= end_date)].copy()
    trades_view = trades[(trades["entry_date"] >= start_date) & (trades["entry_date"] <= end_date)].copy()

    nav_view["returns"] = nav_view["nav"].pct_change()
    nav_view["drawdown"] = nav_view["nav"] / nav_view["nav"].cummax() - 1.0
    merged_view["benchmark_returns"] = merged_view["benchmark_nav"].pct_change()
    merged_view["benchmark_drawdown"] = merged_view["benchmark_nav"] / merged_view["benchmark_nav"].cummax() - 1.0
    return nav_view.reset_index(drop=True), merged_view.reset_index(drop=True), trades_view.reset_index(drop=True)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.965, title, fontsize=22, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.06, 0.932, subtitle, fontsize=11, color=MUTED, ha="left", va="top")


def add_stat_row(fig: plt.Figure, stats: list[tuple[str, str, str]], *, y: float = 0.845) -> None:
    ax = fig.add_axes([0.07, y, 0.86, 0.055])
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    slot_width = 1.0 / max(len(stats), 1)
    for idx, (label, value, color) in enumerate(stats):
        x0 = idx * slot_width
        inset = 0.022
        if idx > 0:
            ax.axvline(x0 - 0.012, ymin=0.18, ymax=0.88, color=GRID, linewidth=0.8, alpha=0.55)
        ax.text(x0 + inset, 0.72, label, fontsize=9.2, color=MUTED, ha="left", va="center", transform=ax.transAxes)
        ax.text(x0 + inset, 0.16, value, fontsize=13.8, color=color, fontweight="bold", ha="left", va="bottom", transform=ax.transAxes)


def add_line_end_label(
    ax: plt.Axes,
    x: pd.Timestamp,
    y: float,
    text: str,
    color: str,
    *,
    dx_days: int = 14,
    text_pad_days: int = 2,
) -> None:
    line_end = x + pd.Timedelta(days=dx_days)
    text_x = line_end + pd.Timedelta(days=text_pad_days)
    ax.plot([x, line_end], [y, y], color=color, linewidth=1.0, alpha=0.9, clip_on=False)
    ax.text(
        text_x,
        y,
        text,
        color=color,
        fontsize=10.2,
        fontweight="bold",
        va="center",
        ha="left",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_strategy_vs_nepse_sleek(
    nav: pd.DataFrame,
    merged: pd.DataFrame,
    summary: dict[str, float],
    *,
    title: str = "Strategy vs NEPSE",
    subtitle: str | None = None,
    date_line: str | None = None,
    stats_line: str | None = None,
    stat_pairs: list[tuple[str, str, str]] | None = None,
    output_name: str = "sleek_01_strategy_vs_nepse.png",
) -> None:
    fig = plt.figure(figsize=(14.4, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.95, 1.10], hspace=0.075, top=0.825, bottom=0.08, left=0.07, right=0.965)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    strategy_index = nav["nav"] / nav["nav"].iloc[0] * 100.0
    benchmark_index = merged["benchmark_nav"] / merged["benchmark_nav"].iloc[0] * 100.0

    if subtitle is None:
        subtitle = (
            f"2020-2025  |  +{summary['strategy_total']*100:.1f}% vs +{summary['benchmark_total']*100:.1f}%"
            f"  |  Sharpe {summary['strategy_sharpe']:.2f}  |  Max DD {summary['strategy_dd']*100:.1f}%"
        )

    fig.text(0.07, 0.952, title, fontsize=22, fontweight="bold", color=TEXT, ha="left", va="top")
    if date_line or stats_line:
        if date_line:
            fig.text(0.07, 0.915, date_line, fontsize=10.5, color=MUTED, ha="left", va="top")
        if stat_pairs:
            add_stat_row(fig, stat_pairs, y=0.848)
        elif stats_line:
            fig.text(0.07, 0.885, stats_line, fontsize=11.0, color=TEXT, ha="left", va="top")
    else:
        fig.text(
            0.07,
            0.915,
            subtitle,
            fontsize=10,
            color=MUTED,
            ha="left",
            va="top",
        )

    style_axis(ax1)
    ax1.plot(nav["date"], strategy_index, color=STRATEGY, linewidth=2.2, solid_capstyle="round", label="Strategy")
    ax1.plot(merged["date"], benchmark_index, color=BENCHMARK, linewidth=1.55, linestyle=(0, (3, 2.6)), solid_capstyle="round", label="NEPSE")
    ax1.fill_between(nav["date"], strategy_index, 100, color=STRATEGY, alpha=0.035)
    ax1.axhline(100, color=GRID, linewidth=0.8)
    ax1.set_ylabel("")
    ax1.set_title("")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    days_span = (nav["date"].iloc[-1] - nav["date"].iloc[0]).days
    if days_span <= 420:
        locator = mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11])
        formatter = mdates.DateFormatter("%b")
    else:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.text(0.0, 1.015, "Indexed to 100", transform=ax1.transAxes, color=MUTED, fontsize=9.2, ha="left", va="bottom")

    end_dates = [nav["date"].iloc[-1], merged["date"].iloc[-1]]
    end_vals = [strategy_index.iloc[-1], benchmark_index.iloc[-1]]
    ax1.scatter(end_dates, end_vals, c=[STRATEGY, BENCHMARK], s=22, zorder=5)
    x_pad_days = 24 if days_span <= 420 else 28
    ax1.set_xlim(nav["date"].iloc[0] - pd.Timedelta(days=4), nav["date"].iloc[-1] + pd.Timedelta(days=x_pad_days))
    if stat_pairs:
        add_line_end_label(ax1, nav["date"].iloc[-1], float(strategy_index.iloc[-1]), f"Strategy  {strategy_index.iloc[-1]:.1f}", STRATEGY)
        add_line_end_label(ax1, merged["date"].iloc[-1], float(benchmark_index.iloc[-1]), f"NEPSE  {benchmark_index.iloc[-1]:.1f}", BENCHMARK)
    else:
        ax1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.995), frameon=False, fontsize=9.5, ncol=2, handlelength=2.2, borderaxespad=0.0)

    style_axis(ax2)
    strategy_dd = nav["drawdown"] * 100
    benchmark_dd = merged["benchmark_drawdown"] * 100
    ax2.plot(nav["date"], strategy_dd, color=STRATEGY, linewidth=1.35)
    ax2.plot(merged["date"], benchmark_dd, color=BENCHMARK, linewidth=1.15, linestyle=(0, (3, 2.4)), alpha=0.95)
    ax2.fill_between(nav["date"], strategy_dd, 0, color=NEGATIVE, alpha=0.055)
    ax2.axhline(0, color=GRID, linewidth=0.8)
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_title("")
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    dd_floor = min(-10.0, float(np.floor(min(strategy_dd.min(), benchmark_dd.min()) / 5.0) * 5.0))
    ax2.set_ylim(dd_floor, 1.5)
    ax2.set_yticks([dd_floor, dd_floor / 2.0, 0])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.text(0.0, 0.88, "Drawdown", transform=ax2.transAxes, color=MUTED, fontsize=9.2, ha="left", va="top")

    save_figure(fig, OUTPUT_DIR / output_name)


def chart_strategy_vs_nepse(nav: pd.DataFrame, merged: pd.DataFrame, summary: dict[str, float]) -> None:
    fig = plt.figure(figsize=(14, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.25], width_ratios=[2.6, 1.15], hspace=0.18, wspace=0.16, top=0.82, bottom=0.08, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[1, 1])

    add_header(
        fig,
        "Strategy vs NEPSE",
        "Upgraded walk-forward backtest | 2020-01-01 to 2025-12-31 | Benchmark rebased to the same starting capital",
    )
    add_metric_card(fig, 0.06, 0.845, 0.18, 0.065, "Strategy total return", f"{summary['strategy_total']*100:.1f}%", STRATEGY)
    add_metric_card(fig, 0.255, 0.845, 0.18, 0.065, "NEPSE total return", f"{summary['benchmark_total']*100:.1f}%", BENCHMARK)
    add_metric_card(fig, 0.45, 0.845, 0.18, 0.065, "Excess return", f"{(summary['strategy_total']-summary['benchmark_total'])*100:+.1f}%", ACCENT)
    add_metric_card(fig, 0.645, 0.845, 0.18, 0.065, "Strategy Sharpe", f"{summary['strategy_sharpe']:.2f}", CYAN)

    strategy_index = nav["nav"] / nav["nav"].iloc[0] * 100.0
    benchmark_index = merged["benchmark_nav"] / merged["benchmark_nav"].iloc[0] * 100.0

    style_axis(ax1)
    ax1.plot(nav["date"], strategy_index, color=STRATEGY, linewidth=3.0)
    ax1.plot(merged["date"], benchmark_index, color=BENCHMARK, linewidth=2.2, linestyle=(0, (4, 2)))
    ax1.fill_between(nav["date"], strategy_index, benchmark_index, where=(strategy_index >= benchmark_index), color=STRATEGY, alpha=0.08)
    ax1.fill_between(nav["date"], strategy_index, benchmark_index, where=(strategy_index < benchmark_index), color=BENCHMARK, alpha=0.05)
    ax1.set_title("Indexed growth of NPR 100", loc="left", fontsize=15, pad=12)
    ax1.set_ylabel("Indexed growth")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.annotate(
        f"Strategy  {strategy_index.iloc[-1]:.0f}",
        xy=(nav["date"].iloc[-1], strategy_index.iloc[-1]),
        xytext=(-12, 12),
        textcoords="offset points",
        color=STRATEGY,
        fontsize=11,
        fontweight="bold",
        ha="right",
    )
    ax1.annotate(
        f"NEPSE  {benchmark_index.iloc[-1]:.0f}",
        xy=(merged["date"].iloc[-1], benchmark_index.iloc[-1]),
        xytext=(-12, -18),
        textcoords="offset points",
        color=BENCHMARK,
        fontsize=11,
        fontweight="bold",
        ha="right",
    )
    ax1.text(0.015, 0.965, "Strategy line stays above NEPSE for most of 2024-2025", transform=ax1.transAxes, color=MUTED, fontsize=9.5, va="top")

    style_axis(ax2)
    ax2.plot(nav["date"], nav["drawdown"] * 100, color=STRATEGY, linewidth=2.0)
    ax2.plot(merged["date"], merged["benchmark_drawdown"] * 100, color=BENCHMARK, linewidth=1.7, linestyle=(0, (4, 2)))
    ax2.fill_between(nav["date"], nav["drawdown"] * 100, 0, color=NEGATIVE, alpha=0.12)
    ax2.axhline(0, color=GRID, linewidth=1.0)
    ax2.set_title("Drawdown comparison", loc="left", fontsize=14, pad=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(["Strategy", "NEPSE"], loc="lower left", ncol=2, fontsize=9, frameon=False)

    comparison_rows = [
        ("Total return", f"{summary['strategy_total']*100:.1f}%", f"{summary['benchmark_total']*100:.1f}%"),
        ("CAGR", f"{summary['strategy_cagr']*100:.1f}%", f"{summary['benchmark_cagr']*100:.1f}%"),
        ("Annualized vol", f"{summary['strategy_vol']*100:.1f}%", f"{summary['benchmark_vol']*100:.1f}%"),
        ("Sharpe ratio", f"{summary['strategy_sharpe']:.2f}", f"{summary['benchmark_sharpe']:.2f}"),
        ("Max drawdown", f"{summary['strategy_dd']*100:.1f}%", f"{summary['benchmark_dd']*100:.1f}%"),
    ]
    add_comparison_table(ax3, comparison_rows)

    save_figure(fig, OUTPUT_DIR / "clean_01_strategy_vs_nepse.png")


def chart_returns_profile(nav: pd.DataFrame, merged: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.28, top=0.84, bottom=0.08, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    add_header(
        fig,
        "Return Profile",
        "Monthly strategy returns with annual comparison against NEPSE",
    )

    nav_monthly = nav.set_index("date")["nav"].resample("ME").last()
    monthly_ret = nav_monthly.pct_change(fill_method=None).dropna()
    monthly_df = pd.DataFrame({"year": monthly_ret.index.year, "month": monthly_ret.index.month, "return": monthly_ret.values})
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = month_labels

    style_axis(ax1)
    cmap = LinearSegmentedColormap.from_list("soft_rg", ["#7F1D1D", "#F8FAFC", "#14532D"])
    heat = ax1.imshow(pivot.values * 100, aspect="auto", cmap=cmap, vmin=-12, vmax=12)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(month_labels)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([str(int(y)) for y in pivot.index])
    ax1.set_title("Monthly returns (%)", loc="left", fontsize=15, pad=12)

    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = BG if abs(val) >= 0.05 else TEXT
                ax1.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(heat, ax=ax1, fraction=0.022, pad=0.02)
    cbar.set_label("Return (%)", color=MUTED)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED)

    strategy_annual = nav.set_index("date")["nav"].resample("YE").last().pct_change().dropna()
    benchmark_annual = merged.set_index("date")["benchmark_nav"].resample("YE").last().pct_change().dropna()
    years = sorted(set(strategy_annual.index.year).union(set(benchmark_annual.index.year)))
    strat_vals = [strategy_annual[strategy_annual.index.year == year].iloc[0] * 100 if year in strategy_annual.index.year else np.nan for year in years]
    bench_vals = [benchmark_annual[benchmark_annual.index.year == year].iloc[0] * 100 if year in benchmark_annual.index.year else np.nan for year in years]

    style_axis(ax2)
    x = np.arange(len(years))
    width = 0.34
    bars1 = ax2.bar(x - width / 2, strat_vals, width, color=STRATEGY, alpha=0.9, label="Strategy")
    bars2 = ax2.bar(x + width / 2, bench_vals, width, color=BENCHMARK, alpha=0.9, label="NEPSE")
    ax2.axhline(0, color=GRID, linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("Annual return (%)")
    ax2.set_title("Annual returns: strategy vs NEPSE", loc="left", fontsize=15, pad=12)
    ax2.legend(loc="upper left", ncol=2, fontsize=10)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.8 if height >= 0 else -1.4),
                f"{height:.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color=TEXT,
            )

    save_figure(fig, OUTPUT_DIR / "clean_02_returns_profile.png")


def chart_returns_profile_sleek(
    nav: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    title: str = "Return Profile",
    subtitle: str = "Monthly heatmap and annual comparison",
    output_name: str = "sleek_02_returns_profile.png",
) -> None:
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.0], hspace=0.24, top=0.865, bottom=0.09, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.text(0.07, 0.935, title, fontsize=21, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.902, subtitle, fontsize=10, color=MUTED, ha="left", va="top")

    nav_monthly = nav.set_index("date")["nav"].resample("ME").last()
    monthly_ret = nav_monthly.pct_change(fill_method=None).dropna()
    monthly_df = pd.DataFrame({"year": monthly_ret.index.year, "month": monthly_ret.index.month, "return": monthly_ret.values})
    pivot = monthly_df.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = month_labels

    style_panel(ax1)
    heat_values = pivot.values * 100.0
    vmax = max(8.0, float(np.nanpercentile(np.abs(heat_values), 92)))
    cmap = LinearSegmentedColormap.from_list("research_div", ["#8B3A3A", "#F2F4F7", "#2D6A4F"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    heat = ax1.imshow(np.ma.masked_invalid(heat_values), aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(month_labels)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([str(int(y)) for y in pivot.index])
    ax1.tick_params(length=0)
    ax1.set_xticks(np.arange(-0.5, len(month_labels), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax1.grid(False)
    ax1.grid(which="minor", color="#314056", linewidth=0.75)
    ax1.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if pd.notna(val) and abs(val) >= 0.018:
                color = TEXT if abs(val * 100.0) >= vmax * 0.62 else "#0F172A"
                ax1.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=8.4, color=color)

    cbar = fig.colorbar(heat, ax=ax1, fraction=0.018, pad=0.02)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=MUTED)
    cbar.set_label("%", color=MUTED, rotation=0, labelpad=9)

    strategy_annual = nav.set_index("date")["nav"].resample("YE").last().pct_change().dropna()
    benchmark_annual = merged.set_index("date")["benchmark_nav"].resample("YE").last().pct_change().dropna()
    years = sorted(set(strategy_annual.index.year).union(set(benchmark_annual.index.year)))
    strat_vals = [strategy_annual[strategy_annual.index.year == year].iloc[0] * 100 if year in strategy_annual.index.year else np.nan for year in years]
    bench_vals = [benchmark_annual[benchmark_annual.index.year == year].iloc[0] * 100 if year in benchmark_annual.index.year else np.nan for year in years]

    style_axis(ax2)
    if not years or (np.isnan(strat_vals).all() and np.isnan(bench_vals).all()):
        years = [int(nav["date"].dt.year.mode().iloc[0])]
        strat_vals = [(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0) * 100.0]
        bench_vals = [(merged["benchmark_nav"].iloc[-1] / merged["benchmark_nav"].iloc[0] - 1.0) * 100.0]

    x = np.arange(len(years))
    width = 0.28
    bars1 = ax2.bar(x - width / 2, strat_vals, width, color=STRATEGY, alpha=0.88, label="Strategy")
    bars2 = ax2.bar(x + width / 2, bench_vals, width, color=BENCHMARK, alpha=0.82, label="NEPSE")
    ax2.axhline(0, color=GRID, linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=2, fontsize=9.5, handlelength=1.4, borderaxespad=0.0)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height) or abs(height) < 4.0:
                continue
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.9 if height >= 0 else -1.1),
                f"{height:.0f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8.2,
                color=MUTED,
            )

    save_figure(fig, OUTPUT_DIR / output_name)


def chart_trade_profile(trades: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22, top=0.84, bottom=0.10, left=0.06, right=0.97)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    add_header(
        fig,
        "Trade Quality",
        "Cleaner breakdown of trade outcomes by size, signal, and contribution",
    )

    style_axis(ax1)
    pnl_k = trades["net_pnl"] / 1000.0
    ax1.hist(pnl_k[pnl_k > 0], bins=18, color=POSITIVE, alpha=0.75, label="Wins")
    ax1.hist(pnl_k[pnl_k <= 0], bins=16, color=NEGATIVE, alpha=0.75, label="Losses")
    ax1.axvline(0, color=GRID, linewidth=1.0)
    ax1.set_title("Trade P&L distribution", loc="left", fontsize=15, pad=12)
    ax1.set_xlabel("P&L (NPR '000)")
    ax1.set_ylabel("Trades")
    ax1.legend(loc="upper right", fontsize=10)

    sig = (
        trades.groupby("signal_type")
        .agg(trades=("net_pnl", "count"), wins=("net_pnl", lambda x: (x > 0).sum()), avg_return=("net_return", "mean"), total_pnl=("net_pnl", "sum"))
        .sort_values("total_pnl", ascending=True)
    )
    sig["win_rate"] = sig["wins"] / sig["trades"] * 100.0
    labels = [label.replace("_", " ") for label in sig.index]

    style_axis(ax2)
    bars = ax2.barh(labels, sig["win_rate"], color=ACCENT, alpha=0.85)
    ax2.axvline(50, color=BENCHMARK, linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_title("Win rate by signal", loc="left", fontsize=15, pad=12)
    ax2.set_xlabel("Win rate (%)")
    for bar, wr in zip(bars, sig["win_rate"]):
        ax2.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2, f"{wr:.0f}%", va="center", fontsize=9)

    style_axis(ax3)
    pnl_totals = sig["total_pnl"] / 1000.0
    colors = [POSITIVE if val >= 0 else NEGATIVE for val in pnl_totals]
    bars = ax3.barh(labels, pnl_totals, color=colors, alpha=0.88)
    ax3.axvline(0, color=GRID, linewidth=1.0)
    ax3.set_title("Total P&L by signal", loc="left", fontsize=15, pad=12)
    ax3.set_xlabel("P&L (NPR '000)")
    for bar, val in zip(bars, pnl_totals):
        x = bar.get_width()
        ax3.text(x + (6 if val >= 0 else -6), bar.get_y() + bar.get_height() / 2, f"{val:.0f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    save_figure(fig, OUTPUT_DIR / "clean_03_trade_profile.png")


def chart_trade_profile_sleek(
    trades: pd.DataFrame,
    *,
    title: str = "Trade Profile",
    subtitle: str = "P&L distribution and signal contribution",
    output_name: str = "sleek_03_trade_profile.png",
) -> None:
    fig = plt.figure(figsize=(14, 7.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.95], wspace=0.18, top=0.865, bottom=0.10, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.text(0.07, 0.935, title, fontsize=21, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.902, subtitle, fontsize=10, color=MUTED, ha="left", va="top")

    style_axis(ax1)
    pnl_k = trades["net_pnl"] / 1000.0
    ax1.hist(pnl_k[pnl_k > 0], bins=18, color=POSITIVE, alpha=0.18, histtype="stepfilled", linewidth=0)
    ax1.hist(pnl_k[pnl_k > 0], bins=18, color=POSITIVE, histtype="step", linewidth=1.4)
    ax1.hist(pnl_k[pnl_k <= 0], bins=16, color=NEGATIVE, alpha=0.18, histtype="stepfilled", linewidth=0)
    ax1.hist(pnl_k[pnl_k <= 0], bins=16, color=NEGATIVE, histtype="step", linewidth=1.4)
    ax1.axvline(0, color=GRID, linewidth=0.9)
    ax1.set_xlabel("P&L (NPR '000)")
    ax1.set_ylabel("Trades")
    ax1.set_title("")

    sig = (
        trades.groupby("signal_type")
        .agg(total_pnl=("net_pnl", "sum"))
        .sort_values("total_pnl", ascending=True)
    )
    labels = [label.replace("_", " ") for label in sig.index]
    vals = sig["total_pnl"] / 1000.0

    style_axis(ax2)
    ax2.grid(True, axis="x", linestyle="-", linewidth=0.6)
    ax2.grid(False, axis="y")
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in vals]
    y = np.arange(len(labels))
    ax2.hlines(y, 0, vals, color=colors, linewidth=2.0, alpha=0.85)
    ax2.scatter(vals, y, color=colors, s=70, zorder=3)
    ax2.axvline(0, color=GRID, linewidth=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("P&L (NPR '000)")
    ax2.set_title("")
    for yi, val in zip(y, vals):
        if val >= 0:
            ax2.text(val + 8, yi, f"{val:.0f}", ha="left", va="center", fontsize=9, color=MUTED)
        else:
            ax2.annotate(
                f"{val:.0f}",
                xy=(val, yi),
                xytext=(8, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9,
                color=MUTED,
            )

    save_figure(fig, OUTPUT_DIR / output_name)


def chart_risk_profile(nav: pd.DataFrame, merged: pd.DataFrame, trades: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22, top=0.84, bottom=0.10, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    add_header(
        fig,
        "Risk Lens",
        "Rolling risk, excess performance, and holding-period efficiency",
    )

    returns = nav.set_index("date")["returns"].dropna()
    benchmark_returns = merged.set_index("date")["benchmark_returns"].dropna()
    rolling_vol = returns.rolling(63).std() * np.sqrt(234) * 100.0
    rolling_vol_bench = benchmark_returns.rolling(63).std() * np.sqrt(234) * 100.0

    style_axis(ax1)
    ax1.plot(rolling_vol.index, rolling_vol, color=STRATEGY, linewidth=2.4, label="Strategy")
    ax1.plot(rolling_vol_bench.index, rolling_vol_bench, color=BENCHMARK, linewidth=2.0, linestyle="--", label="NEPSE")
    ax1.set_title("63-day annualized volatility", loc="left", fontsize=15, pad=12)
    ax1.set_ylabel("Volatility (%)")
    ax1.legend(loc="upper right", ncol=2, fontsize=10)

    strategy_index = nav.set_index("date")["nav"] / nav["nav"].iloc[0] * 100.0
    benchmark_index = merged.set_index("date")["benchmark_nav"] / merged["benchmark_nav"].iloc[0] * 100.0
    excess = strategy_index.reindex(benchmark_index.index) - benchmark_index

    style_axis(ax2)
    ax2.plot(excess.index, excess.values, color=CYAN, linewidth=2.2)
    ax2.fill_between(excess.index, excess.values, 0, where=(excess.values >= 0), color=CYAN, alpha=0.12)
    ax2.axhline(0, color=GRID, linewidth=1.0)
    ax2.set_title("Cumulative excess return vs NEPSE", loc="left", fontsize=15, pad=12)
    ax2.set_ylabel("Indexed points")

    style_axis(ax3)
    colors = np.where(trades["net_return"] > 0, POSITIVE, NEGATIVE)
    ax3.scatter(trades["holding_days"], trades["net_return"] * 100.0, c=colors, alpha=0.6, s=42, edgecolors="none")
    slope, intercept = np.polyfit(trades["holding_days"], trades["net_return"] * 100.0, 1)
    x_line = np.array([trades["holding_days"].min(), trades["holding_days"].max()])
    ax3.plot(x_line, slope * x_line + intercept, color=ACCENT, linewidth=1.8, linestyle="--")
    ax3.axhline(0, color=GRID, linewidth=1.0)
    ax3.axvline(40, color=BENCHMARK, linewidth=1.2, linestyle="--", alpha=0.8)
    ax3.set_title("Holding period vs return", loc="left", fontsize=15, pad=12)
    ax3.set_xlabel("Holding days")
    ax3.set_ylabel("Net return (%)")

    save_figure(fig, OUTPUT_DIR / "clean_04_risk_profile.png")


def chart_risk_profile_sleek(
    nav: pd.DataFrame,
    merged: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    title: str = "Risk Profile",
    subtitle: str = "Rolling volatility and return efficiency",
    output_name: str = "sleek_04_risk_profile.png",
    visible_start: pd.Timestamp | None = None,
    visible_end: pd.Timestamp | None = None,
) -> None:
    fig = plt.figure(figsize=(14, 7.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.92], wspace=0.20, top=0.865, bottom=0.10, left=0.07, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.text(0.07, 0.935, title, fontsize=21, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.902, subtitle, fontsize=10, color=MUTED, ha="left", va="top")

    returns = nav.set_index("date")["returns"].dropna()
    benchmark_returns = merged.set_index("date")["benchmark_returns"].dropna()
    rolling_vol = returns.rolling(63).std() * np.sqrt(234) * 100.0
    rolling_vol_bench = benchmark_returns.rolling(63).std() * np.sqrt(234) * 100.0
    if visible_start is not None:
        rolling_vol = rolling_vol[rolling_vol.index >= visible_start]
        rolling_vol_bench = rolling_vol_bench[rolling_vol_bench.index >= visible_start]
    if visible_end is not None:
        rolling_vol = rolling_vol[rolling_vol.index <= visible_end]
        rolling_vol_bench = rolling_vol_bench[rolling_vol_bench.index <= visible_end]

    style_axis(ax1)
    ax1.plot(rolling_vol.index, rolling_vol, color=STRATEGY, linewidth=1.85, label="Strategy")
    ax1.plot(rolling_vol_bench.index, rolling_vol_bench, color=BENCHMARK, linewidth=1.5, linestyle=(0, (3, 2.4)), label="NEPSE")
    ax1.set_ylabel("Volatility (%)")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", frameon=False, ncol=2, fontsize=9.5, handlelength=2.2)

    style_axis(ax2)
    colors = np.where(trades["net_return"] > 0, POSITIVE, NEGATIVE)
    ax2.scatter(trades["holding_days"], trades["net_return"] * 100.0, c=colors, alpha=0.42, s=28, edgecolors="none")
    slope, intercept = np.polyfit(trades["holding_days"], trades["net_return"] * 100.0, 1)
    x_line = np.array([trades["holding_days"].min(), trades["holding_days"].max()])
    ax2.plot(x_line, slope * x_line + intercept, color=ACCENT, linewidth=1.35, linestyle="--")
    ax2.axhline(0, color=GRID, linewidth=0.9)
    ax2.axvline(40, color=BENCHMARK, linewidth=0.95, linestyle=(0, (3, 2.4)), alpha=0.75)
    ax2.set_xlabel("Holding days")
    ax2.set_ylabel("Net return (%)")
    ax2.set_title("")

    save_figure(fig, OUTPUT_DIR / output_name)


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nav, trades = load_nav_trades()
    merged = load_nepse_index(nav)
    summary = compute_summary(nav, merged)
    nav_recent, merged_recent, trades_recent = filter_window(nav, merged, trades, RECENT_START, RECENT_END)
    recent_summary = compute_summary(nav_recent, merged_recent)

    chart_strategy_vs_nepse(nav, merged, summary)
    chart_returns_profile(nav, merged)
    chart_trade_profile(trades)
    chart_risk_profile(nav, merged, trades)
    chart_strategy_vs_nepse_sleek(nav, merged, summary)
    chart_returns_profile_sleek(nav, merged)
    chart_trade_profile_sleek(trades)
    chart_risk_profile_sleek(nav, merged, trades)
    chart_strategy_vs_nepse_sleek(
        nav_recent,
        merged_recent,
        recent_summary,
        title="Strategy vs NEPSE",
        date_line="2025-01-01 to 2025-12-31",
        stat_pairs=[
            ("Strategy", f"+{recent_summary['strategy_total']*100:.1f}%", STRATEGY),
            ("NEPSE", f"+{recent_summary['benchmark_total']*100:.1f}%", BENCHMARK),
            ("Sharpe", f"{recent_summary['strategy_sharpe']:.2f}", ACCENT),
            ("Max DD", f"{recent_summary['strategy_dd']*100:.1f}%", NEGATIVE),
        ],
        output_name="recent_01_strategy_vs_nepse.png",
    )
    chart_returns_profile_sleek(
        nav_recent,
        merged_recent,
        title="Return Profile",
        subtitle="2025 monthly heatmap and annual comparison",
        output_name="recent_02_returns_profile.png",
    )
    chart_trade_profile_sleek(
        trades_recent,
        title="Trade Profile",
        subtitle="Trades entered in 2025",
        output_name="recent_03_trade_profile.png",
    )
    chart_risk_profile_sleek(
        nav,
        merged,
        trades_recent,
        title="Risk Profile",
        subtitle="2025 window with rolling context retained",
        output_name="recent_04_risk_profile.png",
        visible_start=RECENT_START,
        visible_end=RECENT_END,
    )

    print(f"Saved clean social charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
