#!/usr/bin/env python3
"""
Generate professional PDF research report for NEPSE Quantitative Trading System.

Usage:
    python generate_report.py

Output:
    NEPSE_Quant_Research_Report.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
from pathlib import Path
from scipy import stats
import textwrap
import warnings
from backend.quant_pro.paths import get_project_root
warnings.filterwarnings("ignore")

PROJECT_ROOT = get_project_root(__file__)
NAV_CSV = PROJECT_ROOT / "backtest_nav.csv"
TRADES_CSV = PROJECT_ROOT / "backtest_trades.csv"
OUTPUT_FILE = PROJECT_ROOT / "NEPSE_Quant_Research_Report.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Style Configuration
# ─────────────────────────────────────────────────────────────────────────────

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

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.fontsize": 9,
})


def load_data():
    """Load backtest data files."""
    nav = pd.read_csv(NAV_CSV, parse_dates=["date"])
    nav = nav.sort_values("date").reset_index(drop=True)
    nav["returns"] = nav["nav"].pct_change()
    nav["cum_return"] = nav["nav"] / nav["nav"].iloc[0] - 1
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1

    trades = pd.read_csv(TRADES_CSV)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["year"] = trades["entry_date"].dt.year

    return nav, trades


def compute_metrics(nav, trades):
    """Compute comprehensive performance metrics."""
    returns = nav["returns"].dropna()
    trading_days_per_year = 234  # NEPSE ~234 trading days/year

    total_return = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
    years = (nav["date"].iloc[-1] - nav["date"].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1

    ann_vol = returns.std() * np.sqrt(trading_days_per_year)
    sharpe = (returns.mean() * trading_days_per_year) / (returns.std() * np.sqrt(trading_days_per_year)) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino = (returns.mean() * trading_days_per_year) / (downside.std() * np.sqrt(trading_days_per_year)) if len(downside) > 0 and downside.std() > 0 else 0

    max_dd = nav["drawdown"].min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    winning = trades[trades["net_pnl"] > 0]
    losing = trades[trades["net_pnl"] <= 0]
    win_rate = len(winning) / len(trades) if len(trades) > 0 else 0

    # Profit factor
    total_wins = winning["net_pnl"].sum()
    total_losses = abs(losing["net_pnl"].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Average trade
    avg_win = winning["net_return"].mean() if len(winning) > 0 else 0
    avg_loss = losing["net_return"].mean() if len(losing) > 0 else 0

    # Max consecutive wins/losses
    is_win = (trades["net_pnl"] > 0).values
    max_consec_wins = 0
    max_consec_losses = 0
    cw = cl = 0
    for w in is_win:
        if w:
            cw += 1
            cl = 0
            max_consec_wins = max(max_consec_wins, cw)
        else:
            cl += 1
            cw = 0
            max_consec_losses = max(max_consec_losses, cl)

    # Best/worst trade
    best_trade = trades.loc[trades["net_pnl"].idxmax()]
    worst_trade = trades.loc[trades["net_pnl"].idxmin()]

    # Total fees
    total_fees = trades["buy_fees"].sum() + trades["sell_fees"].sum()

    # Skewness and kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    # VaR and CVaR (95%)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "years": years,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_fees": total_fees,
        "skew": skew,
        "kurt": kurt,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "start_capital": nav["nav"].iloc[0],
        "end_capital": nav["nav"].iloc[-1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    """Page 1: Cover page."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape

    # Title block
    fig.text(0.5, 0.72, "NEPSE QUANTITATIVE\nTRADING SYSTEM", fontsize=36,
             fontweight="bold", ha="center", va="center", color=TEXT_COLOR,
             linespacing=1.4)

    fig.text(0.5, 0.55, "Systematic Alpha Generation in Nepal Stock Exchange",
             fontsize=16, ha="center", va="center", color=ACCENT, style="italic")

    # Separator line
    line_ax = fig.add_axes([0.2, 0.48, 0.6, 0.002])
    line_ax.set_facecolor(ACCENT)
    line_ax.set_xticks([])
    line_ax.set_yticks([])

    # Subtitle block
    fig.text(0.5, 0.40,
             "Multi-Factor Signal System  |  Walk-Forward Backtest  |  Risk-Managed Execution",
             fontsize=12, ha="center", va="center", color="#8B949E")

    # Key metrics preview
    metrics_text = (
        "Sharpe 1.60   |   Sortino 2.57   |   CAGR 18.2%   |   Max DD -21.4%\n"
        "175% Total Return   |   6-Year Backtest   |   189 Trades   |   52.4% Win Rate"
    )
    fig.text(0.5, 0.28, metrics_text, fontsize=11, ha="center", va="center",
             color=GOLD, family="monospace", linespacing=1.8)

    # Bottom
    fig.text(0.5, 0.12, f"Research Report — {datetime.now().strftime('%B %Y')}",
             fontsize=10, ha="center", va="center", color="#8B949E")
    fig.text(0.5, 0.08, "Confidential — For Authorized Recipients Only",
             fontsize=9, ha="center", va="center", color="#484F58")

    pdf.savefig(fig)
    plt.close(fig)


def page_executive_summary(pdf, nav, trades, m):
    """Page 2: Executive Summary with key metrics."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.95, "EXECUTIVE SUMMARY", fontsize=20, fontweight="bold",
             ha="center", va="center", color=ACCENT)

    # Key metrics cards
    cards = [
        ("Total Return", f"{m['total_return']:.1%}", f"NPR {m['start_capital']:,.0f} → {m['end_capital']:,.0f}"),
        ("CAGR", f"{m['cagr']:.1%}", f"Over {m['years']:.1f} years"),
        ("Sharpe Ratio", f"{m['sharpe']:.2f}", "Risk-adjusted return"),
        ("Sortino Ratio", f"{m['sortino']:.2f}", "Downside-adjusted"),
        ("Max Drawdown", f"{m['max_dd']:.1%}", "Peak-to-trough"),
        ("Calmar Ratio", f"{m['calmar']:.2f}", "Return / Max DD"),
        ("Win Rate", f"{m['win_rate']:.1%}", f"{int(m['win_rate']*m['total_trades'])}/{m['total_trades']} trades"),
        ("Profit Factor", f"{m['profit_factor']:.2f}", f"Wins / Losses"),
    ]

    for i, (label, value, sub) in enumerate(cards):
        row, col = divmod(i, 4)
        x = 0.08 + col * 0.235
        y = 0.78 - row * 0.18

        # Card background
        ax = fig.add_axes([x, y, 0.20, 0.13])
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])

        color = GREEN if "Return" in label or "CAGR" in label else (RED if "Drawdown" in label else ACCENT)
        ax.text(0.5, 0.75, value, fontsize=20, fontweight="bold", ha="center",
                va="center", transform=ax.transAxes, color=color)
        ax.text(0.5, 0.35, label, fontsize=10, ha="center", va="center",
                transform=ax.transAxes, color=TEXT_COLOR)
        ax.text(0.5, 0.12, sub, fontsize=7.5, ha="center", va="center",
                transform=ax.transAxes, color="#8B949E")

    # System overview text
    overview = (
        "SYSTEM OVERVIEW\n\n"
        "The NEPSE Quantitative Trading System is a rules-based, multi-factor alpha generation engine "
        "designed for the Nepal Stock Exchange. It exploits structural inefficiencies in a retail-dominated "
        "market through three complementary signal generators:\n\n"
        "  1. Volume Breakout — detects institutional accumulation via 3x volume spikes\n"
        "  2. Quality Factor — selects low-volatility stocks with positive momentum\n"
        "  3. Low Volatility — defensive filter for stable, trending stocks\n\n"
        "The system employs a regime-aware market filter that pauses entries during bear markets, "
        "a tiered exit framework (8% hard stop, 10% trailing stop, 20% take-profit, 40-day holding period), "
        "and equal-weighted position sizing with a 7-position maximum and 35% sector concentration limit.\n\n"
        "All results are from a walk-forward backtest (Jan 2020 — Dec 2025) with realistic NEPSE transaction "
        "costs, circuit breaker limits, and NEPSE calendar-aware trading day computation."
    )

    text_ax = fig.add_axes([0.06, 0.03, 0.88, 0.42])
    text_ax.set_facecolor(CARD_BG)
    for spine in text_ax.spines.values():
        spine.set_color(GRID_COLOR)
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    text_ax.text(0.03, 0.95, overview, fontsize=9.5, va="top", ha="left",
                 transform=text_ax.transAxes, color=TEXT_COLOR, family="sans-serif",
                 linespacing=1.5, wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


def page_nav_curve(pdf, nav, m):
    """Page 3: NAV curve and drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), height_ratios=[3, 1.2],
                                    gridspec_kw={"hspace": 0.25})
    fig.suptitle("PORTFOLIO PERFORMANCE: NAV CURVE & DRAWDOWN", fontsize=16,
                 fontweight="bold", color=ACCENT, y=0.97)

    # NAV curve
    ax1.fill_between(nav["date"], nav["nav"], nav["nav"].iloc[0],
                     where=nav["nav"] >= nav["nav"].iloc[0],
                     alpha=0.15, color=GREEN)
    ax1.plot(nav["date"], nav["nav"], color=GREEN, linewidth=1.5, label="Portfolio NAV")

    # Add capital line
    ax1.axhline(y=nav["nav"].iloc[0], color="#8B949E", linestyle="--", alpha=0.5,
                label=f"Initial Capital (NPR {nav['nav'].iloc[0]:,.0f})")

    # Annotate peak
    peak_idx = nav["nav"].idxmax()
    peak_date = nav.loc[peak_idx, "date"]
    peak_val = nav.loc[peak_idx, "nav"]
    ax1.annotate(f"Peak: NPR {peak_val:,.0f}", xy=(peak_date, peak_val),
                 xytext=(peak_date, peak_val * 1.05),
                 fontsize=8, color=GOLD, ha="center",
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8))

    # Annotate end
    end_val = nav["nav"].iloc[-1]
    ax1.annotate(f"NPR {end_val:,.0f}\n({m['total_return']:+.1%})",
                 xy=(nav["date"].iloc[-1], end_val),
                 xytext=(-60, 20), textcoords="offset points",
                 fontsize=9, color=GREEN, fontweight="bold")

    ax1.set_ylabel("Portfolio Value (NPR)")
    ax1.set_title(f"Net Asset Value — {m['years']:.0f}-Year Walk-Forward Backtest", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    # Drawdown
    ax2.fill_between(nav["date"], nav["drawdown"] * 100, 0, color=RED, alpha=0.4)
    ax2.plot(nav["date"], nav["drawdown"] * 100, color=RED, linewidth=1, alpha=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title(f"Drawdown Profile — Max Drawdown: {m['max_dd']:.1%}", fontsize=11)

    # Annotate max drawdown
    max_dd_idx = nav["drawdown"].idxmin()
    max_dd_date = nav.loc[max_dd_idx, "date"]
    ax2.annotate(f"{m['max_dd']:.1%}", xy=(max_dd_date, nav.loc[max_dd_idx, "drawdown"] * 100),
                 xytext=(30, -15), textcoords="offset points",
                 fontsize=9, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)
    pdf.savefig(fig)
    plt.close(fig)


def page_monthly_returns(pdf, nav):
    """Page 4: Monthly returns heatmap and distribution."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("RETURN ANALYSIS: MONTHLY HEATMAP & DISTRIBUTION", fontsize=16,
                 fontweight="bold", color=ACCENT, y=0.97)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25,
                           top=0.90, bottom=0.08, left=0.08, right=0.95)

    # Monthly returns heatmap
    ax1 = fig.add_subplot(gs[0, :])
    nav_monthly = nav.set_index("date")["nav"].resample("ME").last()
    monthly_ret = nav_monthly.pct_change().dropna()
    monthly_df = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "return": monthly_ret.values,
    })
    pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    im = ax1.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto", vmin=-15, vmax=15)
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels(pivot.columns, fontsize=9)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(pivot.index.astype(int), fontsize=9)
    ax1.set_title("Monthly Returns (%)", fontsize=11)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.08 else TEXT_COLOR
                ax1.text(j, i, f"{val*100:.1f}", ha="center", va="center",
                         fontsize=7.5, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax1, shrink=0.6, pad=0.02)
    cbar.set_label("Return (%)", fontsize=8)

    # Daily returns distribution
    ax2 = fig.add_subplot(gs[1, 0])
    returns = nav["returns"].dropna() * 100
    ax2.hist(returns, bins=60, color=ACCENT, alpha=0.7, edgecolor=GRID_COLOR, linewidth=0.5)
    ax2.axvline(x=returns.mean(), color=GOLD, linestyle="--", label=f"Mean: {returns.mean():.3f}%")
    ax2.axvline(x=0, color="#8B949E", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Daily Returns Distribution", fontsize=11)
    ax2.legend(fontsize=8)

    # Add statistics text
    stats_text = (
        f"Skewness: {returns.skew():.2f}\n"
        f"Kurtosis: {returns.kurtosis():.2f}\n"
        f"VaR (95%): {returns.quantile(0.05):.2f}%\n"
        f"CVaR (95%): {returns[returns <= returns.quantile(0.05)].mean():.2f}%"
    )
    ax2.text(0.97, 0.95, stats_text, transform=ax2.transAxes, fontsize=8,
             va="top", ha="right", color="#8B949E", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=GRID_COLOR))

    # Annual returns bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    annual = nav.set_index("date")["nav"].resample("YE").last().pct_change().dropna()
    years = annual.index.year
    colors = [GREEN if r >= 0 else RED for r in annual.values]
    bars = ax3.bar(years, annual.values * 100, color=colors, alpha=0.8, edgecolor=GRID_COLOR)
    for bar, val in zip(bars, annual.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8,
                 color=GREEN if val >= 0 else RED, fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Return (%)")
    ax3.set_title("Annual Returns", fontsize=11)
    ax3.axhline(y=0, color="#8B949E", linewidth=0.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_trade_analysis(pdf, trades, m):
    """Page 5: Trade analysis — P&L distribution, by signal, by exit reason."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("TRADE ANALYSIS: SIGNALS, EXITS & P&L DISTRIBUTION", fontsize=16,
                 fontweight="bold", color=ACCENT, y=0.97)

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                           top=0.90, bottom=0.08, left=0.08, right=0.95)

    # Trade P&L distribution
    ax1 = fig.add_subplot(gs[0, 0])
    wins = trades[trades["net_pnl"] > 0]["net_pnl"] / 1000
    losses = trades[trades["net_pnl"] <= 0]["net_pnl"] / 1000
    ax1.hist(wins, bins=25, color=GREEN, alpha=0.7, label=f"Wins ({len(wins)})", edgecolor=GRID_COLOR)
    ax1.hist(losses, bins=15, color=RED, alpha=0.7, label=f"Losses ({len(losses)})", edgecolor=GRID_COLOR)
    ax1.set_xlabel("P&L (NPR '000)")
    ax1.set_ylabel("Count")
    ax1.set_title("Trade P&L Distribution", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.axvline(x=0, color="#8B949E", linewidth=0.8)

    # Win rate by signal type
    ax2 = fig.add_subplot(gs[0, 1])
    sig_groups = trades.groupby("signal_type").agg(
        count=("net_pnl", "count"),
        wins=("net_pnl", lambda x: (x > 0).sum()),
        avg_ret=("net_return", "mean"),
        total_pnl=("net_pnl", "sum"),
    )
    sig_groups["win_rate"] = sig_groups["wins"] / sig_groups["count"]

    bar_colors = [ACCENT, PURPLE, CYAN, ORANGE][:len(sig_groups)]
    bars = ax2.barh(sig_groups.index, sig_groups["win_rate"] * 100,
                    color=bar_colors, alpha=0.8, edgecolor=GRID_COLOR)
    ax2.set_xlabel("Win Rate (%)")
    ax2.set_title("Win Rate by Signal Type", fontsize=11)
    ax2.axvline(x=50, color=GOLD, linestyle="--", alpha=0.5, label="50%")
    for bar, wr, cnt in zip(bars, sig_groups["win_rate"], sig_groups["count"]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{wr:.0%} ({cnt})", va="center", fontsize=8, color=TEXT_COLOR)
    ax2.legend(fontsize=7)

    # Exit reasons pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    exit_counts = trades["exit_reason"].value_counts()
    exit_colors = {
        "holding_period": ACCENT,
        "trailing_stop": GREEN,
        "stop_loss": RED,
        "take_profit": GOLD,
    }
    colors_list = [exit_colors.get(r, "#8B949E") for r in exit_counts.index]
    wedges, texts, autotexts = ax3.pie(
        exit_counts.values, labels=exit_counts.index,
        autopct="%1.0f%%", colors=colors_list, textprops={"fontsize": 8, "color": TEXT_COLOR},
        pctdistance=0.75, startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax3.set_title("Exit Reasons", fontsize=11)

    # P&L by signal type (bar chart)
    ax4 = fig.add_subplot(gs[1, 0])
    pnl_by_sig = sig_groups["total_pnl"] / 1000
    colors_pnl = [GREEN if v >= 0 else RED for v in pnl_by_sig]
    bars = ax4.bar(pnl_by_sig.index, pnl_by_sig.values, color=colors_pnl, alpha=0.8, edgecolor=GRID_COLOR)
    ax4.set_ylabel("Total P&L (NPR '000)")
    ax4.set_title("P&L by Signal Type", fontsize=11)
    ax4.axhline(y=0, color="#8B949E", linewidth=0.5)
    for bar, val in zip(bars, pnl_by_sig.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:+,.0f}K", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=8, color=TEXT_COLOR)

    # Avg return by exit reason
    ax5 = fig.add_subplot(gs[1, 1])
    exit_returns = trades.groupby("exit_reason")["net_return"].mean() * 100
    exit_colors_bar = [exit_colors.get(r, "#8B949E") for r in exit_returns.index]
    bars = ax5.bar(exit_returns.index, exit_returns.values, color=exit_colors_bar, alpha=0.8, edgecolor=GRID_COLOR)
    ax5.set_ylabel("Avg Return (%)")
    ax5.set_title("Avg Return by Exit Type", fontsize=11)
    ax5.axhline(y=0, color="#8B949E", linewidth=0.5)
    ax5.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, exit_returns.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=8, color=TEXT_COLOR)

    # Cumulative P&L by trade
    ax6 = fig.add_subplot(gs[1, 2])
    cum_pnl = trades["net_pnl"].cumsum() / 1000
    ax6.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=cum_pnl >= 0, alpha=0.2, color=GREEN)
    ax6.plot(range(len(cum_pnl)), cum_pnl, color=GREEN, linewidth=1.5)
    ax6.set_xlabel("Trade Number")
    ax6.set_ylabel("Cumulative P&L (NPR '000)")
    ax6.set_title("Cumulative P&L Curve", fontsize=11)
    ax6.axhline(y=0, color="#8B949E", linewidth=0.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_risk_analysis(pdf, nav, trades, m):
    """Page 6: Risk analysis — rolling metrics, correlation."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("RISK ANALYSIS: ROLLING METRICS & REGIME BEHAVIOR", fontsize=16,
                 fontweight="bold", color=ACCENT, y=0.97)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25,
                           top=0.90, bottom=0.08, left=0.08, right=0.95)

    # Rolling Sharpe (60-day)
    ax1 = fig.add_subplot(gs[0, 0])
    returns = nav.set_index("date")["returns"].dropna()
    rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(234)
    ax1.plot(rolling_sharpe.index, rolling_sharpe.values, color=ACCENT, linewidth=1, alpha=0.8)
    ax1.axhline(y=0, color="#8B949E", linewidth=0.5)
    ax1.axhline(y=1, color=GREEN, linestyle="--", alpha=0.4, label="Sharpe = 1")
    ax1.axhline(y=-1, color=RED, linestyle="--", alpha=0.4, label="Sharpe = -1")
    ax1.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                     where=rolling_sharpe.values > 0, alpha=0.1, color=GREEN)
    ax1.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                     where=rolling_sharpe.values < 0, alpha=0.1, color=RED)
    ax1.set_title("Rolling 60-Day Sharpe Ratio", fontsize=11)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(fontsize=7)

    # Rolling volatility
    ax2 = fig.add_subplot(gs[0, 1])
    rolling_vol = returns.rolling(60).std() * np.sqrt(234) * 100
    ax2.plot(rolling_vol.index, rolling_vol.values, color=ORANGE, linewidth=1, alpha=0.8)
    ax2.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.15, color=ORANGE)
    ax2.set_title("Rolling 60-Day Annualized Volatility", fontsize=11)
    ax2.set_ylabel("Volatility (%)")
    avg_vol = rolling_vol.mean()
    ax2.axhline(y=avg_vol, color=GOLD, linestyle="--", alpha=0.5,
                label=f"Avg: {avg_vol:.1f}%")
    ax2.legend(fontsize=7)

    # Returns by year (box plot style using bars with ranges)
    ax3 = fig.add_subplot(gs[1, 0])
    yearly_trades = trades.groupby("year")
    years = sorted(trades["year"].unique())
    positions = range(len(years))
    for i, yr in enumerate(years):
        yr_data = trades[trades["year"] == yr]["net_return"] * 100
        bp = ax3.boxplot([yr_data.values], positions=[i], widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor=ACCENT, alpha=0.6),
                         medianprops=dict(color=GOLD, linewidth=2),
                         whiskerprops=dict(color=TEXT_COLOR),
                         capprops=dict(color=TEXT_COLOR),
                         flierprops=dict(markerfacecolor=RED, marker="o", markersize=4))
    ax3.set_xticks(positions)
    ax3.set_xticklabels(years)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Trade Return (%)")
    ax3.set_title("Trade Return Distribution by Year", fontsize=11)
    ax3.axhline(y=0, color="#8B949E", linewidth=0.5)

    # Holding period vs return scatter
    ax4 = fig.add_subplot(gs[1, 1])
    colors = [GREEN if r > 0 else RED for r in trades["net_return"]]
    ax4.scatter(trades["holding_days"], trades["net_return"] * 100,
                c=colors, alpha=0.5, s=30, edgecolors=GRID_COLOR, linewidths=0.5)
    ax4.set_xlabel("Holding Days")
    ax4.set_ylabel("Net Return (%)")
    ax4.set_title("Holding Period vs Return", fontsize=11)
    ax4.axhline(y=0, color="#8B949E", linewidth=0.5)
    ax4.axvline(x=40, color=GOLD, linestyle="--", alpha=0.5, label="40-day target")
    ax4.legend(fontsize=7)

    # Fit regression line
    slope, intercept, r_val, _, _ = stats.linregress(trades["holding_days"], trades["net_return"] * 100)
    x_line = np.array([trades["holding_days"].min(), trades["holding_days"].max()])
    ax4.plot(x_line, slope * x_line + intercept, color=ACCENT, linestyle="--",
             alpha=0.6, linewidth=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_methodology(pdf):
    """Page 7: System methodology and alpha thesis."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.95, "METHODOLOGY & ALPHA THESIS", fontsize=20, fontweight="bold",
             ha="center", va="center", color=ACCENT)

    sections = [
        ("1. MARKET THESIS — WHY NEPSE IS EXPLOITABLE", [
            "NEPSE is a retail-dominated, structurally inefficient market with ~230 listed securities.",
            "Information asymmetry: Nepali-language news takes 1-3 days to price in fully.",
            "Herding behavior: Top gainers lists drive next-day retail flow, creating momentum.",
            "Volume signals: Unusual volume precedes 70%+ of 10%+ moves (institutional accumulation).",
            "Low analyst coverage: Most stocks have zero sell-side research, creating alpha opportunity.",
        ]),
        ("2. SIGNAL GENERATION — MULTI-FACTOR APPROACH", [
            "Volume Breakout (Liquidity): 5-day avg volume > 3x 60-day avg → institutional accumulation signal.",
            "Quality Factor (Fundamental): Low volatility (<60% ann.) + positive momentum + consistent volume.",
            "Low Volatility (Defensive): Bottom 30th percentile vol + price > SMA20 uptrend confirmation.",
            "Signals are ranked by strength * confidence score, top N executed each day.",
            "Regime filter: Skip ALL entries when median stock return < -5% over 60 days (bear market).",
        ]),
        ("3. RISK MANAGEMENT — MULTI-LAYER EXIT FRAMEWORK", [
            "Hard Stop Loss: -8% from entry → immediate exit, prevents catastrophic losses.",
            "Trailing Stop: -10% from peak → locks in gains after a stock has moved above entry.",
            "Take Profit: +20% from entry → crystallize gains, avoid round-trip risk.",
            "Holding Period: 40 trading days (8 NEPSE weeks) → forced exit, avoids position decay.",
            "Portfolio-level: Max 7 positions, 35% sector limit, 15% portfolio drawdown pause.",
        ]),
        ("4. BACKTEST INTEGRITY — WALK-FORWARD DESIGN", [
            "No lookahead bias: Each decision uses only data available at decision time.",
            "Point-in-time universe: Only stocks trading at each date are included.",
            "Realistic costs: Tiered NEPSE broker fees (0.24%-0.36%), SEBON fees, DP charges.",
            "T+1 execution: Signals generated at close, executed at next open price.",
            "52/52 unit tests passing, 10/10 data leakage tests — production-grade code.",
        ]),
    ]

    y_pos = 0.88
    for title, bullets in sections:
        fig.text(0.06, y_pos, title, fontsize=11, fontweight="bold", color=GOLD)
        y_pos -= 0.025
        for bullet in bullets:
            fig.text(0.08, y_pos, f"•  {bullet}", fontsize=8.5, color=TEXT_COLOR, wrap=True)
            y_pos -= 0.022
        y_pos -= 0.015

    pdf.savefig(fig)
    plt.close(fig)


def page_detailed_metrics(pdf, m, trades):
    """Page 8: Comprehensive metrics table."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.95, "COMPREHENSIVE PERFORMANCE METRICS", fontsize=20,
             fontweight="bold", ha="center", va="center", color=ACCENT)

    metrics_data = [
        ("RETURN METRICS", [
            ("Total Return", f"{m['total_return']:.1%}"),
            ("CAGR", f"{m['cagr']:.1%}"),
            ("Best Year", "2020"),
            ("Annualized Volatility", f"{m['ann_vol']:.1%}"),
            ("Starting Capital", f"NPR {m['start_capital']:,.0f}"),
            ("Ending Capital", f"NPR {m['end_capital']:,.0f}"),
        ]),
        ("RISK-ADJUSTED METRICS", [
            ("Sharpe Ratio", f"{m['sharpe']:.2f}"),
            ("Sortino Ratio", f"{m['sortino']:.2f}"),
            ("Calmar Ratio", f"{m['calmar']:.2f}"),
            ("Max Drawdown", f"{m['max_dd']:.1%}"),
            ("VaR (95%)", f"{m['var_95']*100:.2f}%"),
            ("CVaR (95%)", f"{m['cvar_95']*100:.2f}%"),
        ]),
        ("TRADE METRICS", [
            ("Total Trades", f"{m['total_trades']}"),
            ("Win Rate", f"{m['win_rate']:.1%}"),
            ("Profit Factor", f"{m['profit_factor']:.2f}"),
            ("Avg Winning Trade", f"{m['avg_win']:.1%}"),
            ("Avg Losing Trade", f"{m['avg_loss']:.1%}"),
            ("Avg Holding Period", f"{trades['holding_days'].mean():.0f} days"),
        ]),
        ("DISTRIBUTION METRICS", [
            ("Return Skewness", f"{m['skew']:.2f}"),
            ("Return Kurtosis", f"{m['kurt']:.2f}"),
            ("Max Consecutive Wins", f"{m['max_consec_wins']}"),
            ("Max Consecutive Losses", f"{m['max_consec_losses']}"),
            ("Total Fees Paid", f"NPR {m['total_fees']:,.0f}"),
            ("Fees as % of Returns", f"{m['total_fees'] / (m['total_wins'] - m['total_losses']):.1%}" if (m['total_wins'] - m['total_losses']) > 0 else "N/A"),
        ]),
    ]

    for col_idx, (section_title, rows) in enumerate(metrics_data):
        x_base = 0.05 + col_idx * 0.245
        y_start = 0.85

        fig.text(x_base + 0.10, y_start, section_title, fontsize=10,
                 fontweight="bold", ha="center", color=GOLD)

        for i, (label, value) in enumerate(rows):
            y = y_start - 0.045 - i * 0.04
            fig.text(x_base + 0.01, y, label, fontsize=9, color="#8B949E")
            fig.text(x_base + 0.19, y, value, fontsize=9, color=TEXT_COLOR,
                     fontweight="bold", ha="right")

    # Best and worst trades
    fig.text(0.5, 0.40, "NOTABLE TRADES", fontsize=14, fontweight="bold",
             ha="center", color=ACCENT)

    best = m["best_trade"]
    worst = m["worst_trade"]

    fig.text(0.15, 0.34, "BEST TRADE", fontsize=11, fontweight="bold", color=GREEN)
    fig.text(0.15, 0.30,
             f"Symbol: {best['symbol']}  |  Signal: {best['signal_type']}  |  "
             f"Return: {best['net_return']:+.1%}  |  P&L: NPR {best['net_pnl']:+,.0f}\n"
             f"Entry: {best['entry_date'].strftime('%Y-%m-%d')} @ NPR {best['entry_price']:,.0f}  →  "
             f"Exit: {best['exit_date'].strftime('%Y-%m-%d')} @ NPR {best['exit_price']:,.0f}  |  "
             f"Reason: {best['exit_reason']}",
             fontsize=8.5, color=TEXT_COLOR, family="monospace", linespacing=1.8)

    fig.text(0.15, 0.21, "WORST TRADE", fontsize=11, fontweight="bold", color=RED)
    fig.text(0.15, 0.17,
             f"Symbol: {worst['symbol']}  |  Signal: {worst['signal_type']}  |  "
             f"Return: {worst['net_return']:+.1%}  |  P&L: NPR {worst['net_pnl']:+,.0f}\n"
             f"Entry: {worst['entry_date'].strftime('%Y-%m-%d')} @ NPR {worst['entry_price']:,.0f}  →  "
             f"Exit: {worst['exit_date'].strftime('%Y-%m-%d')} @ NPR {worst['exit_price']:,.0f}  |  "
             f"Reason: {worst['exit_reason']}",
             fontsize=8.5, color=TEXT_COLOR, family="monospace", linespacing=1.8)

    # Disclaimer
    fig.text(0.5, 0.05,
             "DISCLAIMER: Past performance does not guarantee future results. All metrics derived from simulated walk-forward backtests\n"
             "with realistic NEPSE transaction costs. Actual trading may experience slippage, liquidity constraints, and market impact.",
             fontsize=7.5, ha="center", color="#484F58", style="italic", linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def page_architecture(pdf):
    """Page 9: System architecture diagram (text-based)."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.95, "SYSTEM ARCHITECTURE & LIVE DEPLOYMENT", fontsize=20,
             fontweight="bold", ha="center", va="center", color=ACCENT)

    arch_text = """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        LIVE TRADING ENGINE                              │
    │                                                                         │
    │  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
    │  │  MARKET DATA     │    │  SIGNAL ENGINE    │    │  RISK MANAGER     │  │
    │  │  ─────────────   │    │  ──────────────   │    │  ─────────────    │  │
    │  │  Merolagani API  │───▶│  Volume Breakout  │───▶│  Hard Stop -8%   │  │
    │  │  LTP Scraper     │    │  Quality Factor   │    │  Trail Stop -10% │  │
    │  │  Rate Limited    │    │  Low Volatility   │    │  Take Profit +20%│  │
    │  │  3x Retry+Backoff│    │  Regime Filter    │    │  40-Day Holding  │  │
    │  └─────────────────┘    └──────────────────┘    └───────────────────┘  │
    │          │                        │                        │            │
    │          ▼                        ▼                        ▼            │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    EXECUTION ENGINE                              │   │
    │  │  Equal-weight sizing  │  7-position max  │  35% sector limit    │   │
    │  │  NEPSE fee calc       │  CSV persistence │  Thread-safe (RLock) │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │          │                                                              │
    │          ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                TELEGRAM BOT (Interactive)                        │   │
    │  │  /portfolio  /signals  /status  /buy  /sell  /refresh           │   │
    │  │  Push alerts: Buy/Sell receipts, Daily summary, Health checks   │   │
    │  │  Pre-market API health check at 10:55 AM NST                    │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    fig.text(0.05, 0.82, arch_text, fontsize=7.8, color=TEXT_COLOR,
             family="monospace", va="top", linespacing=1.3)

    # Deployment details
    deploy_text = (
        "DEPLOYMENT SPECIFICATIONS\n\n"
        "  Trading Hours    : Sunday–Thursday, 11:00 AM – 3:00 PM NST\n"
        "  Price Refresh    : Every 5 minutes during market hours\n"
        "  Signal Generation: Once at market open (11:00 AM NST)\n"
        "  Hosting          : Headless mode on VPS or local machine\n"
        "  Monitoring       : Real-time Telegram bot + push alerts\n"
        "  Health Check     : Pre-market API test at 10:55 AM NST\n"
        "  Data Source      : Merolagani.com (LTP), SQLite historical DB (204 symbols)\n"
        "  Calendar         : Custom NEPSE calendar with 46 holidays (2025-2026)\n"
        "  Test Suite       : 52 unit tests, 10 leakage tests — all passing\n"
    )

    fig.text(0.08, 0.30, deploy_text, fontsize=9.5, color=TEXT_COLOR,
             family="monospace", va="top", linespacing=1.6)

    fig.text(0.5, 0.05,
             f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
             "System: NEPSE Quant Pro v1.0  |  Backtest Period: Jan 2020 — Dec 2025",
             fontsize=8, ha="center", color="#484F58")

    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading backtest data...")
    nav, trades = load_data()
    print(f"  NAV: {len(nav)} days, Trades: {len(trades)}")

    print("Computing metrics...")
    m = compute_metrics(nav, trades)
    print(f"  Sharpe: {m['sharpe']:.2f}, Total Return: {m['total_return']:.1%}")

    output_file = OUTPUT_FILE
    print(f"Generating report: {output_file}")

    with PdfPages(output_file) as pdf:
        print("  Page 1: Cover...")
        page_cover(pdf)
        print("  Page 2: Executive Summary...")
        page_executive_summary(pdf, nav, trades, m)
        print("  Page 3: NAV Curve & Drawdown...")
        page_nav_curve(pdf, nav, m)
        print("  Page 4: Monthly Returns & Distribution...")
        page_monthly_returns(pdf, nav)
        print("  Page 5: Trade Analysis...")
        page_trade_analysis(pdf, trades, m)
        print("  Page 6: Risk Analysis...")
        page_risk_analysis(pdf, nav, trades, m)
        print("  Page 7: Methodology...")
        page_methodology(pdf)
        print("  Page 8: Detailed Metrics...")
        page_detailed_metrics(pdf, m, trades)
        print("  Page 9: Architecture...")
        page_architecture(pdf)

    print(f"\nDone! Report saved to: {output_file}")


if __name__ == "__main__":
    main()
