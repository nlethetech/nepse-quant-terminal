#!/usr/bin/env python3
"""
Dual Portfolio Runner — Long-Term + Short-Term Alpha

Runs both portfolio configs on the same date range, reports individual
and combined metrics, and checks portfolio correlation for diversification.

Usage:
    python -m scripts.portfolio.run_dual_portfolio --start 2020-01-01 --end 2025-12-31
    python -m scripts.portfolio.run_dual_portfolio --start 2025-10-01 --end 2026-02-13  # forward test
    python -m scripts.portfolio.run_dual_portfolio --start 2020-01-01 --end 2025-12-31 --allocation 0.6 0.4
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from configs.long_term import LONG_TERM_CONFIG
from configs.short_term import SHORT_TERM_CONFIG
from backend.backtesting.simple_backtest import run_backtest, BacktestResult
from backend.quant_pro.paths import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = get_project_root(__file__)


def run_portfolio(name: str, config: dict, start_date: str, end_date: str) -> BacktestResult:
    """Run a single portfolio backtest from config dict."""
    logger.info(f"{'='*60}")
    logger.info(f"Running {name} portfolio: {start_date} → {end_date}")
    logger.info(f"Signals: {config['signal_types']}")
    logger.info(f"{'='*60}")

    result = run_backtest(
        start_date=start_date,
        end_date=end_date,
        holding_days=config["holding_days"],
        max_positions=config["max_positions"],
        signal_types=config["signal_types"],
        initial_capital=config["initial_capital"],
        rebalance_frequency=config["rebalance_frequency"],
        use_trailing_stop=True,
        trailing_stop_pct=config["trailing_stop_pct"],
        stop_loss_pct=config["stop_loss_pct"],
        use_regime_filter=config["use_regime_filter"],
        sector_limit=config["sector_limit"],
        regime_max_positions=config.get("regime_max_positions"),
        bear_threshold=config["bear_threshold"],
        profit_target_pct=config.get("profit_target_pct"),
        event_exit_mode=config.get("event_exit_mode", False),
    )

    return result


def compute_combined_nav(
    lt_result: BacktestResult,
    st_result: BacktestResult,
    lt_weight: float = 0.5,
    st_weight: float = 0.5,
) -> list:
    """Merge two portfolio NAV series into a combined series with given allocation weights."""
    lt_nav = pd.DataFrame(lt_result.daily_nav, columns=["date", "lt_nav"])
    st_nav = pd.DataFrame(st_result.daily_nav, columns=["date", "st_nav"])

    lt_nav["date"] = pd.to_datetime(lt_nav["date"])
    st_nav["date"] = pd.to_datetime(st_nav["date"])

    merged = pd.merge(lt_nav, st_nav, on="date", how="outer").sort_values("date")
    merged = merged.ffill().dropna()

    if merged.empty:
        return []

    # Normalize both to start at 1.0, then apply weights
    lt_start = merged["lt_nav"].iloc[0]
    st_start = merged["st_nav"].iloc[0]

    if lt_start > 0 and st_start > 0:
        merged["lt_norm"] = merged["lt_nav"] / lt_start
        merged["st_norm"] = merged["st_nav"] / st_start
        total_capital = lt_result.initial_capital * lt_weight + st_result.initial_capital * st_weight
        merged["combined_nav"] = total_capital * (
            lt_weight * merged["lt_norm"] + st_weight * merged["st_norm"]
        )
    else:
        merged["combined_nav"] = merged["lt_nav"] * lt_weight + merged["st_nav"] * st_weight

    return list(zip(merged["date"].tolist(), merged["combined_nav"].tolist()))


def compute_nav_metrics(daily_nav: list, initial_capital: float) -> dict:
    """Compute standard metrics from a daily NAV series."""
    if len(daily_nav) < 30:
        return {"error": "insufficient data"}

    navs = np.array([nav for _, nav in daily_nav])
    rets = np.diff(navs) / navs[:-1]

    total_ret = navs[-1] / navs[0] - 1
    dates = [d for d, _ in daily_nav]
    total_days = (dates[-1] - dates[0]).days if hasattr(dates[-1], 'days') else (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    ann_ret = (1 + total_ret) ** (365 / max(total_days, 1)) - 1

    vol = float(np.std(rets, ddof=1) * np.sqrt(240))
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(240)) if np.std(rets, ddof=1) > 0 else 0

    running_max = np.maximum.accumulate(navs)
    drawdowns = (navs - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    return {
        "total_return": float(total_ret),
        "annualized_return": float(ann_ret),
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def format_pct(val: float) -> str:
    return f"{val:+.2%}" if val != 0 else "0.00%"


def print_comparison(
    lt: BacktestResult,
    st: BacktestResult,
    combined_metrics: dict,
    correlation: float,
    lt_weight: float,
    st_weight: float,
):
    """Print a side-by-side comparison of both portfolios + combined."""
    w = 22  # column width

    print("\n" + "=" * 80)
    print("DUAL PORTFOLIO COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<28} {'Long-Term':>{w}} {'Short-Term':>{w}} {'Combined':>{w}}")
    print("-" * 80)

    rows = [
        ("Total Return", format_pct(lt.total_return), format_pct(st.total_return),
         format_pct(combined_metrics.get("total_return", 0))),
        ("Annualized Return", format_pct(lt.annualized_return), format_pct(st.annualized_return),
         format_pct(combined_metrics.get("annualized_return", 0))),
        ("Volatility", format_pct(lt.volatility), format_pct(st.volatility),
         format_pct(combined_metrics.get("volatility", 0))),
        ("Sharpe Ratio", f"{lt.sharpe_ratio:.2f}", f"{st.sharpe_ratio:.2f}",
         f"{combined_metrics.get('sharpe_ratio', 0):.2f}"),
        ("Max Drawdown", format_pct(lt.max_drawdown), format_pct(st.max_drawdown),
         format_pct(combined_metrics.get("max_drawdown", 0))),
        ("", "", "", ""),
        ("Total Trades", str(lt.total_trades), str(st.total_trades),
         str(lt.total_trades + st.total_trades)),
        ("Win Rate", f"{lt.win_rate:.1%}", f"{st.win_rate:.1%}", "—"),
        ("Avg Win", format_pct(lt.avg_win), format_pct(st.avg_win), "—"),
        ("Avg Loss", format_pct(lt.avg_loss), format_pct(st.avg_loss), "—"),
        ("Profit Factor", f"{lt.profit_factor:.2f}", f"{st.profit_factor:.2f}", "—"),
        ("Avg Hold (days)", f"{lt.avg_holding_days:.1f}", f"{st.avg_holding_days:.1f}", "—"),
        ("Total P&L", f"NPR {lt.total_pnl:,.0f}", f"NPR {st.total_pnl:,.0f}",
         f"NPR {lt.total_pnl + st.total_pnl:,.0f}"),
    ]

    for label, v1, v2, v3 in rows:
        if label == "":
            print()
        else:
            print(f"  {label:<26} {v1:>{w}} {v2:>{w}} {v3:>{w}}")

    print()
    print(f"  Portfolio Correlation:    {correlation:+.3f}")
    print(f"  Allocation:              LT {lt_weight:.0%} / ST {st_weight:.0%}")

    # Diversification benefit
    if combined_metrics.get("sharpe_ratio", 0) > max(lt.sharpe_ratio, st.sharpe_ratio):
        print(f"  Diversification Benefit: YES (combined Sharpe > individual)")
    else:
        print(f"  Diversification Benefit: NO (combined Sharpe <= best individual)")

    print("=" * 80)


def print_short_term_trades(st: BacktestResult):
    """Print every short-term trade for inspection."""
    trades = sorted(st.completed_trades, key=lambda t: t.entry_date)
    if not trades:
        print("\nNo short-term trades.")
        return

    print(f"\n{'='*90}")
    print(f"SHORT-TERM TRADE LOG ({len(trades)} trades)")
    print(f"{'='*90}")
    print(f"  {'Symbol':<10} {'Entry':>10} {'Exit':>10} {'Signal':<20} {'Return':>8} {'Exit Reason':<16}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*20} {'-'*8} {'-'*16}")

    for t in trades:
        entry_str = t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "?"
        exit_str = t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "open"
        ret_str = f"{t.net_return:+.1%}" if t.net_return is not None else "?"
        print(f"  {t.symbol:<10} {entry_str:>10} {exit_str:>10} {t.signal_type:<20} {ret_str:>8} {t.exit_reason:<16}")

    # Signal type breakdown
    print(f"\n  BY SIGNAL TYPE:")
    for sig_type, stats in st.by_signal_type().items():
        print(f"    {sig_type}: {stats['count']} trades, "
              f"{stats['win_rate']:.1%} win, {stats['avg_return']:+.2%} avg")

    # Exit reason breakdown
    print(f"\n  BY EXIT REASON:")
    for reason, count in sorted(st.by_exit_reason().items(), key=lambda x: -x[1]):
        pct = count / st.total_trades if st.total_trades > 0 else 0
        print(f"    {reason}: {count} ({pct:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Run dual portfolio backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--allocation", nargs=2, type=float, default=[0.5, 0.5],
        help="Allocation weights [long_term short_term] (default: 0.5 0.5)",
    )
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    lt_weight, st_weight = args.allocation
    assert abs(lt_weight + st_weight - 1.0) < 0.01, "Allocation weights must sum to 1.0"

    # Run both portfolios
    lt_result = run_portfolio("LONG-TERM", LONG_TERM_CONFIG, args.start, args.end)
    st_result = run_portfolio("SHORT-TERM", SHORT_TERM_CONFIG, args.start, args.end)

    # Print individual summaries
    print("\n" + lt_result.summary())
    print("\n" + st_result.summary())

    # Compute combined NAV
    combined_nav = compute_combined_nav(lt_result, st_result, lt_weight, st_weight)
    combined_initial = lt_result.initial_capital * lt_weight + st_result.initial_capital * st_weight
    combined_metrics = compute_nav_metrics(combined_nav, combined_initial) if len(combined_nav) >= 30 else {}

    # Compute correlation between portfolio daily returns
    lt_rets = lt_result.daily_returns
    st_rets = st_result.daily_returns
    min_len = min(len(lt_rets), len(st_rets))
    if min_len >= 30:
        correlation = float(np.corrcoef(lt_rets[:min_len], st_rets[:min_len])[0, 1])
    else:
        correlation = 0.0

    # Print comparison
    print_comparison(lt_result, st_result, combined_metrics, correlation, lt_weight, st_weight)

    # Print short-term trade log
    print_short_term_trades(st_result)

    # Verification targets
    print(f"\n{'='*60}")
    print("VERIFICATION TARGETS")
    print(f"{'='*60}")
    targets = [
        ("ST Win Rate >= 55%", st_result.win_rate >= 0.55, f"{st_result.win_rate:.1%}"),
        ("ST Sharpe >= 0.7", st_result.sharpe_ratio >= 0.7, f"{st_result.sharpe_ratio:.2f}"),
        ("Combined Sharpe > LT alone", combined_metrics.get("sharpe_ratio", 0) > lt_result.sharpe_ratio,
         f"{combined_metrics.get('sharpe_ratio', 0):.2f} vs {lt_result.sharpe_ratio:.2f}"),
        ("Correlation < 0.3", abs(correlation) < 0.3, f"{correlation:.3f}"),
    ]
    for label, passed, value in targets:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {value}")
    print(f"{'='*60}")

    # Save results
    if args.save:
        output_dir = PROJECT_ROOT / "reports"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "dual_portfolio_results.json"

        results = {
            "run_date": datetime.now().isoformat(),
            "period": {"start": args.start, "end": args.end},
            "allocation": {"long_term": lt_weight, "short_term": st_weight},
            "long_term": {
                "total_return": lt_result.total_return,
                "sharpe": lt_result.sharpe_ratio,
                "max_dd": lt_result.max_drawdown,
                "win_rate": lt_result.win_rate,
                "total_trades": lt_result.total_trades,
                "total_pnl": lt_result.total_pnl,
            },
            "short_term": {
                "total_return": st_result.total_return,
                "sharpe": st_result.sharpe_ratio,
                "max_dd": st_result.max_drawdown,
                "win_rate": st_result.win_rate,
                "total_trades": st_result.total_trades,
                "total_pnl": st_result.total_pnl,
                "by_signal": st_result.by_signal_type(),
                "by_exit": st_result.by_exit_reason(),
            },
            "combined": combined_metrics,
            "correlation": correlation,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
