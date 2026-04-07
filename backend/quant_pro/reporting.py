"""Shared report builders for owner and viewer Telegram bots."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.trading.live_trader import (
    compute_portfolio_intelligence,
    compute_risk_snapshot,
    count_trading_days_since,
    load_trade_log_df,
    now_nst,
    previous_trading_day,
)
from backend.quant_pro.tms_audit import list_executed_trade_events


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _snapshot_trader(trader) -> Dict[str, Any]:
    with trader._state_lock:
        return {
            "capital": float(trader.capital),
            "cash": float(trader.cash),
            "positions": dict(trader.positions),
            "daily_start_nav": float(trader.daily_start_nav) if getattr(trader, "daily_start_nav", None) else None,
            "trade_log_file": str(trader.trade_log_file),
            "nav_log_file": str(trader.nav_log_file),
            "last_refresh": trader.last_refresh,
            "last_price_source_label": str(getattr(trader, "last_price_source_label", "")),
            "last_price_source_detail": str(getattr(trader, "last_price_source_detail", "")),
            "last_price_snapshot_time_utc": getattr(trader, "last_price_snapshot_time_utc", None),
        }


def _build_recent_trades(trade_log_file: str, *, limit: int = 8) -> List[Dict[str, Any]]:
    trades = load_trade_log_df(trade_log_file)
    if trades.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in trades.tail(limit).iterrows():
        rows.append(
            {
                "date": row["Date"].date().isoformat(),
                "action": str(row["Action"]).upper(),
                "symbol": str(row["Symbol"]),
                "shares": int(row["Shares"]),
                "price": float(row["Price"]),
                "fees": float(row["Fees"]),
                "reason": str(row.get("Reason", "")),
                "pnl": float(row.get("PnL", 0.0)),
                "pnl_pct": float(row.get("PnL_Pct", 0.0)),
            }
        )
    rows.reverse()
    return rows


def _build_live_trades(*, limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in list_executed_trade_events(limit=limit):
        completed_at = str(row.get("completed_at") or "")
        rows.append(
            {
                "date": completed_at[:10] if completed_at else "",
                "action": str(row.get("action") or "").upper(),
                "symbol": str(row.get("symbol") or ""),
                "shares": int(row.get("quantity") or 0),
                "price": float(row.get("limit_price") or 0.0),
                "broker_order_ref": row.get("broker_order_ref"),
                "status_text": row.get("status_text") or "",
            }
        )
    return rows


def _build_daily_report(snapshot: Dict[str, Any], intelligence: Dict[str, Any]) -> Dict[str, Any]:
    today = now_nst().date()
    trades = load_trade_log_df(snapshot["trade_log_file"])
    trades_today = []
    if not trades.empty:
        trades_today_df = trades[trades["Date"].dt.date == today]
        for _, row in trades_today_df.iterrows():
            trades_today.append(
                {
                    "date": row["Date"].date().isoformat(),
                    "action": str(row["Action"]).upper(),
                    "symbol": str(row["Symbol"]),
                    "shares": int(row["Shares"]),
                    "price": float(row["Price"]),
                    "pnl": float(row.get("PnL", 0.0)),
                    "reason": str(row.get("Reason", "")),
                }
            )

    nav = snapshot["cash"] + sum(pos.market_value for pos in snapshot["positions"].values())
    day_pnl = None
    day_return_pct = None
    daily_start_nav = snapshot.get("daily_start_nav")
    if isinstance(daily_start_nav, (int, float)) and daily_start_nav > 0:
        day_pnl = float(nav - daily_start_nav)
        day_return_pct = (day_pnl / float(daily_start_nav)) * 100.0
    last_refresh = snapshot.get("last_refresh")
    benchmark = intelligence.get("global_benchmark")
    benchmark_move_pct = None
    if benchmark and benchmark["base_close"] > 0:
        prev_day = previous_trading_day(datetime.strptime(benchmark["latest_date"].isoformat(), "%Y-%m-%d").date())
        benchmark_history = intelligence["benchmark_histories"].get("NEPSE")
        if benchmark_history is not None and not benchmark_history.empty:
            benchmark_history = benchmark_history.copy()
            benchmark_history["Date"] = benchmark_history["Date"].dt.date
            prev_rows = benchmark_history[benchmark_history["Date"] <= prev_day]
            latest_rows = benchmark_history[benchmark_history["Date"] <= benchmark["latest_date"]]
            if not prev_rows.empty and not latest_rows.empty:
                prev_close = float(prev_rows.iloc[-1]["Close"])
                latest_close = float(latest_rows.iloc[-1]["Close"])
                if prev_close > 0:
                    benchmark_move_pct = ((latest_close / prev_close) - 1.0) * 100.0

    positions = intelligence["positions"]
    biggest_contributor = positions[0] if positions else None
    biggest_detractor = positions[-1] if positions else None
    invested_pct = 100.0 - ((snapshot["cash"] / nav) * 100.0 if nav > 0 else 0.0)
    return {
        "date": today.isoformat(),
        "nav": nav,
        "day_pnl": day_pnl,
        "day_return_pct": day_return_pct,
        "benchmark_move_pct": benchmark_move_pct,
        "trades_today": trades_today,
        "biggest_contributor": biggest_contributor,
        "biggest_detractor": biggest_detractor,
        "invested_pct": invested_pct,
        "confidence_score": intelligence["data_quality"]["confidence_score"],
        "last_refresh_nst": last_refresh.strftime("%Y-%m-%d %H:%M:%S") if last_refresh else None,
    }


def build_owner_report(trader) -> Optional[Dict[str, Any]]:
    snapshot = _snapshot_trader(trader)
    intelligence = compute_portfolio_intelligence(
        snapshot["capital"],
        snapshot["cash"],
        snapshot["positions"],
        snapshot["trade_log_file"],
        snapshot["nav_log_file"],
    )
    if intelligence is None:
        return None

    ranked_holdings: List[Dict[str, Any]] = []
    for row in intelligence["positions"]:
        pos = snapshot["positions"].get(row["symbol"])
        ranked_holdings.append(
            {
                "symbol": row["symbol"],
                "shares": pos.shares if pos else row["shares"],
                "entry_price": float(pos.entry_price) if pos else None,
                "last_ltp": float(pos.last_ltp) if pos and pos.last_ltp is not None else None,
                "holding_days": count_trading_days_since(pos.entry_date) if pos else 0,
                "unrealized_pnl": row["unrealized_pnl"],
                "return_pct": row["aligned_return_pct"],
                "active_vs_nepse_pct": row["active_vs_nepse_pct"],
                "active_vs_sector_pct": row["active_vs_sector_pct"],
                "contribution_vs_nepse_pts": row["contribution_vs_nepse_pts"],
                "mark_source": row["aligned_source"],
                "mark_time_utc": row["aligned_time_utc"],
                "sector": row["sector"],
                "strategy": row["strategy"],
            }
        )

    winners = [row for row in ranked_holdings if row["unrealized_pnl"] >= 0]
    laggards = [row for row in sorted(ranked_holdings, key=lambda item: item["contribution_vs_nepse_pts"])]
    risk = compute_risk_snapshot(
        snapshot["capital"],
        snapshot["cash"],
        snapshot["positions"],
        snapshot["nav_log_file"],
    )

    report = {
        "generated_at_nst": now_nst().isoformat(),
        "summary": {
            "capital": snapshot["capital"],
            "cash": snapshot["cash"],
            "nav": risk["nav"],
            "total_return_pct": ((risk["nav"] / snapshot["capital"]) - 1.0) * 100.0 if snapshot["capital"] > 0 else 0.0,
            "open_positions": len(snapshot["positions"]),
            "realized_pnl": intelligence["performance"]["realized_pnl"],
            "realized_return_pct": intelligence["performance"]["realized_return_pct"],
        },
        "portfolio": {
            "holdings_ranked": ranked_holdings,
            "contributors": ranked_holdings[:5],
            "laggards": laggards[:5],
            "recent_trades": _build_recent_trades(snapshot["trade_log_file"]),
            "live_trades": _build_live_trades(),
        },
        "alpha": intelligence,
        "risk": risk,
        "health": {
            "data_quality": intelligence["data_quality"],
            "last_refresh_nst": snapshot["last_refresh"].strftime("%Y-%m-%d %H:%M:%S") if snapshot["last_refresh"] else None,
            "last_price_source_label": snapshot["last_price_source_label"],
            "last_price_source_detail": snapshot["last_price_source_detail"],
            "last_price_snapshot_time_utc": snapshot["last_price_snapshot_time_utc"],
        },
        "daily": _build_daily_report(snapshot, intelligence),
    }
    return report


def build_viewer_report(trader) -> Optional[Dict[str, Any]]:
    owner = build_owner_report(trader)
    if owner is None:
        return None

    show_holdings = _env_flag("NEPSE_VIEWER_SHOW_HOLDINGS", True)
    show_trades = _env_flag("NEPSE_VIEWER_SHOW_TRADES", True)
    show_benchmark = _env_flag("NEPSE_VIEWER_SHOW_BENCHMARK", True)
    alpha = owner["alpha"]
    benchmark = alpha.get("global_benchmark")
    holdings = owner["portfolio"]["holdings_ranked"]
    recent_trades = owner["portfolio"]["recent_trades"][:6]
    live_trades = owner["portfolio"].get("live_trades") or _build_live_trades()
    viewer_trades = live_trades[:6] if live_trades else recent_trades
    viewer = {
        "generated_at_nst": owner["generated_at_nst"],
        "summary": {
            "nav": owner["summary"]["nav"],
            "open_positions": owner["summary"]["open_positions"],
            "total_return_pct": owner["summary"].get("total_return_pct"),
            "realized_pnl": owner["summary"].get("realized_pnl"),
            "realized_return_pct": owner["summary"].get("realized_return_pct"),
            "deployed_return_pct": alpha["performance"]["deployed_return_pct"],
            "day_return_pct": owner["daily"].get("day_return_pct"),
            "benchmark_return_pct": benchmark["return_pct"] if (benchmark and show_benchmark) else None,
        },
        "portfolio": {
            "holdings": [
                {
                    "symbol": row["symbol"],
                    "last_ltp": row.get("last_ltp"),
                    "entry_price": row.get("entry_price"),
                    "holding_days": row["holding_days"],
                    "return_pct": row.get("return_pct", 0.0),
                    "active_vs_nepse_pct": row.get("active_vs_nepse_pct", 0.0),
                    "contribution_vs_nepse_pts": row.get("contribution_vs_nepse_pts", 0.0),
                }
                for row in holdings[:6]
            ] if show_holdings else [],
            "contributors": [
                {
                    "symbol": row["symbol"],
                    "contribution_vs_nepse_pts": row.get("contribution_vs_nepse_pts", 0.0),
                    "return_pct": row.get("return_pct", 0.0),
                }
                for row in holdings[:3]
            ] if show_holdings else [],
            "laggards": [
                {
                    "symbol": row["symbol"],
                    "contribution_vs_nepse_pts": row.get("contribution_vs_nepse_pts", 0.0),
                    "return_pct": row.get("return_pct", 0.0),
                }
                for row in sorted(holdings, key=lambda item: item["contribution_vs_nepse_pts"])[:3]
            ] if show_holdings else [],
            "recent_trades": [
                {
                    "date": row["date"],
                    "action": row["action"],
                    "symbol": row["symbol"],
                    "shares": row["shares"],
                    "price": row["price"],
                    "pnl": row.get("pnl"),
                    "status_text": row.get("status_text"),
                }
                for row in viewer_trades
            ] if show_trades else [],
        },
        "daily": {
            "date": owner["daily"]["date"],
            "nav": owner["daily"]["nav"],
            "day_pnl": owner["daily"].get("day_pnl"),
            "day_return_pct": owner["daily"].get("day_return_pct"),
            "benchmark_move_pct": owner["daily"]["benchmark_move_pct"] if show_benchmark else None,
            "trades_today": [
                {
                    "action": row["action"],
                    "symbol": row["symbol"],
                    "shares": row["shares"],
                    "price": row["price"],
                }
                for row in owner["daily"]["trades_today"]
            ] if show_trades else [],
            "biggest_contributor": owner["daily"]["biggest_contributor"] if show_holdings else None,
            "biggest_detractor": owner["daily"]["biggest_detractor"] if show_holdings else None,
        },
    }
    return viewer
