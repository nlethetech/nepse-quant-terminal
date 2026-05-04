"""
Interactive Telegram Bot for NEPSE Live Paper Trader.

Provides two-way interaction: view portfolio/signals/status and execute
manual buy/sell orders with inline button confirmations.

Runs as a daemon thread alongside the Rich TUI (or in headless mode).
"""

from __future__ import annotations

import asyncio
import functools
from io import BytesIO
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
)

from backend.quant_pro.message_formatters import (
    format_portfolio_holding_html,
    format_trade_activity_html,
)
from backend.quant_pro.control_plane.command_service import build_live_trader_control_plane

if TYPE_CHECKING:
    from backend.trading.live_trader import LiveTrader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state (accessible from telegram_alerts.py)
# ─────────────────────────────────────────────────────────────────────────────

_applications: Dict[str, Application] = {}
_trader: Optional["LiveTrader"] = None


def get_application(role: str = "owner") -> Optional[Application]:
    """Return the running Application instance for the given role."""
    return _applications.get(str(role))


# ─────────────────────────────────────────────────────────────────────────────
# Authorization decorator
# ─────────────────────────────────────────────────────────────────────────────

async def _authorize_viewer_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    role = _app_role(context)
    if role != "viewer":
        return False

    user = update.effective_user
    if user is None:
        return False

    viewer_channel_id = context.application.bot_data.get("viewer_channel_id")
    if viewer_channel_id is None:
        return False

    try:
        member = await context.application.bot.get_chat_member(
            chat_id=int(viewer_channel_id),
            user_id=int(user.id),
        )
    except Exception as exc:
        logger.warning("Viewer membership check failed for %s: %s", user.id, exc)
        return False

    if getattr(member, "status", "") in {"member", "administrator", "creator"}:
        allowed_chat_ids = set(context.application.bot_data.get("allowed_chat_ids") or set())
        allowed_chat_ids.add(int(update.effective_chat.id))
        context.application.bot_data["allowed_chat_ids"] = allowed_chat_ids
        logger.info(
            "Auto-authorized viewer chat_id=%s via channel %s membership",
            update.effective_chat.id,
            viewer_channel_id,
        )
        return True
    return False


def authorized(func):
    """Decorator: only allow the configured chat ID."""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        allowed_chat_ids = set(context.application.bot_data.get("allowed_chat_ids") or set())
        if chat_id not in allowed_chat_ids:
            if not await _authorize_viewer_member(update, context):
                await update.effective_message.reply_text("Unauthorized.")
                return
        return await func(update, context)
    return wrapper


def authorized_callback(func):
    """Decorator for CallbackQueryHandlers."""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        allowed_chat_ids = set(context.application.bot_data.get("allowed_chat_ids") or set())
        if chat_id not in allowed_chat_ids:
            if not await _authorize_viewer_member(update, context):
                await update.callback_query.answer("Unauthorized.")
                return
        return await func(update, context)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_npr(value: float) -> str:
    """Format NPR value with commas."""
    return f"NPR {value:,.0f}"


def _fmt_npr_precise(value: Optional[float]) -> str:
    if value is None:
        return "NPR --"
    return f"NPR {float(value):,.2f}"


def _pnl_emoji(value: float) -> str:
    return "✅" if value >= 0 else "❌"


def _format_mark_source(source: Optional[str]) -> Optional[str]:
    if not source:
        return None
    mapping = {
        "nepalstock": "Nepalstock",
        "sqlite_cache": "SQLite cache",
        "merolagani": "MeroLagani fallback",
        "manual": "Manual entry",
        "startup": "Startup",
    }
    return mapping.get(str(source).strip().lower(), str(source))


def _parse_snapshot_utc_to_nst(raw: Optional[str]) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=5, minutes=45)


def _latest_mark_nst(trader: "LiveTrader") -> Optional[datetime]:
    last_refresh = getattr(trader, "last_refresh", None)
    snapshot_nst = _parse_snapshot_utc_to_nst(getattr(trader, "last_price_snapshot_time_utc", None))
    if snapshot_nst and last_refresh:
        return snapshot_nst if snapshot_nst >= last_refresh else last_refresh
    return snapshot_nst or last_refresh


def _confidence_label(score: int) -> str:
    if score >= 85:
        return "High"
    if score >= 65:
        return "Medium"
    return "Low"


def _parse_chat_ids(raw: Optional[str]) -> set[int]:
    values: set[int] = set()
    if not raw:
        return values
    for chunk in str(raw).replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            logger.warning("Ignoring invalid Telegram chat id: %s", item)
    return values


def _get_trader() -> Optional["LiveTrader"]:
    return _trader


def _app_role(context: ContextTypes.DEFAULT_TYPE) -> str:
    return str(context.application.bot_data.get("role") or "owner")


def _format_refresh_meta(trader: "LiveTrader") -> list[str]:
    from backend.trading.live_trader import now_nst

    lines: list[str] = []
    latest_mark = _latest_mark_nst(trader)
    if latest_mark:
        delta_secs = max(0, int((now_nst() - latest_mark).total_seconds()))
        mins, _ = divmod(delta_secs, 60)
        lines.append(f"Marks: {latest_mark.strftime('%b %d %H:%M')} NST")
        lines.append(f"  ({mins}m ago)")
    if trader.last_price_source_label or trader.last_price_source_detail:
        label = trader.last_price_source_label or "unknown"
        detail = (trader.last_price_source_detail or "").strip()
        src = f"{label} {detail}".strip()
        lines.append(f"Src: {src[:24]}")
    if trader.last_price_snapshot_time_utc:
        snap = str(trader.last_price_snapshot_time_utc)[:16]
        lines.append(f"UTC: {snap}")
    return lines


async def _maybe_refresh_portfolio_prices(trader: "LiveTrader", *, max_age_secs: int = 90) -> None:
    from backend.trading.live_trader import is_market_open, now_nst

    if not hasattr(trader, "refresh_prices") or not hasattr(trader, "_state_lock"):
        return
    if not is_market_open():
        return
    last_refresh = getattr(trader, "last_refresh", None)
    if last_refresh is not None:
        age_secs = max(0, int((now_nst() - last_refresh).total_seconds()))
        if age_secs <= max_age_secs:
            return
    await asyncio.to_thread(_refresh_prices_locked, trader)


def _refresh_prices_locked(trader: "LiveTrader") -> None:
    with trader._state_lock:
        trader.refresh_prices()


def _parse_int(raw: str) -> Optional[int]:
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _parse_float(raw: str) -> Optional[float]:
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _build_live_preview_keyboard(intent_id: str, confirm_label: str = "Confirm") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(confirm_label, callback_data=f"live_confirm_{intent_id}"),
            InlineKeyboardButton("Cancel", callback_data=f"live_cancel_{intent_id}"),
        ],
    ])


def _build_owner_help_message() -> str:
    return (
        "<b>OWNER COMMANDS</b>\n\n"
        "/portfolio — Ranked current holdings\n"
        "/alpha — Strategy vs NEPSE dashboard\n"
        "/risk — Concentration and drawdown view\n"
        "/health — Data quality and benchmark alignment\n"
        "/daily — What changed today\n"
        "/attribution — Selection, timing, drag breakdown\n"
        "/signals — Today’s buy candidates\n"
        "/trades — Recent trade log\n"
        "/tms_status — Browser session and live status\n"
        "/tms_account — Clean TMS account summary\n"
        "/tms_funds — Collateral, refund/load, payable/receivable\n"
        "/tms_holdings — DP holdings snapshot\n"
        "/tms_trades — Daily and historic trade book snapshot\n"
        "/tms_health — Session, last sync, selector health\n"
        "/status — Market and system state\n"
        "/buy SYMBOL SHARES [PRICE] — Manual buy\n"
        "/sell SYMBOL SHARES|all [PRICE] — Manual sell\n"
        "/refresh — Force price refresh\n"
        "/summary — Force daily summary\n"
        "/short — Short-term sleeve view\n"
        "/calendar — Corporate action calendar\n"
        "/help — This message"
    )


def _build_viewer_help_message() -> str:
    return (
        "<b>VIEWER COMMANDS</b>\n\n"
        "/portfolio — Current holdings and top movers\n"
        "/performance — Growth vs NEPSE\n"
        "/daily — Today’s change summary\n"
        "/trades — Recent buy and sell activity\n"
        "/help — This message"
    )


def _build_owner_trades_message(trader: "LiveTrader", limit: int = 10) -> str:
    from backend.trading.live_trader import load_trade_log

    trades = load_trade_log(trader.trade_log_file, limit=limit)
    if not trades:
        return "<b>RECENT TRADES</b>\n\nNo trades yet."

    lines = ["<b>RECENT TRADES</b>", ""]
    for t in reversed(trades):
        lines.append(
            format_trade_activity_html(
                date=t.date,
                action=t.action,
                symbol=t.symbol,
                shares=t.shares,
                price=t.price,
                pnl=t.pnl if t.action == "SELL" else None,
            )
        )
    return "\n".join(lines)


def _build_owner_status_message(trader: "LiveTrader") -> str:
    from backend.trading.live_trader import is_market_open, now_nst

    nst = now_nst()
    market = "OPEN" if is_market_open() else "CLOSED"
    latest_mark = _latest_mark_nst(trader)
    last_refresh = latest_mark.strftime("%H:%M") if latest_mark else "--"

    with trader._state_lock:
        n_pos = len(trader.positions)
        regime = trader.regime
        n_sig = len(trader.signals_today)
        cash = trader.cash

    msg = (
        f"<b>STATUS</b>\n\n"
        f"<code>"
        f"  Time         : {nst.strftime('%Y-%m-%d %H:%M')} NST\n"
        f"  Market       : {market}\n"
        f"  Regime       : {regime.upper()}\n"
        f"  Last Refresh : {last_refresh}\n"
        f"  Positions    : {n_pos}\n"
        f"  Signals      : {n_sig}\n"
        f"  Cash         : {_fmt_npr(cash)}"
        f"</code>"
    )
    benchmark_lines = _build_benchmark_lines(trader)
    if benchmark_lines:
        msg += "\n\n" + "\n".join(benchmark_lines)
    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        msg += "\n\n" + "\n".join(f"<code>  {line}</code>" for line in refresh_lines)
    return msg


def _build_live_mode_message(trader: "LiveTrader") -> str:
    summary = trader.live_mode_summary()
    lines = [
        "<b>LIVE MODE</b>",
        "",
        "<code>",
        f"  Enabled            : {'yes' if summary['enabled'] else 'no'}",
        f"  Mode               : {summary['mode']}",
        f"  Strategy Automation: {'on' if summary['strategy_automation'] else 'off'}",
        f"  Auto Exits         : {'on' if summary['auto_exits'] else 'off'}",
        f"  Owner Confirm      : {'on' if summary['owner_confirm_required'] else 'off'}",
        f"  Halt Level         : {summary['halt_level']}",
        "</code>",
    ]
    if summary.get("freeze_reason"):
        lines.extend(["", f"<code>  Freeze Reason: {summary['freeze_reason']}</code>"])
    return "\n".join(lines)


def _build_tms_status_message(trader: "LiveTrader", *, force: bool = False) -> str:
    status = trader.live_session_status(force=force)
    lines = [
        "<b>TMS STATUS</b>",
        "",
        "<code>",
        f"  Enabled        : {'yes' if status.get('enabled') else 'no'}",
        f"  Session Ready  : {'yes' if status.get('ready') else 'no'}",
        f"  Login Required : {'yes' if status.get('login_required') else 'no'}",
        f"  Halt Level     : {status.get('halt_level') or 'none'}",
    ]
    if status.get("freeze_reason"):
        lines.append(f"  Freeze Reason  : {status['freeze_reason']}")
    if status.get("detail"):
        lines.append(f"  Detail         : {status['detail']}")
    if status.get("url"):
        lines.append(f"  URL            : {status['url'][:80]}")
    lines.append("</code>")
    return "\n".join(lines)


def _build_tms_account_message(trader: "LiveTrader", *, force: bool = False) -> str:
    account = trader.get_tms_snapshot("account", force=force) or {}
    health = trader.get_tms_health_summary(force=False)
    if not account:
        return "<b>TMS ACCOUNT</b>\n\nNo account snapshot available yet."
    trade = account.get("trade_summary") or {}
    collateral = account.get("collateral_summary") or {}
    dp = account.get("dp_holding_summary") or {}
    lines = [
        "<b>TMS ACCOUNT</b>",
        "",
        "<b>TRADE SUMMARY</b>",
        "<code>",
        f"  Turnover      : {_fmt_npr_precise(trade.get('total_turnover'))}",
        f"  Traded Shares : {int(trade.get('traded_shares') or 0)}",
        f"  Transactions  : {int(trade.get('transactions') or 0)}",
        f"  Scrips        : {int(trade.get('scrips_traded') or 0)}",
        f"  Buy Count     : {int(trade.get('buy_count') or 0)}",
        f"  Sell Count    : {int(trade.get('sell_count') or 0)}",
        "</code>",
        "",
        "<b>COLLATERAL</b>",
        "<code>",
        f"  Total         : {_fmt_npr_precise(collateral.get('collateral_amount'))}",
        f"  Utilized      : {_fmt_npr_precise(collateral.get('collateral_utilized'))}",
        f"  Available     : {_fmt_npr_precise(collateral.get('collateral_available'))}",
        f"  Payable       : {_fmt_npr_precise(collateral.get('payable_amount'))}",
        f"  Receivable    : {_fmt_npr_precise(collateral.get('receivable_amount'))}",
        "</code>",
        "",
        "<b>DP HOLDING</b>",
        "<code>",
        f"  Count         : {int(dp.get('holdings_count') or 0)}",
        f"  Total Amount  : {_fmt_npr_precise(dp.get('total_amount_cp'))}",
        f"  Last Sync     : {dp.get('last_sync') or '--'}",
        "</code>",
    ]
    if health.get("last_sync_utc"):
        lines.extend(["", f"<code>  UTC: {str(health.get('last_sync_utc') or '')[:16]}</code>"])
    return "\n".join(lines)


def _build_tms_funds_message(trader: "LiveTrader", *, force: bool = False) -> str:
    funds = trader.get_tms_snapshot("funds", force=force) or {}
    if not funds:
        return "<b>TMS FUNDS</b>\n\nNo fund snapshot available yet."
    tx_rows = funds.get("recent_transactions") or []
    lines = [
        "<b>TMS FUNDS</b>",
        "",
        "<b>COLLATERAL</b>",
        "<code>",
        f"  Total         : {_fmt_npr_precise(funds.get('collateral_total'))}",
        f"  Utilized      : {_fmt_npr_precise(funds.get('collateral_utilized'))}",
        f"  Available     : {_fmt_npr_precise(funds.get('collateral_available'))}",
        f"  Fund Transfer : {_fmt_npr_precise(funds.get('fund_transfer_amount'))}",
        f"  Cash Collat   : {_fmt_npr_precise(funds.get('cash_collateral_amount'))}",
        "</code>",
        "",
        "<b>REFUND / LOAD</b>",
        "<code>",
        f"  Max Refund    : {_fmt_npr_precise(funds.get('max_refund_allowed'))}",
        f"  Pending Ref   : {_fmt_npr_precise(funds.get('pending_refund_request'))}",
        f"  Trade Limit   : {_fmt_npr_precise(funds.get('available_trading_limit'))}",
        f"  Utilized TL   : {_fmt_npr_precise(funds.get('utilized_trading_limit'))}",
        "</code>",
        "",
        "<b>SETTLEMENT</b>",
        "<code>",
        f"  Payable       : {_fmt_npr_precise(funds.get('payable_amount'))}",
        f"  Receivable    : {_fmt_npr_precise(funds.get('receivable_amount'))}",
        "</code>",
    ]
    if tx_rows:
        lines.extend(["", "<b>RECENT COLLATERAL ACTIVITY</b>"])
        for row in tx_rows[:5]:
            label = row.get("particular") or row.get("description") or row.get("remarks") or "activity"
            amount = row.get("amount") or row.get("credit") or row.get("debit") or "-"
            date = row.get("date") or row.get("transaction_date") or row.get("created_at") or ""
            short_label = str(label)[:18]
            lines.append(f"<code>  {str(date)[:10]}  {short_label:<18}</code>")
            lines.append(f"<code>    {amount}</code>")
    if funds.get("snapshot_time_utc"):
        lines.extend(["", f"<code>  UTC: {str(funds.get('snapshot_time_utc') or '')[:16]}</code>"])
    return "\n".join(lines)


def _build_tms_holdings_message(trader: "LiveTrader", *, force: bool = False) -> str:
    holdings = trader.get_tms_snapshot("holdings", force=force) or {}
    if not holdings:
        return "<b>TMS HOLDINGS</b>\n\nNo holdings snapshot available yet."
    items = holdings.get("items") or []
    lines = [
        "<b>TMS HOLDINGS</b>",
        "",
        "<code>",
        f"  Count         : {int(holdings.get('count') or 0)}",
        f"  Total @ CP    : {_fmt_npr_precise(holdings.get('total_amount_cp'))}",
        f"  Total @ LTP   : {_fmt_npr_precise(holdings.get('total_amount_ltp'))}",
        f"  Last Sync     : {holdings.get('last_sync') or '--'}",
        "</code>",
    ]
    if items:
        lines.extend(["", "<b>POSITIONS</b>"])
        for row in items[:8]:
            sym = str(row.get('symbol') or '-')[:7]
            bal = int(row.get('tms_balance') or 0)
            ltp = float(row.get('ltp') or 0.0)
            lines.append(f"<code>  {sym:<7} {bal:>4}sh @ {ltp:,.0f}</code>")
            val = _fmt_npr_precise(row.get('value_as_of_ltp'))
            lines.append(f"<code>    Val: {val}</code>")
    else:
        lines.extend(["", "<code>  No holdings rows found.</code>"])
    return "\n".join(lines)


    daily = trader.get_tms_snapshot("orders_daily", force=force) or {}
    historic = trader.get_tms_snapshot("orders_historic", force=False) or {}
    lines = [
        "<b>TMS ORDERS</b>",
        "",
        "<b>DAILY ORDER BOOK</b>",
        f"<code>  Rows          : {int(daily.get('row_count') or 0)}</code>",
    ]
    for row in (daily.get("records") or [])[:5]:
        symbol = row.get("symbol") or row.get("scrip") or row.get("scrip_name") or "-"
        qty = row.get("qty") or row.get("quantity") or row.get("order_quantity") or "-"
        price = row.get("price") or row.get("rate") or row.get("limit_price") or "-"
        status = row.get("status") or row.get("order_status") or row.get("buy_sell") or "-"
        lines.append(f"<code>  {str(symbol)[:7]:<7} {str(qty)[:5]:<5} @ {str(price)[:8]}</code>")
        lines.append(f"<code>    {str(status)[:26]}</code>")
    if not (daily.get("records") or []):
        lines.append("<code>  No daily orders.</code>")
    lines.extend([
        "",
        "<b>HISTORIC ORDER BOOK</b>",
        f"<code>  Rows          : {int(historic.get('row_count') or 0)}</code>",
    ])
    for row in (historic.get("records") or [])[:3]:
        symbol = row.get("symbol") or row.get("scrip") or row.get("scrip_name") or "-"
        qty = row.get("qty") or row.get("quantity") or row.get("order_quantity") or "-"
        status = row.get("status") or row.get("order_status") or row.get("buy_sell") or "-"
        date = row.get("date") or row.get("order_date") or row.get("business_date") or ""
        lines.append(f"<code>  {str(date)[:10]}  {str(symbol)[:7]:<7}  {str(qty)[:5]}</code>")
        lines.append(f"<code>    {str(status)[:26]}</code>")
    if daily.get("snapshot_time_utc") or historic.get("snapshot_time_utc"):
        snap = str(daily.get('snapshot_time_utc') or historic.get('snapshot_time_utc') or '')[:16]
        lines.extend(["", f"<code>  UTC: {snap}</code>"])
    return "\n".join(lines)


def _build_tms_trades_message(trader: "LiveTrader", *, force: bool = False) -> str:
    daily = trader.get_tms_snapshot("trades_daily", force=force) or {}
    historic = trader.get_tms_snapshot("trades_historic", force=False) or {}
    lines = [
        "<b>TMS TRADES</b>",
        "",
        "<b>DAILY TRADE BOOK</b>",
        f"<code>  Rows          : {int(daily.get('row_count') or 0)}</code>",
    ]
    for row in (daily.get("records") or [])[:5]:
        symbol = row.get("symbol") or row.get("scrip") or row.get("scrip_name") or "-"
        qty = row.get("qty") or row.get("quantity") or row.get("trade_qty") or "-"
        price = row.get("price") or row.get("rate") or row.get("trade_price") or "-"
        side = row.get("buy_sell") or row.get("type") or row.get("status") or "-"
        lines.append(f"<code>  {str(symbol)[:8]:<8} {str(side)[:6]:<6} {str(qty)[:6]:<6} @ {str(price)[:10]:<10}</code>")
    if not (daily.get("records") or []):
        lines.append("<code>  No daily trades.</code>")
    lines.extend([
        "",
        "<b>HISTORIC TRADE BOOK</b>",
        f"<code>  Rows          : {int(historic.get('row_count') or 0)}</code>",
    ])
    for row in (historic.get("records") or [])[:3]:
        symbol = row.get("symbol") or row.get("scrip") or row.get("scrip_name") or "-"
        date = row.get("trade_date") or row.get("date") or row.get("business_date") or ""
        qty = row.get("qty") or row.get("quantity") or row.get("trade_qty") or "-"
        lines.append(f"<code>  {str(date)[:12]:<12} {str(symbol)[:8]:<8} Qty {str(qty)[:6]:<6}</code>")
    if daily.get("snapshot_time_utc") or historic.get("snapshot_time_utc"):
        snap = str(daily.get('snapshot_time_utc') or historic.get('snapshot_time_utc') or '')[:16]
        lines.extend(["", f"<code>  UTC: {snap}</code>"])
    return "\n".join(lines)


def _build_tms_health_message(trader: "LiveTrader", *, force: bool = False) -> str:
    health = trader.get_tms_health_summary(force=force)
    selector_health = health.get("selector_health") or {}
    lines = [
        "<b>TMS HEALTH</b>",
        "",
        "<code>",
        f"  Monitor Enabled: {'yes' if health.get('enabled') else 'no'}",
        f"  Session Ready  : {'yes' if health.get('ready') else 'no'}",
        f"  Login Required : {'yes' if health.get('login_required') else 'no'}",
        f"  Selector Health: {float(health.get('selector_health_pct') or 0.0):.1f}%",
        f"  Last Sync UTC  : {health.get('last_sync_utc') or '--'}",
        "</code>",
    ]
    if health.get("detail"):
        lines.extend(["", f"<code>  Detail         : {str(health.get('detail'))[:120]}</code>"])
    if selector_health:
        lines.extend(["", "<b>PAGE CHECKS</b>"])
        for key, ok in selector_health.items():
            status = "OK" if ok else "MISS"
            lines.append(f"<code>  {str(key).replace('_', ' ').title():<16} {status}</code>")
    return "\n".join(lines)


def _build_live_orders_message(trader: "LiveTrader", *, limit: int = 10) -> str:
    rows = trader.list_live_orders(limit=limit)
    if not rows:
        return "<b>LIVE ORDERS</b>\n\nNo broker order states recorded yet."
    lines = ["<b>LIVE ORDERS</b>", ""]
    for row in rows[:limit]:
        sym = str(row.get('symbol') or '-')[:7]
        action = str(row.get('action') or '-').upper()[:4]
        qty = int(row.get('quantity') or 0)
        price = float(row.get('price') or 0.0)
        ts = str(row.get('recorded_at') or '')[:16]
        lines.append(f"<code>  {sym:<7} {action:<4} {qty:>4} @ {price:,.0f}</code>")
        lines.append(f"<code>  {ts}</code>")
        status = str(row.get('status_text') or row.get('fill_state') or '-')[:24]
        ref = str(row.get('broker_order_ref') or '-')[:12]
        lines.append(f"<code>  {status}  #{ref}</code>")
    return "\n".join(lines)


def _build_live_positions_message(trader: "LiveTrader", *, limit: int = 20) -> str:
    rows = trader.list_live_positions(limit=limit)
    if not rows:
        return "<b>LIVE POSITIONS</b>\n\nNo broker positions recorded yet."
    lines = ["<b>LIVE POSITIONS</b>", ""]
    for row in rows[:limit]:
        sym = str(row.get('symbol') or '-')[:7]
        qty = int(row.get('quantity') or 0)
        avg = float(row.get('average_price') or 0.0)
        mv = float(row.get('market_value') or 0.0)
        lines.append(f"<code>  {sym:<7} {qty:>4}sh @ {avg:,.0f}</code>")
        lines.append(f"<code>    MV: {_fmt_npr(mv)}</code>")
    return "\n".join(lines)


def _build_benchmark_lines(trader: "LiveTrader") -> list[str]:
    """Build deployed-capital vs NEPSE comparison lines for Telegram views."""
    from backend.trading.live_trader import compute_portfolio_vs_nepse

    comparison = compute_portfolio_vs_nepse(
        trader.capital,
        trader.positions,
        trader.trade_log_file,
        cash=trader.cash,
        nav_log_path=trader.nav_log_file,
    )
    if not comparison:
        return []

    perf = comparison["performance"]
    lines = [
        "",
        "<b>DEPLOYED VS NEPSE</b>",
        (
            f"  Strategy ({perf['start_date']}): "
            f"{perf['deployed_return_pct']:+.2f}%"
        ),
    ]
    benchmark = comparison.get("benchmark")
    quality = comparison.get("data_quality") or {}
    if benchmark:
        alpha = comparison.get("alpha_pct")
        lines.append(
            f"  NEPSE ({benchmark['base_date']}): {benchmark['return_pct']:+.2f}%"
        )
        if quality.get("freeze_alpha"):
            lines.append("  Alpha: frozen (benchmark mismatch)")
        elif alpha is not None:
            lines.append(f"  Alpha: {alpha:+.2f} pts")
    else:
        lines.append("  NEPSE: unavailable")

    lines.append(f"  Open Book: {perf['open_return_pct']:+.2f}%")
    return lines


def _build_alpha_message(trader: "LiveTrader") -> str:
    from backend.trading.live_trader import (
        compute_portfolio_intelligence,
    )

    with trader._state_lock:
        positions = dict(trader.positions)
        cash = trader.cash

    intelligence = compute_portfolio_intelligence(
        trader.capital,
        cash,
        positions,
        trader.trade_log_file,
        trader.nav_log_file,
    )
    if not intelligence:
        return "<b>ALPHA VIEW</b>\n\nNo benchmark data available yet."

    perf = intelligence["performance"]
    benchmark = intelligence.get("global_benchmark")
    quality = intelligence["data_quality"]
    attribution = intelligence["attribution_stack"]
    position_rows = intelligence["positions"]
    gross_alpha_line = "frozen" if quality.get("freeze_alpha") else f"{attribution['gross_alpha_return_pct']:+.2f} pts"

    lines = [
        "<b>ALPHA VIEW</b>",
        "",
    ]
    if quality.get("freeze_alpha"):
        lines.extend(
            [
                "<b>ALPHA STATUS</b>",
                "<code>  FROZEN — benchmark snapshots are not aligned</code>",
                "",
            ]
        )

    lines.extend([
        "<b>SUMMARY</b>",
        "<code>",
        f"  Strategy Return : {perf['deployed_return_pct']:+.2f}%",
        f"  Gross Alpha     : {gross_alpha_line}",
        f"  Net Alpha       : {_fmt_npr(attribution['net_alpha_pnl'])}",
        f"  Confidence      : {quality['confidence_score']}/100",
    ])
    if benchmark:
        lines.extend(
            [
                f"  NEPSE Return    : {benchmark['return_pct']:+.2f}%",
                f"  Benchmark Date  : {quality.get('common_benchmark_date') or 'unknown'}",
            ]
        )
    lines.append("</code>")

    lines.extend(
        [
            "",
            "<b>PROFIT</b>",
            "<code>",
            f"  Realized P&L    : {_fmt_npr(perf['realized_pnl'])}",
            f"  Open P&L        : {_fmt_npr(perf['open_unrealized_pnl'])}",
            f"  Open Book       : {perf['open_return_pct']:+.2f}%",
            "</code>",
            "",
            "<b>ALPHA BREAKDOWN</b>",
            "<code>",
            f"  Gross Alpha     : {_fmt_npr(attribution['gross_alpha_pnl'])}",
            f"  Stock Select    : {_fmt_npr(attribution['stock_selection_alpha_pnl'])}",
            f"  Sector Alloc    : {_fmt_npr(attribution['sector_allocation_alpha_pnl'])}",
            f"  Timing          : {_fmt_npr(attribution['timing_alpha_pnl'])}",
            f"  Cash Drag       : {_fmt_npr(attribution['cash_drag_pnl'])}",
            f"  Fees + Turnover : {_fmt_npr(attribution['alpha_cost_drag_pnl'])}",
            "</code>",
            "",
            "<b>ACTIVE NAMES</b>",
        ]
    )

    if position_rows:
        lines.append("<code>")
        lines.append(f"  {'SYM':<6}  {'α PTS':>7}  {'vs IDX':>7}")
        lines.append(f"  {'─'*6}  {'─'*7}  {'─'*7}")
        for row in position_rows[:4]:
            sym = row["symbol"][:6]
            pts = f"{row['contribution_vs_nepse_pts']:+.2f}p"
            vs  = f"{row['active_vs_nepse_pct']:+.1f}%"
            lines.append(f"  {sym:<6}  {pts:>7}  {vs:>7}")
        lines.append("</code>")
    else:
        lines.append("<code>  No active contribution data yet.</code>")

    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        lines.append("")
        lines.append("<b>MARK DATA</b>")
        for rl in refresh_lines:
            lines.append(f"<code>{rl}</code>")
    alerts = quality.get("alerts", [])[:3]
    if alerts:
        import textwrap
        for alert in alerts:
            wrapped = textwrap.wrap(alert, width=26)
            if wrapped:
                lines.append(f"<code>  ⚠ {wrapped[0]}</code>")
                for chunk in wrapped[1:]:
                    lines.append(f"<code>    {chunk}</code>")

    return "\n".join(lines)


def _render_alpha_chart_image(chart: dict) -> Optional[BytesIO]:
    if not chart:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        logger.warning("Matplotlib unavailable for alpha chart render: %s", exc)
        return None

    sleeve = chart.get("sleeve_index") or []
    benchmark = chart.get("benchmark_index") or []
    dates = chart.get("dates") or []
    if len(sleeve) < 2 or len(benchmark) < 2:
        return None

    x = list(range(len(sleeve)))
    bg = "#223447"
    fg = "#EAF2F8"
    grid = "#3C556B"
    sleeve_color = "#38D27A"
    bench_color = "#D5DEE8"
    pos_fill = "#2ECC71"
    neg_fill = "#E74C3C"

    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=180)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(x, sleeve, color=sleeve_color, linewidth=3.0, solid_capstyle="round", zorder=4)
    ax.plot(x, benchmark, color=bench_color, linewidth=2.0, linestyle=(0, (4, 2)), alpha=0.95, zorder=3)

    ax.fill_between(
        x, sleeve, benchmark,
        where=[a >= b for a, b in zip(sleeve, benchmark)],
        interpolate=True,
        color=pos_fill,
        alpha=0.08,
        zorder=1,
    )
    ax.fill_between(
        x, sleeve, benchmark,
        where=[a < b for a, b in zip(sleeve, benchmark)],
        interpolate=True,
        color=neg_fill,
        alpha=0.06,
        zorder=1,
    )

    ax.scatter(x[-1], sleeve[-1], s=34, color=sleeve_color, edgecolor=bg, linewidth=0.8, zorder=6)
    ax.scatter(x[-1], benchmark[-1], s=30, color=bench_color, edgecolor=bg, linewidth=0.8, zorder=5)

    ax.set_title("Deployed NAV vs NEPSE", color=fg, fontsize=15, weight="bold", loc="left", pad=12)
    ax.text(0.0, 1.02, "Indexed to 100 from strategy start", transform=ax.transAxes, color="#AFC0CF", fontsize=9.5)

    tick_positions = sorted(set({0, len(x) // 2, len(x) - 1}))
    tick_labels = []
    for pos in tick_positions:
        try:
            tick_labels.append(dates[pos].strftime("%b %d"))
        except Exception:
            tick_labels.append(str(pos))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, color="#BFD0DE", fontsize=9)

    y_min = min(min(sleeve), min(benchmark)) - 1.6
    y_max = max(max(sleeve), max(benchmark)) + 1.8
    ax.set_ylim(y_min, y_max)
    label_gap = max((y_max - y_min) * 0.05, 1.4)
    sleeve_label_y = sleeve[-1]
    benchmark_label_y = benchmark[-1]
    if abs(sleeve_label_y - benchmark_label_y) < label_gap:
        midpoint = (sleeve_label_y + benchmark_label_y) / 2.0
        sleeve_label_y = midpoint + (label_gap / 2.0)
        benchmark_label_y = midpoint - (label_gap / 2.0)
    sleeve_label_y = min(y_max - 0.7, max(y_min + 0.7, sleeve_label_y))
    benchmark_label_y = min(y_max - 0.7, max(y_min + 0.7, benchmark_label_y))
    label_x = x[-1] + 0.85
    ax.set_xlim(-0.4, label_x + 1.1)
    for actual_y, label_y, text, color in [
        (sleeve[-1], sleeve_label_y, f"Sleeve {sleeve[-1]:.1f}", sleeve_color),
        (benchmark[-1], benchmark_label_y, f"NEPSE {benchmark[-1]:.1f}", bench_color),
    ]:
        ax.plot(
            [x[-1] + 0.06, label_x - 0.08],
            [actual_y, label_y],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            solid_capstyle="round",
            clip_on=False,
            zorder=6,
        )
        ax.text(
            label_x,
            label_y,
            text,
            color=color,
            fontsize=9.4,
            weight="bold" if "Sleeve" in text else None,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.2", facecolor="#1B2A38", edgecolor=color, linewidth=0.8),
            clip_on=False,
            zorder=7,
        )

    ax.tick_params(axis="y", colors="#BFD0DE", labelsize=9)
    ax.tick_params(axis="x", colors="#BFD0DE", labelsize=9)
    ax.grid(axis="y", color=grid, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.grid(axis="x", color=grid, linestyle=":", linewidth=0.5, alpha=0.18)

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color("#5B748A")
    ax.spines["bottom"].set_color("#5B748A")

    legend_handles = [
        Line2D([0], [0], color=sleeve_color, lw=3.0, label="Sleeve"),
        Line2D([0], [0], color=bench_color, lw=2.0, linestyle=(0, (4, 2)), label="NEPSE"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.19),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_color("#C8D6E3")

    plt.tight_layout(rect=(0, 0.03, 1, 0.98))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    buffer.name = "alpha_chart.png"
    return buffer


def _render_alpha_dashboard_image(trader: "LiveTrader") -> Optional[BytesIO]:
    from backend.trading.live_trader import (
        compute_deployed_nav_chart_data,
        compute_portfolio_intelligence,
    )

    with trader._state_lock:
        positions = dict(trader.positions)
        cash = trader.cash

    intelligence = compute_portfolio_intelligence(
        trader.capital,
        cash,
        positions,
        trader.trade_log_file,
        trader.nav_log_file,
    )
    if not intelligence:
        return None

    chart = compute_deployed_nav_chart_data(
        trader.capital,
        positions,
        trader.trade_log_file,
        trader.nav_log_file,
    )
    if not chart:
        return None

    perf = intelligence["performance"]
    benchmark = intelligence.get("global_benchmark")
    quality = intelligence["data_quality"]
    attribution = intelligence["attribution_stack"]
    position_rows = intelligence["positions"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:
        logger.warning("Matplotlib unavailable for alpha dashboard render: %s", exc)
        return None

    sleeve = chart.get("sleeve_index") or []
    benchmark_values = chart.get("benchmark_index") or []
    dates = chart.get("dates") or []
    if len(sleeve) < 2 or len(benchmark_values) < 2:
        return None

    x = list(range(len(sleeve)))

    # ── Palette ──────────────────────────────────────────────────────────────
    bg       = "#0F1923"
    card_bg  = "#172130"
    card_alt = "#1D2B3A"
    chart_bg = "#141E2A"
    fg       = "#E8F0F8"
    muted    = "#7B9AB5"
    border   = "#243347"
    green    = "#34D17A"
    red      = "#F26355"
    amber    = "#F5C842"
    blue     = "#4A9EE8"
    # ─────────────────────────────────────────────────────────────────────────

    def _color(v, *, frozen=False):
        if frozen: return amber
        if v is None: return fg
        return green if v >= 0 else red

    alpha_frozen  = bool(quality.get("freeze_alpha"))
    status_label  = "FROZEN" if alpha_frozen else "LIVE"
    status_color  = amber if alpha_frozen else green
    confidence    = quality.get("confidence_score", 0)

    strat_ret  = perf.get("deployed_return_pct", 0.0)
    bench_ret  = benchmark.get("return_pct", 0.0) if benchmark else None
    gross_pts  = attribution.get("gross_alpha_return_pct")
    net_pnl    = attribution.get("net_alpha_pnl", 0.0)
    open_pnl   = perf.get("open_unrealized_pnl", 0.0)
    fee_drag   = attribution.get("fee_drag_pnl", 0.0)
    turn_drag  = attribution.get("turnover_drag_pnl", 0.0)
    bench_date = quality.get("common_benchmark_date") or "—"

    # ── Layout: 5 rows — title | cards | chart | drivers | footer ───────────
    fig = plt.figure(figsize=(9.0, 9.0), dpi=160)
    fig.patch.set_facecolor(bg)
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[0.38, 0.72, 3.2, 1.85, 0.22],
        hspace=0.14,
    )
    title_ax   = fig.add_subplot(gs[0])
    cards_ax   = fig.add_subplot(gs[1])
    chart_ax   = fig.add_subplot(gs[2])
    bottom_ax  = fig.add_subplot(gs[3])
    footer_ax  = fig.add_subplot(gs[4])

    for ax in (title_ax, cards_ax, bottom_ax, footer_ax):
        ax.set_facecolor(bg)
        ax.axis("off")
    chart_ax.set_facecolor(chart_bg)

    # ── Helper: draw a metric card ────────────────────────────────────────
    def _card(ax, x0, y0, w, h, label, value, tone=fg, sub=None):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            linewidth=0.8, edgecolor=border, facecolor=card_bg,
            transform=ax.transAxes, zorder=2,
        ))
        # Label (top of card)
        ax.text(x0 + 0.018, y0 + h - 0.08,
                label.upper(), transform=ax.transAxes,
                color=muted, fontsize=7.5, weight="bold", va="top", zorder=3)
        # Value (vertical centre of card)
        ax.text(x0 + 0.018, y0 + h * 0.50,
                value, transform=ax.transAxes,
                color=tone, fontsize=15.0, weight="bold", va="center", zorder=3)
        # Sub label (bottom of card)
        if sub:
            ax.text(x0 + 0.018, y0 + 0.05,
                    sub, transform=ax.transAxes,
                    color=muted, fontsize=7.8, va="bottom", zorder=3)

    # ── Title row — only title text + subtitle + badge ────────────────────
    title_ax.text(0.00, 0.90, "ALPHA VIEW", color=fg,
                  fontsize=18, weight="bold", va="top",
                  transform=title_ax.transAxes)
    title_ax.text(0.00, 0.15, "Strategy vs NEPSE · since deployment", color=muted,
                  fontsize=9.5, va="bottom", transform=title_ax.transAxes)
    # Status badge (top-right)
    title_ax.text(
        0.998, 0.90,
        f"  {status_label}   {confidence}/100  ",
        color=status_color, fontsize=8.5, ha="right", va="top",
        transform=title_ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=card_bg,
                  edgecolor=status_color, linewidth=1.1),
    )

    # ── Cards row — 4 metric cards filling the full row ───────────────────
    cw, ch, gap = 0.2375, 0.82, 0.017
    cy = 0.09
    _card(cards_ax, 0.000,            cy, cw, ch,
          "Strategy",
          f"{strat_ret:+.2f}%",
          tone=_color(strat_ret), sub="Deployed return")
    _card(cards_ax, cw + gap,         cy, cw, ch,
          "NEPSE",
          f"{bench_ret:+.2f}%" if bench_ret is not None else "n/a",
          tone=_color(bench_ret), sub="Benchmark return")
    _card(cards_ax, (cw + gap) * 2,   cy, cw, ch,
          "Gross Alpha" if not alpha_frozen else "Alpha",
          "Frozen" if alpha_frozen else f"{gross_pts:+.2f} pts",
          tone=_color(gross_pts, frozen=alpha_frozen), sub="Before fee drag")
    _card(cards_ax, (cw + gap) * 3,   cy, cw, ch,
          "Net Alpha",
          _fmt_npr(net_pnl),
          tone=_color(net_pnl), sub="After costs")

    # ── Chart ─────────────────────────────────────────────────────────────
    chart_ax.plot(x, sleeve, color=green, linewidth=2.6,
                  solid_capstyle="round", zorder=4, label="Strategy")
    chart_ax.plot(x, benchmark_values, color=blue, linewidth=1.8,
                  linestyle=(0, (6, 3)), alpha=0.85, zorder=3, label="NEPSE")
    chart_ax.fill_between(x, sleeve, benchmark_values,
                          where=[a >= b for a, b in zip(sleeve, benchmark_values)],
                          interpolate=True, color=green, alpha=0.10, zorder=1)
    chart_ax.fill_between(x, sleeve, benchmark_values,
                          where=[a < b for a, b in zip(sleeve, benchmark_values)],
                          interpolate=True, color=red, alpha=0.07, zorder=1)
    # End dots
    chart_ax.scatter([x[-1]], [sleeve[-1]], s=55, color=green,
                     edgecolor=chart_bg, linewidth=1.4, zorder=6)
    chart_ax.scatter([x[-1]], [benchmark_values[-1]], s=40, color=blue,
                     edgecolor=chart_bg, linewidth=1.2, zorder=6)

    # X ticks — show up to 4 dates
    n = len(x)
    tick_idxs = sorted({0, n // 3, 2 * n // 3, n - 1})
    tick_labels = []
    for i in tick_idxs:
        try:    tick_labels.append(dates[i].strftime("%b %d"))
        except: tick_labels.append(str(i))
    chart_ax.set_xticks(tick_idxs)
    chart_ax.set_xticklabels(tick_labels, color=muted, fontsize=9.0)

    y_min = min(min(sleeve), min(benchmark_values)) - 2.0
    y_max = max(max(sleeve), max(benchmark_values)) + 2.5
    chart_ax.set_ylim(y_min, y_max)
    chart_ax.set_xlim(-0.5, x[-1] + 0.5)

    # End-of-line labels (right side, inside chart)
    gap_needed = max((y_max - y_min) * 0.055, 1.8)
    sy, by = sleeve[-1], benchmark_values[-1]
    if abs(sy - by) < gap_needed:
        mid = (sy + by) / 2
        sy = mid + gap_needed / 2
        by = mid - gap_needed / 2
    sy = float(min(y_max - 1.0, max(y_min + 1.0, sy)))
    by = float(min(y_max - 1.0, max(y_min + 1.0, by)))
    for actual, label_y, txt, col in [
        (sleeve[-1], sy, f"▶ {sleeve[-1]:.1f}", green),
        (benchmark_values[-1], by, f"▶ {benchmark_values[-1]:.1f}", blue),
    ]:
        chart_ax.annotate(
            txt, xy=(x[-1], actual), xytext=(x[-1] + 0.4, label_y),
            color=col, fontsize=8.8, weight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=card_alt,
                      edgecolor=col, linewidth=0.8),
            clip_on=False, zorder=7,
        )

    chart_ax.tick_params(axis="y", colors=muted, labelsize=9.0)
    chart_ax.grid(axis="y", color=border, linestyle="--", linewidth=0.6, alpha=0.5)
    chart_ax.grid(axis="x", color=border, linestyle=":", linewidth=0.4, alpha=0.2)
    for spine in ("top", "right"):
        chart_ax.spines[spine].set_visible(False)
    chart_ax.spines["left"].set_color(border)
    chart_ax.spines["bottom"].set_color(border)
    chart_ax.legend(loc="upper left", frameon=False, fontsize=9.0,
                    labelcolor=fg, handlelength=2.2, ncol=2,
                    bbox_to_anchor=(0.0, 1.01))

    # ── Bottom: 2-column grid (drivers left, costs right) ─────────────────
    # We draw everything on bottom_ax using transAxes coordinates
    divider_x = 0.50

    # — TOP DRIVERS —
    bottom_ax.text(0.00, 0.97, "TOP DRIVERS", color=fg, fontsize=11.0,
                   weight="bold", va="top", transform=bottom_ax.transAxes)
    top_rows = sorted(position_rows,
                      key=lambda r: r["contribution_vs_nepse_pts"], reverse=True)[:3]
    if top_rows:
        for i, row in enumerate(top_rows):
            y = 0.73 - i * 0.26
            pts = row["contribution_vs_nepse_pts"]
            vs  = row.get("active_vs_nepse_pct", 0.0)
            bottom_ax.text(0.00, y, row["symbol"],
                           color=fg, fontsize=11.5, weight="bold", va="center",
                           transform=bottom_ax.transAxes)
            bottom_ax.text(0.18, y, f"{pts:+.2f} pts",
                           color=_color(pts), fontsize=10.5, weight="bold", va="center",
                           transform=bottom_ax.transAxes)
            bottom_ax.text(0.36, y, f"vs NEPSE {vs:+.1f}%",
                           color=muted, fontsize=8.5, va="center",
                           transform=bottom_ax.transAxes)
    else:
        bottom_ax.text(0.00, 0.55, "No contribution data yet.",
                       color=muted, fontsize=9.5, va="center",
                       transform=bottom_ax.transAxes)

    # Vertical divider
    bottom_ax.axvline(divider_x, color=border, linewidth=0.8, ymin=0.0, ymax=1.0)

    # — COSTS & P&L —
    bottom_ax.text(divider_x + 0.03, 0.97, "COSTS & P&L", color=fg,
                   fontsize=11.0, weight="bold", va="top",
                   transform=bottom_ax.transAxes)
    cost_rows = [
        ("Open P&L",     open_pnl,  _color(open_pnl)),
        ("Net Alpha",    net_pnl,   _color(net_pnl)),
        ("Fee Drag",     fee_drag,  _color(fee_drag)),
        ("Turnover Drag",turn_drag, _color(turn_drag)),
    ]
    for i, (lbl, val, tone) in enumerate(cost_rows):
        y = 0.73 - i * 0.185
        bottom_ax.text(divider_x + 0.03, y, lbl,
                       color=muted, fontsize=9.0, va="center",
                       transform=bottom_ax.transAxes)
        bottom_ax.text(0.985, y, _fmt_npr(val),
                       color=tone, fontsize=10.5, weight="bold",
                       va="center", ha="right",
                       transform=bottom_ax.transAxes)

    # ── Footer: one-line meta ──────────────────────────────────────────────
    footer_ax.text(0.00, 0.80, f"Benchmark date: {bench_date}",
                   color=muted, fontsize=7.8, va="top",
                   transform=footer_ax.transAxes)

    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.06, right=0.97)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buffer.seek(0)
    buffer.name = "alpha_dashboard.png"
    return buffer


async def _send_alpha_view(target):
    trader = _get_trader()
    if not trader:
        await target.reply_text("Trader not initialized.")
        return

    image_buffer = _render_alpha_dashboard_image(trader)
    if image_buffer is not None:
        await target.reply_photo(
            photo=image_buffer,
            caption="<b>ALPHA VIEW</b>\nStrategy, benchmark, and cost drag in one view.",
            parse_mode="HTML",
        )
    await target.reply_text(_build_alpha_message(trader), parse_mode="HTML")


def _build_health_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_owner_report

    report = build_owner_report(trader)
    if not report:
        return "<b>HEALTH</b>\n\nNo report available."
    health = report["health"]
    quality = health["data_quality"]
    lines = [
        "<b>HEALTH</b>",
        "",
        "<code>",
        f"  Confidence    : {quality['confidence_score']}/100 ({_confidence_label(int(quality['confidence_score']))})",
        f"  Alpha Frozen  : {'YES' if quality['freeze_alpha'] else 'NO'}",
        f"  Bench Date    : {quality.get('common_benchmark_date') or 'unknown'}",
        f"  Primary Marks : {len(quality.get('primary_symbols', []))}",
        f"  Fallback Marks: {len(quality.get('fallback_symbols', []))}",
        f"  Cache Marks   : {len(quality.get('cache_symbols', []))}",
        "</code>",
    ]
    if quality.get("stale_symbols"):
        lines.append("")
        lines.append("<b>STALE MARKS</b>")
        lines.append("<code>  " + ", ".join(quality["stale_symbols"]) + "</code>")
    if quality.get("alerts"):
        lines.append("")
        lines.append("<b>ALERTS</b>")
        for alert in quality["alerts"][:6]:
            lines.append(f"<code>  {alert}</code>")
    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        lines.append("")
        lines.append("<b>MARK DATA</b>")
        lines.extend(f"<code>  {line}</code>" for line in refresh_lines)
    return "\n".join(lines)


def _build_owner_daily_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_owner_report

    report = build_owner_report(trader)
    if not report:
        return "<b>DAILY</b>\n\nNo report available."
    daily = report["daily"]
    trades_today = daily["trades_today"]
    contributor = daily.get("biggest_contributor")
    detractor = daily.get("biggest_detractor")
    snapshot_nst = daily.get("last_snapshot_nst") or daily.get("last_refresh_nst")
    mark_date = str(snapshot_nst).split(" ")[0] if snapshot_nst else daily["date"]
    portfolio_move_line = "  Portfolio Move: n/a"
    if daily.get("day_return_pct") is not None and daily.get("day_pnl") is not None:
        portfolio_move_line = (
            f"  Portfolio Move: {daily['day_return_pct']:+.2f}% ({_fmt_npr(daily['day_pnl'])})"
        )
    lines = [
        "<b>TODAY</b>",
        "",
        "<code>",
        f"  Mark Date    : {mark_date}",
        f"  Last Snapshot: {snapshot_nst} NST" if snapshot_nst else "  Last Snapshot: n/a",
        portfolio_move_line,
        f"  NAV          : {_fmt_npr(daily['nav'])}",
        f"  Invested     : {daily['invested_pct']:.1f}%",
        f"  Bench Move   : {daily['benchmark_move_pct']:+.2f}%" if daily.get("benchmark_move_pct") is not None else "  Bench Move   : n/a",
        f"  Confidence   : {daily['confidence_score']}/100",
        "</code>",
    ]
    if trades_today:
        lines.append("")
        lines.append("<b>TRADES TODAY</b>")
        for row in trades_today[:6]:
            lines.append(
                f"<code>  {row['action']:<4} {row['symbol']:<8} {row['shares']} @ {row['price']:,.1f}</code>"
            )
    if contributor:
        lines.append("")
        lines.append("<b>BIGGEST CONTRIBUTOR</b>")
        lines.append(
            f"<code>  {contributor['symbol']} {contributor['contribution_vs_nepse_pts']:+.2f} pts vs NEPSE</code>"
        )
    if detractor:
        lines.append("")
        lines.append("<b>BIGGEST DETRACTOR</b>")
        lines.append(
            f"<code>  {detractor['symbol']} {detractor['contribution_vs_nepse_pts']:+.2f} pts vs NEPSE</code>"
        )
    return "\n".join(lines)


def _build_attribution_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_owner_report

    report = build_owner_report(trader)
    if not report:
        return "<b>ATTRIBUTION</b>\n\nNo report available."
    alpha = report["alpha"]
    stack = alpha["attribution_stack"]
    positions = alpha["positions"]
    lines = [
        "<b>ATTRIBUTION</b>",
        "",
        "<code>",
        f"  Gross Alpha    : {stack['gross_alpha_pnl']:+,.0f}",
        f"  Stock Select   : {stack['stock_selection_alpha_pnl']:+,.0f}",
        f"  Sector Alloc   : {stack['sector_allocation_alpha_pnl']:+,.0f}",
        f"  Timing         : {stack['timing_alpha_pnl']:+,.0f}",
        f"  Cash Drag      : {stack['cash_drag_pnl']:+,.0f}",
        f"  Turnover Drag  : {stack['turnover_drag_pnl']:+,.0f}",
        f"  Fee Drag       : {stack['fee_drag_pnl']:+,.0f}",
        f"  Net Alpha      : {stack['net_alpha_pnl']:+,.0f}",
        f"  Realized Alpha : {stack['realized_alpha_pnl']:+,.0f} gross",
        f"  Unrealized Alp : {stack['unrealized_alpha_pnl']:+,.0f} gross",
        f"  Turnover Ratio : {stack['turnover_ratio_pct']:.1f}%",
        "</code>",
    ]
    if positions:
        lines.append("")
        lines.append("<b>NAME CONTRIBUTION</b>")
        for row in positions[:8]:
            lines.append(
                f"<code>  {row['symbol']:<8} {row['contribution_vs_nepse_pts']:+.2f} pts  vs S {row['active_vs_sector_pct']:+.1f}%  vs N {row['active_vs_nepse_pct']:+.1f}%</code>"
            )
    return "\n".join(lines)


def _build_viewer_portfolio_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_viewer_report

    report = build_viewer_report(trader)
    if not report:
        return "<b>PORTFOLIO</b>\n\nNo portfolio data available."
    summary = report["summary"]
    holdings = report["portfolio"]["holdings"]
    refresh_lines = _format_refresh_meta(trader)
    marks_line = next((line for line in refresh_lines if line.startswith("Marks:")), None)
    lines = [
        "<b>PORTFOLIO</b>",
        "",
        f"NAV: {_fmt_npr(summary['nav'])} • Open Names: {summary['open_positions']}",
        f"Total Ret: {summary['total_return_pct']:+.2f}%" if summary.get("total_return_pct") is not None else "Total Ret: n/a",
        (
            f"Realized: {_fmt_npr(summary['realized_pnl'])} ({summary['realized_return_pct']:+.2f}%)"
            if summary.get("realized_pnl") is not None and summary.get("realized_return_pct") is not None
            else "Realized: n/a"
        ),
        f"Deployed: {summary['deployed_return_pct']:+.2f}%",
        f"Today: {summary['day_return_pct']:+.2f}%" if summary.get("day_return_pct") is not None else "Today: n/a",
        f"NEPSE: {summary['benchmark_return_pct']:+.2f}%" if summary.get("benchmark_return_pct") is not None else "NEPSE: n/a",
    ]
    if marks_line:
        lines.append(marks_line)
    if holdings:
        lines.append("")
        lines.append("<b>HOLDINGS</b>")
        for row in holdings[:6]:
            extras = []
            if row.get("last_ltp") is not None:
                extras.append(f"LTP NPR {row['last_ltp']:,.1f}")
            if row.get("entry_price") is not None:
                extras.append(f"Entry NPR {row['entry_price']:,.1f}")
            extras.append(f"vs N {row['active_vs_nepse_pct']:+.1f} pts")
            lines.append(
                format_portfolio_holding_html(
                    symbol=row["symbol"],
                    direction_value=row.get("return_pct", 0.0),
                    primary_text=f"Ret {row['return_pct']:+.1f}%",
                    secondary_text=f"LTP NPR {row['last_ltp']:,.1f}" if row.get("last_ltp") is not None else None,
                    holding_days=row["holding_days"],
                    extra_metrics=extras,
                )
            )
    contributors = report["portfolio"]["contributors"]
    laggards = report["portfolio"]["laggards"]
    if contributors:
        lines.append("")
        lines.append("<b>TOP CONTRIBUTORS</b>")
        for row in contributors:
            lines.append(
                format_portfolio_holding_html(
                    symbol=row["symbol"],
                    direction_value=row.get("contribution_vs_nepse_pts", 0.0),
                    primary_text=f"vs N {row['contribution_vs_nepse_pts']:+.2f} pts",
                    secondary_text=f"Ret {row['return_pct']:+.1f}%" if row.get("return_pct") is not None else None,
                )
            )
    if laggards:
        lines.append("")
        lines.append("<b>LAGGARDS</b>")
        for row in laggards:
            lines.append(
                format_portfolio_holding_html(
                    symbol=row["symbol"],
                    direction_value=row.get("contribution_vs_nepse_pts", 0.0),
                    primary_text=f"vs N {row['contribution_vs_nepse_pts']:+.2f} pts",
                    secondary_text=f"Ret {row['return_pct']:+.1f}%" if row.get("return_pct") is not None else None,
                )
            )
    return "\n".join(lines)


def _build_viewer_performance_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_viewer_report

    report = build_viewer_report(trader)
    if not report:
        return "<b>PERFORMANCE</b>\n\nNo performance data available."
    summary = report["summary"]
    lines = [
        "<b>PERFORMANCE</b>",
        "",
        "<code>",
        f"  Portfolio   : {_fmt_npr(summary['nav'])}",
        f"  Total Return: {summary['total_return_pct']:+.2f}%" if summary.get("total_return_pct") is not None else "  Total Return: n/a",
        f"  Deployed    : {summary['deployed_return_pct']:+.2f}%",
        f"  Today       : {summary['day_return_pct']:+.2f}%" if summary.get("day_return_pct") is not None else "  Today       : n/a",
        f"  NEPSE       : {summary['benchmark_return_pct']:+.2f}%" if summary.get("benchmark_return_pct") is not None else "  NEPSE       : n/a",
        f"  Open Names  : {summary['open_positions']}",
        "</code>",
    ]
    return "\n".join(lines)


def _build_viewer_daily_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_viewer_report

    report = build_viewer_report(trader)
    if not report:
        return "<b>DAILY</b>\n\nNo daily report available."
    daily = report["daily"]
    lines = [
        "<b>TODAY</b>",
        "",
        "<code>",
        f"  Date       : {daily['date']}",
        f"  Portfolio  : {_fmt_npr(daily['nav'])}",
        f"  NEPSE Move : {daily['benchmark_move_pct']:+.2f}%" if daily.get("benchmark_move_pct") is not None else "  NEPSE Move : n/a",
        "</code>",
    ]
    if daily["trades_today"]:
        lines.append("")
        lines.append("<b>TRADES TODAY</b>")
        for row in daily["trades_today"][:6]:
            lines.append(
                format_trade_activity_html(
                    date=None,
                    action=row["action"],
                    symbol=row["symbol"],
                    shares=row["shares"],
                    price=row["price"],
                    pnl=row.get("pnl") if row["action"] == "SELL" else None,
                    include_date=False,
                )
            )
    contributor = daily.get("biggest_contributor")
    detractor = daily.get("biggest_detractor")
    if contributor:
        lines.append("")
        lines.append(f"<code>  Top contributor: {contributor['symbol']}</code>")
    if detractor:
        lines.append(f"<code>  Top laggard    : {detractor['symbol']}</code>")
    return "\n".join(lines)


def _build_viewer_trades_message(trader: "LiveTrader") -> str:
    from backend.quant_pro.reporting import build_viewer_report

    report = build_viewer_report(trader)
    if not report:
        return "<b>TRADES</b>\n\nNo trades available."
    lines = ["<b>RECENT TRADES</b>", ""]
    for row in report["portfolio"]["recent_trades"]:
        lines.append(
            format_trade_activity_html(
                date=row["date"],
                action=row["action"],
                symbol=row["symbol"],
                shares=row["shares"],
                price=row["price"],
                pnl=row.get("pnl") if row["action"] == "SELL" else None,
                status_text=row.get("status_text"),
            )
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# /start — Main Menu
# ─────────────────────────────────────────────────────────────────────────────

MAIN_MENU_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("📊 Portfolio", callback_data="menu_portfolio"),
     InlineKeyboardButton("📈 Alpha",     callback_data="menu_alpha")],
    [InlineKeyboardButton("⚠️ Risk",      callback_data="menu_risk"),
     InlineKeyboardButton("🩺 Health",   callback_data="menu_health")],
    [InlineKeyboardButton("📅 Daily",    callback_data="menu_daily"),
     InlineKeyboardButton("🔍 Signals",  callback_data="menu_signals")],
    [InlineKeyboardButton("🏦 TMS Acct", callback_data="menu_tms_account"),
     InlineKeyboardButton("💊 TMS Hlth", callback_data="menu_tms_health")],
    [InlineKeyboardButton("📋 Trades",   callback_data="menu_trades"),
     InlineKeyboardButton("📉 Attrib.",  callback_data="menu_attribution")],
    [InlineKeyboardButton("🔄 Status",   callback_data="menu_status"),
     InlineKeyboardButton("🔃 Refresh",  callback_data="menu_refresh")],
    [InlineKeyboardButton("📆 Calendar", callback_data="menu_calendar"),
     InlineKeyboardButton("⚡ Short-Term", callback_data="menu_short")],
    [InlineKeyboardButton("🟢 Buy",      callback_data="menu_buy"),
     InlineKeyboardButton("🔴 Sell",     callback_data="menu_sell")],
    [InlineKeyboardButton("❓ Help",      callback_data="menu_help")],
])

VIEWER_MENU_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("Portfolio", callback_data="viewer_portfolio"),
     InlineKeyboardButton("Performance", callback_data="viewer_performance")],
    [InlineKeyboardButton("Daily", callback_data="viewer_daily"),
     InlineKeyboardButton("Trades", callback_data="viewer_trades")],
    [InlineKeyboardButton("Help", callback_data="viewer_help")],
])


@authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    role = _app_role(context)
    if role == "viewer":
        await update.message.reply_text(
            "<b>Portfolio Viewer</b>\nChoose a view:",
            parse_mode="HTML",
            reply_markup=VIEWER_MENU_KEYBOARD,
        )
        return
    await update.message.reply_text(
        "<b>NEPSE Paper Trader</b>\nSelect an action:",
        parse_mode="HTML",
        reply_markup=MAIN_MENU_KEYBOARD,
    )


# ─────────────────────────────────────────────────────────────────────────────
# View Commands
# ─────────────────────────────────────────────────────────────────────────────

@authorized
async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _send_portfolio(update.message, role=_app_role(context))


@authorized
async def cmd_alpha(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _send_alpha_view(update.message)


@authorized
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await update.message.reply_text(_build_health_message(trader), parse_mode="HTML")


@authorized
async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    role = _app_role(context)
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    if role == "viewer":
        await update.message.reply_text(_build_viewer_daily_message(trader), parse_mode="HTML")
        return
    await update.message.reply_text(_build_owner_daily_message(trader), parse_mode="HTML")


@authorized
async def cmd_attribution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await update.message.reply_text(_build_attribution_message(trader), parse_mode="HTML")


@authorized
async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    from backend.trading.live_trader import compute_deployed_nav_chart_data

    with trader._state_lock:
        positions = dict(trader.positions)
    chart = compute_deployed_nav_chart_data(
        trader.capital,
        positions,
        trader.trade_log_file,
        trader.nav_log_file,
    )
    image = _render_alpha_chart_image(chart) if chart else None
    if image is not None:
        await update.message.reply_photo(
            photo=image,
            caption="<b>PERFORMANCE</b>\nPortfolio vs NEPSE since strategy start.",
            parse_mode="HTML",
        )
    await update.message.reply_text(_build_viewer_performance_message(trader), parse_mode="HTML")


def _build_risk_message(trader: "LiveTrader") -> str:
    from backend.trading.live_trader import compute_risk_snapshot

    with trader._state_lock:
        cash = trader.cash
        positions = dict(trader.positions)

    snapshot = compute_risk_snapshot(
        trader.capital,
        cash,
        positions,
        trader.nav_log_file,
    )

    lines = [
        "<b>RISK VIEW</b>",
        "",
        "<code>",
        f"  NAV         : {_fmt_npr(snapshot['nav'])}",
        f"  Cash Weight : {snapshot['cash_weight_pct']:.1f}%",
        f"  Drawdown    : {snapshot['drawdown_pct']:+.2f}% vs peak {_fmt_npr(snapshot['peak_nav'])}",
        f"  Roll DD 5   : {snapshot['rolling_drawdown_5d']:+.2f}%",
        f"  Roll DD 20  : {snapshot['rolling_drawdown_20d']:+.2f}%",
        f"  Peak Date   : {snapshot['peak_date']}",
        "</code>",
    ]

    if snapshot["top_positions"]:
        lines.append("")
        lines.append("<b>POSITION CONCENTRATION</b>")
        for row in snapshot["top_positions"]:
            lines.append(
                f"<code>  {row['symbol']:<8} {row['weight_pct']:>5.1f}%  {_fmt_npr(row['market_value'])}</code>"
            )

    if snapshot["top_sectors"]:
        lines.append("")
        lines.append("<b>SECTOR CONCENTRATION</b>")
        for row in snapshot["top_sectors"]:
            lines.append(
                f"<code>  {row['sector'][:18]:<18} {row['weight_pct']:>5.1f}%</code>"
            )

    if snapshot["signals"]:
        lines.append("")
        lines.append("<b>SIGNAL EXPOSURE</b>")
        for row in snapshot["signals"][:5]:
            lines.append(
                f"<code>  {row['signal_type'][:18]:<18} {row['weight_pct']:>5.1f}%  {row['count']} pos</code>"
            )

    if snapshot["holding_buckets"]:
        lines.append("")
        lines.append("<b>HOLDING AGE</b>")
        for row in snapshot["holding_buckets"]:
            lines.append(
                f"<code>  {row['holding_bucket']:<8} {row['weight_pct']:>5.1f}%  {row['count']} pos</code>"
            )

    if snapshot["winners"]:
        lines.append("")
        lines.append("<b>WINNERS</b>")
        for row in snapshot["winners"]:
            lines.append(
                f"<code>  {row['symbol']:<8} {row['pnl']:+,.0f} ({row['return_pct']:+.1f}%)</code>"
            )

    if snapshot["losers"]:
        lines.append("")
        has_negative_losers = any(row["pnl"] < 0 for row in snapshot["losers"])
        lines.append("<b>LOSERS</b>" if has_negative_losers else "<b>LAGGARDS</b>")
        for row in snapshot["losers"]:
            lines.append(
                f"<code>  {row['symbol']:<8} {row['pnl']:+,.0f} ({row['return_pct']:+.1f}%)</code>"
            )

    lines.extend(
        [
            "",
            "<b>WINNER / LAGGARD EXPOSURE</b>",
            f"<code>  Winners      : {snapshot['winner_exposure_pct']:.1f}%</code>",
            f"<code>  Laggards     : {snapshot['laggard_exposure_pct']:.1f}%</code>",
        ]
    )

    if snapshot["alerts"]:
        lines.append("")
        lines.append("<b>RISK ALERTS</b>")
        for alert in snapshot["alerts"]:
            lines.append(f"<code>  {alert}</code>")

    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        lines.append("")
        lines.append("<b>MARK DATA</b>")
        lines.extend(f"<code>  {line}</code>" for line in refresh_lines)

    return "\n".join(lines)


@authorized
async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await update.message.reply_text(_build_risk_message(trader), parse_mode="HTML")


async def _send_portfolio(target, role: str = "owner"):
    """Build and send portfolio message. `target` is a Message to reply to."""
    trader = _get_trader()
    if not trader:
        await target.reply_text("Trader not initialized.")
        return

    await _maybe_refresh_portfolio_prices(trader)

    if role == "viewer":
        await target.reply_text(
            _build_viewer_portfolio_message(trader),
            parse_mode="HTML",
            reply_markup=VIEWER_MENU_KEYBOARD,
        )
        return

    from backend.quant_pro.reporting import build_owner_report

    report = build_owner_report(trader)
    if not report:
        await target.reply_text(
            "<b>PORTFOLIO</b>\n\nNo portfolio data available.",
            parse_mode="HTML",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    summary = report["summary"]
    portfolio = report["portfolio"]
    alpha = report["alpha"]
    quality = report["health"]["data_quality"]
    holdings = portfolio["holdings_ranked"]
    if not holdings:
        await target.reply_text(
            "<b>PORTFOLIO</b>\n\nNo open positions.",
            parse_mode="HTML",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    laggards = [row for row in sorted(holdings, key=lambda item: item["contribution_vs_nepse_pts"]) if row["contribution_vs_nepse_pts"] < 0]
    stale = set(quality.get("stale_symbols") or [])
    refresh_lines = _format_refresh_meta(trader)
    marks_line = next((line for line in refresh_lines if line.startswith("Marks:")), None)

    lines = [
        f"<b>PORTFOLIO ({summary['open_positions']} names)</b>",
        "",
        f"NAV: {_fmt_npr(summary['nav'])} • Cash: {_fmt_npr(summary['cash'])}",
        f"Realized: {_fmt_npr(alpha['performance']['realized_pnl'])} ({alpha['performance']['realized_return_pct']:+.2f}%)",
        f"Deployed: {alpha['performance']['deployed_return_pct']:+.2f}%",
    ]
    benchmark = alpha.get("global_benchmark")
    if benchmark:
        lines.extend(
            [
                f"NEPSE: {benchmark['return_pct']:+.2f}%",
            ]
        )
    if marks_line:
        lines.append(marks_line)

    lines.extend(["", "<b>HOLDINGS</b>"])
    for row in holdings[:6]:
        suffix_parts = []
        source = str(row.get("mark_source") or "")
        if source and source not in {"nepalstock"}:
            suffix_parts.append(_format_mark_source(source) or source)
        if row["symbol"] in stale:
            suffix_parts.append("stale")
        sector_vs = row["active_vs_sector_pct"]
        extras = []
        if row.get("last_ltp") is not None:
            extras.append(f"LTP NPR {row['last_ltp']:,.1f}")
        if row.get("entry_price") is not None:
            extras.append(f"Entry NPR {row['entry_price']:,.1f}")
        extras.append(f"vs N {row['contribution_vs_nepse_pts']:+.2f} pts")
        if sector_vs is not None:
            extras.append(f"vs sector {sector_vs:+.1f}%")
        secondary_text = (
            f"Ret {row['return_pct']:+.1f}%"
            if row.get("return_pct") is not None
            else None
        )
        lines.append(
            format_portfolio_holding_html(
                symbol=row["symbol"],
                direction_value=row.get("unrealized_pnl", 0.0),
                primary_text=f"PnL NPR {row['unrealized_pnl']:+,.0f}",
                secondary_text=secondary_text,
                holding_days=row["holding_days"],
                extra_metrics=extras,
                flags=suffix_parts,
            )
        )

    if laggards:
        lines.extend(["", "<b>NEEDS ATTENTION</b>"])
        for row in laggards[:3]:
            suffix_parts = []
            source = str(row.get("mark_source") or "")
            if source and source not in {"nepalstock"}:
                suffix_parts.append(_format_mark_source(source) or source)
            if row["symbol"] in stale:
                suffix_parts.append("stale")
            sector_vs = row["active_vs_sector_pct"]
            extras = []
            if row.get("last_ltp") is not None:
                extras.append(f"LTP NPR {row['last_ltp']:,.1f}")
            if row.get("entry_price") is not None:
                extras.append(f"Entry NPR {row['entry_price']:,.1f}")
            extras.append(f"vs N {row['contribution_vs_nepse_pts']:+.2f} pts")
            if sector_vs is not None:
                extras.append(f"vs sector {sector_vs:+.1f}%")
            secondary_text = (
                f"Ret {row['return_pct']:+.1f}%"
                if row.get("return_pct") is not None
                else None
            )
            lines.append(
                format_portfolio_holding_html(
                    symbol=row["symbol"],
                    direction_value=row.get("contribution_vs_nepse_pts", 0.0),
                    primary_text=f"PnL NPR {row['unrealized_pnl']:+,.0f}",
                    secondary_text=secondary_text,
                    holding_days=row["holding_days"],
                    extra_metrics=extras,
                    flags=suffix_parts,
                )
            )

    if portfolio["recent_trades"]:
        lines.extend(["", "<b>RECENT ACTIVITY</b>"])
        for row in portfolio["recent_trades"][:3]:
            lines.append(
                format_trade_activity_html(
                    date=row["date"],
                    action=row["action"],
                    symbol=row["symbol"],
                    shares=row["shares"],
                    price=row["price"],
                    pnl=row.get("pnl") if row["action"] == "SELL" else None,
                    status_text=row.get("status_text"),
                )
            )

    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        lines.append("")
        lines.append("<b>MARK DATA</b>")
        lines.extend(f"<code>  {line}</code>" for line in refresh_lines[:2])

    buttons = []
    if not getattr(trader, "live_execution_enabled", False):
        buttons = [
            InlineKeyboardButton(f"Sell {row['symbol']}", callback_data=f"sell_start_{row['symbol']}")
            for row in holdings[: min(6, len(holdings))]
        ]
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)] if buttons else []
    keyboard.append([InlineKeyboardButton("« Menu", callback_data="menu_start")])
    await target.reply_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


@authorized
async def cmd_nav(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return

    with trader._state_lock:
        cash = trader.cash
        capital = trader.capital
        positions_value = sum(p.market_value for p in trader.positions.values())

    nav = cash + positions_value
    invested = capital - cash
    total_ret = (nav / capital - 1) if capital > 0 else 0

    msg = (
        f"<b>NAV SUMMARY</b>\n\n"
        f"<code>"
        f"  Capital  : {_fmt_npr(capital)}\n"
        f"  Cash     : {_fmt_npr(cash)}\n"
        f"  Invested : {_fmt_npr(invested)}\n"
        f"  NAV      : {_fmt_npr(nav)}\n"
        f"  Return   : {total_ret:+.2%}"
        f"</code>"
    )
    benchmark_lines = _build_benchmark_lines(trader)
    if benchmark_lines:
        msg += "\n\n" + "\n".join(benchmark_lines)
    refresh_lines = _format_refresh_meta(trader)
    if refresh_lines:
        msg += "\n\n" + "\n".join(f"<code>  {line}</code>" for line in refresh_lines)
    await update.message.reply_text(msg, parse_mode="HTML")


@authorized
async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _send_signals(update.message)


async def _send_signals(target):
    trader = _get_trader()
    if not trader:
        await target.reply_text("Trader not initialized.")
        return

    with trader._state_lock:
        signals = list(trader.signals_today)
        regime = trader.regime

    if not signals:
        await target.reply_text(
            f"<b>SIGNALS</b> (Regime: {regime.upper()})\n\nNo signals generated yet.",
            parse_mode="HTML",
        )
        return

    lines = [f"<b>TODAY'S SIGNALS</b> (Regime: {regime.upper()})\n"]
    buttons = []
    live_mode = bool(getattr(trader, "live_execution_enabled", False))

    for i, sig in enumerate(signals[:10], 1):
        lines.append(
            f"{i}. <b>{sig['symbol']}</b>  score={sig['score']:.2f}\n"
            f"   {sig['reasoning'][:60]}\n"
        )
        if not live_mode and sig["symbol"] not in (trader.positions or {}):
            buttons.append(
                InlineKeyboardButton(
                    f"Buy {sig['symbol']}",
                    callback_data=f"buy_signal_{sig['symbol']}",
                )
            )

    keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    keyboard.append([InlineKeyboardButton("« Menu", callback_data="menu_start")])
    if live_mode:
        lines.append("Use <code>/buy SYMBOL SHARES PRICE</code> for live orders.")

    await target.reply_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


@authorized
async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    role = _app_role(context)
    if role == "viewer":
        await update.message.reply_text(_build_viewer_trades_message(trader), parse_mode="HTML")
        return
    await update.message.reply_text(_build_owner_trades_message(trader), parse_mode="HTML")


@authorized
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await update.message.reply_text(_build_owner_status_message(trader), parse_mode="HTML")


@authorized
async def cmd_tms_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.live_session_status, force=True))
    await update.message.reply_text(_build_tms_status_message(trader, force=False), parse_mode="HTML")


@authorized
async def cmd_tms_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.refresh_tms_monitor, force=True, emit_alerts=False))
    await update.message.reply_text(_build_tms_account_message(trader, force=False), parse_mode="HTML")


@authorized
async def cmd_tms_funds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.refresh_tms_monitor, force=True, emit_alerts=False))
    await update.message.reply_text(_build_tms_funds_message(trader, force=False), parse_mode="HTML")


@authorized
async def cmd_tms_holdings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.refresh_tms_monitor, force=True, emit_alerts=False))
    await update.message.reply_text(_build_tms_holdings_message(trader, force=False), parse_mode="HTML")

@authorized
async def cmd_tms_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.refresh_tms_monitor, force=True, emit_alerts=False))
    await update.message.reply_text(_build_tms_trades_message(trader, force=False), parse_mode="HTML")


@authorized
async def cmd_tms_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return
    await asyncio.to_thread(functools.partial(trader.refresh_tms_monitor, force=True, emit_alerts=False))
    await update.message.reply_text(_build_tms_health_message(trader, force=False), parse_mode="HTML")


@authorized
async def cmd_short_term(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show short-term portfolio positions and recent trades."""
    target = update.effective_message
    if target is None:
        return
    trader = _get_trader()
    if not trader:
        await target.reply_text("Trader not initialized.")
        return

    with trader._state_lock:
        positions = dict(trader.positions)
        completed = list(trader.completed_trades) if hasattr(trader, 'completed_trades') else []

    # Filter to short-term signal types
    st_types = {"corporate_action", "settlement_pressure", "mean_reversion"}

    st_positions = {s: p for s, p in positions.items() if getattr(p, 'signal_type', '') in st_types}
    st_trades = [t for t in completed if getattr(t, 'signal_type', '') in st_types]
    st_trades = sorted(st_trades, key=lambda t: t.exit_date or t.entry_date, reverse=True)[:10]

    lines = [f"<b>SHORT-TERM PORTFOLIO</b>\n"]

    if st_positions:
        lines.append(f"<b>Open Positions ({len(st_positions)}):</b>")
        for sym, pos in sorted(st_positions.items()):
            ltp = getattr(pos, 'last_ltp', None) or pos.entry_price
            pnl = getattr(pos, 'unrealized_pnl', 0)
            emoji = _pnl_emoji(pnl)
            lines.append(
                f"  <b>{sym}</b> {pos.shares}@{pos.entry_price:,.0f} "
                f"→ {ltp:,.0f} {emoji} {pnl:+,.0f}"
            )
    else:
        lines.append("No open short-term positions.")

    if st_trades:
        wins = sum(1 for t in st_trades if t.net_pnl and t.net_pnl > 0)
        losses = len(st_trades) - wins
        lines.append(f"\n<b>Recent Trades ({wins}W/{losses}L):</b>")
        for t in st_trades[:5]:
            ret = t.net_return
            emoji = _pnl_emoji(t.net_pnl or 0)
            exit_str = t.exit_date.strftime("%m/%d") if t.exit_date else "open"
            lines.append(
                f"  {emoji} {t.symbol} {ret:+.1%} ({t.exit_reason}) {exit_str}"
            )

    await target.reply_text("\n".join(lines), parse_mode="HTML")


@authorized
async def cmd_calendar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show upcoming high-yield corporate actions."""
    target = update.effective_message
    if target is None:
        return
    import sqlite3
    from datetime import timedelta

    try:
        from backend.backtesting.simple_backtest import get_db_path, load_corporate_actions
        conn = sqlite3.connect(str(get_db_path()))
        corp_df = load_corporate_actions(conn)
        conn.close()
    except Exception as e:
        await target.reply_text(f"Could not load corporate actions: {e}")
        return

    if corp_df.empty:
        await target.reply_text("No corporate action data available.")
        return

    from datetime import datetime as dt
    now = dt.now()
    cutoff = now + timedelta(days=21)

    upcoming = corp_df[
        (corp_df["bookclose_date"] > now) &
        (corp_df["bookclose_date"] <= cutoff)
    ].copy()

    # Filter to high-yield only
    upcoming = upcoming[
        (upcoming["cash_dividend_pct"].fillna(0) >= 5) |
        (upcoming["bonus_share_pct"].fillna(0) >= 10)
    ]

    if upcoming.empty:
        await target.reply_text(
            "<b>CORPORATE ACTION CALENDAR</b>\n\n"
            "No high-yield events in next 3 weeks.",
            parse_mode="HTML",
        )
        return

    upcoming = upcoming.sort_values("bookclose_date")

    lines = [f"<b>UPCOMING EVENTS ({len(upcoming)})</b>\n"]
    for _, row in upcoming.iterrows():
        sym = row["symbol"]
        bc = row["bookclose_date"].strftime("%Y-%m-%d")
        cash = row.get("cash_dividend_pct") or 0
        bonus = row.get("bonus_share_pct") or 0
        days = (row["bookclose_date"] - now).days

        # T-7 entry window
        entry_by = (row["bookclose_date"] - timedelta(days=5)).strftime("%m/%d")

        parts = []
        if cash >= 5:
            parts.append(f"Div:{cash}%")
        if bonus >= 10:
            parts.append(f"Bonus:{bonus}%")

        lines.append(
            f"<b>{sym}</b> — BC: {bc} ({days}d)\n"
            f"  {' + '.join(parts)}  Entry by: {entry_by}"
        )

    await target.reply_text("\n".join(lines), parse_mode="HTML")


@authorized
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = _build_viewer_help_message() if _app_role(context) == "viewer" else _build_owner_help_message()
    await update.message.reply_text(msg, parse_mode="HTML")


# ─────────────────────────────────────────────────────────────────────────────
# Action Commands
# ─────────────────────────────────────────────────────────────────────────────

@authorized
async def cmd_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return

    await update.message.reply_text("Refreshing prices...")
    with trader._state_lock:
        trader.refresh_prices()
    await update.message.reply_text("Prices refreshed.")


@authorized
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return

    with trader._state_lock:
        trader.send_daily_summary()
    await update.message.reply_text("Daily summary sent.")


# ─────────────────────────────────────────────────────────────────────────────
# Buy Flow (ConversationHandler)
# ─────────────────────────────────────────────────────────────────────────────

BUY_CONFIRM = 0


@authorized
async def buy_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /buy SYMBOL [SHARES] or /buy (show signals)."""
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return ConversationHandler.END

    args = context.args or []

    if trader.live_execution_enabled:
        await update.message.reply_text("Only paper trading is supported in this build.")
        return ConversationHandler.END

    # /buy with no args → show signals as buttons
    if not args:
        await _send_signals(update.message)
        return ConversationHandler.END

    symbol = args[0].upper()
    shares = int(args[1]) if len(args) > 1 else None

    return await _buy_preview(update.message, context, trader, symbol, shares)


async def _buy_preview(target, context, trader, symbol, shares):
    """Show buy confirmation preview."""
    from backend.quant_pro.vendor_api import fetch_latest_ltp

    ltp = fetch_latest_ltp(symbol)
    if ltp is None or ltp <= 0:
        await target.reply_text(f"Could not fetch LTP for {symbol}.")
        return ConversationHandler.END

    with trader._state_lock:
        if symbol in trader.positions:
            await target.reply_text(f"Already holding {symbol}.")
            return ConversationHandler.END
        cash = trader.cash
        capital = trader.capital
        max_positions = trader.max_positions
        n_positions = len(trader.positions)

    if n_positions >= max_positions:
        await target.reply_text(f"Max positions ({max_positions}) reached.")
        return ConversationHandler.END

    # Auto-size if no shares specified
    if shares is None:
        per_position = capital / max_positions
        available = min(per_position, cash * 0.95)
        shares = int(available / ltp)

    if shares < 1:
        await target.reply_text("Insufficient cash for even 1 share.")
        return ConversationHandler.END

    from backend.backtesting.simple_backtest import NepseFees
    fees = NepseFees.total_fees(shares, ltp)
    total_cost = shares * ltp + fees
    cash_after = cash - total_cost

    if total_cost > cash:
        # Try smaller size
        shares = int((cash - NepseFees.total_fees(1, ltp)) / ltp)
        if shares < 1:
            await target.reply_text("Insufficient cash.")
            return ConversationHandler.END
        fees = NepseFees.total_fees(shares, ltp)
        total_cost = shares * ltp + fees
        cash_after = cash - total_cost

    # Store in context for confirmation
    context.user_data["buy_symbol"] = symbol
    context.user_data["buy_shares"] = shares
    context.user_data["buy_ltp"] = ltp
    context.user_data["buy_fees"] = fees
    context.user_data["buy_total"] = total_cost

    msg = (
        f"<b>BUY ORDER PREVIEW</b>\n\n"
        f"<code>"
        f"  Symbol    : {symbol}\n"
        f"  Shares    : {shares}\n"
        f"  LTP       : {_fmt_npr(ltp)}\n"
        f"  Value     : {_fmt_npr(shares * ltp)}\n"
        f"  Fees      : {_fmt_npr(fees)}\n"
        f"  Total     : {_fmt_npr(total_cost)}\n"
        f"  Cash after: {_fmt_npr(cash_after)}"
        f"</code>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Confirm Buy", callback_data="buy_confirm"),
         InlineKeyboardButton("Cancel", callback_data="buy_cancel")],
    ])

    await target.reply_text(msg, parse_mode="HTML", reply_markup=keyboard)
    return BUY_CONFIRM


@authorized_callback
async def buy_confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == "buy_cancel":
        await query.edit_message_text("Buy cancelled.")
        return ConversationHandler.END

    if data != "buy_confirm":
        return BUY_CONFIRM

    trader = _get_trader()
    if not trader:
        await query.edit_message_text("Trader not initialized.")
        return ConversationHandler.END

    symbol = context.user_data.get("buy_symbol")
    shares = context.user_data.get("buy_shares")
    ltp = context.user_data.get("buy_ltp")

    if not all([symbol, shares, ltp]):
        await query.edit_message_text("Buy data missing. Try again.")
        return ConversationHandler.END

    result = build_live_trader_control_plane(trader).submit_paper_order(
        action="buy",
        symbol=symbol,
        quantity=int(shares),
        limit_price=float(ltp),
        thesis="telegram_manual_buy",
    )

    await query.edit_message_text(result.message, parse_mode="HTML")
    return ConversationHandler.END


# ─────────────────────────────────────────────────────────────────────────────
# Sell Flow (ConversationHandler)
# ─────────────────────────────────────────────────────────────────────────────

SELL_CONFIRM = 0


@authorized
async def sell_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /sell SYMBOL or /sell (show positions)."""
    trader = _get_trader()
    if not trader:
        await update.message.reply_text("Trader not initialized.")
        return ConversationHandler.END

    args = context.args or []

    if trader.live_execution_enabled:
        await update.message.reply_text("Only paper trading is supported in this build.")
        return ConversationHandler.END

    # /sell with no args → show portfolio with sell buttons
    if not args:
        await _send_portfolio(update.message)
        return ConversationHandler.END

    symbol = args[0].upper()
    return await _sell_preview(update.message, context, trader, symbol)


async def _sell_preview(target, context, trader, symbol):
    """Show sell confirmation preview."""
    from backend.trading.live_trader import count_trading_days_since
    from backend.backtesting.simple_backtest import NepseFees

    with trader._state_lock:
        if symbol not in trader.positions:
            await target.reply_text(f"No position in {symbol}.")
            return ConversationHandler.END
        pos = trader.positions[symbol]
        shares = pos.shares
        entry_price = pos.entry_price
        buy_fees = pos.buy_fees
        entry_date = pos.entry_date
        ltp = pos.last_ltp

    if ltp is None:
        from backend.quant_pro.vendor_api import fetch_latest_ltp
        ltp = fetch_latest_ltp(symbol)
        if ltp is None:
            await target.reply_text(f"Could not fetch LTP for {symbol}.")
            return ConversationHandler.END

    days = count_trading_days_since(entry_date)
    sell_fees = NepseFees.total_fees(shares, ltp)
    proceeds = shares * ltp - sell_fees
    gross_pnl = (ltp - entry_price) * shares
    net_pnl = gross_pnl - buy_fees - sell_fees
    cost_basis = shares * entry_price + buy_fees
    pnl_pct = net_pnl / cost_basis if cost_basis > 0 else 0.0
    emoji = _pnl_emoji(net_pnl)

    context.user_data["sell_symbol"] = symbol
    context.user_data["sell_shares"] = shares
    context.user_data["sell_ltp"] = ltp

    msg = (
        f"<b>SELL ORDER PREVIEW</b>\n\n"
        f"<code>"
        f"  Symbol   : {symbol}  ({shares} shares)\n"
        f"  Entry    : {_fmt_npr(entry_price)} → LTP: {_fmt_npr(ltp)}\n"
        f"  Proceeds : {_fmt_npr(proceeds)} (after fees)\n"
        f"  P&L      : {emoji} {net_pnl:+,.0f} ({pnl_pct:+.1%})\n"
        f"  Held     : {days} trading days"
        f"</code>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Confirm Sell", callback_data="sell_confirm"),
         InlineKeyboardButton("Cancel", callback_data="sell_cancel")],
    ])

    await target.reply_text(msg, parse_mode="HTML", reply_markup=keyboard)
    return SELL_CONFIRM


@authorized_callback
async def sell_confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == "sell_cancel":
        await query.edit_message_text("Sell cancelled.")
        return ConversationHandler.END

    if data != "sell_confirm":
        return SELL_CONFIRM

    trader = _get_trader()
    if not trader:
        await query.edit_message_text("Trader not initialized.")
        return ConversationHandler.END

    symbol = context.user_data.get("sell_symbol")
    if not symbol:
        await query.edit_message_text("Sell data missing. Try again.")
        return ConversationHandler.END

    shares = int(context.user_data.get("sell_shares") or 0)
    ltp = float(context.user_data.get("sell_ltp") or 0.0)
    result = build_live_trader_control_plane(trader).submit_paper_order(
        action="sell",
        symbol=symbol,
        quantity=shares,
        limit_price=ltp if ltp > 0 else 1.0,
        thesis="telegram_manual_sell",
    )

    await query.edit_message_text(result.message, parse_mode="HTML")
    return ConversationHandler.END


# ─────────────────────────────────────────────────────────────────────────────
# Callback Router (menu buttons + inline action buttons)
# ─────────────────────────────────────────────────────────────────────────────

@authorized_callback
async def owner_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    # Menu buttons
    if data == "menu_start":
        await query.edit_message_text(
            "<b>NEPSE Paper Trader</b>\nSelect an action:",
            parse_mode="HTML",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return

    if data == "menu_portfolio":
        await _send_portfolio(query.message, role="owner")
        return

    if data == "menu_signals":
        await _send_signals(query.message)
        return

    if data == "menu_status":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_owner_status_message(trader), parse_mode="HTML")
        return

    if data == "menu_trades":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_owner_trades_message(trader), parse_mode="HTML")
        return

    if data == "menu_alpha":
        await _send_alpha_view(query.message)
        return

    if data == "menu_risk":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_risk_message(trader), parse_mode="HTML")
        return

    if data == "menu_health":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_health_message(trader), parse_mode="HTML")
        return

    if data == "menu_daily":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_owner_daily_message(trader), parse_mode="HTML")
        return

    if data == "menu_attribution":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_attribution_message(trader), parse_mode="HTML")
        return

    if data == "menu_tms_account":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_tms_account_message(trader), parse_mode="HTML")
        return

    if data == "menu_tms_health":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text(_build_tms_health_message(trader), parse_mode="HTML")
        return

    if data == "menu_short":
        # Trigger the /short command via the callback
        await cmd_short_term(update, context)
        return

    if data == "menu_calendar":
        await cmd_calendar(update, context)
        return

    if data == "menu_buy":
        trader = _get_trader()
        if trader and trader.live_execution_enabled:
            await query.message.reply_text(
                "Live mode requires explicit pricing.\nUse <code>/buy SYMBOL SHARES PRICE</code>.",
                parse_mode="HTML",
            )
        else:
            await _send_signals(query.message)
        return

    if data == "menu_sell":
        trader = _get_trader()
        if trader and trader.live_execution_enabled:
            await query.message.reply_text(
                "Live mode requires explicit pricing.\nUse <code>/sell SYMBOL SHARES PRICE</code> or <code>/sell SYMBOL all PRICE</code>.",
                parse_mode="HTML",
            )
        else:
            await _send_portfolio(query.message, role="owner")
        return

    if data == "menu_refresh":
        trader = _get_trader()
        if not trader:
            return
        await query.message.reply_text("Refreshing prices...")
        with trader._state_lock:
            trader.refresh_prices()
        await query.message.reply_text("Prices refreshed.")
        return

    if data == "menu_help":
        await query.message.reply_text(_build_owner_help_message(), parse_mode="HTML")
        return

    # Sell from portfolio view: sell_start_SYMBOL
    if data.startswith("sell_start_"):
        symbol = data.removeprefix("sell_start_")
        trader = _get_trader()
        if not trader:
            return
        if trader.live_execution_enabled:
            await query.message.reply_text(
                f"Live mode requires explicit pricing.\nUse <code>/sell {symbol} SHARES PRICE</code>.",
                parse_mode="HTML",
            )
            return
        context.user_data["sell_symbol"] = symbol
        await _sell_preview(query.message, context, trader, symbol)
        return

    # Buy from signals view: buy_signal_SYMBOL
    if data.startswith("buy_signal_"):
        symbol = data.removeprefix("buy_signal_")
        trader = _get_trader()
        if not trader:
            return
        if trader.live_execution_enabled:
            await query.message.reply_text(
                f"Live mode requires explicit pricing.\nUse <code>/buy {symbol} SHARES PRICE</code>.",
                parse_mode="HTML",
            )
            return
        await _buy_preview(query.message, context, trader, symbol, shares=None)
        return


@authorized_callback
async def viewer_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    trader = _get_trader()
    if not trader:
        return

    if data == "viewer_start":
        await query.edit_message_text(
            "<b>Portfolio Viewer</b>\nChoose a view:",
            parse_mode="HTML",
            reply_markup=VIEWER_MENU_KEYBOARD,
        )
        return
    if data == "viewer_portfolio":
        await _maybe_refresh_portfolio_prices(trader)
        await query.message.reply_text(_build_viewer_portfolio_message(trader), parse_mode="HTML")
        return
    if data == "viewer_performance":
        from backend.trading.live_trader import compute_deployed_nav_chart_data

        with trader._state_lock:
            positions = dict(trader.positions)
        chart = compute_deployed_nav_chart_data(
            trader.capital,
            positions,
            trader.trade_log_file,
            trader.nav_log_file,
        )
        image = _render_alpha_chart_image(chart) if chart else None
        if image is not None:
            await query.message.reply_photo(
                photo=image,
                caption="<b>PERFORMANCE</b>\nPortfolio vs NEPSE since strategy start.",
                parse_mode="HTML",
            )
        await query.message.reply_text(_build_viewer_performance_message(trader), parse_mode="HTML")
        return
    if data == "viewer_daily":
        await query.message.reply_text(_build_viewer_daily_message(trader), parse_mode="HTML")
        return
    if data == "viewer_trades":
        await query.message.reply_text(_build_viewer_trades_message(trader), parse_mode="HTML")
        return
    if data == "viewer_help":
        await query.message.reply_text(_build_viewer_help_message(), parse_mode="HTML")
        return


# ─────────────────────────────────────────────────────────────────────────────
# Bot Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

OWNER_COMMANDS: Sequence[str] = (
    "start", "buy", "sell", "p", "portfolio", "nav", "alpha", "risk",
    "signals", "trades", "status", "help", "short", "st", "calendar",
    "refresh", "summary", "health", "daily", "attribution",
    "tms_status", "tms_account", "tms_funds", "tms_holdings",
    "tms_trades", "tms_health",
)

VIEWER_COMMANDS: Sequence[str] = (
    "start", "portfolio", "performance", "daily", "trades", "help",
)


def get_registered_commands_for_role(role: str) -> Sequence[str]:
    return VIEWER_COMMANDS if str(role) == "viewer" else OWNER_COMMANDS


def _build_application(
    token: str,
    *,
    role: str,
    allowed_chat_ids: set[int],
    viewer_channel_id: Optional[int] = None,
) -> Application:
    """Create the Application with role-specific handlers registered."""
    app = Application.builder().token(token).build()
    app.bot_data["role"] = str(role)
    app.bot_data["allowed_chat_ids"] = set(allowed_chat_ids)
    if viewer_channel_id is not None:
        app.bot_data["viewer_channel_id"] = int(viewer_channel_id)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    if role == "viewer":
        app.add_handler(CommandHandler("portfolio", cmd_portfolio))
        app.add_handler(CommandHandler("performance", cmd_performance))
        app.add_handler(CommandHandler("daily", cmd_daily))
        app.add_handler(CommandHandler("trades", cmd_trades))
        app.add_handler(CallbackQueryHandler(viewer_callback_router, pattern=r"^viewer_"))
    else:
        app.add_handler(CommandHandler("buy", buy_entry))
        app.add_handler(CommandHandler("sell", sell_entry))
        app.add_handler(CommandHandler("p", cmd_portfolio))
        app.add_handler(CommandHandler("portfolio", cmd_portfolio))
        app.add_handler(CommandHandler("nav", cmd_nav))
        app.add_handler(CommandHandler("alpha", cmd_alpha))
        app.add_handler(CommandHandler("risk", cmd_risk))
        app.add_handler(CommandHandler("signals", cmd_signals))
        app.add_handler(CommandHandler("trades", cmd_trades))
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("short", cmd_short_term))
        app.add_handler(CommandHandler("st", cmd_short_term))
        app.add_handler(CommandHandler("calendar", cmd_calendar))
        app.add_handler(CommandHandler("refresh", cmd_refresh))
        app.add_handler(CommandHandler("summary", cmd_summary))
        app.add_handler(CommandHandler("health", cmd_health))
        app.add_handler(CommandHandler("daily", cmd_daily))
        app.add_handler(CommandHandler("attribution", cmd_attribution))
        app.add_handler(CommandHandler("tms_status", cmd_tms_status))
        app.add_handler(CommandHandler("tms_account", cmd_tms_account))
        app.add_handler(CommandHandler("tms_funds", cmd_tms_funds))
        app.add_handler(CommandHandler("tms_holdings", cmd_tms_holdings))
        app.add_handler(CommandHandler("tms_trades", cmd_tms_trades))
        app.add_handler(CommandHandler("tms_health", cmd_tms_health))

        app.add_handler(CallbackQueryHandler(buy_confirm_handler, pattern="^buy_(confirm|cancel)$"))
        app.add_handler(CallbackQueryHandler(sell_confirm_handler, pattern="^sell_(confirm|cancel)$"))
        app.add_handler(CallbackQueryHandler(owner_callback_router))

    # Error handler to surface hidden exceptions
    async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram bot exception: %s", context.error, exc_info=context.error)

    app.add_error_handler(_error_handler)

    return app


def _run_bot(
    trader: "LiveTrader",
    token: str,
    *,
    role: str,
    allowed_chat_ids: set[int],
    startup_chat_id: Optional[int] = None,
    viewer_channel_id: Optional[int] = None,
) -> None:
    """Run bot polling in its own asyncio event loop (blocking)."""
    global _trader

    _trader = trader

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = _build_application(
        token,
        role=role,
        allowed_chat_ids=allowed_chat_ids,
        viewer_channel_id=viewer_channel_id,
    )
    _applications[str(role)] = app

    logger.info("Telegram %s bot starting (allowed=%s)...", role, sorted(allowed_chat_ids))

    loop.run_until_complete(app.initialize())
    loop.run_until_complete(app.start())
    loop.run_until_complete(
        app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
        )
    )

    # Send startup message.
    # Keep viewer channels quiet: they should only receive real portfolio updates.
    should_send_startup = startup_chat_id is not None
    if should_send_startup and role == "viewer":
        try:
            should_send_startup = int(startup_chat_id) > 0
        except Exception:
            should_send_startup = False
    if should_send_startup:
        try:
            title = "Portfolio Viewer Bot Started" if role == "viewer" else "NEPSE Paper Trader Bot Started"
            loop.run_until_complete(
                app.bot.send_message(
                    chat_id=startup_chat_id,
                    text=f"<b>{title}</b>\nSend /start for menu.",
                    parse_mode="HTML",
                )
            )
        except Exception as e:
            logger.warning("Failed to send %s startup message: %s", role, e)

    # Block until the thread is killed (daemon thread)
    try:
        loop.run_forever()
    except Exception:
        pass
    finally:
        loop.run_until_complete(app.updater.stop())
        loop.run_until_complete(app.stop())
        loop.run_until_complete(app.shutdown())
        loop.close()
        _applications.pop(str(role), None)


def start_bot_thread(trader: "LiveTrader", token: str, chat_id: str) -> threading.Thread:
    """Start the Telegram bot in a daemon thread.

    Args:
        trader: LiveTrader instance (must have _state_lock).
        token: Telegram bot token.
        chat_id: Authorized chat ID (string, will be converted to int).

    Returns:
        The daemon thread (already started).
    """
    t = threading.Thread(
        target=_run_bot,
            kwargs={
                "trader": trader,
                "token": token,
                "role": "owner",
                "allowed_chat_ids": {int(chat_id)},
                "startup_chat_id": int(chat_id),
                "viewer_channel_id": None,
            },
        name="TelegramOwnerBot",
        daemon=True,
    )
    t.start()
    logger.info("Telegram bot thread started")
    return t


def start_bot_threads(
    trader: "LiveTrader",
    *,
    owner_token: Optional[str],
    owner_chat_id: Optional[str],
    viewer_token: Optional[str] = None,
    viewer_chat_ids: Optional[str] = None,
    viewer_startup_chat_id: Optional[str] = None,
) -> Dict[str, threading.Thread]:
    threads: Dict[str, threading.Thread] = {}

    if owner_token and owner_chat_id:
        threads["owner"] = start_bot_thread(trader, owner_token, owner_chat_id)

    viewer_allowed = _parse_chat_ids(viewer_chat_ids)
    explicit_startup = _parse_chat_ids(viewer_startup_chat_id)
    startup_chat = None
    if explicit_startup:
        startup_chat = next(iter(explicit_startup))
    elif viewer_allowed:
        startup_chat = next(iter(viewer_allowed))

    if viewer_token and (viewer_allowed or startup_chat is not None):

        t = threading.Thread(
            target=_run_bot,
            kwargs={
                "trader": trader,
                "token": viewer_token,
                "role": "viewer",
                "allowed_chat_ids": viewer_allowed,
                "startup_chat_id": startup_chat,
                "viewer_channel_id": startup_chat if (startup_chat is not None and int(startup_chat) < 0) else None,
            },
            name="TelegramViewerBot",
            daemon=True,
        )
        t.start()
        threads["viewer"] = t
        logger.info("Viewer Telegram bot thread started")

    return threads
