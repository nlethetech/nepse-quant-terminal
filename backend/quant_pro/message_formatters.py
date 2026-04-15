"""Compact formatting helpers for Telegram and TUI trade messages."""

from __future__ import annotations

from html import escape
from typing import Optional

from backend.quant_pro.tms_models import (
    ExecutionAction,
    ExecutionIntent,
    ExecutionResult,
    ExecutionSource,
    ExecutionStatus,
)


def _action_prefix(action: str) -> str:
    action_upper = str(action or "").upper()
    return {
        "BUY": "🟢 BUY",
        "SELL": "🔴 SELL",
        "MODIFY": "🟡 MODIFY",
        "CANCEL": "⚪ CANCEL",
    }.get(action_upper, action_upper or "TRADE")


def _action_emoji(action: str) -> str:
    action_upper = str(action or "").upper()
    return {
        "BUY": "🟢",
        "SELL": "🔴",
        "MODIFY": "🟡",
        "CANCEL": "⚪",
    }.get(action_upper, "ℹ️")


def _polarity_emoji(value: float) -> str:
    if float(value) > 0:
        return "🟢"
    if float(value) < 0:
        return "🔴"
    return "⚪"


def _status_label(status: str) -> tuple[str, str]:
    value = str(status or "").lower()
    mapping = {
        str(ExecutionStatus.PENDING_CONFIRMATION): ("Pending Confirmation", "⏳"),
        str(ExecutionStatus.QUEUED): ("Queued", "⏳"),
        str(ExecutionStatus.SUBMITTING): ("Submitting", "⏳"),
        str(ExecutionStatus.ACCEPTED): ("Accepted", "✅"),
        str(ExecutionStatus.SUBMITTED_PENDING): ("Submitted", "⏳"),
        str(ExecutionStatus.PARTIALLY_FILLED): ("Partially Filled", "🟡"),
        str(ExecutionStatus.FILLED): ("Filled", "✅"),
        str(ExecutionStatus.CANCELLED): ("Cancelled", "⚪"),
        str(ExecutionStatus.REJECTED_PRETRADE): ("Failed", "❌"),
        str(ExecutionStatus.SUBMIT_FAILED): ("Failed", "❌"),
        str(ExecutionStatus.MODIFY_FAILED): ("Failed", "❌"),
        str(ExecutionStatus.FROZEN): ("Frozen", "⚠️"),
    }
    return mapping.get(value, ("Updated", "ℹ️"))


def _live_flow_label(intent: ExecutionIntent) -> str:
    if intent.action == ExecutionAction.BUY:
        return "MANUAL BUY" if intent.source == ExecutionSource.OWNER_MANUAL else "AUTO BUY"
    if intent.action == ExecutionAction.SELL:
        return "MANUAL SELL" if intent.source == ExecutionSource.OWNER_MANUAL else "AUTO SELL"
    if intent.action == ExecutionAction.CANCEL:
        return "MANUAL CANCEL" if intent.source == ExecutionSource.OWNER_MANUAL else "AUTO CANCEL"
    if intent.action == ExecutionAction.MODIFY:
        return "MANUAL MODIFY" if intent.source == ExecutionSource.OWNER_MANUAL else "AUTO MODIFY"
    return f"LIVE {str(intent.action).upper()}"


def format_trade_activity_line(
    *,
    date: Optional[str],
    action: str,
    symbol: str,
    shares: int,
    price: float,
    pnl: Optional[float] = None,
    status_text: Optional[str] = None,
    include_date: bool = True,
) -> str:
    parts = []
    if include_date and date:
        parts.append(str(date))
    parts.append(f"{_action_prefix(action)} {symbol}")
    parts.append(f"{int(shares)} @ NPR {float(price):,.1f}")
    if str(action).upper() == "SELL" and pnl is not None:
        parts.append(f"NPR {float(pnl):+,.0f} {'✅' if float(pnl) >= 0 else '❌'}")
    elif status_text:
        parts.append(str(status_text).strip())
    return " • ".join(parts)


def format_trade_activity_html(**kwargs: object) -> str:
    return escape(format_trade_activity_line(**kwargs))


def format_portfolio_holding_html(
    *,
    symbol: str,
    direction_value: float,
    primary_text: str,
    secondary_text: Optional[str] = None,
    holding_days: Optional[int] = None,
    extra_metrics: Optional[list[str]] = None,
    flags: Optional[list[str]] = None,
) -> str:
    emoji = _polarity_emoji(direction_value)
    line_one_parts = [
        f"{emoji} <b>{escape(str(symbol))}</b>",
        escape(primary_text),
    ]
    if secondary_text:
        line_one_parts.append(escape(secondary_text))
    if holding_days is not None:
        line_one_parts.append(f"Day {int(holding_days)}")

    line_two_parts = [escape(metric) for metric in (extra_metrics or []) if metric]
    line_two_parts.extend(escape(flag) for flag in (flags or []) if flag)
    if line_two_parts:
        return " • ".join(line_one_parts) + "\n" + " • ".join(line_two_parts)
    return " • ".join(line_one_parts)


def format_live_order_summary_lines(
    intent: ExecutionIntent,
    result: Optional[ExecutionResult] = None,
) -> list[str]:
    status_value = str(result.status if result is not None else intent.status)
    status_label, status_emoji = _status_label(status_value)
    lines = [
        f"{_action_emoji(str(intent.action))} {_live_flow_label(intent)} order for {intent.symbol} {status_label} {status_emoji}",
    ]

    price = None
    if result is not None and result.observed_price is not None:
        price = float(result.observed_price)
    elif intent.limit_price is not None:
        price = float(intent.limit_price)

    qty = int(result.observed_qty if result is not None and result.observed_qty is not None else intent.quantity)
    detail_parts = []
    if qty > 0:
        detail_parts.append(str(qty))
    if price is not None and price > 0:
        detail_parts.append(f"@ NPR {price:,.2f}")
    if intent.reason:
        detail_parts.append(str(intent.reason).replace("_", " "))
    if detail_parts:
        lines.append(" • ".join(detail_parts))

    meta_parts = []
    order_ref = result.broker_order_ref if result is not None else intent.broker_order_ref
    if order_ref:
        meta_parts.append(f"Ref {order_ref}")
    detail = ""
    if result is not None:
        detail = str(result.status_text or result.dom_error_text or "").strip()
    if detail and detail.lower() not in {"filled", "success"}:
        meta_parts.append(detail)
    if meta_parts:
        lines.append(" • ".join(meta_parts))

    return lines


def format_live_order_summary_html(
    intent: ExecutionIntent,
    result: Optional[ExecutionResult] = None,
) -> str:
    lines = format_live_order_summary_lines(intent, result=result)
    if not lines:
        return ""
    return "\n".join([f"<b>{escape(lines[0])}</b>", *[escape(line) for line in lines[1:]]])
