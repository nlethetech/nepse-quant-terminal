"""
Telegram alert module for NEPSE Live Paper Trader.

Reads owner and viewer routing from environment variables:
  NEPSE_TELEGRAM_BOT_TOKEN              — owner bot token
  NEPSE_TELEGRAM_CHAT_ID                — owner private chat ID
  NEPSE_VIEWER_TELEGRAM_BOT_TOKEN       — viewer bot token
  NEPSE_VIEWER_TELEGRAM_POST_CHAT_ID    — viewer broadcast chat/channel
  NEPSE_VIEWER_SHOW_TRADES              — whether to post public buy/sell events
  NEPSE_VIEWER_DAILY_POSTS              — whether to post public daily updates

Gracefully degrades to console logging if credentials are not set.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from backend.quant_pro.message_formatters import format_trade_activity_html

logger = logging.getLogger(__name__)

OWNER_BOT_TOKEN: Optional[str] = os.environ.get("NEPSE_TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID: Optional[str] = os.environ.get("NEPSE_TELEGRAM_CHAT_ID")
VIEWER_BOT_TOKEN: Optional[str] = os.environ.get("NEPSE_VIEWER_TELEGRAM_BOT_TOKEN")
VIEWER_POST_CHAT_ID: Optional[str] = os.environ.get("NEPSE_VIEWER_TELEGRAM_POST_CHAT_ID")
VIEWER_ALLOWED_CHAT_IDS: Optional[str] = os.environ.get("NEPSE_VIEWER_TELEGRAM_ALLOWED_CHAT_IDS")

_last_send_time: float = 0.0
_MIN_INTERVAL: float = 1.0  # Telegram rate limit: max 1 msg/sec


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


VIEWER_SHOW_TRADES = _env_flag("NEPSE_VIEWER_SHOW_TRADES", True)
VIEWER_DAILY_POSTS = _env_flag("NEPSE_VIEWER_DAILY_POSTS", True)


def _parse_chat_ids(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    ids: list[str] = []
    for chunk in str(raw).replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            ids.append(item)
    return ids


def _owner_is_configured() -> bool:
    return bool(OWNER_BOT_TOKEN and OWNER_CHAT_ID)


def _viewer_targets() -> list[str]:
    explicit = _parse_chat_ids(VIEWER_POST_CHAT_ID)
    if explicit:
        return explicit
    return _parse_chat_ids(VIEWER_ALLOWED_CHAT_IDS)


def _viewer_is_configured() -> bool:
    return bool(VIEWER_BOT_TOKEN and _viewer_targets())


def _rate_limit_wait() -> None:
    """Enforce 1 message/second Telegram API rate limit."""
    global _last_send_time
    now = time.monotonic()
    elapsed = now - _last_send_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_send_time = time.monotonic()


def _get_bot(role: str = "owner"):
    """Get a fresh Bot instance for the target audience."""
    import telegram
    token = OWNER_BOT_TOKEN if role == "owner" else VIEWER_BOT_TOKEN
    return telegram.Bot(token=token)


def _send_message(bot, chat_id: str, message: str) -> None:
    asyncio.get_event_loop().run_until_complete(
        bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
    )


def _send_photo(bot, chat_id: str, photo_path: str, caption: str) -> None:
    with open(photo_path, "rb") as handle:
        asyncio.get_event_loop().run_until_complete(
            bot.send_photo(chat_id=chat_id, photo=handle, caption=caption, parse_mode="HTML")
        )


def _send_to_audience(message: str, *, role: str, chat_ids: list[str]) -> None:
    _rate_limit_wait()
    try:
        bot = _get_bot(role)
        for chat_id in chat_ids:
            _send_message(bot, chat_id, message)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            bot = _get_bot(role)
            for chat_id in chat_ids:
                loop.run_until_complete(
                    bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
                )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        logger.warning("Telegram %s send failed: %s", role, e)


def _send_photo_to_audience(photo_path: str, caption: str, *, role: str, chat_ids: list[str]) -> None:
    _rate_limit_wait()
    try:
        bot = _get_bot(role)
        for chat_id in chat_ids:
            _send_photo(bot, chat_id, photo_path, caption)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            bot = _get_bot(role)
            for chat_id in chat_ids:
                with open(photo_path, "rb") as handle:
                    loop.run_until_complete(
                        bot.send_photo(chat_id=chat_id, photo=handle, caption=caption, parse_mode="HTML")
                    )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        logger.warning("Telegram %s photo send failed: %s", role, e)


def send_alert(message: str) -> None:
    """Send a private owner alert."""
    if not _owner_is_configured():
        logger.info("[TG-disabled] %s", message)
        return
    _send_to_audience(message, role="owner", chat_ids=[str(OWNER_CHAT_ID)])


def send_alert_photo(caption: str, photo_path: str) -> None:
    """Send a private owner alert with a photo attachment."""
    if not _owner_is_configured():
        logger.info("[TG-disabled-photo] %s :: %s", caption, photo_path)
        return
    _send_photo_to_audience(photo_path, caption, role="owner", chat_ids=[str(OWNER_CHAT_ID)])


def send_public_alert(message: str) -> None:
    """Send a sanitized viewer-facing broadcast update."""
    if not _viewer_is_configured():
        logger.info("[Viewer-disabled] %s", message)
        return
    _send_to_audience(message, role="viewer", chat_ids=_viewer_targets())


def send_buy_signal(
    symbol: str,
    shares: int,
    price: float,
    signal_type: str,
    strength: float,
) -> None:
    """Send a buy execution alert."""
    msg = "\n".join(
        [
            f"<b>{format_trade_activity_html(date=None, action='BUY', symbol=symbol, shares=shares, price=price, include_date=False)} Executed ✅</b>",
            f"Signal: {signal_type} • Score: {strength:.2f} • Value: NPR {shares * price:,.0f}",
        ]
    )
    send_alert(msg)
    if VIEWER_SHOW_TRADES:
        send_public_alert(
            f"<b>{format_trade_activity_html(date=None, action='BUY', symbol=symbol, shares=shares, price=price, include_date=False)} Executed ✅</b>\n"
            f"Signal: {signal_type}"
        )


def send_sell_signal(
    symbol: str,
    shares: int,
    price: float,
    reason: str,
    pnl: float,
    pnl_pct: float,
) -> None:
    """Send a sell execution alert."""
    msg = "\n".join(
        [
            f"<b>{format_trade_activity_html(date=None, action='SELL', symbol=symbol, shares=shares, price=price, pnl=pnl, include_date=False)} Executed ✅</b>",
            f"Reason: {reason} • Return: {pnl_pct:+.1%}",
        ]
    )
    send_alert(msg)
    if VIEWER_SHOW_TRADES:
        send_public_alert(
            f"<b>{format_trade_activity_html(date=None, action='SELL', symbol=symbol, shares=shares, price=price, pnl=pnl, include_date=False)} Executed ✅</b>\n"
            f"Return: {pnl_pct:+.1%}"
        )


def send_daily_summary(
    portfolio_value: float,
    day_pnl: float,
    open_positions: int,
    signals_generated: int,
    capital: float,
    since_start_return: Optional[float] = None,
    portfolio_day_return: Optional[float] = None,
    benchmark_day_return: Optional[float] = None,
    benchmark_return: Optional[float] = None,
    alpha_points: Optional[float] = None,
    realized_pnl: Optional[float] = None,
) -> None:
    """Send end-of-day portfolio summary."""
    total_return = (portfolio_value / capital - 1) if capital > 0 else 0
    since_start = total_return if since_start_return is None else float(since_start_return)
    owner_lines = [
        "<b>DAILY SUMMARY</b>",
        "<code>",
        f"  Portfolio  : NPR {portfolio_value:,.0f}",
        f"  Day P&L    : NPR {day_pnl:+,.0f}",
    ]
    if portfolio_day_return is not None:
        owner_lines.append(f"  Today      : {portfolio_day_return:+.2%}")
    if benchmark_day_return is not None:
        owner_lines.append(f"  NEPSE Today: {benchmark_day_return:+.2%}")
    owner_lines.append(f"  Since Start: {since_start:+.2%}")
    if benchmark_return is not None:
        owner_lines.append(f"  NEPSE All  : {benchmark_return:+.2%}")
    if alpha_points is not None:
        owner_lines.append(f"  Alpha      : {alpha_points:+.2f} pts")
    if realized_pnl is not None:
        owner_lines.append(f"  Realized   : NPR {float(realized_pnl):+,.0f}")
    owner_lines.extend([
        f"  Open Names : {open_positions}",
        f"  Signals    : {signals_generated}",
        "</code>",
    ])
    msg = "\n".join(owner_lines)
    send_alert(msg)
    if VIEWER_DAILY_POSTS:
        public_lines = [
            "<b>DAILY SUMMARY</b>",
            "<code>",
            f"  Portfolio  : NPR {portfolio_value:,.0f}",
            f"  Day P&L    : NPR {day_pnl:+,.0f}",
        ]
        if portfolio_day_return is not None:
            public_lines.append(f"  Today      : {portfolio_day_return:+.2%}")
        if benchmark_day_return is not None:
            public_lines.append(f"  NEPSE Today: {benchmark_day_return:+.2%}")
        public_lines.append(f"  Since Start: {since_start:+.2%}")
        if benchmark_return is not None:
            public_lines.append(f"  NEPSE All  : {benchmark_return:+.2%}")
        if alpha_points is not None:
            public_lines.append(f"  Alpha      : {alpha_points:+.2f} pts")
        public_lines.extend([
            f"  Open Names : {open_positions}",
            "</code>",
        ])
        public_msg = "\n".join(public_lines)
        send_public_alert(public_msg)


def send_error(error_message: str) -> None:
    """Send a system error alert."""
    msg = f"<b>ERROR</b>\n<code>{error_message[:500]}</code>"
    send_alert(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Short-Term / Dual-Portfolio Alerts
# ─────────────────────────────────────────────────────────────────────────────

def send_short_term_alert(
    symbol: str,
    action: str,
    price: float,
    target_exit: str,
    yield_pct: float,
    reasoning: str,
) -> None:
    """Send a short-term event-driven trade alert (buy/sell)."""
    is_buy = action.upper() == "BUY"
    emoji = "\U0001f7e2" if is_buy else "\U0001f534"  # green/red circle
    action_text = "BUY" if is_buy else "SELL"

    msg = (
        f"{emoji} <b>[SHORT-TERM] {action_text} {symbol}</b>\n"
        f"<code>"
        f"  Price     : NPR {price:,.1f}\n"
        f"  Target Exit: {target_exit}\n"
        f"  Yield     : {yield_pct:.1f}%\n"
        f"  Catalyst  : {reasoning}"
        f"</code>"
    )
    send_alert(msg)


def send_weekly_scorecard(
    long_term_stats: dict,
    short_term_stats: dict,
) -> None:
    """Send weekly portfolio scorecard showing wins/losses for both portfolios.

    Args:
        long_term_stats: {"wins": int, "losses": int, "pnl": float, "nav": float}
        short_term_stats: {"wins": int, "losses": int, "pnl": float, "nav": float,
                          "upcoming_events": list[str]}
    """
    lt = long_term_stats
    st = short_term_stats

    lt_emoji = "\U0001f7e2" if lt.get("pnl", 0) >= 0 else "\U0001f534"
    st_emoji = "\U0001f7e2" if st.get("pnl", 0) >= 0 else "\U0001f534"

    lines = [
        "<b>WEEKLY SCORECARD</b>\n",
        f"{lt_emoji} <b>Long-Term Portfolio</b>",
        f"<code>"
        f"  W/L     : {lt.get('wins', 0)}W / {lt.get('losses', 0)}L\n"
        f"  Week P&L: NPR {lt.get('pnl', 0):+,.0f}\n"
        f"  NAV     : NPR {lt.get('nav', 0):,.0f}"
        f"</code>\n",
        f"{st_emoji} <b>Short-Term Portfolio</b>",
        f"<code>"
        f"  W/L     : {st.get('wins', 0)}W / {st.get('losses', 0)}L\n"
        f"  Week P&L: NPR {st.get('pnl', 0):+,.0f}\n"
        f"  NAV     : NPR {st.get('nav', 0):,.0f}"
        f"</code>",
    ]

    upcoming = st.get("upcoming_events", [])
    if upcoming:
        lines.append(f"\n<b>Next Week Events:</b>")
        for event in upcoming[:5]:
            lines.append(f"  \u2022 {event}")

    send_alert("\n".join(lines))


def send_corporate_action_calendar(upcoming_events: list) -> None:
    """Send weekly digest of upcoming high-yield corporate actions.

    Args:
        upcoming_events: list of dicts with keys:
            symbol, bookclose_date, cash_dividend_pct, bonus_share_pct, entry_date
    """
    if not upcoming_events:
        send_alert("<b>CORPORATE ACTION CALENDAR</b>\n\nNo high-yield events in the next 2 weeks.")
        return

    lines = [f"<b>CORPORATE ACTION CALENDAR</b> ({len(upcoming_events)} events)\n"]

    for ev in upcoming_events[:10]:
        symbol = ev.get("symbol", "?")
        bc_date = ev.get("bookclose_date", "?")
        cash_div = ev.get("cash_dividend_pct", 0)
        bonus = ev.get("bonus_share_pct", 0)
        entry_date = ev.get("entry_date", "?")

        yield_parts = []
        if cash_div and cash_div >= 5:
            yield_parts.append(f"Div {cash_div}%")
        if bonus and bonus >= 10:
            yield_parts.append(f"Bonus {bonus}%")
        yield_str = " + ".join(yield_parts) if yield_parts else "—"

        lines.append(
            f"<b>{symbol}</b>\n"
            f"<code>"
            f"  Bookclose: {bc_date}\n"
            f"  Entry By : {entry_date}\n"
            f"  Yield    : {yield_str}"
            f"</code>\n"
        )

    send_alert("\n".join(lines))
