"""Rows for the Signals tab's BROKER FLOOR and TOP BROKERS panels.

One renderer for both data sources (the local scraped database and the
nepsequant.com floorsheet), so a row reads the same either way. Records are
mappings with the ``broker_signals_v2`` / ``broker_summary`` column names
(``sqlite3.Row`` works as is); see apps/tui/io/floor_metrics.py.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from rich.text import Text

from apps.tui.render.cells import _dim_text, _sym_text, _vol_text
from apps.tui.theme import DIM, GAIN_HI, LOSS_HI, WHITE

# (label, key, width) — explicit widths for perfect alignment
BROKER_COLS = [
    ("SYMBOL",   "bsym",    8),
    ("SIGNAL",   "btype",  14),
    ("BUY HHI",  "bhhi_b",  8),
    ("SLR HHI",  "bhhi_s",  8),
    ("CIRC",     "bcirc",   6),
    ("SELF%",    "bself",   6),
    ("SM SCORE", "bscore",  9),
    ("TRADES",   "btr",     7),
]
TOP_BROKER_COLS = [
    ("BROKER", "tbroker", 8),
    ("BUY QTY", "tbuy", 10),
    ("SELL QTY", "tsell", 10),
    ("NET QTY", "tnet", 10),
    ("TRADES", "ttrades", 8),
]


def floor_title(date: str, records: list[Mapping[str, Any]]) -> str:
    n_circ = sum(1 for _r in records if float(_r["circular_score"] or 0) > 0.15)
    n_sm = sum(1 for _r in records if float(_r["smart_money_score"] or 0) > 0.08)
    return (
        f"BROKER FLOOR  {date}  |"
        f"  {n_circ} circular/pump  ·  {n_sm} smart money  ·  {len(records)} total"
    )


def message_row(text: str) -> list[Text]:
    return [_dim_text("—"), _dim_text(text), *[Text("")] * 6]


def floor_signal_rows(records: Iterable[Mapping[str, Any]]) -> list[list[Text]]:
    rows: list[list[Text]] = []
    for _r in records:
        circ  = float(_r["circular_score"]  or 0.0)
        self_ = float(_r["self_trade_pct"]  or 0.0)  # self-trade fraction
        sm    = float(_r["smart_money_score"] or 0.0)
        hhi_b = float(_r["hhi_buy"]          or 0.0)
        hhi_s = float(_r["hhi_sell"]         or 0.0)
        trades = int(_r["n_trades"]          or 0)
        # Microstructure-adjusted score
        _ms = _r["micro_score"]
        micro = float(_ms) if _ms is not None else 0.5
        display_score = sm * (0.5 + 0.5 * micro)
        # Signal classification — self_ is now direct self-trade evidence
        if self_ > 0.20 and circ > 0.15:
            sig_label, sig_style = "[!] WASH+SELF ", f"bold {LOSS_HI}"
        elif self_ > 0.10:
            sig_label, sig_style = "[!] SELF-TRADE", LOSS_HI
        elif circ > 0.15:
            sig_label, sig_style = "[!] CIRCULAR  ", LOSS_HI
        elif display_score > 0.25:
            sig_label, sig_style = "[+] SMART $   ", f"bold {GAIN_HI}"
        else:
            sig_label, sig_style = "[~] ACCUM     ", GAIN_HI
        rows.append([
            _sym_text(str(_r["symbol"])),
            Text(sig_label, style=sig_style),
            Text(f"{hhi_b:5.2f}", style=GAIN_HI if hhi_b > 0.30 else WHITE),
            Text(f"{hhi_s:5.2f}", style=LOSS_HI if hhi_s > 0.20 else GAIN_HI),
            Text(f"{circ*100:4.0f}%", style=LOSS_HI if circ > 0.15 else WHITE),
            Text(f"{self_*100:4.0f}%", style=LOSS_HI if self_ > 0.10 else WHITE),
            Text(f"{display_score:7.3f}" if display_score > 0 else "      —",
                 style=GAIN_HI if display_score > 0.15 else WHITE),
            _dim_text(f"{trades:6d}"),
        ])
    return rows


def top_broker_rows(records: Iterable[Mapping[str, Any]]) -> list[list[Text]]:
    rows: list[list[Text]] = []
    for _r in records:
        net_qty = int(_r["net_qty"] or 0)
        net_style = GAIN_HI if net_qty > 0 else (LOSS_HI if net_qty < 0 else DIM)
        rows.append([
            Text(str(_r["broker_code"]), style=WHITE),
            _vol_text(int(_r["buy_qty"] or 0)),
            _vol_text(int(_r["sell_qty"] or 0)),
            Text(f"{net_qty:,}", style=net_style),
            _dim_text(f"{int(_r['total_trades'] or 0):,}"),
        ])
    return rows


SITE = "nepsequant.com"


def website_floor_panels(view: Optional[Any], error: str = "") -> tuple[str, list, list, list]:
    """(title, floor rows, top-broker columns, top-broker rows) for a terminal
    linked to nepsequant.com; ``view`` is a website_floor.FloorView or None."""
    if view is None:
        if error:
            return (f"BROKER FLOOR  —  {SITE} floorsheet not loaded: {error[:120]}",
                    [message_row("Not loaded")], [], [])
        return (f"BROKER FLOOR  —  loading the floorsheet from {SITE}…", [message_row("Loading…")], [], [])
    day = view.day
    if not day.date or not day.count:
        return (f"BROKER FLOOR  —  no floorsheet on {SITE} yet", [message_row("No data yet")], [], [])
    title = f"{floor_title(day.date, view.signals)}  ·  {SITE} · {day.engine}"
    if day.complete is False:
        title += " · day still being captured"
    if error:
        title += " · last refresh failed"
    return title, floor_signal_rows(view.signals), TOP_BROKER_COLS, top_broker_rows(view.top_brokers)
