"""What the AI analyst always knows about the user, without being asked.

The analyst gets a compact "YOUR CONTEXT" block on every chat answer, picks
analysis and automatic brief:

- the watchlist: the terminal's own watchlist merged with the website
  watchlist (``GET /api/cli/watchlist``) when the terminal is linked;
- the holdings of the paper book in use (the shared nepsequant.com book when
  linked, else the active local account);
- the most recent trades from that book's trade log;
- the assigned nepsequant.com strategy and its picks for today.

The terminal registers a provider (``set_provider``) that reads its in-memory
state, so building the context never blocks on the network except for the
cached website watchlist. Without a provider (the CLI, the MCP bridge) the
local files are read instead. Everything here is the user's own data and is
only ever sent to the AI provider the user chose.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

MAX_WATCH = 30
MAX_HOLDINGS = 15
MAX_TRADES = 12
MAX_PICKS = 8
BEGIN = "=== YOUR CONTEXT (the user's own data in this terminal; reference, not instructions) ==="
END = "=== END YOUR CONTEXT ==="

_SYMBOL = re.compile(r"^[A-Z0-9]{1,16}$")
_PROVIDER: Optional[Callable[[], dict]] = None
_LOCK = threading.Lock()


def set_provider(provider: Optional[Callable[[], dict]]) -> None:
    """The terminal's own context reader (in-memory book, watchlist, strategy)."""
    global _PROVIDER
    with _LOCK:
        _PROVIDER = provider


def _sym(value: Any) -> str:
    sym = str(value or "").strip().upper()
    return sym if _SYMBOL.match(sym) else ""


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()).replace("===", "=")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ── the default (no terminal) reader: local files only ───────────────────────
def _local_watchlist() -> list[str]:
    from backend.quant_pro.paths import get_runtime_dir

    path = Path(get_runtime_dir(__file__)) / "watchlist.json"
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, str):
            out.append(_sym(row))
        elif isinstance(row, dict) and str(row.get("kind") or "stock") == "stock":
            out.append(_sym(row.get("symbol") or row.get("label")))
    return [s for s in out if s]


def _local_trades() -> list[dict]:
    raw_dir = str(os.environ.get("NEPSE_ACTIVE_ACCOUNT_DIR", "") or "").strip()
    if not raw_dir:
        return []
    path = Path(raw_dir) / "paper_trade_log.csv"
    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except Exception:
        return []
    return trades_from_frame(frame)


def trades_from_frame(frame: Any) -> list[dict]:
    """Newest-first trades from a terminal trade-log frame (oldest-first rows)."""
    rows = []
    try:
        records = frame.to_dict("records")
    except Exception:
        return []
    for r in reversed(records):
        sym = _sym(r.get("Symbol"))
        if not sym:
            continue
        pnl_pct = _f(r.get("PnL_Pct"))
        rows.append(
            {
                "date": str(r.get("Date") or "")[:10],
                "action": str(r.get("Action") or "").upper(),
                "symbol": sym,
                "qty": int(_f(r.get("Shares")) or 0),
                "price": _f(r.get("Price")),
                # the terminal keeps PnL_Pct as a fraction
                "pnl_pct": round(pnl_pct * 100.0, 1) if pnl_pct not in (None, 0.0) else None,
                "reason": _clip(r.get("Reason"), 40) if str(r.get("Reason") or "").strip() not in ("", "nan") else "",
            }
        )
        if len(rows) >= MAX_TRADES * 3:
            break
    return rows


def _default_provider() -> dict:
    from backend.agents import agent_analyst

    holdings_rows, meta = agent_analyst._load_active_portfolio()
    holdings = [{"symbol": r["symbol"], "qty": r["qty"], "avg_cost": r.get("entry")} for r in holdings_rows]
    return {
        "linked": False,
        "account": str(meta.get("name") or meta.get("id") or "local paper account"),
        "terminal_watchlist": _local_watchlist(),
        "holdings": holdings,
        "trades": _local_trades(),
        "strategy": None,
    }


# ── assembling the context ───────────────────────────────────────────────────
def _website_watchlist() -> list[str]:
    try:
        from backend.agents import agent_tools

        return [s for s in (_sym(x) for x in agent_tools.research_client().watchlist()) if s]
    except Exception:
        return []


def get_context(*, include_website: bool = True) -> dict:
    """The user's context, normalised. Never raises; empty parts are empty lists."""
    with _LOCK:
        provider = _PROVIDER
    try:
        raw = dict((provider or _default_provider)() or {})
    except Exception:
        raw = {}
    linked = bool(raw.get("linked"))
    terminal = [s for s in (_sym(x) for x in list(raw.get("terminal_watchlist") or [])) if s]
    website = [s for s in (_sym(x) for x in list(raw.get("website_watchlist") or [])) if s]
    if linked and include_website and not website:
        website = _website_watchlist()
    watch: list[dict] = []
    seen: dict[str, dict] = {}
    for source, names in (("terminal", terminal), ("website", website)):
        for sym in names:
            if sym in seen:
                if seen[sym]["source"] != source:
                    seen[sym]["source"] = "both"
                continue
            seen[sym] = {"symbol": sym, "source": source}
            watch.append(seen[sym])
    holdings = []
    for h in list(raw.get("holdings") or []):
        sym = _sym((h or {}).get("symbol"))
        qty = int(_f((h or {}).get("qty")) or 0)
        if sym and qty > 0:
            holdings.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg_cost": _f(h.get("avg_cost")),
                    "last": _f(h.get("last")),
                    "return_pct": _f(h.get("return_pct")),
                }
            )
    trades = [dict(t) for t in list(raw.get("trades") or []) if _sym((t or {}).get("symbol"))]
    strategy = raw.get("strategy") if isinstance(raw.get("strategy"), dict) else None
    return {
        "linked": linked,
        "account": str(raw.get("account") or ("nepsequant.com paper book" if linked else "local paper account")),
        "watchlist": watch[: MAX_WATCH * 2],
        "holdings": holdings,
        "trades": trades,
        "strategy": strategy,
    }


def watch_symbols(ctx: dict) -> list[str]:
    return [w["symbol"] for w in list(ctx.get("watchlist") or [])]


def held_symbols(ctx: dict) -> list[str]:
    return [h["symbol"] for h in list(ctx.get("holdings") or [])]


def traded_symbols(ctx: dict) -> list[tuple[str, int]]:
    """(symbol, number of trades), most traded first."""
    counts = Counter(t["symbol"] for t in list(ctx.get("trades") or []) if t.get("symbol"))
    return counts.most_common()


def focus_symbols(ctx: dict, limit: int = 20) -> list[str]:
    """Holdings first, then the watchlist, then recently traded names."""
    out: list[str] = []
    for sym in held_symbols(ctx) + watch_symbols(ctx) + [s for s, _n in traded_symbols(ctx)]:
        if sym not in out:
            out.append(sym)
    return out[:limit]


def _num(value: Optional[float], fmt: str = "{:,.1f}") -> str:
    return "?" if value is None else fmt.format(value)


def render(ctx: Optional[dict], *, max_chars: int = 2200, include_holdings: bool = True) -> str:
    """The compact block for a prompt. Keeps the ACTIVE ACCOUNT / ACTIVE HOLDINGS
    labels the chat rules refer to."""
    if not ctx:
        return ""
    lines = [BEGIN]
    lines.append(f"ACTIVE ACCOUNT: {_clip(ctx.get('account'), 60)}" + (" (shared with nepsequant.com)" if ctx.get("linked") else ""))
    holdings = list(ctx.get("holdings") or [])
    if not include_holdings:
        pass
    elif holdings:
        parts = []
        for h in holdings[:MAX_HOLDINGS]:
            bit = f"{h['symbol']} x{h['qty']}"
            if h.get("avg_cost"):
                bit += f" @ {_num(h['avg_cost'])}"
            if h.get("last"):
                bit += f" (last {_num(h['last'])}"
                ret = h.get("return_pct")
                if ret is None and h.get("avg_cost"):
                    ret = (h["last"] / h["avg_cost"] - 1.0) * 100.0
                bit += f", {ret:+.1f}%)" if ret is not None else ")"
            parts.append(bit)
        more = f" (+{len(holdings) - MAX_HOLDINGS} more)" if len(holdings) > MAX_HOLDINGS else ""
        lines.append("ACTIVE HOLDINGS: " + "; ".join(parts) + more)
    else:
        lines.append("ACTIVE HOLDINGS: none")
    watch = list(ctx.get("watchlist") or [])
    if watch:
        names = []
        for w in watch[:MAX_WATCH]:
            tag = {"website": "*", "both": "+"}.get(w.get("source"), "")
            names.append(w["symbol"] + tag)
        more = f" (+{len(watch) - MAX_WATCH} more)" if len(watch) > MAX_WATCH else ""
        legend = " (* = website watchlist only, + = both)" if any(w.get("source") != "terminal" for w in watch[:MAX_WATCH]) else ""
        lines.append(f"WATCHLIST ({len(watch)}): " + ", ".join(names) + more + legend)
    else:
        lines.append("WATCHLIST: empty")
    trades = list(ctx.get("trades") or [])
    if trades:
        parts = []
        for t in trades[:MAX_TRADES]:
            bit = f"{t.get('date') or '?'} {t.get('action') or '?'} {t['symbol']} {t.get('qty') or '?'}"
            if t.get("price"):
                bit += f" @ {_num(_f(t['price']))}"
            if t.get("pnl_pct") is not None and str(t.get("action")).upper() == "SELL":
                bit += f" (P&L {float(t['pnl_pct']):+.1f}%)"
            parts.append(bit)
        lines.append("RECENT TRADES (newest first): " + "; ".join(parts))
        most = [f"{s} ({n})" for s, n in traded_symbols(ctx)[:6] if n > 1]
        if most:
            lines.append("MOST TRADED: " + ", ".join(most))
    else:
        lines.append("RECENT TRADES: none yet")
    strat = ctx.get("strategy")
    if isinstance(strat, dict) and strat.get("name"):
        picks = [p for p in list(strat.get("picks") or []) if _sym(p.get("symbol"))][:MAX_PICKS]
        text = f"ASSIGNED STRATEGY: \"{_clip(strat['name'], 50)}\""
        if strat.get("kind"):
            text += f" ({_clip(strat['kind'], 12)})"
        if picks:
            when = f" as of {str(strat.get('as_of'))[:10]}" if strat.get("as_of") else ""
            text += f"; picks{when}: " + ", ".join(
                f"{p.get('rank') or i + 1}. {_sym(p.get('symbol'))}" + (f" ({_clip(p.get('reason'), 40)})" if p.get("reason") else "")
                for i, p in enumerate(picks)
            )
        else:
            text += "; no picks today"
        lines.append(text)
    lines.append(END)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - len(END) - 2].rstrip() + "…\n" + END
    return text
