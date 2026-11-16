"""Your nepsequant.com strategies, used by the terminal.

When the terminal is linked to a nepsequant.com account, its strategies come
from the website: the Lab rule sets, Lab signal blends and saved Screener
screens the user built (and backtested) there — and, for the desk owner only,
the website's research strategies (kind ``research``). A research strategy
can carry a ``gate`` (a market filter: invested or cash, since when, and the
gauges behind it, described by the website); while the gate says cash it holds
nothing. The terminal does not build or
edit strategies. It lists them, lets the user assign one, shows that
strategy's picks for today (computed by the website, the same way it shows
them), and can auto-trade the assigned strategy on the shared paper book
while the terminal is open (apps/tui/io/strategy_autotrader.py).

Network calls happen in ``refresh()`` / ``picks()`` / ``assign()``, which the
terminal runs in background threads; the screens read the cached copies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import quote

from apps.cli import connect

PICKS_TTL_S = 300.0  # the website recomputes picks at most a few times a day
WARMING_RETRY_S = 30.0
KIND_LABEL = {"rules": "RULES", "blend": "BLEND", "screen": "SCREEN", "research": "RESEARCH"}

# Trading rules used when a strategy carries none (a saved screen, or a rule
# set that was never backtested). Shown on screen as "default".
DEFAULT_PARAMS = {
    "max_positions": 5,
    "rebalance_days": 5,
    "holding_days": 40,
    "stop_loss_pct": 8.0,
    "trailing_stop_pct": 10.0,
    "sector_limit": 35.0,
}


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _sector_pct(value: Any) -> Optional[float]:
    """The website sends the sector cap as a fraction (0.35 = 35%)."""
    v = _f(value)
    if v is None:
        return None
    return v * 100.0 if 0 < v <= 1.0 else v


def trading_params(strategy: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The strategy's trading rules and where they came from.

    A rule the strategy sets to 0 is OFF: ``stop_loss_pct`` 0 = no stop-loss,
    ``trailing_stop_pct`` 0 = no trailing stop, ``holding_days`` 0 = no
    holding limit. ``sector_limit`` sent as null = no sector cap (None here).
    A rule the strategy does not send keeps the default."""
    raw = dict(strategy.get("params") or {})
    out: dict[str, Any] = dict(DEFAULT_PARAMS)
    used = False
    for key in ("max_positions", "rebalance_days"):
        v = _f(raw.get(key))
        if v is not None and v > 0:
            out[key] = int(v)
            used = True
    v = _f(raw.get("holding_days"))
    if v is not None and v >= 0:
        out["holding_days"] = int(v)  # 0 = off
        used = True
    # *_pct fields arrive in percent (8.0 = 8%); the sector cap as a fraction.
    for key in ("stop_loss_pct", "trailing_stop_pct"):
        v = _f(raw.get(key))
        if v is not None and v >= 0:
            out[key] = v  # 0 = off
            used = True
    if "sector_limit" in raw and raw.get("sector_limit") is None and strategy.get("params"):
        out["sector_limit"] = None  # no cap
        used = True
    else:
        v = _sector_pct(raw.get("sector_limit"))
        if v is not None and v > 0:
            out["sector_limit"] = v
            used = True
    source = str(strategy.get("params_source") or "") if used else "default"
    return out, (source or "strategy")


def gate_of(strategy: Optional[dict[str, Any]], picks: Optional["Picks"] = None) -> Optional[dict[str, Any]]:
    """The strategy's market filter: the picks' copy (fresher) or the list's."""
    if picks is not None and isinstance(picks.gate, dict):
        return picks.gate
    g = (strategy or {}).get("gate")
    return g if isinstance(g, dict) else None


def gate_title(gate: Optional[dict[str, Any]]) -> str:
    """``MARKET FILTER: CASH since 2026-04-07`` (empty without a gate)."""
    if not gate:
        return ""
    pos = "INVESTED" if gate.get("invested") else "CASH"
    since = f" since {gate['since']}" if gate.get("since") else ""
    return f"{gate.get('filter') or 'MARKET FILTER'}: {pos}{since}"


def gate_lines(gate: Optional[dict[str, Any]]) -> list[str]:
    """Plain lines for the detail panel: the position and the gauges, worded
    by the website (``label`` / ``text`` per gauge, ``rule`` for the filter)."""
    if not gate:
        return []
    when = f" (close of {gate['as_of']})" if gate.get("as_of") else ""
    gauges = gate.get("gauges") or {}
    votes = gate.get("votes")
    lines = [gate_title(gate) + when
             + (f" · {votes} of {len(gauges)} gauges on" if votes is not None and gauges else "")
             + (f" ({gate['rule']})" if gate.get("rule") else "")]
    parts = []
    for key, g in gauges.items():
        label = str(g.get("label") or key)
        text = f" {g['text']}" if g.get("text") else ""
        parts.append(f"{label} {'ON' if g.get('on') else 'off'}{text}")
    if parts:
        lines.append("Gauges: " + " · ".join(parts))
    if gate.get("invested") and gate.get("selection_date"):
        lines.append(f"Holding the selection made at the close of {gate['selection_date']}.")
    elif not gate.get("invested"):
        lines.append("Holds nothing while the filter says cash; auto-trade sells what it bought.")
    if gate.get("stale"):
        lines.append("Inputs stale on the website: " + "; ".join(str(x) for x in gate["stale"]))
    return lines


@dataclass
class Picks:
    strategy_id: str
    as_of: Optional[str] = None
    warming: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    fetched_at: float = 0.0
    error: str = ""
    gate: Optional[dict[str, Any]] = None
    selection_date: Optional[str] = None


Request = Callable[..., dict]


class WebsiteStrategies:
    def __init__(self, key: str, *, request: Optional[Request] = None) -> None:
        self._key = key
        self._request: Request = request or connect.request_json
        self._lock = threading.Lock()
        self.strategies: list[dict[str, Any]] = []
        self.assigned: Optional[str] = None
        self.notice = ""
        self.error = ""
        self.loaded = False
        self._picks: dict[str, Picks] = {}

    @classmethod
    def from_book(cls, book: Any) -> Optional["WebsiteStrategies"]:
        """Strategies for the account the shared paper book is linked to."""
        key = getattr(book, "_key", None)
        return cls(key, request=getattr(book, "_request", None)) if key else None

    # ── the list ────────────────────────────────────────────────────────────
    def refresh(self) -> bool:
        try:
            data = self._request("/api/cli/strategies", self._key)
        except connect.ConnectError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:
            self.error = f"Could not read your strategies: {exc}"
            return False
        with self._lock:
            self.strategies = [dict(s) for s in (data.get("strategies") or []) if s.get("id")]
            ids = {s["id"] for s in self.strategies}
            assigned = data.get("assigned")
            self.assigned = assigned if assigned in ids else None
            self.notice = str(data.get("notice") or "")
            self.loaded = True
        self.error = ""
        return True

    def get(self, strategy_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not strategy_id:
            return None
        return next((s for s in self.strategies if s.get("id") == strategy_id), None)

    def assigned_strategy(self) -> Optional[dict[str, Any]]:
        return self.get(self.assigned)

    def assign(self, strategy_id: Optional[str]) -> tuple[bool, str]:
        try:
            data = self._request(
                "/api/cli/strategy-assignment", self._key, method="PUT", body={"id": strategy_id}
            )
        except connect.ConnectError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Could not save the assignment: {exc}"
        self.assigned = data.get("assigned")
        name = (self.get(self.assigned) or {}).get("name") or ""
        return True, (f"Assigned {name}" if self.assigned else "No strategy assigned")

    # ── today's picks ───────────────────────────────────────────────────────
    def cached_picks(self, strategy_id: Optional[str]) -> Optional[Picks]:
        return self._picks.get(strategy_id or "")

    def picks(self, strategy_id: str, *, force: bool = False, limit: int = 20) -> Picks:
        """Today's picks, fetched at most every few minutes (sooner while the
        website is still preparing them)."""
        now = time.time()
        cached = self._picks.get(strategy_id)
        if cached and not force and not cached.error:
            ttl = WARMING_RETRY_S if cached.warming else PICKS_TTL_S
            if now - cached.fetched_at < ttl:
                return cached
        try:
            data = self._request(f"/api/cli/strategies/{quote(strategy_id, safe='')}/signals?limit={int(limit)}", self._key)
            picks = Picks(
                strategy_id=strategy_id,
                as_of=data.get("as_of"),
                warming=bool(data.get("warming")),
                rows=[dict(r) for r in (data.get("picks") or []) if r.get("symbol")],
                note=str(data.get("note") or ""),
                fetched_at=now,
                gate=data.get("gate") if isinstance(data.get("gate"), dict) else None,
                selection_date=data.get("selection_date"),
            )
        except Exception as exc:
            picks = Picks(strategy_id=strategy_id, fetched_at=now, error=str(exc),
                          rows=list(cached.rows) if cached else [], as_of=cached.as_of if cached else None,
                          gate=cached.gate if cached else None,
                          selection_date=cached.selection_date if cached else None)
        self._picks[strategy_id] = picks
        return picks


def metric(strategy: dict[str, Any], key: str) -> Optional[float]:
    return _f((strategy.get("backtest") or {}).get(key))


def describe(strategy: dict[str, Any]) -> list[str]:
    """Plain lines for the strategy detail panel."""
    params, source = trading_params(strategy)
    lines = [
        f"{strategy.get('name') or strategy.get('id')}  ({KIND_LABEL.get(str(strategy.get('kind')), str(strategy.get('kind') or '').upper())})",
    ]
    if strategy.get("summary"):
        lines.append(str(strategy["summary"]))
    if strategy.get("description"):
        lines.append(str(strategy["description"]))
    lines += gate_lines(strategy.get("gate"))
    bt = strategy.get("backtest") or {}
    if bt:
        parts = []
        for label, key, fmt in (
            ("Return", "total_return_pct", "{:+.1f}%"), ("CAGR", "cagr_pct", "{:+.1f}%"),
            ("vs NEPSE", "vs_nepse_pct_points", "{:+.1f}pp"), ("Sharpe", "sharpe", "{:.2f}"),
            ("Max DD", "max_drawdown_pct", "{:.1f}%"), ("Win", "win_rate_pct", "{:.0f}%"),
            ("Trades", "trade_count", "{:.0f}"),
        ):
            v = _f(bt.get(key))
            if v is not None:
                parts.append(f"{label} {fmt.format(v)}")
        window = f"{bt.get('start')} → {bt.get('end')}" if bt.get("start") else ""
        lines.append("Backtest " + (f"{window}: " if window else "") + "  ".join(parts))
        if bt.get("verdict"):
            d = bt.get("design") or {}
            alpha, t, bar = _f(bt.get("alpha_pct")), _f(bt.get("alpha_t")), _f(bt.get("bar_t"))
            line = f"Research verdict {bt['verdict']}"
            if alpha is not None and t is not None:
                line += f": alpha {alpha:+.1f}%/yr, t {t:.2f}" + (f" (bar {bar:.2f})" if bar is not None else "")
            if _f(d.get("alpha_pct")) is not None and _f(d.get("alpha_t")) is not None:
                line += f"; design alpha {float(d['alpha_pct']):+.1f}%, t {float(d['alpha_t']):.2f}"
            lines.append(line)
    else:
        lines.append("Not backtested yet on nepsequant.com.")
    hold = f"sell after {params['holding_days']} days" if params["holding_days"] else "no holding limit"
    stop = f"stop {params['stop_loss_pct']:.0f}%" if params["stop_loss_pct"] else "no stop-loss"
    trail = f"trailing {params['trailing_stop_pct']:.0f}%" if params["trailing_stop_pct"] else "no trailing stop"
    cap = f"sector cap {params['sector_limit']:.0f}%" if params["sector_limit"] is not None else "no sector cap"
    rules = (
        f"Hold up to {params['max_positions']} stocks · rebalance every {params['rebalance_days']} trading days · "
        f"{hold} · {stop} · {trail} · {cap}"
    )
    if strategy.get("gate") is not None or strategy.get("kind") == "research":
        rules += " · only while its market filter says invested"
    where = {"backtest": "from its backtest", "default": "default", "research": "from the research"}.get(source, "from the strategy")
    lines.append(f"Trading rules ({where}): {rules}")
    return lines
