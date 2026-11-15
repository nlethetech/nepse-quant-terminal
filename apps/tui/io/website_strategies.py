"""Your nepsequant.com strategies, used by the terminal.

When the terminal is linked to a nepsequant.com account, its strategies come
from the website: the Lab rule sets, Lab signal blends and saved Screener
screens the user built (and backtested) there. The terminal does not build or
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
KIND_LABEL = {"rules": "RULES", "blend": "BLEND", "screen": "SCREEN"}

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


def _pct(value: Any) -> Optional[float]:
    """A percentage as percent, whether the website sent 0.08 or 8."""
    v = _f(value)
    if v is None:
        return None
    return v * 100.0 if 0 < abs(v) <= 1.0 else v


def trading_params(strategy: dict[str, Any]) -> tuple[dict[str, float], str]:
    """The strategy's trading rules and where they came from."""
    raw = dict(strategy.get("params") or {})
    out = dict(DEFAULT_PARAMS)
    used = False
    for key in ("max_positions", "rebalance_days", "holding_days"):
        v = _f(raw.get(key))
        if v is not None and v > 0:
            out[key] = int(v)
            used = True
    for key in ("stop_loss_pct", "trailing_stop_pct", "sector_limit"):
        v = _pct(raw.get(key))
        if v is not None and v >= 0:
            out[key] = v
            used = True
    source = str(strategy.get("params_source") or "") if used else "default"
    return out, (source or "strategy")


@dataclass
class Picks:
    strategy_id: str
    as_of: Optional[str] = None
    warming: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    fetched_at: float = 0.0
    error: str = ""


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
            )
        except Exception as exc:
            picks = Picks(strategy_id=strategy_id, fetched_at=now, error=str(exc),
                          rows=list(cached.rows) if cached else [], as_of=cached.as_of if cached else None)
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
    else:
        lines.append("Not backtested yet on nepsequant.com.")
    rules = (
        f"Hold up to {params['max_positions']} stocks · rebalance every {params['rebalance_days']} trading days · "
        f"sell after {params['holding_days']} days · stop {params['stop_loss_pct']:.0f}% · "
        f"trailing {params['trailing_stop_pct']:.0f}% · sector cap {params['sector_limit']:.0f}%"
    )
    lines.append(f"Trading rules ({'from its backtest' if source == 'backtest' else 'default' if source == 'default' else 'from the strategy'}): {rules}")
    return lines
