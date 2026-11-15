"""Auto-trade the assigned nepsequant.com strategy on the shared paper book.

While the terminal is open, linked, and the user has switched auto-trading
on, this places PAPER orders on the user's nepsequant.com paper book so the
book follows the assigned strategy with the same trading rules as its
backtest (apps/tui/io/website_strategies.py ``trading_params``):

* **Exits, checked every refresh while NEPSE is open:** stop-loss from the
  buy price, trailing stop from the highest price seen, and the holding
  period. Only positions the auto-trader itself bought are ever sold — stocks
  the user bought by hand are never touched.
* **Entries, on rebalance days only:** the strategy's top picks for today
  (computed by the website), equal-weight (book value / max positions), up to
  the position limit, within the sector cap and the cash available, as market
  orders.
* **Guards:** paper only (the website book), only while the website says
  NEPSE is open, nothing twice in a day for the same stock, a daily order
  cap, picks older than a few days are not bought, and every decision is
  logged. The website itself still applies NEPSE's rules (T+2, same-day,
  price band, hours) and refuses anything that breaks them.

``plan()`` is pure (no I/O) so the rules are easy to test; ``AutoTrader``
holds the small state file and talks to the book.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from backend.quant_pro.paths import ensure_dir, get_runtime_dir

try:  # one auto-trader at a time across terminal windows
    import fcntl  # macOS / Linux
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:  # pragma: no cover - Windows only
    import msvcrt
except ImportError:
    msvcrt = None

STATE_FILE = Path(get_runtime_dir(__file__)) / "website_book" / "autotrader.json"
STALE_PICKS_DAYS = 4  # weekdays: older picks are not bought
FEE_BUFFER = 0.985  # leave room for broker commission, SEBON fee and DP charge
LOG_KEEP = 60


def weekdays_between(start: date, end: date) -> int:
    """Mon–Fri days after ``start`` up to and including ``end`` (NEPSE trades Mon–Fri)."""
    if end <= start:
        return 0
    days = (end - start).days
    full, rest = divmod(days, 7)
    count = full * 5
    for i in range(1, rest + 1):
        if (start + timedelta(days=full * 7 + i)).weekday() < 5:
            count += 1
    return count


def _d(value: Any) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) or math.isinf(out) else out


@dataclass(frozen=True)
class Planned:
    action: str
    symbol: str
    qty: int
    reason: str
    price: float = 0.0
    sector: str = ""


def fresh_state(strategy_id: Optional[str] = None) -> dict[str, Any]:
    return {
        "enabled": False,
        "strategy_id": strategy_id,
        "last_rebalance": None,
        "managed": {},
        "day": None,
        "done_today": [],
        "notes_today": [],
        "log": [],
    }


def plan(
    *,
    today: date,
    params: dict[str, float],
    picks: list[dict[str, Any]],
    picks_as_of: Optional[str],
    positions: list[dict[str, Any]],
    cash: float,
    nav: float,
    open_orders: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[list[Planned], dict[str, Any], list[str]]:
    """The orders the strategy wants now, the updated state, and notes."""
    st = json.loads(json.dumps(state))  # never mutate the caller's copy
    notes: list[str] = []
    if st.get("day") != today.isoformat():
        st["day"] = today.isoformat()
        st["done_today"] = []
        st["notes_today"] = []
    done = {(d["action"], d["symbol"]) for d in st.get("done_today", [])}
    pending = {(str(o.get("action")).upper(), str(o.get("symbol")).upper()) for o in open_orders}

    held = {str(p.get("symbol")).upper(): p for p in positions if int(_f(p.get("quantity"))) > 0}
    managed: dict[str, dict[str, Any]] = st.setdefault("managed", {})
    for sym in list(managed):
        if sym not in held:
            managed.pop(sym)  # sold (by us or by hand): no longer ours to manage

    max_pos = max(1, int(params["max_positions"]))
    max_orders = 2 * max_pos + 2
    budget_orders = max(0, max_orders - len(st.get("done_today", [])))
    orders: list[Planned] = []

    # ── exits: only what the auto-trader bought ──────────────────────────────
    for sym, info in sorted(managed.items()):
        pos = held[sym]
        last = _f(pos.get("last"))
        if last > 0:
            info["hwm"] = max(_f(info.get("hwm")), last)
        bought = _d(info.get("bought"))
        if bought == today or ("SELL", sym) in done or ("SELL", sym) in pending:
            continue
        cost = _f(pos.get("avg_cost"))
        reason = ""
        if last > 0 and cost > 0 and last <= cost * (1 - params["stop_loss_pct"] / 100.0):
            reason = f"stop-loss {params['stop_loss_pct']:.0f}% (bought {cost:,.1f}, now {last:,.1f})"
        elif last > 0 and _f(info.get("hwm")) > 0 and last <= _f(info["hwm"]) * (1 - params["trailing_stop_pct"] / 100.0):
            reason = f"trailing stop {params['trailing_stop_pct']:.0f}% from {float(info['hwm']):,.1f}"
        elif bought and weekdays_between(bought, today) >= int(params["holding_days"]):
            reason = f"held {weekdays_between(bought, today)} trading days (limit {int(params['holding_days'])})"
        if reason and len(orders) < budget_orders:
            orders.append(Planned("SELL", sym, int(_f(pos.get("quantity"))), reason, last))

    # ── entries: rebalance days only ─────────────────────────────────────────
    last_rb = _d(st.get("last_rebalance"))
    due = last_rb is None or weekdays_between(last_rb, today) >= int(params["rebalance_days"])
    if not due:
        return orders, st, notes
    as_of = _d(picks_as_of)
    if not picks:
        notes.append("No picks from the website yet; buying waits.")
        return orders, st, notes
    if as_of is None or weekdays_between(as_of, today) > STALE_PICKS_DAYS:
        notes.append(f"The strategy's picks are from {picks_as_of or 'an unknown date'}; not buying on stale picks.")
        return orders, st, notes

    selling = {o.symbol for o in orders if o.action == "SELL"}
    kept = [s for s in managed if s not in selling]
    pending_buys = [s for (a, s) in pending if a == "BUY"]
    slots = max_pos - len(kept) - len(pending_buys)
    target = nav / max_pos if nav > 0 else 0.0
    budget = max(0.0, cash) * FEE_BUFFER
    sector_value: dict[str, float] = {}
    for s in kept:
        sec = str(managed[s].get("sector") or "")
        if sec:
            sector_value[sec] = sector_value.get(sec, 0.0) + _f(held[s].get("last")) * _f(held[s].get("quantity"))

    for pick in picks:
        if slots <= 0 or len(orders) >= budget_orders:
            break
        sym = str(pick.get("symbol") or "").upper()
        price = _f(pick.get("ltp"))
        sector = str(pick.get("sector") or "")
        if not sym or sym in held or ("BUY", sym) in done or ("SELL", sym) in done or ("BUY", sym) in pending:
            continue
        if price <= 0:
            continue
        if sector and nav > 0 and (sector_value.get(sector, 0.0) + target) / nav * 100.0 > params["sector_limit"] + 1e-9:
            notes.append(f"Skipped {sym}: sector cap {params['sector_limit']:.0f}% for {sector}.")
            continue
        qty = int(min(target, budget) // (price * 1.01))  # 1% room for the market price moving
        if qty < 1:
            notes.append(f"Skipped {sym}: not enough cash for one share.")
            continue
        rank = pick.get("rank")
        orders.append(Planned("BUY", sym, qty, f"rank {rank} in today's picks" if rank else "today's picks", price, sector))
        budget -= qty * price * 1.01
        slots -= 1
        if sector:
            sector_value[sector] = sector_value.get(sector, 0.0) + qty * price
    st["last_rebalance"] = today.isoformat()
    return orders, st, notes


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class AutoTrader:
    """The state file plus one pass of: plan → send → record."""

    def __init__(self, account_key: str, *, path: Path = STATE_FILE) -> None:
        self._account = key_fingerprint(account_key)
        self._path = Path(path)
        self._lock = threading.Lock()
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self._path.read_text())
            if raw.get("account") == self._account:
                return {**fresh_state(), **dict(raw.get("state") or {})}
        except (OSError, ValueError):
            pass
        return fresh_state()

    def save(self) -> None:
        ensure_dir(self._path.parent)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"account": self._account, "state": self.state}, indent=2))
        os.replace(tmp, self._path)

    @property
    def enabled(self) -> bool:
        return bool(self.state.get("enabled"))

    def _other_window_lock(self):
        """An exclusive lock shared by every terminal window on this computer,
        so two open terminals can never auto-trade (and double-buy) at once.
        Returns the open lock file, or None when another window holds it."""
        ensure_dir(self._path.parent)
        handle = open(self._path.with_suffix(".lock"), "a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            return None
        return handle

    @staticmethod
    def _release(handle) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()

    def set_enabled(self, on: bool, strategy_id: Optional[str]) -> None:
        with self._lock:
            self.state = self._load()  # another window may have changed it
            if strategy_id != self.state.get("strategy_id"):
                # a new strategy: keep managing what we bought, rebalance afresh
                self.state["strategy_id"] = strategy_id
                self.state["last_rebalance"] = None
            self.state["enabled"] = bool(on and strategy_id)
            self._log(f"Auto-trade {'ON' if self.state['enabled'] else 'OFF'}" + (f" for {strategy_id}" if on else ""))
            self.save()

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%m-%d %H:%M")
        self.state.setdefault("log", []).append(f"{stamp}  {message}")
        self.state["log"] = self.state["log"][-LOG_KEEP:]

    def run_once(self, *, book: Any, strategies: Any, today: date) -> list[str]:
        """One pass. Returns the lines worth showing; never raises."""
        if not self._lock.acquire(blocking=False):
            return []
        handle = None
        try:
            handle = self._other_window_lock()
            if handle is None:
                return []  # another terminal window is auto-trading right now
            self.state = self._load()  # pick up what other windows recorded
            return self._run(book=book, strategies=strategies, today=today)
        except Exception as exc:  # a bad pass must never take the terminal down
            self._log(f"Auto-trade error: {exc}")
            self.save()
            return [f"Auto-trade error: {exc}"]
        finally:
            if handle is not None:
                self._release(handle)
            self._lock.release()

    def _run(self, *, book: Any, strategies: Any, today: date) -> list[str]:
        sid = self.state.get("strategy_id")
        if not self.enabled or not sid:
            return []
        if strategies.assigned != sid:
            self.state["enabled"] = False
            self._log("Auto-trade OFF: the assigned strategy changed on the website")
            self.save()
            return ["Auto-trade switched off: the assigned strategy changed"]
        if not book.can_trade or book.market_open is not True:
            return []
        from apps.tui.io.website_strategies import trading_params

        strategy = strategies.get(sid) or {}
        params, _source = trading_params(strategy)
        picks = strategies.picks(sid)
        open_rows, _history = book.order_records()
        orders, new_state, notes = plan(
            today=today,
            params=params,
            picks=list(picks.rows),
            picks_as_of=picks.as_of,
            positions=book.positions(),
            cash=book.cash(),
            nav=book.value(),
            open_orders=[{"action": o["action"], "symbol": o["symbol"]} for o in open_rows],
            state=self.state,
        )
        self.state = new_state
        shown: list[str] = []
        for note in notes:
            if note not in self.state.get("notes_today", []):
                self.state.setdefault("notes_today", []).append(note)
                self._log(note)
        for order in orders:
            result = book.submit(order.action, order.symbol, order.qty)
            ok = bool(result.get("ok"))
            self.state.setdefault("done_today", []).append(
                {"action": order.action, "symbol": order.symbol, "ok": ok}
            )
            if ok and order.action == "BUY":
                fill = _f(result.get("fill_price")) or order.price
                self.state["managed"][order.symbol] = {
                    "bought": today.isoformat(), "hwm": fill, "sector": order.sector, "strategy": sid,
                }
            outcome = (
                f"{order.action} {order.symbol} x{order.qty}: "
                + ("sent" if ok else f"refused — {str(result.get('message') or '')[:120]}")
                + f" ({order.reason})"
            )
            self._log(outcome)
            shown.append(f"AUTO {outcome}")
        self.save()
        return shown
