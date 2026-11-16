"""The paper book on nepsequant.com, used by the terminal as its own.

When the terminal is linked to a nepsequant.com account (``./nqt-connect
login``), there is ONE paper book: the website's. The terminal shows it and
sends its orders there, so the website, the phone and the terminal always show
the same cash, holdings and orders. The website runs the simulator (fills,
fees, tax, T+2, price band, NEPSE hours), exactly as for its own order ticket.

Reading needs any key. Placing and cancelling orders needs a key made with
"Paper trading" ticked; with a read-only key the terminal shows the book and
says how to get a trading key. Virtual money only: nothing here reaches a broker.

Network calls happen in ``refresh()`` (run from the terminal's background
refresh) and in ``submit()``/``cancel()``. Everything the screens read comes
from the last snapshot, so a slow or offline website never freezes the screen.

``NQT_SHARED_BOOK=0`` keeps the terminal on its own local paper book even when
a key is saved.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from apps.cli import connect

TRADE_LOG_COLS = ["Date", "Action", "Symbol", "Shares", "Price", "Fees", "Reason", "PnL", "PnL_Pct"]
PORT_COLS = ["Entry_Date", "Symbol", "Quantity", "Buy_Price", "Total_Cost_Basis", "Last_LTP", "Signal_Type"]
NAV_COLS = ["Date", "Cash", "Positions_Value", "NAV", "Num_Positions"]
TICK = 0.1  # NEPSE prices move in Rs 0.10 steps


def shared_book_enabled() -> bool:
    return str(os.environ.get("NQT_SHARED_BOOK", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) or math.isinf(out) else out


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _s(value: Any) -> str:
    return "" if value is None else str(value)


def to_tick(price: float) -> float:
    return round(round(float(price) / TICK) * TICK, 1)


@dataclass
class Snapshot:
    account: dict[str, Any] = field(default_factory=dict)
    open_orders: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    nav: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0


Request = Callable[..., dict]


class WebsiteBook:
    """The website's paper book, cached for the terminal's screens."""

    def __init__(self, key: str, *, request: Optional[Request] = None) -> None:
        self._key = key
        self._request: Request = request or connect.request_json
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self.error = ""  # last refresh problem, shown on screen; "" when fine

    # ── linking ─────────────────────────────────────────────────────────────
    @classmethod
    def from_saved(cls) -> Optional["WebsiteBook"]:
        """The linked book, or None when no key is saved (or it is switched off)."""
        if not shared_book_enabled():
            return None
        try:
            key = connect.load_key()
        except connect.ConnectError:
            return None
        return cls(key) if key else None

    @property
    def site(self) -> str:
        try:
            return connect.base_url().replace("https://", "")
        except connect.ConnectError:
            return "nepsequant.com"

    # ── reading ─────────────────────────────────────────────────────────────
    def refresh(self) -> bool:
        """Fetch the book. On failure the last snapshot stays and ``error`` says why."""
        try:
            account = self._request("/api/cli/paper/account", self._key)
            orders = self._request("/api/cli/paper/orders", self._key)
            trades = self._request("/api/cli/paper/trades?limit=500", self._key)
            try:
                nav = self._request("/api/cli/paper/nav", self._key).get("nav") or []
            except connect.ConnectError:
                nav = []  # value history is a nice-to-have: never fail the book on it
        except connect.ConnectError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:  # malformed reply: keep the last good book
            self.error = f"Could not read your paper book from {self.site}: {exc}"
            return False
        with self._lock:
            self._snap = Snapshot(
                account=dict(account or {}),
                open_orders=list(orders.get("open") or []),
                history=list(orders.get("history") or []),
                trades=list(trades.get("trades") or []),
                nav=list(nav),
                fetched_at=time.time(),
            )
        self.error = ""
        return True

    @property
    def loaded(self) -> bool:
        return self._snap.fetched_at > 0

    @property
    def can_trade(self) -> bool:
        return bool(self._snap.account.get("can_trade"))

    @property
    def market_open(self) -> Optional[bool]:
        market = self._snap.account.get("market") or {}
        return None if not market else bool(market.get("open"))

    def cash(self) -> float:
        return _f(self._snap.account.get("cash"))

    def value(self) -> float:
        """The book's total value (cash + holdings) as the website reports it."""
        return _f(self._snap.account.get("value"))

    def starting_cash(self) -> float:
        return _f(self._snap.account.get("starting_cash")) or 1_000_000.0

    def positions(self) -> list[dict[str, Any]]:
        return list(self._snap.account.get("positions") or [])

    # The terminal's own record shapes, so its screens and checks work unchanged.
    def portfolio_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.positions():
            qty = _i(p.get("quantity"))
            if qty <= 0:
                continue
            avg = _f(p.get("avg_cost"))
            cost = _f(p.get("cost_basis")) or avg * qty
            rows.append({
                "Entry_Date": _s(p.get("acquired_date"))[:10],
                "Symbol": _s(p.get("symbol")).upper(),
                "Quantity": qty,
                "Buy_Price": avg,
                "Total_Cost_Basis": cost,
                "Last_LTP": _f(p.get("last")),
                "Signal_Type": "",
            })
        return pd.DataFrame(rows, columns=PORT_COLS)

    def trade_log_frame(self) -> pd.DataFrame:
        rows = []
        for t in reversed(self._snap.trades):  # the website sends newest first
            pnl_pct = _f(t.get("pnl_pct"))
            rows.append({
                "Date": _s(t.get("date")),
                "Action": _s(t.get("action")).upper(),
                "Symbol": _s(t.get("symbol")).upper(),
                "Shares": _i(t.get("shares")),
                "Price": _f(t.get("price")),
                "Fees": _f(t.get("fees")),
                "Reason": _s(t.get("reason")),
                "PnL": _f(t.get("pnl")),
                # the terminal keeps PnL_Pct as a fraction; the website sends percent
                "PnL_Pct": pnl_pct / 100.0,
            })
        return pd.DataFrame(rows, columns=TRADE_LOG_COLS)

    def nav_frame(self) -> pd.DataFrame:
        rows = [
            {"Date": _s(r.get("date"))[:10], "Cash": float("nan"), "Positions_Value": float("nan"),
             "NAV": _f(r.get("nav")), "Num_Positions": 0}
            for r in self._snap.nav
            if _f(r.get("nav")) > 0
        ]
        return pd.DataFrame(rows, columns=NAV_COLS)

    @staticmethod
    def _record(o: dict[str, Any]) -> dict[str, Any]:
        created = _s(o.get("created_at"))
        price = _f(o.get("limit_price")) or _f(o.get("fill_price"))
        qty = _i(o.get("quantity"))
        return {
            "id": _s(o.get("order_id")),
            "order_id": _s(o.get("order_id")),
            "action": _s(o.get("action")).upper(),
            "symbol": _s(o.get("symbol")).upper(),
            "qty": qty,
            "quantity": qty,
            "price": price,
            "trigger_price": price,
            "order_type": _s(o.get("order_type") or "market").lower(),
            "slippage_pct": 0.0,
            "status": _s(o.get("status")).upper() or "OPEN",
            "filled_qty": _i(o.get("filled_qty")) or (qty if _s(o.get("status")).upper() == "FILLED" else 0),
            "fill_price": _f(o.get("fill_price")),
            "created_at": created.replace("T", " ")[:19],
            "updated_at": _s(o.get("updated_at")).replace("T", " ")[:19],
            "day": created[:10],
            "source": "nepsequant.com",
            "reason": _s(o.get("reason")),
        }

    def order_records(self) -> tuple[list[dict], list[dict]]:
        """(open orders, order history oldest first) in the terminal's shape."""
        with self._lock:
            open_rows = [self._record(o) for o in self._snap.open_orders]
            history = [self._record(o) for o in reversed(self._snap.history)]  # sent newest first
        return open_rows, history

    # ── trading ─────────────────────────────────────────────────────────────
    def submit(self, action: str, symbol: str, qty: int, *, limit_price: Optional[float] = None) -> dict[str, Any]:
        """Place a paper order on the website. A price makes it a limit order."""
        body: dict[str, Any] = {"action": action.upper(), "symbol": symbol.upper(), "quantity": int(qty)}
        if limit_price:
            body.update(order_type="limit", limit_price=to_tick(limit_price))
        else:
            body["order_type"] = "market"
        try:
            result = self._request("/api/cli/paper/orders", self._key, method="POST", body=body)
        except connect.ConnectError as exc:
            # transient: the website was busy or unreachable, not a refusal of this order
            return {"ok": False, "status": "REFUSED", "message": str(exc),
                    "transient": bool(getattr(exc, "transient", False))}
        except Exception as exc:
            return {"ok": False, "status": "ERROR", "message": f"Could not reach {self.site}: {exc}", "transient": True}
        self.refresh()
        return dict(result or {})

    def cancel(self, order_id: str) -> dict[str, Any]:
        try:
            result = self._request(f"/api/cli/paper/orders/{order_id}", self._key, method="DELETE")
        except connect.ConnectError as exc:
            return {"ok": False, "message": str(exc)}
        except Exception as exc:
            return {"ok": False, "message": f"Could not reach {self.site}: {exc}"}
        self.refresh()
        return dict(result or {"ok": True})

    def cancel_all(self) -> tuple[int, list[str]]:
        """Cancel every open order; (number cancelled, problems)."""
        cancelled, problems = 0, []
        open_ids = [_s(o.get("order_id")) for o in self._snap.open_orders if _s(o.get("order_id"))]
        for oid in open_ids:
            try:
                self._request(f"/api/cli/paper/orders/{oid}", self._key, method="DELETE")
                cancelled += 1
            except Exception as exc:
                problems.append(f"{oid[:8]}: {exc}")
        if open_ids:
            self.refresh()
        return cancelled, problems


def describe_result(result: dict[str, Any], action: str, symbol: str, qty: int, limit_price: Optional[float]) -> str:
    """One status line for an order the website answered."""
    what = f"{action} {symbol} x{qty}" + (f" limit {to_tick(limit_price):,.1f}" if limit_price else " at market")
    message = _s(result.get("message")).strip()
    if not result.get("ok"):
        return f"Rejected: {what} — {message or 'the website refused the order'}"
    status = _s(result.get("status")).upper()
    if status == "FILLED" and _f(result.get("fill_price")):
        return f"Filled: {what} @ {_f(result.get('fill_price')):,.1f} on your nepsequant.com paper book"
    if result.get("resting") or status == "OPEN":
        return f"Order {_s(result.get('order_id'))[:8]}: {what} — OPEN on your nepsequant.com paper book"
    return f"{what} — {message or status or 'sent'}"
