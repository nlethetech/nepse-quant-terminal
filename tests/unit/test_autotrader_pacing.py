"""Auto-trade keeps to the website's order allowance and finishes long lists.

The website takes at most 10 paper orders a minute per account. A research
strategy like AK1 holds about 40 names, so on its buying day auto-trade sends
at most 8 a minute and buys the rest on the next passes; a busy or unreachable
website (429, 5xx, no connection) is not a refusal, so those orders are tried
again instead of being given up for the day.
"""

from __future__ import annotations

import io
import urllib.error
from datetime import date

import pytest

from apps.cli import connect
from apps.tui.io import strategy_autotrader as at
from apps.tui.io import website_strategies as ws

TODAY = date(2026, 11, 18)  # a Wednesday
AK1 = {"id": "hs:ak1", "kind": "research", "name": "AK1",
       "params": {"max_positions": 40, "rebalance_days": 60, "holding_days": 0, "stop_loss_pct": 0,
                  "trailing_stop_pct": 0, "sector_limit": None},
       "params_source": "research"}
PARAMS, _ = ws.trading_params(AK1)
GATE = {"filter": "K03", "invested": True, "since": "2026-11-10", "selection_date": "2026-11-17"}
NAMES = [f"S{i:02d}" for i in range(1, 41)]
PICKS = [{"rank": i + 1, "symbol": s, "ltp": 100.0, "sector": "Hydro Power"} for i, s in enumerate(NAMES)]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Book:
    """A paper book whose holdings change as orders fill."""

    can_trade = True
    market_open = True

    def __init__(self, cash=1_000_000.0, fail=None):
        self.cash_ = cash
        self.held = {}
        self.sent = []
        self.fail = fail or (lambda n, action, symbol: None)

    def positions(self):
        return [{"symbol": s, "quantity": q, "avg_cost": 100.0, "last": 100.0} for s, q in self.held.items()]

    def cash(self):
        return self.cash_

    def value(self):
        return self.cash_ + 100.0 * sum(self.held.values())

    def order_records(self):
        return [], []

    def submit(self, action, symbol, qty, *, limit_price=None):
        failure = self.fail(len(self.sent), action, symbol)
        self.sent.append((action, symbol, qty))
        if failure:
            return failure
        if action == "BUY":
            self.held[symbol] = self.held.get(symbol, 0) + qty
            self.cash_ -= qty * 100.0
        else:
            self.held.pop(symbol, None)
            self.cash_ += qty * 100.0
        return {"ok": True, "status": "FILLED", "fill_price": 100.0}


class Strategies:
    assigned = "hs:ak1"

    def get(self, sid):
        return {**AK1, "gate": GATE} if sid == "hs:ak1" else None

    def picks(self, sid):
        return ws.Picks(strategy_id=sid, as_of="2026-11-17", rows=list(PICKS), gate=GATE)


def _trader(tmp_path, clock):
    trader = at.AutoTrader("nqk_one", path=tmp_path / "auto.json", clock=clock)
    trader.set_enabled(True, "hs:ak1")
    return trader


# ── plan(): the allowance cuts the list, and the rest follows ──────────────
def test_a_cut_list_is_not_marked_done_and_the_next_plan_continues():
    state = at.fresh_state("hs:ak1")
    held = []
    for n in range(5):
        positions = [{"symbol": s, "quantity": 250, "avg_cost": 100.0, "last": 100.0} for s in held]
        orders, state, notes = at.plan(
            today=TODAY, params=PARAMS, picks=PICKS, picks_as_of="2026-11-17", positions=positions,
            cash=1_000_000.0 - 24_700.0 * len(held), nav=1_000_000.0, open_orders=[], state=state,
            gate=GATE, max_now=8,
        )
        assert [o.action for o in orders] == ["BUY"] * 8
        # 1/40 of the book each, every pass (the fee reserve may trim the very last one)
        assert all(o.qty == 247 for o in orders[:-1]) and 240 <= orders[-1].qty <= 247
        for o in orders:
            state["managed"][o.symbol] = {"bought": TODAY.isoformat(), "hwm": 100.0, "sector": "Hydro Power"}
            state["done_today"].append({"action": "BUY", "symbol": o.symbol, "ok": True})
        held += [o.symbol for o in orders]
        if n < 4:
            assert state["last_selection"] is None and any("rest of the buys" in x for x in notes)
    assert sorted(held) == NAMES
    assert state["last_selection"] == "2026-11-17"  # all 40 placed: the selection is done


def test_no_allowance_means_no_orders_and_nothing_marked():
    orders, state, _ = at.plan(today=TODAY, params=PARAMS, picks=PICKS, picks_as_of="2026-11-17", positions=[],
                               cash=1e6, nav=1e6, open_orders=[], state=at.fresh_state("hs:ak1"), gate=GATE,
                               max_now=0)
    assert orders == [] and state["last_rebalance"] is None and state["last_selection"] is None


# ── AutoTrader: 8 a minute, across passes and windows ──────────────────────
def test_a_forty_name_selection_is_bought_eight_a_minute(tmp_path):
    clock, book = Clock(), Book()
    trader = _trader(tmp_path, clock)
    trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert len(book.sent) == 8
    # 30 s later (the next refresh): the minute is used up, nothing is sent
    clock.t += 30
    trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert len(book.sent) == 8
    for _ in range(4):
        clock.t += 61
        trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert sorted(book.held) == NAMES and len(book.sent) == 40
    assert trader.state["last_selection"] == "2026-11-17"
    clock.t += 61  # done: later passes send nothing more
    trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert len(book.sent) == 40


def test_the_allowance_is_shared_by_terminal_windows(tmp_path):
    clock, book = Clock(), Book()
    first = _trader(tmp_path, clock)
    first.run_once(book=book, strategies=Strategies(), today=TODAY)
    second = at.AutoTrader("nqk_one", path=tmp_path / "auto.json", clock=clock)  # another window
    clock.t += 20
    second.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert len(book.sent) == 8


def test_a_busy_website_is_retried_not_given_up(tmp_path):
    clock = Clock()
    busy = {"ok": False, "status": "REFUSED", "transient": True,
            "message": "Too many orders at once: at most 10 a minute per account"}
    book = Book(fail=lambda n, action, symbol: busy if n == 2 else None)
    trader = _trader(tmp_path, clock)
    lines = trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert len(book.held) == 2 and len(book.sent) == 3  # stopped at the busy reply
    assert [d["symbol"] for d in trader.state["done_today"]] == ["S01", "S02"]  # S03 not given up
    assert trader.state["last_selection"] is None
    assert any("busy or unreachable" in line for line in lines)
    for _ in range(6):
        clock.t += 61
        trader.run_once(book=book, strategies=Strategies(), today=TODAY)
    assert sorted(book.held) == NAMES


def test_a_real_refusal_is_not_retried_the_same_day(tmp_path):
    clock = Clock()
    refused = {"ok": False, "status": "REFUSED", "transient": False, "message": "Not enough cash"}
    book = Book(fail=lambda n, action, symbol: refused if symbol == "S03" else None)
    trader = _trader(tmp_path, clock)
    for _ in range(6):
        trader.run_once(book=book, strategies=Strategies(), today=TODAY)
        clock.t += 61
    assert "S03" not in book.held and len(book.held) == 39
    assert [s for (_a, s, _q) in book.sent].count("S03") == 1


# ── which website errors are "busy", not a refusal ─────────────────────────
@pytest.mark.parametrize("code, transient", [(429, True), (500, True), (503, True), (403, False), (409, False),
                                             (422, False), (400, False)])
def test_busy_errors_are_marked_transient(monkeypatch, code, transient):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b'{"detail": "no"}'))

    monkeypatch.setattr(connect, "open_keyed", boom)
    with pytest.raises(connect.ConnectError) as err:
        connect.request_json("/api/cli/paper/orders", "nqk_" + "A" * 40, method="POST", body={})
    assert err.value.transient is transient


def test_no_connection_is_transient(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(connect, "open_keyed", boom)
    with pytest.raises(connect.ConnectError) as err:
        connect.request_json("/api/cli/paper/account", "nqk_" + "A" * 40)
    assert err.value.transient is True
