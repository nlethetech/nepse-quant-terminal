"""The terminal's auto-trader for an assigned nepsequant.com strategy.

Pins the trading rules (same as the strategy's backtest: equal weight, max
positions, rebalance days, stop-loss, trailing stop, holding period, sector
cap) and the guards (only auto-bought stocks are sold, nothing twice a day,
stale picks are not bought, only while NEPSE is open, daily order cap).
"""

from __future__ import annotations

from datetime import date

import pytest

from apps.tui.io import strategy_autotrader as at
from apps.tui.io import website_strategies as ws

PARAMS = {"max_positions": 4, "rebalance_days": 5, "holding_days": 40,
          "stop_loss_pct": 8.0, "trailing_stop_pct": 10.0, "sector_limit": 50.0}
TODAY = date(2026, 11, 18)  # a Wednesday


def _pick(rank, sym, ltp=100.0, sector="Banks"):
    return {"rank": rank, "symbol": sym, "ltp": ltp, "sector": sector}


def _pos(sym, qty=100, cost=100.0, last=100.0):
    return {"symbol": sym, "quantity": qty, "avg_cost": cost, "last": last}


def _plan(**kw):
    args = dict(today=TODAY, params=PARAMS, picks=[], picks_as_of="2026-11-18", positions=[],
                cash=1_000_000.0, nav=1_000_000.0, open_orders=[], state=at.fresh_state("rs:a"))
    args.update(kw)
    return at.plan(**args)


def test_weekdays_between_counts_trading_days():
    assert at.weekdays_between(date(2026, 11, 13), date(2026, 11, 16)) == 1  # Fri → Mon
    assert at.weekdays_between(date(2026, 11, 16), date(2026, 11, 23)) == 5
    assert at.weekdays_between(date(2026, 11, 18), date(2026, 11, 18)) == 0


def test_first_rebalance_buys_top_picks_equal_weight():
    picks = [_pick(1, "AAA", 100.0), _pick(2, "BBB", 200.0, "Hydro"), _pick(3, "CCC", 50.0, "Hydro"),
             _pick(4, "DDD", 400.0, "Life"), _pick(5, "EEE", 10.0)]
    orders, st, _ = _plan(picks=picks)
    assert [o.symbol for o in orders] == ["AAA", "BBB", "CCC", "DDD"]  # max 4 positions
    assert all(o.action == "BUY" for o in orders)
    # equal weight: 1,000,000 / 4 = 250,000 each (1% room for the price moving)
    assert orders[0].qty == int(250_000 // 101.0)
    assert orders[1].qty == int(250_000 // 202.0)
    assert st["last_rebalance"] == "2026-11-18"


def test_no_buying_between_rebalance_days():
    state = {**at.fresh_state("rs:a"), "last_rebalance": "2026-11-16"}
    orders, st, _ = _plan(picks=[_pick(1, "AAA")], state=state)
    assert orders == [] and st["last_rebalance"] == "2026-11-16"
    state = {**at.fresh_state("rs:a"), "last_rebalance": "2026-11-11"}  # 5 trading days ago
    orders, _, _ = _plan(picks=[_pick(1, "AAA")], state=state)
    assert [o.symbol for o in orders] == ["AAA"]


def test_stale_picks_are_not_bought():
    orders, st, notes = _plan(picks=[_pick(1, "AAA")], picks_as_of="2026-11-06")
    assert orders == [] and "stale" in notes[0] and st["last_rebalance"] is None


def test_hand_bought_stocks_are_never_sold_or_rebought():
    positions = [_pos("MINE", cost=100.0, last=50.0)]  # down 50%, but bought by hand
    orders, _, _ = _plan(positions=positions, picks=[_pick(1, "MINE"), _pick(2, "AAA")])
    assert [(o.action, o.symbol) for o in orders] == [("BUY", "AAA")]


def test_stop_loss_trailing_and_holding_exits_for_managed_positions():
    state = at.fresh_state("rs:a")
    state["last_rebalance"] = "2026-11-17"
    state["managed"] = {
        "STOP": {"bought": "2026-11-10", "hwm": 100.0},
        "TRAIL": {"bought": "2026-11-10", "hwm": 150.0},
        "OLD": {"bought": "2026-09-01", "hwm": 120.0},
        "FINE": {"bought": "2026-11-10", "hwm": 105.0},
    }
    positions = [_pos("STOP", cost=100.0, last=91.0), _pos("TRAIL", cost=100.0, last=134.0),
                 _pos("OLD", cost=100.0, last=119.0), _pos("FINE", cost=100.0, last=104.0)]
    orders, st, _ = _plan(positions=positions, state=state)
    reasons = {o.symbol: o.reason for o in orders}
    assert set(reasons) == {"STOP", "TRAIL", "OLD"}
    assert "stop-loss" in reasons["STOP"] and "trailing" in reasons["TRAIL"] and "held" in reasons["OLD"]
    assert all(o.action == "SELL" and o.qty == 100 for o in orders)
    assert st["managed"]["FINE"]["hwm"] == 105.0


def test_nothing_sold_on_the_day_it_was_bought():
    state = at.fresh_state("rs:a")
    state["last_rebalance"] = "2026-11-18"
    state["managed"] = {"NEW": {"bought": "2026-11-18", "hwm": 100.0}}
    orders, _, _ = _plan(positions=[_pos("NEW", last=80.0)], state=state)
    assert orders == []


def test_sold_positions_stop_being_managed():
    state = at.fresh_state("rs:a")
    state["managed"] = {"GONE": {"bought": "2026-11-10", "hwm": 100.0}}
    _, st, _ = _plan(state=state)
    assert st["managed"] == {}


def test_nothing_twice_in_a_day_and_open_orders_respected():
    state = at.fresh_state("rs:a")
    state["day"] = "2026-11-18"
    state["done_today"] = [{"action": "BUY", "symbol": "AAA", "ok": False}]
    orders, _, _ = _plan(picks=[_pick(1, "AAA"), _pick(2, "BBB"), _pick(3, "CCC")], state=state,
                         open_orders=[{"action": "BUY", "symbol": "BBB"}])
    assert [o.symbol for o in orders] == ["CCC"]


def test_slots_count_kept_positions_and_pending_buys():
    state = at.fresh_state("rs:a")
    state["managed"] = {"H1": {"bought": "2026-11-10", "hwm": 100.0}, "H2": {"bought": "2026-11-10", "hwm": 100.0}}
    positions = [_pos("H1"), _pos("H2")]
    orders, _, _ = _plan(picks=[_pick(i, f"P{i}") for i in range(1, 6)], positions=positions, state=state,
                         open_orders=[{"action": "BUY", "symbol": "X"}])
    assert [o.symbol for o in orders] == ["P1"]  # 4 max − 2 held − 1 pending


def test_sector_cap_and_cash_limit():
    params = {**PARAMS, "sector_limit": 30.0}
    picks = [_pick(1, "B1"), _pick(2, "B2"), _pick(3, "H1", sector="Hydro")]
    orders, _, notes = _plan(params=params, picks=picks)
    assert [o.symbol for o in orders] == ["B1", "H1"]  # a second bank would make Banks 50%
    assert any("sector cap" in n for n in notes)
    orders, _, notes = _plan(picks=[_pick(1, "AAA", 100.0)], cash=50.0)
    assert orders == [] and any("not enough cash" in n for n in notes)


def test_daily_order_cap():
    state = at.fresh_state("rs:a")
    state["day"] = "2026-11-18"
    state["done_today"] = [{"action": "BUY", "symbol": f"Z{i}", "ok": True} for i in range(10)]  # cap = 2×4+2
    orders, _, _ = _plan(picks=[_pick(1, "AAA")], state=state)
    assert orders == []


def test_plan_does_not_mutate_the_callers_state():
    state = at.fresh_state("rs:a")
    _plan(picks=[_pick(1, "AAA")], state=state)
    assert state["last_rebalance"] is None and state["day"] is None


# ── trading params from the website ─────────────────────────────────────────


def test_trading_params_from_strategy_or_defaults():
    params, source = ws.trading_params({"params": {"max_positions": 6, "stop_loss_pct": 0.05, "trailing_stop_pct": 12},
                                        "params_source": "backtest"})
    assert params["max_positions"] == 6 and params["stop_loss_pct"] == 5.0 and params["trailing_stop_pct"] == 12.0
    assert params["rebalance_days"] == ws.DEFAULT_PARAMS["rebalance_days"] and source == "backtest"
    params, source = ws.trading_params({"params": None})
    assert params == ws.DEFAULT_PARAMS and source == "default"


# ── one pass against a fake book and website ────────────────────────────────


class FakeBook:
    def __init__(self, *, can_trade=True, market_open=True, positions=None, cash=1_000_000.0):
        self.can_trade = can_trade
        self.market_open = market_open
        self._positions = list(positions or [])
        self._cash = cash
        self.sent: list[tuple] = []

    def positions(self):
        return list(self._positions)

    def cash(self):
        return self._cash

    def value(self):
        return self._cash + sum(p["quantity"] * p["last"] for p in self._positions)

    def order_records(self):
        return [], []

    def submit(self, action, symbol, qty, *, limit_price=None):
        self.sent.append((action, symbol, qty, limit_price))
        if symbol == "REFUSE":
            return {"ok": False, "status": "BLOCKED", "message": "Price is outside the band."}
        return {"ok": True, "status": "FILLED", "fill_price": 101.0}


class FakeStrategies:
    def __init__(self, assigned="rs:a", rows=None, as_of="2026-11-18"):
        self.assigned = assigned
        self._rows = rows if rows is not None else [_pick(1, "AAA"), _pick(2, "REFUSE")]
        self._as_of = as_of

    def get(self, sid):
        return {"id": sid, "params": PARAMS, "params_source": "backtest"} if sid == self.assigned else None

    def picks(self, sid):
        return ws.Picks(strategy_id=sid, as_of=self._as_of, rows=list(self._rows))


def _trader(tmp_path, key="nqk_one"):
    return at.AutoTrader(key, path=tmp_path / "auto.json")


def test_off_by_default_and_needs_open_market_and_trading_key(tmp_path):
    trader = _trader(tmp_path)
    book, strategies = FakeBook(), FakeStrategies()
    assert trader.run_once(book=book, strategies=strategies, today=TODAY) == [] and book.sent == []
    trader.set_enabled(True, "rs:a")
    for closed in (FakeBook(market_open=False), FakeBook(market_open=None), FakeBook(can_trade=False)):
        assert trader.run_once(book=closed, strategies=strategies, today=TODAY) == [] and closed.sent == []


def test_a_pass_sends_market_orders_and_remembers_what_it_bought(tmp_path):
    trader = _trader(tmp_path)
    trader.set_enabled(True, "rs:a")
    book = FakeBook()
    lines = trader.run_once(book=book, strategies=FakeStrategies(), today=TODAY)
    assert [s[:2] for s in book.sent] == [("BUY", "AAA"), ("BUY", "REFUSE")]
    assert all(s[3] is None for s in book.sent)  # market orders
    assert "AAA" in trader.state["managed"] and "REFUSE" not in trader.state["managed"]
    assert any("refused" in line for line in lines)
    # the second pass the same day sends nothing again (the bought stock is now in the book)
    book2 = FakeBook(positions=[_pos("AAA", qty=2475, cost=101.0, last=101.0)])
    trader.run_once(book=book2, strategies=FakeStrategies(), today=TODAY)
    assert book2.sent == []
    # state survives a restart, for the same account only
    assert _trader(tmp_path).state["managed"].keys() == {"AAA"}
    assert _trader(tmp_path, key="nqk_other").state["managed"] == {}


def test_a_changed_assignment_switches_auto_trading_off(tmp_path):
    trader = _trader(tmp_path)
    trader.set_enabled(True, "rs:a")
    book = FakeBook()
    lines = trader.run_once(book=book, strategies=FakeStrategies(assigned="bl:other"), today=TODAY)
    assert book.sent == [] and not trader.enabled and "switched off" in lines[0]


def test_errors_never_escape(tmp_path):
    trader = _trader(tmp_path)
    trader.set_enabled(True, "rs:a")

    class Broken(FakeStrategies):
        def picks(self, sid):
            raise RuntimeError("boom")

    lines = trader.run_once(book=FakeBook(), strategies=Broken(), today=TODAY)
    assert "Auto-trade error: boom" in lines[0]


# ── the website strategies client ───────────────────────────────────────────


def test_client_lists_assigns_and_caches_picks():
    calls = []

    def fake(path, key, *, method="GET", body=None, timeout=30):
        calls.append((method, path, body))
        if path == "/api/cli/strategies":
            return {"strategies": [{"id": "rs:a", "kind": "rules", "name": "Value banks"},
                                   {"id": "bl:b", "kind": "blend", "name": "Momentum"}],
                    "assigned": "rs:gone", "notice": "Your own screen results."}
        if path == "/api/cli/strategy-assignment":
            return {"assigned": body["id"]}
        if path.startswith("/api/cli/strategies/rs%3Aa/signals"):
            return {"as_of": "2026-11-18", "warming": False, "picks": [{"rank": 1, "symbol": "NABIL"}]}
        raise AssertionError(path)

    client = ws.WebsiteStrategies("nqk_x", request=fake)
    assert client.refresh() and len(client.strategies) == 2
    assert client.assigned is None  # an assignment to a deleted strategy is ignored
    ok, msg = client.assign("rs:a")
    assert ok and client.assigned == "rs:a" and "Value banks" in msg
    picks = client.picks("rs:a")
    assert picks.rows[0]["symbol"] == "NABIL" and picks.as_of == "2026-11-18"
    client.picks("rs:a")
    assert sum(1 for c in calls if "signals" in c[1]) == 1  # cached
    lines = ws.describe(client.get("rs:a"))
    assert lines[0].startswith("Value banks") and any("Trading rules (default)" in line for line in lines)
