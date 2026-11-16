"""Auto-trading a gated research strategy (the website's AK1 + K03 filter).

* 0 means OFF for stop-loss / trailing stop / holding days; a null sector cap
  means no cap (a 0% stop used to sell anything at or below its cost).
* The gate: while it says cash, every position auto-trade bought is sold
  ("K03 says cash") and nothing is bought; same-day and T+2 guards hold.
* The selection is a holding list: a new selection date is the rebalance day —
  managed names no longer selected are sold (and retried until they go), new
  names are bought at equal weight of the selection, kept names are left alone.
"""

from __future__ import annotations

from datetime import date

from apps.tui.io import strategy_autotrader as at
from apps.tui.io import website_strategies as ws

TODAY = date(2026, 11, 18)  # a Wednesday
AK1 = {"id": "hs:ak1", "kind": "research", "name": "AK1 · illiquid stocks + K03 filter",
       "params": {"max_positions": 40, "rebalance_days": 60, "holding_days": 0, "stop_loss_pct": 0,
                  "trailing_stop_pct": 0, "sector_limit": None},
       "params_source": "research"}
PARAMS, _ = ws.trading_params(AK1)
CASH_GATE = {"filter": "K03", "invested": False, "position": "CASH", "since": "2026-04-07", "as_of": "2026-11-17",
             "selection_date": None,
             "gauges": {"rates": {"on": False, "value": 2.75, "detail": {"rate": 2.75, "mean_12m": 2.75}},
                        "funds": {"on": False, "value": 0.0155, "detail": {}},
                        "trend": {"on": True, "value": 0.0046, "detail": {}}}, "votes": 1}


def invested(selection_date="2026-11-17"):
    return {**CASH_GATE, "invested": True, "position": "INVESTED", "since": "2026-11-10", "selection_date": selection_date}


def _pick(rank, sym, ltp=100.0, sector="Hydro Power"):
    return {"rank": rank, "symbol": sym, "ltp": ltp, "sector": sector}


def _pos(sym, qty=100, cost=100.0, last=100.0):
    return {"symbol": sym, "quantity": qty, "avg_cost": cost, "last": last}


def _managed(*syms, bought="2026-11-02"):
    return {s: {"bought": bought, "hwm": 100.0, "sector": "Hydro Power", "strategy": "hs:ak1"} for s in syms}


def _plan(**kw):
    args = dict(today=TODAY, params=PARAMS, picks=[], picks_as_of="2026-11-17", positions=[],
                cash=1_000_000.0, nav=1_000_000.0, open_orders=[], state=at.fresh_state("hs:ak1"), gate=None)
    args.update(kw)
    return at.plan(**args)


# ── trading params: 0 = off, null sector cap = none ─────────────────────────
def test_research_params_turn_the_stops_and_the_holding_limit_off():
    assert PARAMS == {"max_positions": 40, "rebalance_days": 60, "holding_days": 0, "stop_loss_pct": 0.0,
                      "trailing_stop_pct": 0.0, "sector_limit": None}
    assert ws.trading_params(AK1)[1] == "research"
    # keys not sent keep the defaults; a sent cap is a fraction
    params, _ = ws.trading_params({"params": {"max_positions": 6, "sector_limit": 0.25}, "params_source": "strategy"})
    assert params["holding_days"] == 40 and params["stop_loss_pct"] == 8.0 and params["sector_limit"] == 25.0


def test_zero_stops_never_sell_and_zero_holding_never_expires():
    state = at.fresh_state("rs:x")
    state["last_rebalance"] = "2026-11-17"
    state["managed"] = _managed("DOWN", "OLD", bought="2025-01-01")
    state["managed"]["DOWN"]["hwm"] = 200.0
    positions = [_pos("DOWN", cost=100.0, last=60.0), _pos("OLD", cost=100.0, last=100.0)]
    orders, _, _ = _plan(positions=positions, state=state)  # no gate: an ordinary strategy with these rules
    assert orders == []


def test_no_sector_cap_when_the_strategy_sends_none():
    picks = [_pick(i, f"H{i}", sector="Hydro Power") for i in range(1, 6)]
    orders, _, notes = _plan(params={**PARAMS, "max_positions": 5}, picks=picks)
    assert [o.symbol for o in orders] == ["H1", "H2", "H3", "H4", "H5"] and not any("sector cap" in n for n in notes)


# ── the gate ─────────────────────────────────────────────────────────────────
def test_cash_sells_everything_auto_trade_bought_and_buys_nothing():
    state = at.fresh_state("hs:ak1")
    state.update(last_rebalance="2026-09-01", last_selection="2026-08-31")
    state["managed"] = _managed("A", "B")
    state["managed"]["TODAY"] = {"bought": "2026-11-18", "hwm": 100.0}   # same day: cannot sell yet
    state["managed"]["YDAY"] = {"bought": "2026-11-17", "hwm": 100.0}    # not settled (T+2) yet
    positions = [_pos("A"), _pos("B", qty=40), _pos("TODAY"), _pos("YDAY"), _pos("MINE")]  # MINE: bought by hand
    orders, st, notes = _plan(positions=positions, state=state, gate=CASH_GATE,
                              picks=[_pick(1, "NEW1"), _pick(2, "NEW2")])
    assert [(o.action, o.symbol, o.qty) for o in orders] == [("SELL", "A", 100), ("SELL", "B", 40)]
    assert all(o.reason == "K03 says cash since 2026-04-07" for o in orders)
    assert st["last_rebalance"] is None and st["last_selection"] is None  # re-entry = a fresh selection
    assert any("K03 says cash" in n for n in notes)
    # the next settled day the rest go too
    st["managed"] = {k: v for k, v in st["managed"].items() if k in ("TODAY", "YDAY")}
    later, _, _ = _plan(today=date(2026, 11, 20), positions=[_pos("TODAY"), _pos("YDAY"), _pos("MINE")],
                        state=st, gate=CASH_GATE, picks_as_of="2026-11-19")
    assert sorted(o.symbol for o in later) == ["TODAY", "YDAY"]


def test_cash_with_nothing_held_does_nothing():
    orders, st, notes = _plan(gate=CASH_GATE, picks=[])
    assert orders == [] and notes == [] and st["last_selection"] is None


def test_turning_invested_buys_the_whole_selection_at_equal_weight_of_the_selection():
    picks = [_pick(i, f"S{i:02d}", ltp=50.0) for i in range(1, 26)]  # 25 selected (fewer than max 40)
    orders, st, _ = _plan(picks=picks, gate=invested())
    assert [o.symbol for o in orders] == [f"S{i:02d}" for i in range(1, 26)]
    assert all(o.action == "BUY" for o in orders)
    assert orders[0].qty == int(1_000_000 / 25 // (50.0 * 1.01))  # 1/n of the book, not 1/40
    assert "selection of 2026-11-17" in orders[0].reason
    assert st["last_selection"] == "2026-11-17" and st["last_rebalance"] == "2026-11-18"
    # the same selection on a later pass: nothing (no trades between selections)
    st["managed"] = _managed(*[o.symbol for o in orders], bought="2026-11-18")
    again, _, _ = _plan(today=date(2026, 11, 25), picks=picks, gate=invested(), state=st,
                        positions=[_pos(o.symbol, qty=o.qty, last=50.0) for o in orders], picks_as_of="2026-11-24")
    assert again == []


def test_a_new_selection_rotates_sells_first_and_keeps_names_still_selected():
    state = at.fresh_state("hs:ak1")
    state.update(last_rebalance="2026-08-20", last_selection="2026-08-19")
    state["managed"] = _managed("A", "B", "C")
    positions = [_pos("A", qty=1000, last=300.0), _pos("B"), _pos("C"), _pos("MINE")]
    picks = [_pick(1, "B"), _pick(2, "C"), _pick(3, "D", ltp=100.0), _pick(4, "E", ltp=100.0)]
    orders, st, _ = _plan(positions=positions, state=state, gate=invested("2026-11-17"), picks=picks,
                          cash=0.0, nav=320_000.0)
    assert [(o.action, o.symbol) for o in orders] == [("SELL", "A"), ("BUY", "D"), ("BUY", "E")]
    assert orders[0].reason == "no longer in the selection"
    # paid for by A's sale (300,000 at the last price); equal weight = book / 4 selected
    assert orders[1].qty == int(80_000 // 101.0) and orders[2].qty == int(80_000 // 101.0)
    assert st["last_selection"] == "2026-11-17"


def test_a_sell_that_did_not_go_through_is_retried_between_selections():
    state = at.fresh_state("hs:ak1")
    state.update(last_rebalance="2026-11-17", last_selection="2026-11-17")
    state["managed"] = _managed("B", "LEFT")
    picks = [_pick(1, "B")]
    orders, _, _ = _plan(positions=[_pos("B"), _pos("LEFT")], state=state, gate=invested("2026-11-17"), picks=picks)
    assert [(o.action, o.symbol) for o in orders] == [("SELL", "LEFT")]


def test_stale_picks_never_rotate_or_buy():
    state = at.fresh_state("hs:ak1")
    state["managed"] = _managed("LEFT")
    orders, st, notes = _plan(positions=[_pos("LEFT")], state=state, gate=invested("2026-11-17"),
                              picks=[_pick(1, "B")], picks_as_of="2026-11-06")
    assert orders == [] and "stale" in notes[0] and st["last_selection"] is None


def test_a_gate_without_picks_waits():
    orders, _, notes = _plan(gate=invested("2026-11-17"), picks=[])
    assert orders == [] and "No picks" in notes[0]


# ── the auto-trader pass + the client ───────────────────────────────────────
class Book:
    can_trade = True
    market_open = True

    def __init__(self, positions=None, cash=1_000_000.0):
        self._positions = list(positions or [])
        self._cash = cash
        self.sent = []

    def positions(self):
        return list(self._positions)

    def cash(self):
        return self._cash

    def value(self):
        return self._cash + sum(p["quantity"] * p["last"] for p in self._positions)

    def order_records(self):
        return [], []

    def submit(self, action, symbol, qty, *, limit_price=None):
        self.sent.append((action, symbol, qty))
        return {"ok": True, "status": "FILLED", "fill_price": 100.0}


class Strategies:
    assigned = "hs:ak1"

    def __init__(self, gate, rows, error=""):
        self.gate, self.rows, self.error = gate, rows, error

    def get(self, sid):
        return {**AK1, "gate": None} if sid == "hs:ak1" else None

    def picks(self, sid):
        return ws.Picks(strategy_id=sid, as_of="2026-11-17", rows=list(self.rows), gate=self.gate, error=self.error)


def test_a_pass_reads_the_gate_from_the_picks(tmp_path):
    trader = at.AutoTrader("nqk_one", path=tmp_path / "auto.json")
    trader.set_enabled(True, "hs:ak1")
    book = Book()
    trader.run_once(book=book, strategies=Strategies(invested(), [_pick(1, "AAA"), _pick(2, "BBB")]), today=TODAY)
    assert [s[:2] for s in book.sent] == [("BUY", "AAA"), ("BUY", "BBB")]
    assert trader.state["last_selection"] == "2026-11-17"
    # K03 turns cash: the next settled day everything auto-trade bought is sold
    book2 = Book(positions=[_pos("AAA", qty=4950), _pos("BBB", qty=4950)], cash=0.0)
    trader.run_once(book=book2, strategies=Strategies(CASH_GATE, []), today=date(2026, 11, 20))
    assert sorted(s[:2] for s in book2.sent) == [("SELL", "AAA"), ("SELL", "BBB")]


def test_a_failed_read_of_a_gated_strategy_trades_nothing(tmp_path):
    trader = at.AutoTrader("nqk_one", path=tmp_path / "auto.json")
    trader.set_enabled(True, "hs:ak1")
    trader.state["managed"] = _managed("AAA")
    trader.save()
    book = Book(positions=[_pos("AAA")])
    assert trader.run_once(book=book, strategies=Strategies(CASH_GATE, [], error="timed out"), today=TODAY) == []
    assert book.sent == [] and any("Could not read" in line for line in trader.state["log"])


def test_client_keeps_the_gate_and_the_screens_say_it():
    payload = {"as_of": "2026-11-06", "warming": False, "picks": [], "gate": CASH_GATE, "selection_date": None,
               "note": "K03 says cash since 2026-04-07; AK1 holds nothing."}

    def fake(path, key, *, method="GET", body=None, timeout=30):
        if path == "/api/cli/strategies":
            return {"strategies": [{**AK1, "gate": CASH_GATE, "description": "Research verdict WATCH.",
                                    "backtest": {"verdict": "WATCH", "cagr_pct": 8.7, "alpha_pct": 11.0,
                                                 "alpha_t": 1.63, "bar_t": 2.24, "start": "2022-01-02",
                                                 "end": "2024-06-30", "design": {"alpha_pct": 15.8, "alpha_t": 2.06}}}],
                    "assigned": "hs:ak1"}
        return payload

    client = ws.WebsiteStrategies("nqk_x", request=fake)
    assert client.refresh() and client.assigned == "hs:ak1"
    picks = client.picks("hs:ak1")
    assert picks.gate["invested"] is False and picks.rows == []
    assert ws.gate_title(ws.gate_of(client.get("hs:ak1"), picks)) == "K03: CASH since 2026-04-07"
    lines = ws.describe(client.get("hs:ak1"))
    assert lines[0] == "AK1 · illiquid stocks + K03 filter  (RESEARCH)"
    assert any(line.startswith("K03: CASH since 2026-04-07 (close of 2026-11-17) · 1 of 3") for line in lines)
    assert any("interest rates off 2.75% vs 12-month mean 2.75%" in line and "trend ON" in line for line in lines)
    assert any(line.startswith("Research verdict WATCH: alpha +11.0%/yr, t 1.63 (bar 2.24)") for line in lines)
    rules = next(line for line in lines if line.startswith("Trading rules"))
    assert "(from the research)" in rules and "no holding limit" in rules and "no stop-loss" in rules
    assert "no trailing stop" in rules and "no sector cap" in rules and "market filter" in rules
