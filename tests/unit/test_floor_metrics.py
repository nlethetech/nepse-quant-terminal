"""Broker floor metrics: the same numbers from Rust, from Python, and from the scraper.

* the Python engine reproduces what ``floorsheet_scraper._recompute_aggregates``
  stores (every column of ``broker_signals_v2``) and what the panel's SQL picks
  (``local_floor_records``), on random days and on the edge cases;
* the optional Rust engine gives bit-identical arrays (skipped when the native
  module is not installed: the terminal never needs it).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from apps.tui.io import floor_metrics as fm
from backend.quant_pro.floorsheet_scraper import FloorTrade, _insert_trades, _recompute_aggregates, ensure_floorsheet_tables

DAY = "2026-11-13"
CODES = [c for c in range(1, 70) if c not in (2, 9, 12, 15, 23)]
native, _why = fm.native_module()
needs_rust = pytest.mark.skipif(native is None, reason="native nqt_floor module not installed")


# ── days to test on ──────────────────────────────────────────────────────────
def random_day(seed: int, n_trades: int = 2500, n_symbols: int = 25):
    """(symbols, symbol_idx, buyer codes, seller codes, qty) with planted patterns."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    sym = list(rng.integers(0, n_symbols, n_trades))
    buyer = list(rng.choice(CODES, n_trades))
    seller = list(rng.choice(CODES, n_trades))
    qty = [int(q) for q in rng.integers(1, 900, n_trades)]
    for i in range(n_trades):
        roll = rng.random()
        if roll < 0.08:
            seller[i] = buyer[i]  # self-trade
        elif roll < 0.10:
            qty[i] = 0
    plants = [
        # one broker buys from many sellers (concentrated buying, spread selling)
        [(1, int(c), 400) for c in CODES[5:35]],
        # two brokers trade back and forth (circular)
        [(5, 6, 300), (6, 5, 300)] * 15,
        # a broker trading with itself
        [(7, 7, 500)] * 12 + [(8, 10, 100)] * 10,
        # thin: a handful of trades, below the smart-money floor
        [(11, 13, 50), (13, 11, 50), (11, 13, 40), (13, 11, 40)],
    ]
    for k, plant in enumerate(plants):
        symbols.append(f"P{k}")
        for b, s, q in plant:
            sym.append(len(symbols) - 1)
            buyer.append(b)
            seller.append(s)
            qty.append(q)
    return symbols, sym, buyer, seller, qty


EDGE_DAYS = {
    "one trade": (["ONE"], [0], [4], [8], [100]),
    "self-trades only": (["SELF"], [0] * 30, [7] * 30, [7] * 30, [250] * 30),
    "all one broker": (["MONO"], [0] * 25, [3] * 25, [3] * 20 + [4] * 5, [400] * 25),
    "zero quantity": (["ZERO", "SOME"], [0, 0, 1], [1, 2, 1], [2, 1, 3], [0, 0, 60]),
    "big broker codes": (["BIG"], [0] * 22, [5000, 7000] * 11, [7000, 9999] * 11, [300] * 22),
    "an empty symbol in the list": (["USED", "UNUSED"], [0, 0], [1, 2], [2, 1], [10, 10]),
}


# ── the scraper's own numbers ────────────────────────────────────────────────
def scraper_db(day, tmp_path) -> sqlite3.Connection:
    symbols, sym, buyer, seller, qty = day
    conn = sqlite3.connect(tmp_path / "floor.db")
    ensure_floorsheet_tables(conn)
    trades = [
        FloorTrade(f"T{i:07d}", symbols[s], DAY, int(b), int(sl), int(q), 100.0, 100.0 * q, "test", "now")
        for i, (s, b, sl, q) in enumerate(zip(sym, buyer, seller, qty))
    ]
    _insert_trades(conn, trades)
    for name in sorted({symbols[s] for s in sym}):
        _recompute_aggregates(conn, name, DAY)
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


SIGNAL_COLS = (
    "hhi_buy", "hhi_sell", "hhi_buy_norm", "hhi_sell_norm", "circular_score", "top_pair_pct",
    "self_trade_pct", "smart_money_score", "pump_score", "total_volume", "n_trades", "n_brokers_buy", "n_brokers_sell",
)


def assert_matches_scraper(day, tmp_path):
    symbols, sym, buyer, seller, qty = day
    got = fm.floor_day(symbols, sym, buyer, seller, qty, date=DAY, engine="python")
    conn = scraper_db(day, tmp_path)
    stored = {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM broker_signals_v2")}
    mine = {r["symbol"]: r for r in got.symbols}
    assert set(mine) == set(stored)
    for name, row in stored.items():
        for col in SIGNAL_COLS:
            assert mine[name][col] == row[col], (name, col, mine[name][col], row[col])

    # the panels: same rows, same order (SQL leaves exact ties unordered)
    latest, rows, top = fm.local_floor_records(conn)
    assert latest == DAY
    key = lambda r: (0 if (r["circular_score"] > 0.15 or r["pump_score"] > 0.05) else 1,  # noqa: E731
                     -r["circular_score"], -r["smart_money_score"], r["symbol"])
    cols = ("symbol", "hhi_buy", "hhi_sell", "circular_score", "pump_score", "self_trade_pct",
            "smart_money_score", "n_trades", "micro_score")
    local = sorted((dict(r) for r in rows), key=key)
    if len(local) < fm.SIGNAL_ROWS:
        assert [tuple(r[c] for c in cols) for r in fm.signal_records(got)] == [tuple(r[c] for c in cols) for r in local]
    tkey = lambda r: (-abs(r["net_qty"]), -r["total_trades"], r["broker_code"])  # noqa: E731
    tcols = ("broker_code", "buy_qty", "sell_qty", "net_qty", "total_trades")
    local_top = sorted((dict(r) for r in top), key=tkey)
    assert [tuple(r[c] for c in tcols) for r in fm.top_broker_records(got)] == [tuple(r[c] for c in tcols) for r in local_top]
    conn.close()
    return got


@pytest.mark.parametrize("seed", range(6))
def test_python_reproduces_the_scraper_on_random_days(seed, tmp_path):
    got = assert_matches_scraper(random_day(seed), tmp_path)
    labels = {r["symbol"]: r for r in fm.signal_records(got)}
    assert labels["P1"]["circular_score"] > 0.15  # the planted patterns really show up
    assert labels["P2"]["self_trade_pct"] > 0.10
    assert got.engine == "python"


@pytest.mark.parametrize("name", sorted(EDGE_DAYS))
def test_python_reproduces_the_scraper_on_edge_cases(name, tmp_path):
    assert_matches_scraper(EDGE_DAYS[name], tmp_path)


def test_smart_money_and_thin_stocks(tmp_path):
    got = assert_matches_scraper(random_day(11), tmp_path)
    rows = {r["symbol"]: r for r in got.symbols}
    assert rows["P0"]["smart_money_score"] > 0.08 and rows["P0"]["hhi_buy_norm"] == 1.0
    # below 20 trades / 5,000 shares: no smart-money or pump score, but circularity
    # is still computed and can flag a thin stock (the scraper does the same)
    assert rows["P3"]["smart_money_score"] == 0.0 and rows["P3"]["pump_score"] == 0.0
    assert rows["P3"]["circular_score"] > 0.15


# ── Rust == Python, to the bit ───────────────────────────────────────────────
def _raw(day, engine):
    symbols, sym, buyer, seller, qty = day
    codes = sorted(set(buyer) | set(seller))
    pos = {c: i for i, c in enumerate(codes)}
    return fm.compute(sym, [pos[b] for b in buyer], [pos[s] for s in seller], qty, len(symbols), len(codes), engine=engine)


def _same(a: dict, b: dict) -> None:
    for k in fm.RESULT_KEYS:
        assert a[k].dtype == b[k].dtype, k
        assert a[k].tobytes() == b[k].tobytes(), (k, a[k], b[k])


@needs_rust
@pytest.mark.parametrize("seed", range(20))
def test_rust_equals_python_on_random_days(seed):
    day = random_day(100 + seed, n_trades=int(np.random.default_rng(seed).integers(1, 6000)), n_symbols=1 + seed * 3)
    (py, e1), (rs, e2) = _raw(day, "python"), _raw(day, "rust")
    assert (e1, e2) == ("python", "rust")
    _same(py, rs)


@needs_rust
@pytest.mark.parametrize("name", sorted(EDGE_DAYS))
def test_rust_equals_python_on_edge_cases(name):
    _same(_raw(EDGE_DAYS[name], "python")[0], _raw(EDGE_DAYS[name], "rust")[0])
    a = fm.floor_day(*EDGE_DAYS[name], engine="python")
    b = fm.floor_day(*EDGE_DAYS[name], engine="rust")
    assert (a.symbols, a.brokers) == (b.symbols, b.brokers)


@needs_rust
def test_both_engines_refuse_bad_input():
    for engine in ("python", "rust"):
        with pytest.raises(ValueError):
            fm.compute([0, 0], [0], [0], [1], 1, 1, engine=engine)
        with pytest.raises(ValueError):
            fm.compute([2], [0], [0], [1], 1, 1, engine=engine)
        with pytest.raises(ValueError):
            fm.compute([0], [0], [0], [2**53], 1, 1, engine=engine)


def test_an_empty_day_works_in_python():
    raw, engine = fm.compute([], [], [], [], 0, 0, engine="python")
    assert engine == "python" and all(len(v) == 0 for v in raw.values())
    day = fm.floor_day([], [], [], [], [], date=DAY, engine="python")
    assert day.symbols == [] and day.brokers == [] and fm.signal_records(day) == []


# ── which engine runs ────────────────────────────────────────────────────────
def test_python_can_be_forced(monkeypatch):
    monkeypatch.setenv("NQT_FLOOR_ENGINE", "python")
    assert fm.engine_name() == "python"
    assert fm.compute([0], [0], [1], [5], 1, 2)[1] == "python"


def test_without_the_native_module_python_runs(monkeypatch):
    monkeypatch.setattr(fm, "native_module", lambda: (None, "native module not installed (ImportError)"))
    assert fm.engine_name() == "python"
    assert fm.floor_day(["A"], [0], [1], [2], [10]).engine == "python"
    with pytest.raises(RuntimeError):
        fm.engine_name("rust")


def test_a_native_module_of_another_version_is_not_used(monkeypatch):
    class Old:
        ABI = fm.ENGINE_ABI + 1

    import sys

    monkeypatch.setitem(sys.modules, "nqt_floor", Old)
    monkeypatch.setattr(fm, "_native_cache", {})
    mod, why = fm.native_module()
    assert mod is None and "another version" in why


# ── a website reply ──────────────────────────────────────────────────────────
def test_a_website_reply_becomes_a_day():
    payload = {
        "date": DAY, "version": f"{DAY}:5:x", "available_dates": [DAY], "complete": True, "notice": "n",
        "symbols": ["AHPC", "NABIL"], "brokers": [{"id": 1, "name": "Kumari Securities"}, {"id": 58, "name": None}],
        "symbol_idx": [1, 1, 0, 1, 1], "buyer": [1, 58, 34, 1, 58], "seller": [58, 1, 34, 58, 1],
        "qty": [100, 50, 1000, 10, 5], "rate": [520.5, 521, 210, 519.9, 520], "count": 5,
    }
    day = fm.floor_day_from_payload(payload, engine="python")
    assert (day.date, day.complete, day.version, day.count) == (DAY, True, f"{DAY}:5:x", 5)
    assert [r["symbol"] for r in day.symbols] == ["AHPC", "NABIL"]
    nabil = next(r for r in day.symbols if r["symbol"] == "NABIL")
    assert nabil["n_trades"] == 4 and nabil["total_volume"] == 165
    names = {b["broker_code"]: b["name"] for b in day.brokers}
    assert names == {1: "Kumari Securities", 34: None, 58: None}
    assert next(b for b in day.brokers if b["broker_code"] == 34)["total_trades"] == 2  # a self-trade counts both sides

    broken = dict(payload, qty=[1])
    with pytest.raises(ValueError):
        fm.floor_day_from_payload(broken)
    with pytest.raises(ValueError):
        fm.floor_day_from_payload(dict(payload, symbol_idx=[5, 1, 0, 1, 1]))


def test_signal_rows_are_capped_and_ordered():
    rows = [
        {"symbol": f"X{i:02d}", "circular_score": 0.2 + i / 1000, "pump_score": 0.0, "smart_money_score": 0.0,
         "self_trade_pct": 0.0, "hhi_buy": 0, "hhi_sell": 0, "n_trades": 1}
        for i in range(70)
    ] + [{"symbol": "SM", "circular_score": 0.0, "pump_score": 0.0, "smart_money_score": 0.5,
          "self_trade_pct": 0.0, "hhi_buy": 0, "hhi_sell": 0, "n_trades": 1}]
    day = fm.FloorDay(date=DAY, engine="python", symbols=rows)
    picked = fm.signal_records(day)
    assert len(picked) == 60 and picked[0]["symbol"] == "X69"
    assert "SM" not in {r["symbol"] for r in picked}  # flagged rows come first
    assert all(r["micro_score"] is None for r in picked)
