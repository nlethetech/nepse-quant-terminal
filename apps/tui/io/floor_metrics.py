"""Broker floor metrics for one trading day, computed from the day's floorsheet.

These are the numbers behind the Signals tab's BROKER FLOOR and TOP BROKERS
panels. They describe where the day's trades were placed (how concentrated
buying and selling were across brokers, how much volume went back and forth
between the same brokers, how much a broker traded with itself). They are
analytics, not trade signals: a broker acts for many clients.

The definitions are exactly those of ``backend/quant_pro/floorsheet_scraper.py``
(``_hhi_normalized``, ``_compute_signal_metrics``, ``_recompute_aggregates``),
which fill the panels from a locally scraped floorsheet, and the panels pick
and order rows the same way the local SQL does (``local_floor_records``).

Two engines compute them, with identical results:

* ``rust``   the optional native module ``nqt_floor`` (native/nqt_floor/), when
  it is installed: ``pip install maturin && maturin develop --release -m
  native/nqt_floor/Cargo.toml``
* ``python`` numpy, always available.

``compute()`` uses the native module when it imports (``NQT_FLOOR_ENGINE=python``
forces the fallback) and says which one ran. Both add brokers in ascending
order and only add whole numbers otherwise, so they agree to the last bit.
Against the scraper's stored values they agree after its rounding to 6 decimals
(Python 3.12's ``sum()`` compensates rounding error; plain addition here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

ENGINE_ABI = 1

# backend/quant_pro/floorsheet_scraper.py
MIN_SIGNAL_TRADES = 20
MIN_SIGNAL_VOLUME = 5000
NET_RATIO_MAX = 0.15
MIN_CIRC_SHARE = 0.05
# Sums stay exact (float64 and int64 alike) below 2**53 shares.
MAX_ABS_QTY_SUM = 1 << 53

# The local panel's SQL (apps/tui/state/mixins/tab_refresh.py, broker floor block)
FLAG_CIRCULAR = 0.15
FLAG_PUMP = 0.05
FLAG_SMART = 0.08
SIGNAL_ROWS = 60
TOP_BROKER_ROWS = 25

SYMBOL_FLOATS = (
    "hhi_buy", "hhi_sell", "hhi_buy_norm", "hhi_sell_norm",
    "circular_score", "top_pair_pct", "self_trade_pct", "smart_money_score", "pump_score",
)
SYMBOL_INTS = ("n_trades", "total_qty", "n_brokers_buy", "n_brokers_sell")
BROKER_INTS = ("buy_qty", "sell_qty", "buy_trades", "sell_trades")
RESULT_KEYS = SYMBOL_INTS + SYMBOL_FLOATS + BROKER_INTS

_DIRECT_BROKER_CODES = 4096  # codes below this are used as array positions as-is
_DENSE_CELLS = 1 << 21  # symbol x broker table size for the dense numpy path

_native_cache: dict[str, Any] = {}


# ── engines ──────────────────────────────────────────────────────────────────
def native_module() -> tuple[Optional[Any], str]:
    """(the native module or None, why it is not used)."""
    if "mod" not in _native_cache:
        try:
            import nqt_floor  # type: ignore[import-not-found]
        except Exception as exc:  # not built / not installed / wrong platform
            _native_cache.update(mod=None, why=f"native module not installed ({type(exc).__name__})")
        else:
            if getattr(nqt_floor, "ABI", None) == ENGINE_ABI:
                _native_cache.update(mod=nqt_floor, why="")
            else:
                _native_cache.update(mod=None, why="native module is from another version; rebuild it")
    return _native_cache["mod"], _native_cache["why"]


def engine_name(engine: Optional[str] = None) -> str:
    """The engine ``compute()`` will use: "rust" or "python"."""
    wanted = (engine or os.environ.get("NQT_FLOOR_ENGINE", "")).strip().lower()
    if wanted == "python":
        return "python"
    mod, _ = native_module()
    if mod is None:
        if wanted == "rust":
            raise RuntimeError(native_module()[1])
        return "python"
    return "rust"


def _check(sym: np.ndarray, buyer: np.ndarray, seller: np.ndarray, qty: np.ndarray, n_symbols: int, n_brokers: int) -> None:
    n = len(sym)
    if not (len(buyer) == len(seller) == len(qty) == n):
        raise ValueError(
            f"arrays differ in length: symbol_idx {n}, buyer {len(buyer)}, seller {len(seller)}, qty {len(qty)}"
        )
    for name, arr, limit in (("symbol", sym, n_symbols), ("buyer", buyer, n_brokers), ("seller", seller, n_brokers)):
        if n and (int(arr.min()) < 0 or int(arr.max()) >= limit):
            raise ValueError(f"{name} index out of range (limit {limit})")
    if n and int(np.abs(qty).sum(dtype=np.float64)) >= MAX_ABS_QTY_SUM:
        raise ValueError("quantities too large to add up exactly")


def compute_python(
    symbol_idx: Sequence[int], buyer: Sequence[int], seller: Sequence[int], qty: Sequence[int],
    n_symbols: int, n_brokers: int,
) -> dict[str, np.ndarray]:
    """The numpy engine. Every sum over brokers runs in ascending broker order
    (``np.bincount`` adds in input order), every other sum is over whole numbers."""
    sym = np.asarray(symbol_idx, dtype=np.int64)
    buy_b = np.asarray(buyer, dtype=np.int64)
    sell_b = np.asarray(seller, dtype=np.int64)
    q = np.asarray(qty, dtype=np.int64)
    S, B = int(n_symbols), int(n_brokers)
    _check(sym, buy_b, sell_b, q, S, B)
    qf = q.astype(np.float64)

    def per_symbol(idx: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return np.bincount(idx, weights=weights, minlength=S)[:S]

    n_trades = np.bincount(sym, minlength=S)[:S].astype(np.int64)
    total_f = per_symbol(sym, qf)
    total = total_f.astype(np.int64)

    # (symbol, broker) cells, symbol-major and brokers ascending within a symbol
    if S * B <= _DENSE_CELLS:
        buy_cells = np.bincount(sym * B + buy_b, weights=qf, minlength=S * B)
        sell_cells = np.bincount(sym * B + sell_b, weights=qf, minlength=S * B)
        cell = np.flatnonzero((buy_cells != 0) | (sell_cells != 0))
        cs, bv, sv = cell // B, buy_cells[cell], sell_cells[cell]
    else:
        keys = np.concatenate([sym * B + buy_b, sym * B + sell_b])
        cell, inv = np.unique(keys, return_inverse=True)
        n = len(sym)
        bv = np.bincount(inv[:n], weights=qf, minlength=len(cell))
        sv = np.bincount(inv[n:], weights=qf, minlength=len(cell))
        cs = cell // B

    def hhi_side(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = values > 0
        s_pos, v_pos = cs[pos], values[pos]
        count = per_symbol(s_pos).astype(np.int64)
        pos_total = per_symbol(s_pos, v_pos)
        share = v_pos / pos_total[s_pos]
        hhi = per_symbol(s_pos, share * share)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_n = 1.0 / count.astype(np.float64)
            norm = (hhi - inv_n) / (1.0 - inv_n)
        norm = np.where(norm < 1.0, norm, 1.0)
        norm = np.where(norm > 0.0, norm, 0.0)
        norm = np.where(count == 1, 1.0, np.where(count == 0, 0.0, norm))
        hhi = np.where(count == 0, 0.0, hhi)
        return hhi, norm, count

    hhi_buy, hhi_buy_norm, n_buy = hhi_side(bv)
    hhi_sell, hhi_sell_norm, n_sell = hhi_side(sv)

    has_qty = total > 0
    safe_total = np.where(has_qty, total_f, 1.0)

    self_mask = buy_b == sell_b
    self_pct = np.where(has_qty, per_symbol(sym[self_mask], qf[self_mask]) / safe_total, 0.0)

    # largest (buyer, seller) pair per symbol
    top_pair = np.zeros(S, dtype=np.float64)
    if len(sym):
        if S * B * B < (1 << 62):
            pair_keys = (sym * B + buy_b) * B + sell_b
            pairs, inv = np.unique(pair_keys, return_inverse=True)
            pair_sym = pairs // (B * B)
        else:  # pragma: no cover - only for absurd broker counts
            pairs, inv = np.unique(np.stack([sym, buy_b, sell_b], axis=1), axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            pair_sym = pairs[:, 0]
        pair_sum = np.bincount(inv, weights=qf, minlength=len(pairs))
        starts = np.flatnonzero(np.r_[True, pair_sym[1:] != pair_sym[:-1]])
        top_pair[pair_sym[starts]] = np.maximum.reduceat(pair_sum, starts)
    top_pct = np.where(has_qty, top_pair / safe_total, 0.0)

    total_buy = per_symbol(cs, bv)
    total_sell = per_symbol(cs, sv)
    tb, ts = total_buy[cs], total_sell[cs]
    with np.errstate(divide="ignore", invalid="ignore"):
        balanced = np.abs(bv - sv) / (bv + sv + 1e-9) < NET_RATIO_MAX
    circular_cell = (tb > 0) & (bv >= MIN_CIRC_SHARE * tb) & (ts > 0) & (sv >= MIN_CIRC_SHARE * ts) & balanced
    circ_qty = per_symbol(cs[circular_cell], (bv + sv)[circular_cell])
    circular = np.where(has_qty, circ_qty / (total_buy + total_sell + 1e-9), 0.0)

    eligible = has_qty & (n_trades >= MIN_SIGNAL_TRADES) & (total >= MIN_SIGNAL_VOLUME)
    pump = np.where(
        self_pct >= 0.05, self_pct, np.where((top_pct > 0.40) & (circular > 0.20), top_pct * 0.5, 0.0)
    )
    pump = np.where(eligible, pump, 0.0)
    buy_component = (hhi_buy_norm - 0.20) / 0.60
    buy_component = np.where(1.0 < buy_component, 1.0, buy_component)
    sell_component = 1.0 - hhi_sell_norm / 0.40
    sell_component = np.where(0.0 > sell_component, 0.0, sell_component)
    circ_penalty = 1.0 - circular / 0.30
    circ_penalty = np.where(0.0 > circ_penalty, 0.0, circ_penalty)
    smart_ok = eligible & (hhi_buy_norm > 0.20) & (hhi_sell_norm < 0.40) & (circular < 0.30) & (self_pct < 0.10)
    smart = np.where(smart_ok, buy_component * sell_component * circ_penalty, 0.0)

    def per_broker(idx: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return np.bincount(idx, weights=weights, minlength=B)[:B].astype(np.int64)

    return {
        "n_trades": n_trades,
        "total_qty": total,
        "hhi_buy": hhi_buy,
        "hhi_sell": hhi_sell,
        "hhi_buy_norm": hhi_buy_norm,
        "hhi_sell_norm": hhi_sell_norm,
        "n_brokers_buy": n_buy,
        "n_brokers_sell": n_sell,
        "circular_score": circular,
        "top_pair_pct": top_pct,
        "self_trade_pct": self_pct,
        "smart_money_score": smart,
        "pump_score": pump,
        "buy_qty": per_broker(buy_b, qf),
        "sell_qty": per_broker(sell_b, qf),
        "buy_trades": per_broker(buy_b),
        "sell_trades": per_broker(sell_b),
    }


def compute_rust(
    symbol_idx: Sequence[int], buyer: Sequence[int], seller: Sequence[int], qty: Sequence[int],
    n_symbols: int, n_brokers: int,
) -> dict[str, np.ndarray]:
    mod, why = native_module()
    if mod is None:
        raise RuntimeError(why)
    sym = np.asarray(symbol_idx, dtype=np.int64)
    buy_b = np.asarray(buyer, dtype=np.int64)
    sell_b = np.asarray(seller, dtype=np.int64)
    q = np.asarray(qty, dtype=np.int64)
    _check(sym, buy_b, sell_b, q, int(n_symbols), int(n_brokers))
    raw = mod.compute_floor(
        sym.astype("<u4").tobytes(), buy_b.astype("<u4").tobytes(), sell_b.astype("<u4").tobytes(),
        q.astype("<i8").tobytes(), int(n_symbols), int(n_brokers),
    )
    return {
        key: np.asarray(raw[key], dtype=np.float64 if key in SYMBOL_FLOATS else np.int64) for key in RESULT_KEYS
    }


def compute(
    symbol_idx: Sequence[int], buyer: Sequence[int], seller: Sequence[int], qty: Sequence[int],
    n_symbols: int, n_brokers: int, *, engine: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], str]:
    """Raw per-symbol and per-broker arrays, and the engine that made them."""
    name = engine_name(engine)
    run = compute_rust if name == "rust" else compute_python
    return run(symbol_idx, buyer, seller, qty, n_symbols, n_brokers), name


# ── one day, ready for the panels ────────────────────────────────────────────
@dataclass
class FloorDay:
    date: Optional[str]
    engine: str
    symbols: list[dict[str, Any]] = field(default_factory=list)  # one per traded symbol
    brokers: list[dict[str, Any]] = field(default_factory=list)  # one per broker that traded
    count: int = 0
    complete: Optional[bool] = None
    version: str = ""
    available_dates: list[str] = field(default_factory=list)
    notice: str = ""


def _symbol_row(symbol: str, raw: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    """A ``broker_signals_v2`` row as ``_compute_signal_metrics`` stores it."""
    total = int(raw["total_qty"][i])
    hhi = {k: float(raw[k][i]) for k in ("hhi_buy", "hhi_sell", "hhi_buy_norm", "hhi_sell_norm")}
    base = {
        "symbol": symbol,
        "n_trades": int(raw["n_trades"][i]),
        "n_brokers_buy": int(raw["n_brokers_buy"][i]),
        "n_brokers_sell": int(raw["n_brokers_sell"][i]),
    }
    if total <= 0:  # stored unrounded, as the scraper does
        return {
            **base, **hhi, "circular_score": 0.0, "top_pair_pct": 0.0, "self_trade_pct": 0.0,
            "smart_money_score": 0.0, "pump_score": 0.0, "total_volume": 0,
        }
    return {
        **base,
        **{k: round(v, 6) for k, v in hhi.items()},
        "circular_score": round(float(raw["circular_score"][i]), 6),
        "top_pair_pct": round(float(raw["top_pair_pct"][i]), 6),
        "self_trade_pct": round(float(raw["self_trade_pct"][i]), 6),
        "smart_money_score": round(min(float(raw["smart_money_score"][i]), 1.0), 6),
        "pump_score": round(min(float(raw["pump_score"][i]), 1.0), 6),
        "total_volume": total,
    }


def floor_day(
    symbols: Sequence[str], symbol_idx: Sequence[int], buyer_codes: Sequence[int], seller_codes: Sequence[int],
    qty: Sequence[int], *, broker_names: Optional[dict[int, Optional[str]]] = None,
    date: Optional[str] = None, engine: Optional[str] = None,
) -> FloorDay:
    """Metrics for one day from its trades (broker NEPSE codes, not positions)."""
    symbols = [str(s) for s in symbols]
    buyer = np.asarray(buyer_codes, dtype=np.int64)
    seller = np.asarray(seller_codes, dtype=np.int64)
    if len(buyer) and (min(int(buyer.min()), int(seller.min())) < 0 or max(int(buyer.max()), int(seller.max())) >= _DIRECT_BROKER_CODES):
        codes, inv = np.unique(np.concatenate([buyer, seller]), return_inverse=True)
        b_idx, s_idx, n_brokers = inv[: len(buyer)], inv[len(buyer):], len(codes)
    else:
        n_brokers = (max(int(buyer.max()), int(seller.max())) + 1) if len(buyer) else 0
        codes, b_idx, s_idx = np.arange(n_brokers), buyer, seller
    raw, used = compute(symbol_idx, b_idx, s_idx, qty, len(symbols), n_brokers, engine=engine)
    names = broker_names or {}
    rows = [_symbol_row(symbols[i], raw, i) for i in np.flatnonzero(raw["n_trades"] > 0)]
    brokers = []
    for i in np.flatnonzero((raw["buy_trades"] + raw["sell_trades"]) > 0):
        code = int(codes[i])
        buy_q, sell_q = int(raw["buy_qty"][i]), int(raw["sell_qty"][i])
        brokers.append({
            "broker_code": code,
            "name": names.get(code),
            "buy_qty": buy_q,
            "sell_qty": sell_q,
            "net_qty": buy_q - sell_q,
            "total_trades": int(raw["buy_trades"][i] + raw["sell_trades"][i]),
        })
    return FloorDay(date=date, engine=used, symbols=rows, brokers=brokers, count=len(buyer))


def floor_day_from_payload(payload: dict[str, Any], *, engine: Optional[str] = None) -> FloorDay:
    """Metrics for the day in a nepsequant.com ``/api/cli/floorsheet`` reply."""
    symbols = list(payload.get("symbols") or [])
    cols = {k: list(payload.get(k) or []) for k in ("symbol_idx", "buyer", "seller", "qty")}
    n = len(cols["symbol_idx"])
    if any(len(v) != n for v in cols.values()):
        raise ValueError("the floorsheet reply has columns of different lengths")
    if n and (min(cols["symbol_idx"]) < 0 or max(cols["symbol_idx"]) >= len(symbols)):
        raise ValueError("the floorsheet reply points at a symbol it does not list")
    names = {}
    for b in payload.get("brokers") or []:
        try:
            names[int(b.get("id"))] = b.get("name") or None
        except (TypeError, ValueError, AttributeError):
            continue
    day = floor_day(
        symbols, cols["symbol_idx"], cols["buyer"], cols["seller"], cols["qty"],
        broker_names=names, date=payload.get("date"), engine=engine,
    )
    day.complete = payload.get("complete")
    day.version = str(payload.get("version") or "")
    day.available_dates = [str(d) for d in payload.get("available_dates") or []]
    day.notice = str(payload.get("notice") or "")
    return day


# ── what the panels show ─────────────────────────────────────────────────────
def _flagged(row: dict[str, Any]) -> bool:
    return row["circular_score"] > FLAG_CIRCULAR or row["pump_score"] > FLAG_PUMP


def signal_records(day: FloorDay) -> list[dict[str, Any]]:
    """The BROKER FLOOR rows, chosen and ordered like the local panel's SQL:
    circular > 0.15 or pump > 0.05 or smart-money > 0.08; circular/pump rows
    first, then circular score and smart-money score, highest first; 60 rows.
    Ties (SQL leaves them unordered) go by symbol."""
    rows = [
        r for r in day.symbols
        if r["circular_score"] > FLAG_CIRCULAR or r["pump_score"] > FLAG_PUMP or r["smart_money_score"] > FLAG_SMART
    ]
    rows.sort(key=lambda r: (0 if _flagged(r) else 1, -r["circular_score"], -r["smart_money_score"], r["symbol"]))
    # No microstructure score comes with a floorsheet (the local table is empty
    # in this repo too), so the panel's display score uses its neutral 0.5.
    return [{**r, "micro_score": None} for r in rows[:SIGNAL_ROWS]]


def top_broker_records(day: FloorDay) -> list[dict[str, Any]]:
    """The TOP BROKERS rows, ordered like the local panel's SQL: largest
    absolute net quantity first, then most trades; 25 rows. Ties go by code."""
    rows = sorted(day.brokers, key=lambda b: (-abs(b["net_qty"]), -b["total_trades"], b["broker_code"]))
    return rows[:TOP_BROKER_ROWS]


def local_floor_records(conn) -> tuple[Optional[str], list[Any], list[Any]]:
    """The same two panels from the local scraped database (rows support
    ``row["column"]``): (latest date or None, signal rows, top broker rows).

    This is the terminal's original query, unchanged. A failure reading the
    signal rows raises; a failure reading the broker rows gives no broker rows.
    """
    try:
        latest = (conn.execute("SELECT MAX(as_of_date) FROM broker_signals_v2").fetchone() or [None])[0]
    except Exception:
        latest = None
    if not latest:
        return None, [], []
    rows = conn.execute("""
        SELECT b.symbol,
               b.hhi_buy, b.hhi_sell,
               b.circular_score, b.pump_score,
               COALESCE(b.self_trade_pct, b.pump_score) AS self_trade_pct,
               b.smart_money_score, b.n_trades,
               m.micro_score
        FROM broker_signals_v2 b
        LEFT JOIN broker_microstructure m
          ON b.symbol = m.symbol AND m.as_of_date = b.as_of_date
        WHERE b.as_of_date = ?
          AND (b.circular_score > 0.15 OR b.pump_score > 0.05 OR b.smart_money_score > 0.08)
        ORDER BY
            CASE WHEN b.circular_score > 0.15 OR b.pump_score > 0.05 THEN 0 ELSE 1 END ASC,
            b.circular_score DESC,
            b.smart_money_score DESC
        LIMIT 60
    """, (latest,)).fetchall()
    try:
        top = conn.execute("""
            SELECT broker_code,
                   SUM(buy_qty) AS buy_qty,
                   SUM(sell_qty) AS sell_qty,
                   SUM(net_qty) AS net_qty,
                   SUM(total_trades) AS total_trades
            FROM broker_summary
            WHERE as_of_date = ?
            GROUP BY broker_code
            ORDER BY ABS(SUM(net_qty)) DESC, SUM(total_trades) DESC
            LIMIT 25
        """, (latest,)).fetchall()
    except Exception:
        top = []
    return str(latest), list(rows), list(top)
