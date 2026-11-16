from __future__ import annotations

import json as _json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.quant_pro.dashboard_data import MD, _db, load_port

INITIAL_CAPITAL = 1_000_000.0

# Persistence helpers live in dashboard_tui until Step 5; dashboard_tui binds the
# real callables onto these names after import so _compute_portfolio_stats and
# _compute_account_portfolio_stats resolve them in this module's namespace.
_load_nav_log = None
_load_trade_log = None
_load_manual_paper_cash = None
_account_initial_capital_from_files = None


_SIGNAL_LABEL_MAP = {
    "fundamental": "Fundamental",
    "quality": "Quality",
    "momentum": "Momentum",
    "residual_momentum": "Residual Momentum",
    "xsec_momentum": "Cross-Sectional Momentum",
    "mean_reversion": "Mean Reversion",
    "liquidity": "Liquidity",
    "volume": "Volume",
    "volume_breakout": "Volume Breakout",
    "accumulation": "Accumulation",
    "corporate_action": "Corporate Action",
    "dividend": "Dividend",
    "sentiment": "Sentiment",
    "nlp_sentiment": "NLP Sentiment",
    "disposition": "Disposition",
    "anchoring_52wk": "52-Week Anchoring",
    "pairs_trade": "Pairs Trade",
    "macro_remittance": "Macro Remittance",
    "satellite_hydro": "Satellite Hydro",
    "settlement_pressure": "Settlement Pressure",
    "manual": "Manual",
    "tms": "TMS",
    "unknown": "Unknown",
}


def _signal_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Unknown"
    norm = raw.lower().replace("-", "_").replace(" ", "_")
    return _SIGNAL_LABEL_MAP.get(norm, raw.replace("_", " ").replace("-", " ").title())


def _signal_code(value: str) -> str:
    norm = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    code_map = {
        "fundamental": "F",
        "mean_reversion": "MR",
        "xsec_momentum": "XS",
        "accumulation": "A",
        "volume": "V",
        "quality": "Q",
        "low_vol": "LV",
        "residual_momentum": "RM",
            "tms": "TMS",
    }
    if not norm:
        return "?"
    return code_map.get(norm, norm[:3].upper())

def _get_regime(md: MD) -> str:
    """Get market regime from simple_backtest."""
    try:
        from backend.backtesting.simple_backtest import load_all_prices, compute_market_regime
        conn = _db()
        prices_df = load_all_prices(conn)
        conn.close()
        today = datetime.strptime(md.latest, "%Y-%m-%d")
        return compute_market_regime(prices_df, today)
    except Exception:
        return "unknown"

def _compute_portfolio_stats(md: MD) -> dict:
    """Compute full portfolio stats for portfolio/risk tabs."""
    ltps = md.ltps()
    quote_map = {}
    if not md.quotes.empty:
        quote_map = {
            str(r.symbol): {
                "ltp": float(getattr(r, "ltp", 0) or 0),
                "prev_close": float(getattr(r, "prev_close", 0) or 0),
                "pc": float(getattr(r, "pc", 0) or 0),
            }
            for r in md.quotes.itertuples()
        }
    port = load_port()
    nav_log = _load_nav_log()
    trade_log = _load_trade_log()

    positions = []
    total_cost = total_value = 0.0
    total_prev_value = 0.0
    sector_exposure = {}

    if not port.empty:
        for _, r in port.iterrows():
            sym = str(r["Symbol"]); qty = int(r["Quantity"])
            entry = float(r["Buy_Price"])
            cost = float(r.get("Total_Cost_Basis", entry * qty))
            quote = quote_map.get(sym, {})
            cur = ltps.get(sym) or float(r.get("Last_LTP") or entry)
            prev_close = float(quote.get("prev_close") or 0) or cur
            val = cur * qty; pnl = val - cost
            ret = pnl / cost * 100 if cost else 0
            day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
            day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
            entry_dt = str(r.get("Entry_Date", ""))[:10]
            days = 0
            try:
                days = (datetime.now() - datetime.strptime(entry_dt, "%Y-%m-%d")).days
            except Exception:
                pass
            total_cost += cost; total_value += val
            total_prev_value += prev_close * qty

            # Sector
            try:
                from backend.backtesting.simple_backtest import get_symbol_sector
                sec = get_symbol_sector(sym) or "Other"
            except Exception:
                sec = "Other"
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val

            positions.append({
                "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                "signal": _signal_label(str(r.get("Signal_Type", ""))),
                "date": entry_dt, "days": days, "sector": sec,
            })

    # NAV
    cash = _load_manual_paper_cash(total_cost, nav_log)
    nav = cash + total_value
    prev_nav = cash + total_prev_value
    day_pnl_total = nav - prev_nav
    day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
    baseline_nav = float(INITIAL_CAPITAL)
    if not nav_log.empty and "NAV" in nav_log.columns:
        try:
            baseline_nav = float(nav_log.iloc[0]["NAV"] or INITIAL_CAPITAL)
        except Exception:
            baseline_nav = float(INITIAL_CAPITAL)
    total_return = (nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    # Realized P&L and fees paid from trade log
    realized = 0.0
    fees_paid = 0.0
    if not trade_log.empty and "PnL" in trade_log.columns:
        realized = trade_log["PnL"].sum()
    if not trade_log.empty and "Fees" in trade_log.columns:
        fees_paid = float(pd.to_numeric(trade_log["Fees"], errors="coerce").fillna(0.0).sum())
    elif not port.empty and "Buy_Fees" in port.columns:
        fees_paid = float(pd.to_numeric(port["Buy_Fees"], errors="coerce").fillna(0.0).sum())

    gross_nav = nav + fees_paid
    gross_return = (gross_nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    # Max drawdown from NAV log
    max_dd = 0.0; peak_nav = baseline_nav; dd_date = ""
    if not nav_log.empty and "NAV" in nav_log.columns:
        for _, nr in nav_log.iterrows():
            n = float(nr["NAV"])
            if n > peak_nav:
                peak_nav = n
            dd = (n - peak_nav) / peak_nav * 100
            if dd < max_dd:
                max_dd = dd
                dd_date = str(nr.get("Date", ""))[:10]

    # NEPSE benchmark return
    nepse_ret = 0.0
    if len(md.nepse) >= 2:
        ni = md.nepse.iloc[0]["close"]
        # Get NEPSE at portfolio start
        try:
            conn = _db()
            start = pd.read_sql_query(
                "SELECT close FROM stock_prices WHERE symbol='NEPSE' AND date>=? "
                "ORDER BY date LIMIT 1", conn, params=("2026-02-09",))
            conn.close()
            if not start.empty:
                nepse_ret = (ni - start.iloc[0]["close"]) / start.iloc[0]["close"] * 100
        except Exception:
            pass

    # Concentration
    positions.sort(key=lambda x: x["val"], reverse=True)
    top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0

    # Winners/losers
    winners = [p for p in positions if p["pnl"] > 0]
    losers = [p for p in positions if p["pnl"] < 0]

    # Holding age buckets
    age_0_5 = sum(1 for p in positions if p["days"] <= 5)
    age_6_15 = sum(1 for p in positions if 6 <= p["days"] <= 15)
    age_16 = sum(1 for p in positions if p["days"] > 15)

    return {
        "positions": positions,
        "total_cost": total_cost, "total_value": total_value,
        "cash": cash, "nav": nav, "total_return": total_return,
        "gross_nav": gross_nav, "gross_return": gross_return, "fees_paid": fees_paid,
        "day_pnl": day_pnl_total, "day_ret": day_ret_total,
        "realized": realized, "unrealized": total_value - total_cost,
        "max_dd": max_dd, "dd_date": dd_date, "peak_nav": peak_nav,
        "nepse_ret": nepse_ret, "alpha": total_return - nepse_ret,
        "n_positions": len(positions),
        "sector_exposure": sector_exposure,
        "top3_conc": top3_conc,
        "winners": winners, "losers": losers,
        "age_0_5": age_0_5, "age_6_15": age_6_15, "age_16": age_16,
        "trade_log": trade_log, "nav_log": nav_log,
    }


def _compute_account_portfolio_stats(md: MD, account_dir: Path) -> dict:
    """Compute portfolio stats from account-specific files (paper_portfolio.csv etc.)."""
    _port_path = account_dir / "paper_portfolio.csv"
    _nav_path = account_dir / "paper_nav_log.csv"
    _tl_path = account_dir / "paper_trade_log.csv"
    _state_path = account_dir / "paper_state.json"

    port = pd.read_csv(_port_path) if _port_path.exists() else pd.DataFrame()
    nav_log = pd.read_csv(_nav_path) if _nav_path.exists() else pd.DataFrame()
    trade_log = pd.read_csv(_tl_path) if _tl_path.exists() else pd.DataFrame()
    account_capital = _account_initial_capital_from_files(account_dir)

    cash: Optional[float] = None
    if _state_path.exists():
        try:
            _ps = _json.loads(_state_path.read_text())
            _c = float(_ps.get("cash", 0))
            if _c >= 0:
                cash = _c
        except Exception:
            pass
    return _stats_from_frames(
        md, port, nav_log, trade_log,
        account_capital=account_capital,
        cash=cash,
        derive_cash=not _state_path.exists(),
    )


def _stats_from_frames(
    md: MD,
    port: pd.DataFrame,
    nav_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    account_capital: float,
    cash: Optional[float],
    derive_cash: bool = False,
    prefer_book_price: bool = False,
) -> dict:
    """Portfolio stats from a book's frames: the account files, or the website's
    paper book (``prefer_book_price``: value holdings at the book's own last
    price, so the terminal shows the same value as the website)."""
    ltps = md.ltps()
    quote_map = {}
    if not md.quotes.empty:
        quote_map = {
            str(r.symbol): {
                "ltp": float(getattr(r, "ltp", 0) or 0),
                "prev_close": float(getattr(r, "prev_close", 0) or 0),
                "pc": float(getattr(r, "pc", 0) or 0),
            }
            for r in md.quotes.itertuples()
        }

    positions = []
    total_cost = total_value = 0.0
    total_prev_value = 0.0
    sector_exposure = {}

    if not port.empty:
        for _, r in port.iterrows():
            sym = str(r["Symbol"]); qty = int(r["Quantity"])
            entry = float(r["Buy_Price"])
            cost = float(r.get("Total_Cost_Basis", entry * qty))
            quote = quote_map.get(sym, {})
            book_px = float(r.get("Last_LTP") or 0) if prefer_book_price else 0.0
            cur = book_px or ltps.get(sym) or float(r.get("Last_LTP") or entry)
            prev_close = float(quote.get("prev_close") or 0) or cur
            val = cur * qty; pnl = val - cost
            ret = pnl / cost * 100 if cost else 0
            day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
            day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
            entry_dt = str(r.get("Entry_Date", ""))[:10]
            days = 0
            try:
                days = (datetime.now() - datetime.strptime(entry_dt, "%Y-%m-%d")).days
            except Exception:
                pass
            total_cost += cost; total_value += val
            total_prev_value += prev_close * qty
            try:
                from backend.backtesting.simple_backtest import get_symbol_sector
                sec = get_symbol_sector(sym) or "Other"
            except Exception:
                sec = "Other"
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val
            positions.append({
                "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                "signal": _signal_label(str(r.get("Signal_Type", ""))),
                "date": entry_dt, "days": days, "sector": sec,
            })

    # Cash: the book's own figure (paper_state.json / the website); a book with
    # no state file derives it from the starting capital.
    if cash is None:
        cash = float(account_capital)
        if derive_cash and total_cost > 0:
            cash = max(0.0, float(account_capital) - total_cost)

    nav = cash + total_value
    baseline_nav = float(account_capital)
    if not nav_log.empty and "NAV" in nav_log.columns:
        try:
            baseline_nav = float(nav_log.iloc[0]["NAV"] or account_capital)
        except Exception:
            pass
    total_return = (nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    realized = 0.0
    fees_paid = 0.0
    if not trade_log.empty and "PnL" in trade_log.columns:
        realized = float(pd.to_numeric(trade_log["PnL"], errors="coerce").fillna(0).sum())
    if not trade_log.empty and "Fees" in trade_log.columns:
        fees_paid = float(pd.to_numeric(trade_log["Fees"], errors="coerce").fillna(0).sum())

    max_dd = 0.0; peak_nav = baseline_nav; dd_date = ""
    if not nav_log.empty and "NAV" in nav_log.columns:
        for _, nr in nav_log.iterrows():
            n = float(nr["NAV"])
            if n > peak_nav:
                peak_nav = n
            dd = (n - peak_nav) / peak_nav * 100
            if dd < max_dd:
                max_dd = dd
                dd_date = str(nr.get("Date", ""))[:10]

    prev_nav = cash + total_prev_value
    day_pnl_total = nav - prev_nav
    day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
    positions.sort(key=lambda x: x["val"], reverse=True)
    top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0
    winners = [p for p in positions if p["pnl"] > 0]
    losers = [p for p in positions if p["pnl"] < 0]
    age_0_5 = sum(1 for p in positions if p["days"] <= 5)
    age_6_15 = sum(1 for p in positions if 6 <= p["days"] <= 15)
    age_16 = sum(1 for p in positions if p["days"] > 15)

    return {
        "positions": positions,
        "total_cost": total_cost, "total_value": total_value,
        "cash": cash, "nav": nav, "total_return": total_return,
        "gross_nav": nav + fees_paid, "gross_return": total_return, "fees_paid": fees_paid,
        "day_pnl": day_pnl_total, "day_ret": day_ret_total,
        "realized": realized, "unrealized": total_value - total_cost,
        "max_dd": max_dd, "dd_date": dd_date, "peak_nav": peak_nav,
        "nepse_ret": 0, "alpha": total_return,
        "n_positions": len(positions), "sector_exposure": sector_exposure,
        "top3_conc": top3_conc, "winners": winners, "losers": losers,
        "age_0_5": age_0_5, "age_6_15": age_6_15, "age_16": age_16,
        "trade_log": trade_log, "nav_log": nav_log,
    }
