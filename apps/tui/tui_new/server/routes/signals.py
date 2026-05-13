"""Signals endpoints — wraps signal generation and corporate actions."""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Request

from backend.quant_pro.database import get_db_path

router = APIRouter()

# Cache
_signals_cache: dict = {"data": None, "ts": None, "date": None}


def _signal_to_dict(signal, signal_type: str, date: str) -> dict:
    return {
        "symbol": str(getattr(signal, "symbol", "")),
        "signal_type": signal_type,
        "score": float(getattr(signal, "score", 0.0) or 0.0),
        "strength": float(getattr(signal, "strength", 0.0) or 0.0),
        "confidence": float(getattr(signal, "confidence", 0.0) or 0.0),
        "reasoning": str(getattr(signal, "reasoning", "")),
        "date": date,
    }


@router.get("")
async def get_signals(request: Request):
    today = datetime.now().strftime("%Y-%m-%d")

    # Return cached if same day
    if _signals_cache["date"] == today and _signals_cache["data"] is not None:
        return _signals_cache["data"]

    try:
        from backend.backtesting.simple_backtest import (
            build_symbol_price_cache,
            generate_volume_breakout_signals_at_date,
            generate_quality_signals_at_date,
        )

        conn = sqlite3.connect(get_db_path())
        prices = pd.read_sql_query(
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM stock_prices WHERE symbol != 'NEPSE' ORDER BY date",
            conn,
            parse_dates=["date"],
        )
        conn.close()
        symbol_cache = build_symbol_price_cache(prices)

        signals_out = []

        # Generate volume signals
        try:
            vol_sigs = generate_volume_breakout_signals_at_date(
                prices,
                date=pd.Timestamp(today),
                symbol_cache=symbol_cache,
            )
            for s in (vol_sigs or []):
                signals_out.append(_signal_to_dict(s, "volume", today))
        except Exception:
            pass

        # Generate quality signals
        try:
            qual_sigs = generate_quality_signals_at_date(
                prices,
                date=pd.Timestamp(today),
                symbol_cache=symbol_cache,
            )
            for s in (qual_sigs or []):
                signals_out.append(_signal_to_dict(s, "quality", today))
        except Exception:
            pass

        _signals_cache["data"] = signals_out
        _signals_cache["ts"] = datetime.now()
        _signals_cache["date"] = today
        return signals_out

    except Exception:
        return []


@router.get("/calendar")
async def get_calendar(request: Request):
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query(
            "SELECT symbol, fiscal_year, bookclose_date, cash_dividend_pct, "
            "bonus_share_pct, right_share_ratio, agenda "
            "FROM corporate_actions ORDER BY bookclose_date DESC LIMIT 50",
            conn,
        )
        conn.close()
        if df.empty:
            return []
        return [
            {
                "symbol": str(r.get("symbol", "")),
                "fiscal_year": str(r.get("fiscal_year", "")),
                "bookclose_date": str(r.get("bookclose_date", "")),
                "cash_dividend_pct": float(r.get("cash_dividend_pct", 0) or 0),
                "bonus_share_pct": float(r.get("bonus_share_pct", 0) or 0),
                "right_share_ratio": str(r.get("right_share_ratio", "") or ""),
                "agenda": str(r.get("agenda", "") or ""),
            }
            for _, r in df.iterrows()
        ]
    except Exception:
        return []


@router.get("/screener")
async def get_screener(request: Request):
    """Active stocks screener with basic metrics."""
    try:
        md = request.app.state.md
        if hasattr(md, 'df') and not md.df.empty:
            df = md.df.copy()
            df = df.sort_values("vol", ascending=False).head(50)
            return [
                {
                    "symbol": str(r.get("symbol", "")),
                    "ltp": float(r.get("ltp", r.get("close", 0))),
                    "change_pct": float(r.get("pc", r.get("chg_pct", 0))),
                    "volume": int(r.get("vol", r.get("volume", 0))),
                    "turnover": float(r.get("turnover", 0)),
                }
                for _, r in df.iterrows()
            ]
    except Exception:
        pass
    return []
