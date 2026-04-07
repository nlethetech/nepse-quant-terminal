"""
Corporate actions ingestion (AGM / dividend / bonus / rights) from MeroLagani.

This module focuses on the "AGM" / actions table visible on:
https://merolagani.com/CompanyDetail.aspx?symbol=SYMBOL

Design goals:
- Robust parsing across minor HTML/layout changes
- Idempotent persistence into SQLite (see database_extended.py helpers)
- Explicit, explainable outputs suitable for UI display
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Tuple, Dict, Any

import re
import sqlite3

import requests
from bs4 import BeautifulSoup

from .database import get_db_connection


MEROLAGANI_BASE = "https://merolagani.com"


@dataclass(frozen=True)
class CorporateActionRow:
    symbol: str
    fiscal_year: Optional[str]
    bookclose_date_ad: Optional[date]
    description: str
    agenda: Optional[str]
    cash_dividend_pct: Optional[float]
    bonus_share_pct: Optional[float]
    right_share_ratio: Optional[str]
    source_url: str
    scraped_at_utc: datetime


def _parse_percent(text: str) -> Optional[float]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("%", "").strip()
    try:
        return float(raw)
    except Exception:
        return None


def _parse_bookclose_date(text: str) -> Optional[date]:
    raw = (text or "").strip()
    if not raw:
        return None
    # Common: "YYYY/MM/DD AD (....)" or "YYYY-MM-DD"
    m = re.search(r"(\d{4}/\d{2}/\d{2})", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y/%m/%d").date()
        except Exception:
            pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass
    return None


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def fetch_company_detail_html(
    symbol: str,
    *,
    session: Optional[requests.Session] = None,
    timeout_s: int = 25,
) -> Tuple[str, str]:
    sym = str(symbol).strip().upper()
    url = f"{MEROLAGANI_BASE}/CompanyDetail.aspx?symbol={sym}"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text, url


def fetch_company_agm_tab_html(
    symbol: str,
    *,
    session: Optional[requests.Session] = None,
    timeout_s: int = 25,
) -> Tuple[str, str]:
    """
    CompanyDetail.aspx loads AGM history via an ASP.NET postback + pager postback.

    This function returns the fully populated HTML for the AGM tab (with the table).
    """
    sym = str(symbol).strip().upper()
    url = f"{MEROLAGANI_BASE}/CompanyDetail.aspx?symbol={sym}"
    sess = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    html0 = sess.get(url, timeout=timeout_s, headers=headers).text
    soup0 = BeautifulSoup(html0, "html.parser")
    payload0 = {i.get("name"): i.get("value", "") for i in soup0.select("input[type=hidden]") if i.get("name")}
    payload0["__EVENTTARGET"] = "ctl00$ContentPlaceHolder1$CompanyDetail1$lnkAgmTab"
    payload0["__EVENTARGUMENT"] = ""
    resp1 = sess.post(url, data=payload0, timeout=max(timeout_s, 45), headers=headers)
    resp1.raise_for_status()

    soup1 = BeautifulSoup(resp1.text, "html.parser")
    payload1 = {i.get("name"): i.get("value", "") for i in soup1.select("input[type=hidden]") if i.get("name")}
    payload1["ctl00$ContentPlaceHolder1$CompanyDetail1$PagerControlAgm1$hdnCurrentPage"] = "1"
    payload1["ctl00$ContentPlaceHolder1$CompanyDetail1$PagerControlAgm1$hdnPCID"] = "PC1"
    payload1["__EVENTTARGET"] = "ctl00$ContentPlaceHolder1$CompanyDetail1$PagerControlAgm1$btnPaging"
    payload1["__EVENTARGUMENT"] = ""
    resp2 = sess.post(url, data=payload1, timeout=max(timeout_s, 60), headers=headers)
    resp2.raise_for_status()
    return resp2.text, url


def parse_corporate_actions_from_company_detail_html(
    html: str,
    *,
    symbol: str,
    source_url: str,
    scraped_at_utc: Optional[datetime] = None,
) -> List[CorporateActionRow]:
    """
    Parse the corporate actions (AGM table) from company detail page HTML.

    Returns:
        List of CorporateActionRow (newest-first if parsing preserves order).
    """
    scraped_at_utc = scraped_at_utc or datetime.utcnow()
    sym = str(symbol).strip().upper()
    soup = BeautifulSoup(html or "", "html.parser")

    target_headers = {
        "fiscal year",
        "bookclose date",
        "cash dividend",
        "bonus share",
        "right share",
    }

    best_table = None
    best_score = 0
    for tbl in soup.find_all("table"):
        th_texts = [_clean_cell(th.get_text(" ", strip=True)).lower() for th in tbl.find_all("th")]
        if not th_texts:
            continue
        score = sum(1 for h in target_headers if any(h in th for th in th_texts))
        if score > best_score:
            best_score = score
            best_table = tbl

    if not best_table or best_score < 3:
        return []

    # Map column indexes by header
    headers = [_clean_cell(th.get_text(" ", strip=True)).lower() for th in best_table.find_all("th")]
    col_idx: Dict[str, int] = {}
    for i, h in enumerate(headers):
        if "fiscal" in h and "year" in h:
            col_idx["fiscal_year"] = i
        elif "book" in h and "close" in h:
            col_idx["bookclose_date"] = i
        elif "description" in h:
            col_idx["description"] = i
        elif "agenda" in h:
            col_idx["agenda"] = i
        elif "cash" in h and "dividend" in h:
            col_idx["cash_dividend"] = i
        elif "bonus" in h:
            col_idx["bonus_share"] = i
        elif "right" in h:
            col_idx["right_share"] = i

    out: List[CorporateActionRow] = []
    for tr in best_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        def get(key: str) -> str:
            idx = col_idx.get(key)
            if idx is None or idx < 0 or idx >= len(tds):
                return ""
            return _clean_cell(tds[idx].get_text(" ", strip=True))

        description = get("description")
        agenda = get("agenda") or None
        fiscal_year = get("fiscal_year") or None
        bookclose = _parse_bookclose_date(get("bookclose_date"))
        cash = _parse_percent(get("cash_dividend"))
        bonus = _parse_percent(get("bonus_share"))
        rights = get("right_share") or None

        # Skip empty rows that sometimes appear from layout tables
        if not (description or agenda or cash or bonus or rights):
            continue

        out.append(
            CorporateActionRow(
                symbol=sym,
                fiscal_year=fiscal_year,
                bookclose_date_ad=bookclose,
                description=description,
                agenda=agenda,
                cash_dividend_pct=cash,
                bonus_share_pct=bonus,
                right_share_ratio=rights,
                source_url=source_url,
                scraped_at_utc=scraped_at_utc,
            )
        )

    return out


def ensure_corporate_actions_tables(conn: sqlite3.Connection) -> None:
    """
    Keep the DDL close to the ingestion logic for robustness.
    database_extended.py calls a similar helper; this is used by callers that
    don’t import the extended schema module.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS corporate_actions (
            symbol TEXT NOT NULL,
            fiscal_year TEXT,
            bookclose_date DATE,
            description TEXT,
            agenda TEXT,
            cash_dividend_pct REAL,
            bonus_share_pct REAL,
            right_share_ratio TEXT,
            source_url TEXT,
            scraped_at_utc TEXT NOT NULL,
            PRIMARY KEY (symbol, fiscal_year, bookclose_date, description)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol ON corporate_actions(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_corp_actions_bookclose ON corporate_actions(bookclose_date)")


def upsert_corporate_actions(rows: List[CorporateActionRow]) -> int:
    if not rows:
        return 0
    conn = get_db_connection()
    ensure_corporate_actions_tables(conn)
    cur = conn.cursor()
    inserted = 0
    for r in rows:
        cur.execute(
            """
            INSERT OR REPLACE INTO corporate_actions (
                symbol, fiscal_year, bookclose_date, description, agenda,
                cash_dividend_pct, bonus_share_pct, right_share_ratio, source_url, scraped_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.symbol,
                r.fiscal_year,
                r.bookclose_date_ad.isoformat() if r.bookclose_date_ad else None,
                r.description,
                r.agenda,
                r.cash_dividend_pct,
                r.bonus_share_pct,
                r.right_share_ratio,
                r.source_url,
                r.scraped_at_utc.isoformat(),
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return inserted


def load_latest_corporate_actions(symbol: str, *, limit: int = 12) -> List[Dict[str, Any]]:
    sym = str(symbol).strip().upper()
    conn = get_db_connection()
    ensure_corporate_actions_tables(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, fiscal_year, bookclose_date, description, agenda,
               cash_dividend_pct, bonus_share_pct, right_share_ratio, source_url, scraped_at_utc
        FROM corporate_actions
        WHERE symbol = ?
        ORDER BY COALESCE(bookclose_date, '') DESC, scraped_at_utc DESC
        LIMIT ?
        """,
        (sym, int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for row in rows:
        out.append(
            {
                "symbol": row[0],
                "fiscal_year": row[1],
                "bookclose_date": row[2],
                "description": row[3],
                "agenda": row[4],
                "cash_dividend_pct": row[5],
                "bonus_share_pct": row[6],
                "right_share_ratio": row[7],
                "source_url": row[8],
                "scraped_at_utc": row[9],
            }
        )
    return out


def refresh_corporate_actions(symbol: str) -> List[CorporateActionRow]:
    # Use AGM tab (contains bookclose + cash/bonus/rights + description/agenda).
    html, url = fetch_company_agm_tab_html(symbol)
    rows = parse_corporate_actions_from_company_detail_html(html, symbol=symbol, source_url=url)
    upsert_corporate_actions(rows)
    return rows


# === PRICE ADJUSTMENT FOR CORPORATE ACTIONS ===


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_adjustment_factor(
    bonus_share_pct: Optional[float] = None,
    right_share_ratio: Optional[str] = None,
    cash_dividend_pct: Optional[float] = None,
    prev_close: Optional[float] = None,
) -> float:
    """
    Compute the adjustment factor for a single corporate action.

    Parameters
    ----------
    bonus_share_pct : float, optional
        Bonus share percentage (e.g., 20 means 20% bonus = 1.2x shares).
    right_share_ratio : str, optional
        Right share ratio string (e.g., "1:5" means 1 right for every 5 held).
    cash_dividend_pct : float, optional
        Cash dividend as percentage of paid-up value (minimal price impact).
    prev_close : float, optional
        Previous close price for dividend adjustment.

    Returns
    -------
    float
        Adjustment factor (multiply old prices by this to get adjusted prices).
        Values < 1.0 indicate split/bonus (prices should be lower).
    """
    factor = 1.0

    # Bonus share adjustment
    # Bonus 20% = 120 shares for every 100 held = factor of 1/1.20
    if bonus_share_pct is not None and bonus_share_pct > 0:
        bonus_mult = 1.0 + (bonus_share_pct / 100.0)
        factor /= bonus_mult

    # Right share adjustment
    # "1:5" means 1 new share for every 5 held at par value (Rs 100)
    if right_share_ratio is not None:
        try:
            parts = right_share_ratio.replace(" ", "").split(":")
            if len(parts) == 2:
                new_shares = float(parts[0])
                old_shares = float(parts[1])
                if old_shares > 0:
                    # Simplified: assume rights are at par (Rs 100)
                    # True adjustment needs subscription price
                    right_mult = 1.0 + (new_shares / old_shares)
                    factor /= right_mult
        except (ValueError, ZeroDivisionError):
            pass

    # Cash dividend adjustment (usually minimal impact)
    # Only adjust if we have prev_close and dividend is significant
    if cash_dividend_pct is not None and prev_close is not None and prev_close > 0:
        # Paid-up value in Nepal is typically Rs 100
        paid_up_value = 100.0
        cash_per_share = paid_up_value * (cash_dividend_pct / 100.0)
        # Dividend impact as fraction of price
        div_impact = cash_per_share / prev_close
        if div_impact > 0.01:  # Only adjust if >1% impact
            factor *= (1.0 - div_impact * 0.5)  # Partial adjustment (market usually prices in)

    return factor


def adjust_prices_for_corporate_actions(
    prices: pd.DataFrame,
    symbol: str,
    corp_actions: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Retroactively adjust OHLCV for corporate actions.

    This applies cumulative adjustment factors backwards from each
    corporate action date. Volume is adjusted inversely (more shares
    after bonus = volume divided by factor).

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    symbol : str
        Stock symbol for loading corporate actions.
    corp_actions : list of dict, optional
        Pre-loaded corporate actions. If None, loads from database.

    Returns
    -------
    pd.DataFrame
        Adjusted price DataFrame.
    """
    if prices is None or prices.empty:
        return prices

    adjusted = prices.copy()

    # Load corporate actions if not provided
    if corp_actions is None:
        corp_actions = load_latest_corporate_actions(symbol, limit=50)

    if not corp_actions:
        return adjusted

    # Sort actions by bookclose date (oldest first for cumulative adjustment)
    actions_with_dates = [
        a for a in corp_actions
        if a.get("bookclose_date") is not None
    ]
    actions_with_dates.sort(key=lambda x: x["bookclose_date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in adjusted.columns]

    for action in actions_with_dates:
        try:
            bookclose = pd.to_datetime(action["bookclose_date"])
        except (ValueError, TypeError):
            continue

        # Get previous close for dividend adjustment
        prev_close = None
        if "Close" in adjusted.columns:
            pre_action = adjusted[adjusted.index < bookclose]
            if not pre_action.empty:
                prev_close = float(pre_action["Close"].iloc[-1])

        # Compute adjustment factor
        factor = compute_adjustment_factor(
            bonus_share_pct=action.get("bonus_share_pct"),
            right_share_ratio=action.get("right_share_ratio"),
            cash_dividend_pct=action.get("cash_dividend_pct"),
            prev_close=prev_close,
        )

        # Skip if no adjustment needed
        if abs(factor - 1.0) < 0.001:
            continue

        # Apply adjustment to all dates BEFORE bookclose
        mask = adjusted.index < bookclose

        if mask.any():
            for col in price_cols:
                adjusted.loc[mask, col] = adjusted.loc[mask, col] * factor

            # Adjust volume inversely (more shares = lower volume per share)
            if "Volume" in adjusted.columns:
                adjusted.loc[mask, "Volume"] = adjusted.loc[mask, "Volume"] / factor

            logger.debug(
                f"{symbol}: Applied adjustment factor {factor:.4f} for "
                f"bookclose {bookclose.date()} (bonus={action.get('bonus_share_pct')}, "
                f"rights={action.get('right_share_ratio')})"
            )

    return adjusted


def detect_unadjusted_gaps(
    prices: pd.DataFrame,
    threshold: float = 0.25,
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Detect potential corporate action gaps in price data.

    Returns dates where overnight return exceeds threshold,
    which may indicate missing corporate action adjustment.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLCV DataFrame.
    threshold : float
        Minimum absolute return to flag (default 25%).

    Returns
    -------
    list of (date, return)
        Dates and returns for potential unadjusted actions.
    """
    if prices is None or prices.empty or "Close" not in prices.columns:
        return []

    returns = prices["Close"].pct_change()
    gaps = returns[returns.abs() > threshold]

    return [(idx, float(ret)) for idx, ret in gaps.items()]


__all__ = [
    "CorporateActionRow",
    "fetch_company_detail_html",
    "fetch_company_agm_tab_html",
    "parse_corporate_actions_from_company_detail_html",
    "ensure_corporate_actions_tables",
    "upsert_corporate_actions",
    "load_latest_corporate_actions",
    "refresh_corporate_actions",
    "compute_adjustment_factor",
    "adjust_prices_for_corporate_actions",
    "detect_unadjusted_gaps",
]
