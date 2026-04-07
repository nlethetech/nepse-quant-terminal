"""
Quarterly earnings scraper for NEPSE stocks.

Data sources:
1. ShareSansar (www.sharesansar.com) - Full quarterly financial data via AJAX endpoint
   - Balance sheet, P&L, key metrics (EPS, book value, ROE, etc.)
   - Returns the LATEST quarter only per company
2. MeroLagani (merolagani.com) - Quarterly report announcement dates
   - Paginated list of quarterly report publication dates with fiscal year + quarter
   - Company detail pages show current EPS with FY/Q designation

Strategy:
- Use ShareSansar's /company-quarterly-report POST endpoint for financial data
- Use MeroLagani's CompanyReports.aspx?type=QUARTERLY for announcement dates
- Combine into the quarterly_earnings table

Rate limiting: 1 request/sec (configurable).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, RequestException, Timeout
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiter (shared, matches vendor_api.py pattern)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter (1 req/sec default)."""

    def __init__(self, rate: float = 1.0):
        self._min_interval = 1.0 / rate
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                _time.sleep(self._min_interval - elapsed)
            self._last_call = _time.monotonic()


_rate_limiter = _RateLimiter(rate=1.0)  # 1 req/sec

# ---------------------------------------------------------------------------
# Default user-agent and session factory
# ---------------------------------------------------------------------------

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    return s


# ---------------------------------------------------------------------------
# Target symbols (top NEPSE stocks by market cap / liquidity)
# ---------------------------------------------------------------------------

# Banking (16), Hydro (6), Insurance (4), Others (4) = 30 target stocks
TARGET_SYMBOLS: List[str] = [
    # Commercial Banks
    "NABIL", "SCB", "SBI", "GBIME", "EBL", "NICA", "PCBL", "MBL",
    "KBL", "NBL", "ADBL", "CZBIL", "SANIMA", "NMB", "PRVU", "NIMB",
    # Hydropower
    "NHPC", "UPPER", "AHPC", "API", "CHL", "BARUN",
    # Insurance
    "NLIC", "LICN", "ALICL", "NIL",
    # Others
    "NTC", "SHIVM", "UNL", "CHCL",
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Resolve the database file path."""
    import os
    raw = os.environ.get("NEPSE_DB_FILE", "nepse_market_data.db")
    return str(Path(raw).resolve())


def create_quarterly_earnings_table(db_path: Optional[str] = None) -> None:
    """Create the quarterly_earnings table if it does not exist."""
    db_path = db_path or _get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quarterly_earnings (
            symbol TEXT NOT NULL,
            fiscal_year TEXT NOT NULL,
            quarter INTEGER NOT NULL,
            eps REAL,
            net_profit REAL,
            revenue REAL,
            book_value REAL,
            announcement_date TEXT,
            report_date TEXT,
            source TEXT DEFAULT 'sharesansar',
            scraped_at_utc TEXT NOT NULL,
            PRIMARY KEY (symbol, fiscal_year, quarter)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qe_symbol
        ON quarterly_earnings (symbol)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qe_announcement
        ON quarterly_earnings (announcement_date)
    """)
    conn.commit()
    conn.close()
    logger.info("quarterly_earnings table ready at %s", db_path)


def create_fundamentals_table(db_path: Optional[str] = None) -> None:
    """Create the fundamentals snapshot table if it does not exist."""
    db_path = db_path or _get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol TEXT,
            date DATE,
            market_cap REAL,
            pe_ratio REAL,
            pb_ratio REAL,
            eps REAL,
            book_value_per_share REAL,
            roe REAL,
            debt_to_equity REAL,
            dividend_yield REAL,
            payout_ratio REAL,
            current_ratio REAL,
            shares_outstanding REAL,
            sector TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol
        ON fundamentals (symbol, date DESC)
    """)
    conn.commit()
    conn.close()
    logger.info("fundamentals table ready at %s", db_path)


def upsert_quarterly_earnings(
    db_path: str,
    rows: List[Dict[str, Any]],
) -> int:
    """Insert or update quarterly earnings rows. Returns count inserted."""
    if not rows:
        return 0
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    count = 0
    for row in rows:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO quarterly_earnings
                    (symbol, fiscal_year, quarter, eps, net_profit, revenue,
                     book_value, announcement_date, report_date, source, scraped_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["symbol"],
                row["fiscal_year"],
                row["quarter"],
                row.get("eps"),
                row.get("net_profit"),
                row.get("revenue"),
                row.get("book_value"),
                row.get("announcement_date"),
                row.get("report_date"),
                row.get("source", "sharesansar"),
                row.get("scraped_at_utc", datetime.now(timezone.utc).isoformat()),
            ))
            count += 1
        except sqlite3.Error as e:
            logger.warning("Failed to upsert %s FY%s Q%s: %s",
                           row.get("symbol"), row.get("fiscal_year"),
                           row.get("quarter"), e)
    conn.commit()
    conn.close()
    return count


def upsert_fundamentals_snapshots(
    db_path: str,
    rows: List[Dict[str, Any]],
    *,
    as_of_date: Optional[str] = None,
) -> int:
    """Persist latest per-symbol valuation metrics into fundamentals."""
    if not rows:
        return 0

    snapshot_date = as_of_date or datetime.now(timezone.utc).date().isoformat()
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        snapshot = by_symbol.setdefault(
            symbol,
            {
                "symbol": symbol,
                "date": snapshot_date,
                "market_cap": None,
                "pe_ratio": None,
                "pb_ratio": None,
                "eps": None,
                "book_value_per_share": None,
                "roe": None,
                "debt_to_equity": None,
                "dividend_yield": None,
                "sector": None,
                "shares_outstanding": None,
            },
        )
        for source_key, target_key in [
            ("market_cap", "market_cap"),
            ("pe_ratio", "pe_ratio"),
            ("pb_ratio", "pb_ratio"),
            ("eps", "eps"),
            ("book_value", "book_value_per_share"),
            ("roe", "roe"),
            ("debt_to_equity", "debt_to_equity"),
            ("dividend_yield", "dividend_yield"),
            ("sector", "sector"),
            ("shares_outstanding", "shares_outstanding"),
        ]:
            value = row.get(source_key)
            if value not in (None, ""):
                snapshot[target_key] = value

    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    count = 0
    for snapshot in by_symbol.values():
        has_metrics = any(
            snapshot.get(key) not in (None, "")
            for key in (
                "market_cap",
                "pe_ratio",
                "pb_ratio",
                "eps",
                "book_value_per_share",
                "roe",
                "debt_to_equity",
                "dividend_yield",
                "shares_outstanding",
                "sector",
            )
        )
        if not has_metrics:
            continue
        try:
            cursor.execute(
                """
                INSERT INTO fundamentals (
                    symbol, date, market_cap, pe_ratio, pb_ratio, eps,
                    book_value_per_share, roe, debt_to_equity, dividend_yield,
                    shares_outstanding, sector
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    market_cap = COALESCE(excluded.market_cap, fundamentals.market_cap),
                    pe_ratio = COALESCE(excluded.pe_ratio, fundamentals.pe_ratio),
                    pb_ratio = COALESCE(excluded.pb_ratio, fundamentals.pb_ratio),
                    eps = COALESCE(excluded.eps, fundamentals.eps),
                    book_value_per_share = COALESCE(excluded.book_value_per_share, fundamentals.book_value_per_share),
                    roe = COALESCE(excluded.roe, fundamentals.roe),
                    debt_to_equity = COALESCE(excluded.debt_to_equity, fundamentals.debt_to_equity),
                    dividend_yield = COALESCE(excluded.dividend_yield, fundamentals.dividend_yield),
                    shares_outstanding = COALESCE(excluded.shares_outstanding, fundamentals.shares_outstanding),
                    sector = COALESCE(excluded.sector, fundamentals.sector)
                """,
                (
                    snapshot["symbol"],
                    snapshot["date"],
                    snapshot.get("market_cap"),
                    snapshot.get("pe_ratio"),
                    snapshot.get("pb_ratio"),
                    snapshot.get("eps"),
                    snapshot.get("book_value_per_share"),
                    snapshot.get("roe"),
                    snapshot.get("debt_to_equity"),
                    snapshot.get("dividend_yield"),
                    snapshot.get("shares_outstanding"),
                    snapshot.get("sector"),
                ),
            )
            count += 1
        except sqlite3.Error as e:
            logger.warning("Failed to upsert fundamentals snapshot for %s: %s", snapshot["symbol"], e)
    conn.commit()
    conn.close()
    return count


# ---------------------------------------------------------------------------
# ShareSansar: Company ID resolver
# ---------------------------------------------------------------------------

_company_id_cache: Dict[str, Dict[str, Any]] = {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((RequestException, Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _load_sharesansar_company_map(session: requests.Session) -> Dict[str, Dict[str, Any]]:
    """
    Load the full symbol -> {id, name} mapping from ShareSansar.
    The mapping is embedded as a JS variable `cmpjson` in any company page.
    """
    global _company_id_cache
    if _company_id_cache:
        return _company_id_cache

    _rate_limiter.wait()
    resp = session.get(
        "https://www.sharesansar.com/company/nabil",
        timeout=20,
    )
    resp.raise_for_status()

    match = re.search(r"var\s+cmpjson\s*=\s*(\[.*?\]);", resp.text, re.S)
    if not match:
        raise ValueError("Could not find cmpjson in ShareSansar page")

    cmpjson = json.loads(match.group(1))
    for c in cmpjson:
        _company_id_cache[c["symbol"].upper()] = {
            "id": str(c["id"]),
            "name": c.get("companyname", ""),
        }
    logger.info("Loaded %d company IDs from ShareSansar", len(_company_id_cache))
    return _company_id_cache


def _get_sharesansar_company_info(
    session: requests.Session, symbol: str
) -> Optional[Dict[str, str]]:
    """
    Get company ID and sector from ShareSansar for a given symbol.
    Returns dict with 'id', 'symbol', 'sector' or None.
    """
    _rate_limiter.wait()
    url = f"https://www.sharesansar.com/company/{symbol.lower()}"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except RequestException as e:
        logger.warning("Failed to fetch ShareSansar page for %s: %s", symbol, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    company_id_el = soup.find("div", id="companyid")
    symbol_el = soup.find("div", id="symbol")
    sector_el = soup.find("div", id="sector")

    if not company_id_el or not symbol_el:
        logger.warning("Could not find company ID for %s on ShareSansar", symbol)
        return None

    csrf_meta = soup.find("meta", {"name": "_token"})
    csrf_token = csrf_meta["content"] if csrf_meta else ""

    return {
        "id": company_id_el.get_text(strip=True),
        "symbol": symbol_el.get_text(strip=True),
        "sector": sector_el.get_text(strip=True) if sector_el else "",
        "csrf_token": csrf_token,
    }


# ---------------------------------------------------------------------------
# ShareSansar: Quarterly financial data scraper
# ---------------------------------------------------------------------------

def _parse_number(text: str) -> Optional[float]:
    """Parse a number string, handling commas and parentheses (negative)."""
    if not text:
        return None
    text = text.strip().replace(",", "")
    # Handle parenthesized negatives: (123.45) -> -123.45
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def normalize_fiscal_year(fy: str) -> str:
    """
    Normalize NEPSE fiscal year to standard format 'XXX-XXX' (short BS year).

    Examples:
        '2082/2083' -> '082-083'
        '082-083'   -> '082-083'
        '2081/82'   -> '081-082'
        '081-082'   -> '081-082'
        'FY:082-083' -> '082-083'
    """
    if not fy:
        return fy

    # Remove 'FY:' prefix if present
    fy = re.sub(r"^FY[:\s]*", "", fy.strip())

    # Pattern: 4-digit/4-digit (e.g. 2082/2083)
    m = re.match(r"(\d{4})[/-](\d{4})", fy)
    if m:
        return f"{m.group(1)[-3:]}-{m.group(2)[-3:]}"

    # Pattern: 4-digit/2-digit (e.g. 2081/82)
    m = re.match(r"(\d{4})[/-](\d{2})", fy)
    if m:
        return f"{m.group(1)[-3:]}-0{m.group(2)}" if len(m.group(2)) == 2 else f"{m.group(1)[-3:]}-{m.group(2)}"

    # Pattern: 3-digit-3-digit (e.g. 082-083) -- already normalized
    m = re.match(r"(\d{3})[/-](\d{3})", fy)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Pattern: 3-digit-2-digit (e.g. 081-82)
    m = re.match(r"(\d{3})[/-](\d{2})", fy)
    if m:
        return f"{m.group(1)}-0{m.group(2)}"

    return fy  # Return as-is if no pattern matched


def _parse_quarter_header(header_text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse quarter and fiscal year from header like '2nd Quarter 2082/2083'.
    Returns (quarter_num, normalized_fiscal_year) or (None, None).
    """
    header_text = header_text.strip()

    # Match patterns: '1st Quarter 2082/2083', '2nd Quarter 2082/2083', etc.
    q_match = re.search(r"(\d+)(?:st|nd|rd|th)\s+Quarter\s+(\d{4}/\d{4})", header_text, re.I)
    if q_match:
        return int(q_match.group(1)), normalize_fiscal_year(q_match.group(2))

    # Also try: 'Q1 2082/2083' pattern
    q_match2 = re.search(r"Q(\d)\s+(\d{4}/\d{4})", header_text, re.I)
    if q_match2:
        return int(q_match2.group(1)), normalize_fiscal_year(q_match2.group(2))

    return None, None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((RequestException, Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def scrape_sharesansar_quarterly(
    session: requests.Session,
    company_info: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Scrape quarterly financial data for a company from ShareSansar.

    Parameters
    ----------
    session : requests.Session
    company_info : dict with 'id', 'symbol', 'sector', 'csrf_token'

    Returns
    -------
    dict with parsed financial data or None on failure.
    Keys: symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
          roe, pe_ratio, net_worth_per_share, total_equity, total_assets
    """
    _rate_limiter.wait()

    ajax_headers = {
        "X-CSRF-Token": company_info.get("csrf_token", ""),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://www.sharesansar.com/company/{company_info['symbol'].lower()}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = session.post(
        "https://www.sharesansar.com/company-quarterly-report",
        data={
            "company": company_info["id"],
            "symbol": company_info["symbol"],
            "sector": company_info.get("sector", ""),
        },
        headers=ajax_headers,
        timeout=15,
    )
    resp.raise_for_status()

    if len(resp.text) < 100:
        logger.info("No quarterly data for %s (response too short)", company_info["symbol"])
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        logger.info("No tables in quarterly response for %s", company_info["symbol"])
        return None

    result: Dict[str, Any] = {
        "symbol": company_info["symbol"].upper(),
        "source": "sharesansar",
        "sector": str(company_info.get("sector", "")).strip(),
    }

    # Parse quarter/fiscal year from any table header
    for table in tables:
        ths = table.find_all("th")
        for th in ths:
            text = th.get_text(strip=True)
            q, fy = _parse_quarter_header(text)
            if q is not None and fy is not None:
                result["quarter"] = q
                result["fiscal_year"] = fy
                break
        if "quarter" in result:
            break

    if "quarter" not in result:
        logger.warning("Could not parse quarter/FY from tables for %s", company_info["symbol"])
        return None

    # Parse financial data from all tables
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower()
            value_text = cells[1].get_text(strip=True)
            value = _parse_number(value_text)

            # Key metrics (match various label patterns)
            if "basic earnings per share" in label or "eps annualized" in label:
                result["eps"] = value
            elif label.startswith("diluted earnings per share"):
                if "eps" not in result:
                    result["eps"] = value
            elif "net worth per share" in label or "networth per share" in label:
                result["book_value"] = value
            elif "return on equity" in label:
                result["roe"] = value
            elif "return on assets" in label:
                result["roa"] = value
            elif "p/e ratio" in label or "pe ratio" in label:
                result["pe_ratio"] = value
            elif "price to book" in label or "pb ratio" in label:
                result["pb_ratio"] = value
            elif "total equity" in label:
                result["total_equity"] = value
            elif "total assets" in label:
                result["total_assets"] = value

            # P&L items (Rs in '000)
            # Note: label matching order matters - check specific patterns first
            if "profit for the period" in label and "comprehensive" not in label:
                if "net_profit" not in result:
                    result["net_profit"] = value
            elif label == "net profit" or label.startswith("net profit"):
                if "net_profit" not in result:
                    result["net_profit"] = value
            elif "total operating income" in label:
                # Best revenue proxy (works for all sectors)
                result["revenue"] = value  # Always overwrite with total operating income
            elif "operating income" in label and "total" not in label and "non" not in label and "net" not in label:
                # "Operating Income" for hydro companies (not "Non-Operating" or "Net Operating")
                if "revenue" not in result:
                    result["revenue"] = value
            elif "net interest income" in label:
                # For banks, use net interest income as primary revenue
                result["net_interest_income"] = value
            elif "income from sales" in label or "sales revenue" in label:
                # For non-banks (hydro, manufacturing, etc.)
                if "revenue" not in result:
                    result["revenue"] = value
            elif "interest income" in label and "net" not in label and "expense" not in label:
                # Gross interest income for banks
                if "gross_interest_income" not in result:
                    result["gross_interest_income"] = value

    # Fallback: use net interest income or gross interest income as revenue
    if "revenue" not in result:
        result["revenue"] = result.get("net_interest_income") or result.get("gross_interest_income")

    logger.info(
        "Scraped %s FY%s Q%d: EPS=%.2f, Profit=%s, BV=%s",
        result["symbol"],
        result.get("fiscal_year", "?"),
        result.get("quarter", 0),
        result.get("eps") or 0,
        result.get("net_profit"),
        result.get("book_value"),
    )

    return result


def _sharesansar_datatable_request(
    session: requests.Session,
    company_info: Dict[str, str],
    *,
    endpoint: str,
    extra_data: Optional[Dict[str, Any]] = None,
    start: int = 0,
    length: int = 50,
    draw: int = 1,
) -> Dict[str, Any]:
    """Replay Sharesansar DataTables AJAX requests used on company pages."""
    _rate_limiter.wait()
    payload = {
        "draw": str(draw),
        "start": str(start),
        "length": str(length),
        "search[value]": "",
        "search[regex]": "false",
        "company": company_info["id"],
        "columns[0][data]": "published_date",
        "columns[0][name]": "",
        "columns[0][searchable]": "true",
        "columns[0][orderable]": "false",
        "columns[0][search][value]": "",
        "columns[0][search][regex]": "false",
        "columns[1][data]": "title",
        "columns[1][name]": "",
        "columns[1][searchable]": "true",
        "columns[1][orderable]": "false",
        "columns[1][search][value]": "",
        "columns[1][search][regex]": "false",
    }
    if extra_data:
        for key, value in extra_data.items():
            payload[str(key)] = value

    resp = session.post(
        endpoint,
        data=payload,
        headers={
            "X-CSRF-Token": company_info.get("csrf_token", ""),
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://www.sharesansar.com/company/{company_info['symbol'].lower()}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_sharesansar_financial_report_item(
    item: Dict[str, Any],
    *,
    symbol: str,
    scraped_at_utc: str,
) -> Optional[Dict[str, Any]]:
    """Parse Sharesansar Financial Reports table rows into quarterly earnings records."""
    title_html = str(item.get("title") or "")
    title_text = BeautifulSoup(title_html, "html.parser").get_text(" ", strip=True)
    title_text = re.sub(r"\s+", " ", title_text).strip()
    if not title_text:
        return None

    quarter_match = re.search(r"(\d)(?:st|nd|rd|th)\s+quarter", title_text, re.I)
    fy_match = re.search(r"fiscal year\s+(\d{4}/\d{2,4})", title_text, re.I)
    if not quarter_match or not fy_match:
        return None

    net_profit = None
    pnl_match = re.search(
        r"net\s+(profit|loss)\s+of\s+Rs\s+([0-9][0-9,]*(?:\.\d+)?)\s+million",
        title_text,
        re.I,
    )
    if pnl_match:
        amount = _parse_number(pnl_match.group(2))
        if amount is not None:
            net_profit = amount * 1_000_000.0
            if pnl_match.group(1).lower() == "loss":
                net_profit *= -1.0

    published_date = str(item.get("published_date") or "").strip()
    return {
        "symbol": symbol.upper(),
        "fiscal_year": normalize_fiscal_year(fy_match.group(1)),
        "quarter": int(quarter_match.group(1)),
        "net_profit": net_profit,
        "announcement_date": published_date or None,
        "report_date": published_date or None,
        "source": "sharesansar_financial_reports",
        "scraped_at_utc": scraped_at_utc,
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((RequestException, Timeout, HTTPError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def scrape_sharesansar_financial_report_history(
    session: requests.Session,
    company_info: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Scrape quarterly profit/loss history from ShareSansar Financial Reports."""
    rows: List[Dict[str, Any]] = []
    start = 0
    draw = 1
    length = 50
    total_records: Optional[int] = None
    now_utc = datetime.now(timezone.utc).isoformat()

    while total_records is None or start < total_records:
        payload = _sharesansar_datatable_request(
            session,
            company_info,
            endpoint="https://www.sharesansar.com/company-announcement-category",
            extra_data={"category": "11"},
            start=start,
            length=length,
            draw=draw,
        )
        if total_records is None:
            total_records = int(payload.get("recordsTotal") or 0)
        data_rows = list(payload.get("data") or [])
        if not data_rows:
            break
        for item in data_rows:
            parsed = _parse_sharesansar_financial_report_item(
                item,
                symbol=company_info["symbol"],
                scraped_at_utc=now_utc,
            )
            if parsed:
                rows.append(parsed)
        start += length
        draw += 1
        if len(data_rows) < length:
            break

    logger.info(
        "Scraped %d Sharesansar financial-report history rows for %s",
        len(rows),
        company_info["symbol"],
    )
    return rows


# ---------------------------------------------------------------------------
# MeroLagani: EPS from company detail page
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((RequestException, Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def scrape_merolagani_eps(
    session: requests.Session, symbol: str
) -> Optional[Dict[str, Any]]:
    """
    Scrape current EPS and key metrics from MeroLagani company detail page.

    The page shows EPS like: '35.18(FY:082-083, Q:2)'
    Also shows: Book Value, P/E Ratio, PBV

    Returns dict with: eps, fiscal_year, quarter, book_value, pe_ratio, etc.
    """
    _rate_limiter.wait()
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    resp = session.get(url, headers={"Referer": "https://merolagani.com/"}, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    result: Dict[str, Any] = {"symbol": symbol.upper(), "source": "merolagani"}

    # Parse tables for financial metrics
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower()
            value_text = cells[1].get_text(strip=True)

            if label == "eps":
                # Parse EPS value and FY/Q: '35.18(FY:082-083, Q:2)'
                eps_match = re.match(r"([\d,.]+)\s*\(FY:(\S+),\s*Q:(\d+)\)", value_text)
                if eps_match:
                    result["eps"] = _parse_number(eps_match.group(1))
                    result["fiscal_year"] = normalize_fiscal_year(eps_match.group(2))
                    result["quarter"] = int(eps_match.group(3))
                else:
                    result["eps"] = _parse_number(value_text.split("(")[0])
            elif label == "book value":
                result["book_value"] = _parse_number(value_text)
            elif label == "p/e ratio":
                result["pe_ratio"] = _parse_number(value_text)
            elif label == "pbv":
                result["pb_ratio"] = _parse_number(value_text)
            elif label == "market price":
                result["market_price"] = _parse_number(value_text)
            elif label == "market capitalization":
                result["market_cap"] = _parse_number(value_text)
            elif label in {"shares outstanding", "listed shares"}:
                result["shares_outstanding"] = _parse_number(value_text)
            elif label == "sector":
                result["sector"] = value_text.strip()
            elif label == "% dividend":
                result["dividend_yield"] = _parse_number(value_text)

    if "eps" not in result:
        logger.warning("No EPS found on MeroLagani for %s", symbol)
        return None

    return result


# ---------------------------------------------------------------------------
# MeroLagani: Quarterly announcement dates
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((RequestException, Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def scrape_merolagani_announcement_dates(
    session: requests.Session,
    symbol: str,
) -> List[Dict[str, Any]]:
    """
    Scrape quarterly report announcement dates for a symbol from MeroLagani.

    Uses the company detail page -> quarterly tab postback.
    Parses the announcement list to extract fiscal year, quarter, and date.

    Returns list of dicts: [{symbol, fiscal_year, quarter, announcement_date}, ...]
    """
    _rate_limiter.wait()
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    resp = session.get(url, headers={"Referer": "https://merolagani.com/"}, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract ASP.NET form fields
    viewstate = soup.find("input", {"name": "__VIEWSTATE"})
    viewstate_gen = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})
    event_validation = soup.find("input", {"name": "__EVENTVALIDATION"})

    if not viewstate or not event_validation:
        logger.warning("Missing ASP.NET form fields for %s", symbol)
        return []

    # Trigger quarterly tab postback
    _rate_limiter.wait()
    form_data = {
        "__VIEWSTATE": viewstate["value"],
        "__VIEWSTATEGENERATOR": viewstate_gen["value"] if viewstate_gen else "",
        "__EVENTVALIDATION": event_validation["value"],
        "__EVENTTARGET": "ctl00$ContentPlaceHolder1$CompanyDetail1$btnQuaterlyTab",
        "__EVENTARGUMENT": "",
    }

    resp2 = session.post(
        url,
        data=form_data,
        headers={
            "Referer": url,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=20,
    )
    resp2.raise_for_status()

    soup2 = BeautifulSoup(resp2.text, "html.parser")

    # Parse quarterly announcements from the quarterly data div
    # The div contains a table with: #, Fiscal Year, Date, Description
    # The description mentions the company name and quarter
    quarter_map = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
    }

    announcements: List[Dict[str, Any]] = []

    for row in soup2.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        desc = cells[3].get_text(strip=True) if len(cells) > 3 else cells[-1].get_text(strip=True)

        # Only match THIS company's quarterly announcements
        if symbol.lower() not in desc.lower() and "provisional financial statement" not in desc.lower():
            continue

        # Check the description mentions this symbol's company
        # MeroLagani uses full company names in descriptions
        if "financial statement" not in desc.lower():
            continue

        fy_text = cells[1].get_text(strip=True)
        date_text = cells[2].get_text(strip=True)

        # Parse quarter from description
        q_num = None
        for q_word, q_val in quarter_map.items():
            if q_word in desc.lower():
                q_num = q_val
                break

        if q_num is None:
            continue

        # Parse AD date: '2026/01/29 AD(2082/10/15 BS)'
        ad_match = re.search(r"(\d{4}/\d{2}/\d{2})\s*AD", date_text)
        if ad_match:
            ad_date = ad_match.group(1).replace("/", "-")
        else:
            # Try plain date
            date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", date_text)
            ad_date = date_match.group(1).replace("/", "-") if date_match else None

        if ad_date:
            announcements.append({
                "symbol": symbol.upper(),
                "fiscal_year": fy_text,
                "quarter": q_num,
                "announcement_date": ad_date,
            })

    logger.info("Found %d announcement dates for %s", len(announcements), symbol)
    return announcements


# ---------------------------------------------------------------------------
# Combined scraper: scrape all sources for a symbol
# ---------------------------------------------------------------------------

def scrape_symbol_earnings(
    symbol: str,
    session: Optional[requests.Session] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape all available quarterly earnings data for a symbol.

    Combines:
    1. ShareSansar: latest quarter with full financial data
    2. MeroLagani: current EPS with FY/Q info
    3. MeroLagani: historical announcement dates

    Returns list of quarterly earnings records ready for DB insertion.
    """
    session = session or _new_session()
    results: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc).isoformat()
    company_info: Optional[Dict[str, str]] = None

    # --- 1. ShareSansar: latest quarter full data ---
    try:
        company_info = _get_sharesansar_company_info(session, symbol)
        if company_info:
            ss_data = scrape_sharesansar_quarterly(session, company_info)
            if ss_data and "quarter" in ss_data and "fiscal_year" in ss_data:
                results.append({
                    "symbol": symbol.upper(),
                    "fiscal_year": ss_data["fiscal_year"],
                    "quarter": ss_data["quarter"],
                    "eps": ss_data.get("eps"),
                    "net_profit": ss_data.get("net_profit"),
                    "revenue": ss_data.get("revenue"),
                    "book_value": ss_data.get("book_value"),
                    "pe_ratio": ss_data.get("pe_ratio"),
                    "pb_ratio": ss_data.get("pb_ratio"),
                    "roe": ss_data.get("roe"),
                    "sector": ss_data.get("sector"),
                    "source": "sharesansar",
                    "scraped_at_utc": now_utc,
                })
    except Exception as e:
        logger.warning("ShareSansar scrape failed for %s: %s", symbol, e)

    # --- 1b. ShareSansar: historical financial-report titles for net profit/loss ---
    try:
        if company_info:
            history_rows = scrape_sharesansar_financial_report_history(session, company_info)
            existing_by_quarter = {
                (normalize_fiscal_year(str(row["fiscal_year"])), int(row["quarter"])): row
                for row in results
            }
            for history_row in history_rows:
                key = (
                    normalize_fiscal_year(str(history_row["fiscal_year"])),
                    int(history_row["quarter"]),
                )
                existing = existing_by_quarter.get(key)
                if existing is None:
                    results.append(history_row)
                    existing_by_quarter[key] = history_row
                    continue
                if history_row.get("net_profit") is not None:
                    existing["net_profit"] = history_row.get("net_profit")
                for date_key in ("announcement_date", "report_date"):
                    incoming = history_row.get(date_key)
                    current = existing.get(date_key)
                    if incoming and (not current or str(incoming) > str(current)):
                        existing[date_key] = incoming
                if not existing.get("source") or existing.get("source") == "merolagani":
                    existing["source"] = history_row.get("source", existing.get("source"))
    except Exception as e:
        logger.warning("ShareSansar financial-report history scrape failed for %s: %s", symbol, e)

    # --- 2. MeroLagani: current EPS (may differ from ShareSansar quarter) ---
    try:
        ml_data = scrape_merolagani_eps(session, symbol)
        if ml_data and "eps" in ml_data and "fiscal_year" in ml_data and "quarter" in ml_data:
            # Check if we already have this quarter from ShareSansar
            # Normalize both fiscal years for comparison
            ml_fy_norm = normalize_fiscal_year(ml_data["fiscal_year"])
            existing_key = None
            for r in results:
                r_fy_norm = normalize_fiscal_year(r["fiscal_year"])
                if r_fy_norm == ml_fy_norm and r["quarter"] == ml_data["quarter"]:
                    existing_key = r
                    break

            if existing_key:
                # Merge: prefer ShareSansar for detailed financials, use ML for cross-check
                if existing_key.get("eps") is None:
                    existing_key["eps"] = ml_data.get("eps")
                if existing_key.get("book_value") is None:
                    existing_key["book_value"] = ml_data.get("book_value")
                for key in ("pe_ratio", "pb_ratio", "market_price", "market_cap", "shares_outstanding", "dividend_yield"):
                    if ml_data.get(key) is not None:
                        existing_key[key] = ml_data.get(key)
                if not existing_key.get("sector") and ml_data.get("sector"):
                    existing_key["sector"] = ml_data.get("sector")
            else:
                # New quarter from MeroLagani
                results.append({
                    "symbol": symbol.upper(),
                    "fiscal_year": ml_data["fiscal_year"],
                    "quarter": ml_data["quarter"],
                    "eps": ml_data.get("eps"),
                    "net_profit": None,
                    "revenue": None,
                    "book_value": ml_data.get("book_value"),
                    "pe_ratio": ml_data.get("pe_ratio"),
                    "pb_ratio": ml_data.get("pb_ratio"),
                    "market_price": ml_data.get("market_price"),
                    "market_cap": ml_data.get("market_cap"),
                    "shares_outstanding": ml_data.get("shares_outstanding"),
                    "dividend_yield": ml_data.get("dividend_yield"),
                    "sector": ml_data.get("sector"),
                    "source": "merolagani",
                    "scraped_at_utc": now_utc,
                })
    except Exception as e:
        logger.warning("MeroLagani EPS scrape failed for %s: %s", symbol, e)

    # --- 3. MeroLagani: announcement dates (update existing records) ---
    # This is expensive (2 requests per symbol), so only do it if we have results
    # The announcement dates are critical for preventing lookahead bias
    # Note: The quarterly tab shows ALL companies' announcements, not just this one
    # So we only get dates for the current company via description matching

    return results


# ---------------------------------------------------------------------------
# Bulk scraper: scrape multiple symbols
# ---------------------------------------------------------------------------

def scrape_all_earnings(
    symbols: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Scrape quarterly earnings for multiple symbols and store in DB.

    Parameters
    ----------
    symbols : list of str, optional
        Symbols to scrape. Defaults to TARGET_SYMBOLS.
    db_path : str, optional
        Database path. Defaults to nepse_market_data.db.
    skip_existing : bool
        If True, skip symbols that already have data for the latest quarter.

    Returns
    -------
    dict with 'total_scraped', 'total_rows', 'errors'.
    """
    symbols = symbols or TARGET_SYMBOLS
    db_path = db_path or _get_db_path()

    # Ensure tables exist
    create_quarterly_earnings_table(db_path)
    create_fundamentals_table(db_path)

    session = _new_session()
    stats = {"total_scraped": 0, "total_rows": 0, "errors": 0, "skipped": 0}

    # Load company ID map from ShareSansar (one request)
    try:
        _load_sharesansar_company_map(session)
    except Exception as e:
        logger.warning("Failed to load ShareSansar company map: %s", e)

    for i, symbol in enumerate(symbols):
        logger.info("[%d/%d] Scraping %s...", i + 1, len(symbols), symbol)

        try:
            rows = scrape_symbol_earnings(symbol, session=session, db_path=db_path)
            if rows:
                inserted = upsert_quarterly_earnings(db_path, rows)
                upsert_fundamentals_snapshots(db_path, rows)
                stats["total_rows"] += inserted
                stats["total_scraped"] += 1
                logger.info("  -> %d rows for %s", inserted, symbol)
            else:
                logger.info("  -> No data for %s", symbol)
        except Exception as e:
            logger.error("  -> Error for %s: %s", symbol, e)
            stats["errors"] += 1

    logger.info(
        "Scraping complete: %d symbols, %d rows, %d errors",
        stats["total_scraped"], stats["total_rows"], stats["errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# MeroLagani: Announcement date scraper (bulk)
# ---------------------------------------------------------------------------

def scrape_announcement_dates_bulk(
    db_path: Optional[str] = None,
    max_pages: int = 10,
) -> int:
    """
    Scrape quarterly report announcement dates from MeroLagani's
    CompanyReports.aspx?type=QUARTERLY page.

    This page lists ALL companies' quarterly announcements (paginated).
    We parse each announcement to extract: symbol (from company name),
    fiscal_year, quarter, and announcement_date.

    Then we UPDATE existing quarterly_earnings rows with the announcement_date.

    Returns count of updated rows.
    """
    db_path = db_path or _get_db_path()
    session = _new_session()

    all_announcements: List[Dict[str, Any]] = []

    # Fetch the reports page
    _rate_limiter.wait()
    try:
        resp = session.get(
            "https://merolagani.com/CompanyReports.aspx?type=QUARTERLY",
            headers={"Referer": "https://merolagani.com/"},
            timeout=20,
        )
        resp.raise_for_status()
    except RequestException as e:
        logger.error("Failed to fetch quarterly reports page: %s", e)
        return 0

    soup = BeautifulSoup(resp.text, "html.parser")

    # Parse announcement links
    quarter_map = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
    }

    # Build symbol lookup from company names
    # We need to match company names like "Nabil Bank Limited" to symbols
    # Use the ShareSansar company map for name-to-symbol resolution
    try:
        cmp_map = _load_sharesansar_company_map(session)
        name_to_symbol = {}
        for sym, info in cmp_map.items():
            name = info["name"].lower().strip()
            name_to_symbol[name] = sym
    except Exception:
        name_to_symbol = {}

    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        href = link.get("href", "")

        if "AnnouncementDetail" not in href:
            continue
        if "financial statement" not in text.lower():
            continue

        # Parse quarter
        q_num = None
        for q_word, q_val in quarter_map.items():
            if q_word in text.lower():
                q_num = q_val
                break
        if q_num is None:
            continue

        # Parse fiscal year from text
        fy_match = re.search(r"fiscal\s+year\s+(\d{4}/\d{2,4})", text, re.I)
        if not fy_match:
            continue
        fiscal_year = fy_match.group(1)

        # Try to extract symbol from company name in the text
        # Text format: "Company Name has published its provisional financial statement for..."
        name_match = re.match(r"^(.+?)\s+has\s+published", text, re.I)
        if not name_match:
            continue
        company_name = name_match.group(1).strip().lower()

        # Look up symbol
        matched_symbol = name_to_symbol.get(company_name)
        if not matched_symbol:
            # Try partial match
            for name, sym in name_to_symbol.items():
                if company_name in name or name in company_name:
                    matched_symbol = sym
                    break

        if matched_symbol:
            all_announcements.append({
                "symbol": matched_symbol,
                "fiscal_year": fiscal_year,
                "quarter": q_num,
            })

    logger.info("Parsed %d announcements from MeroLagani reports page", len(all_announcements))

    # Now update the quarterly_earnings table with today's date as announcement_date
    # (since these are recent announcements, the date is approximately today)
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    updated = 0

    for ann in all_announcements:
        try:
            cursor.execute("""
                UPDATE quarterly_earnings
                SET announcement_date = ?
                WHERE symbol = ? AND fiscal_year LIKE ? AND quarter = ?
                  AND announcement_date IS NULL
            """, (today, ann["symbol"], f"%{ann['fiscal_year'][-4:]}%", ann["quarter"]))
            if cursor.rowcount > 0:
                updated += cursor.rowcount
        except sqlite3.Error as e:
            logger.warning("Failed to update announcement date for %s: %s", ann["symbol"], e)

    conn.commit()
    conn.close()

    logger.info("Updated %d announcement dates", updated)
    return updated


# ---------------------------------------------------------------------------
# Convenience: get all quarterly earnings from DB
# ---------------------------------------------------------------------------

def get_quarterly_earnings(
    db_path: Optional[str] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Load quarterly earnings from DB, optionally filtered by symbol."""
    db_path = db_path or _get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)

    if symbol:
        query = "SELECT * FROM quarterly_earnings WHERE symbol = ? ORDER BY fiscal_year DESC, quarter DESC"
        df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    else:
        query = "SELECT * FROM quarterly_earnings ORDER BY symbol, fiscal_year DESC, quarter DESC"
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the earnings scraper."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="NEPSE Quarterly Earnings Scraper")
    parser.add_argument(
        "--symbols", nargs="*",
        help="Symbols to scrape (default: top 30 NEPSE stocks)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Database path (default: nepse_market_data.db)",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Do not skip symbols with existing data",
    )
    args = parser.parse_args()

    stats = scrape_all_earnings(
        symbols=args.symbols,
        db_path=args.db,
        skip_existing=not args.no_skip,
    )
    print(f"\nScraping Summary:")
    print(f"  Symbols scraped: {stats['total_scraped']}")
    print(f"  Rows inserted:   {stats['total_rows']}")
    print(f"  Errors:          {stats['errors']}")


if __name__ == "__main__":
    main()
