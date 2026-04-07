"""
Kalimati Market — scraper + SQLite data layer.
Fetches daily vegetable/fruit prices from kalimatimarket.gov.np and stores them locally.
"""
from __future__ import annotations

import datetime
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String,
    create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from backend.market.kalimati_translations import translate_name, translate_unit

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "kalimati.db"
_engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class _Base(DeclarativeBase):
    pass


class KCommodity(_Base):
    __tablename__ = "k_commodities"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    name_nepali   = Column(String, unique=True, nullable=False)
    name_english  = Column(String, nullable=False)
    unit_nepali   = Column(String)
    unit_english  = Column(String)
    prices        = relationship("KPrice", back_populates="commodity",
                                 cascade="all, delete-orphan")


class KPrice(_Base):
    __tablename__ = "k_prices"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    commodity_id  = Column(Integer, ForeignKey("k_commodities.id"), nullable=False)
    date          = Column(String, nullable=False)   # YYYY-MM-DD
    min_price     = Column(Float)
    max_price     = Column(Float)
    avg_price     = Column(Float)
    scraped_at    = Column(DateTime, default=datetime.datetime.now)
    commodity     = relationship("KCommodity", back_populates="prices")


def init_kalimati_db() -> None:
    _Base.metadata.create_all(_engine)


# ── Scraper ───────────────────────────────────────────────────────────────────

_PRICE_URLS = [
    "https://kalimatimarket.gov.np/lang/en/price",
    "https://kalimatimarket.gov.np/price",
]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ne;q=0.8",
}
_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def _to_float(text: str) -> float | None:
    cleaned = text.replace("रू", "").replace("Rs", "").replace(",", "").strip()
    cleaned = cleaned.translate(_DIGIT_MAP)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_english(text: str) -> bool:
    if not text:
        return False
    return sum(1 for c in text if c.isascii() and c.isalpha()) > len(text) * 0.5


def fetch_kalimati_prices(timeout: int = 15) -> list[dict]:
    """Scrape Kalimati market. Returns list of dicts with name/unit/min/max/avg."""
    last_err = None
    for url in _PRICE_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"id": "commodityPriceParticular"})
            if not table:
                for t in soup.find_all("table"):
                    if "न्यूनतम" in t.get_text() or "minimum" in t.get_text().lower():
                        table = t
                        break
            if not table:
                continue

            rows = []
            is_en = "/lang/en/" in url
            for tr in (table.find("tbody") or table).find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                name_raw = tds[0].get_text(strip=True)
                unit_raw = tds[1].get_text(strip=True) if len(tds) > 4 else ""
                min_v  = _to_float(tds[-3].get_text(strip=True))
                max_v  = _to_float(tds[-2].get_text(strip=True))
                avg_v  = _to_float(tds[-1].get_text(strip=True))
                if not name_raw or avg_v is None:
                    continue

                if is_en and _is_english(name_raw):
                    name_en = name_raw
                    name_np = name_raw
                else:
                    name_np = name_raw
                    name_en = translate_name(name_raw)

                unit_en = translate_unit(unit_raw)
                rows.append({
                    "name_nepali": name_np,
                    "name_english": name_en,
                    "unit_nepali": unit_raw,
                    "unit_english": unit_en,
                    "min": min_v or 0.0,
                    "max": max_v or 0.0,
                    "avg": avg_v,
                })
            if rows:
                return rows
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Kalimati fetch failed: {last_err}")


# ── DB write ──────────────────────────────────────────────────────────────────

def store_kalimati_prices(rows: list[dict], date: str | None = None) -> int:
    """Store scraped rows into DB. Returns number stored."""
    date = date or datetime.date.today().isoformat()
    with Session(_engine) as session:
        for r in rows:
            comm = session.query(KCommodity).filter_by(name_nepali=r["name_nepali"]).first()
            if comm is None:
                comm = KCommodity(
                    name_nepali=r["name_nepali"],
                    name_english=r["name_english"],
                    unit_nepali=r["unit_nepali"],
                    unit_english=r["unit_english"],
                )
                session.add(comm)
                session.flush()
            else:
                comm.unit_english = r["unit_english"]
                if r["name_english"] != r["name_nepali"]:
                    comm.name_english = r["name_english"]

            existing = session.query(KPrice).filter_by(
                commodity_id=comm.id, date=date
            ).first()
            if existing:
                existing.min_price = r["min"]
                existing.max_price = r["max"]
                existing.avg_price = r["avg"]
                existing.scraped_at = datetime.datetime.now()
            else:
                session.add(KPrice(
                    commodity_id=comm.id,
                    date=date,
                    min_price=r["min"],
                    max_price=r["max"],
                    avg_price=r["avg"],
                ))
        session.commit()
    return len(rows)


# ── DB read ───────────────────────────────────────────────────────────────────

def get_kalimati_display_rows() -> list[dict]:
    """Return latest price row per commodity with prev-day change calculation."""
    with Session(_engine) as session:
        latest_date = session.query(func.max(KPrice.date)).scalar()
        if not latest_date:
            return []
        prev_date = (
            session.query(func.max(KPrice.date))
            .filter(KPrice.date < latest_date)
            .scalar()
        )

        # Build prev-day map
        prev_map: dict[int, float] = {}
        if prev_date:
            for p, c in (
                session.query(KPrice, KCommodity)
                .join(KCommodity)
                .filter(KPrice.date == prev_date)
                .all()
            ):
                prev_map[c.id] = p.avg_price

        rows = []
        for price, comm in (
            session.query(KPrice, KCommodity)
            .join(KCommodity)
            .filter(KPrice.date == latest_date)
            .order_by(KCommodity.name_english)
            .all()
        ):
            prev = prev_map.get(comm.id)
            if prev and prev > 0:
                change = price.avg_price - prev
                change_pct = (change / prev) * 100
            else:
                change = None
                change_pct = None

            rows.append({
                "id": comm.id,
                "name_english": comm.name_english,
                "name_nepali": comm.name_nepali,
                "unit": comm.unit_english or comm.unit_nepali or "KG",
                "min": price.min_price,
                "max": price.max_price,
                "avg": price.avg_price,
                "prev_avg": prev,
                "change": change,
                "change_pct": change_pct,
                "date": price.date,
                "scraped_at": price.scraped_at,
            })
        return rows


def refresh_kalimati() -> tuple[list[dict], str]:
    """Fetch + store + return display rows. Returns (rows, status_msg)."""
    try:
        raw = fetch_kalimati_prices()
        n = store_kalimati_prices(raw)
        rows = get_kalimati_display_rows()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        return rows, f"Kalimati: {n} items  Updated {ts}"
    except Exception as e:
        rows = get_kalimati_display_rows()  # return cached
        return rows, f"Kalimati fetch failed: {e}"
