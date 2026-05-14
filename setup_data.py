#!/usr/bin/env python3
"""
NEPSE Quant Terminal — Initial data setup.

Downloads the pre-built database from the GitHub release (fast, ~13 MB),
or falls back to scraping from Merolagani if preferred.

Usage:
    python setup_data.py               # download pre-built DB (recommended)
    python setup_data.py --scrape      # scrape fresh from Merolagani instead
    python setup_data.py --scrape --days 90   # scrape only last 90 days
    python setup_data.py --symbol NABIL       # scrape a single symbol
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 14):
    print(
        "ERROR: Python 3.14+ is not yet supported.\n"
        "The nepse package and numba both cap at Python <3.14.\n"
        "Please use Python 3.10 – 3.13 (recommended: 3.12).\n"
        "  pyenv install 3.12 && pyenv local 3.12"
    )
    sys.exit(1)

import argparse
import gzip
import shutil
import sqlite3
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

DB_RELEASE_URL = (
    "https://github.com/nlethetech/nepse-quant-terminal"
    "/releases/download/v3.0/nepse_data_public.db.gz"
)
ALL_SYMBOLS_FILE = Path(__file__).parent / "all_symbols.txt"
DELAY_BETWEEN_SYMBOLS = 1.2  # seconds — stay under Merolagani rate limit


# ── helpers ───────────────────────────────────────────────────────────────────

def _db_path() -> Path:
    from backend.quant_pro.database import get_db_path
    return Path(get_db_path())


def _progress(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


# ── download path ─────────────────────────────────────────────────────────────

def download_db():
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = db_path.parent / "nepse_data_public.db.gz"

    print(f"Downloading pre-built database from GitHub release...")
    print(f"  {DB_RELEASE_URL}")
    try:
        urllib.request.urlretrieve(DB_RELEASE_URL, gz_path, _progress)
        print()
    except Exception as e:
        print(f"\nERROR: download failed: {e}")
        print("Try --scrape to fetch data directly from Merolagani instead.")
        return False

    print(f"Extracting to {db_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    # Quick sanity check
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
    conn.close()
    print(f"Done — {rows:,} price rows in {db_path}  ({db_path.stat().st_size // 1_000_000} MB)")
    return True


# ── scrape path ───────────────────────────────────────────────────────────────

def load_symbols() -> list[str]:
    if ALL_SYMBOLS_FILE.exists():
        from backend.quant_pro.signal_ranking import is_tradeable_signal_symbol
        syms = [s.strip() for s in ALL_SYMBOLS_FILE.read_text().splitlines() if s.strip()]
        return [s for s in syms if is_tradeable_signal_symbol(s)]
    return []


def ts(dt: datetime) -> int:
    return int(dt.timestamp())


def backfill_symbol(symbol: str, start: datetime, end: datetime) -> int:
    from backend.quant_pro.database import save_to_db
    from backend.quant_pro.vendor_api import fetch_ohlcv_chunk
    try:
        df = fetch_ohlcv_chunk(symbol, ts(start), ts(end))
        if df is None or df.empty:
            return 0
        df["symbol"] = symbol
        save_to_db(df, symbol)
        return len(df)
    except Exception as e:
        print(f"  WARN {symbol}: {e}")
        return 0


def scrape_db(days: int, symbol: str | None):
    from backend.quant_pro.database import get_db_path, init_db
    print("Initialising database...")
    init_db()

    end   = datetime.now()
    start = end - timedelta(days=days)
    symbols = [symbol] if symbol else load_symbols()

    if not symbols:
        print("ERROR: no symbols found. Check all_symbols.txt exists.")
        return

    print(f"Scraping {len(symbols)} symbols from Merolagani  |  {start.date()} → {end.date()}")
    print(f"DB: {get_db_path()}\n")

    total = 0
    for i, sym in enumerate(symbols, 1):
        rows = backfill_symbol(sym, start, end)
        total += rows
        print(f"[{i:3d}/{len(symbols)}] {sym:<12} {rows:>5} rows")
        if i < len(symbols):
            time.sleep(DELAY_BETWEEN_SYMBOLS)

    print(f"\nDone — {total:,} rows written to {get_db_path()}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEPSE Quant — data setup")
    parser.add_argument("--scrape", action="store_true",
                        help="Scrape fresh data from Merolagani instead of downloading")
    parser.add_argument("--days", type=int, default=760,
                        help="Days of history to scrape (default 760 ≈ 2 yr). Only used with --scrape.")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to scrape. Only used with --scrape.")
    args = parser.parse_args()

    if args.scrape:
        scrape_db(days=args.days, symbol=args.symbol)
    else:
        ok = download_db()
        if not ok:
            print("\nFalling back to scrape...")
            scrape_db(days=760, symbol=None)

    print("\nLaunch the terminal with:  python3 -m apps.tui.dashboard_tui")


if __name__ == "__main__":
    main()
