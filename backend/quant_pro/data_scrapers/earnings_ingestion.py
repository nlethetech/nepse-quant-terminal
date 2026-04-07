#!/usr/bin/env python3
"""
Standalone quarterly earnings ingestion script.

Scrapes quarterly financial data from ShareSansar and MeroLagani,
stores in the quarterly_earnings SQLite table.

Usage:
    python3 -m backend.quant_pro.data_scrapers.earnings_ingestion
    python3 -m backend.quant_pro.data_scrapers.earnings_ingestion --symbols NABIL SCB SBI
    python3 -m backend.quant_pro.data_scrapers.earnings_ingestion --all
    python3 -m backend.quant_pro.data_scrapers.earnings_ingestion --summary
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from backend.quant_pro.paths import get_project_root

# Ensure the project root is on sys.path
project_root = str(get_project_root(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.quant_pro.earnings_scraper import (
    TARGET_SYMBOLS,
    create_quarterly_earnings_table,
    get_quarterly_earnings,
    scrape_all_earnings,
)


def _get_db_path() -> str:
    raw = os.environ.get("NEPSE_DB_FILE", "nepse_market_data.db")
    return str(Path(raw).resolve())


def show_summary(db_path: str) -> None:
    """Display a summary of quarterly earnings data in the database."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='quarterly_earnings'"
        )
        if not cursor.fetchone():
            print("quarterly_earnings table does not exist yet.")
            print("Run: python3 -m backend.quant_pro.data_scrapers.earnings_ingestion")
            return

        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM quarterly_earnings")
        total, n_symbols = cursor.fetchone()
        print(f"\n{'='*60}")
        print(f"QUARTERLY EARNINGS DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total rows:      {total}")
        print(f"Unique symbols:  {n_symbols}")

        if total == 0:
            print("\nNo data yet. Run the scraper to populate.")
            conn.close()
            return

        # Per-symbol summary
        cursor.execute("""
            SELECT symbol, COUNT(*) as quarters,
                   GROUP_CONCAT(fiscal_year || ' Q' || quarter, ', ') as quarters_list,
                   MAX(eps) as max_eps,
                   MIN(eps) as min_eps,
                   MAX(announcement_date) as latest_ann
            FROM quarterly_earnings
            GROUP BY symbol
            ORDER BY symbol
        """)
        rows = cursor.fetchall()

        print(f"\n{'Symbol':>10s} {'Qtrs':>5s} {'EPS Range':>14s} {'Latest Ann':>12s}  Quarters")
        print(f"{'-'*10:>10s} {'-'*5:>5s} {'-'*14:>14s} {'-'*12:>12s}  {'-'*30}")
        for sym, qtrs, qlist, max_eps, min_eps, latest_ann in rows:
            eps_range = f"{min_eps or 0:6.1f}-{max_eps or 0:6.1f}" if max_eps else "N/A"
            ann = latest_ann or "NULL"
            print(f"{sym:>10s} {qtrs:>5d} {eps_range:>14s} {ann:>12s}  {qlist[:60]}")

        # Source breakdown
        cursor.execute("""
            SELECT source, COUNT(*) FROM quarterly_earnings GROUP BY source
        """)
        sources = cursor.fetchall()
        print(f"\nData sources: {', '.join(f'{s}: {c}' for s, c in sources)}")

        # Announcement date coverage
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN announcement_date IS NOT NULL THEN 1 ELSE 0 END) as with_ann
            FROM quarterly_earnings
        """)
        total, with_ann = cursor.fetchone()
        pct = (with_ann / total * 100) if total > 0 else 0
        print(f"Announcement dates: {with_ann}/{total} ({pct:.0f}%)")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="NEPSE Quarterly Earnings Ingestion",
        epilog=(
            "Examples:\n"
            "  python3 -m backend.quant_pro.data_scrapers.earnings_ingestion\n"
            "  python3 -m backend.quant_pro.data_scrapers.earnings_ingestion --symbols NABIL SCB\n"
            "  python3 -m backend.quant_pro.data_scrapers.earnings_ingestion --summary\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Symbols to scrape (default: top 30 NEPSE stocks)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scrape ALL available symbols from ShareSansar (200+)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Database path (default: nepse_market_data.db)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show data summary and exit",
    )
    parser.add_argument(
        "--create-table", action="store_true",
        help="Create the quarterly_earnings table and exit",
    )

    args = parser.parse_args()
    db_path = args.db or _get_db_path()

    if args.create_table:
        create_quarterly_earnings_table(db_path)
        print(f"quarterly_earnings table created at {db_path}")
        return

    if args.summary:
        show_summary(db_path)
        return

    # Determine symbols to scrape
    symbols = args.symbols
    if args.all:
        # Get all symbols from the stock_prices table
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT symbol FROM stock_prices
                WHERE symbol NOT LIKE 'SECTOR::%' AND symbol != 'NEPSE'
                ORDER BY symbol
            """)
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            print(f"Found {len(symbols)} symbols in database")
        except sqlite3.Error:
            print("Could not read symbols from database, using defaults")
            symbols = None

    # Run scraper
    print(f"\nStarting earnings scraper...")
    print(f"Database: {db_path}")
    print(f"Symbols:  {len(symbols) if symbols else len(TARGET_SYMBOLS)} targets")
    print(f"{'='*60}\n")

    stats = scrape_all_earnings(
        symbols=symbols,
        db_path=db_path,
        skip_existing=True,
    )

    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"  Symbols scraped: {stats['total_scraped']}")
    print(f"  Rows inserted:   {stats['total_rows']}")
    print(f"  Errors:          {stats['errors']}")
    print(f"  Skipped:         {stats.get('skipped', 0)}")

    # Show summary
    print()
    show_summary(db_path)


if __name__ == "__main__":
    main()
