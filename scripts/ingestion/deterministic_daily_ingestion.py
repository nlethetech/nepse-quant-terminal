#!/usr/bin/env python3
"""
Deterministic daily ingestion for full NEPSE universe.

Features:
1) Stable symbol ordering and explicit fetch windows.
2) Run metadata (ingestion_runs / ingestion_run_symbols).
3) Freshness SLA enforcement.
"""

from __future__ import annotations

import argparse
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.quant_pro.database import get_db_path, save_to_db
from backend.quant_pro.institutional import (
    INGESTION_RUN_STATUS,
    INGESTION_SYMBOL_STATUS,
    get_db_latest_market_date,
    init_institutional_tables,
    utc_now_iso,
)
from backend.quant_pro.vendor_api import fetch_ohlcv_chunk

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[ingestion] Shutdown requested (signal {signum}), finishing current symbol...")


DEFAULT_DB_FILE = str(get_db_path())


@dataclass
class SymbolResult:
    symbol: str
    status: str
    rows_fetched: int
    rows_added: int
    latest_before: Optional[str]
    latest_after: Optional[str]
    error: Optional[str] = None


def _fetch_all_rows(conn: sqlite3.Connection, query: str, params: Sequence = ()) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(query, params)
    return cur.fetchall()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return cur.fetchone() is not None


def get_symbol_universe(
    conn: sqlite3.Connection,
    source: str,
    watchlist_id: Optional[int],
    symbols_file: Optional[Path],
) -> List[str]:
    symbols: Set[str] = set()

    if source in {"db", "both", "all"}:
        if _table_exists(conn, "stock_prices"):
            rows = _fetch_all_rows(
                conn,
                """
                SELECT DISTINCT symbol
                FROM stock_prices
                WHERE symbol NOT LIKE 'SECTOR::%'
                """,
            )
            symbols.update(str(r[0]).strip().upper() for r in rows if r[0])
        else:
            print("[ingestion] table 'stock_prices' not found; skipping db source")

    if source in {"quotes", "both", "all"}:
        # Include ALL symbols from live market quotes (full NEPSE universe)
        if _table_exists(conn, "market_quotes"):
            rows = _fetch_all_rows(
                conn,
                """
                SELECT DISTINCT symbol
                FROM market_quotes
                WHERE symbol NOT LIKE 'SECTOR::%'
                  AND symbol NOT LIKE '%/%'
                  AND length(symbol) <= 10
                """,
            )
            symbols.update(str(r[0]).strip().upper() for r in rows if r[0])
        else:
            print("[ingestion] table 'market_quotes' not found; skipping quotes source")

    if source in {"watchlist", "both", "all"}:
        if _table_exists(conn, "watchlist_items"):
            if watchlist_id is not None:
                rows = _fetch_all_rows(
                    conn,
                    "SELECT DISTINCT symbol FROM watchlist_items WHERE watchlist_id = ?",
                    (watchlist_id,),
                )
            else:
                rows = _fetch_all_rows(conn, "SELECT DISTINCT symbol FROM watchlist_items")
            symbols.update(str(r[0]).strip().upper() for r in rows if r[0])
        else:
            print("[ingestion] table 'watchlist_items' not found; skipping watchlist source")

    if source in {"file", "all"} and symbols_file is not None:
        for line in symbols_file.read_text(encoding="utf-8").splitlines():
            sym = line.strip().upper()
            if sym and not sym.startswith("#"):
                symbols.add(sym)

    # Drop obvious empties/noise and keep deterministic order.
    clean = sorted(s for s in symbols if s)
    return clean


def get_symbol_latest_date(conn: sqlite3.Connection, symbol: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stock_prices WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def get_symbol_row_count(conn: sqlite3.Connection, symbol: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM stock_prices WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    return int(row[0] or 0)


def start_run(
    conn: sqlite3.Connection,
    *,
    source: str,
    universe_size: int,
    latest_market_date_before: Optional[str],
    sla_max_staleness_days: int,
    notes: str,
) -> int:
    init_institutional_tables(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ingestion_runs (
            started_at_utc, status, source, universe_size,
            latest_market_date_before, sla_max_staleness_days, notes
        )
        VALUES (?, 'RUNNING', ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            source,
            int(universe_size),
            latest_market_date_before,
            int(sla_max_staleness_days),
            notes,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_symbol_result(conn: sqlite3.Connection, run_id: int, result: SymbolResult) -> None:
    if result.status not in INGESTION_SYMBOL_STATUS:
        raise ValueError(f"Invalid ingestion symbol status: {result.status}")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO ingestion_run_symbols (
            run_id, symbol, started_at_utc, ended_at_utc, status, rows_fetched,
            rows_added, latest_date_before, latest_date_after, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            result.symbol,
            utc_now_iso(),
            utc_now_iso(),
            result.status,
            int(result.rows_fetched),
            int(result.rows_added),
            result.latest_before,
            result.latest_after,
            result.error,
        ),
    )
    conn.commit()


def end_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    status: str,
    symbols_succeeded: int,
    symbols_failed: int,
    rows_fetched: int,
    latest_market_date_after: Optional[str],
    freshness_days_after: Optional[int],
) -> None:
    if status not in INGESTION_RUN_STATUS:
        raise ValueError(f"Invalid ingestion run status: {status}")
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE ingestion_runs
        SET ended_at_utc = ?,
            status = ?,
            symbols_succeeded = ?,
            symbols_failed = ?,
            rows_fetched = ?,
            latest_market_date_after = ?,
            freshness_days_after = ?
        WHERE run_id = ?
        """,
        (
            utc_now_iso(),
            status,
            int(symbols_succeeded),
            int(symbols_failed),
            int(rows_fetched),
            latest_market_date_after,
            freshness_days_after,
            int(run_id),
        ),
    )
    conn.commit()


def fetch_symbol_history(
    symbol: str,
    latest_before: Optional[str],
    history_days: int,
    backfill_days: int,
) -> pd.DataFrame:
    now_ts = int(time.time())
    if latest_before is None:
        start_dt = datetime.now() - timedelta(days=history_days)
    else:
        start_dt = pd.Timestamp(latest_before).to_pydatetime() - timedelta(days=backfill_days)
    start_ts = int(start_dt.timestamp())
    df = fetch_ohlcv_chunk(symbol, start_ts=start_ts, end_ts=now_ts)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")


def cleanup_invalid_ohlcv_rows(conn: sqlite3.Connection) -> int:
    """
    Remove invalid historical bars to keep data-quality gates deterministic.
    """
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM stock_prices
        WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR volume < 0
        """
    )
    deleted = int(cur.rowcount or 0)
    conn.commit()
    return deleted


def execute_ingestion(
    *,
    conn: sqlite3.Connection,
    symbols: List[str],
    source: str,
    history_days: int,
    backfill_days: int,
    max_staleness_days: int,
    sleep_ms: int,
    strict: bool,
) -> int:
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    latest_before = get_db_latest_market_date(conn)
    run_id = start_run(
        conn,
        source=source,
        universe_size=len(symbols),
        latest_market_date_before=latest_before,
        sla_max_staleness_days=max_staleness_days,
        notes=f"history_days={history_days}, backfill_days={backfill_days}",
    )
    print(f"[ingestion] started run_id={run_id} symbols={len(symbols)} source={source}")

    deleted_bad_rows = cleanup_invalid_ohlcv_rows(conn)
    if deleted_bad_rows > 0:
        print(f"[ingestion] removed invalid_ohlcv_rows={deleted_bad_rows}")

    succeeded = 0
    failed = 0
    rows_fetched_total = 0

    for idx, symbol in enumerate(symbols, start=1):
        latest_symbol_before = get_symbol_latest_date(conn, symbol)
        rows_before = get_symbol_row_count(conn, symbol)
        try:
            df = fetch_symbol_history(
                symbol=symbol,
                latest_before=latest_symbol_before,
                history_days=history_days,
                backfill_days=backfill_days,
            )
            rows_fetched = int(len(df))
            rows_fetched_total += rows_fetched

            if rows_fetched > 0:
                save_to_db(df, symbol)

            rows_after = get_symbol_row_count(conn, symbol)
            latest_symbol_after = get_symbol_latest_date(conn, symbol)
            rows_added = max(rows_after - rows_before, 0)

            status = "SUCCESS" if rows_fetched > 0 else "NO_DATA"
            save_symbol_result(
                conn,
                run_id,
                SymbolResult(
                    symbol=symbol,
                    status=status,
                    rows_fetched=rows_fetched,
                    rows_added=rows_added,
                    latest_before=latest_symbol_before,
                    latest_after=latest_symbol_after,
                ),
            )
            succeeded += 1
            print(
                f"[{idx:04d}/{len(symbols):04d}] {symbol:<12} status={status:<7} "
                f"fetched={rows_fetched:<5d} added={rows_added:<5d} "
                f"last_before={latest_symbol_before} last_after={latest_symbol_after}"
            )
        except Exception as exc:  # keep run deterministic; isolate symbol failures
            failed += 1
            save_symbol_result(
                conn,
                run_id,
                SymbolResult(
                    symbol=symbol,
                    status="FAILED",
                    rows_fetched=0,
                    rows_added=0,
                    latest_before=latest_symbol_before,
                    latest_after=get_symbol_latest_date(conn, symbol),
                    error=str(exc),
                ),
            )
            print(f"[{idx:04d}/{len(symbols):04d}] {symbol:<12} status=FAILED  error={exc}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        if _shutdown_requested:
            print(f"[ingestion] Graceful shutdown after {idx}/{len(symbols)} symbols")
            status = "PARTIAL"
            break

    latest_after = get_db_latest_market_date(conn)
    freshness_days: Optional[int] = None
    if latest_after is not None:
        freshness_days = (datetime.now().date() - pd.Timestamp(latest_after).date()).days

    if failed == 0 and freshness_days is not None and freshness_days <= max_staleness_days:
        status = "SUCCESS"
    elif succeeded > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"

    if _shutdown_requested and status == "SUCCESS":
        status = "PARTIAL"

    end_run(
        conn,
        run_id=run_id,
        status=status,
        symbols_succeeded=succeeded,
        symbols_failed=failed,
        rows_fetched=rows_fetched_total,
        latest_market_date_after=latest_after,
        freshness_days_after=freshness_days,
    )

    print(
        "[ingestion] completed "
        f"run_id={run_id} status={status} succeeded={succeeded} failed={failed} "
        f"rows_fetched={rows_fetched_total} latest_market_date={latest_after} "
        f"freshness_days={freshness_days}"
    )

    if strict:
        if status != "SUCCESS":
            return 2
        if freshness_days is None or freshness_days > max_staleness_days:
            return 3
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic NEPSE ingestion job")
    parser.add_argument("--db-file", default=DEFAULT_DB_FILE, help="SQLite database file path")
    parser.add_argument(
        "--source",
        default="both",
        choices=["db", "watchlist", "quotes", "both", "file", "all"],
        help="How to build symbol universe (quotes = full NEPSE from market_quotes)",
    )
    parser.add_argument("--watchlist-id", type=int, default=None, help="Optional watchlist id filter")
    parser.add_argument("--symbols-file", type=Path, default=None, help="Optional file with one symbol per line")
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit symbols for controlled runs")
    parser.add_argument("--history-days", type=int, default=3650, help="Backfill horizon for new symbols")
    parser.add_argument("--backfill-days", type=int, default=5, help="Overlap window for existing symbols")
    parser.add_argument("--max-staleness-days", type=int, default=7, help="Freshness SLA")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between symbols")
    parser.add_argument("--strict", action="store_true", help="Fail on non-success run or SLA breach")
    return parser.parse_args(argv)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db_file)
    if not db_path.exists():
        print(f"[ingestion] database file not found: {db_path}")
        return 1

    conn = connect_db(db_path)
    try:
        init_institutional_tables(conn)
        symbols = get_symbol_universe(
            conn=conn,
            source=args.source,
            watchlist_id=args.watchlist_id,
            symbols_file=args.symbols_file,
        )
        if args.max_symbols and args.max_symbols > 0:
            symbols = symbols[: args.max_symbols]
        if not symbols:
            print("[ingestion] no symbols resolved for selected source")
            return 1

        return execute_ingestion(
            conn=conn,
            symbols=symbols,
            source=args.source,
            history_days=args.history_days,
            backfill_days=args.backfill_days,
            max_staleness_days=args.max_staleness_days,
            sleep_ms=args.sleep_ms,
            strict=args.strict,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
