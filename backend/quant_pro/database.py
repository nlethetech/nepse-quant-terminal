import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .exceptions import DatabaseError
from .paths import get_data_dir

# Configure logging
logger = logging.getLogger(__name__)

# Flag to track if WAL mode has been initialized
_wal_initialized = False


def get_db_path() -> Path:
    """
    Return the canonical database file path.

    Reads ``NEPSE_DB_FILE`` from the environment (if set) and resolves it to an
    absolute path. Falls back to ``data/nepse_market_data.db`` in the project root.
    """
    raw = os.environ.get("NEPSE_DB_FILE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return get_data_dir(__file__) / "nepse_market_data.db"


# Legacy module-level constant kept for backwards compat during migration.
DB_FILE = str(get_db_path())


def _is_nepse_trading_day(date_like: object) -> bool:
    """
    Prefer deriving trading days from benchmark history when available:
    - If NEPSE index has a row for the date, treat it as a trading day.
    - Otherwise fall back to the NEPSE weekmask (Sun–Thu) used throughout the codebase.
    """
    try:
        day = pd.Timestamp(date_like).normalize()
    except (ValueError, TypeError):
        return False
    day_str = day.strftime("%Y-%m-%d")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM stock_prices WHERE symbol = ? AND date = ? LIMIT 1", ("NEPSE", day_str))
        row = cur.fetchone()
        conn.close()
        if row is not None:
            return True
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass
    # Fallback: Sunday(6) ... Thursday(3) on pandas dayofweek scale (Mon=0).
    try:
        return int(day.dayofweek) in {6, 0, 1, 2, 3}
    except (ValueError, AttributeError):
        return False


def get_db_connection(timeout: float = 60.0, retries: int = 3) -> sqlite3.Connection:
    """
    Get database connection with proper pragmas for concurrency.

    Retries with exponential backoff on ``OperationalError`` (database locked).
    Raises ``DatabaseError`` after exhausting retries.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = sqlite3.connect(str(get_db_path()), timeout=timeout)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 60000")
            return conn
        except sqlite3.OperationalError as exc:
            last_err = exc
            if attempt < retries:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "DB connection attempt %d/%d failed (%s), retrying in %ds",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)
    raise DatabaseError(f"Failed to connect after {retries} attempts: {last_err}") from last_err


def init_db():
    """Creates the table if it doesn't exist and configures WAL mode."""
    global _wal_initialized
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable WAL mode for better concurrency (only needs to be done once per database file)
    if not _wal_initialized:
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")  # Balance durability/speed
            cursor.execute("PRAGMA cache_size=-64000")   # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")
            _wal_initialized = True
            logger.debug("SQLite WAL mode enabled")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to enable WAL mode: {e}")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')

    # Create index for faster symbol lookups if not exists
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol
        ON stock_prices (symbol)
    ''')

    # Corporate actions table (used by signal generators and backtest)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corporate_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            fiscal_year TEXT,
            bookclose_date DATE,
            cash_dividend_pct REAL DEFAULT 0,
            bonus_share_pct REAL DEFAULT 0,
            right_share_ratio TEXT,
            agenda TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, bookclose_date)
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_corporate_actions_bookclose
        ON corporate_actions (bookclose_date)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_corporate_actions_symbol
        ON corporate_actions (symbol)
    ''')

    # Raw intraday market snapshots for audit/replay of upstream payloads.
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS market_data_raw (
            raw_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            source TEXT NOT NULL,
            symbol TEXT,
            business_date TEXT,
            fetched_at_utc TEXT NOT NULL,
            record_count INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            metadata_json TEXT
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_market_data_raw_dataset_fetched
        ON market_data_raw (dataset, fetched_at_utc DESC)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_market_data_raw_symbol_fetched
        ON market_data_raw (symbol, fetched_at_utc DESC)
        '''
    )

    # Normalized quote snapshots for fast symbol-level lookup.
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS market_quotes (
            raw_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            security_id TEXT,
            security_name TEXT,
            last_traded_price REAL,
            close_price REAL,
            previous_close REAL,
            percentage_change REAL,
            total_trade_quantity REAL,
            source TEXT NOT NULL,
            fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (raw_id, symbol),
            FOREIGN KEY (raw_id) REFERENCES market_data_raw(raw_id)
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol_fetched
        ON market_quotes (symbol, fetched_at_utc DESC)
        '''
    )

    # Daily benchmark history snapshots for local performance comparison.
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS benchmark_index_history (
            benchmark TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume REAL,
            source TEXT NOT NULL,
            fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY (benchmark, date)
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_benchmark_index_history_date
        ON benchmark_index_history (benchmark, date DESC)
        '''
    )

    # Earnings / fundamentals snapshots used by lookup, signals, and reports.
    cursor.execute(
        '''
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
            scraped_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, fiscal_year, quarter)
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_qe_symbol
        ON quarterly_earnings (symbol)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_qe_announcement
        ON quarterly_earnings (announcement_date)
        '''
    )

    cursor.execute(
        '''
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
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol
        ON fundamentals (symbol, date DESC)
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date DATE,
            headline TEXT,
            url TEXT UNIQUE,
            source TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            category TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_news_symbol
        ON news(symbol)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_news_date
        ON news(date DESC)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_news_sentiment
        ON news(sentiment_label)
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            date TEXT NOT NULL,
            symbol TEXT,
            source TEXT NOT NULL,
            model TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL,
            n_documents INTEGER,
            scraped_at_utc TEXT NOT NULL,
            PRIMARY KEY (date, symbol, source, model)
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_sentiment_date
        ON sentiment_scores(date DESC)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_sentiment_symbol
        ON sentiment_scores(symbol)
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS news_event_scores (
            event_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            window_start_utc TEXT NOT NULL,
            window_end_utc TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            impact_direction TEXT NOT NULL,
            impact_score REAL NOT NULL,
            confidence REAL NOT NULL,
            event_type TEXT NOT NULL,
            source_count INTEGER NOT NULL DEFAULT 0,
            source_refs_json TEXT NOT NULL,
            rationale_short TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        )
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_news_event_scores_run_entity
        ON news_event_scores(run_date DESC, entity_type, entity_key)
        '''
    )
    cursor.execute(
        '''
        CREATE INDEX IF NOT EXISTS idx_news_event_scores_created
        ON news_event_scores(created_at_utc DESC)
        '''
    )

    conn.commit()
    conn.close()

def get_latest_date(symbol):
    """Returns the most recent date we have for a symbol, or None."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM stock_prices WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()[0]
    conn.close()
    return pd.to_datetime(result).date() if result else None

def save_to_db(df, symbol):
    """Saves new data to SQLite, ignoring duplicates."""
    if df.empty: return

    conn = get_db_connection()
    df = df.copy()
    df["symbol"] = symbol
    # Ensure date is string YYYY-MM-DD for SQLite
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d')

    # Data hygiene: reject accidental non-trading-day inserts when Volume==0 and the date
    # is not a known trading session (prevents stale "live candles" polluting EOD history).
    try:
        vol = pd.to_numeric(df.get("Volume", 0.0), errors="coerce").fillna(0.0)
        dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        non_trading = ~dates.apply(_is_nepse_trading_day)
        bad = (vol <= 0.0) & non_trading
        if bad.any():
            df = df.loc[~bad].copy()
            if df.empty:
                conn.close()
                return
    except (KeyError, ValueError, TypeError) as exc:
        logger.debug("Data hygiene check skipped: %s", exc)

    # Rename columns to match DB schema
    df_to_save = df[["symbol", "date", "Open", "High", "Low", "Close", "Volume"]]
    df_to_save.columns = ["symbol", "date", "open", "high", "low", "close", "volume"]

    # Efficient Upsert (Insert or Ignore)
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', df_to_save.values.tolist())

    conn.commit()
    conn.close()

def load_from_db(symbol):
    """Loads full history for a symbol."""
    conn = get_db_connection()
    query = """
        SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume
        FROM stock_prices
        WHERE symbol = ?
        ORDER BY date ASC
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    return df


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def save_market_data_raw(
    *,
    dataset: str,
    source: str,
    payload: Any,
    symbol: Optional[str] = None,
    business_date: Optional[str] = None,
    fetched_at_utc: Optional[str] = None,
    record_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Persist raw upstream payloads for later audit/replay."""
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    fetched_at_utc = fetched_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if record_count is None:
        if isinstance(payload, list):
            record_count = len(payload)
        elif isinstance(payload, dict):
            record_count = len(payload)
        else:
            record_count = 1
    cur.execute(
        '''
        INSERT INTO market_data_raw (
            dataset, source, symbol, business_date, fetched_at_utc,
            record_count, payload_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            dataset,
            source,
            symbol,
            business_date,
            fetched_at_utc,
            int(record_count),
            _json_dumps(payload),
            _json_dumps(metadata) if metadata is not None else None,
        ),
    )
    raw_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return raw_id


def save_market_quotes(raw_id: int, quotes: Iterable[Dict[str, Any]]) -> int:
    """Persist normalized symbol-level quotes linked to a raw snapshot."""
    rows = []
    for quote in quotes:
        symbol = str(quote.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        rows.append(
            (
                int(raw_id),
                symbol,
                str(quote.get("security_id")) if quote.get("security_id") is not None else None,
                quote.get("security_name"),
                quote.get("last_traded_price"),
                quote.get("close_price"),
                quote.get("previous_close"),
                quote.get("percentage_change"),
                quote.get("total_trade_quantity"),
                str(quote.get("source") or ""),
                str(quote.get("fetched_at_utc") or ""),
            )
        )
    if not rows:
        return 0

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        '''
        INSERT OR REPLACE INTO market_quotes (
            raw_id, symbol, security_id, security_name, last_traded_price,
            close_price, previous_close, percentage_change,
            total_trade_quantity, source, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    count = len(rows)
    conn.commit()
    conn.close()
    return count


def load_latest_market_quotes(
    symbols: Iterable[str],
    *,
    max_age_seconds: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load the most recent normalized quote per symbol from SQLite."""
    clean_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not clean_symbols:
        return {}

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    results: Dict[str, Dict[str, Any]] = {}

    age_cutoff = None
    if max_age_seconds is not None:
        age_cutoff = (
            datetime.now(timezone.utc) - pd.Timedelta(seconds=int(max_age_seconds))
        ).replace(microsecond=0).isoformat()

    query = (
        '''
        SELECT symbol, security_id, security_name, last_traded_price, close_price,
               previous_close, percentage_change, total_trade_quantity, source,
               fetched_at_utc
        FROM market_quotes
        WHERE symbol = ?
        '''
        + (" AND fetched_at_utc >= ?" if age_cutoff is not None else "")
        + '''
        ORDER BY fetched_at_utc DESC, raw_id DESC
        LIMIT 1
        '''
    )

    for symbol in clean_symbols:
        params = (symbol, age_cutoff) if age_cutoff is not None else (symbol,)
        cur.execute(query, params)
        row = cur.fetchone()
        if row is None:
            continue
        results[symbol] = {
            "symbol": row[0],
            "security_id": row[1],
            "security_name": row[2],
            "last_traded_price": row[3],
            "close_price": row[4],
            "previous_close": row[5],
            "percentage_change": row[6],
            "total_trade_quantity": row[7],
            "source": row[8],
            "fetched_at_utc": row[9],
        }

    conn.close()
    return results


def save_benchmark_history(
    benchmark: str,
    rows: Iterable[Dict[str, Any]],
    *,
    source: str,
    fetched_at_utc: Optional[str] = None,
) -> int:
    """Persist daily benchmark index history."""
    benchmark = str(benchmark).strip().upper()
    if not benchmark:
        return 0
    fetched_at_utc = fetched_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = []
    for row in rows:
        date_value = row.get("date") or row.get("Date")
        if date_value is None:
            continue
        try:
            date_str = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        payload.append(
            (
                benchmark,
                date_str,
                row.get("open") if row.get("open") is not None else row.get("Open"),
                row.get("high") if row.get("high") is not None else row.get("High"),
                row.get("low") if row.get("low") is not None else row.get("Low"),
                row.get("close") if row.get("close") is not None else row.get("Close"),
                row.get("volume") if row.get("volume") is not None else row.get("Volume"),
                source,
                fetched_at_utc,
            )
        )
    if not payload:
        return 0

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        '''
        INSERT OR REPLACE INTO benchmark_index_history (
            benchmark, date, open, high, low, close, volume, source, fetched_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        payload,
    )
    conn.commit()
    conn.close()
    return len(payload)


def load_benchmark_history(
    benchmark: str,
    *,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
) -> pd.DataFrame:
    """Load benchmark history from local SQLite snapshots."""
    benchmark = str(benchmark).strip().upper()
    if not benchmark:
        return pd.DataFrame()

    init_db()
    conn = get_db_connection()
    query = '''
        SELECT date AS Date, open AS Open, high AS High, low AS Low, close AS Close,
               volume AS Volume, source AS Source, fetched_at_utc AS Fetched_At
        FROM benchmark_index_history
        WHERE benchmark = ?
    '''
    params: list[Any] = [benchmark]
    if start_date is not None:
        query += " AND date >= ?"
        params.append(pd.Timestamp(start_date).strftime("%Y-%m-%d"))
    if end_date is not None:
        query += " AND date <= ?"
        params.append(pd.Timestamp(end_date).strftime("%Y-%m-%d"))
    query += " ORDER BY date ASC"
    df = pd.read_sql(query, conn, params=params, parse_dates=["Date"])
    conn.close()
    return df
