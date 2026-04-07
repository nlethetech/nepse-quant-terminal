"""Unit tests for database module."""

import sqlite3
import tempfile
import os
import pytest
import pandas as pd

from backend.quant_pro.database import (
    get_db_connection,
    get_db_path,
    init_db,
    load_latest_market_quotes,
    save_market_data_raw,
    save_market_quotes,
)
from backend.quant_pro.exceptions import DatabaseError


class TestGetDbPath:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("NEPSE_DB_FILE", raising=False)
        path = get_db_path()
        assert path.name == "nepse_market_data.db"
        assert path.is_absolute()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEPSE_DB_FILE", "/tmp/test_custom.db")
        path = get_db_path()
        assert path.name == "test_custom.db"


class TestInitDb:
    def test_creates_tables(self, tmp_path, monkeypatch):
        db_file = tmp_path / "test.db"
        monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

        # Reset WAL flag
        import backend.quant_pro.database as db_mod
        db_mod._wal_initialized = False

        init_db()

        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()

        # Check stock_prices table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_prices'")
        assert cur.fetchone() is not None

        # Check corporate_actions table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='corporate_actions'")
        assert cur.fetchone() is not None

        # Check intraday audit tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_raw'")
        assert cur.fetchone() is not None
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_quotes'")
        assert cur.fetchone() is not None
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news'")
        assert cur.fetchone() is not None
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news_event_scores'")
        assert cur.fetchone() is not None

        # Check WAL mode
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"

        conn.close()


class TestSaveLoadDb:
    def test_round_trip(self, tmp_path, monkeypatch):
        db_file = tmp_path / "test.db"
        monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

        import backend.quant_pro.database as db_mod
        db_mod._wal_initialized = False

        init_db()

        from backend.quant_pro.database import save_to_db, load_from_db

        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-07", "2024-01-08"]),
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [103.0, 104.0],
            "Volume": [1000.0, 1200.0],
        })
        save_to_db(df, "TEST")

        loaded = load_from_db("TEST")
        assert not loaded.empty
        assert len(loaded) == 2
        assert loaded["Close"].iloc[0] == 103.0

    def test_empty_df_noop(self, tmp_path, monkeypatch):
        db_file = tmp_path / "test.db"
        monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

        import backend.quant_pro.database as db_mod
        db_mod._wal_initialized = False
        init_db()

        from backend.quant_pro.database import save_to_db, load_from_db

        save_to_db(pd.DataFrame(), "EMPTY")
        loaded = load_from_db("EMPTY")
        assert loaded.empty

    def test_market_quote_round_trip(self, tmp_path, monkeypatch):
        db_file = tmp_path / "test.db"
        monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

        import backend.quant_pro.database as db_mod
        db_mod._wal_initialized = False
        init_db()

        raw_id = save_market_data_raw(
            dataset="intraday_snapshot",
            source="nepalstock",
            payload=[{"symbol": "ADBL"}],
            fetched_at_utc="2026-03-26T00:00:00+00:00",
        )
        rows_saved = save_market_quotes(
            raw_id,
            [
                {
                    "symbol": "ADBL",
                    "security_id": "397",
                    "security_name": "Agricultural Development Bank Limited",
                    "last_traded_price": 333.1,
                    "close_price": 333.1,
                    "previous_close": 330.0,
                    "percentage_change": 0.94,
                    "total_trade_quantity": 85296,
                    "source": "nepalstock",
                    "fetched_at_utc": "2026-03-26T00:00:00+00:00",
                }
            ],
        )
        assert rows_saved == 1

        latest = load_latest_market_quotes(["ADBL"])
        assert latest["ADBL"]["last_traded_price"] == 333.1
        assert latest["ADBL"]["source"] == "nepalstock"
