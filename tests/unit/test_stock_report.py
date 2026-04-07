"""Unit tests for deterministic stock report builder."""

from __future__ import annotations

import json
import sqlite3

from backend.quant_pro.stock_report import build_stock_report


def _prepare_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE fundamentals (
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
        """
    )
    conn.execute(
        """
        CREATE TABLE quarterly_earnings (
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
        """
    )
    conn.commit()
    conn.close()


def test_build_stock_report_prefers_cached_financials(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    cache_dir = tmp_path / "quarterly_reports"
    cache_dir.mkdir()
    _prepare_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO fundamentals (
            symbol, date, pe_ratio, pb_ratio, eps, book_value_per_share, roe, debt_to_equity, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("NABIL", "2026-01-09", 18.0, 2.0, 25.0, 240.0, 15.0, 7.5, "Banking"),
    )
    conn.commit()
    conn.close()

    cached = {
        "symbol": "NABIL",
        "reports": [
            {
                "date": "2026-02-01",
                "financials": {
                    "sector": "banking",
                    "fiscal_year": "2082/2083",
                    "quarter": "Q2",
                    "balance_sheet": {
                        "total_assets": 1000,
                        "total_liabilities": 700,
                        "shareholders_equity": 300,
                        "retained_earnings": 40,
                        "total_deposits": 800,
                        "total_loans": 640,
                    },
                    "income_statement": {
                        "total_revenue": 500,
                        "operating_profit": 180,
                        "net_profit": 120,
                    },
                    "per_share": {
                        "eps": 12,
                        "book_value": 150,
                    },
                    "ratios": {
                        "npl_pct": 1.8,
                        "capital_adequacy_pct": 14.2,
                    },
                    "notes": "Strong quarter with good asset quality.",
                },
            },
            {
                "date": "2025-11-10",
                "financials": {
                    "sector": "banking",
                    "fiscal_year": "2082/2083",
                    "quarter": "Q1",
                    "balance_sheet": {
                        "total_assets": 950,
                        "total_liabilities": 680,
                        "shareholders_equity": 270,
                        "retained_earnings": 20,
                        "total_deposits": 770,
                        "total_loans": 600,
                    },
                    "income_statement": {
                        "total_revenue": 420,
                        "operating_profit": 140,
                        "net_profit": 90,
                    },
                    "per_share": {
                        "eps": 9,
                        "book_value": 140,
                    },
                    "ratios": {
                        "npl_pct": 2.1,
                        "capital_adequacy_pct": 13.7,
                    },
                    "notes": "Prior quarter.",
                },
            },
        ],
    }
    (cache_dir / "NABIL.json").write_text(json.dumps(cached))

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", cache_dir)

    report = build_stock_report("NABIL", current_price=420.0)

    assert report["has_data"] is True
    assert report["signal"] == "ACCUMULATE"
    assert report["financial_rows"][0]["period"] == "2082/83 Q2"
    assert any(label == "P/E" and value == "17.5" for label, value in report["snapshot"])
    assert "score" in report["summary"]
    assert report["latest_notes"] == "Strong quarter with good asset quality."


def test_build_stock_report_falls_back_to_quarterly_earnings_db(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    _prepare_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO quarterly_earnings (
            symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
            announcement_date, report_date, source, scraped_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("UPPER", "082-083", 2, 14.0, 400000000.0, 800000000.0, 120.0, "2026-02-13", None, "sharesansar", "2026-02-13T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO quarterly_earnings (
            symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
            announcement_date, report_date, source, scraped_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("UPPER", "082-083", 1, 10.0, 300000000.0, 650000000.0, 112.0, "2025-11-10", None, "sharesansar", "2025-11-10T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    empty_cache = tmp_path / "quarterly_reports"
    empty_cache.mkdir()

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", empty_cache)

    report = build_stock_report("UPPER", current_price=300.0)

    assert report["has_data"] is True
    assert report["signal"] in {"WATCH", "ACCUMULATE"}
    assert report["financial_rows"][0]["period"] == "082/83 Q2"
    assert any(label == "Revenue QoQ" and value.startswith("+") for label, value in report["snapshot"])


def test_build_stock_report_skips_low_confidence_unified_ocr_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    _prepare_db(db_path)

    quarterly_cache = tmp_path / "quarterly_reports"
    quarterly_cache.mkdir()
    unified_cache = tmp_path / "financial_reports"
    unified_cache.mkdir()

    cached = {
        "symbol": "NABIL",
        "reports": [
            {
                "report_type": "quarterly",
                "announcement_date": "2026-02-01",
                "extraction_status": "local_review",
                "extracted_json": {
                    "sector": "banking",
                    "fiscal_year": "082-083",
                    "quarter": "Q2",
                    "income_statement": {"total_revenue": 999999999, "net_profit": 1},
                    "per_share": {"eps": 9999, "book_value": 1},
                    "quality": {
                        "confidence": 0.42,
                        "review_flags": ["missing_core_field:per_share.book_value"],
                    },
                },
            }
        ],
    }
    (unified_cache / "NABIL.json").write_text(json.dumps(cached))

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", quarterly_cache)
    monkeypatch.setattr("backend.quant_pro.stock_report.FINANCIAL_REPORTS_DIR", unified_cache)

    report = build_stock_report("NABIL", current_price=420.0)

    assert report["has_data"] is False


def test_build_stock_report_uses_stored_fundamentals_without_quarter_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    _prepare_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO fundamentals (
            symbol, date, pe_ratio, pb_ratio, eps, book_value_per_share, roe, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("RURU", "2026-02-13", 35.34, 4.52, 18.59, 145.43, 12.8, "Hydro Power"),
    )
    conn.commit()
    conn.close()

    empty_quarterly_cache = tmp_path / "quarterly_reports"
    empty_quarterly_cache.mkdir()
    empty_unified_cache = tmp_path / "financial_reports"
    empty_unified_cache.mkdir()

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", empty_quarterly_cache)
    monkeypatch.setattr("backend.quant_pro.stock_report.FINANCIAL_REPORTS_DIR", empty_unified_cache)

    report = build_stock_report("RURU")
    snapshot = dict(report["snapshot"])

    assert report["has_data"] is True
    assert report["signal"] in {"WATCH", "ACCUMULATE"}
    assert snapshot["Book Value"] == "145.43"
    assert snapshot["P/E"] == "35.3"
    assert snapshot["P/BV"] == "4.52x"
    assert report["summary"].startswith("RURU screens")
    assert report["company_name"] == "Ru Ru Jalbidhyut Pariyojana Limited"
    assert report["company_profile"]["board"][0]["name"] == "Ashish Subedi"
    assert report["company_profile"]["board"][0]["role"] == "Chairman"
    assert report["company_profile"]["officers"][0]["name"] == "Kishor Prasad Ghimire"
    assert report["intelligence"]["headline"].startswith("Upper Hugdi Khola")
    assert report["intelligence"]["sections"][0]["title"] == "Asset"
    assert ("Offtaker", "Nepal Electricity Authority") in report["intelligence"]["sections"][2]["rows"]
    assert report["intelligence"]["sections"][4]["title"] == "Recent Watch"
    assert "Jan 1, 2026" in report["intelligence"]["sections"][4]["bullets"][0]


def test_build_stock_report_includes_rsml_board_override(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    _prepare_db(db_path)

    empty_quarterly_cache = tmp_path / "quarterly_reports"
    empty_quarterly_cache.mkdir()
    empty_unified_cache = tmp_path / "financial_reports"
    empty_unified_cache.mkdir()

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", empty_quarterly_cache)
    monkeypatch.setattr("backend.quant_pro.stock_report.FINANCIAL_REPORTS_DIR", empty_unified_cache)

    report = build_stock_report("RSML")

    assert report["company_name"] == "Reliance Spinning Mills Limited"
    assert report["company_profile"]["board"][0]["name"] == "Pawan Kumar Golyan"
    assert report["company_profile"]["board"][0]["role"] == "Chairman"
    assert report["company_profile"]["board"][2]["name"] == "Akshay Golyan"
    assert report["company_profile"]["board"][2]["role"] == "Managing Director"
    assert report["intelligence"]["headline"].startswith("Reliance Spinning Mills")
    assert report["intelligence"]["sections"][0]["title"] == "Asset"
    assert ("Primary Export Markets", "India, Turkey") in report["intelligence"]["sections"][2]["rows"]
    assert report["intelligence"]["sections"][4]["title"] == "Recent Watch"
    assert "Apr 2, 2026" in report["intelligence"]["sections"][4]["bullets"][2]


def test_build_stock_report_uses_override_financials_when_db_tables_are_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "report.db"
    sqlite3.connect(str(db_path)).close()

    empty_quarterly_cache = tmp_path / "quarterly_reports"
    empty_quarterly_cache.mkdir()
    empty_unified_cache = tmp_path / "financial_reports"
    empty_unified_cache.mkdir()

    monkeypatch.setenv("NEPSE_DB_FILE", str(db_path))
    monkeypatch.setattr("backend.quant_pro.stock_report.QUARTERLY_REPORTS_DIR", empty_quarterly_cache)
    monkeypatch.setattr("backend.quant_pro.stock_report.FINANCIAL_REPORTS_DIR", empty_unified_cache)

    report = build_stock_report("RURU", current_price=654.0)

    assert report["has_data"] is True
    assert report["financial_rows"]
    assert report["financial_rows"][0]["period"] == "2082/83 Q2"
    assert report["summary"].startswith("RURU screens")
