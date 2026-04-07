from __future__ import annotations

import json
import sqlite3

from scripts.signals.generate_daily_signals import load_fundamentals
from backend.quant_pro.alpha_practical import FundamentalData, FundamentalScanner


def _prepare_fundamental_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE fundamentals (
            symbol TEXT,
            pe_ratio REAL,
            pb_ratio REAL,
            dividend_yield REAL,
            eps REAL,
            book_value_per_share REAL,
            roe REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE quarterly_earnings (
            symbol TEXT,
            fiscal_year TEXT,
            quarter INTEGER,
            eps REAL,
            net_profit REAL,
            revenue REAL,
            book_value REAL,
            announcement_date TEXT,
            report_date TEXT
        )
        """
    )
    conn.commit()


def test_load_fundamentals_merges_quarterly_reports_and_db(tmp_path, monkeypatch):
    cache_dir = tmp_path / "quarterly_reports"
    cache_dir.mkdir()

    payload = {
        "symbol": "NABIL",
        "reports": [
            {
                "date": "2026-02-01",
                "financials": {
                    "sector": "banking",
                    "fiscal_year": "2082/83",
                    "quarter": "Q2",
                    "balance_sheet": {
                        "shareholders_equity": 300.0,
                        "total_liabilities": 700.0,
                    },
                    "income_statement": {
                        "total_revenue": 610.0,
                        "net_profit": 155.0,
                    },
                    "per_share": {
                        "eps": 12.0,
                        "book_value": 155.0,
                    },
                    "ratios": {
                        "npl_pct": 1.7,
                        "capital_adequacy_pct": 14.3,
                        "cost_income_ratio": 42.0,
                    },
                },
            }
        ],
    }
    (cache_dir / "NABIL.json").write_text(json.dumps(payload))
    monkeypatch.setattr("scripts.signals.generate_daily_signals.QUARTERLY_REPORTS_DIR", cache_dir)

    conn = sqlite3.connect(":memory:")
    _prepare_fundamental_db(conn)
    conn.execute(
        """
        INSERT INTO fundamentals (
            symbol, pe_ratio, pb_ratio, dividend_yield, eps, book_value_per_share, roe
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("NABIL", 18.8, 2.0, 0.03, 20.0, 240.0, 0.18),
    )
    conn.executemany(
        """
        INSERT INTO quarterly_earnings (
            symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
            announcement_date, report_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("NABIL", "082-083", 2, 12.0, 150.0, 600.0, 160.0, "2026-02-01", "2026-02-01"),
            ("NABIL", "082-083", 1, 9.0, 110.0, 520.0, 150.0, "2025-11-10", "2025-11-10"),
        ],
    )
    conn.commit()

    data = load_fundamentals(conn, ["NABIL"], latest_prices={"NABIL": 420.0})
    conn.close()

    fd = data["NABIL"]
    assert round(fd.pe_ratio, 2) == 17.50
    assert round(fd.pb_ratio, 2) == round(420.0 / 155.0, 2)
    assert round(fd.eps_growth_qoq, 4) == round((12.0 - 9.0) / 9.0, 4)
    assert round(fd.revenue_growth_qoq, 4) == round((600.0 - 520.0) / 520.0, 4)
    assert fd.npl_pct == 1.7
    assert fd.capital_adequacy_pct == 14.3
    assert "quarterly_earnings" in (fd.data_source or "")
    assert "quarterly_cache" in (fd.data_source or "")


def test_fundamental_scanner_rewards_growth_and_filters_risks():
    scanner = FundamentalScanner()
    scanner.update_fundamentals(
        FundamentalData(
            symbol="PEER",
            sector="Commercial Banks",
            pe_ratio=18.0,
            pb_ratio=1.8,
            roe=0.10,
            latest_net_profit=50_000_000,
        )
    )
    scanner.update_fundamentals(
        FundamentalData(
            symbol="NABIL",
            sector="Commercial Banks",
            pe_ratio=11.0,
            pb_ratio=0.95,
            dividend_yield=0.06,
            roe=0.18,
            revenue_growth_qoq=0.12,
            profit_growth_qoq=0.20,
            eps_growth_qoq=0.25,
            capital_adequacy_pct=14.0,
            npl_pct=1.8,
            cost_income_ratio=45.0,
            latest_net_profit=150_000_000,
            data_source="quarterly_earnings+quarterly_cache",
        )
    )

    signals = scanner.scan()
    assert "NABIL" in [signal.symbol for signal in signals]
    nabil_signal = next(signal for signal in signals if signal.symbol == "NABIL")
    assert nabil_signal.confidence >= 0.75
    assert "EPS QoQ" in nabil_signal.reasoning
    assert "CAR 14.0%" in nabil_signal.reasoning

    risk_scanner = FundamentalScanner()
    risk_scanner.update_fundamentals(
        FundamentalData(
            symbol="PEER",
            sector="Commercial Banks",
            pe_ratio=18.0,
            pb_ratio=1.8,
            roe=0.10,
            latest_net_profit=50_000_000,
        )
    )
    risk_scanner.update_fundamentals(
        FundamentalData(
            symbol="RISK",
            sector="Commercial Banks",
            pe_ratio=8.0,
            pb_ratio=0.8,
            roe=0.08,
            capital_adequacy_pct=6.5,
            npl_pct=8.0,
            latest_net_profit=-10_000_000,
        )
    )

    risk_symbols = [signal.symbol for signal in risk_scanner.scan()]
    assert "RISK" not in risk_symbols
