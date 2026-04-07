"""Unit tests for earnings scraper persistence helpers."""

from __future__ import annotations

import sqlite3

from backend.quant_pro.earnings_scraper import (
    _parse_sharesansar_financial_report_item,
    create_fundamentals_table,
    scrape_symbol_earnings,
    upsert_fundamentals_snapshots,
)


def test_upsert_fundamentals_snapshots_persists_merolagani_metrics(tmp_path):
    db_path = tmp_path / "earnings.db"
    create_fundamentals_table(str(db_path))

    rows = [
        {
            "symbol": "RURU",
            "eps": 18.59,
            "book_value": 145.43,
            "source": "merolagani",
        },
        {
            "symbol": "RURU",
            "pe_ratio": 35.34,
            "pb_ratio": 4.52,
            "market_cap": 3727306873.71,
            "shares_outstanding": 5673222.03,
            "dividend_yield": 5.79,
            "sector": "Hydro Power",
            "source": "merolagani",
        },
    ]

    inserted = upsert_fundamentals_snapshots(
        str(db_path),
        rows,
        as_of_date="2026-02-13",
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT symbol, date, pe_ratio, pb_ratio, eps, book_value_per_share,
               market_cap, shares_outstanding, dividend_yield, sector
        FROM fundamentals
        WHERE symbol = ?
        """,
        ("RURU",),
    ).fetchone()
    conn.close()

    assert inserted == 1
    assert row is not None
    assert row["date"] == "2026-02-13"
    assert row["pe_ratio"] == 35.34
    assert row["pb_ratio"] == 4.52
    assert row["eps"] == 18.59
    assert row["book_value_per_share"] == 145.43
    assert row["shares_outstanding"] == 5673222.03
    assert row["dividend_yield"] == 5.79
    assert row["sector"] == "Hydro Power"


def test_parse_sharesansar_financial_report_item_parses_profit_and_loss_titles():
    profit = _parse_sharesansar_financial_report_item(
        {
            "title": "<a href='#'>Sonapur Minerals and Oil Limited has posted a net profit of Rs 47.50 million and published its 2nd quarter company analysis of the fiscal year 2082/83.</a>",
            "published_date": "2026-02-12",
        },
        symbol="SONA",
        scraped_at_utc="2026-04-05T00:00:00+00:00",
    )
    loss = _parse_sharesansar_financial_report_item(
        {
            "title": "<a href='#'>Sonapur Minerals and Oil Limited has posted a net loss of Rs 75.93 million and published its 3rd quarter company analysis of the fiscal year 2081/82.</a>",
            "published_date": "2025-05-13",
        },
        symbol="SONA",
        scraped_at_utc="2026-04-05T00:00:00+00:00",
    )

    assert profit is not None
    assert profit["symbol"] == "SONA"
    assert profit["fiscal_year"] == "082-083"
    assert profit["quarter"] == 2
    assert profit["net_profit"] == 47_500_000.0
    assert profit["announcement_date"] == "2026-02-12"

    assert loss is not None
    assert loss["fiscal_year"] == "081-082"
    assert loss["quarter"] == 3
    assert loss["net_profit"] == -75_930_000.0


def test_scrape_symbol_earnings_prefers_sharesansar_financial_report_profit(monkeypatch):
    monkeypatch.setattr(
        "backend.quant_pro.earnings_scraper._get_sharesansar_company_info",
        lambda session, symbol: {"id": "1193", "symbol": "SONA", "sector": "Manufacturing", "csrf_token": "tok"},
    )
    monkeypatch.setattr(
        "backend.quant_pro.earnings_scraper.scrape_sharesansar_quarterly",
        lambda session, company_info: {
            "symbol": "SONA",
            "fiscal_year": "082-083",
            "quarter": 2,
            "eps": 3.09,
            "net_profit": 2.81,
            "book_value": 191.42,
            "sector": "Manufacturing",
        },
    )
    monkeypatch.setattr(
        "backend.quant_pro.earnings_scraper.scrape_sharesansar_financial_report_history",
        lambda session, company_info: [
            {
                "symbol": "SONA",
                "fiscal_year": "082-083",
                "quarter": 2,
                "net_profit": 47_500_000.0,
                "announcement_date": "2026-02-12",
                "report_date": "2026-02-12",
                "source": "sharesansar_financial_reports",
            },
            {
                "symbol": "SONA",
                "fiscal_year": "082-083",
                "quarter": 1,
                "net_profit": 43_680_000.0,
                "announcement_date": "2025-11-16",
                "report_date": "2025-11-16",
                "source": "sharesansar_financial_reports",
            },
        ],
    )
    monkeypatch.setattr("backend.quant_pro.earnings_scraper.scrape_merolagani_eps", lambda session, symbol: None)

    rows = scrape_symbol_earnings("SONA")
    keyed = {(row["fiscal_year"], row["quarter"]): row for row in rows}

    assert keyed[("082-083", 2)]["eps"] == 3.09
    assert keyed[("082-083", 2)]["book_value"] == 191.42
    assert keyed[("082-083", 2)]["net_profit"] == 47_500_000.0
    assert keyed[("082-083", 2)]["announcement_date"] == "2026-02-12"
    assert keyed[("082-083", 1)]["net_profit"] == 43_680_000.0
