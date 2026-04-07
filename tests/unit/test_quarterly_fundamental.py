from __future__ import annotations

import pandas as pd

from backend.quant_pro.quarterly_fundamental import QuarterlyFundamentalModel


def test_build_fundamentals_respects_point_in_time_announcement_dates():
    quarterly = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "fiscal_year": "082-083",
                "quarter": 1,
                "eps": 10.0,
                "net_profit": 20.0,
                "revenue": 100.0,
                "book_value": 100.0,
                "announcement_date": "2026-01-15",
                "report_date": "2026-01-15",
                "scraped_at_utc": "2026-01-16T00:00:00+00:00",
            },
            {
                "symbol": "AAA",
                "fiscal_year": "082-083",
                "quarter": 2,
                "eps": 14.0,
                "net_profit": 28.0,
                "revenue": 120.0,
                "book_value": 110.0,
                "announcement_date": "2026-04-15",
                "report_date": "2026-04-15",
                "scraped_at_utc": "2026-04-16T00:00:00+00:00",
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": "2026-01-20",
                "pe_ratio": 15.0,
                "pb_ratio": 1.5,
                "roe": 0.18,
                "dividend_yield": 0.06,
                "sector": "Commercial Banks",
            }
        ]
    )

    model = QuarterlyFundamentalModel.from_frames(quarterly, fundamentals)

    early = model.build_fundamentals(
        pd.Timestamp("2026-03-01"),
        {"AAA": 400.0},
        sector_lookup=lambda _symbol: "Commercial Banks",
        symbols=["AAA"],
    )["AAA"]
    assert round(early.eps, 2) == 40.0
    assert round(early.pe_ratio, 2) == 10.0
    assert early.eps_growth_qoq is None

    later = model.build_fundamentals(
        pd.Timestamp("2026-05-01"),
        {"AAA": 400.0},
        sector_lookup=lambda _symbol: "Commercial Banks",
        symbols=["AAA"],
    )["AAA"]
    assert round(later.eps, 2) == 28.0
    assert round(later.eps_growth_qoq, 2) == 0.40
    assert round(later.revenue_growth_qoq, 2) == 0.20
    assert round(later.book_value_growth_qoq, 2) == 0.10


def test_generate_signals_emits_quarterly_fundamental_signal():
    quarterly = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "fiscal_year": "082-083",
                "quarter": 1,
                "eps": 12.0,
                "net_profit": 24.0,
                "revenue": 120.0,
                "book_value": 150.0,
                "announcement_date": "2026-01-15",
                "report_date": "2026-01-15",
                "scraped_at_utc": "2026-01-16T00:00:00+00:00",
            },
            {
                "symbol": "AAA",
                "fiscal_year": "082-083",
                "quarter": 2,
                "eps": 16.0,
                "net_profit": 32.0,
                "revenue": 144.0,
                "book_value": 165.0,
                "announcement_date": "2026-04-15",
                "report_date": "2026-04-15",
                "scraped_at_utc": "2026-04-16T00:00:00+00:00",
            },
            {
                "symbol": "PEER",
                "fiscal_year": "082-083",
                "quarter": 2,
                "eps": 10.0,
                "net_profit": 18.0,
                "revenue": 110.0,
                "book_value": 120.0,
                "announcement_date": "2026-04-15",
                "report_date": "2026-04-15",
                "scraped_at_utc": "2026-04-16T00:00:00+00:00",
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "date": "2026-04-20",
                "pe_ratio": 14.0,
                "pb_ratio": 1.8,
                "roe": 0.18,
                "dividend_yield": 0.06,
                "sector": "Commercial Banks",
            },
            {
                "symbol": "PEER",
                "date": "2026-04-20",
                "pe_ratio": 20.0,
                "pb_ratio": 2.2,
                "roe": 0.10,
                "dividend_yield": 0.02,
                "sector": "Commercial Banks",
            },
        ]
    )

    model = QuarterlyFundamentalModel.from_frames(quarterly, fundamentals)
    signals = model.generate_signals(
        pd.Timestamp("2026-05-01"),
        {"AAA": 420.0, "PEER": 620.0},
        sector_lookup=lambda _symbol: "Commercial Banks",
        symbols=["AAA", "PEER"],
    )

    symbols = [signal.symbol for signal in signals]
    assert "AAA" in symbols
    aaa = next(signal for signal in signals if signal.symbol == "AAA")
    assert aaa.signal_type.value == "quarterly_fundamental"
    assert "EPS QoQ" in aaa.reasoning
