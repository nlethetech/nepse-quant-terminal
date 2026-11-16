"""The Lookup tab from nepsequant.com: volume chart (no candlesticks), key
facts, related stocks, quarterly results, dividends, news, broker flow — and,
for the NEPSE index, the day's movers. Fake website replies only."""

from __future__ import annotations

import datetime as dt

import pytest

from apps.cli import connect
from apps.tui.io.website_lookup import SESSIONS, WebsiteLookup
from apps.tui.render import lookup_page as lp
from apps.tui.render.volume_chart import fmt_volume, render_volume_chart
from apps.tui.theme import GAIN, LOSS


def _rows(n=30, start=dt.date(2026, 9, 1), base=500.0):
    out, close = [], base
    for i in range(n):
        close = close * (1.01 if i % 3 else 0.98)
        out.append([(start + dt.timedelta(days=i)).isoformat(), close - 2, close + 3, close - 4, round(close, 1), 1000 * (i + 1)])
    return out


SECURITY = {
    "symbol": "NABIL", "kind": "security",
    "history": {"columns": ["date", "open", "high", "low", "close", "volume"], "rows": _rows(), "volume_unit": "shares"},
    "quote": {"name": "Nabil Bank Limited", "sector": "Commercial Banks", "ltp": 512.3, "change_pct": 2.47},
    "facts": {"market_cap": 1.38e11, "pe": 14.2, "pb": 1.88, "roe": 13.2, "eps": 35.99, "dividend_yield_pct": 3.46,
              "high_52w": 560.0, "low_52w": 440.0, "pct_from_52w_high": -8.5, "rsi": 55, "ret_1w": 1.2, "ret_1m": -2.0,
              "ret_3m": 5.6, "ret_1y": 10.0, "fundamentals_as_of": "2026-10-30"},
    "related": {"sector": "Commercial Banks", "sector_size": 20, "rows": [
        {"symbol": "BIGB", "ltp": 300.0, "change_pct": -1.0, "pe": 12.0, "market_cap": 2e11},
        {"symbol": "NABIL", "ltp": 512.3, "change_pct": 2.47, "pe": 14.2, "market_cap": 1.38e11, "is_subject": True},
        {"symbol": "SBL", "ltp": 400.0, "change_pct": 0.5, "pe": 11.0, "market_cap": 9e10}]},
    "quarters": [{"fiscal_year": "2082/83", "quarter": 1, "eps": 9.5, "net_profit": 1.2e9, "reported": "2025-11-10"}],
    "corporate_actions": [{"fiscal_year": "2081/82", "book_close": "2026-01-05", "cash_dividend_pct": 12.5}],
    "news": [{"published_at": "2026-11-14T05:00:00+00:00", "source": "Kantipur", "title": "Nabil profit up 20%"}],
    "filings": [{"published_at": "2026-11-15T04:00:00+00:00", "type": "dividend", "title": "Book close for dividend"}],
    "brokers": {"sessions": 20, "from": "2026-10-16", "to": "2026-11-13", "top5_buy_share_pct": 45.1, "brokers": 40,
                "buyers": [{"broker": 58, "name": "Naasa Securities", "net_qty": 12000, "net_amount": 6.1e6}],
                "sellers": [{"broker": 34, "name": "Vision Securities", "net_qty": -9000, "net_amount": -4.6e6}]},
    "parts": {"facts": "ok"},
}
INDEX = {
    "symbol": "NEPSE", "kind": "index",
    "history": {"columns": ["date", "open", "high", "low", "close", "volume"], "rows": _rows(base=2700.0),
                "volume_unit": "rupees"},
    "related": {"gainers": [{"symbol": "G1", "ltp": 110.0, "change_pct": 9.95, "volume": 5000,
                             "news": [{"published_at": "2026-11-15T03:00:00+00:00", "source": "Karobar", "title": "G1 wins licence"}]}],
                "losers": [{"symbol": "L1", "ltp": 50.0, "change_pct": -7.5, "volume": 3000}],
                "most_active": [{"symbol": "G1", "ltp": 110.0, "change_pct": 9.95, "volume": 5000}]},
}


def _plain(text) -> str:
    return text.plain if hasattr(text, "plain") else str(text)


# ── the volume chart ─────────────────────────────────────────────────────────
def test_volume_chart_has_bars_axis_average_and_dates():
    d = [f"2026-11-{i:02d}" for i in range(1, 11)]
    closes = [100, 101, 99, 99, 102, 103, 101, 104, 105, 103]
    vols = [1e5 * (i + 1) for i in range(10)]
    t = render_volume_chart(d, closes, vols, width=60, height=10)
    lines = t.plain.split("\n")
    assert len(lines) == 10 and lines[0].startswith("  VOLUME 10 sessions")
    assert "1.00M" in lines[1] and "┤" in lines[1]  # the top of the scale is the biggest day
    assert "┈" in t.plain  # the average line
    assert lines[-1].strip().startswith("2026-11-01") and lines[-1].rstrip().endswith("2026-11-10")
    styles = {str(span.style) for span in t.spans}
    assert any(GAIN in s for s in styles) and any(LOSS in s for s in styles)  # up and down days


def test_volume_chart_keeps_the_newest_sessions_when_narrow_and_handles_rupees():
    d = [f"d{i:03d}" for i in range(300)]
    t = render_volume_chart(d, [100.0] * 300, [5e9] * 300, width=80, height=8, unit="rupees")
    assert "Rs 5.00 arba" in t.plain and "d299" in t.plain and "d000" not in t.plain
    assert all(len(line) <= 80 for line in t.plain.split("\n")[1:])
    assert fmt_volume(7_840_000) == "7.84M" and fmt_volume(812) == "812"
    assert fmt_volume(15.1e9, "rupees") == "Rs 15.1 arba" and fmt_volume(4.6e7, "rupees") == "Rs 4.60 cr"
    assert render_volume_chart([], [], [], width=60, height=8).plain.strip() == "No volume history yet."


# ── the page ─────────────────────────────────────────────────────────────────
def test_a_stock_page_from_the_website():
    p = lp.build_payload("NABIL", "D", SECURITY, width=120, height=14)
    assert p["found"] and p["source"] == "nepsequant.com"
    assert "Nabil Bank Limited · Commercial Banks" in p["title"] and "(30 sessions)" in p["title"]
    assert "VOLUME 30 sessions" in _plain(p["chart"]) and "candle" not in _plain(p["chart"]).lower()
    facts = _plain(p["stats"])
    assert "Rs 138.00 arba" in facts and "P/E 14.2" in facts and "ROE 13.2%" in facts and "52W 440.0 – 560.0" in facts
    assert p["related_syms"] == ["BIGB", "NABIL", "SBL"] and "Commercial Banks" in p["related_title"]
    assert _plain(p["related_rows"][1][0]).startswith("▸ NABIL")  # the stock itself, marked
    assert _plain(p["fin_rows"][0][0]) == "2082/83 Q1" and _plain(p["fin_rows"][0][2]) == "Rs 1.20 arba"
    assert _plain(p["ca_rows"][0][2]) == "12.50%"
    news = _plain(p["intel"]).split("\n")
    assert "FILING DIVIDEND" in news[0] and "Book close for dividend" in news[1]  # newest first, filings included
    assert "Kantipur" in news[2] and news[3].startswith("    Nabil profit up 20%")
    brokers = _plain(p["report"])
    assert "Naasa Securities" in brokers and "+12,000 sh" in brokers and "-9,000 sh" in brokers
    assert _plain(p["ohlcv_rows"][0][0]) == "2026-09-30"  # newest first
    assert "CORPORATE INTELLIGENCE" not in str(p) and "Pipeline" not in str(p)


def test_weekly_view_and_missing_parts():
    bare = {"symbol": "XYZ", "kind": "security", "history": SECURITY["history"], "quote": {}, "facts": {},
            "related": {"sector": None, "rows": []}, "quarters": [], "corporate_actions": [], "news": [], "filings": [],
            "parts": {"facts": "warming"}}
    p = lp.build_payload("XYZ", "W", bare, width=100, height=12)
    assert "Weekly (5 weeks)" in p["title"] and p["ohlcv_cols"][0][0] == "WEEK"
    assert "being prepared" in _plain(p["stats"])
    assert "No broker flow" in _plain(p["report"]) and "No recent news" in _plain(p["intel"])
    assert p["ca_title"].endswith("none recorded") and p["related_syms"] == [""]


def test_the_index_page_shows_what_moved_the_market():
    p = lp.build_payload("NEPSE", "D", INDEX, width=120, height=14)
    assert "NEPSE INDEX" in _plain(p["stats"]) and "Rs " in _plain(p["chart"])
    assert p["related_syms"] == ["G1", "L1"] and "WHAT MOVED THE MARKET" in p["related_title"]
    assert "G1 wins licence" in _plain(p["report"])
    assert p["fin_title"] == "MOST ACTIVE TODAY" and p["ca_title"] == "" and p["intel_title"] == ""


def test_rupees_read_the_nepse_way():
    assert lp.rupees(1.38e11) == "Rs 138.00 arba"
    assert lp.rupees(4.6e7) == "Rs 4.60 cr"
    assert lp.rupees(-4.6e6) == "-Rs 46.00 lakh"
    assert lp.rupees(None) == "—"


# ── the client ───────────────────────────────────────────────────────────────
def test_client_asks_once_per_stock_and_view_and_keeps_it_five_minutes():
    seen = []
    now = [1000.0]

    def fake(path, key, **kw):
        seen.append(path)
        return {"symbol": "NABIL"}

    c = WebsiteLookup("nqk_x", request=fake, clock=lambda: now[0])
    c.get("nabil")
    c.get("NABIL")
    c.get("NABIL", timeframe="W")
    assert seen == [f"/api/cli/lookup/NABIL?sessions={SESSIONS['D']}", f"/api/cli/lookup/NABIL?sessions={SESSIONS['W']}"]
    now[0] += 301
    c.get("NABIL")
    assert len(seen) == 3


def test_client_passes_the_websites_refusal_through_and_needs_a_saved_key():
    def refuse(path, key, **kw):
        raise connect.ConnectError("ZZZ is not a listed NEPSE symbol.")

    with pytest.raises(connect.ConnectError, match="not a listed"):
        WebsiteLookup("nqk_x", request=refuse).get("ZZZ")
    assert WebsiteLookup.from_saved() is None  # the test home has no key
