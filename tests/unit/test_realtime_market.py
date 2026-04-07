"""Unit tests for Nepalstock-first intraday market routing."""

from __future__ import annotations

from backend.quant_pro.database import init_db, load_latest_market_quotes
from backend.quant_pro.realtime_market import RealtimeMarketDataProvider


class FakeNepseClient:
    def getMarketStatus(self):
        return {"isOpen": "OPEN", "asOf": "2026-03-26T11:15:00"}

    def getLiveMarket(self):
        return [
            {
                "symbol": "ADBL",
                "securityId": 397,
                "securityName": "Agricultural Development Bank Limited",
                "lastTradedPrice": 333.1,
                "closePrice": 333.1,
                "previousClose": 330.0,
                "percentageChange": 0.939393939,
                "totalTradeQuantity": 85296,
            }
        ]

    def getPriceVolume(self):
        return []


class FakePriceVolumeOnlyClient:
    def getMarketStatus(self):
        return {"isOpen": "CLOSE", "asOf": "2026-03-26T15:00:00"}

    def getLiveMarket(self):
        return []

    def getPriceVolume(self):
        return [
            {
                "symbol": "NABIL",
                "securityId": 131,
                "securityName": "Nabil Bank Limited",
                "lastTradedPrice": 500.0,
                "previousClose": 490.0,
                "closePrice": 500.0,
                "totalTradeQuantity": 1000,
            }
        ]


def test_fetch_snapshot_persists_quotes(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

    import backend.quant_pro.database as db_mod

    db_mod._wal_initialized = False
    init_db()

    provider = RealtimeMarketDataProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: FakeNepseClient())

    snapshot = provider.fetch_snapshot(force=True)

    assert snapshot.endpoint == "live_market"
    assert snapshot.quotes["ADBL"]["last_traded_price"] == 333.1

    latest = load_latest_market_quotes(["ADBL"])
    assert latest["ADBL"]["last_traded_price"] == 333.1
    assert latest["ADBL"]["security_id"] == "397"


def test_get_latest_ltps_uses_fallbacks(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("NEPSE_DB_FILE", str(db_file))

    import backend.quant_pro.database as db_mod

    db_mod._wal_initialized = False
    init_db()

    provider = RealtimeMarketDataProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: FakePriceVolumeOnlyClient())
    monkeypatch.setattr(
        provider,
        "_fetch_merolagani_ltps",
        lambda symbols: {symbol: 612.3 for symbol in symbols},
    )

    prices = provider.get_latest_ltps(["NABIL", "NICA"], force_refresh=True)

    assert prices["NABIL"] == 500.0
    assert prices["NICA"] == 612.3
    batch_info = provider.get_last_batch_info()
    assert batch_info["source_map"]["NABIL"] == "nepalstock"
    assert batch_info["source_map"]["NICA"] == "merolagani"
    assert batch_info["timestamp_map"]["NABIL"]
    assert "NICA" in batch_info["timestamp_map"]
