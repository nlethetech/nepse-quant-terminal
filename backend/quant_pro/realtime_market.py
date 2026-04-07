"""Primary intraday market data provider with Nepalstock-first routing.

This module centralizes intraday access so the trading system reads from a
cached market-wide snapshot instead of scraping one symbol at a time.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup

from .database import load_latest_market_quotes, save_market_data_raw, save_market_quotes
from .institutional import utc_now_iso


logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


INTRADAY_CACHE_SECS = int(os.environ.get("NEPSE_INTRADAY_CACHE_SECS", "15"))
INTRADAY_DB_FALLBACK_MAX_AGE_SECS = int(
    os.environ.get("NEPSE_INTRADAY_DB_FALLBACK_MAX_AGE_SECS", "900")
)
NEPALSTOCK_TLS_VERIFY = _env_bool("NEPALSTOCK_TLS_VERIFY", False)


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class MarketSnapshot:
    quotes: Dict[str, Dict[str, Any]]
    source: str
    endpoint: str
    fetched_at_utc: str
    market_status: Optional[str]
    market_status_as_of: Optional[str]


class RealtimeMarketDataProvider:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None
        self._snapshot: Optional[MarketSnapshot] = None
        self._cache_deadline = 0.0
        self._last_batch_info: Dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            from nepse import Nepse

            client = Nepse()
            client.setTLSVerification(NEPALSTOCK_TLS_VERIFY)
            self._client = client
        return self._client

    def _normalize_quote(
        self,
        row: Dict[str, Any],
        *,
        source: str,
        fetched_at_utc: str,
    ) -> Optional[Dict[str, Any]]:
        symbol = _as_text(row.get("symbol"))
        if not symbol:
            return None
        symbol = symbol.upper()

        last_traded_price = None
        for key in ("lastTradedPrice", "ltp", "closePrice", "closingPrice", "lastPrice", "close"):
            last_traded_price = _as_float(row.get(key))
            if last_traded_price is not None and last_traded_price > 0:
                break

        close_price = None
        for key in ("closePrice", "closingPrice", "lastTradedPrice", "close"):
            close_price = _as_float(row.get(key))
            if close_price is not None and close_price > 0:
                break

        previous_close = _as_float(row.get("previousClose"))
        percentage_change = _as_float(row.get("percentageChange"))
        if percentage_change is None and last_traded_price and previous_close:
            if previous_close > 0:
                percentage_change = ((last_traded_price - previous_close) / previous_close) * 100.0

        total_trade_quantity = None
        for key in ("totalTradeQuantity", "shareTraded", "shareVolume", "volume"):
            total_trade_quantity = _as_float(row.get(key))
            if total_trade_quantity is not None:
                break

        normalized = {
            "symbol": symbol,
            "security_id": _as_text(row.get("securityId") or row.get("id")),
            "security_name": _as_text(row.get("securityName") or row.get("companyName")),
            "last_traded_price": last_traded_price or close_price,
            "close_price": close_price or last_traded_price,
            "previous_close": previous_close,
            "percentage_change": percentage_change,
            "total_trade_quantity": total_trade_quantity,
            "source": source,
            "fetched_at_utc": fetched_at_utc,
        }
        if normalized["last_traded_price"] is None or normalized["last_traded_price"] <= 0:
            return None
        return normalized

    def _persist_snapshot(
        self,
        *,
        dataset: str,
        source: str,
        endpoint: str,
        fetched_at_utc: str,
        payload: Any,
        quotes: List[Dict[str, Any]],
        market_status: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_id = save_market_data_raw(
            dataset=dataset,
            source=source,
            payload=payload,
            fetched_at_utc=fetched_at_utc,
            record_count=len(quotes),
            metadata={
                "endpoint": endpoint,
                "market_status": market_status or {},
            },
        )
        save_market_quotes(raw_id, quotes)

    def _fetch_primary_snapshot(self) -> MarketSnapshot:
        client = self._get_client()
        fetched_at_utc = utc_now_iso()
        market_status = None
        try:
            market_status = client.getMarketStatus()
        except Exception as exc:
            logger.warning("Failed to fetch Nepalstock market status: %s", exc)

        endpoint = "live_market"
        rows: List[Dict[str, Any]] = []
        try:
            live_rows = client.getLiveMarket()
            if isinstance(live_rows, list):
                rows = live_rows
        except Exception as exc:
            logger.warning("Nepalstock live market fetch failed: %s", exc)

        if not rows:
            endpoint = "price_volume"
            rows = client.getPriceVolume()
            if not isinstance(rows, list):
                raise ValueError("Nepalstock price volume response was not a list")

        quotes = []
        quotes_by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            quote = self._normalize_quote(row, source="nepalstock", fetched_at_utc=fetched_at_utc)
            if quote is None:
                continue
            quotes.append(quote)
            quotes_by_symbol[quote["symbol"]] = quote

        self._persist_snapshot(
            dataset="intraday_snapshot",
            source="nepalstock",
            endpoint=endpoint,
            fetched_at_utc=fetched_at_utc,
            payload=rows,
            quotes=quotes,
            market_status=market_status if isinstance(market_status, dict) else None,
        )

        return MarketSnapshot(
            quotes=quotes_by_symbol,
            source="nepalstock",
            endpoint=endpoint,
            fetched_at_utc=fetched_at_utc,
            market_status=_as_text((market_status or {}).get("isOpen")) if isinstance(market_status, dict) else None,
            market_status_as_of=_as_text((market_status or {}).get("asOf")) if isinstance(market_status, dict) else None,
        )

    def fetch_snapshot(self, *, force: bool = False) -> MarketSnapshot:
        if not force and self._snapshot is not None and time.monotonic() < self._cache_deadline:
            return self._snapshot

        with self._lock:
            if not force and self._snapshot is not None and time.monotonic() < self._cache_deadline:
                return self._snapshot
            snapshot = self._fetch_primary_snapshot()
            self._snapshot = snapshot
            self._cache_deadline = time.monotonic() + max(1, INTRADAY_CACHE_SECS)
            return snapshot

    def get_latest_ltps(
        self,
        symbols: Iterable[str],
        *,
        force_refresh: bool = False,
    ) -> Dict[str, Optional[float]]:
        normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        if not normalized_symbols:
            return {}

        prices: Dict[str, Optional[float]] = {symbol: None for symbol in normalized_symbols}
        missing = set(normalized_symbols)
        source_map: Dict[str, str] = {}
        timestamp_map: Dict[str, str] = {}
        snapshot: Optional[MarketSnapshot] = None

        try:
            snapshot = self.fetch_snapshot(force=force_refresh)
            for symbol in list(missing):
                quote = snapshot.quotes.get(symbol)
                if quote is None:
                    continue
                price = _as_float(quote.get("last_traded_price"))
                if price is None or price <= 0:
                    continue
                prices[symbol] = price
                missing.discard(symbol)
                source_map[symbol] = "nepalstock"
                if snapshot and snapshot.fetched_at_utc:
                    timestamp_map[symbol] = str(snapshot.fetched_at_utc)
        except Exception as exc:
            logger.warning("Primary intraday snapshot fetch failed: %s", exc)

        if missing:
            db_quotes = load_latest_market_quotes(
                missing,
                max_age_seconds=INTRADAY_DB_FALLBACK_MAX_AGE_SECS,
            )
            for symbol in list(missing):
                quote = db_quotes.get(symbol)
                if quote is None:
                    continue
                price = _as_float(quote.get("last_traded_price"))
                if price is None or price <= 0:
                    continue
                prices[symbol] = price
                missing.discard(symbol)
                source_map[symbol] = "sqlite_cache"
                fetched_at = quote.get("fetched_at_utc")
                if fetched_at:
                    timestamp_map[symbol] = str(fetched_at)

        if missing:
            fallback_prices = self._fetch_merolagani_ltps(missing)
            fallback_timestamp = utc_now_iso()
            for symbol, price in fallback_prices.items():
                prices[symbol] = price
                if price is not None and price > 0:
                    source_map[symbol] = "merolagani"
                    timestamp_map[symbol] = fallback_timestamp

        self._last_batch_info = {
            "requested_symbols": normalized_symbols,
            "resolved_symbols": [symbol for symbol, price in prices.items() if price is not None],
            "missing_symbols": [symbol for symbol, price in prices.items() if price is None],
            "source_map": source_map,
            "timestamp_map": timestamp_map,
            "primary_count": sum(1 for src in source_map.values() if src == "nepalstock"),
            "db_cache_count": sum(1 for src in source_map.values() if src == "sqlite_cache"),
            "fallback_count": sum(1 for src in source_map.values() if src == "merolagani"),
            "snapshot_source": snapshot.source if snapshot else None,
            "snapshot_endpoint": snapshot.endpoint if snapshot else None,
            "snapshot_fetched_at_utc": snapshot.fetched_at_utc if snapshot else None,
            "market_status": snapshot.market_status if snapshot else None,
            "market_status_as_of": snapshot.market_status_as_of if snapshot else None,
        }

        return prices

    def get_latest_ltp(self, symbol: str, *, force_refresh: bool = False) -> Optional[float]:
        symbol = str(symbol).strip().upper()
        if not symbol:
            return None
        return self.get_latest_ltps([symbol], force_refresh=force_refresh).get(symbol)

    def fetch_market_depth(self, symbol: str) -> Any:
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ValueError("Symbol is required for market depth")
        client = self._get_client()
        payload = client.getSymbolMarketDepth(symbol)
        save_market_data_raw(
            dataset="market_depth",
            source="nepalstock",
            symbol=symbol,
            payload=payload,
            fetched_at_utc=utc_now_iso(),
        )
        return payload

    def get_last_batch_info(self) -> Dict[str, Any]:
        return dict(self._last_batch_info)

    def fetch_floorsheet(
        self,
        *,
        symbol: Optional[str] = None,
        business_date: Optional[str] = None,
        show_progress: bool = False,
    ) -> Any:
        client = self._get_client()
        dataset = "floorsheet"
        payload = None
        if symbol:
            symbol = str(symbol).strip().upper()
            payload = client.getFloorSheetOf(symbol, business_date=business_date)
            dataset = "floorsheet_symbol"
        else:
            payload = client.getFloorSheet(show_progress=show_progress)
        save_market_data_raw(
            dataset=dataset,
            source="nepalstock",
            symbol=symbol,
            business_date=business_date,
            payload=payload,
            fetched_at_utc=utc_now_iso(),
        )
        return payload

    def _fetch_merolagani_ltps(self, symbols: Iterable[str]) -> Dict[str, Optional[float]]:
        prices: Dict[str, Optional[float]] = {}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        for symbol in symbols:
            symbol = str(symbol).strip().upper()
            if not symbol:
                continue
            try:
                url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                ltp = soup.select_one("#ctl00_ContentPlaceHolder1_CompanyDetail1_lblMarketPrice")
                price = _as_float(ltp.text if ltp is not None else None)
                prices[symbol] = price
            except Exception as exc:
                logger.warning("Fallback MeroLagani fetch failed for %s: %s", symbol, exc)
                prices[symbol] = None
        return prices


_provider = RealtimeMarketDataProvider()


def get_market_data_provider() -> RealtimeMarketDataProvider:
    return _provider
