"""
Vendor API client utilities (non-Streamlit, backend-safe).
"""

from __future__ import annotations

import threading
import time as _time
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException, Timeout
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import logging

from .realtime_market import get_market_data_provider


logger = logging.getLogger(__name__)


class _RateLimiter:
    """Token-bucket rate limiter (0.5 req/sec = 1 request per 2 seconds)."""

    def __init__(self, rate: float = 0.5):
        self._min_interval = 1.0 / rate
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                _time.sleep(self._min_interval - elapsed)
            self._last_call = _time.monotonic()


_rate_limiter = _RateLimiter(rate=0.5)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RequestException, Timeout, HTTPError, ValueError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_ohlcv_chunk(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Merolagani chart handler.
    """
    _rate_limiter.wait()
    url = "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
    params = {
        "type": "get_advanced_chart",
        "symbol": symbol,
        "resolution": "1D",
        "rangeStartDate": int(start_ts),
        "rangeEndDate": int(end_ts),
        "isAdjust": "1",
        "currencyCode": "NPR",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}",
        "X-Requested-With": "XMLHttpRequest",
    }
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    # Merolagani returns {"s":"no_data"} for some index symbols/ranges.
    # Treat that as a valid empty result, not a schema failure worth retrying.
    if isinstance(data, dict) and str(data.get("s") or "").lower() == "no_data":
        return pd.DataFrame()

    required_keys = {"t", "o", "h", "l", "c", "v"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(f"Invalid schema for symbol={symbol}")

    if len(data["t"]) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"],
        }
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RequestException, Timeout, HTTPError, ValueError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_latest_ltp(symbol: str) -> Optional[float]:
    """
    Get the latest traded price using the shared intraday market data provider.
    Nepalstock is primary; MeroLagani is used only as a fallback.
    """
    provider = get_market_data_provider()
    return provider.get_latest_ltp(symbol)


def fetch_latest_ltps(symbols: Iterable[str]) -> Dict[str, Optional[float]]:
    """Batch fetch LTPs from the shared intraday market data provider."""
    provider = get_market_data_provider()
    return provider.get_latest_ltps(symbols)


def get_latest_ltps_context() -> Dict[str, object]:
    """Return metadata about the most recent batch LTP fetch."""
    provider = get_market_data_provider()
    return provider.get_last_batch_info()
