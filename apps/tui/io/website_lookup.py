"""The Lookup tab's data from nepsequant.com: one request per stock.

When the terminal is linked to a nepsequant.com account, the Lookup tab shows
what the website knows about a stock — its daily history, key facts, related
stocks in the same sector, quarterly results, dividends, news, filings and
broker flow — from ``GET /api/cli/lookup/{symbol}`` (for the NEPSE index:
its history and the day's movers). Replies are kept a few minutes, so moving
between tabs and back costs nothing.

Network calls happen in ``get()``, which the terminal runs in a background
worker; nothing here touches the screen.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional
from urllib.parse import quote

from apps.cli import connect

TTL_S = 300.0
KEEP = 24  # stocks remembered at once
SESSIONS = {"D": 250, "W": 260, "M": 500, "Y": 500, "I": 250}  # daily rows asked for per view (a year for Daily)

Request = Callable[..., dict]


class WebsiteLookup:
    def __init__(self, key: str, *, request: Optional[Request] = None, clock: Callable[[], float] = time.time) -> None:
        self._key = key
        self._request: Request = request or connect.request_json
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}

    @classmethod
    def from_saved(cls) -> Optional["WebsiteLookup"]:
        """From the saved nepsequant.com key; None when the terminal is not linked."""
        try:
            key = connect.load_key()
        except connect.ConnectError:
            return None
        return cls(key) if key else None

    @classmethod
    def from_book(cls, book: Any) -> Optional["WebsiteLookup"]:
        """The lookup client for the account the shared paper book is linked to."""
        key = getattr(book, "_key", None)
        return cls(key, request=getattr(book, "_request", None)) if key else None

    def get(self, symbol: str, *, timeframe: str = "D", force: bool = False) -> dict[str, Any]:
        """The stock's lookup page. Raises ``connect.ConnectError`` (the website's
        own reason) when it cannot be read."""
        sym = str(symbol or "").strip().upper()
        sessions = SESSIONS.get(str(timeframe or "D").upper(), 120)
        key = (sym, sessions)
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit and not force and now - hit[0] < TTL_S:
                return hit[1]
        data = self._request(f"/api/cli/lookup/{quote(sym, safe='')}?sessions={sessions}", self._key)
        with self._lock:
            self._cache[key] = (now, data)
            while len(self._cache) > KEEP:
                self._cache.pop(min(self._cache, key=lambda k: self._cache[k][0]))
        return data
