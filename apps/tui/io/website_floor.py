"""The Signals tab's broker panels from nepsequant.com's floorsheet.

When the terminal is linked to a nepsequant.com account, it does not need a
locally scraped floorsheet: it downloads the latest day's trades from the
website (``GET /api/cli/floorsheet``, about 0.3 MB gzipped for a full day)
and computes the BROKER FLOOR and TOP BROKERS panels itself, with the native
Rust module when it is installed and plain Python otherwise
(apps/tui/io/floor_metrics.py).

Network calls happen only in ``refresh()``, which the terminal runs in a
background thread at most every few minutes; the screen reads the last copy.
Re-checking a day the terminal already holds costs a tiny "unchanged" reply,
so a finished day is never downloaded twice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import quote

from apps.cli import connect
from apps.tui.io import floor_metrics as fm

TTL_S = 300.0  # look for a newer day at most every five minutes
RETRY_S = 60.0  # after a failed download
TIMEOUT_S = 60.0

Request = Callable[..., dict]


@dataclass
class FloorView:
    day: fm.FloorDay
    signals: list[dict[str, Any]] = field(default_factory=list)
    top_brokers: list[dict[str, Any]] = field(default_factory=list)
    checked_at: float = 0.0


class WebsiteFloor:
    """The latest floorsheet day on nepsequant.com, computed for the panels."""

    def __init__(
        self,
        key: str,
        *,
        request: Optional[Request] = None,
        clock: Callable[[], float] = time.monotonic,
        engine: Optional[str] = None,
    ) -> None:
        self._key = key
        self._request: Request = request or connect.request_json
        self._clock = clock
        self._engine = engine
        self._lock = threading.Lock()
        self._fetching = threading.Lock()  # one download at a time
        self._view: Optional[FloorView] = None
        self._last_try: Optional[float] = None
        self._failed = False
        self.error = ""  # last problem, shown on screen; "" when fine

    @classmethod
    def from_book(cls, book: Any) -> Optional["WebsiteFloor"]:
        """Linked (a website book exists)? Then the same key reads the floorsheet."""
        key = getattr(book, "_key", "") if book is not None else ""
        return cls(key) if key else None

    @property
    def view(self) -> Optional[FloorView]:
        with self._lock:
            return self._view

    def due(self) -> bool:
        if self._last_try is None:
            return True
        wait = RETRY_S if self._failed else TTL_S
        return self._clock() - self._last_try >= wait

    def refresh(self, *, force: bool = False) -> bool:
        """Fetch the latest day if it is time to. True when the panels changed."""
        if not force and not self.due():
            return False
        if not self._fetching.acquire(blocking=False):
            return False  # another thread is already downloading
        try:
            return self._refresh()
        finally:
            self._fetching.release()

    def _refresh(self) -> bool:
        self._last_try = self._clock()
        current = self.view
        have = current.day.version if current is not None else ""
        path = "/api/cli/floorsheet" + (f"?have={quote(have, safe='')}" if have else "")
        try:
            payload = self._request(path, self._key, timeout=TIMEOUT_S, compressed=True)
            if payload.get("unchanged") and current is not None and payload.get("version") == have:
                current.checked_at = self._clock()
                changed = bool(self.error)
                self._failed, self.error = False, ""
                return changed
            day = fm.floor_day_from_payload(payload, engine=self._engine)
        except connect.ConnectError as exc:
            self._failed, self.error = True, str(exc)
            return True
        except Exception as exc:  # malformed reply: keep the last good day
            self._failed, self.error = True, f"Could not read the floorsheet from nepsequant.com: {exc}"
            return True
        view = FloorView(day, fm.signal_records(day), fm.top_broker_records(day), self._clock())
        with self._lock:
            self._view = view
        self._failed, self.error = False, ""
        return True
