from __future__ import annotations

import pandas as pd

from backend.quant_pro import vendor_api


class _DummyResponse:
    def __init__(self, payload, *, content_type="application/json"):
        self._payload = payload
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        return None

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_fetch_ohlcv_chunk_treats_no_data_as_empty(monkeypatch):
    monkeypatch.setattr(
        vendor_api.requests,
        "get",
        lambda *args, **kwargs: _DummyResponse({"s": "no_data", "nextTime": 1775107220}),
    )

    df = vendor_api.fetch_ohlcv_chunk("MANUFACTURING", 1704067200, 1775097600)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_ohlcv_chunk_normalizes_millisecond_timestamps(monkeypatch):
    seen = {}

    def fake_get(_url, *, params, **_kwargs):
        seen.update(params)
        return _DummyResponse({"s": "no_data"})

    monkeypatch.setattr(vendor_api.requests, "get", fake_get)

    vendor_api.fetch_ohlcv_chunk("NABIL", 1704067200000, 1775097600000)

    assert seen["rangeStartDate"] == 1704067200
    assert seen["rangeEndDate"] == 1775097600


def test_fetch_ohlcv_chunk_treats_html_error_page_as_empty(monkeypatch):
    monkeypatch.setattr(
        vendor_api.requests,
        "get",
        lambda *args, **kwargs: _DummyResponse("<html>error</html>", content_type="text/html; charset=utf-8"),
    )

    df = vendor_api.fetch_ohlcv_chunk("ACLBSL", 1704067200, 1775097600)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
