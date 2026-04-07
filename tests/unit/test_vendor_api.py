from __future__ import annotations

import pandas as pd

from backend.quant_pro import vendor_api


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
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
