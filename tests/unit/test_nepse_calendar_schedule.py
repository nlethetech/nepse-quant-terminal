from __future__ import annotations

import importlib
from datetime import date, datetime


def test_default_schedule_is_mon_fri_with_preopen():
    import backend.quant_pro.nepse_calendar as calendar

    importlib.reload(calendar)

    schedule = calendar.get_market_schedule()
    assert schedule["trading_week"] == "Monday-Friday"
    assert schedule["preopen"] == "10:30-10:45 NPT"
    assert schedule["regular"] == "11:00-15:00 NPT"
    assert calendar.is_trading_day(date(2026, 4, 10)) is True  # Friday
    assert calendar.is_trading_day(date(2026, 4, 12)) is False  # Sunday
    assert calendar.market_session_phase(datetime(2026, 4, 6, 9, 4)) == "PREMARKET"
    assert calendar.market_session_phase(datetime(2026, 4, 6, 10, 35)) == "PREOPEN"
    assert calendar.market_session_phase(datetime(2026, 4, 6, 11, 15)) == "OPEN"
    assert calendar.market_session_phase(datetime(2026, 4, 6, 15, 5)) == "POSTCLOSE"


def test_legacy_sun_thu_override(monkeypatch):
    monkeypatch.setenv("NEPSE_TRADING_WEEK", "sun_thu")
    import backend.quant_pro.nepse_calendar as calendar

    importlib.reload(calendar)

    assert calendar.get_market_schedule()["trading_week"] == "Sunday-Thursday"
    assert calendar.is_trading_day(date(2026, 4, 10)) is False  # Friday
    assert calendar.is_trading_day(date(2026, 4, 12)) is True  # Sunday

    monkeypatch.delenv("NEPSE_TRADING_WEEK", raising=False)
    importlib.reload(calendar)
