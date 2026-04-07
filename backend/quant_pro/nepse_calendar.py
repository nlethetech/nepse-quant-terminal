"""
NEPSE Trading Calendar

Handles Nepal Stock Exchange schedule and holiday logic.

The trading week is centralized here so the rest of the app doesn't hard-code
weekday assumptions in scattered modules. By default this runtime uses:
- Trading days: Monday through Friday
- Weekends: Saturday and Sunday
- Pre-open: 10:30-10:45 NPT
- Regular session: 11:00-15:00 NPT

Set `NEPSE_TRADING_WEEK=sun_thu` to switch back to the legacy Sunday-Thursday
calendar if needed.
"""

import os
import sqlite3
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


NST_OFFSET = timedelta(hours=5, minutes=45)
TRADING_WEEK_MODE = str(os.environ.get("NEPSE_TRADING_WEEK", "mon_fri") or "mon_fri").strip().lower()


def _configured_weekend_days() -> Set[int]:
    if TRADING_WEEK_MODE in {"sun_thu", "sunday_thursday", "legacy"}:
        return {4, 5}  # Friday, Saturday
    return {5, 6}  # Saturday, Sunday


NEPAL_WEEKEND_DAYS = _configured_weekend_days()

PREOPEN_START_HOUR = 10
PREOPEN_START_MINUTE = 30
PREOPEN_END_HOUR = 10
PREOPEN_END_MINUTE = 45
REGULAR_OPEN_HOUR = 11
REGULAR_OPEN_MINUTE = 0
REGULAR_CLOSE_HOUR = 15
REGULAR_CLOSE_MINUTE = 0


def current_nepal_datetime() -> datetime:
    """Return the current Nepal time as a naive datetime."""
    return (datetime.now(timezone.utc) + NST_OFFSET).replace(tzinfo=None)


def get_market_schedule() -> dict:
    """Return the configured NEPSE trading-week and session descriptors."""
    if TRADING_WEEK_MODE in {"sun_thu", "sunday_thursday", "legacy"}:
        trading_week = "Sunday-Thursday"
        weekend = "Friday-Saturday"
    else:
        trading_week = "Monday-Friday"
        weekend = "Saturday-Sunday"
    return {
        "trading_week": trading_week,
        "weekend": weekend,
        "preopen": f"{PREOPEN_START_HOUR:02d}:{PREOPEN_START_MINUTE:02d}-{PREOPEN_END_HOUR:02d}:{PREOPEN_END_MINUTE:02d} NPT",
        "special_preopen": f"{PREOPEN_START_HOUR:02d}:{PREOPEN_START_MINUTE:02d}-{PREOPEN_END_HOUR:02d}:{PREOPEN_END_MINUTE:02d} NPT",
        "regular": f"{REGULAR_OPEN_HOUR:02d}:{REGULAR_OPEN_MINUTE:02d}-{REGULAR_CLOSE_HOUR:02d}:{REGULAR_CLOSE_MINUTE:02d} NPT",
    }


# Known NEPSE holidays (AD dates) from SEBON/government announcements.
# These are used for forward-looking scheduling (daily_run.sh, signal timing).
# Source: timeanddate.com/holidays/nepal, SEBON public-holidays page
KNOWN_HOLIDAYS_2025 = {
    date(2025, 1, 11),   # Prithvi Jayanti
    date(2025, 1, 14),   # Maghe Sankranti
    date(2025, 1, 30),   # Martyrs' Memorial Day / Sonam Losar
    date(2025, 2, 19),   # National Democracy Day
    date(2025, 2, 26),   # Maha Shivaratri
    date(2025, 3, 8),    # Nari Dibas
    date(2025, 4, 6),    # Ram Nawami
    date(2025, 4, 14),   # Nepali New Year (BS 2082)
    date(2025, 5, 1),    # Majdoor Divas
    date(2025, 5, 12),   # Buddha Jayanti
    date(2025, 5, 29),   # Ganatantra Diwas (Republic Day)
    date(2025, 6, 7),    # Eid ul-Adha
    date(2025, 8, 9),    # Janai Purnima
    date(2025, 8, 16),   # Krishna Janmashtami
    date(2025, 9, 17),   # National Mourning Day
    date(2025, 9, 19),   # Constitution Day
    date(2025, 9, 22),   # Ghatasthapana (Dashain start)
    date(2025, 9, 29),   # Phulpati (Dashain)
    date(2025, 9, 30),   # Astami (Dashain)
    date(2025, 10, 1),   # Nawami (Dashain)
    date(2025, 10, 2),   # Dashami (Dashain)
    date(2025, 10, 3),   # Ekadashi (Dashain)
    date(2025, 10, 4),   # Duwadashi (Dashain)
    date(2025, 10, 5),   # Post-Dashain
    date(2025, 10, 6),   # Post-Dashain
    date(2025, 10, 20),  # Laxmi Puja (Tihar)
    date(2025, 10, 21),  # Gai Tihar
    date(2025, 10, 22),  # Gobhardan Pujan (Tihar)
    date(2025, 10, 23),  # Bhai Tika (Tihar)
    date(2025, 10, 24),  # Tihar Holiday
    date(2025, 10, 27),  # Chhath Parwa
}

KNOWN_HOLIDAYS_2026 = {
    date(2026, 1, 15),   # Maghe Sankranti
    date(2026, 1, 19),   # Sonam Losar
    date(2026, 2, 15),   # Maha Shivaratri
    date(2026, 2, 19),   # National Democracy Day
    date(2026, 3, 8),    # Nari Dibas
    date(2026, 5, 1),    # Majdoor Divas
    date(2026, 9, 19),   # Constitution Day
    date(2026, 10, 17),  # Phulpati (Dashain)
    date(2026, 10, 18),  # Astami (Dashain)
    date(2026, 10, 20),  # Nawami (Dashain)
    date(2026, 10, 21),  # Dashami (Dashain)
    date(2026, 10, 22),  # Ekadashi (Dashain)
    date(2026, 11, 8),   # Laxmi Puja (Tihar)
    date(2026, 11, 9),   # Gobhardan Pujan (Tihar)
    date(2026, 11, 10),  # Bhai Tika (Tihar)
}

# Combined set of all known future holidays
KNOWN_HOLIDAYS = KNOWN_HOLIDAYS_2025 | KNOWN_HOLIDAYS_2026


# === Dashain start dates (Ghatasthapana) by year ===
# Extracted from KNOWN_HOLIDAYS above
DASHAIN_START_DATES = {
    2025: date(2025, 9, 22),   # Ghatasthapana (Dashain start)
    2026: date(2026, 10, 17),  # Phulpati (earliest Dashain holiday in 2026)
}

# === Tihar start dates (Laxmi Puja) by year ===
TIHAR_START_DATES = {
    2025: date(2025, 10, 20),  # Laxmi Puja (Tihar)
    2026: date(2026, 11, 8),   # Laxmi Puja (Tihar)
}


def is_dashain_period(dt, pre_days: int = 21, post_days: int = 3) -> bool:
    """
    Check if date falls within the Dashain rally period.

    The rally period spans from `pre_days` before Dashain start
    to `post_days` after Dashain start.

    Args:
        dt: date or datetime to check
        pre_days: days before Dashain start to begin rally window
        post_days: days after Dashain start to end rally window

    Returns:
        True if date falls within the Dashain rally period.
    """
    if hasattr(dt, 'date'):
        dt = dt.date()
    year = dt.year
    dashain_start = DASHAIN_START_DATES.get(year)
    if dashain_start is None:
        return False
    window_start = dashain_start - timedelta(days=pre_days)
    window_end = dashain_start + timedelta(days=post_days)
    return window_start <= dt <= window_end


def is_tihar_period(dt, pre_days: int = 7, post_days: int = 3) -> bool:
    """
    Check if date falls within the Tihar rally period.

    Args:
        dt: date or datetime to check
        pre_days: days before Tihar start to begin rally window
        post_days: days after Tihar start to end rally window

    Returns:
        True if date falls within the Tihar rally period.
    """
    if hasattr(dt, 'date'):
        dt = dt.date()
    year = dt.year
    tihar_start = TIHAR_START_DATES.get(year)
    if tihar_start is None:
        return False
    window_start = tihar_start - timedelta(days=pre_days)
    window_end = tihar_start + timedelta(days=post_days)
    return window_start <= dt <= window_end


def days_until_dashain(dt) -> Optional[int]:
    """
    Return calendar days until the next Dashain start.

    Checks current year's Dashain first; if already past, checks next year.
    Returns None if no Dashain date is known for the relevant years.

    Args:
        dt: date or datetime

    Returns:
        Number of calendar days until Dashain start, or None if unknown.
        Returns 0 if today IS the Dashain start date.
        Returns negative if past this year's Dashain and no next year data.
    """
    if hasattr(dt, 'date'):
        dt = dt.date()

    # Check current year
    dashain = DASHAIN_START_DATES.get(dt.year)
    if dashain is not None and dashain >= dt:
        return (dashain - dt).days

    # Check next year
    dashain_next = DASHAIN_START_DATES.get(dt.year + 1)
    if dashain_next is not None:
        return (dashain_next - dt).days

    return None


def is_nepal_weekend(d: date) -> bool:
    """Check if a date falls on Nepal's weekend (Friday or Saturday)."""
    return d.weekday() in NEPAL_WEEKEND_DAYS


def is_known_holiday(d: date) -> bool:
    """Check if a date is a known NEPSE holiday."""
    return d in KNOWN_HOLIDAYS


def is_trading_day(d: date) -> bool:
    """
    Check if a date is likely a NEPSE trading day.

    Uses known holidays and weekend rules. For historical accuracy,
    use derive_holidays_from_db() which uses actual trading data.
    """
    if is_nepal_weekend(d):
        return False
    if is_known_holiday(d):
        return False
    return True


def market_session_phase(dt: Optional[datetime] = None) -> str:
    """Return PREMARKET, PREOPEN, OPEN, POSTCLOSE, WEEKEND, or HOLIDAY."""
    current = dt or current_nepal_datetime()
    current_date = current.date()
    if is_nepal_weekend(current_date):
        return "WEEKEND"
    if is_known_holiday(current_date):
        return "HOLIDAY"

    hhmm = (current.hour * 60) + current.minute
    preopen_start = (PREOPEN_START_HOUR * 60) + PREOPEN_START_MINUTE
    preopen_end = (PREOPEN_END_HOUR * 60) + PREOPEN_END_MINUTE
    regular_start = (REGULAR_OPEN_HOUR * 60) + REGULAR_OPEN_MINUTE
    regular_end = (REGULAR_CLOSE_HOUR * 60) + REGULAR_CLOSE_MINUTE

    if hhmm < preopen_start:
        return "PREMARKET"
    if preopen_start <= hhmm < regular_start:
        return "PREOPEN"
    if regular_start <= hhmm < regular_end:
        return "OPEN"
    return "POSTCLOSE"


def derive_holidays_from_db(db_path: Path) -> Set[date]:
    """
    Derive historical NEPSE holidays from the database.

    Logic: Any configured trading-week date that has no trading data,
    but where surrounding dates DO have data, is a holiday.

    Returns: Set of holiday dates derived from actual trading gaps.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Get all unique trading dates from the database
    cur.execute("SELECT DISTINCT date FROM stock_prices ORDER BY date")
    trading_dates_raw = cur.fetchall()
    conn.close()

    if not trading_dates_raw:
        return set()

    trading_dates = set()
    for (d,) in trading_dates_raw:
        try:
            if isinstance(d, str):
                trading_dates.add(datetime.strptime(d, "%Y-%m-%d").date())
            else:
                trading_dates.add(d)
        except (ValueError, TypeError):
            continue

    if not trading_dates:
        return set()

    # Find date range
    min_date = min(trading_dates)
    max_date = max(trading_dates)

    # Iterate through all configured trading-week dates and find gaps
    holidays = set()
    current = min_date
    while current <= max_date:
        if not is_nepal_weekend(current) and current not in trading_dates:
            holidays.add(current)
        current += timedelta(days=1)

    logger.info(f"Derived {len(holidays)} historical holidays from DB "
                f"({min_date} to {max_date})")
    return holidays


def get_trading_calendar(db_path: Path) -> Set[date]:
    """
    Get the complete set of actual NEPSE trading dates from the database.

    This is the most reliable way to determine which days had trading.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT date FROM stock_prices ORDER BY date")
    rows = cur.fetchall()
    conn.close()

    trading_dates = set()
    for (d,) in rows:
        try:
            if isinstance(d, str):
                trading_dates.add(datetime.strptime(d, "%Y-%m-%d").date())
            else:
                trading_dates.add(d)
        except (ValueError, TypeError):
            continue
    return trading_dates


def count_trading_days(start: date, end: date,
                       trading_dates: Optional[Set[date]] = None) -> int:
    """
    Count the number of NEPSE trading days between two dates (inclusive).

    If trading_dates is provided (from DB), uses exact historical data.
    Otherwise, uses known holidays + weekend rules (approximate for future).
    """
    if trading_dates is not None:
        return len([d for d in trading_dates if start <= d <= end])

    # Approximate: count configured trading days minus known holidays
    count = 0
    current = start
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def next_trading_day(d: date, trading_dates: Optional[Set[date]] = None) -> date:
    """
    Find the next NEPSE trading day after the given date.

    If trading_dates is provided, uses exact data.
    Otherwise uses known holidays + weekend rules.
    """
    candidate = d + timedelta(days=1)
    for _ in range(30):  # Look up to 30 days ahead
        if trading_dates is not None:
            if candidate in trading_dates:
                return candidate
        elif is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    return candidate  # Fallback


def trading_days_until(target: date, from_date: Optional[date] = None,
                       trading_dates: Optional[Set[date]] = None) -> int:
    """Count trading days from from_date (default: today) until target date."""
    if from_date is None:
        from_date = date.today()
    return count_trading_days(from_date, target, trading_dates)


def is_today_trading_day(trading_dates: Optional[Set[date]] = None) -> bool:
    """Check if today is a NEPSE trading day."""
    today = current_nepal_datetime().date()
    if trading_dates is not None:
        return today in trading_dates
    return is_trading_day(today)


def print_calendar_summary(db_path: Optional[Path] = None) -> None:
    """Print a summary of the NEPSE trading calendar."""
    today = date.today()
    print(f"Date: {today} ({today.strftime('%A')})")
    print(f"Is Nepal weekend: {is_nepal_weekend(today)}")
    print(f"Is known holiday: {is_known_holiday(today)}")
    print(f"Is trading day (rule-based): {is_trading_day(today)}")

    if not is_trading_day(today):
        nxt = next_trading_day(today)
        print(f"Next trading day: {nxt} ({nxt.strftime('%A')})")

    # Upcoming holidays
    upcoming = sorted(h for h in KNOWN_HOLIDAYS if h >= today)[:10]
    if upcoming:
        print(f"\nUpcoming holidays ({len(upcoming)} shown):")
        for h in upcoming:
            print(f"  {h} ({h.strftime('%A')})")

    if db_path is not None:
        historical = derive_holidays_from_db(db_path)
        trading = get_trading_calendar(db_path)
        year_holidays = len([h for h in historical if h.year == today.year])
        year_trading = len([d for d in trading if d.year == today.year])
        print(f"\n{today.year}: {year_trading} trading days, "
              f"{year_holidays} holidays (from DB)")
        print(f"Total historical: {len(trading)} trading days, "
              f"{len(historical)} holidays")
