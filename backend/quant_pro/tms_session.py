"""Browser/session settings and helpers for local TMS execution."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tms_audit import get_live_audit_db_path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _load_selector_overrides() -> Dict[str, List[str]]:
    raw_json = os.environ.get("NEPSE_TMS_SELECTORS_JSON")
    raw_file = os.environ.get("NEPSE_TMS_SELECTORS_FILE")
    payload: Dict[str, Any] = {}
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            payload = {}
    elif raw_file:
        path = Path(raw_file).expanduser()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
    selectors: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            selectors[str(key)] = [str(item) for item in value if str(item).strip()]
        elif value:
            selectors[str(key)] = [str(value)]
    return selectors


def default_selectors() -> Dict[str, List[str]]:
    return {
        "login_form": ["form:has(input[type='password'])", "input[type='password']"],
        "dashboard_ready": [
            "text=/dashboard/i",
            "text=/portfolio/i",
            "text=/order/i",
            "text=/market watch/i",
        ],
        "order_nav": ["text=/order entry/i", "text=/order/i", "a[href*='order']"],
        "order_form": ["form:has(input)", "div:has(input)"],
        "buy_tab": ["text=/buy/i", "button:has-text('Buy')"],
        "sell_tab": ["text=/sell/i", "button:has-text('Sell')"],
        "symbol_input": [
            "input[name='symbol']",
            "input[placeholder*='Symbol']",
            "input[aria-label*='Symbol']",
        ],
        "quantity_input": [
            "input[name='quantity']",
            "input[placeholder*='Qty']",
            "input[aria-label*='Qty']",
        ],
        "price_input": [
            "input[name='price']",
            "input[placeholder*='Price']",
            "input[aria-label*='Price']",
        ],
        "submit_buy": ["button:has-text('Buy')", "input[value='Buy']"],
        "submit_sell": ["button:has-text('Sell')", "input[value='Sell']"],
        "confirm_submit": ["button:has-text('Confirm')", "button:has-text('Submit')"],
        "success_text": ["text=/success/i", "text=/submitted/i", "text=/accepted/i"],
        "error_text": ["text=/error/i", "text=/failed/i", "text=/invalid/i", ".error, .alert-danger"],
        "order_reference": [".order-ref", "text=/order no/i", "text=/reference/i"],
        "order_book_nav": ["text=/order book/i", "a[href*='orderbook']"],
        "positions_nav": ["text=/position/i", "a[href*='position']"],
        "market_watch_nav": [
            "text=/market watch/i",
            "text=/watchlist/i",
            "a[href*='member-market-watch']",
            "a[href*='market-watch']",
            "a[href*='watchlist']",
        ],
        "market_watch_ready": [
            "text=/market watch/i",
            "text=/watchlist/i",
            "input[placeholder*='Symbol']",
            "input[aria-label*='Symbol']",
        ],
        "market_watch_table": [
            "table",
            "[role='table']",
            ".table",
        ],
        "market_watch_symbol_input": [
            "input[name='symbol']",
            "input[placeholder*='Symbol']",
            "input[aria-label*='Symbol']",
            "input[placeholder*='Scrip']",
        ],
        "market_watch_add": [
            "button:has-text('Add')",
            "button:has-text('Save')",
            "button[title*='Add']",
            "input[value='Add']",
        ],
        "market_watch_remove": [
            "button:has-text('Remove')",
            "button:has-text('Delete')",
            "button[title*='Remove']",
            "button[title*='Delete']",
        ],
        "market_watch_remove_in_row": [
            "button:has-text('Remove')",
            "button:has-text('Delete')",
            "button[title*='Remove']",
            "button[title*='Delete']",
            "a:has-text('Remove')",
            "a:has-text('Delete')",
            "[aria-label*='Remove']",
            "[aria-label*='Delete']",
            ".btn-danger",
        ],
        "market_watch_confirm": [
            "button:has-text('Confirm')",
            "button:has-text('Yes')",
            "button:has-text('OK')",
        ],
        "orders_table": ["table"],
        "positions_table": ["table"],
    }


def _load_kv_secret_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[str(key).strip()] = str(value).strip()
    return data


def _load_secret_credentials() -> tuple[Optional[str], Optional[str], str]:
    username = str(os.environ.get("NEPSE_TMS_USERNAME", "")).strip() or None
    password = str(os.environ.get("NEPSE_TMS_PASSWORD", "")).strip() or None
    if username and password:
        return username, password, "env"

    secret_file_raw = str(os.environ.get("NEPSE_TMS_SECRET_FILE", "")).strip()
    if not secret_file_raw:
        return username, password, "none"

    secret_file = Path(secret_file_raw).expanduser().resolve()
    data = _load_kv_secret_file(secret_file)
    username = username or str(data.get("NEPSE_TMS_USERNAME", "")).strip() or None
    password = password or str(data.get("NEPSE_TMS_PASSWORD", "")).strip() or None
    if username and password:
        return username, password, f"secret_file:{secret_file}"
    return username, password, f"secret_file_missing_fields:{secret_file}"


@dataclass
class TMSSettings:
    enabled: bool = False
    mode: str = "paper"
    base_url: str = "https://tms19.nepsetms.com.np/tms/client/dashboard"
    browser: str = "chrome"
    profile_dir: Path = field(default_factory=lambda: Path(".tms_chrome_profile").resolve())
    headless: bool = False
    owner_confirm_required: bool = True
    strategy_automation_enabled: bool = False
    auto_exits_enabled: bool = False
    max_order_notional: float = 250000.0
    max_daily_orders: int = 20
    symbol_cooldown_secs: int = 120
    max_price_deviation_pct: float = 2.5
    screenshot_dir: Path = field(default_factory=lambda: Path("artifacts/tms_screens").resolve())
    failsafe_freeze_on_uncertain_submit: bool = True
    human_delay_min: float = 0.35
    human_delay_max: float = 1.05
    selectors: Dict[str, List[str]] = field(default_factory=default_selectors)
    username: Optional[str] = None
    password: Optional[str] = None
    credentials_source: str = "none"

    @property
    def has_credentials(self) -> bool:
        return bool(self.username and self.password)


def load_tms_settings() -> TMSSettings:
    selectors = default_selectors()
    selectors.update(_load_selector_overrides())
    username, password, credentials_source = _load_secret_credentials()
    settings = TMSSettings(
        enabled=_env_flag("NEPSE_LIVE_EXECUTION_ENABLED", False),
        mode=str(os.environ.get("NEPSE_LIVE_EXECUTION_MODE", "paper")).strip().lower() or "paper",
        base_url=str(os.environ.get("NEPSE_TMS_BASE_URL", "https://tms19.nepsetms.com.np/tms/client/dashboard")).strip(),
        browser=str(os.environ.get("NEPSE_TMS_BROWSER", "chrome")).strip().lower() or "chrome",
        profile_dir=Path(os.environ.get("NEPSE_TMS_PROFILE_DIR", ".tms_chrome_profile")).expanduser().resolve(),
        headless=_env_flag("NEPSE_TMS_HEADLESS", False),
        owner_confirm_required=_env_flag("NEPSE_LIVE_OWNER_CONFIRM_REQUIRED", True),
        strategy_automation_enabled=_env_flag("NEPSE_LIVE_STRATEGY_AUTOMATION_ENABLED", False),
        auto_exits_enabled=_env_flag("NEPSE_LIVE_AUTO_EXITS_ENABLED", False),
        max_order_notional=_env_float("NEPSE_LIVE_MAX_ORDER_NOTIONAL", 250000.0),
        max_daily_orders=_env_int("NEPSE_LIVE_MAX_DAILY_ORDERS", 20),
        symbol_cooldown_secs=_env_int("NEPSE_LIVE_SYMBOL_COOLDOWN_SECS", 120),
        max_price_deviation_pct=_env_float("NEPSE_LIVE_MAX_PRICE_DEVIATION_PCT", 2.5),
        screenshot_dir=Path(os.environ.get("NEPSE_LIVE_SCREENSHOT_DIR", "artifacts/tms_screens")).expanduser().resolve(),
        failsafe_freeze_on_uncertain_submit=_env_flag("NEPSE_LIVE_FAILSAFE_FREEZE_ON_UNCERTAIN_SUBMIT", True),
        selectors=selectors,
        username=username,
        password=password,
        credentials_source=credentials_source,
    )
    settings.profile_dir.mkdir(parents=True, exist_ok=True)
    settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
    return settings


def validate_live_setup(settings: TMSSettings, *, interactive: bool = False) -> List[str]:
    errors: List[str] = []
    active_live_mode = bool(settings.enabled and settings.mode in {"live", "dual", "shadow_live"})
    if not active_live_mode:
        return errors

    if settings.mode in {"live", "dual"} and not settings.owner_confirm_required:
        errors.append("Live mode requires owner confirmation to stay enabled.")

    audit_path = get_live_audit_db_path()
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(audit_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    except Exception as exc:
        errors.append(f"Live audit DB is not writable: {exc}")

    requires_noninteractive_creds = settings.headless or settings.strategy_automation_enabled
    if requires_noninteractive_creds and not settings.has_credentials:
        errors.append(
            "Headless/automated live mode requires NEPSE_TMS_USERNAME and NEPSE_TMS_PASSWORD "
            "from environment or NEPSE_TMS_SECRET_FILE."
        )
    if not interactive and settings.mode in {"live", "dual"} and settings.strategy_automation_enabled and not settings.has_credentials:
        errors.append("Strategy automation in live mode cannot start without broker credentials.")
    return errors


def human_pause(settings: TMSSettings, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> None:
    low = settings.human_delay_min if minimum is None else minimum
    high = settings.human_delay_max if maximum is None else maximum
    time.sleep(random.uniform(float(low), float(high)))
