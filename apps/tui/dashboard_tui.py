#!/usr/bin/env python3
"""
NEPSE Bloomberg-Style Terminal Dashboard — Textual TUI

Run:  python3 dashboard_tui.py
Keys: 1-9 tabs │ / search │ L lookup │ R refresh │ Q quit
"""
from __future__ import annotations

import copy
import json as _json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import textwrap
import unicodedata
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, unquote

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt
else:
    msvcrt = None

import pandas as pd
import requests as _requests
from bs4 import BeautifulSoup
from rich.markup import escape as _escape_markup
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import ContentSwitcher, DataTable, Input, Static, Button, Label, OptionList
from textual.widgets.option_list import Option


def _silence_tui_noisy_loggers() -> None:
    """Keep background logger output from painting over the TUI."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "urllib3",
        "urllib3.connectionpool",
        "backend",
        "backend.backtesting.simple_backtest",
        "backend.trading.live_trader",
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.caching",
        "streamlit.runtime.caching.cache_data_api",
    ]
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for handler in list(root_logger.handlers):
        try:
            handler.setLevel(logging.WARNING)
        except Exception:
            pass


_silence_tui_noisy_loggers()





# ── Import data layer from existing dashboard.py ─────────────────────────────
from apps.classic.dashboard import (
    MD, _db, _vol, _pct, _npr, load_port, save_port, exec_buy, exec_sell,
    AMBER, WHITE, DIM, LABEL, GAIN_HI, GAIN, LOSS_HI, LOSS, CYAN, YELLOW, PURPLE, BLUE,
)
from configs.long_term import LONG_TERM_CONFIG

from backend.quant_pro.paths import (
    ensure_dir,
    get_project_root,
    get_runtime_dir,
    get_trading_runtime_dir,
    migrate_legacy_path,
)
from backend.agents.agent_analyst import (
    analyze as agent_analyze,
    append_external_agent_chat_message,
    build_algo_shortlist_snapshot,
    publish_agent_signal_snapshot,
    check_trade_approval,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
)
from backend.quant_pro.stock_report import build_stock_report
from backend.trading.tui_trading_engine import TUITradingEngine
from backend.market.kalimati_market import init_kalimati_db, refresh_kalimati, get_kalimati_display_rows
from backend.quant_pro.data_scrapers.gold_silver_ingestion import (
    store_nepal_metals_prices,
    get_latest_nepal_metals,
)
from backend.trading.live_trader import (
    NAV_LOG_COLS,
    PORTFOLIO_COLS,
    TRADE_LOG_COLS,
    calculate_cash_from_trade_log,
    load_runtime_state,
    save_runtime_state,
)
from backend.trading import strategy_registry

INITIAL_CAPITAL = 1_000_000.0
MAX_ACCOUNTS = 5
PROJECT_ROOT = get_project_root(__file__)

# ── Signal picker definitions (label, signal_type) ────────────────────────────
_SIGNAL_DEFS: list[tuple[str, str]] = [
    # Row 1 — factor / fundamental
    ("QUALITY",    "quality"),
    ("LOW VOL",    "low_vol"),
    ("XSEC MOM",  "xsec_momentum"),
    ("QF",         "quarterly_fundamental"),
    ("MEAN REV",  "mean_reversion"),
    # Row 2 — price / technical
    ("VOLUME",     "volume"),
    ("MOMENTUM",  "momentum"),
    ("ACCUM",     "accumulation"),
    ("52W HIGH",  "52wk_high"),
    ("VAL BNCE",  "value_bounce"),
    # Row 3 — alternative / microstructure / event
    ("CORP ACT",  "corp_action"),
    ("SMART $",   "smart_money"),
    ("SATELLITE", "satellite_hydro"),
]
_SIG_ID_PREFIX = "sig-btn-"

def _sig_btn_id(sig_type: str) -> str:
    return _SIG_ID_PREFIX + sig_type.replace("_", "-")

def _sig_type_from_id(btn_id: str) -> str:
    return btn_id[len(_SIG_ID_PREFIX):].replace("-", "_")
RUNTIME_DIR = ensure_dir(get_runtime_dir(__file__))
TRADING_RUNTIME_DIR = ensure_dir(get_trading_runtime_dir(__file__))
HEDGE_TRADE_LOG_FILE = TRADING_RUNTIME_DIR / "hedge_trade_log.json"
WATCHLIST_FILE = migrate_legacy_path(RUNTIME_DIR / "watchlist.json", [PROJECT_ROOT / "watchlist.json"])
PAPER_NAV_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_nav_log.csv", [PROJECT_ROOT / "paper_nav_log.csv"])
PAPER_TRADE_LOG_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_trade_log.csv", [PROJECT_ROOT / "paper_trade_log.csv"])
PAPER_STATE_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_state.json", [PROJECT_ROOT / "paper_state.json"])
PAPER_PORTFOLIO_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "paper_portfolio.csv", [PROJECT_ROOT / "paper_portfolio.csv"])
TUI_PAPER_PORTFOLIO_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_portfolio.csv",
    [PROJECT_ROOT / "tui_paper_portfolio.csv"],
)
TUI_PAPER_NAV_LOG_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_nav_log.csv",
    [PROJECT_ROOT / "tui_paper_nav_log.csv"],
)
TUI_PAPER_TRADE_LOG_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_trade_log.csv",
    [PROJECT_ROOT / "tui_paper_trade_log.csv"],
)
TUI_PAPER_STATE_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_state.json",
    [PROJECT_ROOT / "tui_paper_state.json"],
)
PAPER_PROFILE_FILE = TRADING_RUNTIME_DIR / "paper_profile.json"
PAPER_IMPORT_BACKUP_DIR = RUNTIME_DIR / "imports"
PAPER_ACCOUNTS_DIR = RUNTIME_DIR / "accounts"
PAPER_ACCOUNTS_REGISTRY_FILE = PAPER_ACCOUNTS_DIR / "registry.json"
MACRO_INDICATOR_HISTORY_FILE = migrate_legacy_path(
    RUNTIME_DIR / "macro_indicator_history.json",
    [PROJECT_ROOT / "macro_indicator_history.json"],
)
TUI_PAPER_ORDERS_FILE = migrate_legacy_path(TRADING_RUNTIME_DIR / "tui_paper_orders.json", [PROJECT_ROOT / "tui_paper_orders.json"])
TUI_PAPER_ORDER_HISTORY_FILE = migrate_legacy_path(
    TRADING_RUNTIME_DIR / "tui_paper_order_history.json",
    [PROJECT_ROOT / "tui_paper_order_history.json"],
)
AGENT_ARCHIVE_RENDER_LIMIT = 60
AGENT_CHAT_TIMEOUT_SECS = 90

ACTIVE_ACCOUNT_FILES = {
    "paper_portfolio.csv": PAPER_PORTFOLIO_FILE,
    "paper_trade_log.csv": PAPER_TRADE_LOG_FILE,
    "paper_nav_log.csv": PAPER_NAV_LOG_FILE,
    "paper_state.json": PAPER_STATE_FILE,
    "watchlist.json": WATCHLIST_FILE,
    "tui_paper_portfolio.csv": TUI_PAPER_PORTFOLIO_FILE,
    "tui_paper_trade_log.csv": TUI_PAPER_TRADE_LOG_FILE,
    "tui_paper_nav_log.csv": TUI_PAPER_NAV_LOG_FILE,
    "tui_paper_state.json": TUI_PAPER_STATE_FILE,
    "tui_paper_orders.json": TUI_PAPER_ORDERS_FILE,
    "tui_paper_order_history.json": TUI_PAPER_ORDER_HISTORY_FILE,
}


def _split_agent_messages_by_cutoff(items: list[dict] | None, cutoff_ts: float) -> tuple[list[dict], list[dict]]:
    visible: list[dict] = []
    hidden: list[dict] = []
    cutoff = float(cutoff_ts or 0.0)
    for item in list(items or []):
        ts = float((item or {}).get("ts") or 0.0)
        if ts >= cutoff:
            visible.append(item)
        else:
            hidden.append(item)
    return visible, hidden

# ── Default watchlist (NEPSE blue chips) ────────────────────────────────────
DEFAULT_WATCHLIST = [
    "NABIL", "NLIC", "UPPER", "CHDC", "SBL", "SHIVM", "NRIC",
    "NTC", "NICA", "GBIME", "KBL", "MEGA", "PRVU", "SBI",
]

STRATEGY_DISPLAY_NAMES = {
    "default_c5": "C5 Baseline",
    "temp_forward_winner": "TFW",
    "strat_3_p2r25": "P2R25",
    "strat_4_p3r32": "P3R32",
    "strat_6_r83": "R83",
    "strat_7_h53": "H53",
    
    
    
}

_TMS_AUDIT_SNAPSHOT_MAP = {
    "health": "tms_health",
    "account": "tms_account",
    "watchlist": "tms_watchlist",
    "funds": "tms_funds",
    "holdings": "tms_holdings",
    "orders_daily": "tms_orders_daily",
    "orders_historic": "tms_orders_historic",
    "trades_daily": "tms_trades_daily",
    "trades_historic": "tms_trades_historic",
}

def _stock_watchlist_entry(symbol: str) -> dict:
    sym = str(symbol or "").strip().upper()
    return {
        "kind": "stock",
        "key": f"stock:{sym}",
        "label": sym,
        "symbol": sym,
    }


def _normalize_watchlist_entry(item) -> Optional[dict]:
    if isinstance(item, str):
        sym = item.strip().upper()
        return _stock_watchlist_entry(sym) if sym else None
    if not isinstance(item, dict):
        return None

    kind = str(item.get("kind") or "stock").strip().lower()
    if kind == "stock":
        sym = str(item.get("symbol") or item.get("label") or "").strip().upper()
        return _stock_watchlist_entry(sym) if sym else None

    label = str(item.get("label") or item.get("symbol") or item.get("currency_code") or "").strip()
    if not label:
        return None
    key = str(item.get("key") or f"{kind}:{label}").strip()
    normalized = dict(item)
    normalized["kind"] = kind
    normalized["label"] = label
    normalized["key"] = key
    return normalized


def _watchlist_entry_key(item: dict) -> str:
    return str((item or {}).get("key") or "")


def _build_sell_holdings_map(positions: list[dict] | None) -> dict[str, int]:
    holdings: dict[str, int] = {}
    for pos in positions or []:
        sym = str((pos or {}).get("sym") or "").strip().upper()
        qty = int((pos or {}).get("qty") or 0)
        if sym and qty > 0:
            holdings[sym] = qty
    return holdings


def _format_sell_holdings_summary(positions: list[dict] | None) -> str:
    holdings = _build_sell_holdings_map(positions)
    if not holdings:
        return "No holdings loaded"
    parts = [f"{sym} {qty}" for sym, qty in sorted(holdings.items())]
    lines: list[str] = []
    chunk = 3
    for idx in range(0, len(parts), chunk):
        lines.append("   ".join(parts[idx:idx + chunk]))
    return "\n".join(lines)


def _resolve_sell_qty(symbol: str, raw_shares: str, holdings: dict[str, int]) -> int:
    sym = str(symbol or "").strip().upper()
    total = int(holdings.get(sym) or 0)
    if total <= 0:
        raise ValueError(f"{sym or 'Symbol'} not in holdings")
    token = str(raw_shares or "").strip().lower()
    if not token or token == "all":
        return total
    try:
        qty = int(token)
    except ValueError as exc:
        raise ValueError("Shares must be a number or 'all'") from exc
    if qty <= 0 or qty > total:
        raise ValueError(f"Invalid qty — holding {total}")
    return qty


def _load_watchlist() -> list[dict]:
    if WATCHLIST_FILE.exists():
        try:
            data = _json.loads(WATCHLIST_FILE.read_text())
            if isinstance(data, list) and data:
                rows = [_normalize_watchlist_entry(item) for item in data]
                return [row for row in rows if row]
        except Exception:
            pass
    return [_stock_watchlist_entry(sym) for sym in DEFAULT_WATCHLIST]

def _save_watchlist(entries: list[dict]) -> None:
    ensure_dir(WATCHLIST_FILE.parent)
    WATCHLIST_FILE.write_text(_json.dumps(entries, indent=2))


def _ensure_csv_file(path: Path, columns: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if target.exists():
        return
    pd.DataFrame(columns=columns).to_csv(target, index=False)


def _ensure_paper_runtime_files() -> None:
    _ensure_csv_file(PAPER_PORTFOLIO_FILE, PORTFOLIO_COLS)
    _ensure_csv_file(PAPER_TRADE_LOG_FILE, TRADE_LOG_COLS)
    _ensure_csv_file(PAPER_NAV_LOG_FILE, NAV_LOG_COLS)
    _ensure_csv_file(TUI_PAPER_PORTFOLIO_FILE, PORTFOLIO_COLS)
    _ensure_csv_file(TUI_PAPER_TRADE_LOG_FILE, TRADE_LOG_COLS)
    _ensure_csv_file(TUI_PAPER_NAV_LOG_FILE, NAV_LOG_COLS)
    if not PAPER_STATE_FILE.exists():
        save_runtime_state(
            str(PAPER_STATE_FILE),
            {"cash": float(INITIAL_CAPITAL), "daily_start_nav": float(INITIAL_CAPITAL)},
        )
    if not TUI_PAPER_STATE_FILE.exists():
        save_runtime_state(
            str(TUI_PAPER_STATE_FILE),
            {"cash": float(INITIAL_CAPITAL), "daily_start_nav": float(INITIAL_CAPITAL)},
        )


def _paper_filled_orders_for_day(order_history: list[dict] | None, day: str) -> list[dict]:
    filled: list[dict] = []
    for order in list(order_history or []):
        row = dict(order or {})
        if str(row.get("status") or "").upper() != "FILLED":
            continue
        stamp = str(row.get("updated_at") or row.get("created_at") or "")
        if stamp[:10] != str(day):
            continue
        filled.append(row)
    filled.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""))
    return filled


def _lock_file_exclusive(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")


def _unlock_file(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")


def _write_json_locked(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        _lock_file_exclusive(handle)
        try:
            handle.seek(0)
            handle.truncate()
            _json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def _load_macro_indicator_history() -> dict[str, dict]:
    path = Path(MACRO_INDICATOR_HISTORY_FILE)
    if not path.exists():
        return {}
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_macro_indicator_history(payload: dict[str, dict]) -> None:
    _write_json_locked(Path(MACRO_INDICATOR_HISTORY_FILE), payload)


def _apply_indicator_history_change(
    history: dict[str, dict],
    *,
    key: str,
    value: float,
    timestamp: Optional[str] = None,
) -> tuple[Optional[float], Optional[float]]:
    prev = history.get(str(key)) or {}
    prev_value = float(prev.get("value") or 0.0)
    change = None
    change_pct = None
    if prev_value > 0:
        change = float(value) - prev_value
        change_pct = (change / prev_value) * 100.0
    history[str(key)] = {
        "value": float(value),
        "timestamp": str(timestamp or datetime.utcnow().isoformat()),
    }
    return change, change_pct


def _display_live_override_enabled() -> bool:
    return False


def _load_accounts_registry() -> dict:
    if PAPER_ACCOUNTS_REGISTRY_FILE.exists():
        try:
            payload = _json.loads(PAPER_ACCOUNTS_REGISTRY_FILE.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"accounts": []}


def _save_accounts_registry(payload: dict) -> None:
    ensure_dir(PAPER_ACCOUNTS_REGISTRY_FILE.parent)
    PAPER_ACCOUNTS_REGISTRY_FILE.write_text(_json.dumps(payload, indent=2, sort_keys=True))


def _account_dir(account_id: str) -> Path:
    return ensure_dir(PAPER_ACCOUNTS_DIR / str(account_id))


def _copy_file_if_exists(source: Path, target: Path) -> None:
    src = Path(source)
    dst = Path(target)
    ensure_dir(dst.parent)
    if src.exists():
        shutil.copy2(src, dst)


def _blank_account_files(target_dir: Path) -> None:
    ensure_dir(target_dir)
    _ensure_csv_file(target_dir / "paper_portfolio.csv", PORTFOLIO_COLS)
    _ensure_csv_file(target_dir / "paper_trade_log.csv", TRADE_LOG_COLS)
    _ensure_csv_file(target_dir / "paper_nav_log.csv", NAV_LOG_COLS)
    _ensure_csv_file(target_dir / "tui_paper_portfolio.csv", PORTFOLIO_COLS)
    _ensure_csv_file(target_dir / "tui_paper_trade_log.csv", TRADE_LOG_COLS)
    _ensure_csv_file(target_dir / "tui_paper_nav_log.csv", NAV_LOG_COLS)
    if not (target_dir / "paper_state.json").exists():
        save_runtime_state(
            str(target_dir / "paper_state.json"),
            {"cash": float(INITIAL_CAPITAL), "daily_start_nav": float(INITIAL_CAPITAL)},
        )
    if not (target_dir / "tui_paper_state.json").exists():
        save_runtime_state(
            str(target_dir / "tui_paper_state.json"),
            {"cash": float(INITIAL_CAPITAL), "daily_start_nav": float(INITIAL_CAPITAL)},
        )
    if not (target_dir / "watchlist.json").exists():
        (target_dir / "watchlist.json").write_text(_json.dumps([_stock_watchlist_entry(sym) for sym in DEFAULT_WATCHLIST], indent=2))
    if not (target_dir / "tui_paper_orders.json").exists():
        (target_dir / "tui_paper_orders.json").write_text("[]")
    if not (target_dir / "tui_paper_order_history.json").exists():
        (target_dir / "tui_paper_order_history.json").write_text("[]")


def _next_account_id(accounts: list[dict]) -> str:
    highest = 0
    for account in accounts or []:
        account_id = str(account.get("id") or "")
        match = re.fullmatch(r"account_(\d+)", account_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"account_{highest + 1}"


def _portfolio_mark_value(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = 0.0
    for _, row in df.iterrows():
        qty = int(float(row.get("Quantity") or 0))
        price = float(row.get("Last_LTP") or row.get("Buy_Price") or 0)
        total += qty * price
    return float(total)


def _build_account_seed_state(portfolio_df: pd.DataFrame, target_nav: float) -> tuple[dict, pd.DataFrame]:
    positions_value = round(_portfolio_mark_value(portfolio_df), 2)
    cash = round(float(target_nav) - positions_value, 2)
    if cash < 0:
        raise ValueError(f"Target NAV is below current marked portfolio value {_npr_k(positions_value)}")
    today = datetime.now().strftime("%Y-%m-%d")
    state = {"cash": cash, "daily_start_nav": round(float(target_nav), 2)}
    nav_log = pd.DataFrame(
        [
            {
                "Date": today,
                "Cash": cash,
                "Positions_Value": positions_value,
                "NAV": round(float(target_nav), 2),
                "Num_Positions": len(portfolio_df.index),
            }
        ],
        columns=NAV_LOG_COLS,
    )
    return state, nav_log

def _load_profile_config() -> dict:
    if PAPER_PROFILE_FILE.exists():
        try:
            payload = _json.loads(PAPER_PROFILE_FILE.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _save_profile_config(payload: dict) -> None:
    ensure_dir(PAPER_PROFILE_FILE.parent)
    PAPER_PROFILE_FILE.write_text(_json.dumps(payload, indent=2, sort_keys=True))


def _bootstrap_paper_accounts() -> tuple[list[dict], str]:
    ensure_dir(PAPER_ACCOUNTS_DIR)
    strategy_registry.ensure_builtin_strategies()
    registry = _load_accounts_registry()
    accounts = strategy_registry.ensure_account_strategy_ids(list(registry.get("accounts") or []))
    profile = _load_profile_config()
    current_account_id = str(profile.get("current_account_id") or "").strip()
    if not accounts:
        current_account_id = "account_1"
        account = {
            "id": current_account_id,
            "name": "Account 1",
            "strategy_id": strategy_registry.default_strategy_for_account(current_account_id),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        accounts = [account]
    known_ids = {str(account.get("id") or "") for account in accounts}
    if current_account_id not in known_ids:
        current_account_id = str(accounts[0].get("id") or "account_1")
    target_dir = _account_dir(current_account_id)
    has_existing_runtime = any((target_dir / name).exists() for name in ACTIVE_ACCOUNT_FILES)
    if not has_existing_runtime:
        for name, active_path in ACTIVE_ACCOUNT_FILES.items():
            _copy_file_if_exists(Path(active_path), target_dir / name)
    _blank_account_files(target_dir)
    registry["accounts"] = accounts
    _save_accounts_registry(registry)
    profile["current_account_id"] = current_account_id
    _save_profile_config(profile)
    return accounts, current_account_id


def _coerce_dragdrop_path(raw_value: str) -> Optional[Path]:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    try:
        parts = shlex.split(raw)
        if len(parts) == 1:
            raw = parts[0]
    except Exception:
        raw = raw.strip("\"'")
    raw = raw.replace("\\ ", " ")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _normalize_frame_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())
        normalized[key] = str(col)
    return normalized


def _pick_column(df: pd.DataFrame, *aliases: str) -> Optional[str]:
    normalized = _normalize_frame_columns(df)
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", str(alias).strip().lower())
        match = normalized.get(key)
        if match:
            return match
    return None


def _is_meroshare_csv(df: pd.DataFrame) -> bool:
    """Detect MeroShare 'My Shares Values' export format."""
    cols_lower = {str(c).strip().lower() for c in df.columns}
    return "scrip" in cols_lower and "current balance" in cols_lower


def _normalize_meroshare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MeroShare 'My Shares Values' CSV to internal portfolio format.

    MeroShare columns:
      S.N | Scrip | Current Balance | Last Closing Price | Value as of Last Closing Price
        | Last Transaction Price (LTP) | Value as of LTP
    The last row is a 'Total :' summary — skip it.
    Since MeroShare doesn't export actual buy prices, we use LTP as the
    entry price (mark-to-market seed) and flag entries as 'meroshare_import'.
    """
    # Identify columns by loose matching
    scrip_col = _pick_column(df, "Scrip")
    qty_col = _pick_column(df, "Current Balance", "Balance")
    ltp_col = _pick_column(df, "Last Transaction Price (LTP)", "LTP", "Last Closing Price")
    close_col = _pick_column(df, "Last Closing Price")

    if not scrip_col or not qty_col:
        raise ValueError("Not a valid MeroShare CSV — expected 'Scrip' and 'Current Balance' columns")

    today = datetime.now().strftime("%Y-%m-%d")
    rows: list[dict] = []

    for _, row in df.iterrows():
        symbol = str(row.get(scrip_col) or "").strip().upper()
        # Skip blank rows and the 'Total :' footer row
        if not symbol or symbol in ("TOTAL :", "TOTAL:", " ", ""):
            continue
        # Skip non-scrip rows (S.N cell is not numeric)
        sn = str(row.get(_pick_column(df, "S.N", "SN", "S N") or "", "")).strip()
        if sn and not sn.replace(".", "").isdigit():
            continue

        raw_qty = row.get(qty_col)
        qty = 0
        try:
            qty = int(round(float(str(raw_qty).replace(",", "") or 0)))
        except (ValueError, TypeError):
            pass
        if qty <= 0:
            continue

        # Price: prefer LTP, fall back to last close
        price = 0.0
        for col in [ltp_col, close_col]:
            if col:
                try:
                    price = float(str(row.get(col) or "0").replace(",", ""))
                    if price > 0:
                        break
                except (ValueError, TypeError):
                    pass
        if price <= 0:
            continue

        amount = round(qty * price, 2)
        rows.append({
            "Entry_Date": today,
            "Symbol": symbol,
            "Quantity": qty,
            "Buy_Price": price,
            "Buy_Amount": amount,
            "Buy_Fees": 0.0,
            "Total_Cost_Basis": amount,
            "Signal_Type": "meroshare_import",
            "High_Watermark": price,
            "Last_LTP": price,
            "Last_LTP_Source": "meroshare",
            "Last_LTP_Time_UTC": None,
        })

    result = pd.DataFrame(rows, columns=PORTFOLIO_COLS) if rows else pd.DataFrame(columns=PORTFOLIO_COLS)
    return result.reset_index(drop=True)


def _normalize_import_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PORTFOLIO_COLS)
    # Auto-detect MeroShare 'My Shares Values' CSV
    if _is_meroshare_csv(df):
        return _normalize_meroshare_portfolio(df)
    if set(PORTFOLIO_COLS).issubset(df.columns):
        normalized = df.copy()
    else:
        symbol_col = _pick_column(df, "Symbol", "Ticker", "Item", "Stock")
        qty_col = _pick_column(df, "Quantity", "Qty", "Shares", "Units")
        price_col = _pick_column(df, "Buy_Price", "Avg_Price", "Average_Price", "Entry_Price", "Price", "Cost")
        if not symbol_col or not qty_col or not price_col:
            raise ValueError("Portfolio CSV needs symbol, quantity, and buy price columns")
        entry_date_col = _pick_column(df, "Entry_Date", "Buy_Date", "Date", "Purchase_Date")
        signal_col = _pick_column(df, "Signal_Type", "Signal", "Strategy", "Reason")
        watermark_col = _pick_column(df, "High_Watermark", "High", "Peak")
        last_ltp_col = _pick_column(df, "Last_LTP", "LTP", "Current_Price", "Market_Price")
        last_ltp_source_col = _pick_column(df, "Last_LTP_Source", "Price_Source", "Source")
        last_ltp_time_col = _pick_column(df, "Last_LTP_Time_UTC", "Price_Time", "Updated_At")
        rows: list[dict] = []
        today = datetime.now().strftime("%Y-%m-%d")
        for _, row in df.iterrows():
            symbol = str(row.get(symbol_col) or "").strip().upper()
            if not symbol:
                continue
            qty = int(float(row.get(qty_col) or 0))
            price = float(row.get(price_col) or 0)
            if qty <= 0 or price <= 0:
                continue
            fees = float(row.get(_pick_column(df, "Buy_Fees", "Fees") or "", 0) or 0)
            amount = float(row.get(_pick_column(df, "Buy_Amount", "Amount", "Gross_Amount") or "", qty * price) or (qty * price))
            total_cost = float(
                row.get(_pick_column(df, "Total_Cost_Basis", "Cost_Basis", "Net_Amount") or "", amount + fees)
                or (amount + fees)
            )
            entry_date = str(row.get(entry_date_col) or today)[:10]
            signal_type = str(row.get(signal_col) or "imported").strip().lower() or "imported"
            high_watermark = float(row.get(watermark_col) or price)
            last_ltp = row.get(last_ltp_col) if last_ltp_col else None
            last_ltp_source = row.get(last_ltp_source_col) if last_ltp_source_col else None
            last_ltp_time = row.get(last_ltp_time_col) if last_ltp_time_col else None
            rows.append(
                {
                    "Entry_Date": entry_date,
                    "Symbol": symbol,
                    "Quantity": qty,
                    "Buy_Price": price,
                    "Buy_Amount": amount,
                    "Buy_Fees": fees,
                    "Total_Cost_Basis": total_cost,
                    "Signal_Type": signal_type,
                    "High_Watermark": high_watermark,
                    "Last_LTP": last_ltp,
                    "Last_LTP_Source": last_ltp_source,
                    "Last_LTP_Time_UTC": last_ltp_time,
                }
            )
        normalized = pd.DataFrame(rows, columns=PORTFOLIO_COLS)
    normalized = normalized.reindex(columns=PORTFOLIO_COLS)
    if normalized.empty:
        return pd.DataFrame(columns=PORTFOLIO_COLS)
    normalized["Symbol"] = normalized["Symbol"].astype(str).str.strip().str.upper()
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce").fillna(0).astype(int)
    normalized["Buy_Price"] = pd.to_numeric(normalized["Buy_Price"], errors="coerce").fillna(0.0)
    normalized["Buy_Amount"] = pd.to_numeric(normalized["Buy_Amount"], errors="coerce").fillna(
        normalized["Quantity"] * normalized["Buy_Price"]
    )
    normalized["Buy_Fees"] = pd.to_numeric(normalized["Buy_Fees"], errors="coerce").fillna(0.0)
    normalized["Total_Cost_Basis"] = pd.to_numeric(normalized["Total_Cost_Basis"], errors="coerce").fillna(
        normalized["Buy_Amount"] + normalized["Buy_Fees"]
    )
    normalized["High_Watermark"] = pd.to_numeric(normalized["High_Watermark"], errors="coerce").fillna(normalized["Buy_Price"])
    return normalized[normalized["Symbol"] != ""].reset_index(drop=True)


def _normalize_import_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLS)
    if set(TRADE_LOG_COLS).issubset(df.columns):
        normalized = df.copy()
    else:
        date_col = _pick_column(df, "Date", "Trade_Date")
        action_col = _pick_column(df, "Action", "Side")
        symbol_col = _pick_column(df, "Symbol", "Ticker", "Item")
        shares_col = _pick_column(df, "Shares", "Quantity", "Qty")
        price_col = _pick_column(df, "Price", "Rate", "Execution_Price")
        if not date_col or not action_col or not symbol_col or not shares_col or not price_col:
            raise ValueError("Trade log CSV needs date, action, symbol, shares, and price columns")
        fees_col = _pick_column(df, "Fees", "Fee", "Commission")
        reason_col = _pick_column(df, "Reason", "Signal", "Strategy", "Note")
        pnl_col = _pick_column(df, "PnL", "P&L", "Profit")
        pnl_pct_col = _pick_column(df, "PnL_Pct", "Return_Pct", "P&L_Pct")
        rows: list[dict] = []
        for _, row in df.iterrows():
            symbol = str(row.get(symbol_col) or "").strip().upper()
            action = str(row.get(action_col) or "").strip().upper()
            if not symbol or action not in {"BUY", "SELL"}:
                continue
            rows.append(
                {
                    "Date": str(row.get(date_col) or "")[:10],
                    "Action": action,
                    "Symbol": symbol,
                    "Shares": int(float(row.get(shares_col) or 0)),
                    "Price": float(row.get(price_col) or 0),
                    "Fees": float(row.get(fees_col) or 0) if fees_col else 0.0,
                    "Reason": str(row.get(reason_col) or "imported").strip() or "imported",
                    "PnL": float(row.get(pnl_col) or 0) if pnl_col else 0.0,
                    "PnL_Pct": float(row.get(pnl_pct_col) or 0) if pnl_pct_col else 0.0,
                }
            )
        normalized = pd.DataFrame(rows, columns=TRADE_LOG_COLS)
    normalized = normalized.reindex(columns=TRADE_LOG_COLS)
    if normalized.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLS)
    normalized["Symbol"] = normalized["Symbol"].astype(str).str.strip().str.upper()
    return normalized[normalized["Symbol"] != ""].reset_index(drop=True)


def _normalize_import_nav_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=NAV_LOG_COLS)
    if set(NAV_LOG_COLS).issubset(df.columns):
        normalized = df.copy()
    else:
        date_col = _pick_column(df, "Date", "Session_Date")
        nav_col = _pick_column(df, "NAV", "Net_Asset_Value")
        if not date_col or not nav_col:
            raise ValueError("NAV CSV needs at least date and NAV columns")
        cash_col = _pick_column(df, "Cash")
        pos_col = _pick_column(df, "Positions_Value", "Invested", "Positions")
        num_col = _pick_column(df, "Num_Positions", "Positions_Count", "Holdings")
        rows: list[dict] = []
        for _, row in df.iterrows():
            nav = float(row.get(nav_col) or 0)
            if nav <= 0:
                continue
            rows.append(
                {
                    "Date": str(row.get(date_col) or "")[:10],
                    "Cash": float(row.get(cash_col) or 0) if cash_col else 0.0,
                    "Positions_Value": float(row.get(pos_col) or 0) if pos_col else 0.0,
                    "NAV": nav,
                    "Num_Positions": int(float(row.get(num_col) or 0)) if num_col else 0,
                }
            )
        normalized = pd.DataFrame(rows, columns=NAV_LOG_COLS)
    normalized = normalized.reindex(columns=NAV_LOG_COLS)
    return normalized.reset_index(drop=True)


def _build_holdings_watchlist_entries(port: pd.DataFrame, ltps: Optional[dict[str, float]] = None) -> list[dict]:
    if port is None or port.empty or "Symbol" not in port.columns:
        return []
    rows: list[tuple[float, str]] = []
    last_ltp_map = {}
    if "Last_LTP" in port.columns:
        last_ltp_map = {
            str(row.get("Symbol") or "").strip().upper(): float(row.get("Last_LTP") or 0)
            for _, row in port.iterrows()
        }
    for _, row in port.iterrows():
        sym = str(row.get("Symbol") or "").strip().upper()
        qty = int(float(row.get("Quantity") or 0))
        if not sym or qty <= 0:
            continue
        price = float((ltps or {}).get(sym) or last_ltp_map.get(sym) or row.get("Buy_Price") or 0)
        rows.append((qty * price, sym))
    rows.sort(reverse=True)
    return [_stock_watchlist_entry(sym) for _, sym in rows]


def _merge_watchlist_entries(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            normalized = _normalize_watchlist_entry(item)
            if not normalized:
                continue
            key = _watchlist_entry_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _load_manual_paper_cash(total_cost: float, nav_log: Optional[pd.DataFrame] = None) -> float:
    state = load_runtime_state(str(PAPER_STATE_FILE))
    saved_cash = state.get("cash")
    if isinstance(saved_cash, (int, float)):
        return float(saved_cash)
    rebuilt_cash = calculate_cash_from_trade_log(INITIAL_CAPITAL, str(PAPER_TRADE_LOG_FILE))
    if rebuilt_cash is not None:
        return float(rebuilt_cash)
    if nav_log is not None and not nav_log.empty and "Cash" in nav_log.columns:
        try:
            latest_cash = float(nav_log.iloc[-1]["Cash"])
            return latest_cash
        except Exception:
            pass
    return max(0.0, float(INITIAL_CAPITAL) - float(total_cost))


def _tms_health_flag(health: dict, key: str) -> bool:
    """Support both legacy nested status payloads and the current flat payload."""
    if not isinstance(health, dict):
        return False
    status = health.get("status")
    if isinstance(status, dict) and key in status:
        return bool(status.get(key))
    return bool(health.get(key))


def _load_cached_tms_bundle() -> dict:
    # TMS live brokerage not included in public release — always returns empty.
    return {}


def _merge_tms_bundle_with_cache(bundle: Optional[dict]) -> dict:
    merged = _load_cached_tms_bundle()
    if isinstance(bundle, dict):
        for key, payload in bundle.items():
            if payload:
                merged[key] = payload
    return merged

# ── Ticker scroll speed ─────────────────────────────────────────────────────
TICKER_SPEED = 0.15  # seconds between scroll steps

# ── OSINT API ────────────────────────────────────────────────────────────────
OSINT_BASE = "http://3.148.250.92/api/v1"
OSINT_TIMEOUT = 8

SEVERITY_STYLE = {
    "critical": f"bold {LOSS_HI}", "high": LOSS, "medium": YELLOW,
    "low": LABEL, "": LABEL,
}
TYPE_STYLE = {
    "political": PURPLE, "security": LOSS, "economic": CYAN,
    "disaster": LOSS_HI, "social": BLUE, "": LABEL,
}

# ── Nepali → English translation cache ──────────────────────────────────────
_translation_cache: dict[str, str] = {}


def _contains_non_ascii(text: str) -> bool:
    return bool(text and any(ord(c) > 127 for c in text[:20]))


# ── Unicode display-width helpers (Devanagari-aware) ─────────────────────────

def _disp_width(text: str) -> int:
    """Visual column width matching the patched Rich cell_len.

    Mn (non-spacing marks like virama) = 0 cells.
    Mc (spacing combining marks like ा ि ो) = 1 cell (macOS terminals).
    """
    return sum(
        0 if unicodedata.category(c) in ('Mn', 'Me', 'Cf') else 1
        for c in text
    )


def _truncate_display(text: str, max_cols: int, suffix: str = "…") -> str:
    """Truncate to max_cols display columns without splitting combining sequences."""
    text = unicodedata.normalize('NFC', text)
    w = 0
    suffix_w = len(suffix)
    for i, c in enumerate(text):
        cw = 0 if unicodedata.category(c) in ('Mn', 'Me', 'Cf') else 1
        if w + cw > max_cols - suffix_w:
            return text[:i] + suffix
        w += cw
    return text


def _wrap_display(text: str, width: int) -> list[str]:
    """Word-wrap by display column width (handles Devanagari combining chars)."""
    text = unicodedata.normalize('NFC', text)
    result: list[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            if result and result[-1]:
                result.append('')
            continue
        words = para.split(' ')
        line = ''
        line_w = 0
        for word in words:
            word_w = _disp_width(word)
            if line_w == 0:
                line, line_w = word, word_w
            elif line_w + 1 + word_w <= width:
                line += ' ' + word
                line_w += 1 + word_w
            else:
                if line:
                    result.append(line)
                line, line_w = word, word_w
        if line:
            result.append(line)
    return result

def _translate_nepali(text: str) -> str:
    """Translate Nepali text to English using Google Translate. Cached."""
    if not text or not any(ord(c) > 127 for c in text[:10]):
        return text
    if text in _translation_cache:
        return _translation_cache[text]
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source='ne', target='en').translate(text[:200])
        if result:
            _translation_cache[text] = result
            return result
    except Exception:
        pass
    return text

def _translate_batch(texts: list[str]) -> list[str]:
    """Translate a batch of Nepali texts. Uses single API call where possible."""
    to_translate = []
    indices = []
    results = list(texts)
    for i, t in enumerate(texts):
        if t and any(ord(c) > 127 for c in t[:10]) and t not in _translation_cache:
            to_translate.append(t[:200])
            indices.append(i)
        elif t in _translation_cache:
            results[i] = _translation_cache[t]
    if to_translate:
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='ne', target='en').translate_batch(to_translate)
            for idx, orig, trans in zip(indices, to_translate, translated):
                if trans:
                    _translation_cache[texts[idx]] = trans
                    results[idx] = trans
        except Exception:
            pass
    return results


def _news_display_headline(story: dict) -> str:
    """Return the best headline — prefer English, fall back to Nepali."""
    translated = str(story.get("_translated") or "").strip()
    if translated:
        return translated
    summary = str(story.get("summary") or "").strip()
    if summary and not _contains_non_ascii(summary):
        return summary
    translated_summary = str(story.get("_translated_summary") or "").strip()
    if translated_summary:
        return translated_summary
    url = str(story.get("url") or "").strip()
    if url:
        slug = _headline_fallback_from_url(url)
        if slug:
            return slug
    headline = str(story.get("canonical_headline") or "").strip()
    return headline or "Untitled story"


def _news_display_summary(story: dict) -> str:
    """Return the best summary — prefer English, fall back to Nepali."""
    summary = str(story.get("summary") or "").strip()
    if summary and not _contains_non_ascii(summary):
        return summary
    translated_summary = str(story.get("_translated_summary") or "").strip()
    if translated_summary:
        return translated_summary
    translated = str(story.get("_translated") or "").strip()
    if translated:
        return translated
    url = str(story.get("url") or "").strip()
    if url:
        slug = _headline_fallback_from_url(url)
        if slug:
            return slug
    if summary:
        return summary
    return "No summary available."


def _truncate_text(text: str, width: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + "…"


def _fetch_osint_stories(limit: int = 40) -> list[dict]:
    """Fetch latest stories from Nepal OSINT API."""
    try:
        r = _requests.get(f"{OSINT_BASE}/analytics/consolidated-stories",
                          params={"limit": limit}, timeout=OSINT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def _fetch_osint_brief() -> dict:
    """Fetch latest intelligence brief."""
    try:
        r = _requests.get(f"{OSINT_BASE}/briefs/latest", timeout=OSINT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# ── Color helpers for DataTable cells ─────────────────────────────────────────

def _chg_text(v: float) -> Text:
    s = "+" if v >= 0 else ""
    if v > 2:    c = GAIN_HI
    elif v > 0:  c = GAIN
    elif v < -2: c = LOSS_HI
    elif v < 0:  c = LOSS
    else:        c = WHITE
    return Text(f"{s}{v:.2f}%", style=c)

def _sym_text(s: str) -> Text:
    return Text(s, style=f"bold {WHITE}")

def _dim_text(s: str) -> Text:
    return Text(s, style=LABEL)

def _vol_text(v: float) -> Text:
    return Text(_vol(v), style=LABEL)

def _price_text(v: float) -> Text:
    return Text(f"{v:.1f}", style=WHITE)

def _pnl_color(v: float) -> str:
    return GAIN_HI if v > 0 else (LOSS_HI if v < 0 else WHITE)

def _npr_k(v: float) -> str:
    """Format NPR value compactly."""
    if abs(v) >= 1_000_000: return f"NPR {v/1_000_000:.2f}M"
    if abs(v) >= 1_000: return f"NPR {v/1_000:.1f}K"
    return f"NPR {v:.0f}"


def _render_stock_report(report: dict) -> Text:
    """Render deterministic financial report for the lookup pane."""
    def _append_wrapped_block(
        target: Text,
        body: str,
        *,
        style: str = WHITE,
        indent: str = "  ",
        continuation: str = "  ",
        width: int = 44,
    ) -> None:
        content = str(body or "").strip()
        if not content:
            return
        wrapped = textwrap.fill(content, width=width, initial_indent=indent, subsequent_indent=continuation)
        target.append(f"{wrapped}\n", style=style)

    text = Text()
    signal = report.get("signal", "NO DATA")
    score = int(report.get("score", 0) or 0)
    score_text = "N/A" if signal == "NO DATA" and score <= 0 else f"{score}/100"
    signal_style = {
        "ACCUMULATE": f"bold {GAIN_HI}",
        "WATCH": f"bold {YELLOW}",
        "CAUTION": f"bold {LOSS_HI}",
        "NO DATA": LABEL,
    }.get(signal, WHITE)

    text.append("  FINANCIAL REPORT\n", style=f"bold {AMBER}")
    text.append("  Signal", style=LABEL)
    text.append(f"  {signal}", style=signal_style)
    text.append("   ", style=WHITE)
    text.append("Score", style=LABEL)
    text.append(f"  {score_text}\n", style=f"bold {WHITE}")
    _append_wrapped_block(
        text,
        report.get("summary", "No summary available."),
        continuation="    ",
        width=42,
    )

    company_profile = report.get("company_profile") or {}
    board = company_profile.get("board") or []
    officers = company_profile.get("officers") or []
    company_name = str(company_profile.get("company_name") or "").strip()
    if company_name or board or officers:
        text.append("\n  Company Profile\n", style=f"bold {AMBER}")
        if company_name:
            text.append("  Name   ", style=LABEL)
            text.append(f"{company_name}\n", style=WHITE)
        if board:
            text.append("  Board of Directors\n", style=LABEL)
            for item in board:
                name = str((item or {}).get("name") or "").strip()
                role = str((item or {}).get("role") or "").strip()
                if not name:
                    continue
                text.append("    ", style=WHITE)
                text.append(f"{name:<24}", style=WHITE)
                if role:
                    text.append(f" {role}", style=LABEL)
                text.append("\n", style=WHITE)
        if officers:
            text.append("  Officers\n", style=LABEL)
            for item in officers:
                name = str((item or {}).get("name") or "").strip()
                role = str((item or {}).get("role") or "").strip()
                if not name:
                    continue
                text.append("    ", style=WHITE)
                text.append(f"{name:<24}", style=WHITE)
                if role:
                    text.append(f" {role}", style=LABEL)
                text.append("\n", style=WHITE)

    snapshot = report.get("snapshot") or []
    visible_snapshot = []
    for label, value in snapshot:
        rendered = str(value or "").strip()
        if not rendered or rendered == "—":
            continue
        visible_snapshot.append((label, rendered))
    if visible_snapshot:
        text.append("\n  Snapshot\n", style=f"bold {CYAN}")
        for label, value in visible_snapshot[:8]:
            text.append(f"  {label:<12}", style=LABEL)
            text.append(f" {value}\n", style=WHITE)

    for title, items, style in [
        ("Bull Case", report.get("positives") or [], GAIN_HI),
        ("Risk Case", report.get("risks") or [], LOSS_HI),
        ("Monitor", report.get("monitors") or [], YELLOW),
    ]:
        if not items:
            continue
        text.append(f"\n  {title}\n", style=f"bold {style}")
        for item in items:
            wrapped = textwrap.fill(
                str(item),
                width=42,
                initial_indent="  • ",
                subsequent_indent="    ",
            )
            text.append(f"{wrapped}\n", style=WHITE)

    notes = (report.get("latest_notes") or "").strip()
    if notes:
        text.append("\n  Latest Notes\n", style=f"bold {PURPLE}")
        display_notes = notes[:260] + ("..." if len(notes) > 260 else "")
        _append_wrapped_block(text, display_notes)

    return text


def _render_lookup_intelligence(report: dict, symbol: str) -> Text:
    """Render symbol-scoped intelligence / supply-chain brief."""
    intel_payload = report.get("intelligence") or {}
    text = Text()

    def _append_wrapped(target: Text, value: str, *, indent: str = "  ", continuation: str = "  ", width: int = 40, style: str = WHITE) -> None:
        body = str(value or "").strip()
        if not body:
            return
        wrapped = textwrap.fill(body, width=width, initial_indent=indent, subsequent_indent=continuation)
        target.append(f"{wrapped}\n", style=style)

    headline = str(intel_payload.get("headline") or "").strip()
    text.append("  CORPORATE INTELLIGENCE\n", style=f"bold {AMBER}")
    if headline:
        _append_wrapped(text, headline, width=38)

    sections = intel_payload.get("sections") or []
    for section in sections:
        title = str((section or {}).get("title") or "").strip()
        rows = (section or {}).get("rows") or []
        bullets = (section or {}).get("bullets") or []
        if title:
            text.append(f"\n  {title}\n", style=f"bold {AMBER}")
        for label, value in rows:
            clean_label = str(label).strip()
            clean_value = str(value).strip()
            if not clean_value:
                continue
            text.append(f"  {clean_label}\n", style=LABEL)
            _append_wrapped(text, clean_value, indent="    ", continuation="    ", width=36)
        for item in bullets:
            _append_wrapped(text, str(item), indent="  • ", continuation="    ", width=38)

    if not headline and not sections:
        text.append("\n  Pipeline Ready\n", style=f"bold {AMBER}")
        text.append("  Supply Chain", style=LABEL)
        text.append("  Planned\n", style=WHITE)
        text.append("  Threat / Political", style=LABEL)
        text.append("  Planned\n", style=WHITE)
        text.append("  News Catalyst", style=LABEL)
        text.append("  Use the news/OSINT feed as source\n", style=WHITE)
        text.append("  Next Step", style=LABEL)
        text.append(f"  Add symbol-scoped risk, supplier, and route signals for {symbol}", style=WHITE)

    return text

# ── Data helpers ──────────────────────────────────────────────────────────────

def _headline_fallback_from_url(url: str) -> str:
    """Return a readable headline candidate from a URL, or empty string."""
    clean_url = str(url or "").strip()
    if not clean_url:
        return ""
    try:
        parsed = urlparse(clean_url)
    except Exception:
        return ""

    candidate = unquote((parsed.path or "").rstrip("/").split("/")[-1]).strip()
    if not candidate:
        return ""

    candidate = re.sub(r"\.(html?|aspx|php|jsp)$", "", candidate, flags=re.I)
    candidate = candidate.replace("-", " ").replace("_", " ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate_lower = candidate.lower()

    generic_tokens = {
        "newsdetail", "newsdetails", "detail", "details", "article",
        "articles", "news", "story", "stories", "index",
    }
    if candidate_lower in generic_tokens or candidate_lower.startswith("newsdetail"):
        return ""
    if parsed.query and re.fullmatch(r"[a-z]+detail", candidate_lower):
        return ""
    if candidate.isdigit() or len(candidate) <= 5:
        return ""
    if not re.search(r"[A-Za-z]", candidate):
        return ""
    return candidate.title()


def _nst_today_str() -> str:
    """Return today's Nepal trading date as YYYY-MM-DD."""
    return (datetime.utcnow() + timedelta(hours=5, minutes=45)).strftime("%Y-%m-%d")


def _format_nst_hm(ts: float | None) -> str:
    """Render an epoch timestamp in Nepal time as HH:MM."""
    if not ts:
        return ""
    try:
        return (datetime.utcfromtimestamp(float(ts)) + timedelta(hours=5, minutes=45)).strftime("%H:%M")
    except Exception:
        return ""


def _extract_decimal_price(text: str) -> Optional[Decimal]:
    cleaned = re.sub(r"[^\d]", "", str(text or ""))
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except Exception:
        return None


def _fetch_usd_npr_rate() -> Optional[dict]:
    """Fetch latest USD/NPR rates from NRB."""
    api_url = "https://www.nrb.org.np/api/forex/v1/rates"
    try:
        today = datetime.utcnow().date()
        from_date = today - timedelta(days=7)
        response = _requests.get(
            api_url,
            params={
                "from": from_date.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "per_page": 50,
                "page": 1,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "Nepse-TUI/1.0",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status", {}).get("code") != 200:
            return None

        latest_match = None
        for rate_data in data.get("data", {}).get("payload", []):
            date_str = rate_data.get("date")
            try:
                rate_date = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            except Exception:
                rate_date = datetime.utcnow()
            for rate in rate_data.get("rates", []):
                currency = rate.get("currency", {})
                if currency.get("iso3") != "USD":
                    continue
                candidate = {
                    "currency_code": "USD",
                    "currency_name": currency.get("name", "US Dollar"),
                    "buy_rate": float(rate.get("buy", 0) or 0),
                    "sell_rate": float(rate.get("sell", 0) or 0),
                    "unit": int(currency.get("unit", 1) or 1),
                    "date": rate_date,
                    "source": "NRB",
                }
                if latest_match is None or candidate["date"] > latest_match["date"]:
                    latest_match = candidate
        return latest_match
    except Exception:
        return None


def _fetch_nrb_forex_rates(codes: tuple[str, ...] = ("USD", "EUR", "GBP", "INR", "CNY", "JPY")) -> list[dict]:
    """Fetch a compact set of NRB forex rows."""
    api_url = "https://www.nrb.org.np/api/forex/v1/rates"
    code_set = {c.upper() for c in codes}
    try:
        today = datetime.utcnow().date()
        from_date = today - timedelta(days=7)
        response = _requests.get(
            api_url,
            params={
                "from": from_date.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "per_page": 50,
                "page": 1,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "Nepse-TUI/1.0",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status", {}).get("code") != 200:
            return []

        latest_by_code: dict[str, dict] = {}
        previous_by_code: dict[str, dict] = {}
        for rate_data in data.get("data", {}).get("payload", []):
            date_str = rate_data.get("date")
            try:
                rate_date = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            except Exception:
                rate_date = datetime.utcnow()
            for rate in rate_data.get("rates", []):
                currency = rate.get("currency", {})
                code = str(currency.get("iso3") or "").upper()
                if code not in code_set:
                    continue
                candidate = {
                    "currency_code": code,
                    "currency_name": currency.get("name", code),
                    "buy_rate": float(rate.get("buy", 0) or 0),
                    "sell_rate": float(rate.get("sell", 0) or 0),
                    "unit": int(currency.get("unit", 1) or 1),
                    "date": rate_date,
                    "source": "NRB",
                }
                current = latest_by_code.get(code)
                if current is None or candidate["date"] > current["date"]:
                    if current is not None and current["date"] != candidate["date"]:
                        previous = previous_by_code.get(code)
                        if previous is None or current["date"] > previous["date"]:
                            previous_by_code[code] = current
                    latest_by_code[code] = candidate
                elif candidate["date"] != current["date"]:
                    previous = previous_by_code.get(code)
                    if previous is None or candidate["date"] > previous["date"]:
                        previous_by_code[code] = candidate
        rows = []
        for code in codes:
            row = latest_by_code.get(code)
            if not row:
                continue
            previous = previous_by_code.get(code) or {}
            prev_buy = float(previous.get("buy_rate") or 0)
            change_rate = (row["buy_rate"] - prev_buy) if prev_buy > 0 else None
            change_pct = ((row["buy_rate"] - prev_buy) / prev_buy * 100) if prev_buy > 0 else None
            enriched = dict(row)
            enriched["previous_buy_rate"] = prev_buy if prev_buy > 0 else None
            enriched["change_rate"] = change_rate
            enriched["change_pct"] = change_pct
            rows.append(enriched)
        return rows
    except Exception:
        return []


def _fetch_gold_silver_prices() -> Optional[dict]:
    """Fetch Nepal gold and silver prices from FENEGOSIDA."""
    url = "https://fenegosida.org/"
    try:
        response = _requests.get(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "Nepse-TUI/1.0",
            },
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text("\n")

        gold_tola = None
        silver_tola = None
        date_bs = None

        date_pattern = (
            r"(\d{1,2}\s+"
            r"(?:Baishakh|Jestha|Ashadh|Shrawan|Bhadra|Ashwin|Kartik|Mangsir|Poush|Magh|Falgun|Chaitra)"
            r"\s+\d{4})"
        )
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            date_bs = re.sub(r"\s+", " ", date_match.group(1)).strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        current_section = None

        for i, line in enumerate(lines):
            line_lower = line.lower()
            window = lines[i:i + 6]
            window_text = " ".join(window).lower()
            if "fine gold" in line_lower and "per 1 tola" in window_text and not gold_tola:
                for candidate in window:
                    price = _extract_decimal_price(candidate)
                    if price and price > 100000:
                        gold_tola = price
                        break
            if line_lower == "silver" and "per 1 tola" in window_text and not silver_tola:
                for candidate in window:
                    price = _extract_decimal_price(candidate)
                    if price and 1000 < price < 10000:
                        silver_tola = price
                        break

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if "gold" in line_lower and "silver" not in line_lower:
                current_section = "gold"
            elif "silver" in line_lower:
                current_section = "silver"

            price_matches = re.findall(r"रु\s*([\d,]+)", line) or re.findall(r"([\d,]{5,})", line)
            for price_str in price_matches:
                price = _extract_decimal_price(price_str)
                if not price or price <= 1000:
                    continue
                context = " ".join(lines[max(0, i - 2): i + 3]).lower()
                if "tola" in context or "तोला" in context:
                    if current_section == "gold" and not gold_tola:
                        gold_tola = price
                    elif current_section == "silver" and not silver_tola:
                        silver_tola = price

        if not gold_tola and not silver_tola:
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    row_text = " ".join(cell.get_text(" ", strip=True) for cell in cells).lower()
                    if "gold" in row_text or "सुन" in row_text:
                        for cell in cells:
                            price = _extract_decimal_price(cell.get_text(" ", strip=True))
                            if price and price > 100000 and not gold_tola:
                                gold_tola = price
                    if "silver" in row_text or "चाँदी" in row_text:
                        for cell in cells:
                            price = _extract_decimal_price(cell.get_text(" ", strip=True))
                            if price and price > 1000 and not silver_tola:
                                silver_tola = price

        if not gold_tola and not silver_tola:
            return None

        gold_val = float(gold_tola or 0)
        silver_val = float(silver_tola or 0)

        # ── Persist to macro_indicators so CHG% can be computed ──────────────
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            from backend.quant_pro.database import get_db_path as _get_db_path
            store_nepal_metals_prices(
                gold_npr_per_tola=gold_val,
                silver_npr_per_tola=silver_val,
                date_str=today_str,
                db_path=str(_get_db_path()),
            )
        except Exception:
            pass  # persistence failure must not break display

        # ── Load CHG% from stored history ─────────────────────────────────────
        gold_chg_pct = None
        silver_chg_pct = None
        gold_chg_abs = None
        silver_chg_abs = None
        try:
            from backend.quant_pro.database import get_db_path as _get_db_path2
            hist = get_latest_nepal_metals(db_path=str(_get_db_path2()))
            if hist.get("date") == today_str:
                gold_chg_pct = hist.get("gold_chg_pct")
                silver_chg_pct = hist.get("silver_chg_pct")
                gold_chg_abs = hist.get("gold_chg_abs")
                silver_chg_abs = hist.get("silver_chg_abs")
        except Exception:
            pass

        return {
            "gold_per_tola": gold_val,
            "silver_per_tola": silver_val,
            "gold_chg_pct": gold_chg_pct,
            "silver_chg_pct": silver_chg_pct,
            "gold_chg_abs": gold_chg_abs,
            "silver_chg_abs": silver_chg_abs,
            "date": datetime.utcnow(),
            "date_bs": date_bs,
            "source": "FENEGOSIDA",
        }
    except Exception:
        return None


def _fetch_yahoo_futures_price(symbol: str, label: str) -> Optional[dict]:
    """Fetch a lightweight global futures quote from Yahoo Finance."""
    try:
        response = _requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "5d", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
        meta = result.get("meta") or {}
        price = float(meta.get("regularMarketPrice") or 0)
        prev = meta.get("previousClose")
        if prev is None:
            prev = meta.get("chartPreviousClose")
        prev = float(prev or 0)
        pct = ((price - prev) / prev * 100) if prev > 0 else None
        change = (price - prev) if prev > 0 else None
        return {
            "label": label,
            "value": price,
            "unit": meta.get("currency", "USD"),
            "change": change,
            "change_pct": pct,
            "source": "Yahoo Finance",
        }
    except Exception:
        return None


def _fetch_noc_fuel_prices() -> Optional[dict]:
    """Fetch latest Nepal Oil Corporation retail prices."""
    try:
        response = _requests.get(
            "https://noc.org.np/retailprice",
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="table")
        if not table:
            return None

        headers = [th.get_text(" ", strip=True).lower() for th in table.find("tr").find_all(["th", "td"])]
        col_map: dict[str, int] = {}
        for idx, header in enumerate(headers):
            if "date" in header and "time" not in header:
                col_map["date"] = idx
            elif "petrol" in header or "ms" in header:
                col_map["petrol"] = idx
            elif "diesel" in header or "hsd" in header:
                col_map["diesel"] = idx
            elif "kerosene" in header or "sko" in header:
                col_map["kerosene"] = idx
            elif "lpg" in header:
                col_map["lpg"] = idx

        best_cells = None
        best_key = None
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue
            first_cell = cells[0].get_text(" ", strip=True)
            if "प्रेस" in first_cell.lower():
                continue
            match = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", first_cell)
            if not match:
                continue
            sort_key = tuple(int(x) for x in match.groups())
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best_cells = cells

        if not best_cells:
            return None

        def _cell_price(key: str) -> float:
            idx = col_map.get(key)
            if idx is None or idx >= len(best_cells):
                return 0.0
            raw = re.sub(r"[^\d.]", "", best_cells[idx].get_text(" ", strip=True))
            try:
                return float(raw) if raw else 0.0
            except Exception:
                return 0.0

        return {
            "date_bs": best_cells[col_map["date"]].get_text(" ", strip=True) if "date" in col_map else "",
            "petrol": _cell_price("petrol"),
            "diesel": _cell_price("diesel"),
            "kerosene": _cell_price("kerosene"),
            "lpg": _cell_price("lpg"),
            "source": "NOC",
        }
    except Exception:
        return None


def _format_compact_npr(value: float) -> str:
    return f"NPR {value:,.0f}"


def _load_intraday_ohlcv(
    symbol: str,
    *,
    preferred_session_date: Optional[str] = None,
    bucket_minutes: int = 5,
) -> tuple[pd.DataFrame, Optional[str], int]:
    """Build intraday OHLCV bars from stored quote snapshots."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame(), None, 0

    conn = _db()
    try:
        target_date = preferred_session_date
        if target_date:
            probe = pd.read_sql_query(
                """
                SELECT COUNT(*) AS cnt
                FROM market_quotes
                WHERE symbol = ?
                  AND date(datetime(fetched_at_utc, '+5 hours', '+45 minutes')) = ?
                """,
                conn,
                params=(sym, target_date),
            )
            if int(probe["cnt"].iloc[0] or 0) <= 0:
                target_date = None

        if not target_date:
            latest = pd.read_sql_query(
                """
                SELECT MAX(date(datetime(fetched_at_utc, '+5 hours', '+45 minutes'))) AS session_date
                FROM market_quotes
                WHERE symbol = ?
                """,
                conn,
                params=(sym,),
            )
            target_date = latest["session_date"].iloc[0]

        if not target_date:
            return pd.DataFrame(), None, 0

        raw = pd.read_sql_query(
            """
            SELECT fetched_at_utc, last_traded_price, close_price, total_trade_quantity
            FROM market_quotes
            WHERE symbol = ?
              AND date(datetime(fetched_at_utc, '+5 hours', '+45 minutes')) = ?
            ORDER BY fetched_at_utc ASC
            """,
            conn,
            params=(sym, target_date),
        )
    finally:
        conn.close()

    if raw.empty:
        return pd.DataFrame(), str(target_date), 0

    rows = raw.copy()
    rows["price"] = pd.to_numeric(rows["last_traded_price"], errors="coerce")
    rows["close_price"] = pd.to_numeric(rows["close_price"], errors="coerce")
    rows["price"] = rows["price"].fillna(rows["close_price"])
    rows = rows[rows["price"] > 0].copy()
    if rows.empty:
        return pd.DataFrame(), str(target_date), 0

    rows["cum_qty"] = pd.to_numeric(rows["total_trade_quantity"], errors="coerce").ffill().fillna(0.0)
    rows["volume"] = rows["cum_qty"].diff().clip(lower=0).fillna(0.0)
    rows["ts_nst"] = pd.to_datetime(rows["fetched_at_utc"], utc=True) + pd.Timedelta(hours=5, minutes=45)
    rows["bucket"] = rows["ts_nst"].dt.floor(f"{max(1, int(bucket_minutes))}min").dt.tz_localize(None)

    bars = (
        rows.groupby("bucket", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"bucket": "date"})
    )

    # If a coarse bucket collapses sparse snapshots into one bar, fall back to raw points
    # so the user still gets a visible same-session chart.
    if len(bars) < 2 and len(rows) >= 2:
        bars = pd.DataFrame(
            {
                "date": rows["ts_nst"].dt.tz_localize(None),
                "open": rows["price"],
                "high": rows["price"],
                "low": rows["price"],
                "close": rows["price"],
                "volume": rows["volume"],
            }
        )

    return bars, str(target_date), int(len(raw))


def _ensure_lookup_history(symbol: str, *, min_sessions: int = 2, history_days: int = 3650) -> int:
    """Backfill local daily OHLCV when lookup history is missing or too sparse."""
    sym = str(symbol or "").strip().upper()
    if not sym or sym == "NEPSE":
        return 0

    conn = _db()
    try:
        existing = pd.read_sql_query(
            "SELECT COUNT(*) AS cnt FROM stock_prices WHERE symbol = ?",
            conn,
            params=(sym,),
        )
        current_count = int(existing["cnt"].iloc[0] or 0)
    finally:
        conn.close()

    if current_count >= max(1, int(min_sessions)):
        return current_count

    try:
        from backend.quant_pro.database import save_to_db
        from backend.quant_pro.vendor_api import fetch_ohlcv_chunk

        end_ts = int(time.time())
        start_ts = int((datetime.now() - timedelta(days=max(30, int(history_days)))).timestamp())
        fetched = fetch_ohlcv_chunk(sym, start_ts=start_ts, end_ts=end_ts)
        if fetched is not None and not fetched.empty:
            save_to_db(fetched, sym)
            return int(len(fetched))
    except Exception:
        return current_count
    return current_count


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly, monthly, or yearly candles."""
    if timeframe in ("D", "I") or df.empty:
        return df
    rows = df.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.sort_values("date").set_index("date")
    rule = {"W": "W", "M": "ME", "Y": "YE"}.get(timeframe, "W")
    agg = rows.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open"])
    agg = agg.reset_index()
    return agg


def _render_candlestick_chart(df: pd.DataFrame, width: int = 120, height: int = 24,
                               timeframe: str = "D") -> Text:
    """Render candlestick chart using py-candlestick-chart library.

    Professional quality candles with proper wicks, bodies, and scaling.
    Supports D/W/M/Y/I timeframes via resampling.
    Adds date labels on the X-axis below the chart.
    """
    from candlestick_chart import Candle, Chart, constants

    if df.empty or len(df) < 2:
        return Text("  No data for chart", style=LABEL)

    rows = _resample_ohlcv(df, timeframe).sort_values("date").reset_index(drop=True)
    if rows.empty or len(rows) < 2:
        return Text("  Insufficient data for chart", style=LABEL)

    # Overall change for title
    first_close = float(rows.iloc[0]["close"])
    last_close = float(rows.iloc[-1]["close"])
    total_chg = (last_close - first_close) / first_close * 100 if first_close else 0

    chart_width = max(width, 40)

    # Build candles
    candles = []
    for _, r in rows.iterrows():
        candles.append(Candle(
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=float(r.get("volume", 0)),
        ))

    # Tighten library layout — reduce wasted vertical space
    constants.MARGIN_TOP = 1    # default 3 → 1 (less empty rows above candles)
    constants.Y_AXIS_SPACING = 3  # default 4 → 3 (denser price labels)

    # Chart title with timeframe + change
    tf_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
    tf_name = tf_labels.get(timeframe, "Daily")
    chg_sign = "+" if total_chg >= 0 else ""
    title = f"{tf_name}  {chg_sign}{total_chg:.2f}%"

    chart = Chart(candles, title=title, width=chart_width, height=max(height, 8))

    # Colors: teal green bull, red bear
    chart.set_bull_color(38, 166, 154)
    chart.set_bear_color(239, 83, 80)
    chart.set_vol_bull_color(38, 166, 154)
    chart.set_vol_bear_color(239, 83, 80)

    # Disable volume pane — saves vertical space, volume shown in header already
    chart.set_volume_pane_enabled(False)

    # Clean up info labels
    chart.set_label("average", "")
    chart.set_label("volume", "")
    chart.set_label("currency", "NPR")

    # Render chart to ANSI string
    ansi_str = chart._render()

    # Strip the library's bottom cruft (scrollbar, date axis, info line)
    # Keep only lines up to and including the last candle/Y-axis line
    ansi_lines = ansi_str.split('\n')

    # Find last line that contains Y-axis price labels or candle characters
    # The library's own date axis and info line come after the candle area
    # Info line pattern: contains "Price:" or "Highest:" or "Lowest:" or "Var.:"
    # Library date axis: contains only spaces, digits, dashes, and month names
    cut_idx = len(ansi_lines)
    for i in range(len(ansi_lines) - 1, -1, -1):
        stripped = ansi_lines[i].strip()
        if not stripped:
            cut_idx = i
            continue
        # Detect library info line (contains Price:/Highest:/Lowest:/Var.)
        plain = stripped.replace('\x1b', '')  # rough strip of ANSI for detection
        if any(kw in plain for kw in ['Price:', 'Highest:', 'Lowest:', 'Var.:']):
            cut_idx = i
            continue
        # Detect library date axis (mostly dashes, digits, month abbreviations)
        # and scrollbar lines (box drawing chars like ─ ┬ └ ┘ █ ░)
        if all(c in ' ─┬└┘┼│░▓█▒▄▀0123456789-' for c in stripped):
            cut_idx = i
            continue
        break  # hit a real candle/axis line, stop

    candle_lines = '\n'.join(ansi_lines[:cut_idx])

    # Build our own date axis labels
    y_axis_area = constants.WIDTH
    margin_r = constants.MARGIN_RIGHT if not constants.Y_AXIS_ON_THE_RIGHT else 0
    candle_area = chart_width - y_axis_area - margin_r
    n_visible = min(len(rows), candle_area)
    visible_rows = rows.tail(n_visible).reset_index(drop=True)
    if timeframe == "I":
        labels = [pd.Timestamp(d).strftime("%H:%M") for d in pd.to_datetime(visible_rows["date"])]
        label_len = 5
    else:
        labels = [str(d)[:10] for d in visible_rows["date"]]
        label_len = 10

    date_axis = Text()
    if n_visible >= 2:
        min_gap = label_len + 2
        max_labels = max(1, candle_area // min_gap)
        n_labels = min(max_labels, min(8, n_visible))

        if n_labels >= 2:
            step = max(1, (n_visible - 1) // (n_labels - 1))
            tick_line = [" "] * (y_axis_area + candle_area)
            label_positions = []
            for li in range(n_labels):
                idx = min(li * step, n_visible - 1)
                pos = y_axis_area + idx
                if pos < len(tick_line):
                    tick_line[pos] = "┬"
                    label_positions.append((idx, pos))
            date_axis.append("".join(tick_line[:y_axis_area]), style=DIM)
            date_axis.append("".join(tick_line[y_axis_area:y_axis_area + candle_area]).replace(" ", "─"), style=DIM)
            date_axis.append("\n")

            last_end = 0
            for idx, pos in label_positions:
                if pos < last_end:
                    continue
                gap = pos - last_end
                if gap > 0:
                    date_axis.append(" " * gap)
                date_axis.append(labels[idx], style=LABEL)
                last_end = pos + label_len

    # Assemble: candle chart + our date axis (no library info line)
    result = Text.from_ansi(candle_lines)
    if date_axis.plain.strip():
        result.append("\n")
        result.append_text(date_axis)

    return result


def _render_volume_chart(df: pd.DataFrame, width: int = 120, height: int = 6) -> str:
    """Render volume bar chart using plotext. Returns ANSI string."""
    import plotext as plt
    if df.empty:
        return ""

    rows = df.sort_values("date").reset_index(drop=True)
    dates = [str(d)[:10] for d in rows["date"]]
    vols = rows["volume"].tolist()
    colors = [
        "green" if float(rows.iloc[i]["close"]) >= float(rows.iloc[i]["open"]) else "red"
        for i in range(len(rows))
    ]

    plt.clear_figure()
    plt.date_form("Y-m-d")
    plt.bar(dates, vols, color=colors, width=0.8)
    plt.plotsize(max(width, 40), max(height, 4))
    plt.theme("dark")
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("yellow")
    plt.title("Volume")
    return plt.build()


def _render_sparkline(values: list[float], width: int = 30) -> Text:
    """Tiny inline sparkline using Unicode blocks. Green=up, Red=down vs previous bar."""
    if not values or len(values) < 2:
        return Text("—", style=LABEL)
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    blocks = " ▁▂▃▄▅▆▇█"
    result = Text()
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]
    for i, v in enumerate(sampled):
        idx = int((v - mn) / rng * 8)
        if i == 0:
            color = LABEL
        elif v > sampled[i - 1]:
            color = GAIN
        elif v < sampled[i - 1]:
            color = LOSS_HI
        else:
            color = LABEL
        result.append(blocks[idx], style=color)
    return result




def _load_nav_log() -> pd.DataFrame:
    return pd.read_csv(PAPER_NAV_LOG_FILE) if PAPER_NAV_LOG_FILE.exists() else pd.DataFrame()

def _load_trade_log() -> pd.DataFrame:
    return pd.read_csv(PAPER_TRADE_LOG_FILE) if PAPER_TRADE_LOG_FILE.exists() else pd.DataFrame()


_SIGNAL_LABEL_MAP = {
    "fundamental": "Fundamental",
    "quality": "Quality",
    "momentum": "Momentum",
    "residual_momentum": "Residual Momentum",
    "xsec_momentum": "Cross-Sectional Momentum",
    "mean_reversion": "Mean Reversion",
    "liquidity": "Liquidity",
    "volume": "Volume",
    "volume_breakout": "Volume Breakout",
    "accumulation": "Accumulation",
    "corporate_action": "Corporate Action",
    "dividend": "Dividend",
    "sentiment": "Sentiment",
    "nlp_sentiment": "NLP Sentiment",
    "disposition": "Disposition",
    "anchoring_52wk": "52-Week Anchoring",
    "pairs_trade": "Pairs Trade",
    "macro_remittance": "Macro Remittance",
    "satellite_hydro": "Satellite Hydro",
    "settlement_pressure": "Settlement Pressure",
    "manual": "Manual",
    "tms": "TMS",
    "unknown": "Unknown",
}


def _signal_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Unknown"
    norm = raw.lower().replace("-", "_").replace(" ", "_")
    return _SIGNAL_LABEL_MAP.get(norm, raw.replace("_", " ").replace("-", " ").title())


def _signal_code(value: str) -> str:
    norm = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    code_map = {
        "fundamental": "F",
        "mean_reversion": "MR",
        "xsec_momentum": "XS",
        "accumulation": "A",
        "volume": "V",
        "quality": "Q",
        "low_vol": "LV",
        "residual_momentum": "RM",
            "tms": "TMS",
    }
    if not norm:
        return "?"
    return code_map.get(norm, norm[:3].upper())

def _get_regime(md: MD) -> str:
    """Get market regime from simple_backtest."""
    try:
        from backend.backtesting.simple_backtest import load_all_prices, compute_market_regime
        conn = _db()
        prices_df = load_all_prices(conn)
        conn.close()
        today = datetime.strptime(md.latest, "%Y-%m-%d")
        return compute_market_regime(prices_df, today)
    except Exception:
        return "unknown"

def _compute_portfolio_stats(md: MD) -> dict:
    """Compute full portfolio stats for portfolio/risk tabs."""
    ltps = md.ltps()
    quote_map = {}
    if not md.quotes.empty:
        quote_map = {
            str(r.symbol): {
                "ltp": float(getattr(r, "ltp", 0) or 0),
                "prev_close": float(getattr(r, "prev_close", 0) or 0),
                "pc": float(getattr(r, "pc", 0) or 0),
            }
            for r in md.quotes.itertuples()
        }
    port = load_port()
    nav_log = _load_nav_log()
    trade_log = _load_trade_log()

    positions = []
    total_cost = total_value = 0.0
    total_prev_value = 0.0
    sector_exposure = {}

    if not port.empty:
        for _, r in port.iterrows():
            sym = str(r["Symbol"]); qty = int(r["Quantity"])
            entry = float(r["Buy_Price"])
            cost = float(r.get("Total_Cost_Basis", entry * qty))
            quote = quote_map.get(sym, {})
            cur = ltps.get(sym) or float(r.get("Last_LTP") or entry)
            prev_close = float(quote.get("prev_close") or 0) or cur
            val = cur * qty; pnl = val - cost
            ret = pnl / cost * 100 if cost else 0
            day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
            day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
            entry_dt = str(r.get("Entry_Date", ""))[:10]
            days = 0
            try:
                days = (datetime.now() - datetime.strptime(entry_dt, "%Y-%m-%d")).days
            except Exception:
                pass
            total_cost += cost; total_value += val
            total_prev_value += prev_close * qty

            # Sector
            try:
                from backend.backtesting.simple_backtest import get_symbol_sector
                sec = get_symbol_sector(sym) or "Other"
            except Exception:
                sec = "Other"
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val

            positions.append({
                "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                "signal": _signal_label(str(r.get("Signal_Type", ""))),
                "date": entry_dt, "days": days, "sector": sec,
            })

    # NAV
    cash = _load_manual_paper_cash(total_cost, nav_log)
    nav = cash + total_value
    prev_nav = cash + total_prev_value
    day_pnl_total = nav - prev_nav
    day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
    baseline_nav = float(INITIAL_CAPITAL)
    if not nav_log.empty and "NAV" in nav_log.columns:
        try:
            baseline_nav = float(nav_log.iloc[0]["NAV"] or INITIAL_CAPITAL)
        except Exception:
            baseline_nav = float(INITIAL_CAPITAL)
    total_return = (nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    # Realized P&L and fees paid from trade log
    realized = 0.0
    fees_paid = 0.0
    if not trade_log.empty and "PnL" in trade_log.columns:
        realized = trade_log["PnL"].sum()
    if not trade_log.empty and "Fees" in trade_log.columns:
        fees_paid = float(pd.to_numeric(trade_log["Fees"], errors="coerce").fillna(0.0).sum())
    elif not port.empty and "Buy_Fees" in port.columns:
        fees_paid = float(pd.to_numeric(port["Buy_Fees"], errors="coerce").fillna(0.0).sum())

    gross_nav = nav + fees_paid
    gross_return = (gross_nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    # Max drawdown from NAV log
    max_dd = 0.0; peak_nav = baseline_nav; dd_date = ""
    if not nav_log.empty and "NAV" in nav_log.columns:
        for _, nr in nav_log.iterrows():
            n = float(nr["NAV"])
            if n > peak_nav:
                peak_nav = n
            dd = (n - peak_nav) / peak_nav * 100
            if dd < max_dd:
                max_dd = dd
                dd_date = str(nr.get("Date", ""))[:10]

    # NEPSE benchmark return
    nepse_ret = 0.0
    if len(md.nepse) >= 2:
        ni = md.nepse.iloc[0]["close"]
        # Get NEPSE at portfolio start
        try:
            conn = _db()
            start = pd.read_sql_query(
                "SELECT close FROM stock_prices WHERE symbol='NEPSE' AND date>=? "
                "ORDER BY date LIMIT 1", conn, params=("2026-02-09",))
            conn.close()
            if not start.empty:
                nepse_ret = (ni - start.iloc[0]["close"]) / start.iloc[0]["close"] * 100
        except Exception:
            pass

    # Concentration
    positions.sort(key=lambda x: x["val"], reverse=True)
    top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0

    # Winners/losers
    winners = [p for p in positions if p["pnl"] > 0]
    losers = [p for p in positions if p["pnl"] < 0]

    # Holding age buckets
    age_0_5 = sum(1 for p in positions if p["days"] <= 5)
    age_6_15 = sum(1 for p in positions if 6 <= p["days"] <= 15)
    age_16 = sum(1 for p in positions if p["days"] > 15)

    return {
        "positions": positions,
        "total_cost": total_cost, "total_value": total_value,
        "cash": cash, "nav": nav, "total_return": total_return,
        "gross_nav": gross_nav, "gross_return": gross_return, "fees_paid": fees_paid,
        "day_pnl": day_pnl_total, "day_ret": day_ret_total,
        "realized": realized, "unrealized": total_value - total_cost,
        "max_dd": max_dd, "dd_date": dd_date, "peak_nav": peak_nav,
        "nepse_ret": nepse_ret, "alpha": total_return - nepse_ret,
        "n_positions": len(positions),
        "sector_exposure": sector_exposure,
        "top3_conc": top3_conc,
        "winners": winners, "losers": losers,
        "age_0_5": age_0_5, "age_6_15": age_6_15, "age_16": age_16,
        "trade_log": trade_log, "nav_log": nav_log,
    }


def _compute_account_portfolio_stats(md: MD, account_dir: Path) -> dict:
    """Compute portfolio stats from account-specific files (paper_portfolio.csv etc.)."""
    ltps = md.ltps()
    quote_map = {}
    if not md.quotes.empty:
        quote_map = {
            str(r.symbol): {
                "ltp": float(getattr(r, "ltp", 0) or 0),
                "prev_close": float(getattr(r, "prev_close", 0) or 0),
                "pc": float(getattr(r, "pc", 0) or 0),
            }
            for r in md.quotes.itertuples()
        }

    _port_path = account_dir / "paper_portfolio.csv"
    _nav_path = account_dir / "paper_nav_log.csv"
    _tl_path = account_dir / "paper_trade_log.csv"
    _state_path = account_dir / "paper_state.json"

    port = pd.read_csv(_port_path) if _port_path.exists() else pd.DataFrame()
    nav_log = pd.read_csv(_nav_path) if _nav_path.exists() else pd.DataFrame()
    trade_log = pd.read_csv(_tl_path) if _tl_path.exists() else pd.DataFrame()

    positions = []
    total_cost = total_value = 0.0
    total_prev_value = 0.0
    sector_exposure = {}

    if not port.empty:
        for _, r in port.iterrows():
            sym = str(r["Symbol"]); qty = int(r["Quantity"])
            entry = float(r["Buy_Price"])
            cost = float(r.get("Total_Cost_Basis", entry * qty))
            quote = quote_map.get(sym, {})
            cur = ltps.get(sym) or float(r.get("Last_LTP") or entry)
            prev_close = float(quote.get("prev_close") or 0) or cur
            val = cur * qty; pnl = val - cost
            ret = pnl / cost * 100 if cost else 0
            day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
            day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
            entry_dt = str(r.get("Entry_Date", ""))[:10]
            days = 0
            try:
                days = (datetime.now() - datetime.strptime(entry_dt, "%Y-%m-%d")).days
            except Exception:
                pass
            total_cost += cost; total_value += val
            total_prev_value += prev_close * qty
            try:
                from backend.backtesting.simple_backtest import get_symbol_sector
                sec = get_symbol_sector(sym) or "Other"
            except Exception:
                sec = "Other"
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val
            positions.append({
                "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                "signal": _signal_label(str(r.get("Signal_Type", ""))),
                "date": entry_dt, "days": days, "sector": sec,
            })

    # Cash from paper_state.json
    cash = float(INITIAL_CAPITAL)
    if _state_path.exists():
        try:
            _ps = _json.loads(_state_path.read_text())
            _c = float(_ps.get("cash", 0))
            if _c >= 0:
                cash = _c
        except Exception:
            pass
    elif total_cost > 0:
        cash = max(0.0, float(INITIAL_CAPITAL) - total_cost)

    nav = cash + total_value
    baseline_nav = float(INITIAL_CAPITAL)
    if not nav_log.empty and "NAV" in nav_log.columns:
        try:
            baseline_nav = float(nav_log.iloc[0]["NAV"] or INITIAL_CAPITAL)
        except Exception:
            pass
    total_return = (nav - baseline_nav) / baseline_nav * 100 if baseline_nav > 0 else 0.0

    realized = 0.0
    fees_paid = 0.0
    if not trade_log.empty and "PnL" in trade_log.columns:
        realized = float(pd.to_numeric(trade_log["PnL"], errors="coerce").fillna(0).sum())
    if not trade_log.empty and "Fees" in trade_log.columns:
        fees_paid = float(pd.to_numeric(trade_log["Fees"], errors="coerce").fillna(0).sum())

    max_dd = 0.0; peak_nav = baseline_nav; dd_date = ""
    if not nav_log.empty and "NAV" in nav_log.columns:
        for _, nr in nav_log.iterrows():
            n = float(nr["NAV"])
            if n > peak_nav:
                peak_nav = n
            dd = (n - peak_nav) / peak_nav * 100
            if dd < max_dd:
                max_dd = dd
                dd_date = str(nr.get("Date", ""))[:10]

    prev_nav = cash + total_prev_value
    day_pnl_total = nav - prev_nav
    day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
    positions.sort(key=lambda x: x["val"], reverse=True)
    top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0
    winners = [p for p in positions if p["pnl"] > 0]
    losers = [p for p in positions if p["pnl"] < 0]
    age_0_5 = sum(1 for p in positions if p["days"] <= 5)
    age_6_15 = sum(1 for p in positions if 6 <= p["days"] <= 15)
    age_16 = sum(1 for p in positions if p["days"] > 15)

    return {
        "positions": positions,
        "total_cost": total_cost, "total_value": total_value,
        "cash": cash, "nav": nav, "total_return": total_return,
        "gross_nav": nav + fees_paid, "gross_return": total_return, "fees_paid": fees_paid,
        "day_pnl": day_pnl_total, "day_ret": day_ret_total,
        "realized": realized, "unrealized": total_value - total_cost,
        "max_dd": max_dd, "dd_date": dd_date, "peak_nav": peak_nav,
        "nepse_ret": 0, "alpha": total_return,
        "n_positions": len(positions), "sector_exposure": sector_exposure,
        "top3_conc": top3_conc, "winners": winners, "losers": losers,
        "age_0_5": age_0_5, "age_6_15": age_6_15, "age_16": age_16,
        "trade_log": trade_log, "nav_log": nav_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class MarketPanel(Widget):
    """Titled panel with a DataTable inside."""

    def __init__(self, panel_title: str = "", title_color: str = AMBER, **kw):
        super().__init__(**kw)
        self._panel_title = panel_title
        self._title_color = title_color

    def compose(self) -> ComposeResult:
        yield Static(self._panel_title, classes="panel-title")
        yield DataTable(zebra_stripes=True, cursor_type="row")

    def set_data(self, columns: list[tuple[str, str]], rows: list[list]):
        dt = self.query_one(DataTable)
        dt.clear(columns=True)
        for label, key in columns:
            dt.add_column(label, key=key)
        for row in rows:
            dt.add_row(*row)

    def update_title(self, title: str):
        self.query_one(".panel-title", Static).update(title)


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL DIALOGS
# ═══════════════════════════════════════════════════════════════════════════════

class ModalDialog(ModalScreen[dict | None]):
    """Buy/Sell/Lookup modal dialog."""

    DEFAULT_CSS = """
    ModalDialog { align: center middle; }
    """

    def __init__(
        self,
        mode: str = "buy",
        *,
        initial_symbol: str = "",
        initial_shares: str = "",
        initial_price: str = "",
        default_slippage: str = "2.0",
        holdings_positions: list[dict] | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.initial_symbol = initial_symbol
        self.initial_shares = initial_shares
        self.initial_price = initial_price
        self.default_slippage = default_slippage
        self.holdings_summary = _format_sell_holdings_summary(holdings_positions)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog-box", classes=f"dialog-shell dialog-{self.mode}"):
            title_map = {"buy": "BUY", "sell": "SELL", "lookup": "LOOKUP"}
            yield Static(title_map.get(self.mode, ""), classes="dialog-title")
            if self.mode == "buy":
                yield Static("Create a paper ticket for the daily order book.", id="dialog-kicker")
            elif self.mode == "sell":
                yield Static("Holdings available to sell", id="dialog-kicker")
                yield Static(self.holdings_summary, id="dialog-holdings")
            yield Input(id="inp-symbol", placeholder="Symbol e.g. NABIL")
            if self.mode != "lookup":
                yield Input(id="inp-shares",
                            placeholder="Shares" if self.mode == "buy" else "Shares (or all)")
                yield Input(id="inp-price", placeholder="Price (blank=last close)")
                yield Input(id="inp-slippage", placeholder="Slippage %", value=self.default_slippage)
            yield Static("", id="dialog-result")
            with Horizontal(id="dialog-buttons"):
                yield Button("Submit", id="btn-submit", variant="success")
                yield Button("Cancel", id="btn-cancel", variant="error")

    def on_mount(self) -> None:
        symbol = self.query_one("#inp-symbol", Input)
        symbol.value = self.initial_symbol
        if self.mode != "lookup":
            self.query_one("#inp-shares", Input).value = self.initial_shares
            self.query_one("#inp-price", Input).value = self.initial_price
        symbol.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self._submit()
        else:
            self.dismiss(None)

    def _submit(self):
        sym = self.query_one("#inp-symbol", Input).value.strip().upper()
        if not sym:
            self.query_one("#dialog-result", Static).update(
                Text("Symbol required", style=LOSS_HI))
            return
        if self.mode == "lookup":
            self.dismiss({"symbol": sym})
        else:
            shares = self.query_one("#inp-shares", Input).value.strip()
            price = self.query_one("#inp-price", Input).value.strip()
            slippage = self.query_one("#inp-slippage", Input).value.strip()
            if not shares:
                self.query_one("#dialog-result", Static).update(
                    Text("Shares required", style=LOSS_HI))
                return
            self.dismiss({"symbol": sym, "shares": shares, "price": price, "slippage": slippage})

    def key_escape(self) -> None:
        self.dismiss(None)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SELECT SCREEN — shown on startup
# ═══════════════════════════════════════════════════════════════════════════════


class ModeSelectScreen(ModalScreen):
    """Startup screen for the paper trading workspace."""

    DEFAULT_CSS = """
    ModeSelectScreen {
        align: center middle;
        background: #060606;
    }
    #mode-box {
        width: 64;
        height: auto;
        max-height: 20;
        background: #0a0a0a;
        border: none;
        padding: 0;
    }
    #mode-header {
        height: 4; width: 100%; padding: 1 2;
        background: #0a0a0a;
    }
    .mode-brand {
        width: 100%; text-align: center;
        color: #888888; height: 1;
    }
    .mode-version {
        width: 100%; text-align: center;
        color: #5d6670; height: 1;
    }
    #mode-options {
        height: auto; width: 100%;
        padding: 1 3;
        background: #0a0a0a;
    }
    .mode-row {
        width: 100%; height: 1;
        padding: 0 2;
        margin: 0 0;
    }
    #mode-footer {
        height: 3; width: 100%;
        padding: 1 2;
        background: #0a0a0a;
    }
    .mode-keys {
        width: 100%; text-align: center;
        height: auto;
    }
    """

    _selected: int = 0
    _options = [
        ("paper", "PAPER", "Paper trading workspace and local portfolio"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="mode-box"):
            with Vertical(id="mode-header"):
                yield Static(
                    Text.from_markup(
                        "[bold #ffaf00]N E P S E[/]  [bold #e8e8e8]Q U A N T[/]"
                    ),
                    classes="mode-brand",
                )
                yield Static(
                    Text.from_markup("[#5d6670]Terminal v3  ◆  Paper Trading[/]"),
                    classes="mode-version",
                )
            with Vertical(id="mode-options"):
                yield Static("", id="mode-opt-0", classes="mode-row")
            with Vertical(id="mode-footer"):
                yield Static(
                    Text.from_markup(
                        "[bold #ffaf00]\u2191\u2193[/][#555555] select   [/]"
                        "[bold #ffaf00]ENTER[/][#555555] confirm   [/]"
                        "[bold #ffaf00]Q[/][#555555] quit[/]"
                    ),
                    classes="mode-keys",
                )

    def on_mount(self) -> None:
        self._render_options()

    def _render_options(self) -> None:
        for i, (key, label, desc) in enumerate(self._options):
            widget = self.query_one(f"#mode-opt-{i}", Static)
            if i == self._selected:
                t = Text()
                t.append("  \u25b6 ", style="bold #ffaf00")
                t.append(f"{label:<10}", style="bold #e8e8e8")
                t.append(desc, style="#888888")
                widget.update(t)
            else:
                t = Text()
                t.append("    ", style="")
                t.append(f"{label:<10}", style="#555555")
                t.append(desc, style="#333333")
                widget.update(t)

    def key_up(self) -> None:
        self._selected = max(0, self._selected - 1)
        self._render_options()

    def key_down(self) -> None:
        self._selected = min(len(self._options) - 1, self._selected + 1)
        self._render_options()

    def key_enter(self) -> None:
        self.dismiss(self._options[self._selected][0])

    def key_q(self) -> None:
        self.app.exit()

    def key_escape(self) -> None:
        self.app.exit()


class TMSLoginScreen(ModalScreen):
    """Credential entry screen for live TMS auto-login."""

    DEFAULT_CSS = """
    TMSLoginScreen {
        align: center middle;
        background: #060606;
    }
    #tms-box {
        width: 56;
        height: auto;
        max-height: 18;
        background: #0a0a0a;
        border: tall #333333;
        padding: 1 2;
    }
    .tms-title {
        width: 100%; text-align: center;
        color: #ffaf00; text-style: bold;
        height: 1; margin-bottom: 1;
    }
    .tms-label {
        width: 100%; height: 1;
        color: #666666; padding: 0 1;
    }
    .tms-input {
        width: 100%;
        background: #111111;
        border: none;
        border-bottom: solid #222222;
        color: #e8e8e8;
        margin-bottom: 1;
    }
    .tms-input:focus {
        border-bottom: solid #ffaf00;
    }
    .tms-status {
        width: 100%; text-align: center;
        height: 1; color: #666666;
        margin-top: 1;
    }
    .tms-hint {
        width: 100%; text-align: center;
        height: 1; margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="tms-box"):
            yield Static(
                Text.from_markup("[bold #ffaf00]TMS19 LOGIN[/]"),
                classes="tms-title",
            )
            yield Static("Client ID / Username", classes="tms-label")
            yield Input(id="tms-user", placeholder="e.g. 12345678", classes="tms-input")
            yield Static("Password", classes="tms-label")
            yield Input(id="tms-pass", placeholder="", password=True, classes="tms-input")
            yield Static("", id="tms-status", classes="tms-status")
            yield Static(
                Text.from_markup(
                    "[bold #ffaf00]ENTER[/][#555555] login   [/]"
                    "[bold #ffaf00]ESC[/][#555555] back[/]"
                ),
                classes="tms-hint",
            )

    def on_mount(self) -> None:

        settings = load_tms_settings()
        user_env = settings.username or ""
        pass_env = settings.password or ""
        if user_env:
            self.query_one("#tms-user", Input).value = user_env
        if pass_env:
            self.query_one("#tms-pass", Input).value = pass_env
        if user_env and pass_env:
            # Both pre-filled, focus submit hint
            self.query_one("#tms-pass", Input).focus()
        else:
            self.query_one("#tms-user", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "tms-user":
            self.query_one("#tms-pass", Input).focus()
        elif event.input.id == "tms-pass":
            self._do_login()

    def key_escape(self) -> None:
        self.dismiss(None)

    def _do_login(self) -> None:
        user = self.query_one("#tms-user", Input).value.strip()
        pwd = self.query_one("#tms-pass", Input).value.strip()
        if not user or not pwd:
            self.query_one("#tms-status", Static).update(
                Text("Enter both client ID and password", style="#ff4444")
            )
            return
        self.query_one("#tms-status", Static).update(
            Text("Connecting to TMS19...", style="#ffaf00")
        )
        self.dismiss({"username": user, "password": pwd})


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND PALETTE (Bloomberg GO bar)
# ═══════════════════════════════════════════════════════════════════════════════

class CommandPalette(ModalScreen[dict | None]):
    """Bloomberg-style GO bar. Type symbol name → lookup, tab name → switch, action → execute."""

    DEFAULT_CSS = """
    CommandPalette {
        align: center top;
        background: rgba(0, 0, 0, 0.80);
    }
    #cmd-box {
        width: 64;
        height: auto;
        max-height: 22;
        background: #090d12;
        border: solid #3f474f;
        margin-top: 2;
        padding: 0;
    }
    #cmd-header {
        height: 1;
        width: 100%;
        background: #090d12;
        layout: horizontal;
        padding: 0 1;
    }
    #cmd-header-label {
        width: auto;
        height: 1;
        color: #f2b94b;
        text-style: bold;
        padding: 0 1 0 0;
    }
    #cmd-input {
        width: 1fr;
        background: #0c1116;
        border: none;
        color: #e8edf3;
        padding: 0 1;
        height: 1;
    }
    #cmd-input:focus {
        background: #0c1116;
        color: #fff6de;
        border: none;
    }
    #cmd-results {
        height: auto;
        max-height: 18;
        width: 100%;
        background: #090d12;
        padding: 0;
        border-top: solid #1a1f25;
    }
    .cmd-row {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: #6b7785;
    }
    .cmd-row-selected {
        height: 1;
        width: 100%;
        padding: 0 1;
        background: #111820;
        color: #f2b94b;
    }
    #cmd-hint {
        height: 1;
        width: 100%;
        background: #090d12;
        color: #3a4450;
        padding: 0 1;
        border-top: solid #1a1f25;
    }
    """

    _selected: int = 0
    _items: list[dict] = []
    _filtered: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="cmd-box"):
            with Horizontal(id="cmd-header"):
                yield Static("GO", id="cmd-header-label")
                yield Input(id="cmd-input", placeholder="symbol, tab, or action")
            yield Vertical(id="cmd-results")
            yield Static(" ↑↓ Navigate  ENTER Execute  ESC Close", id="cmd-hint")

    def on_mount(self) -> None:
        self._build_items()
        self._selected = 0
        self._filter("")
        self.query_one("#cmd-input", Input).focus()

    def _build_items(self) -> None:
        self._items = []
        # Tab navigation
        tabs = [
            ("1 MARKET", "tab", "market", "Market overview — gainers, losers, volume"),
            ("2 PORTFOLIO", "tab", "portfolio", "Portfolio holdings, NAV, risk"),
            ("3 SIGNALS", "tab", "signals", "Trading signals, screener & calendar"),
            ("4 LOOKUP", "tab", "lookup", "Stock lookup & charts"),
            ("5 AGENTS", "tab", "agents", "AI agent analysis & chat"),
            ("6 ORDERS", "tab", "orders", "Order management"),
            ("7 WATCHLIST", "tab", "watchlist", "Your watched stocks"),
            ("8 RATES & COMMODITIES", "tab", "kalimati", "FX, metals and local commodity prices"),
            ("9 ACCOUNT", "tab", "account", "Paper account profiles and runtime"),
            ("0 STRATEGIES", "tab", "strategies", "Saved strategies, bindings and backtests"),
        ]
        for name, kind, target, desc in tabs:
            self._items.append({"name": name, "kind": kind, "target": target, "desc": desc})
        # Actions
        actions = [
            ("BUY", "action", "buy", "Open buy dialog"),
            ("SELL", "action", "sell", "Open sell dialog"),
            ("REFRESH", "action", "refresh", "Refresh all market data"),
            ("AGENT", "action", "agent", "Run AI agent analysis"),
        ]
        for name, kind, target, desc in actions:
            self._items.append({"name": name, "kind": kind, "target": target, "desc": desc})
        # Stock symbols from market data (will be populated dynamically)
        try:
            conn = _db()
            import sqlite3
            syms = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM stock_prices WHERE symbol != 'NEPSE' "
                "ORDER BY symbol").fetchall()]
            conn.close()
            for sym in syms:
                self._items.append({"name": sym, "kind": "stock", "target": sym,
                                    "desc": "Look up stock"})
        except Exception:
            pass

    def _filter(self, query: str) -> None:
        q = query.strip().upper()
        if not q:
            # Show tabs and actions first
            self._filtered = [i for i in self._items if i["kind"] != "stock"][:15]
        else:
            # Fuzzy match: items starting with query first, then contains
            starts = [i for i in self._items if i["name"].upper().startswith(q)]
            contains = [i for i in self._items if q in i["name"].upper() and i not in starts]
            desc_match = [i for i in self._items if q in i["desc"].upper()
                          and i not in starts and i not in contains]
            self._filtered = (starts + contains + desc_match)[:15]
        self._selected = min(self._selected, max(0, len(self._filtered) - 1))
        self._render_results()

    def _render_results(self) -> None:
        container = self.query_one("#cmd-results", Vertical)
        container.remove_children()
        for i, item in enumerate(self._filtered):
            kind_icon = {"tab": "◧", "action": "▶", "stock": "◆"}.get(item["kind"], "·")
            kind_color = {"tab": CYAN, "action": GAIN_HI, "stock": AMBER}.get(item["kind"], LABEL)
            t = Text()
            if i == self._selected:
                t.append(f"  ▸ {kind_icon} ", style=f"bold {kind_color}")
                t.append(f"{item['name']:<16}", style=f"bold {WHITE}")
                t.append(item["desc"], style=LABEL)
                cls = "cmd-row-selected"
            else:
                t.append(f"    {kind_icon} ", style=kind_color)
                t.append(f"{item['name']:<16}", style=DIM)
                t.append(item["desc"], style=DIM)
                cls = "cmd-row"
            container.mount(Static(t, classes=cls))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "cmd-input":
            self._selected = 0
            self._filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "cmd-input":
            self._execute()

    def key_up(self) -> None:
        self._selected = max(0, self._selected - 1)
        self._render_results()

    def key_down(self) -> None:
        self._selected = min(len(self._filtered) - 1, self._selected + 1)
        self._render_results()

    def key_escape(self) -> None:
        self.dismiss(None)

    def _execute(self) -> None:
        if not self._filtered:
            # Treat input as stock symbol
            val = self.query_one("#cmd-input", Input).value.strip().upper()
            if val:
                self.dismiss({"kind": "stock", "target": val})
            return
        item = self._filtered[self._selected]
        self.dismiss(item)


class WatchlistAddScreen(ModalScreen[dict | str | None]):
    """Bloomberg-style watchlist picker with live suggestions."""

    DEFAULT_CSS = """
    WatchlistAddScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.88);
    }
    #wl-add-box {
        width: 76;
        height: auto;
        max-height: 24;
        border: solid #5d6670;
        background: #090b0e;
        padding: 0;
    }
    #wl-add-title {
        height: 1;
        width: 100%;
        background: #17191d;
        color: #ffaf00;
        text-style: bold;
        padding: 0 2;
    }
    #wl-add-row {
        width: 100%;
        height: 3;
        background: #101419;
        border-top: solid #232a31;
        border-bottom: solid #232a31;
        padding: 0 2;
        layout: horizontal;
        content-align: left middle;
    }
    #wl-add-label {
        height: 1;
        width: 8;
        padding-right: 2;
        color: #8fa0b1;
        text-style: bold;
        content-align: left middle;
    }
    #wl-add-input {
        width: 1fr;
        height: 1;
        background: #101419;
        border: none;
        color: #f6fbff;
        padding: 0 1;
        content-align: left middle;
    }
    #wl-add-input:focus {
        background: #151a20;
        color: #ffffff;
        border: none;
    }
    #wl-add-query {
        height: 1;
        width: 100%;
        background: #0d1014;
        color: #8fa0b1;
        padding: 0 2;
        border-bottom: solid #1b2128;
    }
    #wl-add-results {
        height: auto;
        max-height: 17;
        width: 100%;
        background: #0b0d10;
        padding: 1 0;
    }
    .wl-add-row-item {
        height: 1;
        width: 100%;
        padding: 0 2;
        color: #7e8791;
    }
    .wl-add-row-selected {
        height: 1;
        width: 100%;
        padding: 0 2;
        background: #191d22;
        color: #ffcf70;
        text-style: bold;
    }
    #wl-add-hint {
        height: 1;
        width: 100%;
        background: #101318;
        color: #56616d;
        padding: 0 2;
        border-top: solid #1b2128;
    }
    """

    _selected: int = 0
    _items: list[dict] = []
    _filtered: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="wl-add-box"):
            yield Static("ADD TO WATCHLIST", id="wl-add-title")
            with Horizontal(id="wl-add-row"):
                yield Static("SEARCH", id="wl-add-label")
                yield Input(id="wl-add-input", placeholder="NABIL, USD, Gold / Tola, Petrol...")
            yield Static("", id="wl-add-query")
            yield Vertical(id="wl-add-results")
            yield Static("↑↓ Navigate   ENTER Add   ESC Cancel   Stocks, forex, macro, commodities", id="wl-add-hint")

    def on_mount(self) -> None:
        self._build_items()
        self._selected = 0
        self._filter("")
        self._update_query_preview("")
        self.query_one("#wl-add-input", Input).focus()

    def _build_items(self) -> None:
        self._items = []
        try:
            conn = _db()
            symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM stock_prices WHERE symbol != 'NEPSE' ORDER BY symbol"
            ).fetchall()]
            conn.close()
            for sym in symbols:
                self._items.append({
                    "name": sym,
                    "kind": "stock",
                    "target": _stock_watchlist_entry(sym),
                    "desc": "NEPSE stock",
                })
        except Exception:
            pass

        app = self.app
        kalimati_rows = list(getattr(app, "_kalimati_rows", []) or [])
        macro_rates = dict(getattr(app, "_macro_rates", {}) or {})
        if not kalimati_rows:
            try:
                kalimati_rows = list(get_kalimati_display_rows() or [])
            except Exception:
                kalimati_rows = []
        if not macro_rates:
            try:
                metals = _fetch_gold_silver_prices()
                indicators: list[dict] = []
                if metals:
                    gold = float(metals.get("gold_per_tola") or 0)
                    silver = float(metals.get("silver_per_tola") or 0)
                    if gold > 0:
                        indicators.append({"item": "Gold / Tola", "group": "Metals", "unit": "NPR/tola"})
                    if silver > 0:
                        indicators.append({"item": "Silver / Tola", "group": "Metals", "unit": "NPR/tola"})
                forex_rows = _fetch_nrb_forex_rates(("USD", "EUR", "GBP", "INR", "CNY", "JPY"))
                macro_rates = {"indicators": indicators, "forex_rows": forex_rows}
            except Exception:
                macro_rates = {}

        for row in kalimati_rows:
            label = str(row.get("name_english") or "").strip()
            if not label:
                continue
            self._items.append({
                "name": label,
                "kind": "commodity",
                "target": {
                    "kind": "commodity",
                    "key": f"commodity:{label}",
                    "label": label,
                    "unit": str(row.get("unit") or ""),
                },
                "desc": f"Kalimati commodity  {row.get('unit') or ''}".strip(),
            })

        for row in list(macro_rates.get("indicators", []) or []):
            label = str(row.get("item") or "").strip()
            if not label:
                continue
            self._items.append({
                "name": label,
                "kind": "macro",
                "target": {
                    "kind": "macro",
                    "key": f"macro:{label}",
                    "label": label,
                    "group": str(row.get("group") or ""),
                },
                "desc": f"{row.get('group') or 'Macro'}  {row.get('unit') or ''}".strip(),
            })

        for row in list(macro_rates.get("forex_rows", []) or []):
            code = str(row.get("currency_code") or "").strip().upper()
            name = str(row.get("currency_name") or "").strip()
            if not code:
                continue
            self._items.append({
                "name": code,
                "kind": "forex",
                "target": {
                    "kind": "forex",
                    "key": f"forex:{code}",
                    "label": code,
                    "currency_name": name,
                },
                "desc": name or "NRB forex rate",
            })

    def _filter(self, query: str) -> None:
        q = query.strip().upper()
        if not q:
            priority = ("stock", "forex", "macro", "commodity")
            ordered: list[dict] = []
            for kind in priority:
                ordered.extend([item for item in self._items if item["kind"] == kind][:4])
            self._filtered = ordered[:12]
        else:
            exact = [item for item in self._items if item["name"].upper() == q]
            starts = [item for item in self._items if item["name"].upper().startswith(q) and item not in exact]
            contains = [item for item in self._items if q in item["name"].upper() and item not in exact and item not in starts]
            desc = [item for item in self._items if q in item["desc"].upper() and item not in exact and item not in starts and item not in contains]
            self._filtered = (exact + starts + contains + desc)[:12]
        self._selected = min(self._selected, max(0, len(self._filtered) - 1))
        self._render_results()

    def _render_results(self) -> None:
        container = self.query_one("#wl-add-results", Vertical)
        container.remove_children()
        if not self._filtered:
            container.mount(Static(Text("  No matches", style=DIM), classes="wl-add-row-item"))
            return
        kind_icon = {"stock": "◆", "forex": "FX", "macro": "●", "commodity": "◧"}
        kind_style = {"stock": AMBER, "forex": CYAN, "macro": GAIN_HI, "commodity": WHITE}
        for idx, item in enumerate(self._filtered):
            t = Text()
            icon = kind_icon.get(item["kind"], "·")
            color = kind_style.get(item["kind"], LABEL)
            if idx == self._selected:
                t.append(f"  ▸ {icon} ", style=f"bold {color}")
                t.append(f"{item['name']:<20}", style=f"bold {WHITE}")
                t.append(item["desc"][:42], style=LABEL)
                cls = "wl-add-row-selected"
            else:
                t.append(f"    {icon} ", style=color)
                t.append(f"{item['name']:<20}", style=DIM)
                t.append(item["desc"][:42], style=DIM)
                cls = "wl-add-row-item"
            container.mount(Static(t, classes=cls))

    def _update_query_preview(self, query: str) -> None:
        widget = self.query_one("#wl-add-query", Static)
        clean = query.strip()
        if clean:
            widget.update(Text.from_markup(
                f"[#6e7c89]QUERY[/] [bold {WHITE}]{_escape_markup(clean)}[/]   "
                f"[#6e7c89]MATCHES[/] [bold {AMBER}]{len(self._filtered)}[/]"
            ))
        else:
            widget.update(Text.from_markup(
                f"[#6e7c89]QUERY[/] [#55616d]Start typing to search stocks, FX, macro and commodities[/]"
            ))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "wl-add-input":
            self._selected = 0
            self._filter(event.value)
            self._update_query_preview(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "wl-add-input":
            self._submit()

    def key_up(self) -> None:
        self._selected = max(0, self._selected - 1)
        self._render_results()

    def key_down(self) -> None:
        self._selected = min(len(self._filtered) - 1, self._selected + 1)
        self._render_results()

    def _submit(self) -> None:
        raw = self.query_one("#wl-add-input", Input).value.strip()
        if self._filtered:
            self.dismiss(self._filtered[self._selected]["target"])
            return
        if raw:
            self.dismiss(raw.upper())
        else:
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class LookupScreen(ModalScreen[dict | None]):
    """Compact symbol lookup prompt."""

    DEFAULT_CSS = """
    LookupScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.82);
    }
    #lookup-box {
        width: 38;
        height: 3;
        border: solid #3f474f;
        background: #090d12;
        padding: 0 1;
        layout: horizontal;
        align: left middle;
    }
    #lookup-label {
        color: #f2b94b;
        text-style: bold;
        height: 1;
        width: auto;
        padding: 0 1 0 0;
    }
    #lookup-input {
        width: 1fr;
        height: 1;
        background: #0c1116;
        border: none;
        color: #e8edf3;
        padding: 0 1;
    }
    #lookup-input:focus {
        background: #0c1116;
        color: #fff6de;
        border: none;
    }
    #lookup-hint {
        height: 1;
        width: auto;
        color: #4a5562;
        padding: 0 0 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="lookup-box"):
            yield Static("LOOKUP", id="lookup-label")
            yield Input(id="lookup-input", placeholder="NABIL")
            yield Static("↵ GO  ESC ✕", id="lookup-hint")

    def on_mount(self) -> None:
        self.query_one("#lookup-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        val = self.query_one("#lookup-input", Input).value.strip().upper()
        if val:
            self.dismiss({"symbol": val})
        else:
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

TAB_NAMES = {
    "market": "1", "portfolio": "2", "signals": "3",
    "lookup": "4", "agents": "5", "orders": "6",
    "watchlist": "7", "kalimati": "8", "account": "9", "strategies": "0",
}

TAB_LABELS = {
    "market": "MARKET",
    "portfolio": "PORTFOLIO",
    "signals": "SIGNALS",
    "lookup": "LOOKUP",
    "agents": "AGENTS",
    "orders": "ORDERS",
    "watchlist": "WATCHLIST",
    "kalimati": "RATES & COMMODITIES",
    "account": "ACCOUNT",
    "strategies": "STRATEGIES",
}

def _load_hedge_trade_log() -> list:
    """Load persisted hedge trade log from disk."""
    try:
        if HEDGE_TRADE_LOG_FILE.exists():
            import json as _json
            data = _json.loads(HEDGE_TRADE_LOG_FILE.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save_hedge_trade_log(trades: list) -> None:
    """Persist hedge trade log to disk."""
    try:
        import json as _json
        HEDGE_TRADE_LOG_FILE.write_text(_json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


class NepseDashboard(App):
    CSS_PATH = Path(__file__).with_name("dashboard_tui.tcss")

    BINDINGS = [
        Binding("1", "tab('market')", "Market", show=False),
        Binding("2", "tab('portfolio')", "Portfolio", show=False),
        Binding("3", "tab('signals')", "Signals", show=False),
        Binding("4", "tab('lookup')", "Lookup", show=False),
        Binding("5", "tab('agents')", "Agents", show=False),
        Binding("6", "tab('orders')", "Orders", show=False),
        Binding("7", "tab('watchlist')", "Watchlist", show=False),
        Binding("8", "tab('kalimati')", "Rates & Commodities", show=False),
        Binding("9", "tab('account')", "Account", show=False),
        Binding("0", "tab('strategies')", "Strategies", show=False),
        # B/S hotkeys removed — orders placed via order book only
        Binding("l", "lookup", "Lookup", show=False),
        Binding("a", "run_agent", "Agent", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("f", "force_signals_reload", "Force Signals", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("slash", "command_palette", "GO", show=False),
        Binding("ctrl+p", "command_palette", "GO", show=False),
        Binding("equals_sign", "watchlist_add", "=Watch", show=False),
        Binding("minus", "watchlist_remove", "-Watch", show=False),
        Binding("d", "tf('D')", "Daily", show=False),
        Binding("w", "tf('W')", "Weekly", show=False),
        Binding("m", "tf('M')", "Monthly", show=False),
        Binding("y", "tf('Y')", "Yearly", show=False),
        Binding("i", "tf('I')", "Intraday", show=False),
    ]

    active_tab: str = "market"
    lookup_sym: str = ""
    lookup_tf: str = "D"  # D=Daily, W=Weekly, M=Monthly, I=Intraday
    trade_mode: str = "paper"
    tms_service = None
    _tms_bundle = None   # cached fetch_monitor_bundle result
    _last_tms_watchlist_fetch_at: float = 0.0
    _tms_watchlist_refresh_inflight: bool = False
    _trading_engine: Optional[TUITradingEngine] = None
    _account_engines: dict = {}  # account_id -> TUITradingEngine
    _news_search_query: str = ""  # current news search filter
    _vector_search_results: list = []  # semantic search results from OSINT API
    _order_action: str = "BUY"   # current selected action in order form
    _paper_orders: list = []     # paper mode order book
    _paper_order_history: list = []  # filled/cancelled orders
    _paper_trades_today: list = []   # today's filled trades
    _paper_watchlist: list[dict] = []
    _live_watchlist: list[dict] = []
    _watchlist: list[dict] = []   # user watchlist entries
    _paper_accounts: list[dict] = []
    _current_account_id: str = "account_1"
    _selected_account_id: str = "account_1"
    _account_help_visible: bool = False
    _strategies: list[dict] = []
    _selected_strategy_id: str = "default_c5"
    _strategy_backtest_result: dict = {}
    _agent_chat_process: Optional[subprocess.Popen] = None
    _agent_chat_request_id: int = 0
    _agent_chat_stop_requested: bool = False
    _screener_sort: str = "chg"  # screener sort column
    _screener_filter: str = ""   # screener sector filter
    _refresh_inflight: bool = False

    PAPER_ORDERS_FILE = TUI_PAPER_ORDERS_FILE
    PAPER_ORDER_HISTORY_FILE = TUI_PAPER_ORDER_HISTORY_FILE

    def compose(self) -> ComposeResult:
        with Vertical(id="top-bars"):
            yield Static(id="header-bar")
            yield Static(id="ticker-bar")
            yield Static(id="index-bar")

        with ContentSwitcher(id="content", initial="market"):
            # ── 1 MARKET: 3-panel top + 2-panel bottom ──
            with Vertical(id="market", classes="tab-pane"):
                with Horizontal(id="market-top"):
                    yield MarketPanel(panel_title="1) GAINERS", title_color=GAIN, id="p-gainers")
                    yield MarketPanel(panel_title="2) LOSERS", title_color=LOSS, id="p-losers")
                    yield MarketPanel(panel_title="3) VOLUME LEADERS", title_color=CYAN, id="p-volume")
                with Horizontal(id="market-bot"):
                    yield MarketPanel(panel_title="4) 52-WEEK EXTREMES", title_color=YELLOW, id="p-52wk")
                    yield MarketPanel(panel_title="5) LIVE QUOTES", title_color=PURPLE, id="p-quotes")

            # ── 2 PORTFOLIO: NAV + risk bar + holdings/risk split + trades ──
            with Vertical(id="portfolio", classes="tab-pane"):
                yield Static("", id="nav-summary", classes="panel-title")
                yield Static("", id="risk-summary", classes="panel-title")
                with Horizontal(id="port-split"):
                    with Vertical(id="port-left", classes="full-pane"):
                        yield Static("HOLDINGS", classes="panel-title")
                        yield DataTable(id="dt-portfolio", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="port-right", classes="full-pane"):
                        yield Static("CONCENTRATION & SECTOR", classes="panel-title")
                        yield DataTable(id="dt-concentration", zebra_stripes=True, cursor_type="row")
                        yield Static("WINNERS / LOSERS", classes="panel-title")
                        yield DataTable(id="dt-winloss", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="port-bottom"):
                    with Vertical(id="port-hedge", classes="full-pane"):
                        with Horizontal(id="hedge-header"):
                            yield Static("GOLD / SILVER HEDGE", classes="panel-title", id="hedge-title")
                            yield Button("● HEDGE ON", id="hedge-toggle-btn", classes="hedge-btn-on")
                        yield Static("", id="hedge-info-bar")
                        yield Static("", id="hedge-rec-bar")
                        yield Static("", id="hedge-trade-bar")

            # ── 3 SIGNALS WORKSPACE ──
            with Vertical(id="signals", classes="tab-pane"):
                yield Static("", id="screener-status-bar")
                with Horizontal(id="signals-main"):
                    with Vertical(id="signals-table-pane", classes="full-pane"):
                        with Horizontal(id="signals-header-row"):
                            yield Static("SIGNALS", classes="panel-title", id="signals-table-title")
                            yield Input(value="0.0", placeholder="min score", id="signal-min-score", classes="signal-threshold-input")
                        yield DataTable(id="dt-signals", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="signals-screener-pane", classes="full-pane"):
                        yield Static("ACTIVE STOCKS", classes="panel-title", id="screener-list-title")
                        yield DataTable(id="dt-screener", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="signals-broker"):
                    with Vertical(id="broker-floor-left"):
                        with Horizontal(id="broker-floor-header"):
                            yield Static("BROKER FLOOR SIGNALS", classes="panel-title", id="broker-floor-title")
                            yield Button("ALL",      id="bf-filter-all",   classes="bf-filter-btn bf-active")
                            yield Button("CIRCULAR", id="bf-filter-circ",  classes="bf-filter-btn")
                            yield Button("PUMP",     id="bf-filter-pump",  classes="bf-filter-btn")
                            yield Button("SMART $",  id="bf-filter-smart", classes="bf-filter-btn")
                        yield DataTable(id="dt-broker-floor", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="broker-floor-right"):
                        yield Static("TOP BROKERS", classes="panel-title", id="broker-top-title")
                        yield DataTable(id="dt-broker-top", zebra_stripes=True, cursor_type="row")
                with Horizontal(id="signals-bottom"):
                    with Vertical(id="signals-calendar", classes="full-pane"):
                        yield Static("CORPORATE ACTIONS — Next 30 Days", classes="panel-title")
                        yield DataTable(id="dt-calendar", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="screener-heatmap-pane", classes="full-pane"):
                        yield Static("SECTOR PERFORMANCE", classes="panel-title")
                        yield Static("", id="heatmap-content")

            # ── 4 LOOKUP ──

            with Vertical(id="lookup", classes="tab-pane"):
                yield Static("Press L to look up any stock", classes="panel-title", id="lookup-title")
                yield Static("", id="lookup-header")
                yield Static("", id="lookup-chart")
                with Horizontal(id="lookup-split"):
                    with Vertical(id="lookup-table-pane"):
                        yield Static("OHLCV", classes="panel-title", id="lookup-ohlcv-title")
                        yield DataTable(id="dt-lookup", zebra_stripes=True, cursor_type="row")
                    with VerticalScroll(id="lookup-summary-pane"):
                        yield Static("", id="lookup-stats")
                        yield Static("", id="lookup-report")
                    with VerticalScroll(id="lookup-side-pane"):
                        yield Static("", id="lookup-fin-title")
                        yield DataTable(id="dt-lookup-fin", zebra_stripes=True, cursor_type="row")
                        yield Static("", id="lookup-ca-title")
                        yield DataTable(id="dt-lookup-ca", zebra_stripes=True, cursor_type="row")
                        yield Static("", id="lookup-intel-title")
                        yield Static("", id="lookup-intel")

            # ── 5 AGENTS ──
            with Vertical(id="agents", classes="tab-pane"):
                yield Static("", id="agent-status-bar")
                yield Static("", id="agent-market-view")
                with Horizontal(id="agent-split"):
                    with Vertical(id="agent-left", classes="full-pane"):
                        yield Static("TOP 10 AGENT PICKS", classes="panel-title")
                        yield Static("", id="agent-picks-subtitle")
                        yield DataTable(id="dt-agent-verdicts", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="agent-left-bottom"):
                            yield Static("FOCUS", classes="panel-title", id="agent-detail-title")
                            with VerticalScroll(id="agent-detail-pane"):
                                yield Static("", id="agent-detail")
                    with Vertical(id="agent-right", classes="full-pane"):
                        yield Static("CHAT", classes="panel-title")
                        yield Static("", id="agent-chat-subtitle")
                        yield VerticalScroll(id="agent-chat-scroll")
                        with Vertical(id="agent-right-bottom"):
                            yield Static("COMPOSER", classes="panel-title", id="agent-compose-title")
                            with Vertical(id="agent-chat-footer"):
                                yield Static("", id="agent-chat-hint")
                                yield Input(id="agent-input", placeholder="Ask the agent...  (/history, /recent, /stop)")

            # ── 6 ORDERS (Order Management) ──
            with Vertical(id="orders", classes="tab-pane"):
                yield Static("", id="order-status-bar")
                with Horizontal(id="order-form-bar"):
                    yield Static("SIDE", classes="order-chip order-chip-neutral")
                    yield Button("BUY", id="order-btn-buy")
                    yield Button("SELL", id="order-btn-sell")
                    yield Static("•", classes="order-sep")
                    yield Static("TICKET", classes="order-chip order-chip-neutral")
                    yield Static("SYM", classes="order-field-label")
                    yield Input(id="order-inp-symbol", placeholder="NABIL")
                    yield Static("QTY", classes="order-field-label")
                    yield Input(id="order-inp-qty", placeholder="10")
                    yield Static("LIMIT", classes="order-field-label")
                    yield Input(id="order-inp-price", placeholder="LTP")
                    yield Static("SLIP%", classes="order-field-label")
                    yield Input(id="order-inp-slippage", placeholder="2.0")
                    yield Static("•", classes="order-sep")
                    yield Static("EXEC", classes="order-chip order-chip-neutral")
                    yield Button("SUBMIT", id="order-btn-submit")
                    yield Button("CANCEL", id="order-btn-cancel-all")
                with Horizontal(id="order-books-split"):
                    with Vertical(id="order-daily-pane", classes="full-pane"):
                        yield Static("DAILY ORDER BOOK", classes="panel-title", id="order-daily-title")
                        yield DataTable(id="dt-orders-daily", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="order-historic-pane", classes="full-pane"):
                        yield Static("HISTORIC ORDER BOOK", classes="panel-title", id="order-historic-title")
                        yield DataTable(id="dt-orders-historic", zebra_stripes=True, cursor_type="row")
                with Vertical(id="order-trades-pane", classes="full-pane"):
                    yield Static("TODAY'S TRADES (FILLED)", classes="panel-title", id="order-trades-title")
                    yield DataTable(id="dt-orders-trades", zebra_stripes=True, cursor_type="row")

            # ── 7 WATCHLIST ──
            with Vertical(id="watchlist", classes="tab-pane"):
                yield Static("", id="wl-status-bar")
                with Horizontal(id="wl-split"):
                    with Vertical(id="wl-main", classes="full-pane"):
                        yield Static("STOCK WATCHLIST", classes="panel-title", id="wl-main-title")
                        yield DataTable(id="dt-watchlist", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="wl-side", classes="full-pane"):
                        with Vertical(id="wl-rates-pane", classes="full-pane"):
                            yield Static("FOREX & MACRO", classes="panel-title", id="wl-rates-title")
                            yield DataTable(id="dt-watchlist-rates", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="wl-commodities-pane", classes="full-pane"):
                            yield Static("COMMODITIES", classes="panel-title", id="wl-commodities-title")
                            yield DataTable(id="dt-watchlist-commodities", zebra_stripes=True, cursor_type="row")

            # ── 8 RATES & COMMODITIES ───────────────────────────────────────
            with Vertical(id="kalimati", classes="tab-pane"):
                yield Static("", id="kalimati-status-bar")
                with Horizontal(id="kalimati-search-bar"):
                    yield Static("SEARCH", id="kalimati-search-label")
                    yield Input(
                        id="kalimati-search-input",
                        placeholder="Search commodities, gold, silver, crude, petrol, forex..."
                    )
                    yield Button("CLEAR", id="kalimati-search-clear")
                with Horizontal(id="kalimati-split"):
                    with Vertical(id="kalimati-left-pane", classes="full-pane"):
                        yield Static("KALIMATI COMMODITIES", classes="panel-title", id="kalimati-left-title")
                        yield Static("", id="kalimati-movers-bar")
                        with Vertical(id="kalimati-main", classes="full-pane"):
                            yield DataTable(id="dt-kalimati", zebra_stripes=True, cursor_type="row")
                    with Vertical(id="kalimati-right-pane", classes="full-pane"):
                        yield Static("GLOBAL RATES & PRICES", classes="panel-title", id="kalimati-right-title")
                        with Vertical(id="kalimati-macro-pane", classes="full-pane"):
                            yield Static("METALS, ENERGY & NOC", classes="panel-title", id="macro-top-title")
                            yield DataTable(id="dt-macro", zebra_stripes=True, cursor_type="row")
                        with Vertical(id="kalimati-forex-pane", classes="full-pane"):
                            yield Static("FOREX RATES", classes="panel-title", id="macro-forex-title")
                            yield DataTable(id="dt-forex", zebra_stripes=True, cursor_type="row")

            with Vertical(id="account", classes="tab-pane"):
                with Horizontal(id="account-split"):
                  with Vertical(id="account-left", classes="full-pane"):
                    with Vertical(id="account-main", classes="full-pane"):
                      with Vertical(id="profile-pane"):
                        yield Static("PAPER ACCOUNT", classes="panel-title")
                        yield Static("", id="profile-summary")
                        yield Static("ACCOUNTS", classes="panel-title")
                        yield DataTable(id="dt-account-list", zebra_stripes=True, cursor_type="row")
                        yield Static("ACCOUNT SETUP", classes="panel-title")
                        yield Static("", id="profile-shortcuts")
                        with Horizontal(id="profile-primary-row"):
                            yield Static("NAME", classes="profile-inline-label")
                            yield Input(id="profile-inp-account-name", placeholder="Account 2", classes="profile-inline-input profile-name-input")
                            yield Static("NAV", classes="profile-inline-label")
                            yield Input(id="profile-inp-target-nav", placeholder="1000000", classes="profile-inline-input profile-nav-input")
                            with Horizontal(id="profile-primary-actions"):
                                yield Button("N NEW", id="profile-btn-create-account", classes="profile-btn profile-btn-primary")
                                yield Button("A ACTIVATE", id="profile-btn-activate-account", classes="profile-btn")
                        with Horizontal(id="profile-seed-row"):
                            yield Static("SEED", classes="profile-inline-label")
                            yield Input(id="profile-inp-portfolio", placeholder="paper_portfolio.csv or MeroShare CSV  —  or press B BROWSE")
                            yield Button("B BROWSE", id="profile-btn-browse-seed", classes="profile-btn profile-btn-browse")
                        with Horizontal(id="profile-actions"):
                            yield Button("W WATCHLIST", id="profile-btn-sync-watchlist", classes="profile-btn profile-action-button")
                            yield Button("V SET NAV", id="profile-btn-set-nav", classes="profile-btn profile-action-button")
                            yield Button("S SNAPSHOT", id="profile-btn-save-account", classes="profile-btn profile-action-button")
                            yield Button("DEL ACCOUNT", id="profile-btn-delete-account", classes="profile-btn profile-action-button profile-btn-delete-account")
                        yield Static("", id="account-help")
                        yield Static(
                            "Create a blank account with target NAV, or press B BROWSE to pick a SEED file. "
                            "Accepts paper_portfolio.csv or MeroShare 'My Shares Values.csv' (auto-detected). "
                            "Selecting and activating an account swaps the full paper runtime.",
                            id="profile-note",
                        )
                  with Vertical(id="account-right", classes="full-pane"):
                      with Vertical(id="port-trades", classes="full-pane"):
                          yield Static("TRADE HISTORY", classes="panel-title", id="trades-title")
                          yield DataTable(id="dt-trades-full", zebra_stripes=True, cursor_type="row")
                      with Vertical(id="port-activity", classes="full-pane"):
                          yield Static("ENGINE LOG", classes="panel-title", id="activity-title")
                          yield VerticalScroll(id="activity-scroll")

            with Vertical(id="strategies", classes="tab-pane"):
                with Vertical(id="strategies-main", classes="full-pane"):
                    yield Static("STRATEGY REGISTRY", classes="panel-title")
                    yield Static("", id="strategy-summary")
                    with Horizontal(id="strategies-split"):
                        with Vertical(id="strategies-left", classes="full-pane"):
                            yield Static("SAVED STRATEGIES", classes="panel-title")
                            yield DataTable(id="dt-strategy-list", zebra_stripes=True, cursor_type="row")
                            yield Static("", id="strategy-accounts-note")
                        with Vertical(id="strategies-right", classes="full-pane"):
                            yield Static("STRATEGY BUILDER", classes="panel-title")
                            with Horizontal(id="strategy-name-row", classes="strategy-row"):
                                yield Static("NAME", classes="profile-inline-label")
                                yield Input(id="strategy-inp-name", placeholder="My Strategy", classes="profile-inline-input profile-name-input")
                                yield Static("DESC", classes="profile-inline-label")
                                yield Input(id="strategy-inp-description", placeholder="Notes", classes="profile-inline-input")
                            yield Static("SIGNALS", classes="panel-title", id="signal-picker-label")
                            with Vertical(id="signal-picker-area"):
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[:5]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[5:10]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                with Horizontal(classes="signal-picker-row"):
                                    for _lbl, _sig in _SIGNAL_DEFS[10:]:
                                        yield Button(_lbl, id=_sig_btn_id(_sig), classes="signal-btn")
                                yield Static("", id="signal-picker-active", classes="signal-active-display")
                            yield Static("PARAMETERS", classes="panel-title", id="strategy-params-label")
                            with Horizontal(id="strategy-config-row-a", classes="strategy-row"):
                                yield Static("HOLD", classes="profile-inline-label")
                                yield Input(id="strategy-inp-holding-days", placeholder="40", classes="profile-inline-input strategy-small-input")
                                yield Static("REBAL", classes="profile-inline-label")
                                yield Input(id="strategy-inp-rebalance", placeholder="5", classes="profile-inline-input strategy-small-input")
                                yield Static("MAX POS", classes="profile-inline-label")
                                yield Input(id="strategy-inp-max-positions", placeholder="5", classes="profile-inline-input strategy-small-input")
                            with Horizontal(id="strategy-config-row-b", classes="strategy-row"):
                                yield Static("STOP", classes="profile-inline-label")
                                yield Input(id="strategy-inp-stop-loss", placeholder="0.08", classes="profile-inline-input strategy-small-input")
                                yield Static("TRAIL", classes="profile-inline-label")
                                yield Input(id="strategy-inp-trailing-stop", placeholder="0.10", classes="profile-inline-input strategy-small-input")
                                yield Static("SECTOR", classes="profile-inline-label")
                                yield Input(id="strategy-inp-sector-limit", placeholder="0.35", classes="profile-inline-input strategy-small-input")
                            with Horizontal(id="strategy-actions-row", classes="strategy-row"):
                                yield Button("NEW", id="strategy-btn-new", classes="profile-btn")
                                yield Button("SAVE", id="strategy-btn-save", classes="profile-btn profile-btn-primary")
                                yield Button("→ ACTIVE ACCT", id="strategy-btn-assign-current", classes="profile-btn")
                                yield Button("→ SELECTED ACCT", id="strategy-btn-assign-selected", classes="profile-btn")
                                yield Button("DELETE", id="strategy-btn-delete", classes="profile-btn strategy-btn-delete")
                            yield Static("BACKTEST", classes="panel-title", id="strategy-backtest-label")
                            with Horizontal(id="strategy-backtest-row", classes="strategy-row"):
                                yield Static("START", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-start", placeholder="2025-01-01", classes="profile-inline-input strategy-date-input")
                                yield Static("END", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-end", placeholder="2026-04-11", classes="profile-inline-input strategy-date-input")
                                yield Static("CAP", classes="profile-inline-label")
                                yield Input(id="strategy-inp-backtest-capital", placeholder="1000000", classes="profile-inline-input strategy-capital-input")
                                yield Button("RUN", id="strategy-btn-backtest", classes="profile-btn profile-action-button")
                                yield Button("CHART", id="strategy-btn-chart", classes="profile-btn strategy-btn-chart")
                            yield Static("", id="strategy-backtest-summary")

        yield Static(id="status-bar")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.push_screen(ModeSelectScreen(), callback=self._on_mode_selected)

    def _on_mode_selected(self, mode: str | None) -> None:
        if not mode:
            self.exit()
            return
        self.trade_mode = "paper"
        self._init_dashboard()

    def _on_tms_credentials(self, result: dict | None) -> None:
        if not result:
            # User pressed ESC — fall back to mode select
            self.push_screen(ModeSelectScreen(), callback=self._on_mode_selected)
            return
        self._tms_credentials = result
        self._init_dashboard()
        self._init_tms()

    def _init_dashboard(self) -> None:
        _ensure_paper_runtime_files()
        self._paper_accounts, self._current_account_id = _bootstrap_paper_accounts()
        self._selected_account_id = self._current_account_id
        self._load_strategies_registry()
        self._apply_active_strategy_runtime()
        self._sync_agent_account_context_env()
        # Pull live quotes into DB before loading market data
        try:
            from backend.quant_pro.realtime_market import get_market_data_provider
            snap = get_market_data_provider().fetch_snapshot(force=True)
            if snap and snap.quotes:
                self._upsert_live_prices(snap)
        except Exception:
            pass
        self.md = MD(top_n=25)
        self._regime = "loading..."
        self._stats: dict = {}
        self._lookup_cache: dict[tuple[str, str, str], dict] = {}
        self._lookup_request_key: Optional[tuple[str, str, str]] = None
        self._signals_table_cache_key: str = ""
        self._signals_table_cache_payload: Optional[tuple[list[tuple[str, str]], list[list[Text]], int]] = None
        self._signals_workspace_cache_key: str = ""
        self._signals_workspace_cache_payload: Optional[dict] = None
        self._signals_last_loaded_at: str = ""
        self._signals_last_strategy_name: str = ""
        self._signals_last_count: int = 0
        self._signal_min_score: float = 0.0
        self._broker_floor_filter: str = "all"  # all | circ | pump | smart
        self._ticker_text = ""
        self._ticker_offset = 0
        self._rates_search_query: str = ""
        self._kalimati_rows: list[dict] = []
        self._kalimati_status: str = "Loading..."
        self._macro_rates: dict = {}
        self._hedge_enabled: bool = True    # ON/OFF toggle for gold/silver hedge overlay
        self._hedge_trade_log: list = _load_hedge_trade_log()
        self._active_signals: set[str] = set()  # signal picker state
        self._new_strategy_mode: bool = False   # True after NEW pressed, False after list selection or SAVE
        self._watchlist_stock_rows: list[dict] = []
        self._watchlist_rates_rows: list[dict] = []
        self._watchlist_commodity_rows: list[dict] = []
        self._last_macro_rates_fetch_at: float = 0.0
        self._build_ticker()
        self._update_header()
        self._update_index()
        self._load_profile_inputs()
        self._populate_strategies_tab()
        self._populate_market()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._osint_stories: list[dict] = []
        self._agent_analysis: dict = {}
        self._agent_history: list[dict] = []
        self._agent_archived_history: list[dict] = []
        self._agent_archive_count: int = 0
        self._agent_show_archived = False
        self._agent_hidden_recent_history: list[dict] = []
        self._agent_preview_override: Optional[dict] = None
        self._last_agent_auto_order_key: Optional[str] = None
        self._agent_typing_visible = False
        self._agent_typing_frame = 0
        self._agent_session_started_at = time.time()
        self._agent_visible_since = self._agent_session_started_at
        self._agent_chat_loaded = False
        self._load_agent_runtime_state()
        self._populate_agent_tab()
        if not list((self._agent_analysis or {}).get("stocks") or []):
            self._run_agent_analysis(force=False)
        self._load_paper_orders()
        self._populate_orders_tab()
        self._refresh_order_action_buttons()
        self._paper_watchlist = _load_watchlist()
        self._watchlist = list(self._paper_watchlist)
        self._populate_watchlist()
        self._populate_signals_workspace()
        self._load_signals_async()
        self._load_regime_async()
        self._load_osint_async()
        init_kalimati_db()
        self._load_kalimati_async()
        self._load_macro_rates_async(force=True)
        mode_label = self._display_mode_label()
        self._set_status(
            f"Mode: {mode_label}  │  Session: {self.md.latest}  │  "
            f"▲{self.md.adv} ▼{self.md.dec}  │  Auto-refresh 30s"
        )
        self.set_interval(30, self._auto_refresh)
        self.set_interval(TICKER_SPEED, self._scroll_ticker)
        self.set_interval(0.45, self._animate_agent_typing)

        # Start one auto-trading engine per account in background threads
        if self.trade_mode == "paper":
            self._start_all_account_engines()

    @work(thread=True)
    def _init_tms(self) -> None:
        """Live brokerage not available in public release."""
        pass

    def _attach_tms_live_source(self) -> None:
        """Live brokerage not available in public release."""
        pass

    # ── News ticker ──────────────────────────────────────────────────────────

    def _build_ticker(self) -> None:
        """Build the ticker string from market data."""
        items = []
        # Top gainers
        if not self.md.gainers.empty:
            for _, r in self.md.gainers.head(5).iterrows():
                chg = r["chg"]
                arrow = "▲" if chg >= 0 else "▼"
                items.append(f"{r['symbol']} {r['close']:.1f} {arrow}{chg:+.2f}%")
        # Top losers
        if not self.md.losers.empty:
            for _, r in self.md.losers.head(5).iterrows():
                items.append(f"{r['symbol']} {r['close']:.1f} ▼{r['chg']:+.2f}%")
        # Volume leaders
        if not self.md.vol_top.empty:
            for _, r in self.md.vol_top.head(3).iterrows():
                items.append(f"{r['symbol']} Vol:{_vol(r['volume'])}")
        # NEPSE index
        if len(self.md.nepse) >= 2:
            ni = self.md.nepse.iloc[0]["close"]
            np_ = self.md.nepse.iloc[1]["close"]
            chg = (ni - np_) / np_ * 100
            items.insert(0, f"NEPSE {ni:,.1f} {chg:+.2f}%")
        # Corporate actions
        if not self.md.corp.empty:
            for _, r in self.md.corp.head(3).iterrows():
                bc = r["bookclose_date"]
                days = (bc - datetime.now()).days
                cash = float(r.get("cash_dividend_pct") or 0)
                bonus = float(r.get("bonus_share_pct") or 0)
                parts = []
                if cash > 0: parts.append(f"Div {cash:.0f}%")
                if bonus > 0: parts.append(f"Bonus {bonus:.0f}%")
                if parts:
                    items.append(f"{r['symbol']} {' '.join(parts)} BookClose:{days}d")
        # OSINT headlines in ticker (prefer translated English display text)
        news_count = 0
        for s in getattr(self, "_osint_stories", []):
            sev = str(s.get("severity") or "")
            if sev not in ("high", "medium", "critical"):
                continue
            headline = _news_display_headline(s)
            if not headline or _contains_non_ascii(headline):
                continue
            tag = sev.upper()[:4]
            items.append(f"[{tag}] {_truncate_text(headline, 65)}")
            news_count += 1
            if news_count >= 10:
                break

        # Build the full ticker line with separators
        sep = "  ◆  "
        self._ticker_text = sep + sep.join(items) + sep
        self._ticker_offset = 0

    def _scroll_ticker(self) -> None:
        """Advance the ticker by one character."""
        text = self._ticker_text
        if not text:
            return
        # Get terminal width
        try:
            w = self.size.width
        except Exception:
            w = 120
        # Build the visible window
        doubled = text + text  # loop seamlessly
        start = self._ticker_offset % len(text)
        visible = doubled[start:start + w]
        # Color the ticker: symbols in amber, numbers in white, arrows colored
        from rich.text import Text as RichText
        styled = RichText()
        i = 0
        while i < len(visible):
            ch = visible[i]
            if ch == "▲":
                styled.append(ch, style=f"bold {GAIN_HI}")
            elif ch == "▼":
                styled.append(ch, style=f"bold {LOSS_HI}")
            elif ch == "◆":
                styled.append(ch, style="#555555")
            elif ch == "+" or (ch == "-" and i + 1 < len(visible) and visible[i + 1].isdigit()):
                # Collect the number
                j = i + 1
                while j < len(visible) and (visible[j].isdigit() or visible[j] in ".%"):
                    j += 1
                num_str = visible[i:j]
                if ch == "+":
                    styled.append(num_str, style=GAIN)
                else:
                    styled.append(num_str, style=LOSS)
                i = j
                continue
            elif ch == "[":
                # Severity tag like [HIGH] or [MED ]
                end = visible.find("]", i)
                if end != -1 and end - i <= 6:
                    tag = visible[i+1:end].strip()
                    tag_style = {"CRIT": f"bold {LOSS_HI}", "HIGH": LOSS,
                                 "MEDI": YELLOW, "LOW": LABEL}.get(tag, LABEL)
                    styled.append(visible[i:end+1], style=tag_style)
                    i = end + 1
                    continue
                styled.append(ch, style="#888888")
            elif ch.isalpha() or ord(ch) > 127:
                # Collect word (ASCII or Nepali unicode)
                j = i
                while j < len(visible) and (visible[j].isalpha() or visible[j] == ":" or ord(visible[j]) > 127):
                    j += 1
                word = visible[i:j]
                if word in ("NEPSE", "Vol", "Div", "Bonus", "BookClose"):
                    styled.append(word, style=f"bold {AMBER}")
                else:
                    styled.append(word, style=WHITE)
                i = j
                continue
            else:
                styled.append(ch, style="#888888")
            i += 1
        self.query_one("#ticker-bar", Static).update(styled)
        self._ticker_offset += 1

    # ── Tab switching ─────────────────────────────────────────────────────────

    def action_tab(self, name: str) -> None:
        if name == "screener":
            name = "signals"
        if name == "news":
            name = "agents"
        self.active_tab = name
        self.query_one("#content", ContentSwitcher).current = name
        self._update_header()
        self._refresh_active_tab_view(force_watchlist_sync=True)
        if name == "signals":
            self._load_signals_async()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_buy(self) -> None:
        self.push_screen(
            ModalDialog(
                "buy",
                initial_symbol=self._preferred_order_symbol(),
                initial_price=self._preferred_order_price_text(self._preferred_order_symbol()),
            ),
            callback=self._on_buy,
        )

    def action_sell(self) -> None:
        symbol, shares = self._preferred_sell_ticket_defaults()
        self.push_screen(
            ModalDialog(
                "sell",
                initial_symbol=symbol,
                initial_shares=shares,
                initial_price=self._preferred_order_price_text(symbol),
                holdings_positions=self._stats.get("positions", []) if hasattr(self, "_stats") else [],
            ),
            callback=self._on_sell,
        )

    def action_lookup(self) -> None:
        self.push_screen(LookupScreen(), callback=self._on_lookup)

    def action_run_agent(self) -> None:
        if self.active_tab == "account":
            try:
                self._set_status(self._account_activate_selected())
            except Exception as exc:
                self._set_status(f"Account activate failed: {exc}")
            return
        self.action_tab("agents")
        self._run_agent_analysis(force=True)

    def action_tf(self, tf: str) -> None:
        """Switch lookup chart timeframe: D/W/M/I."""
        if self.active_tab == "account":
            if tf == "W":
                try:
                    self._set_status(self._account_sync_watchlist())
                except Exception as exc:
                    self._set_status(f"Watchlist sync failed: {exc}")
            return
        if self.active_tab != "lookup":
            return
        self.lookup_tf = tf
        self._populate_lookup()
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
        self._set_status(f"Chart: {tf_names.get(tf, tf)}  │  D=Daily  W=Weekly  M=Monthly  Y=Yearly  I=Intraday")

    def action_refresh(self) -> None:
        if self.active_tab == "signals":
            self.action_force_signals_reload()
            self._do_refresh()
            return
        if self.active_tab == "watchlist" and self.trade_mode == "live" and self.tms_service:
            self._refresh_watchlist_live(force=True)
        self._do_refresh()

    def action_force_signals_reload(self) -> None:
        if self.active_tab != "signals":
            self._set_status("Force signals reload is available on the Signals tab")
            return
        self._reload_account_bindings_from_disk()
        self._signals_table_cache_key = ""
        self._signals_table_cache_payload = None
        self._set_signals_table_loading()
        self._load_signals_async(force=True)

    def _on_buy(self, result: dict | None) -> None:
        if not result: return
        # Agent veto gate (reads cached analysis only — no Sonnet call)
        approved, reason = check_trade_approval(result["symbol"], "BUY")
        if not approved:
            self._set_status(f"AGENT BLOCKED BUY {result['symbol']}: {reason}")
            return
        if self.trade_mode == "live":
            self._submit_tms_order(result, "BUY", reason)
            return
        try:
            qty = int(str(result["shares"]).strip())
            if qty <= 0:
                raise ValueError
        except ValueError:
            self._set_status(f"Order: Invalid buy quantity for {result['symbol']}")
            return
        price = self._resolve_ticket_price(result["symbol"], result.get("price", ""))
        if price is None:
            self._set_status(f"Order: No price available for {result['symbol']}")
            return
        slippage = self._parse_slippage(result.get("slippage", ""))
        msg = self._submit_paper_order("BUY", result["symbol"], qty, price, slippage)
        self._set_status(f"{msg}  |  Agent: {reason[:60]}")
        self.action_tab("orders")

    def _on_sell(self, result: dict | None) -> None:
        if not result: return
        # Agent veto gate
        approved, reason = check_trade_approval(result["symbol"], "SELL")
        if not approved:
            self._set_status(f"AGENT BLOCKED SELL {result['symbol']}: {reason}")
            return
        holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
        try:
            qty = _resolve_sell_qty(result["symbol"], result["shares"], holdings)
        except ValueError as exc:
            self._set_status(f"Order: {exc}")
            return
        price = self._resolve_ticket_price(result["symbol"], result.get("price", ""))
        if price is None:
            self._set_status(f"Order: No price available for {result['symbol']}")
            return
        slippage = self._parse_slippage(result.get("slippage", ""))
        msg = self._submit_paper_order("SELL", result["symbol"], qty, price, slippage)
        self._set_status(f"{msg}  |  Agent: {reason[:60]}")
        self.action_tab("orders")

    @work(thread=True)
    def _submit_tms_order(self, result: dict, action: str, agent_reason: str) -> None:
        """Live TMS order submission is disabled in the public paper build."""
        self.app.call_from_thread(self._set_status, "Only paper trading is supported in this build.")

    def _on_lookup(self, result: dict | None) -> None:
        if not result: return
        self.lookup_sym = result["symbol"]
        self.action_tab("lookup")

    # ── Order Management ─────────────────────────────────────────────────────

    def _load_paper_orders(self) -> None:
        """Load paper orders from JSON files."""
        self._paper_orders = []
        self._paper_order_history = []
        if self.PAPER_ORDERS_FILE.exists():
            try:
                self._paper_orders = _json.loads(self.PAPER_ORDERS_FILE.read_text())
            except Exception:
                self._paper_orders = []
        if self.PAPER_ORDER_HISTORY_FILE.exists():
            try:
                self._paper_order_history = _json.loads(self.PAPER_ORDER_HISTORY_FILE.read_text())
            except Exception:
                self._paper_order_history = []
        from backend.trading.live_trader import now_nst
        self._paper_trades_today = _paper_filled_orders_for_day(
            self._paper_order_history,
            now_nst().strftime("%Y-%m-%d"),
        )

    def _save_paper_orders(self) -> None:
        """Persist paper orders to JSON files."""
        _write_json_locked(Path(self.PAPER_ORDERS_FILE), list(self._paper_orders))
        _write_json_locked(Path(self.PAPER_ORDER_HISTORY_FILE), list(self._paper_order_history))

    def _create_paper_order(self, action: str, symbol: str, qty: int, price: float, slippage: float = 2.0) -> dict:
        """Create a new paper order dict."""
        from backend.trading.live_trader import now_nst
        ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
        order = {
            "id": uuid.uuid4().hex[:12],
            "action": action,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "slippage_pct": slippage,
            "status": "OPEN",
            "filled_qty": 0,
            "fill_price": 0.0,
            "trigger_price": price,
            "created_at": ts,
            "updated_at": ts,
            "day": now_nst().strftime("%Y-%m-%d"),
            "source": "dashboard_tui",
            "reason": "",
        }
        return order

    def _paper_has_same_day_order(self, symbol: str, *, action: str) -> bool:
        from backend.trading.live_trader import now_nst
        sym = str(symbol or "").strip().upper()
        act = str(action or "").strip().upper()
        today = now_nst().strftime("%Y-%m-%d")
        for row in [*self._paper_orders, *self._paper_order_history]:
            if str(row.get("symbol") or "").strip().upper() != sym:
                continue
            if str(row.get("action") or "").strip().upper() != act:
                continue
            row_day = str(row.get("day") or "")[:10] or str(row.get("created_at") or "")[:10]
            if row_day != today:
                continue
            if str(row.get("status") or "").strip().upper() in {"OPEN", "FILLED"}:
                return True
        return False

    def _paper_position_opened_today(self, symbol: str) -> bool:
        from backend.trading.live_trader import now_nst
        sym = str(symbol or "").strip().upper()
        today = now_nst().strftime("%Y-%m-%d")
        port = load_port()
        if port.empty or "Symbol" not in port.columns:
            return False
        rows = port[port["Symbol"].astype(str).str.upper() == sym]
        if rows.empty:
            return False
        entry_dates = rows.get("Entry_Date")
        if entry_dates is None:
            return False
        return any(str(value)[:10] == today for value in entry_dates.tolist())

    def _paper_same_day_trade_block(self, action: str, symbol: str) -> Optional[str]:
        sym = str(symbol or "").strip().upper()
        act = str(action or "").strip().upper()
        if act == "BUY" and self._paper_has_same_day_order(sym, action="SELL"):
            return f"Rejected: NEPSE same-day rule blocks buying {sym} after a sell today."
        if act == "SELL":
            if self._paper_position_opened_today(sym) or self._paper_has_same_day_order(sym, action="BUY"):
                return f"Rejected: NEPSE same-day rule blocks selling {sym} on the same day as a buy."
        return None

    def _submit_paper_order(self, action: str, symbol: str, qty: int, price: float, slippage: float = 2.0) -> str:
        """Submit a new paper limit order."""
        self._load_paper_orders()
        block_reason = self._paper_same_day_trade_block(action, symbol)
        if block_reason:
            return block_reason
        order = self._create_paper_order(action, symbol, qty, price, slippage)
        self._paper_orders.append(order)
        self._save_paper_orders()
        self._populate_orders_tab()
        return f"Order {order['id']}: {action} {symbol} x{qty} @ {price:,.1f} slip:{slippage:.1f}% — OPEN"

    def _cancel_paper_order(self, order_id: str) -> str:
        """Cancel a specific paper order by ID."""
        from backend.trading.live_trader import now_nst
        for o in self._paper_orders:
            if o["id"] == order_id and o["status"] == "OPEN":
                o["status"] = "CANCELLED"
                o["updated_at"] = now_nst().strftime("%Y-%m-%d %H:%M:%S")
                self._paper_order_history.append(o)
                self._paper_orders = [x for x in self._paper_orders if x["id"] != order_id]
                self._save_paper_orders()
                self._populate_orders_tab()
                return f"Order {order_id} cancelled"
        return f"Order {order_id} not found or already filled"

    def _cancel_all_paper_orders(self) -> str:
        """Cancel all open paper orders."""
        from backend.trading.live_trader import now_nst
        ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
        count = 0
        for o in self._paper_orders:
            if o["status"] == "OPEN":
                o["status"] = "CANCELLED"
                o["updated_at"] = ts
                self._paper_order_history.append(o)
                count += 1
        self._paper_orders = [x for x in self._paper_orders if x["status"] == "OPEN"]
        self._save_paper_orders()
        self._populate_orders_tab()
        return f"Cancelled {count} open orders"

    def _match_paper_orders(self) -> None:
        """Check open orders against current prices and fill if matched."""
        from backend.trading.live_trader import now_nst
        if not self._paper_orders:
            return
        # Get current LTPs from market data
        ltps: dict[str, float] = {}
        if hasattr(self, 'md') and not self.md.quotes.empty:
            for _, row in self.md.quotes.iterrows():
                sym = row.get("symbol", "")
                price = row.get("close", 0)
                if sym and price > 0:
                    ltps[sym] = float(price)
        # Also check from gainers/losers for broader coverage
        for df in [self.md.gainers, self.md.losers]:
            if not df.empty:
                for _, row in df.iterrows():
                    sym = row.get("symbol", "")
                    price = row.get("close", 0)
                    if sym and price > 0:
                        ltps[sym] = float(price)

        ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
        filled = []
        for o in self._paper_orders:
            if o["status"] != "OPEN":
                continue
            if str(o.get("source") or "").strip().lower() in {"strategy_paper", "strategy_exit_paper"}:
                continue
            sym = o["symbol"]
            if sym not in ltps:
                continue
            ltp = ltps[sym]
            slip_pct = o.get("slippage_pct", 2.0) / 100.0
            matched = False
            if o["action"] == "BUY" and ltp <= o["price"] * (1 + slip_pct):
                matched = True
            elif o["action"] == "SELL" and ltp >= o["price"] * (1 - slip_pct):
                matched = True
            if matched:
                block_reason = self._paper_same_day_trade_block(o["action"], sym)
                if block_reason:
                    o["status"] = "CANCELLED"
                    o["updated_at"] = ts
                    o["reason"] = "same_day_rule"
                    filled.append(o)
                    status_msg = f"Order cancelled: {o['action']} {sym} — {block_reason}"
                    self._append_activity(f"ORDER CANCELLED: {o['action']} {sym} — {block_reason}")
                    self._set_status(status_msg)
                    continue
                o["status"] = "FILLED"
                o["filled_qty"] = o["qty"]
                o["fill_price"] = ltp
                o["updated_at"] = ts
                filled.append(o)
                # Execute the trade in paper portfolio
                if o["action"] == "BUY":
                    msg = exec_buy(o["symbol"], str(o["qty"]), str(ltp))
                else:
                    msg = exec_sell(o["symbol"], str(o["qty"]), str(ltp))
                self._append_activity(f"ORDER FILLED: {o['action']} {sym} x{o['qty']} @ {ltp:,.1f} — {msg}")

        if filled:
            today = now_nst().strftime("%Y-%m-%d")
            for o in filled:
                self._paper_order_history.append(o)
                self._paper_trades_today.append(o)
            self._paper_orders = [x for x in self._paper_orders if x["status"] == "OPEN"]
            self._save_paper_orders()
            self._populate_orders_tab()
            self._populate_portfolio_and_risk()
            self._populate_trades_full()
        self._render_hedge_panel()

    def _populate_orders_tab(self) -> None:
        """Populate the Orders tab DataTables."""
        if self.trade_mode == "live":
            self._populate_orders_tab_live()
            return
        self._populate_orders_tab_paper()

    def _populate_orders_tab_paper(self) -> None:
        """Populate orders tab from paper order book."""
        self._load_paper_orders()
        # -- Daily order book (open orders) --
        dt_daily = self.query_one("#dt-orders-daily", DataTable)
        dt_daily.clear(columns=True)
        dt_daily.add_columns("ID", "ACTION", "SYMBOL", "QTY", "LIMIT", "SLIP%", "BAND", "STATUS", "TIME")
        from backend.trading.live_trader import now_nst
        today = now_nst().strftime("%Y-%m-%d")
        daily = [o for o in self._paper_orders if o.get("day") == today]
        for o in daily:
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            status_style = YELLOW if o["status"] == "OPEN" else (GAIN if o["status"] == "FILLED" else LOSS)
            slip_pct = float(o.get("slippage_pct", 2.0) or 0.0)
            band_text = self._format_order_band(o["action"], float(o["price"]), slip_pct)
            dt_daily.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["qty"]), style=WHITE),
                Text(f'{o["price"]:,.1f}', style=AMBER),
                Text(f"{slip_pct:.1f}%", style=DIM),
                Text(band_text, style=CYAN),
                Text(o["status"], style=status_style),
                Text(o.get("created_at", "")[-8:], style=DIM),
            )
        if not daily:
            dt_daily.add_row(
                Text("—", style=DIM), Text("—", style=DIM), Text("No open orders", style=DIM),
                Text("—", style=DIM), Text("—", style=DIM), Text("—", style=DIM),
                Text("—", style=DIM), Text("—", style=DIM), Text("—", style=DIM),
            )

        # -- Historic order book (filled + cancelled) --
        dt_hist = self.query_one("#dt-orders-historic", DataTable)
        dt_hist.clear(columns=True)
        dt_hist.add_columns("ID", "ACTION", "SYMBOL", "QTY", "PRICE", "STATUS", "FILL PX", "TIME")
        for o in reversed(self._paper_order_history[-50:]):
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            if o["status"] == "FILLED":
                status_style = GAIN
            elif o["status"] == "CANCELLED":
                status_style = LOSS
            else:
                status_style = YELLOW
            fill_px = f'{o["fill_price"]:,.1f}' if o.get("fill_price") else "—"
            dt_hist.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["qty"]), style=WHITE),
                Text(f'{o["price"]:,.1f}', style=AMBER),
                Text(o["status"], style=status_style),
                Text(fill_px, style=WHITE),
                Text(str(o.get("created_at") or o.get("updated_at") or "")[-8:], style=DIM),
            )

        # -- Today's trades (filled orders) --
        dt_trades = self.query_one("#dt-orders-trades", DataTable)
        dt_trades.clear(columns=True)
        dt_trades.add_columns("ID", "ACTION", "SYMBOL", "QTY", "FILL PRICE", "VALUE", "TIME")
        for o in reversed(self._paper_trades_today):
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            value = o.get("fill_price", 0) * o.get("filled_qty", 0)
            dt_trades.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["filled_qty"]), style=WHITE),
                Text(f'{o.get("fill_price", 0):,.1f}', style=AMBER),
                Text(f'{value:,.0f}', style=WHITE),
                Text(o.get("updated_at", "")[-8:], style=DIM),
            )

        # -- Update order status bar --
        open_count = len([o for o in self._paper_orders if o["status"] == "OPEN"])
        filled_today = len(self._paper_trades_today)
        bar = self.query_one("#order-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold #ffaf00]ORDER MANAGEMENT[/]  │  "
            f"[{GAIN}]Open: {open_count}[/]  │  "
            f"[{CYAN}]Filled today: {filled_today}[/]  │  "
            f"[#888888]Mode: PAPER  │  Place orders via the order book[/]"
        ))

    def _populate_orders_tab_live(self) -> None:
        """Populate orders tab from TMS live order book."""
        bundle = self._tms_bundle or {}
        orders_daily = bundle.get("orders_daily", {})
        orders_historic = bundle.get("orders_historic", {})
        trades_daily = bundle.get("trades_daily", {})

        # -- Daily orders --
        dt_daily = self.query_one("#dt-orders-daily", DataTable)
        dt_daily.clear(columns=True)
        daily_records = orders_daily.get("records", [])
        if daily_records:
            cols = list(daily_records[0].keys())[:8]
            dt_daily.add_columns(*[c.upper() for c in cols])
            for rec in daily_records:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_daily.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_daily.add_columns("STATUS")
            dt_daily.add_row(Text("No daily orders", style=DIM))

        # -- Historic orders --
        dt_hist = self.query_one("#dt-orders-historic", DataTable)
        dt_hist.clear(columns=True)
        hist_records = orders_historic.get("records", [])
        if hist_records:
            cols = list(hist_records[0].keys())[:8]
            dt_hist.add_columns(*[c.upper() for c in cols])
            for rec in hist_records[:50]:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_hist.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_hist.add_columns("STATUS")
            dt_hist.add_row(Text("No historic orders", style=DIM))

        # -- Today's trades --
        dt_trades = self.query_one("#dt-orders-trades", DataTable)
        dt_trades.clear(columns=True)
        trade_records = trades_daily.get("records", [])
        if trade_records:
            cols = list(trade_records[0].keys())[:7]
            dt_trades.add_columns(*[c.upper() for c in cols])
            for rec in trade_records:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_trades.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_trades.add_columns("STATUS")
            dt_trades.add_row(Text("No trades today", style=DIM))

        # -- Status bar --
        health = bundle.get("health", {})
        session_ok = _tms_health_flag(health, "ready")
        open_count = len([r for r in daily_records if "open" in str(r).lower() or "pending" in str(r).lower()])
        bar = self.query_one("#order-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold #ffaf00]ORDER MANAGEMENT[/]  │  "
            f"[{GAIN if session_ok else LOSS}]TMS: {'CONNECTED' if session_ok else 'DISCONNECTED'}[/]  │  "
            f"[{CYAN}]Daily orders: {len(daily_records)}[/]  │  "
            f"[{YELLOW}]Open: {open_count}[/]  │  "
            f"[#888888]Mode: PAPER[/]"
        ))

    def _on_order_submit(self) -> None:
        """Handle order form submission."""
        try:
            sym = self.query_one("#order-inp-symbol", Input).value.strip().upper()
            qty_str = self.query_one("#order-inp-qty", Input).value.strip()
            price_str = self.query_one("#order-inp-price", Input).value.strip()
            slip_str = self.query_one("#order-inp-slippage", Input).value.strip()
        except Exception:
            return

        if not sym:
            self._set_status("Order: Symbol required")
            return
        action = self._order_action
        allow_all = action == "SELL" and qty_str.strip().lower() == "all"
        if not qty_str or (not allow_all and (not qty_str.isdigit() or int(qty_str) <= 0)):
            self._set_status("Order: Valid quantity required")
            return

        qty = int(qty_str) if not allow_all else 0
        # If no price, use last close from market data
        if not price_str:
            price = self._get_ltp_for_symbol(sym)
            if not price:
                self._set_status(f"Order: No price available for {sym}")
                return
            self.query_one("#order-inp-price", Input).value = f"{price:.1f}"
        else:
            try:
                price = float(price_str.replace(",", ""))
            except ValueError:
                self._set_status("Order: Invalid price")
                return

        # Slippage %
        try:
            slippage = float(slip_str) if slip_str else 2.0
        except ValueError:
            slippage = 2.0

        if action == "SELL":
            holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
            try:
                qty = _resolve_sell_qty(sym, qty_str, holdings)
            except ValueError as exc:
                self._set_status(f"Order: {exc}")
                return

        if self.trade_mode == "live":
            self._submit_live_order(action, sym, qty, price)
        else:
            msg = self._submit_paper_order(action, sym, qty, price, slippage)
            self._set_status(msg)

        # Clear inputs except slippage (reusable)
        self.query_one("#order-inp-symbol", Input).value = ""
        self.query_one("#order-inp-qty", Input).value = ""
        self.query_one("#order-inp-price", Input).value = ""

    def _get_ltp_for_symbol(self, sym: str) -> Optional[float]:
        """Get last traded price for symbol from market data."""
        if hasattr(self, 'md'):
            for df in [self.md.quotes, self.md.gainers, self.md.losers]:
                if not df.empty and "symbol" in df.columns:
                    match = df[df["symbol"] == sym]
                    if not match.empty:
                        return float(match.iloc[0].get("close", 0))
        return None

    def _refresh_order_action_buttons(self) -> None:
        """Apply active styling to BUY / SELL toggle without mutating labels."""
        try:
            buy_btn = self.query_one("#order-btn-buy", Button)
            sell_btn = self.query_one("#order-btn-sell", Button)
        except Exception:
            return
        if self._order_action == "BUY":
            buy_btn.add_class("order-action-active")
            sell_btn.remove_class("order-action-active")
        else:
            sell_btn.add_class("order-action-active")
            buy_btn.remove_class("order-action-active")

    def _preferred_order_symbol(self) -> str:
        if self.active_tab == "lookup" and self.lookup_sym:
            return str(self.lookup_sym).strip().upper()
        try:
            dt = self.query_one("#dt-portfolio", DataTable)
            row_index = dt.cursor_row
            if row_index is not None:
                row = dt.get_row_at(row_index)
                sym = str(row[0]).strip().upper()
                if sym and sym != "NO POSITIONS":
                    return sym
        except Exception:
            pass
        positions = self._stats.get("positions", []) if hasattr(self, "_stats") else []
        if positions:
            return str(positions[0].get("sym") or "").strip().upper()
        return ""

    def _preferred_sell_ticket_defaults(self) -> tuple[str, str]:
        sym = self._preferred_order_symbol()
        holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
        qty = holdings.get(sym) if sym else None
        if qty:
            return sym, str(qty)
        if holdings:
            fallback_sym = next(iter(holdings))
            return fallback_sym, str(holdings[fallback_sym])
        return "", ""

    def _preferred_order_price_text(self, sym: str) -> str:
        price = self._get_ltp_for_symbol(sym) if sym else None
        return f"{price:.1f}" if price else ""

    def _resolve_ticket_price(self, sym: str, raw_price: str) -> Optional[float]:
        if raw_price:
            try:
                return float(str(raw_price).replace(",", "").strip())
            except ValueError:
                return None
        return self._get_ltp_for_symbol(sym)

    def _parse_slippage(self, raw_value: str) -> float:
        try:
            return max(0.0, float(str(raw_value).strip() or "2.0"))
        except ValueError:
            return 2.0

    def _format_order_band(self, action: str, price: float, slippage_pct: float) -> str:
        slip = max(0.0, slippage_pct) / 100.0
        if action == "BUY":
            upper = price * (1 + slip)
            return f"≤ {upper:,.1f}"
        lower = price * (1 - slip)
        return f"≥ {lower:,.1f}"

    @work(thread=True)
    def _submit_live_order(self, action: str, sym: str, qty: int, price: float) -> None:
        """Live TMS order submission is disabled in the public paper build."""
        self.app.call_from_thread(self._set_status, "Only paper trading is supported in this build.")

    def on_unmount(self) -> None:
        """Graceful shutdown — stop trading engine."""
        try:
            if self.trade_mode == "paper":
                self._persist_active_account_snapshot()
        except Exception:
            pass
        for _eng in list(getattr(self, "_account_engines", {}).values()):
            try:
                _eng.stop()
            except Exception:
                pass
        try:
            src = getattr(self, "_tms_live_src", None)
            if src is not None:
                src.stop()
        except Exception:
            pass

    @staticmethod
    def _upsert_live_prices(snap) -> None:
        """Write live quote LTPs + NEPSE index into stock_prices."""
        import sqlite3
        from backend.quant_pro.database import get_db_path
        from backend.trading.live_trader import now_nst
        today = now_nst().strftime("%Y-%m-%d")
        rows = []
        for sym, q in snap.quotes.items():
            ltp = q.get("last_traded_price") or q.get("close_price")
            if not ltp or float(ltp) <= 0:
                continue
            ltp = float(ltp)
            vol = int(float(q.get("total_trade_quantity") or 0))
            rows.append((sym, today, ltp, ltp, ltp, ltp, vol))

        # Fetch NEPSE index from API + compute total market volume
        try:
            from nepse import Nepse
            client = Nepse()
            client.setTLSVerification(False)
            indices = client.getNepseIndex()
            for idx in (indices or []):
                if idx.get("index") == "NEPSE Index":
                    cur = float(idx.get("currentValue") or idx.get("close") or 0)
                    prev = float(idx.get("close") or idx.get("previousClose") or 0)
                    high = float(idx.get("high") or cur)
                    low = float(idx.get("low") or cur)
                    # Total market volume = sum of all stock volumes today
                    market_vol = sum(r[6] for r in rows)
                    if cur > 0:
                        rows.append(("NEPSE", today, prev, high, low, cur, market_vol))
                    break
        except Exception:
            pass

        if not rows:
            return
        try:
            conn = sqlite3.connect(str(get_db_path()))
            conn.executemany(
                "INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ── Trading Engine ─────────────────────────────────────────────────────────

    @work(thread=True)
    def _start_trading_loop(self) -> None:
        """Run the auto-trading engine in a background thread (legacy single-account)."""
        if self._trading_engine:
            self._trading_engine.run_loop()

    # ── Multi-account engine management ────────────────────────────────────────

    def _start_all_account_engines(self) -> None:
        """Create and start one TUITradingEngine per account."""
        self._account_engines = {}
        for account in list(getattr(self, "_paper_accounts", []) or []):
            account_id = str(account.get("id") or "")
            if not account_id:
                continue
            strategy_id = str(account.get("strategy_id") or "")
            strategy = strategy_registry.load_strategy(strategy_id) if strategy_id else None
            config = dict((strategy or {}).get("config") or {}) or dict(LONG_TERM_CONFIG)
            account_dir = _account_dir(account_id)
            engine = TUITradingEngine(
                capital=float(config.get("initial_capital") or INITIAL_CAPITAL),
                signal_types=list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or [])),
                max_positions=int(config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 5),
                holding_days=int(config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 40),
                sector_limit=float(config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.35),
                stop_loss_pct=float(config.get("stop_loss_pct") or LONG_TERM_CONFIG.get("stop_loss_pct") or 0.08),
                trailing_stop_pct=float(config.get("trailing_stop_pct") or LONG_TERM_CONFIG.get("trailing_stop_pct") or 0.10),
                hedge_enabled=bool(getattr(self, "_hedge_enabled", True)),
                portfolio_file=account_dir / "tui_paper_portfolio.csv",
                trade_log_file=account_dir / "tui_paper_trade_log.csv",
                nav_log_file=account_dir / "tui_paper_nav_log.csv",
                state_file=account_dir / "tui_paper_state.json",
                on_status=self._make_engine_status_cb(account_id),
                on_activity=self._make_engine_activity_cb(account_id),
                on_portfolio_changed=self._make_engine_portfolio_cb(account_id),
                on_agent_updated=self._make_engine_agent_cb(account_id),
            )
            self._account_engines[account_id] = engine
            if account_id == self._current_account_id:
                self._trading_engine = engine
            self._start_account_engine_worker(account_id)

    @work(thread=True)
    def _start_account_engine_worker(self, account_id: str) -> None:
        """Run one account's trading engine in a dedicated background worker thread."""
        engine = self._account_engines.get(account_id)
        if engine:
            engine.run_loop()

    def _make_engine_status_cb(self, account_id: str):
        """Status callback — only updates UI when this account is active."""
        def cb(msg: str) -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._set_status, msg)
        return cb

    def _make_engine_activity_cb(self, account_id: str):
        """Activity log callback — only updates UI when this account is active."""
        def cb(msg: str) -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._append_activity, msg)
        return cb

    def _make_engine_portfolio_cb(self, account_id: str):
        """Portfolio-changed callback — only refreshes UI when this account is active."""
        def cb() -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._on_engine_portfolio_changed)
        return cb

    def _make_engine_agent_cb(self, account_id: str):
        """Agent-updated callback — only refreshes UI when this account is active."""
        def cb() -> None:
            if account_id == getattr(self, "_current_account_id", ""):
                self.call_from_thread(self._on_engine_agent_updated)
        return cb

    def _append_activity(self, msg: str) -> None:
        """Append a message to the activity log in the portfolio tab."""
        try:
            scroll = self.query_one("#activity-scroll", VerticalScroll)
            label = Static(msg, classes="activity-line")
            scroll.mount(label)
            scroll.scroll_end(animate=False)
            # Keep max 100 lines
            children = list(scroll.children)
            if len(children) > 100:
                for child in children[:len(children) - 100]:
                    child.remove()
        except Exception:
            pass

    def _on_engine_portfolio_changed(self) -> None:
        """Called by engine when positions change — refresh portfolio display."""
        if self._trading_engine:
            self._stats = self._trading_engine.get_portfolio_stats()
            self._populate_portfolio_tab(self._stats)
            self._populate_risk_tab(self._stats)
            self._populate_trades_full()
        self._render_hedge_panel()

    def _on_engine_agent_updated(self) -> None:
        """Called by engine after agent analysis — refresh agents tab."""
        try:
            self._load_agent_runtime_state()
            self._populate_agent_tab()
        except Exception:
            pass

    def _load_agent_runtime_state(self) -> None:
        self._agent_analysis = load_agent_analysis() or {}
        current_account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        current_strategy_id = str((self._strategy_account_binding(current_account_id) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(current_account_id))
        if (
            str((self._agent_analysis or {}).get("account_id") or "") != current_account_id
            or str((self._agent_analysis or {}).get("strategy_id") or "") != current_strategy_id
        ):
            self._agent_analysis = {}
        all_recent = list(load_agent_history() or [])
        visible_cutoff = float(getattr(self, "_agent_visible_since", 0.0) or 0.0)
        self._agent_history, self._agent_hidden_recent_history = _split_agent_messages_by_cutoff(all_recent, visible_cutoff)
        archived_items = list(load_agent_archive_history() or [])
        self._agent_archive_count = len(archived_items) + len(self._agent_hidden_recent_history)
        if self._agent_show_archived:
            older = archived_items + list(self._agent_hidden_recent_history or [])
            self._agent_archived_history = older[-AGENT_ARCHIVE_RENDER_LIMIT:]
        else:
            self._agent_archived_history = []

    def _active_account_name(self) -> str:
        account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        for account in list(getattr(self, "_paper_accounts", []) or []):
            if str(account.get("id") or "") == account_id:
                return str(account.get("name") or account_id)
        return account_id

    def _sync_agent_account_context_env(self) -> None:
        account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        account_dir = _account_dir(account_id)
        strategy_id = str((self._strategy_account_binding(account_id) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(account_id))
        os.environ["NEPSE_ACTIVE_ACCOUNT_ID"] = account_id
        os.environ["NEPSE_ACTIVE_ACCOUNT_NAME"] = self._active_account_name()
        os.environ["NEPSE_ACTIVE_ACCOUNT_DIR"] = str(account_dir)
        os.environ["NEPSE_ACTIVE_PORTFOLIO_FILE"] = str(account_dir / "paper_portfolio.csv")
        os.environ["NEPSE_ACTIVE_STRATEGY_ID"] = strategy_id
        os.environ["NEPSE_ACTIVE_STRATEGY_NAME"] = strategy_registry.strategy_name(strategy_id)

    def _current_agent_provider_label(self) -> str:
        try:
            cfg = load_active_agent_config()
        except Exception:
            cfg = {}
        configured = str(
            cfg.get("provider_label")
            or cfg.get("backend")
            or os.environ.get("NEPSE_AGENT_PROVIDER_LABEL")
            or "gemma4_mlx"
        ).strip()
        if configured:
            return configured
        meta = dict((getattr(self, "_agent_analysis", {}) or {}).get("agent_runtime_meta") or {})
        return str(meta.get("provider") or "gemma4_mlx")

    def _stop_active_agent_chat(self, *, announce: bool = True) -> bool:
        proc = getattr(self, "_agent_chat_process", None)
        if proc is None:
            return False
        self._agent_chat_stop_requested = True
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._agent_chat_process = None
        self.call_from_thread(self._set_typing, False)
        if announce:
            provider = self._current_agent_provider_label()
            append_external_agent_chat_message("AGENT", "Stopped by user.", source="tui_chat", provider=provider)
            self.call_from_thread(self._append_chat, "AGENT", "Stopped by user.", PURPLE)
            self.call_from_thread(self._set_status, "Agent response stopped")
        return True

    def _update_agent_chat_hint(self) -> None:
        hint = self.query_one("#agent-chat-hint", Static)
        if getattr(self, "_agent_show_archived", False):
            text = "Archive view loaded. Type /recent to collapse."
        else:
            text = ""
        hint.styles.height = 1 if text else 0
        hint.update(Text(text, style=LABEL))

    def _get_agent_typing_widget(self) -> Static:
        widget = getattr(self, "_agent_typing_widget", None)
        if widget is None:
            widget = Static("", classes="chat-msg chat-note", id="agent-typing-line")
            self._agent_typing_widget = widget
        return widget

    def _detach_agent_typing_widget(self) -> None:
        widget = getattr(self, "_agent_typing_widget", None)
        if widget is not None and widget.parent is not None:
            widget.remove()

    def _ensure_agent_typing_widget(self) -> Static:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        widget = self._get_agent_typing_widget()
        if widget.parent is None:
            scroll.mount(widget)
        elif widget.parent is not scroll:
            widget.remove()
            scroll.mount(widget)
        return widget

    def _append_chat_note(self, message: str) -> None:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        note_text = Text.assemble(
            ("[SYSTEM]: ", f"bold {LABEL}"),
            (str(message or "").strip(), LABEL),
        )
        widget = Static(note_text, classes="chat-msg chat-note")
        scroll.mount(widget)
        scroll.scroll_end(animate=False)

    def _render_agent_chat_history(self) -> None:
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        scroll.remove_children()
        archived = list(getattr(self, "_agent_archived_history", []) or [])
        recent = list(getattr(self, "_agent_history", []) or [])
        if getattr(self, "_agent_show_archived", False) and archived:
            self._append_chat_note(f"ARCHIVE · showing {len(archived)} older messages")
            for item in archived:
                role = str(item.get("role") or "AGENT").upper()
                color = BLUE if role == "YOU" else AMBER if role == "AGENT" else LABEL
                self._append_chat(
                    role,
                    str(item.get("message") or ""),
                    color,
                    ts=item.get("ts"),
                    provider=item.get("provider"),
                )
            self._append_chat_note("RECENT")
        for item in recent:
            role = str(item.get("role") or "AGENT").upper()
            color = BLUE if role == "YOU" else AMBER if role == "AGENT" else LABEL
            self._append_chat(
                role,
                str(item.get("message") or ""),
                color,
                ts=item.get("ts"),
                provider=item.get("provider"),
            )
        if bool(getattr(self, "_agent_typing_visible", False)):
            self._animate_agent_typing()
        else:
            self._detach_agent_typing_widget()
        self._update_agent_chat_hint()

    # ── Auto-refresh ──────────────────────────────────────────────────────────

    def _auto_refresh(self) -> None:
        if self._refresh_inflight:
            return
        self._do_refresh()

    def _data_version(self) -> str:
        ts = getattr(self, "md", None)
        if ts is None:
            return "unknown"
        return f"{self.md.latest}:{int(self.md.ts.timestamp())}"

    def _signals_cache_key(self) -> str:
        payload = self._active_strategy_payload() or {}
        config = dict(payload.get("config") or {})
        signal_types = ",".join(str(item).strip() for item in list(config.get("signal_types") or []))
        return ":".join([
            self._data_version(),
            str(payload.get("id") or "default_c5"),
            signal_types,
            "regime" if bool(config.get("use_regime_filter", True)) else "no_regime",
        ])

    def _refresh_active_tab_view(self, *, force_watchlist_sync: bool = False) -> None:
        """Refresh only the currently visible tab to keep redraws stable."""
        tab = str(self.active_tab or "market")
        if tab == "market":
            self._populate_market()
            return
        if tab == "portfolio":
            self._populate_portfolio_and_risk()
            self._populate_trades_full()
            self._render_hedge_panel()
            return
        if tab == "signals":
            self._populate_signals_workspace()
            self._load_signals_async()
            return
        if tab == "lookup":
            if not str(self.lookup_sym or "").strip():
                self.lookup_sym = "NEPSE"
            self._populate_lookup()
            return
        if tab == "agents":
            self._populate_agent_tab()
            return
        if tab == "orders":
            self._populate_orders_tab()
            return
        if tab == "watchlist":
            self._populate_watchlist()
            if self.trade_mode == "live" and self.tms_service:
                self._refresh_watchlist_live(force=force_watchlist_sync)
            return
        if tab == "kalimati":
            self._load_macro_rates_async()
            self._populate_kalimati()
            return
        if tab == "account":
            self._populate_portfolio_and_risk()
            self._populate_paper_profile_panel(self._stats)
            return
        if tab == "strategies":
            self._populate_strategies_tab()
            return

    @work(thread=True)
    def _do_refresh(self) -> None:
        if self._refresh_inflight:
            return
        self._refresh_inflight = True
        try:
            # Pull live quotes from API + upsert today's prices into stock_prices
            try:
                from backend.quant_pro.realtime_market import get_market_data_provider
                provider = get_market_data_provider()
                snap = provider.fetch_snapshot(force=True)
                if snap and snap.quotes:
                    self._upsert_live_prices(snap)
            except Exception:
                pass

            self.md.refresh()
            self._osint_stories = _fetch_osint_stories(80)

            # Keep hidden tabs stable; only repaint the visible view.
            self.call_from_thread(self._update_header)
            self.call_from_thread(self._update_index)
            self.call_from_thread(self._build_ticker)
            self.call_from_thread(self._match_paper_orders)

            # Refresh kalimati data in the background, but repaint only if visible.
            rows, status = refresh_kalimati()
            self._kalimati_rows = rows
            self._kalimati_status = status

            self.call_from_thread(self._refresh_active_tab_view)
            ts = self.md.ts.strftime("%H:%M:%S")
            self.call_from_thread(
                self._set_status,
                f"Refreshed {ts}  │  ▲{self.md.adv} ▼{self.md.dec}  ={self.md.unch}",
            )
        finally:
            self._refresh_inflight = False

    @work(thread=True)
    def _load_regime_async(self) -> None:
        self._regime = _get_regime(self.md)
        self.call_from_thread(self._update_index)

    # ── Signal generation ─────────────────────────────────────────────────────

    @work(thread=True)
    def _load_signals_async(self, force: bool = False) -> None:
        cache_key = self._signals_cache_key()
        if (
            not force
            and cache_key == self._signals_table_cache_key
            and self._signals_table_cache_payload is not None
        ):
            cols, rows, count = self._signals_table_cache_payload
            self.call_from_thread(self._set_signals_table, cols, rows)
            return
        payload = self._active_strategy_payload() or {}
        config = dict(payload.get("config") or {})
        strategy_name = str(payload.get("name") or "Active Strategy")
        signal_types = list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or []))
        use_regime_filter = bool(config.get("use_regime_filter", True))
        self.call_from_thread(self._set_signals_table_loading, strategy_name)
        self.call_from_thread(self._set_status, f"Loading signals for {strategy_name}...")
        try:
            from backend.backtesting.simple_backtest import load_all_prices
            from backend.trading.live_trader import generate_signals
            conn = _db()
            prices_df = load_all_prices(conn)
            conn.close()
            sigs, _regime = generate_signals(
                prices_df,
                signal_types,
                use_regime_filter=use_regime_filter,
            )
            min_score = float(getattr(self, "_signal_min_score", 0.0))
            if min_score > 0:
                sigs = [s for s in sigs if float(s.get("score") or 0.0) >= min_score]
            sigs = list(sorted(sigs, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:50])

            cols = [("  #", "n"), ("SYMBOL", "sym"), ("SCORE", "score"),
                    ("TYPE", "type"), ("STR", "str"), ("CONF", "conf"), ("DIR", "dir")]
            rows = []
            for i, s in enumerate(sigs, 1):
                score = float(s.get("score") or 0.0)
                strength = float(s.get("strength") or 0.0)
                confidence = float(s.get("confidence") or 0.0)
                signal_type = str(s.get("signal_type") or "")
                score_style = GAIN_HI if score > 0 else LOSS if score < 0 else LABEL
                rows.append([
                    _dim_text(str(i)), _sym_text(str(s.get("symbol") or "")),
                    Text(f"{score:.3f}", style=score_style),
                    Text(signal_type, style=CYAN),
                    Text(f"{strength:.2f}", style=WHITE),
                    Text(f"{confidence:.2f}", style=WHITE),
                    Text("▲ LONG", style=f"bold {GAIN_HI}"),
                ])
            self._signals_last_strategy_name = strategy_name
            self._signals_last_count = len(sigs)
            self._signals_last_loaded_at = datetime.now().strftime("%H:%M:%S")
            publish_agent_signal_snapshot(
                {
                    "account_id": str(getattr(self, "_current_account_id", "account_1") or "account_1"),
                    "strategy_id": str((self._strategy_account_binding(getattr(self, "_current_account_id", "account_1")) or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(getattr(self, "_current_account_id", "account_1"))),
                    "strategy_name": strategy_name,
                    "context_date": str(getattr(self.md, "latest", datetime.now().strftime("%Y-%m-%d")) or datetime.now().strftime("%Y-%m-%d")),
                    "regime": _regime,
                    "signals": list(sigs),
                }
            )
            self.call_from_thread(self._render_signals_table_payload, cache_key, cols, rows, len(sigs))
            self.call_from_thread(self._update_signals_table_title)
            self.call_from_thread(self._set_status,
                f"Signals loaded: {len(sigs)} │ {strategy_name} │ Session: {self.md.latest}")
        except Exception as e:
            self.call_from_thread(self._set_status, f"Signal error: {e}")

    def _render_signals_table_payload(
        self,
        cache_key: str,
        cols: list[tuple[str, str]],
        rows: list[list[Text]],
        count: int,
    ) -> None:
        self._signals_table_cache_key = cache_key
        self._signals_table_cache_payload = (cols, rows, count)
        self._set_signals_table(cols, rows)

    def _set_signals_table_loading(self, strategy_name: str = "") -> None:
        label = str(strategy_name or self._signals_last_strategy_name or self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))).strip()
        suffix = f" — {label}" if label else ""
        try:
            self.query_one("#signals-table-title", Static).update(f"SIGNALS{suffix} — Loading...  |  R/F force reload")
        except Exception:
            pass

    def _update_signals_table_title(self) -> None:
        strategy_name = str(self._signals_last_strategy_name or self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))).strip()
        loaded_at = str(self._signals_last_loaded_at or "not loaded")
        count = int(self._signals_last_count or 0)
        title = f"SIGNALS — {strategy_name}  |  {count} loaded  |  {loaded_at}  |  R/F force reload"
        try:
            self.query_one("#signals-table-title", Static).update(title)
        except Exception:
            pass

    def _set_signals_table(self, cols, rows):
        dt = self.query_one("#dt-signals", DataTable)
        dt.clear(columns=True)
        for label, key in cols:
            dt.add_column(label, key=key)
        for row in rows:
            dt.add_row(*row)

    # ── Rates & Commodities ──────────────────────────────────────────────────

    @work(thread=True)
    def _load_kalimati_async(self) -> None:
        self.call_from_thread(self._set_status, "Loading commodity prices...")
        try:
            rows, status = refresh_kalimati()
            self._kalimati_rows = rows
            self._kalimati_status = status
        except Exception as e:
            self._kalimati_rows = get_kalimati_display_rows()
            self._kalimati_status = f"Rates & Commodities: {e}"
        self.call_from_thread(self._populate_kalimati)

    @work(thread=True)
    def _load_macro_rates_async(self, force: bool = False) -> None:
        now_ts = time.monotonic()
        if not force and (now_ts - self._last_macro_rates_fetch_at) < 1800:
            return
        metals = _fetch_gold_silver_prices()
        noc = _fetch_noc_fuel_prices()
        forex_rows = _fetch_nrb_forex_rates(("USD", "EUR", "GBP", "INR", "CNY", "JPY", "AUD", "CAD"))
        indicator_rows: list[dict] = []
        indicator_history = _load_macro_indicator_history()
        snapshot_ts = datetime.utcnow().isoformat()

        if metals:
            gold = float(metals.get("gold_per_tola") or 0)
            silver = float(metals.get("silver_per_tola") or 0)
            if gold > 0:
                # Prefer DB-stored CHG% (cross-session, today vs yesterday)
                # Fall back to in-session memory tracker if DB has no previous day yet
                db_gold_chg_pct = metals.get("gold_chg_pct")
                db_gold_chg_abs = metals.get("gold_chg_abs")
                if db_gold_chg_pct is not None:
                    gold_change = db_gold_chg_abs
                    gold_change_pct = db_gold_chg_pct
                else:
                    gold_change, gold_change_pct = _apply_indicator_history_change(
                        indicator_history, key="gold_per_tola",
                        value=gold, timestamp=snapshot_ts,
                    )
                indicator_rows.append(
                    {
                        "item": "Gold / Tola",
                        "group": "Metals",
                        "value": gold,
                        "unit": "NPR/tola",
                        "change": gold_change,
                        "change_pct": gold_change_pct,
                        "source": "FENEGOSIDA",
                    }
                )
            if silver > 0:
                db_silver_chg_pct = metals.get("silver_chg_pct")
                db_silver_chg_abs = metals.get("silver_chg_abs")
                if db_silver_chg_pct is not None:
                    silver_change = db_silver_chg_abs
                    silver_change_pct = db_silver_chg_pct
                else:
                    silver_change, silver_change_pct = _apply_indicator_history_change(
                        indicator_history, key="silver_per_tola",
                        value=silver, timestamp=snapshot_ts,
                    )
                indicator_rows.append(
                    {
                        "item": "Silver / Tola",
                        "group": "Metals",
                        "value": silver,
                        "unit": "NPR/tola",
                        "change": silver_change,
                        "change_pct": silver_change_pct,
                        "source": "FENEGOSIDA",
                    }
                )

        for symbol, label in (("CL=F", "WTI Crude"), ("BZ=F", "Brent Crude")):
            quote = _fetch_yahoo_futures_price(symbol, label)
            if quote:
                indicator_rows.append(
                    {
                        "item": label,
                        "group": "Energy",
                        "value": float(quote.get("value") or 0),
                        "unit": quote.get("unit") or "USD",
                        "change_pct": quote.get("change_pct"),
                        "source": quote.get("source") or "Yahoo Finance",
                    }
                )

        if noc:
            for label, key, unit in (
                ("Petrol", "petrol", "NPR/L"),
                ("Diesel", "diesel", "NPR/L"),
                ("Kerosene", "kerosene", "NPR/L"),
                ("LPG", "lpg", "NPR/cylinder"),
            ):
                value = float(noc.get(key) or 0)
                if value <= 0:
                    continue
                change, change_pct = _apply_indicator_history_change(
                    indicator_history,
                    key=f"noc_{key}",
                    value=value,
                    timestamp=snapshot_ts,
                )
                indicator_rows.append(
                    {
                        "item": label,
                        "group": "NOC",
                        "value": value,
                        "unit": unit,
                        "change": change,
                        "change_pct": change_pct,
                        "source": f"NOC {noc.get('date_bs') or ''}".strip(),
                    }
                )

        _save_macro_indicator_history(indicator_history)
        rates = {
            "metals": metals,
            "noc": noc,
            "indicators": indicator_rows,
            "forex_rows": forex_rows,
        }
        self._last_macro_rates_fetch_at = now_ts
        self._macro_rates = rates
        self.call_from_thread(self._populate_kalimati)

    def _populate_kalimati(self) -> None:
        rows = self._kalimati_rows
        status = self._kalimati_status
        filtered_rows, filtered_indicators, filtered_forex = self._get_filtered_rates_payload()

        # ── Status bar ────────────────────────────────────────────────────────
        with_chg = [r for r in filtered_rows if r.get("change_pct") is not None]
        gainers = sum(1 for r in with_chg if r["change_pct"] > 0)
        losers  = sum(1 for r in with_chg if r["change_pct"] < 0)
        date_str = rows[0]["date"] if rows else "—"
        bar_text = (
            f"[bold #ffaf00] RATES & COMMODITIES [/]"
            f"[#444444] │ [/]"
            f"[#888888]{len(filtered_rows)}/{len(rows)} commodities[/]  "
            f"[#444444] │ [/]"
            f"[#888888]{len(filtered_indicators)} macro[/]  "
            f"[#888888]{len(filtered_forex)} forex[/]"
            f"[#444444] │ [/]"
            f"[#00C853]▲ {gainers}[/]  [#ff4444]▼ {losers}[/]"
            f"[#444444] │ [/][#888888]{date_str}[/]"
            f"[#444444] │ [/][#555555]{status}[/]"
        )
        self.query_one("#kalimati-status-bar", Static).update(bar_text)

        self.query_one("#kalimati-left-title", Static).update(
            f"KALIMATI COMMODITIES [{len(filtered_rows)}]"
        )
        self.query_one("#kalimati-right-title", Static).update(
            f"GLOBAL RATES & PRICES [{len(filtered_indicators) + len(filtered_forex)}]"
        )
        self.query_one("#macro-top-title", Static).update(
            f"METALS, ENERGY & NOC [{len(filtered_indicators)}]"
        )
        self.query_one("#macro-forex-title", Static).update(
            f"FOREX RATES [{len(filtered_forex)}]"
        )

        # ── Top movers bar (only shown if we have change data) ────────────────
        if with_chg:
            top_gain = sorted(with_chg, key=lambda r: r["change_pct"], reverse=True)[:5]
            top_lose = sorted(with_chg, key=lambda r: r["change_pct"])[:5]
            parts: list[str] = ["[#555555] MOVERS [/]"]
            for r in top_gain:
                nm = r["name_english"].split("(")[0].strip()[:14]
                parts.append(f"[bold #00C853]▲[/][#00C853]{nm} +{r['change_pct']:.1f}%[/]")
            parts.append("[#333333] ┃ [/]")
            for r in top_lose:
                nm = r["name_english"].split("(")[0].strip()[:14]
                parts.append(f"[bold #ff4444]▼[/][#ff4444]{nm} {r['change_pct']:.1f}%[/]")
            movers_markup = "  ".join(parts)
        else:
            # No prev-day data yet — show price stats instead
            if rows:
                most_exp = max(rows, key=lambda r: r["avg"])
                cheapest = min(rows, key=lambda r: r["avg"])
                avg_price = sum(r["avg"] for r in rows) / len(rows)
                movers_markup = (
                    f"[#555555] MARKET STATS [/]  "
                    f"[#888888]Avg Price:[/] [#ffaf00]{avg_price:.2f}[/]  "
                    f"[#555555]│[/]  "
                    f"[#888888]Most Expensive:[/] [#ffaf00]{most_exp['name_english'].split('(')[0].strip()[:20]}[/] "
                    f"[bold #ffaf00]{most_exp['avg']:,.0f}[/]  "
                    f"[#555555]│[/]  "
                    f"[#888888]Cheapest:[/] [#00C853]{cheapest['name_english'].split('(')[0].strip()[:20]}[/] "
                    f"[bold #00C853]{cheapest['avg']:,.0f}[/]  "
                    f"[#444444]│  Price change available after 2nd fetch[/]"
                )
            else:
                movers_markup = "[#444444] Fetching commodities data...[/]"
        self.query_one("#kalimati-movers-bar", Static).update(movers_markup)

        # ── Kalimati table ────────────────────────────────────────────────────
        dt = self.query_one("#dt-kalimati", DataTable)
        dt.clear(columns=True)

        dt.add_column("COMMODITY", key="name", width=26)
        dt.add_column("U",         key="unit", width=4)
        dt.add_column("AVG",    key="avg",  width=9)
        dt.add_column("CHANGE", key="chg",  width=9)
        dt.add_column("CHG %",  key="pct",  width=7)
        dt.add_column("MIN",    key="min",  width=9)
        dt.add_column("MAX",    key="max",  width=9)
        dt.add_column("▕  RANGE  ▏", key="rng", width=12)

        def _range_bar(mn: float, mx: float, avg: float, width: int = 10) -> Text:
            """Show where avg sits within the [min, max] band."""
            if mx <= mn:
                return Text("▕" + "─" * width + "▏", style=LABEL)
            pos = int((avg - mn) / (mx - mn) * width)
            pos = max(0, min(width - 1, pos))
            bar = "░" * pos + "█" + "░" * (width - pos - 1)
            # Colour: green if avg > midpoint, red if below
            mid = (mn + mx) / 2
            color = GAIN if avg >= mid else LOSS
            return Text(f"▕{bar}▏", style=color)

        def _chg_text(chg: float | None, pct: float | None) -> tuple[Text, Text]:
            """Return (change_text, pct_text) coloured appropriately."""
            if chg is None or pct is None:
                return Text("   ─", style=LABEL), Text("  ─", style=LABEL)
            if pct > 0:
                style, arrow = GAIN_HI if pct > 5 else GAIN, "▲"
            elif pct < 0:
                style, arrow = LOSS_HI if pct < -5 else LOSS, "▼"
            else:
                style, arrow = LABEL, "─"
            chg_str = f"{arrow}{abs(chg):,.1f}"
            pct_str = f"{'+' if pct > 0 else ''}{pct:.1f}%"
            return Text(chg_str, style=style), Text(pct_str, style=style)

        for r in filtered_rows:
            pct = r.get("change_pct")
            chg_t, pct_t = _chg_text(r.get("change"), pct)

            name = r["name_english"]
            is_nepali = any(ord(c) > 0x900 for c in name[:6])
            if is_nepali:
                name_text = Text(name[:25], style=f"italic {LABEL}")
            elif pct is not None and abs(pct) >= 5:
                name_text = Text(name[:25], style=f"bold {WHITE}")
            else:
                name_text = Text(name[:25], style=WHITE)

            dt.add_row(
                name_text,
                Text(r["unit"][:4], style=LABEL),
                Text(f"{r['avg']:>8,.1f}", style=f"bold {AMBER}"),
                chg_t,
                pct_t,
                Text(f"{r['min']:>8,.1f}", style=DIM),
                Text(f"{r['max']:>8,.1f}", style=DIM),
                _range_bar(r["min"], r["max"], r["avg"], width=10),
            )

        # ── Macro indicators ──────────────────────────────────────────────────
        dt_macro = self.query_one("#dt-macro", DataTable)
        dt_macro.clear(columns=True)
        dt_macro.add_column("ITEM", key="item", width=16)
        dt_macro.add_column("GROUP", key="group", width=9)
        dt_macro.add_column("VALUE", key="value", width=14)
        dt_macro.add_column("CHG %", key="pct", width=8)
        dt_macro.add_column("SOURCE", key="source", width=16)

        for row in filtered_indicators:
            value = float(row.get("value") or 0)
            unit = str(row.get("unit") or "")
            if unit.startswith("NPR"):
                value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}")
            else:
                value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}")

            pct = row.get("change_pct")
            if pct is None:
                pct_text = Text("—", style=LABEL)
            elif float(pct) > 0:
                pct_text = Text(f"+{float(pct):.2f}%", style=GAIN)
            elif float(pct) < 0:
                pct_text = Text(f"{float(pct):.2f}%", style=LOSS)
            else:
                pct_text = Text("0.00%", style=LABEL)

            dt_macro.add_row(
                Text(str(row.get("item") or "")[:16], style=WHITE),
                Text(str(row.get("group") or "")[:9], style=CYAN),
                value_text,
                pct_text,
                Text(str(row.get("source") or "")[:16], style=DIM),
            )

        # ── Forex table ───────────────────────────────────────────────────────
        dt_forex = self.query_one("#dt-forex", DataTable)
        dt_forex.clear(columns=True)
        dt_forex.add_column("CCY", key="ccy", width=6)
        dt_forex.add_column("NAME", key="name", width=18)
        dt_forex.add_column("BUY", key="buy", width=10)
        dt_forex.add_column("SELL", key="sell", width=10)
        dt_forex.add_column("CHG %", key="pct", width=8)
        dt_forex.add_column("UNIT", key="unit", width=6)
        dt_forex.add_column("SOURCE", key="source", width=8)

        for row in filtered_forex:
            buy = float(row.get("buy_rate") or 0)
            sell = float(row.get("sell_rate") or 0)
            pct = row.get("change_pct")
            if pct is None:
                pct_text = Text("—", style=LABEL)
            elif float(pct) > 0:
                pct_text = Text(f"+{float(pct):.2f}%", style=GAIN)
            elif float(pct) < 0:
                pct_text = Text(f"{float(pct):.2f}%", style=LOSS)
            else:
                pct_text = Text("0.00%", style=LABEL)
            dt_forex.add_row(
                Text(str(row.get("currency_code") or ""), style=f"bold {CYAN}"),
                Text(str(row.get("currency_name") or "")[:18], style=WHITE),
                Text(f"{buy:,.2f}", style=WHITE),
                Text(f"{sell:,.2f}", style=WHITE),
                pct_text,
                Text(str(row.get("unit") or 1), style=LABEL),
                Text(str(row.get("source") or "")[:8], style=DIM),
            )

    def _get_filtered_rates_payload(self) -> tuple[list[dict], list[dict], list[dict]]:
        query = self._rates_search_query.strip().lower()
        filtered_rows = [
            r for r in self._kalimati_rows
            if not query
            or query in str(r.get("name_english") or "").lower()
            or query in str(r.get("unit") or "").lower()
        ]
        macro_data = self._macro_rates or {}
        indicator_rows = list(macro_data.get("indicators") or [])
        forex_rows = list(macro_data.get("forex_rows") or [])
        filtered_indicators = [
            row for row in indicator_rows
            if not query
            or query in str(row.get("item") or "").lower()
            or query in str(row.get("group") or "").lower()
            or query in str(row.get("source") or "").lower()
        ]
        filtered_forex = [
            row for row in forex_rows
            if not query
            or query in str(row.get("currency_code") or "").lower()
            or query in str(row.get("currency_name") or "").lower()
            or query in str(row.get("source") or "").lower()
        ]
        return filtered_rows, filtered_indicators, filtered_forex

    # ── OSINT news feed ───────────────────────────────────────────────────────

    @work(thread=True)
    def _load_osint_async(self) -> None:
        self.call_from_thread(self._set_status, "Loading OSINT headlines...")
        self._osint_stories = _fetch_osint_stories(80)
        # Translate enough OSINT text to keep the ticker English-first.
        nepali_headlines = []
        nepali_indices = []
        nepali_summaries = []
        nepali_summary_indices = []
        for i, s in enumerate(self._osint_stories):
            hl = s.get("canonical_headline", "")
            summary = s.get("summary", "") or ""
            url = s.get("url", "") or ""
            is_nepali = hl and _contains_non_ascii(hl)
            has_english_summary = summary and not _contains_non_ascii(summary)
            has_slug = bool(_headline_fallback_from_url(url))
            if is_nepali and not has_english_summary and not has_slug:
                nepali_headlines.append(hl)
                nepali_indices.append(i)
            if summary and _contains_non_ascii(summary):
                nepali_summaries.append(summary)
                nepali_summary_indices.append(i)
        if nepali_headlines:
            self.call_from_thread(self._set_status,
                f"Translating {len(nepali_headlines)} Nepali headlines...")
            translated = _translate_batch(nepali_headlines)
            for idx, trans in zip(nepali_indices, translated):
                self._osint_stories[idx]["_translated"] = trans
        if nepali_summaries:
            translated_summaries = _translate_batch(nepali_summaries)
            for idx, trans in zip(nepali_summary_indices, translated_summaries):
                self._osint_stories[idx]["_translated_summary"] = trans
        self.call_from_thread(self._build_ticker)
        n = len(self._osint_stories)
        self.call_from_thread(
            self._set_status,
            f"OSINT headlines loaded: {n} stories │ Session: {self.md.latest}",
        )

    def _populate_news(self) -> None:
        stories = self._osint_stories
        query = self._news_search_query.strip()
        is_vector = bool(self._vector_search_results)

        # If we have vector search results, show those instead
        if query and is_vector:
            stories = self._vector_search_results

        self._news_visible_stories = list(stories)
        if not stories:
            self._news_selected_index = 0
        else:
            self._news_selected_index = max(0, min(self._news_selected_index, len(stories) - 1))

        # News list (OptionList — avoids DataTable cell-level positioning
        # which garbles Devanagari combining characters)
        ol = self.query_one("#news-list", OptionList)
        ol.clear_options()

        if stories:
            options: list[Option] = []
            for i, s in enumerate(stories, 1):
                ts_raw = s.get("first_reported_at", "")
                ts = ts_raw[11:16] if ts_raw else ""
                sev = s.get("severity", "")
                stype = str(s.get("story_type", "") or "").strip()
                src = s.get("source_name", "")
                src = src.replace(" (Nepali)", "").replace("(Nepal", "")[:14]
                headline = unicodedata.normalize('NFC', _news_display_headline(s))

                sev_style = SEVERITY_STYLE.get(sev, LABEL)
                sim = s.get("_similarity")
                hl_display = headline
                if sim is not None:
                    hl_display = f"[{sim:.0%}] {headline}"
                hl_display = _truncate_text(hl_display, 64)

                row = Text()
                row.append(f"{hl_display}\n", style=f"bold {WHITE}")
                row.append(f"{ts:5s}", style=LABEL)
                row.append("  ")
                row.append(sev.upper()[:4], style=sev_style)
                row.append("  ")
                row.append(src, style=DIM)
                if stype:
                    row.append("  •  ", style=LABEL)
                    row.append(_truncate_text(stype, 10), style=TYPE_STYLE.get(stype, CYAN))
                options.append(Option(row, id=f"news-{i}"))
            ol.add_options(options)
            if self._news_selected_index < len(stories):
                ol.highlighted = self._news_selected_index
        else:
            ol.add_option(Option(Text("No stories available right now.", style=LABEL)))

        # Summary bar
        brief_bar = self.query_one("#news-brief-bar", Static)
        n = len(stories)
        high_n = sum(1 for s in stories if s.get("severity") in ("high", "critical"))
        med_n = sum(1 for s in stories if s.get("severity") == "medium")
        types = {}
        for s in stories:
            t = s.get("story_type", "other")
            types[t] = types.get(t, 0) + 1
        type_parts = "  ".join(
            f"[{TYPE_STYLE.get(t, LABEL)}]{t}:{c}[/]"
            for t, c in sorted(types.items(), key=lambda x: -x[1])
        )
        search_info = ""
        if query and is_vector:
            search_info = f"[bold #ffaf00]  ⌕ VECTOR \"{query}\"  {n} results[/]   "
        elif query:
            search_info = f"[bold #ffaf00]  ⌕ \"{query}\"[/]   "
        brief_bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ LIVE NEWS FEED[/]   "
            f"{search_info}"
            f"[#888888]Total:[/] [bold {WHITE}]{n}[/]   "
            f"[{LOSS_HI}]▲{high_n} high[/]   "
            f"[{YELLOW}]{med_n} medium[/]   "
            f"{type_parts}"
        ))

    @work(thread=True)
    def _do_vector_search(self, query: str) -> None:
        """Semantic search via OSINT embeddings API, with local text fallback."""
        self.call_from_thread(self._set_status, f"Vector search: \"{query}\"...")
        vector_ok = False
        try:
            r = _requests.post(
                f"{OSINT_BASE}/embeddings/search",
                json={"query": query, "top_k": 40, "hours": 8760, "min_similarity": 0.3},
                timeout=OSINT_TIMEOUT + 4,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if results:
                stories = []
                for item in results:
                    stories.append({
                        "canonical_headline": item.get("title", ""),
                        "source_name": item.get("source_name", ""),
                        "first_reported_at": item.get("published_at", ""),
                        "severity": item.get("severity", ""),
                        "story_type": item.get("story_type", item.get("category", "")),
                        "summary": "",
                        "display_location": "",
                        "_similarity": item.get("similarity", 0),
                        "_story_id": item.get("story_id", ""),
                    })
                self._vector_search_results = stories
                vector_ok = True
                self.call_from_thread(self._populate_news)
                self.call_from_thread(
                    self._set_status,
                    f"Vector: {len(stories)} results for \"{query}\""
                )
        except Exception:
            pass

        # Fallback: local text filter on loaded stories
        if not vector_ok:
            terms = query.lower().split()
            filtered = []
            for s in self._osint_stories:
                haystack = " ".join([
                    s.get("canonical_headline", ""),
                    s.get("summary", "") or "",
                    s.get("_translated", "") or "",
                    s.get("source_name", "") or "",
                    s.get("story_type", "") or "",
                ]).lower()
                if all(t in haystack for t in terms):
                    filtered.append(s)
            self._vector_search_results = filtered
            self.call_from_thread(self._populate_news)
            self.call_from_thread(
                self._set_status,
                f"Text search: {len(filtered)} results for \"{query}\" (vector empty, local fallback)"
            )

    # ── Agent tab ─────────────────────────────────────────────────────────────

    @work(thread=True)
    def _run_agent_analysis(self, force: bool = True) -> None:
        self._sync_agent_account_context_env()
        self.call_from_thread(self._set_status, "⧖ Agent analyzing..." if force else "⧖ Loading agent...")
        self.call_from_thread(self._update_agent_status, "ANALYZING..." if force else "LOADING...", "running")
        try:
            preview = build_algo_shortlist_snapshot()
            self.call_from_thread(self._set_agent_shortlist_preview, preview)
            result = agent_analyze(force=force)
            self._agent_analysis = result
            self.call_from_thread(self._populate_agent_tab)
            self.call_from_thread(self._maybe_submit_agent_super_signal, result)
            age = time.time() - result.get("timestamp", 0)
            if age < 5:
                self.call_from_thread(self._set_status,
                    f"Agent analysis complete │ {len(result.get('stocks', []))} stocks reviewed")
            else:
                self.call_from_thread(self._set_status,
                    f"Agent loaded from cache ({int(age/60)}m ago) │ Press A+Enter to refresh")
        except Exception as e:
            self.call_from_thread(self._set_status, f"Agent error: {e}")

    def _set_agent_shortlist_preview(self, preview: dict) -> None:
        current_rows = list((getattr(self, "_agent_analysis", {}) or {}).get("stocks") or [])
        if current_rows and not all(str(row.get("verdict") or "").upper() == "REVIEW" for row in current_rows):
            return
        self._agent_preview_override = dict(preview or {})
        self._populate_agent_tab()

    def _set_typing(self, visible: bool):
        self._agent_typing_visible = bool(visible)
        if not visible:
            self._agent_typing_frame = 0
            self._detach_agent_typing_widget()
        else:
            self._animate_agent_typing()

    def _animate_agent_typing(self) -> None:
        if not bool(getattr(self, "_agent_typing_visible", False)):
            self._detach_agent_typing_widget()
            return
        typing_w = self._ensure_agent_typing_widget()
        frames = ["[AGENT] : Typing.", "[AGENT] : Typing..", "[AGENT] : Typing..."]
        frame = frames[int(getattr(self, "_agent_typing_frame", 0)) % len(frames)]
        self._agent_typing_frame = int(getattr(self, "_agent_typing_frame", 0)) + 1
        typing_w.update(Text(frame, style=f"italic {LABEL}"))
        try:
            self.query_one("#agent-chat-scroll", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    @work(thread=True)
    def _agent_ask_async(self, question: str) -> None:
        existing = getattr(self, "_agent_chat_process", None)
        if existing is not None and existing.poll() is None:
            self.call_from_thread(self._set_status, "Agent is already thinking | use /stop to cancel")
            return

        self._sync_agent_account_context_env()
        self._agent_chat_request_id = int(getattr(self, "_agent_chat_request_id", 0)) + 1
        request_id = self._agent_chat_request_id
        self._agent_chat_stop_requested = False
        provider = self._current_agent_provider_label()
        append_external_agent_chat_message("YOU", question, source="tui_chat", provider=provider)
        self.call_from_thread(self._append_chat, "YOU", question, BLUE)
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "agents" / "run_active_agent.py"),
                "--question",
                question,
            ]
            env = dict(os.environ)
            env["NEPSE_AGENT_DISABLE_HISTORY"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._agent_chat_process = proc
            self.call_from_thread(self._set_typing, True)
            self.call_from_thread(self._set_status, "⧖ Agent thinking...")

            started = time.monotonic()
            while True:
                if request_id != int(getattr(self, "_agent_chat_request_id", 0)):
                    self._stop_active_agent_chat(announce=False)
                    return
                if bool(getattr(self, "_agent_chat_stop_requested", False)):
                    self._stop_active_agent_chat(announce=False)
                    return
                if proc.poll() is not None:
                    break
                if (time.monotonic() - started) > AGENT_CHAT_TIMEOUT_SECS:
                    self._stop_active_agent_chat(announce=False)
                    append_external_agent_chat_message(
                        "AGENT",
                        f"Timed out after {AGENT_CHAT_TIMEOUT_SECS}s. Ask a tighter question or try again.",
                        source="tui_chat",
                        provider=provider,
                    )
                    self.call_from_thread(self._set_typing, False)
                    self.call_from_thread(
                        self._append_chat,
                        "AGENT",
                        f"Timed out after {AGENT_CHAT_TIMEOUT_SECS}s. Ask a tighter question or try again.",
                        PURPLE,
                    )
                    self.call_from_thread(self._set_status, "Agent timed out")
                    return
                time.sleep(0.1)

            stdout, stderr = proc.communicate(timeout=1)
            self._agent_chat_process = None
            answer = str(stdout or "").strip()
            if not answer:
                answer = " ".join(str(stderr or "Agent returned no output.").split())[:280]
            append_external_agent_chat_message("AGENT", answer, source="tui_chat", provider=provider)
            self.call_from_thread(self._set_typing, False)
            self.call_from_thread(self._append_chat, "AGENT", answer, PURPLE)
            self.call_from_thread(self._set_status, "Agent responded")
        except Exception as e:
            self._agent_chat_process = None
            self.call_from_thread(self._set_typing, False)
            self.call_from_thread(self._append_chat, "ERROR", str(e), LOSS_HI)

    def _update_agent_status(self, text_str: str, state: str = "idle"):
        bar = self.query_one("#agent-status-bar", Static)
        a = getattr(self, '_agent_analysis', {})
        if state == "running":
            bar.update(Text.from_markup(
                f"[bold #bb88ff]◆ AGENT[/]   [{YELLOW}]⧖ {text_str}[/]"))
            return
        trade = a.get("trade_today", None)
        tc = GAIN_HI if trade else LOSS_HI if trade is False else LABEL
        trade_str = "YES" if trade else "NO" if trade is False else "?"
        stocks = list(a.get("stocks", []) or a.get("shortlist", []) or [])
        n_approve = sum(1 for s in stocks if s.get("verdict", "").upper() == "APPROVE")
        n_reject = sum(1 for s in stocks if s.get("verdict", "").upper() == "REJECT")
        n_total = len(stocks)
        n_super = sum(1 for s in stocks if bool(s.get("auto_entry_candidate")))
        regime = a.get("regime", "?")
        meta = dict(a.get("agent_runtime_meta") or {})
        provider = self._current_agent_provider_label().upper()
        import time as _t
        age = _t.time() - a.get("timestamp", 0)
        age_str = f"{int(age/60)}m ago" if age < 3600 else f"{int(age/3600)}h ago" if age < 86400 else "stale"
        bar.update(Text.from_markup(
            f"[bold #bb88ff]◆ {provider} AGENT[/]   "
            f"[#888888]Trade today:[/] [bold {tc}]{trade_str}[/]   "
            f"[#888888]Verdicts:[/] [bold {GAIN_HI}]{n_approve}✓[/] "
            f"[bold {LOSS_HI}]{n_reject}✗[/] "
            f"[#888888]of {n_total}[/]   "
            f"[#888888]Super:[/] [bold {AMBER}]{n_super}[/]   "
            f"[#888888]Regime:[/] [{YELLOW}]{regime}[/]   "
            f"[#888888]Updated:[/] [{LABEL}]{age_str}[/]   "
            f"[#555555]A=Analyze  │  /history  │  /recent[/]"
        ))

    def _populate_agent_meta_headers(self, analysis: dict) -> None:
        stocks = list((analysis or {}).get("stocks") or [])
        left = self.query_one("#agent-picks-subtitle", Static)
        right = self.query_one("#agent-chat-subtitle", Static)
        provider = self._current_agent_provider_label().upper()
        if stocks:
            buy_count = sum(1 for row in stocks if str(row.get("action_label") or "").upper() == "BUY")
            super_count = sum(1 for row in stocks if bool(row.get("auto_entry_candidate")))
            left.update(
                Text(
                    f"Ranked shortlist · {len(stocks)} names · {buy_count} buys · {super_count} super signals",
                    style=LABEL,
                )
            )
        else:
            left.update(Text("Refreshing live snapshot and building the ranked shortlist.", style=LABEL))
        archive_count = int(getattr(self, "_agent_archive_count", 0) or 0)
        right.update(
            Text(
                f"{provider} conversation stream · live session",
                style=LABEL,
            )
        )

    def _populate_agent_market_banner(self, analysis: dict) -> None:
        banner = self.query_one("#agent-market-view", Static)
        stocks = list((analysis or {}).get("stocks") or [])
        fresh_market = dict((analysis or {}).get("fresh_market") or {})
        parts: list[str] = []
        regime = str((analysis or {}).get("regime") or "unknown").upper()
        if regime and regime != "UNKNOWN":
            parts.append(f"[#888888]Regime[/] [bold {YELLOW}]{regime}[/]")
        session_date = str(fresh_market.get("session_date") or (analysis or {}).get("context_date") or "")
        if session_date:
            parts.append(f"[#888888]Session[/] [bold {WHITE}]{session_date}[/]")
        quote_count = int(fresh_market.get("quote_count") or 0)
        if quote_count > 0:
            parts.append(f"[#888888]Quotes[/] [bold {AMBER}]{quote_count}[/]")
        if stocks:
            top = stocks[0]
            parts.append(
                f"[#888888]Top[/] [bold {AMBER}]{str(top.get('symbol') or '—')}[/] "
                f"[{WHITE}]{str(top.get('action_label') or top.get('verdict') or 'REVIEW').upper()}[/] "
                f"[#888888]score[/] [bold {AMBER}]{float(top.get('signal_score') or 0.0):.2f}[/]"
            )
        elif not parts:
            parts.append("[#888888]Agent snapshot is loading[/]")
        banner.update(Text.from_markup("   [#444444]│[/]   ".join(parts)))

    def _populate_agent_detail_default(self, analysis: dict) -> None:
        try:
            self.query_one("#agent-detail-title", Static).update("FOCUS")
        except Exception:
            pass
        detail = self.query_one("#agent-detail", Static)
        stocks = list((analysis or {}).get("stocks") or [])
        fresh_market = dict((analysis or {}).get("fresh_market") or {})
        if stocks:
            top = stocks[0]
            top_symbol = str(top.get("symbol") or "—")
            top_action = str(top.get("action_label") or top.get("verdict") or "REVIEW").upper()
            top_score = float(top.get("signal_score") or 0.0)
            top_conv = float(top.get("conviction") or 0.0)
            summary = (
                f"Top setup: {top_symbol} {top_action}  "
                f"score {top_score:.2f}  conv {top_conv:.0%}\n"
                f"{str(top.get('what_matters') or top.get('reasoning') or '')[:220]}"
            )
            detail.update(Text(summary, style=WHITE))
            return
        session_date = str(fresh_market.get("session_date") or (analysis or {}).get("context_date") or "—")
        source = str(fresh_market.get("source") or "snapshot")
        quote_count = int(fresh_market.get("quote_count") or 0)
        detail.update(
            Text(
                f"Waiting for ranked picks.\n"
                f"Session: {session_date}  Source: {source}  Quotes: {quote_count}",
                style=LABEL,
            )
        )

    def _show_agent_focus_row(self, index: int) -> None:
        analysis = getattr(self, "_agent_analysis", {}) or {}
        stocks = list(analysis.get("stocks") or [])
        if not (0 <= int(index) < len(stocks)):
            self._populate_agent_detail_default(analysis)
            return

        stock = dict(stocks[int(index)] or {})
        symbol = str(stock.get("symbol") or "").upper()
        verdict = str(stock.get("verdict") or "?").upper()
        action_label = str(stock.get("action_label") or verdict or "REVIEW").upper()
        action_color = {
            "BUY": GAIN_HI,
            "SELL": LOSS_HI,
            "PASS": LOSS_HI,
            "HOLD": YELLOW,
            "REVIEW": LABEL,
        }.get(action_label, WHITE)
        conviction = float(stock.get("conviction", 0) or 0.0)
        signal_score = float(stock.get("signal_score") or 0.0)
        detail = Text.assemble(
            (f"[{symbol}] ", f"bold {AMBER}"),
            (f"{action_label} ", f"bold {action_color}"),
            (f"score {signal_score:.2f} ", YELLOW),
            (f"conv {conviction:.0%}\n", YELLOW),
            (str(stock.get("what_matters") or stock.get("reasoning") or ""), WHITE),
            ("\n", WHITE),
            (f"Bull: {str(stock.get('bull_case') or 'n/a')}\n", GAIN),
            (f"Risk: {str(stock.get('bear_case') or 'n/a')}", LOSS if stock.get("bear_case") else LABEL),
        )
        self.query_one("#agent-detail-title", Static).update(f"FOCUS · {symbol}")
        self.query_one("#agent-detail", Static).update(detail)

    def _populate_agent_tab(self):
        self._load_agent_runtime_state()
        preview = dict(getattr(self, "_agent_preview_override", {}) or {})
        live_rows = list((self._agent_analysis or {}).get("stocks") or [])
        if preview and not live_rows:
            self._agent_analysis = preview
        elif live_rows and not all(str(row.get("verdict") or "").upper() == "REVIEW" for row in live_rows):
            self._agent_preview_override = None
        a = getattr(self, '_agent_analysis', {})
        self._update_agent_status("", "idle")
        self._populate_agent_market_banner(a)
        self._populate_agent_meta_headers(a)
        self._render_agent_chat_history()

        # Verdicts table — short summary, full reasoning on row select
        dt = self.query_one("#dt-agent-verdicts", DataTable)
        dt.clear(columns=True)
        for label, key, width in [
            (" #", "n", 2),
            ("SYMBOL", "sym", 6),
            ("SIGNAL", "algo", 9),
            ("ACTION", "v", 8),
            ("CONV", "conv", 4),
            ("SCORE", "score", 5),
            ("KEY POINT", "kp", 32),
        ]:
            dt.add_column(label, key=key, width=width)

        stocks = list(a.get("stocks", []) or a.get("shortlist", []) or [])
        if stocks:
            for i, s in enumerate(stocks, 1):
                verdict = s.get("verdict", "?").upper()
                action_label = str(s.get("action_label") or verdict or "REVIEW").upper()
                vc = {
                    "BUY": f"bold {GAIN_HI}",
                    "SELL": f"bold {LOSS_HI}",
                    "PASS": f"bold {LOSS_HI}",
                    "HOLD": f"bold {YELLOW}",
                    "REVIEW": LABEL,
                }.get(action_label, LABEL)
                conv = s.get("conviction", 0)
                conv_c = GAIN_HI if conv >= 0.7 else YELLOW if conv >= 0.4 else LOSS
                # Extract first sentence as key point
                reasoning = str(s.get("what_matters") or s.get("reasoning") or "")
                first_sentence = reasoning.split(". ")[0].split(" — ")[0][:32]
                score = float(s.get("signal_score") or 0.0)
                score_style = GAIN_HI if score >= 1.0 else YELLOW if score >= 0.7 else LABEL
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    _sym_text(s.get("symbol", "")),
                    Text(str(s.get("signal_type") or s.get("algo_signal") or "")[:8], style=f"bold {AMBER}"),
                    Text(f" {action_label} ", style=vc),
                    Text(f"{conv:.0%}", style=conv_c),
                    Text(f"{score:.2f}", style=score_style),
                    Text(first_sentence, style=WHITE),
                )
        else:
            dt.add_row(
                _dim_text("—"), _dim_text("Loading top picks"),
                *[Text("")] * 5)
        self._populate_agent_detail_default(a)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection across tabs."""
        # Watchlist: Enter → lookup
        if event.data_table.id == "dt-watchlist":
            idx = event.cursor_row
            stock_rows = getattr(self, "_watchlist_stock_rows", [])
            if 0 <= idx < len(stock_rows):
                entry = stock_rows[idx]
                if str(entry.get("kind") or "stock") == "stock":
                    self.lookup_sym = str(entry.get("symbol") or entry.get("label") or "").upper()
                    if self.lookup_sym:
                        self.action_tab("lookup")
                else:
                    self._rates_search_query = str(entry.get("label") or "").strip()
                    self.action_tab("kalimati")
                    try:
                        self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                    except Exception:
                        pass
            return
        if event.data_table.id == "dt-watchlist-rates":
            idx = event.cursor_row
            if 0 <= idx < len(getattr(self, "_watchlist_rates_rows", [])):
                item = self._watchlist_rates_rows[idx]
                self._rates_search_query = str(item.get("label") or "").strip()
                self.action_tab("kalimati")
                try:
                    self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                except Exception:
                    pass
            return
        if event.data_table.id == "dt-watchlist-commodities":
            idx = event.cursor_row
            if 0 <= idx < len(getattr(self, "_watchlist_commodity_rows", [])):
                item = self._watchlist_commodity_rows[idx]
                self._rates_search_query = str(item.get("label") or "").strip()
                self.action_tab("kalimati")
                try:
                    self.query_one("#kalimati-search-input", Input).value = self._rates_search_query
                except Exception:
                    pass
            return
        if event.data_table.id == "dt-account-list":
            idx = event.cursor_row
            accounts = list(getattr(self, "_paper_accounts", []) or [])
            if 0 <= idx < len(accounts):
                selected = accounts[idx]
                self._selected_account_id = str(selected.get("id") or self._current_account_id)
                try:
                    self.query_one("#profile-inp-account-name", Input).value = str(selected.get("name") or "")
                except Exception:
                    pass
                try:
                    snap = self._account_snapshot(selected)
                    self.query_one("#profile-inp-target-nav", Input).value = f"{float(snap['nav']):,.2f}".replace(",", "")
                except Exception:
                    pass
                self._populate_account_list()
                self._populate_strategies_tab()
                selected_name = str(selected.get("name") or self._selected_account_id)
                self._set_status(f"Selected {selected_name} | press ACTIVATE to switch")
            return
        if event.data_table.id == "dt-strategy-list":
            idx = event.cursor_row
            strategies = list(getattr(self, "_strategies", []) or [])
            if 0 <= idx < len(strategies):
                selected = strategies[idx]
                self._selected_strategy_id = str(selected.get("id") or self._selected_strategy_id)
                self._set_strategy_form_from_payload(selected)
                self._populate_strategies_tab()
                self._set_status(f"Selected strategy {selected.get('name')}")
            return
        # dt-news replaced by OptionList — handled in on_option_list_option_highlighted

        # Screener: Enter → lookup
        if event.data_table.id == "dt-screener":
            try:
                row_data = event.data_table.get_row_at(event.cursor_row)
                sym = str(row_data[0]).strip()
                if sym and sym != "—":
                    self.lookup_sym = sym
                    self.action_tab("lookup")
            except Exception:
                pass
            return
        # Agent verdicts
        if event.data_table.id != "dt-agent-verdicts":
            return
        a = getattr(self, '_agent_analysis', {})
        stocks = a.get("stocks", [])
        idx = event.cursor_row
        self._show_agent_focus_row(idx)
        if 0 <= idx < len(stocks):
            symbol = str(stocks[idx].get("symbol") or "").strip().upper()
            if symbol:
                self.lookup_sym = symbol

        # Market view
        mv = self.query_one("#agent-market-view", Static)
        market_view = a.get("market_view", "")
        trade_reason = a.get("trade_today_reason", "")
        risks = a.get("risks", [])
        parts = []
        if market_view:
            parts.append(f"[bold #bb88ff]VIEW:[/] [{WHITE}]{market_view}[/]")
        if trade_reason:
            parts.append(f"[bold #bb88ff]TRADE:[/] [{YELLOW}]{trade_reason}[/]")
        if risks:
            parts.append(f"[bold {LOSS_HI}]RISKS:[/] [{LABEL}]{' | '.join(risks[:3])}[/]")
        if parts:
            mv.update(Text.from_markup("   ".join(parts)))
        else:
            mv.update(Text("Press A to run agent analysis, or type a question below", style=LABEL))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "dt-strategy-list":
            idx = event.cursor_row
            strategies = list(getattr(self, "_strategies", []) or [])
            if 0 <= idx < len(strategies):
                selected = strategies[idx]
                self._selected_strategy_id = str(selected.get("id") or self._selected_strategy_id)
                # Selecting from the list exits new-strategy mode
                self._new_strategy_mode = False
                self._set_strategy_form_from_payload(selected)
            return
        if event.data_table.id != "dt-agent-verdicts":
            return
        self._show_agent_focus_row(event.cursor_row)

    def _append_chat(
        self,
        role: str,
        message: str,
        color: str,
        *,
        ts: float | None = None,
        provider: str | None = None,
    ):
        scroll = self.query_one("#agent-chat-scroll", VerticalScroll)
        self._detach_agent_typing_widget()
        css_class = "chat-user" if role == "YOU" else "chat-agent"
        label = "YOU" if role == "YOU" else "AGENT" if role == "AGENT" else role.upper()
        ts_text = _format_nst_hm(ts)
        flat_message = " ".join(part.strip() for part in str(message or "").splitlines() if part.strip())
        if not flat_message:
            flat_message = "—"
        prefix = f"[{label}]"
        msg_text = Text.assemble(
            (prefix, f"bold {color}"),
            (" ", WHITE),
            (f"{ts_text} " if ts_text else "", LABEL),
            (": ", LABEL),
            (flat_message, WHITE),
        )
        widget = Static(msg_text, classes=f"chat-msg {css_class}")
        scroll.mount(widget)
        scroll.mount(Static("", classes="chat-gap"))
        scroll.scroll_end(animate=False)
        if bool(getattr(self, "_agent_typing_visible", False)):
            self._animate_agent_typing()

    def _handle_agent_chat_command(self, raw: str) -> bool:
        command = str(raw or "").strip().lower()
        if command in {"/history", "/archive", "/archived"}:
            self._agent_show_archived = True
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent archive loaded")
            return True
        if command in {"/recent", "/hide", "/collapse"}:
            self._agent_show_archived = False
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent archive hidden")
            return True
        if command == "/clear":
            self._agent_show_archived = False
            self._agent_visible_since = time.time()
            self._load_agent_runtime_state()
            self._render_agent_chat_history()
            self._set_status("Agent chat screen cleared")
            return True
        if command == "/stop":
            if self._stop_active_agent_chat():
                return True
            self._set_status("No active agent response to stop")
            return True
        if command == "/help":
            self._set_status("Agent chat commands: /history, /recent, /clear, /stop")
            return True
        return False

    def _build_agent_auto_order_spec(self, analysis: dict) -> Optional[dict]:
        if str(getattr(self, "trade_mode", "paper")) != "paper":
            return None
        if not bool((analysis or {}).get("trade_today")):
            return None
        stocks = sorted(
            [dict(item) for item in list((analysis or {}).get("stocks") or []) if bool(item.get("auto_entry_candidate"))],
            key=lambda item: (
                -float(item.get("signal_score") or 0.0),
                -float(item.get("conviction") or 0.0),
            ),
        )
        if not stocks:
            return None
        positions = list((self._stats or {}).get("positions") or [])
        held_symbols = {str(pos.get("sym") or "").upper() for pos in positions}
        open_buy_symbols = {
            str(order.get("symbol") or "").upper()
            for order in list(getattr(self, "_paper_orders", []) or [])
            if str(order.get("status") or "").upper() == "OPEN" and str(order.get("action") or "").upper() == "BUY"
        }
        max_positions = int(
            LONG_TERM_CONFIG.get("regime_max_positions", {}).get(
                str((analysis or {}).get("regime") or "").lower(),
                LONG_TERM_CONFIG.get("max_positions", 5),
            )
        )
        if len(held_symbols | open_buy_symbols) >= max_positions:
            return None
        prices = self.md.ltps() if hasattr(self, "md") else {}
        cash = float((self._stats or {}).get("cash") or 0.0)
        nav = float((self._stats or {}).get("nav") or cash or INITIAL_CAPITAL)
        per_position_budget = min(nav / max_positions, cash * 0.95) if max_positions > 0 else 0.0
        for stock in stocks:
            symbol = str(stock.get("symbol") or "").upper()
            if symbol in held_symbols or symbol in open_buy_symbols:
                continue
            price = float(prices.get(symbol) or stock.get("last_price") or 0.0)
            if price <= 0:
                continue
            quantity = int(per_position_budget / price)
            if quantity < 10:
                continue
            order_key = f"{analysis.get('context_date') or ''}:{symbol}:{int(float(analysis.get('timestamp') or 0))}"
            if order_key == self._last_agent_auto_order_key:
                return None
            return {
                "symbol": symbol,
                "quantity": quantity,
                "price": round(price, 2),
                "order_key": order_key,
                "signal_score": float(stock.get("signal_score") or 0.0),
                "conviction": float(stock.get("conviction") or 0.0),
            }
        return None

    def _maybe_submit_agent_super_signal(self, analysis: dict) -> None:
        spec = self._build_agent_auto_order_spec(analysis or {})
        if not spec:
            return
        result = build_tui_control_plane(self).submit_paper_order(
            action="buy",
            symbol=spec["symbol"],
            quantity=int(spec["quantity"]),
            limit_price=float(spec["price"]),
            thesis="agent_super_signal",
            confidence=float(spec["conviction"]),
            source_signals=["agent_super_signal"],
        )
        if not result.ok:
            return
        self._last_agent_auto_order_key = str(spec["order_key"])
        self._append_chat_note(
            f"AUTO BUY queued {spec['symbol']} x{spec['quantity']} @ {spec['price']:.2f} "
            f"(score {spec['signal_score']:.2f}, conv {spec['conviction']:.0%})"
        )
        self._set_status(f"Agent auto-buy queued {spec['symbol']} x{spec['quantity']} @ {spec['price']:.2f}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        bid = event.button.id
        if bid == "hedge-toggle-btn":
            self._hedge_enabled = not self._hedge_enabled
            try:
                btn = self.query_one("#hedge-toggle-btn", Button)
                if self._hedge_enabled:
                    btn.label = "● HEDGE ON"
                    btn.remove_class("hedge-btn-off")
                    btn.add_class("hedge-btn-on")
                else:
                    btn.label = "○ HEDGE OFF"
                    btn.remove_class("hedge-btn-on")
                    btn.add_class("hedge-btn-off")
            except Exception:
                pass
            # Propagate to all running engines
            for _eng in list(getattr(self, "_account_engines", {}).values()):
                try:
                    _eng.set_hedge_enabled(self._hedge_enabled)
                except Exception:
                    pass
            self._render_hedge_panel()
            return
        if bid in ("bf-filter-all", "bf-filter-circ", "bf-filter-pump", "bf-filter-smart"):
            self._broker_floor_filter = {"bf-filter-all": "all", "bf-filter-circ": "circ",
                                         "bf-filter-pump": "pump", "bf-filter-smart": "smart"}[bid]
            for btn_id in ("bf-filter-all", "bf-filter-circ", "bf-filter-pump", "bf-filter-smart"):
                try:
                    btn = self.query_one(f"#{btn_id}", Button)
                    if btn_id == bid:
                        btn.add_class("bf-active")
                    else:
                        btn.remove_class("bf-active")
                except Exception:
                    pass
            self._apply_broker_floor_filter()
            return
        if bid == "news-search-clear":
            self._news_search_query = ""
            self._vector_search_results = []
            try:
                self.query_one("#news-search-input", Input).value = ""
            except Exception:
                pass
            self._populate_news()
            return
        if bid == "kalimati-search-clear":
            self._rates_search_query = ""
            try:
                self.query_one("#kalimati-search-input", Input).value = ""
            except Exception:
                pass
            self._populate_kalimati()
            return
        if bid == "order-btn-buy":
            self._order_action = "BUY"
            self._refresh_order_action_buttons()
            self._set_status("Order action: BUY")
        elif bid == "order-btn-sell":
            self._order_action = "SELL"
            self._refresh_order_action_buttons()
            self._set_status("Order action: SELL")
        elif bid == "order-btn-submit":
            self._on_order_submit()
        elif bid == "order-btn-cancel-all":
            if self.trade_mode == "paper":
                msg = self._cancel_all_paper_orders()
                self._set_status(msg)
            else:
                self._set_status("Cancel all is only available for paper orders")
        elif bid == "profile-btn-browse-seed":
            self._browse_seed_file()
        elif bid == "profile-btn-create-account":
            try:
                self._set_status(self._account_create_from_form())
            except Exception as exc:
                self._set_status(f"Account create failed: {exc}")
        elif bid == "profile-btn-activate-account":
            try:
                self._set_status(self._account_activate_selected())
            except Exception as exc:
                self._set_status(f"Account activate failed: {exc}")
        elif bid == "profile-btn-delete-account":
            try:
                self._set_status(self._delete_paper_account(self._selected_account_id))
            except Exception as exc:
                self._set_status(f"Account delete failed: {exc}")
        elif bid == "profile-btn-set-nav":
            try:
                self._set_status(self._account_set_nav_from_form())
            except Exception as exc:
                self._set_status(f"NAV update failed: {exc}")
        elif bid == "profile-btn-sync-watchlist":
            try:
                self._set_status(self._account_sync_watchlist())
            except Exception as exc:
                self._set_status(f"Watchlist sync failed: {exc}")
        elif bid == "profile-btn-save-account":
            try:
                self._set_status(self._save_current_account_snapshot())
            except Exception as exc:
                self._set_status(f"Snapshot save failed: {exc}")
        elif bid.startswith(_SIG_ID_PREFIX):
            sig_type = _sig_type_from_id(bid)
            active = getattr(self, "_active_signals", set())
            if sig_type in active:
                active.discard(sig_type)
            else:
                active.add(sig_type)
            self._active_signals = active
            self._sync_signal_buttons()
            return
        elif bid == "strategy-btn-new":
            self._new_strategy_mode = True
            self._active_signals = set()
            self._sync_signal_buttons()
            # Pre-fill sensible defaults — user only needs to enter name, description, and signals
            _new_defaults = {
                "strategy-inp-name": "",
                "strategy-inp-description": "",
                "strategy-inp-holding-days": "40",
                "strategy-inp-rebalance": "5",
                "strategy-inp-max-positions": "5",
                "strategy-inp-stop-loss": "0.08",
                "strategy-inp-trailing-stop": "0.15",
                "strategy-inp-sector-limit": "0.35",
            }
            for _wid, _val in _new_defaults.items():
                try:
                    self.query_one(f"#{_wid}", Input).value = _val
                except Exception:
                    pass
            try:
                self.query_one("#strategy-inp-name", Input).focus()
            except Exception:
                pass
            self._set_status("New strategy — enter name, pick signals, then SAVE  (parameters pre-filled with defaults)")
            return
        elif bid == "strategy-btn-save":
            try:
                self._set_status(self._save_strategy_from_form())
            except Exception as exc:
                self._set_status(f"Strategy save failed: {exc}")
        elif bid == "strategy-btn-delete":
            try:
                self._set_status(self._delete_custom_strategy())
            except Exception as exc:
                self._set_status(f"Strategy delete failed: {exc}")
        elif bid == "strategy-btn-assign-current":
            try:
                self._set_status(self._assign_strategy_to_account(self._selected_strategy_id, self._current_account_id))
            except Exception as exc:
                self._set_status(f"Strategy assign failed: {exc}")
        elif bid == "strategy-btn-assign-selected":
            try:
                self._set_status(self._assign_strategy_to_account(self._selected_strategy_id, self._selected_account_id))
            except Exception as exc:
                self._set_status(f"Strategy assign failed: {exc}")
        elif bid == "strategy-btn-backtest":
            try:
                self._set_status(self._start_strategy_backtest())
            except Exception as exc:
                self._set_status(f"Strategy backtest failed: {exc}")
        elif bid == "strategy-btn-chart":
            result = getattr(self, "_strategy_backtest_result", None)
            if not result:
                self._set_status("Run a backtest first — no result to chart")
            else:
                try:
                    from validation.quick_chart import generate_quick_chart
                    from backend.quant_pro.database import get_db_path as _get_db_path
                    selected = self._selected_strategy_payload()
                    name   = str((selected or {}).get("name") or "Strategy")
                    start  = str(self.query_one("#strategy-inp-backtest-start", Input).value or "").strip()
                    end    = str(self.query_one("#strategy-inp-backtest-end", Input).value or "").strip()
                    path   = generate_quick_chart(
                        result,
                        strategy_name=name,
                        start_date=start or "2025-01-01",
                        end_date=end or "2026-04-11",
                        db_path=str(_get_db_path()),
                        auto_open=True,
                    )
                    self._set_status(f"Chart saved → {path}" if path else "Chart generation failed")
                except Exception as exc:
                    self._set_status(f"Chart failed: {exc}")
        elif bid == "order-btn-cancel":
            # Cancel selected order from daily order book
            try:
                dt = self.query_one("#dt-orders-daily", DataTable)
                row_key = dt.cursor_row
                if row_key is not None:
                    row_data = dt.get_row_at(row_key)
                    order_id = str(row_data[0]).strip()
                    if order_id and order_id != "—":
                        msg = self._cancel_paper_order(order_id)
                        self._set_status(msg)
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle agent chat input + order form + news search."""
        if event.input.id == "news-search-input":
            q = event.value.strip()
            self._news_search_query = q
            if q:
                self._do_vector_search(q)
            else:
                self._vector_search_results = []
                self._populate_news()
            return
        if event.input.id == "profile-inp-portfolio":
            self._persist_profile_paths(portfolio_path=event.value.strip())
            self._set_status("Portfolio seed path updated")
            return
        if event.input.id == "profile-inp-account-name":
            self._set_status("Account name updated")
            return
        if event.input.id == "profile-inp-target-nav":
            try:
                self._set_status(self._seed_manual_nav(event.value))
                self._populate_portfolio_and_risk()
            except Exception as exc:
                self._set_status(f"NAV update failed: {exc}")
            return
        if event.input.id in ("order-inp-price", "order-inp-slippage"):
            self._on_order_submit()
            return
        if event.input.id in ("order-inp-symbol", "order-inp-qty"):
            return
        if event.input.id != "agent-input":
            return
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        if question.upper() == "A":
            self._run_agent_analysis()
        elif question.startswith("/"):
            if not self._handle_agent_chat_command(question):
                self._set_status(f"Unknown agent command: {question}")
        else:
            self._agent_ask_async(question)

    def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id or ""
        if input_id == "kalimati-search-input":
            self._rates_search_query = event.value.strip()
            self._populate_kalimati()
            return
        if input_id == "signal-min-score":
            try:
                self._signal_min_score = float(event.value.strip() or "0")
            except ValueError:
                self._signal_min_score = 0.0
            # Invalidate cache so next load respects new threshold
            self._signals_table_cache_key = ""
            self._signals_table_cache_payload = None
            return

    def on_key(self, event: events.Key) -> None:
        if str(self.active_tab or "") != "account":
            return
        if isinstance(self.focused, Input):
            return
        key = str(event.key or "").lower()
        try:
            if key == "n":
                self._set_status(self._account_create_from_form())
                event.stop()
            elif key == "v":
                self._set_status(self._account_set_nav_from_form())
                event.stop()
            elif key == "h":
                self._set_status(self._toggle_account_help())
                event.stop()
        except Exception as exc:
            event.stop()
            if key == "n":
                self._set_status(f"Account create failed: {exc}")
            elif key == "v":
                self._set_status(f"NAV update failed: {exc}")
            elif key == "h":
                self._set_status(f"Help failed: {exc}")

    def on_option_list_option_highlighted(self, event) -> None:
        if getattr(getattr(event, "option_list", None), "id", "") != "news-list":
            return
        highlighted = getattr(event.option_list, "highlighted", None)
        if highlighted is None:
            return
        stories = getattr(self, "_news_visible_stories", [])
        if 0 <= highlighted < len(stories):
            self._news_selected_index = highlighted

    def on_option_list_option_selected(self, event) -> None:
        if getattr(getattr(event, "option_list", None), "id", "") != "news-list":
            return
        self.on_option_list_option_highlighted(event)

    # ── Header / Index / Status bars ──────────────────────────────────────────

    def _display_live_badge(self) -> bool:
        return False

    def _display_mode_label(self) -> str:
        return "PAPER AUTO"

    def _display_nav_mode_tag(self) -> str:
        if self._trading_engine:
            phase = self._trading_engine.phase.upper().replace("_", " ")
            return f"[bold #00cfff]AUTO[/] [dim]{phase}[/]"
        return "[bold #00ff7f]PAPER[/]"

    def _update_header(self):
        parts = []
        # Mode indicator
        parts.append("[bold #111820 on #79ffb3] PAPER AUTO [/]")
        for name, key in TAB_NAMES.items():
            label = TAB_LABELS.get(name, name.upper())
            if name == self.active_tab:
                parts.append(f"[bold #111820 on #f2b94b] {key}:{label} [/]")
            else:
                parts.append(f"[#7d8b99] {key}:{label} [/]")
        if self.active_tab == "lookup" and self.lookup_sym:
            tf = self.lookup_tf
            tf_parts = []
            for key, label in [("D", "Daily"), ("W", "Weekly"), ("M", "Monthly"), ("Y", "Yearly"), ("I", "Intraday")]:
                if key == tf:
                    tf_parts.append(f"[bold #f2b94b]{key}={label}[/]")
                else:
                    tf_parts.append(f"[#708091]{key}={label}[/]")
            parts.append("  [#3d4f5e]│[/]  " + "  [#3d4f5e]·[/]  ".join(tf_parts) + "  [#3d4f5e]│[/]  [#7a9ab0]l[/][#4a6070] lookup[/]  [#3d4f5e]·[/]  [#7a9ab0]q[/][#4a6070] quit[/]")
        else:
            parts.append("  [#3d4f5e]│[/]  [#7a9ab0]/[/][#4a6070] search[/]  [#3d4f5e]·[/]  [#7a9ab0]+[/][#4a6070] watch[/]  [#3d4f5e]·[/]  [#7a9ab0]r[/][#4a6070] refresh[/]  [#3d4f5e]·[/]  [#7a9ab0]q[/][#4a6070] quit[/]")
        self.query_one("#header-bar", Static).update(Text.from_markup(" ".join(parts)))

    def _update_index(self):
        parts = []
        # NEPSE index — prefer live TMS WS value when available.
        live_nepse = None
        tms_src = getattr(self.md, "tms", None)
        if tms_src is not None:
            try:
                if tms_src.is_live():
                    idx = (self.md.tms_indices or {}).get("NEPSE")
                    if idx is None:
                        idx = tms_src.indices().get("NEPSE")
                    if idx is not None:
                        live_nepse = idx
            except Exception:
                live_nepse = None

        if live_nepse is not None:
            ni = float(getattr(live_nepse, "value", 0.0))
            chg = float(getattr(live_nepse, "pct_change", 0.0))
            sign = "+" if chg >= 0 else ""
            cc = GAIN_HI if chg >= 0 else LOSS_HI
            parts.append(
                f"[bold #8da1b5]NEPSE[/] [bold {WHITE}]{ni:,.2f}[/] "
                f"[{cc}]{sign}{chg:.2f}%[/] [dim #6a8899]live[/]"
            )
        elif len(self.md.nepse) >= 2:
            ni = self.md.nepse.iloc[0]["close"]
            np_ = self.md.nepse.iloc[1]["close"]
            chg = (ni - np_) / np_ * 100
            sign = "+" if chg >= 0 else ""
            cc = GAIN_HI if chg >= 0 else LOSS_HI
            parts.append(f"[bold #8da1b5]NEPSE[/] [bold {WHITE}]{ni:,.1f}[/] [{cc}]{sign}{chg:.2f}%[/]")
        else:
            parts.append("[#8da1b5]NEPSE N/A[/]")
        # Breadth
        parts.append(f"[bold {GAIN_HI}]▲{self.md.adv}[/] [bold {LOSS_HI}]▼{self.md.dec}[/] [#888888]={self.md.unch}[/]")
        # Regime
        regime = getattr(self, '_regime', 'unknown')
        rc = {
            "bull": f"bold {GAIN_HI}", "neutral": f"bold {YELLOW}",
            "bear": f"bold {LOSS_HI}"
        }.get(regime, LABEL)
        parts.append(f"[{rc}]REGIME {regime.upper()}[/]")
        # Session
        parts.append(f"[#8da1b5]SESSION[/] [bold {WHITE}]{self.md.latest}[/]")
        # Quotes timestamp
        ts_q = ""
        if not self.md.quotes.empty and "ts" in self.md.quotes.columns:
            ts_q = self.md.quotes["ts"].iloc[0][:16]
        if ts_q:
            parts.append(f"[#708091]QUOTES {ts_q}[/]")
        parts.append(f"[#8da1b5]LOCAL[/] [bold {WHITE}]{self.md.ts.strftime('%H:%M:%S')}[/]")
        self.query_one("#index-bar", Static).update(
            Text.from_markup("   │   ".join(parts)))

    def _set_status(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        compact = " ".join(str(msg).split())
        if len(compact) > 220:
            compact = compact[:217] + "..."
        lowered = compact.lower()
        body_style = "#95a4b5"
        if lowered.startswith("rejected:") or lowered.startswith("order cancelled:") or "same-day rule" in lowered:
            body_style = LOSS_HI
        self.query_one("#status-bar", Static).update(
            Text.from_markup(
                f"[bold #f2b94b]{stamp}[/] [#516273]::[/] [{body_style}]{_escape_markup(compact)}[/]"
            )
        )

    @staticmethod
    def _summarize_tms_watchlist_error(exc: Exception, action: str = "sync") -> str:
        detail = " ".join(str(exc).split())
        lowered = detail.lower()
        if "profile is already in use" in lowered or "processsingleton" in lowered or "singletonlock" in lowered:
            return f"TMS watchlist {action} paused: browser profile already in use"
        if "login required" in lowered:
            return f"TMS watchlist {action} paused: login required"
        if "member market watch page not ready" in lowered:
            return f"TMS watchlist {action} paused: member market watch not ready"
        if len(detail) > 140:
            detail = detail[:137] + "..."
        return f"TMS watchlist {action} failed: {detail}"

    # ── Market tab ────────────────────────────────────────────────────────────

    def _populate_market(self):
        self.query_one("#p-gainers", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), _vol_text(r["volume"])]
             for _, r in self.md.gainers.iterrows()])
        self.query_one("#p-losers", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), _vol_text(r["volume"])]
             for _, r in self.md.losers.iterrows()])
        self.query_one("#p-volume", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("CHG%", "chg"), ("VOL", "vol")],
            [[_sym_text(r["symbol"]), _price_text(r["close"]),
              _chg_text(r["chg"]), Text(_vol(r["volume"]), style=CYAN)]
             for _, r in self.md.vol_top.iterrows()])
        # 52-week
        rows_52 = []
        for _, r in self.md.near_hi.iterrows():
            rows_52.append([_sym_text(r["symbol"]), _price_text(r["close"]),
                            Text(f"{r['h']:.1f}", style=WHITE),
                            Text("▲ NEAR HIGH", style=f"bold {GAIN_HI}")])
        for _, r in self.md.near_lo.iterrows():
            rows_52.append([_sym_text(r["symbol"]), _price_text(r["close"]),
                            Text(f"{r['l']:.1f}", style=WHITE),
                            Text("▼ NEAR LOW", style=f"bold {LOSS_HI}")])
        if not rows_52:
            rows_52.append([_dim_text("—"), _dim_text("None today"), _dim_text(""), _dim_text("")])
        self.query_one("#p-52wk", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("PRICE", "price"), ("REF", "ref"), ("SIGNAL", "sig")], rows_52)
        # Quotes
        quote_df = self.md.quotes if isinstance(self.md.quotes, pd.DataFrame) else pd.DataFrame()
        if not quote_df.empty and {"symbol", "ltp", "pc", "vol"}.issubset(quote_df.columns):
            qq = quote_df[quote_df["vol"] > 0].nlargest(60, "vol")
        else:
            qq = pd.DataFrame(columns=["symbol", "ltp", "pc", "vol"])
        ts_q = ""
        if not quote_df.empty and "ts" in quote_df.columns:
            ts_q = str(quote_df["ts"].iloc[0])[:16]
        self.query_one("#p-quotes", MarketPanel).update_title(f"5) LIVE QUOTES  {ts_q}")
        quote_rows = [
            [_sym_text(str(r["symbol"])), _price_text(r["ltp"]),
             _chg_text(r["pc"]), _vol_text(r["vol"])]
            for _, r in qq.iterrows()
        ]
        if not quote_rows:
            quote_rows = [[_dim_text("—"), _dim_text("No live quotes"), _dim_text(""), _dim_text("")]]
        self.query_one("#p-quotes", MarketPanel).set_data(
            [("SYMBOL", "sym"), ("LTP", "ltp"), ("CHG%", "chg"), ("VOL", "vol")],
            quote_rows)

    # ── Portfolio + Risk tabs ─────────────────────────────────────────────────

    def _populate_portfolio_and_risk(self):
        if self._trading_engine or self.trade_mode == "paper":
            # Always use account-specific paper_portfolio.csv as the authoritative view.
            # This captures both manual trades and any engine trades persisted there.
            # The engine's tui_paper_portfolio.csv is used only as a fallback.
            _acc_dir = _account_dir(self._current_account_id)
            _paper_port = _acc_dir / "paper_portfolio.csv"
            if _paper_port.exists():
                self._stats = _compute_account_portfolio_stats(self.md, _acc_dir)
            elif self._trading_engine:
                self._stats = self._trading_engine.get_portfolio_stats()
            else:
                self._stats = _compute_portfolio_stats(self.md)
        elif self.trade_mode == "live":
            # Fetch TMS bundle once, cache for _populate_trades_full
            if self.tms_service:
                try:
                    self._set_status("TMS monitor  |  Scraping TMS pages...")
                    live_bundle = self.tms_service.executor.fetch_monitor_bundle()
                    self._tms_bundle = _merge_tms_bundle_with_cache(live_bundle)
                    h = self._tms_bundle.get("holdings", {})
                    items = h.get("items", []) if isinstance(h, dict) else h
                    health = self._tms_bundle.get("health", {})
                    login_req = _tms_health_flag(health, "login_required")
                    status_note = "using cached snapshots" if login_req else "live snapshots"
                    self._set_status(
                        f"TMS monitor  |  Bundle: {len(items)} holdings, "
                        f"login_required={login_req}  |  {status_note}"
                    )
                except Exception as e:
                    self._tms_bundle = _load_cached_tms_bundle()
                    cached_items = ((self._tms_bundle.get("holdings") or {}).get("items") or [])
                    self._set_status(
                        f"TMS fetch failed: {e}  |  cached holdings={len(cached_items)}"
                    )
            self._stats = self._compute_tms_portfolio_stats()
        else:
            self._stats = _compute_portfolio_stats(self.md)
        s = self._stats
        self._populate_portfolio_tab(s)
        self._populate_risk_tab(s)

    def _compute_tms_portfolio_stats(self) -> dict:
        """Compute portfolio stats from TMS broker data."""
        if not self.tms_service:
            # TMS not ready yet — show empty portfolio with message
            return {
                "positions": [], "total_cost": 0, "total_value": 0,
                "cash": 0, "nav": 0, "total_return": 0,
                "day_pnl": 0, "day_ret": 0,
                "realized": 0, "unrealized": 0,
                "max_dd": 0, "dd_date": "", "peak_nav": 0,
                "nepse_ret": 0, "alpha": 0,
                "n_positions": 0, "sector_exposure": {},
                "top3_conc": 0, "winners": [], "losers": [],
                "age_0_5": 0, "age_6_15": 0, "age_16": 0,
                "trade_log": pd.DataFrame(), "nav_log": pd.DataFrame(),
            }
        try:
            bundle = getattr(self, '_tms_bundle', None)
            if not bundle:
                bundle = _merge_tms_bundle_with_cache(
                    self.tms_service.executor.fetch_monitor_bundle()
                )
                self._tms_bundle = bundle

            # Check if login is required (session expired)
            health = bundle.get("health", {})
            if _tms_health_flag(health, "login_required"):
                self._set_status("TMS session expired — showing last cached broker snapshot")

            holdings_data = bundle.get("holdings", {})
            holdings = holdings_data.get("items", []) if isinstance(holdings_data, dict) else holdings_data
            funds = bundle.get("funds", {})
            ltps = self.md.ltps()
            quote_map = {}
            if not self.md.quotes.empty:
                quote_map = {
                    str(r.symbol): {
                        "ltp": float(getattr(r, "ltp", 0) or 0),
                        "prev_close": float(getattr(r, "prev_close", 0) or 0),
                        "pc": float(getattr(r, "pc", 0) or 0),
                    }
                    for r in self.md.quotes.itertuples()
                }

            positions = []
            total_cost = total_value = 0.0
            total_prev_value = 0.0
            sector_exposure = {}

            for h in holdings:
                sym = str(h.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                qty = int(h.get("tms_balance") or h.get("cds_total_balance") or h.get("quantity", 0))
                if qty <= 0:
                    continue
                # TMS doesn't expose avg cost; use close_price as proxy for entry
                entry = float(h.get("close_price") or h.get("avg_cost", 0))
                cur = ltps.get(sym) or float(h.get("ltp") or h.get("close_price") or entry)
                quote = quote_map.get(sym, {})
                prev_close = float(quote.get("prev_close") or h.get("close_price") or 0) or cur
                cost = entry * qty
                val = cur * qty
                pnl = val - cost
                ret = pnl / cost * 100 if cost else 0
                day_pnl = (cur - prev_close) * qty if prev_close > 0 else 0.0
                day_ret = ((cur - prev_close) / prev_close * 100) if prev_close > 0 else float(quote.get("pc") or 0)
                total_cost += cost
                total_value += val
                total_prev_value += prev_close * qty

                try:
                    from backend.backtesting.simple_backtest import get_symbol_sector
                    sec = get_symbol_sector(sym) or "Other"
                except Exception:
                    sec = "Other"
                sector_exposure[sec] = sector_exposure.get(sec, 0) + val

                positions.append({
                    "sym": sym, "qty": qty, "entry": entry, "cur": cur,
                    "cost": cost, "val": val, "pnl": pnl, "ret": ret,
                    "prev_close": prev_close, "day_pnl": day_pnl, "day_ret": day_ret,
                    "signal": _signal_label("TMS"), "date": "", "days": 0, "sector": sec,
                })

            cash = float(
                funds.get("collateral_available")
                or funds.get("available_trading_limit")
                or funds.get("collateral_total")
                or funds.get("available_cash")
                or funds.get("cash_collateral_amount")
                or 0
            )
            nav = cash + total_value
            prev_nav = cash + total_prev_value
            day_pnl_total = nav - prev_nav
            day_ret_total = (day_pnl_total / prev_nav * 100) if prev_nav > 0 else 0.0
            total_return = (nav - total_cost - cash) / (total_cost + cash) * 100 if (total_cost + cash) > 0 else 0

            positions.sort(key=lambda x: x["val"], reverse=True)
            top3_conc = sum(p["val"] for p in positions[:3]) / total_value * 100 if total_value > 0 else 0
            winners = [p for p in positions if p["pnl"] > 0]
            losers = [p for p in positions if p["pnl"] < 0]

            nepse_ret = 0.0
            if len(self.md.nepse) >= 2:
                nepse_ret = (self.md.nepse.iloc[0]["close"] - self.md.nepse.iloc[1]["close"]) / self.md.nepse.iloc[1]["close"] * 100

            return {
                "positions": positions,
                "total_cost": total_cost, "total_value": total_value,
                "cash": cash, "nav": nav, "total_return": total_return,
                "day_pnl": day_pnl_total, "day_ret": day_ret_total,
                "realized": 0, "unrealized": total_value - total_cost,
                "max_dd": 0, "dd_date": "", "peak_nav": nav,
                "nepse_ret": nepse_ret, "alpha": total_return - nepse_ret,
                "n_positions": len(positions),
                "sector_exposure": sector_exposure,
                "top3_conc": top3_conc,
                "winners": winners, "losers": losers,
                "age_0_5": 0, "age_6_15": 0, "age_16": len(positions),
                "trade_log": pd.DataFrame(), "nav_log": pd.DataFrame(),
            }
        except Exception as e:
            self._set_status(f"TMS portfolio fetch failed: {e}")
            return _compute_portfolio_stats(self.md)

    def _load_profile_inputs(self) -> None:
        profile = _load_profile_config()
        try:
            self.query_one("#profile-inp-portfolio", Input).value = str(profile.get("portfolio_path") or "")
        except Exception:
            pass
        try:
            target_nav = profile.get("target_nav")
            self.query_one("#profile-inp-target-nav", Input).value = (
                f"{float(target_nav):,.2f}".replace(",", "") if isinstance(target_nav, (int, float)) else ""
            )
        except Exception:
            pass
        try:
            accounts = list(getattr(self, "_paper_accounts", []) or [])
            next_name = f"Account {len(accounts) + 1}"
            self.query_one("#profile-inp-account-name", Input).value = next_name
        except Exception:
            pass

    def _strategy_account_binding(self, account_id: str) -> Optional[dict]:
        aid = str(account_id or "").strip()
        for account in list(getattr(self, "_paper_accounts", []) or []):
            if str(account.get("id") or "") == aid:
                return account
        return None

    def _strategy_name_for_account(self, account_id: str) -> str:
        account = self._strategy_account_binding(account_id)
        strategy_id = str((account or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(account_id))
        return strategy_registry.strategy_name(strategy_id)

    def _strategy_display_name(self, strategy_id: str, fallback_name: str = "") -> str:
        sid = str(strategy_id or "").strip()
        return str(STRATEGY_DISPLAY_NAMES.get(sid) or fallback_name or sid)

    def _reload_account_bindings_from_disk(self) -> None:
        registry = _load_accounts_registry()
        accounts = strategy_registry.ensure_account_strategy_ids(list(registry.get("accounts") or []))
        if not accounts:
            return
        self._paper_accounts = accounts
        known_ids = {str(account.get("id") or "") for account in accounts}
        profile = _load_profile_config()
        current_account_id = str(
            profile.get("current_account_id")
            or getattr(self, "_current_account_id", "account_1")
            or "account_1"
        ).strip()
        if current_account_id not in known_ids:
            current_account_id = str(accounts[0].get("id") or "account_1")
        self._current_account_id = current_account_id
        selected_account_id = str(getattr(self, "_selected_account_id", current_account_id) or current_account_id).strip()
        if selected_account_id not in known_ids:
            selected_account_id = current_account_id
        self._selected_account_id = selected_account_id
        binding = self._strategy_account_binding(current_account_id) or {}
        self._selected_strategy_id = str(
            binding.get("strategy_id")
            or getattr(self, "_selected_strategy_id", "")
            or strategy_registry.default_strategy_for_account(current_account_id)
        )
        self._sync_agent_account_context_env()

    def _sync_signal_buttons(self) -> None:
        """Refresh signal picker button styles and active-display label."""
        active = getattr(self, "_active_signals", set())
        for _, sig_type in _SIGNAL_DEFS:
            try:
                btn = self.query_one(f"#{_sig_btn_id(sig_type)}", Button)
                if sig_type in active:
                    btn.add_class("signal-active")
                else:
                    btn.remove_class("signal-active")
            except Exception:
                pass
        try:
            label = ", ".join(s for _, s in _SIGNAL_DEFS if s in active) or "No signals selected"
            self.query_one("#signal-picker-active", Static).update(
                f"  [#555555]Active:[/]  [bold #d8e4f2]{label}[/]"
            )
        except Exception:
            pass

    def _active_strategy_payload(self) -> Optional[dict]:
        account_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        account = self._strategy_account_binding(account_id)
        strategy_id = str((account or {}).get("strategy_id") or strategy_registry.default_strategy_for_account(account_id))
        return strategy_registry.load_strategy(strategy_id)

    def _active_strategy_config(self) -> dict:
        payload = self._active_strategy_payload() or {}
        config = dict(payload.get("config") or {})
        if not config:
            config = copy.deepcopy(LONG_TERM_CONFIG)
        return config

    def _apply_active_strategy_runtime(self) -> None:
        config = self._active_strategy_config()
        self.signal_types = list(config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or []))
        self.max_positions = int(config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 0)
        self.holding_days = int(config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 0)
        self.sector_limit = float(config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.0)
        self.use_regime_filter = bool(config.get("use_regime_filter", True))

    def _load_strategies_registry(self) -> None:
        strategy_registry.ensure_builtin_strategies()
        self._strategies = strategy_registry.list_strategies()
        self._strategy_comparison_snapshot = strategy_registry.load_strategy_comparison_snapshot()
        current_account = self._strategy_account_binding(getattr(self, "_current_account_id", "account_1"))
        preferred = str((current_account or {}).get("strategy_id") or "default_c5")
        known = {str(item.get("id") or "") for item in self._strategies}
        self._selected_strategy_id = preferred if preferred in known else (next(iter(known)) if known else "default_c5")

    def _selected_strategy_payload(self) -> Optional[dict]:
        sid = str(getattr(self, "_selected_strategy_id", "") or "").strip()
        for item in list(getattr(self, "_strategies", []) or []):
            if str(item.get("id") or "") == sid:
                return item
        return strategy_registry.load_strategy(sid)

    def _set_strategy_form_from_payload(self, payload: Optional[dict]) -> None:
        if not payload:
            return
        config = dict(payload.get("config") or {})
        # Sync signal picker buttons
        active = set(str(s) for s in list(config.get("signal_types") or []))
        self._active_signals = active
        self._sync_signal_buttons()

        values = {
            "strategy-inp-name": str(payload.get("name") or ""),
            "strategy-inp-description": str(payload.get("description") or ""),
            "strategy-inp-holding-days": str(int(config.get("holding_days") or 0) or ""),
            "strategy-inp-rebalance": str(int(config.get("rebalance_frequency") or 0) or ""),
            "strategy-inp-max-positions": str(int(config.get("max_positions") or 0) or ""),
            "strategy-inp-stop-loss": str(config.get("stop_loss_pct") if config.get("stop_loss_pct") is not None else ""),
            "strategy-inp-trailing-stop": str(config.get("trailing_stop_pct") if config.get("trailing_stop_pct") is not None else ""),
            "strategy-inp-sector-limit": str(config.get("sector_limit") if config.get("sector_limit") is not None else ""),
        }
        for widget_id, value in values.items():
            try:
                self.query_one(f"#{widget_id}", Input).value = value
            except Exception:
                pass
        try:
            start_widget = self.query_one("#strategy-inp-backtest-start", Input)
            if not start_widget.value:
                start_widget.value = "2025-01-01"
        except Exception:
            pass
        try:
            end_widget = self.query_one("#strategy-inp-backtest-end", Input)
            if not end_widget.value:
                end_widget.value = str(getattr(self.md, "latest", datetime.now().strftime("%Y-%m-%d")) or datetime.now().strftime("%Y-%m-%d"))
        except Exception:
            pass
        try:
            cap_widget = self.query_one("#strategy-inp-backtest-capital", Input)
            if not cap_widget.value:
                cap_widget.value = "1000000"
        except Exception:
            pass

    def _strategy_form_payload(self) -> dict:
        if getattr(self, "_new_strategy_mode", False):
            # New strategy: start from clean defaults, not the currently selected one
            base = {}
            config = {
                "holding_days": 40,
                "max_positions": 5,
                "rebalance_frequency": 5,
                "stop_loss_pct": 0.08,
                "trailing_stop_pct": 0.15,
                "sector_limit": 0.35,
                "use_regime_filter": True,
                "use_trailing_stop": True,
                "signal_types": [],
            }
        else:
            base = self._selected_strategy_payload() or strategy_registry.load_strategy("default_c5") or {}
            config = copy.deepcopy(base.get("config") or {})

        def _read_input(widget_id: str) -> str:
            return str(self.query_one(f"#{widget_id}", Input).value or "").strip()

        name = _read_input("strategy-inp-name") or str(base.get("name") or "Strategy")
        description = _read_input("strategy-inp-description")
        # Signals come from the picker buttons (not a text input)
        active = set(getattr(self, "_active_signals", set()))
        if active:
            # Preserve original ordering where possible, append new ones
            orig_order = list(config.get("signal_types") or [])
            ordered = [s for s in orig_order if s in active]
            ordered += [s for s in active if s not in ordered]
            config["signal_types"] = ordered

        numeric_fields = {
            "holding_days": ("strategy-inp-holding-days", int),
            "rebalance_frequency": ("strategy-inp-rebalance", int),
            "max_positions": ("strategy-inp-max-positions", int),
            "stop_loss_pct": ("strategy-inp-stop-loss", float),
            "trailing_stop_pct": ("strategy-inp-trailing-stop", float),
            "sector_limit": ("strategy-inp-sector-limit", float),
        }
        for key, (widget_id, caster) in numeric_fields.items():
            raw = _read_input(widget_id)
            if raw:
                config[key] = caster(raw)

        return {
            "id": str(base.get("id") or ""),
            "name": name,
            "description": description,
            "runner_mode": str(base.get("runner_mode") or "temp_patched"),
            "execution_mode": str(base.get("execution_mode") or "paper_runtime"),
            "config": config,
            "ranking_overlay": copy.deepcopy(base.get("ranking_overlay") or {"mode": "baseline"}),
            "notes": {
                **dict(base.get("notes") or {}),
                "base_strategy_id": str(base.get("id") or ""),
            },
        }

    def _render_strategy_backtest_summary(self) -> None:
        try:
            widget = self.query_one("#strategy-backtest-summary", Static)
        except Exception:
            return
        result = dict(getattr(self, "_strategy_backtest_result", {}) or {})
        if not result:
            widget.update(
                Text.from_markup(
                    f"[#8aa0b5]Backtest artifacts[/] data/strategy_registry/backtests/\n"
                    f"[#708091]Run a backtest from this tab to store the latest result for the selected strategy.[/]"
                )
            )
            return
        summary = dict(result.get("summary") or {})
        nepse = dict(result.get("nepse") or {})
        widget.update(
            Text.from_markup(
                f"[bold {WHITE}]Window[/] {result.get('window', {}).get('start')} → {result.get('window', {}).get('end')}   "
                f"[bold {WHITE}]Capital[/] {_npr_k(float(result.get('window', {}).get('capital') or 0.0))}\n"
                f"[bold {WHITE}]Strategy[/] {summary.get('total_return_pct', 0.0):+.2f}%   "
                f"[bold {WHITE}]NEPSE[/] {float(nepse.get('return_pct') or 0.0):+.2f}%   "
                f"[bold {WHITE}]Alpha[/] {float(summary.get('vs_nepse_pct_points') or 0.0):+.2f}pp   "
                f"[bold {WHITE}]Sharpe[/] {float(summary.get('sharpe_ratio') or 0.0):.2f}   "
                f"[bold {WHITE}]MaxDD[/] {float(summary.get('max_drawdown_pct') or 0.0):+.2f}%   "
                f"[bold {WHITE}]Trades[/] {int(summary.get('trade_count') or 0)}\n"
                f"[#8aa0b5]Saved[/] data/strategy_registry/backtests/{str(result.get('strategy', {}).get('id') or 'strategy')}_latest.json"
            )
        )

    def _strategy_saved_metrics(self, strategy_id: str) -> Optional[dict]:
        return strategy_registry.comparison_metrics_for_strategy(strategy_id)

    def _populate_strategy_list(self) -> None:
        try:
            dt = self.query_one("#dt-strategy-list", DataTable)
        except Exception:
            return
        dt.clear(columns=True)
        for label, key, width in [
            ("NAME", "name", 8),
            ("SIG", "sig", 4),
            ("HOLD", "hold", 5),
            ("RET", "ret", 9),
            ("VS NP", "vs_np", 10),
            ("SHRP", "sharpe", 6),
            ("DD", "dd", 9),
            ("TRD", "trades", 4),
            ("WR", "win_rate", 6),
        ]:
            dt.add_column(label, key=key, width=width)
        selected_index = 0
        for idx, strategy in enumerate(list(getattr(self, "_strategies", []) or [])):
            sid = str(strategy.get("id") or "")
            config = dict(strategy.get("config") or {})
            signals = list(config.get("signal_types") or [])
            metrics = self._strategy_saved_metrics(sid) or {}
            if sid == str(getattr(self, "_selected_strategy_id", "") or ""):
                selected_index = idx
            dt.add_row(
                Text(self._strategy_display_name(sid, str(strategy.get("name") or sid)), style=WHITE),
                Text(str(len(signals)), style=WHITE),
                Text(str(config.get("holding_days") or ""), style=WHITE),
                Text(f"{float(metrics.get('total_return_pct') or 0.0):+.2f}%" if metrics else "—", style=GAIN_HI if float(metrics.get("total_return_pct") or 0.0) >= 0 else LOSS_HI),
                Text(f"{float(metrics.get('total_return_pct') or 0.0) - float((metrics.get('nepse') or {}).get('return_pct') or 0.0):+.2f}pp" if metrics else "—", style=GAIN_HI if (float(metrics.get("total_return_pct") or 0.0) - float((metrics.get("nepse") or {}).get("return_pct") or 0.0)) >= 0 else LOSS_HI),
                Text(f"{float(metrics.get('sharpe_ratio') or 0.0):.2f}" if metrics else "—", style=WHITE),
                Text(f"{float(metrics.get('max_drawdown_pct') or 0.0):+.2f}%" if metrics else "—", style=WHITE),
                Text(str(int(metrics.get("trade_count") or 0)) if metrics else "—", style=WHITE),
                Text(f"{float(metrics.get('win_rate_pct') or 0.0):.0f}%" if metrics else "—", style=WHITE),
            )
        if self._strategies:
            try:
                dt.move_cursor(row=selected_index)
            except Exception:
                pass
        else:
            dt.add_row(_dim_text("No strategies"), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""), Text(""))
        try:
            row_count = max(1, len(list(getattr(self, "_strategies", []) or [])))
            dt.styles.height = min(max(row_count + 3, 7), 10)
        except Exception:
            pass

    def _populate_strategies_tab(self) -> None:
        if not getattr(self, "_strategies", None):
            self._load_strategies_registry()
        self._populate_strategy_list()
        selected = self._selected_strategy_payload()
        self._set_strategy_form_from_payload(selected)
        current_strategy = self._strategy_name_for_account(getattr(self, "_current_account_id", "account_1"))
        selected_account_strategy = self._strategy_name_for_account(getattr(self, "_selected_account_id", self._current_account_id))
        try:
            metrics = self._strategy_saved_metrics(str((selected or {}).get("id") or ""))
            metrics_line = ""
            if metrics:
                window = dict(metrics.get("window") or {})
                nepse = dict(metrics.get("nepse") or {})
                alpha = float(metrics.get("total_return_pct") or 0.0) - float(nepse.get("return_pct") or 0.0)
                metrics_line = (
                    f"\n[bold {WHITE}]Saved baseline[/] {window.get('start')} → {window.get('end')}   "
                    f"[bold {WHITE}]Growth[/] {float(metrics.get('total_return_pct') or 0.0):+.2f}%   "
                    f"[bold {WHITE}]vs NEPSE[/] {alpha:+.2f}pp   "
                    f"[bold {WHITE}]Sharpe[/] {float(metrics.get('sharpe_ratio') or 0.0):.2f}   "
                    f"[bold {WHITE}]MaxDD[/] {float(metrics.get('max_drawdown_pct') or 0.0):+.2f}%   "
                    f"[bold {WHITE}]Trades[/] {int(metrics.get('trade_count') or 0)}   "
                    f"[bold {WHITE}]Win rate[/] {float(metrics.get('win_rate_pct') or 0.0):.0f}%"
                )
            self.query_one("#strategy-summary", Static).update(
                Text.from_markup(
                    f"[bold {WHITE}]Selected strategy[/] {str((selected or {}).get('name') or self._selected_strategy_id)}   "
                    f"[bold {WHITE}]Current account[/] {current_strategy}   "
                    f"[bold {WHITE}]Selected account[/] {selected_account_strategy}\n"
                    f"[#8aa0b5]Storage[/] data/strategy_registry/{metrics_line}"
                )
            )
        except Exception:
            pass
        try:
            account_lines = []
            for account in list(getattr(self, "_paper_accounts", []) or []):
                aid = str(account.get("id") or "")
                account_lines.append(
                    f"{str(account.get('name') or aid)} → {strategy_registry.strategy_name(str(account.get('strategy_id') or ''))}"
                )
            note = "\n".join(account_lines) if account_lines else "No account bindings"
            self.query_one("#strategy-accounts-note", Static).update(
                Text.from_markup(f"[#8aa0b5]Bindings[/]\n{note}")
            )
        except Exception:
            pass
        self._render_strategy_backtest_summary()

    def _assign_strategy_to_account(self, strategy_id: str, account_id: str) -> str:
        sid = str(strategy_id or "").strip()
        aid = str(account_id or "").strip()
        if not sid:
            raise ValueError("Select a strategy first")
        if not aid:
            raise ValueError("Select an account first")
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        now_stamp = datetime.now().isoformat(timespec="seconds")
        updated = False
        for account in accounts:
            if str(account.get("id") or "") == aid:
                account["strategy_id"] = sid
                account["updated_at"] = now_stamp
                updated = True
        if not updated:
            raise ValueError("Account not found")
        registry["accounts"] = accounts
        _save_accounts_registry(registry)
        self._paper_accounts = strategy_registry.ensure_account_strategy_ids(accounts)
        if aid == str(getattr(self, "_current_account_id", "") or ""):
            self._apply_active_strategy_runtime()
            self._signals_table_cache_key = ""
            self._signals_table_cache_payload = None
            self._sync_agent_account_context_env()
            self._load_signals_async(force=True)
        self._populate_account_list()
        self._populate_paper_profile_panel(self._stats)
        self._populate_strategies_tab()
        return f"Assigned {strategy_registry.strategy_name(sid)} to {aid}"

    def _delete_custom_strategy(self) -> str:
        selected = self._selected_strategy_payload()
        if not selected:
            return "No strategy selected"
        if str(selected.get("source") or "") != "custom":
            return f"Cannot delete built-in strategy '{selected.get('name')}'"
        sid = str(selected.get("id") or "").strip()
        if not sid:
            return "Invalid strategy ID"
        from backend.trading.strategy_registry import CUSTOM_STRATEGY_DIR
        path = CUSTOM_STRATEGY_DIR / f"{sid}.json"
        if path.exists():
            path.unlink()
        name = str(selected.get("name") or sid)
        self._load_strategies_registry()
        self._populate_strategies_tab()
        # Clear the form
        self._active_signals = set()
        self._sync_signal_buttons()
        return f"Deleted strategy '{name}'"

    def _save_strategy_from_form(self) -> str:
        new_mode = getattr(self, "_new_strategy_mode", False)
        payload = self._strategy_form_payload()
        if not str(payload.get("name") or "").strip():
            raise ValueError("Enter a strategy name")
        selected = self._selected_strategy_payload() or {}
        # Never overwrite an existing strategy when NEW was pressed
        overwrite = (
            not new_mode
            and str(selected.get("source") or "") == "custom"
            and str(selected.get("id") or "") == str(payload.get("id") or "")
        )
        if not overwrite:
            payload = dict(payload)
            payload.pop("id", None)
        saved = strategy_registry.save_custom_strategy(
            payload,
            strategy_id=(str(selected.get("id") or "") if overwrite else None),
            overwrite=overwrite,
        )
        self._new_strategy_mode = False
        self._load_strategies_registry()
        self._selected_strategy_id = str(saved.get("id") or self._selected_strategy_id)
        self._populate_strategies_tab()
        return f"Saved strategy {saved.get('name')} ({saved.get('id')})"

    @work(thread=True)
    def _run_strategy_backtest_async(self, strategy_id: str, start_date: str, end_date: str, capital: float) -> None:
        try:
            payload = strategy_registry.load_strategy(strategy_id)
            if not payload:
                raise ValueError("Strategy not found")
            self.call_from_thread(self._set_status, f"Backtesting {payload.get('name')}...")
            # Suppress noisy loggers during backtest to keep TUI clean
            _bt_loggers = [logging.getLogger(n) for n in ("backend", "simple_backtest", "__main__", "root")]
            _bt_prev_levels = [(lg, lg.level) for lg in _bt_loggers]
            for lg in _bt_loggers:
                lg.setLevel(logging.WARNING)
            root_logger = logging.getLogger()
            _root_prev = root_logger.level
            root_logger.setLevel(logging.WARNING)
            try:
                result = strategy_registry.run_strategy_backtest(
                    payload,
                    start_date=start_date,
                    end_date=end_date,
                    capital=capital,
                )
            finally:
                for lg, lvl in _bt_prev_levels:
                    lg.setLevel(lvl)
                root_logger.setLevel(_root_prev)
            self._strategy_backtest_result = result
            summary = dict(result.get("summary") or {})
            nepse   = dict(result.get("nepse") or {})
            self.call_from_thread(self._populate_strategies_tab)
            strat_ret = float(summary.get("total_return_pct") or 0.0)
            nepse_ret = float(nepse.get("return_pct") or 0.0)
            self.call_from_thread(
                self._set_status,
                f"Backtest done | {payload.get('name')} {strat_ret:+.2f}% vs NEPSE {nepse_ret:+.2f}%  |  Generating chart...",
            )
            # ── Generate quick chart (non-blocking, runs in same thread) ──────
            try:
                from validation.quick_chart import generate_quick_chart
                from backend.quant_pro.database import get_db_path as _get_db_path
                chart_path = generate_quick_chart(
                    result,
                    strategy_name=str(payload.get("name") or strategy_id),
                    start_date=start_date,
                    end_date=end_date,
                    db_path=str(_get_db_path()),
                    auto_open=True,
                )
                chart_note = f"  Chart → {chart_path}" if chart_path else ""
            except Exception:
                chart_note = ""
            self.call_from_thread(
                self._set_status,
                f"Backtest done | {payload.get('name')} {strat_ret:+.2f}% vs NEPSE {nepse_ret:+.2f}%{chart_note}",
            )
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Strategy backtest failed: {exc}")

    def _start_strategy_backtest(self) -> str:
        selected = self._selected_strategy_payload()
        if not selected:
            raise ValueError("Select a strategy first")
        start = str(self.query_one("#strategy-inp-backtest-start", Input).value or "").strip()
        end = str(self.query_one("#strategy-inp-backtest-end", Input).value or "").strip()
        capital_raw = str(self.query_one("#strategy-inp-backtest-capital", Input).value or "").strip().replace(",", "")
        if not start or not end:
            raise ValueError("Enter start and end dates")
        capital = float(capital_raw or INITIAL_CAPITAL)
        self._run_strategy_backtest_async(str(selected.get("id") or ""), start, end, capital)
        return f"Backtest launched for {selected.get('name')}"

    def _profile_runtime_snapshot(self, *, account_dir: Optional[Path] = None) -> dict:
        if account_dir:
            portfolio_path = account_dir / "paper_portfolio.csv"
            trade_log_path = account_dir / "paper_trade_log.csv"
            nav_log_path = account_dir / "paper_nav_log.csv"
            state_path = account_dir / "paper_state.json"
            portfolio = pd.read_csv(portfolio_path) if portfolio_path.exists() else pd.DataFrame(columns=PORTFOLIO_COLS)
            trade_log = pd.read_csv(trade_log_path) if trade_log_path.exists() else pd.DataFrame(columns=TRADE_LOG_COLS)
            nav_log = pd.read_csv(nav_log_path) if nav_log_path.exists() else pd.DataFrame(columns=NAV_LOG_COLS)
            state = load_runtime_state(str(state_path))
        else:
            portfolio = load_port()
            nav_log = _load_nav_log()
            trade_log = _load_trade_log()
            state = load_runtime_state(str(PAPER_STATE_FILE))
        holdings = len(portfolio.index) if isinstance(portfolio, pd.DataFrame) else 0
        trade_count = len(trade_log.index) if isinstance(trade_log, pd.DataFrame) else 0
        nav_rows = len(nav_log.index) if isinstance(nav_log, pd.DataFrame) else 0
        cash = state.get("cash")
        if not isinstance(cash, (int, float)):
            cash = _load_manual_paper_cash(0.0, nav_log)
        nav_value = None
        if isinstance(nav_log, pd.DataFrame) and not nav_log.empty and "NAV" in nav_log.columns:
            try:
                nav_value = float(nav_log.iloc[-1]["NAV"])
            except Exception:
                nav_value = None
        return {
            "holdings": holdings,
            "trades": trade_count,
            "nav_rows": nav_rows,
            "cash": float(cash),
            "nav": float(nav_value) if isinstance(nav_value, (int, float)) else (float(cash) + _portfolio_mark_value(portfolio)),
            "runtime_dir": str(TRADING_RUNTIME_DIR),
        }

    def _populate_paper_profile_panel(self, stats: Optional[dict] = None) -> None:
        snapshot = self._profile_runtime_snapshot()
        profile = _load_profile_config()
        current_id = str(getattr(self, "_current_account_id", "account_1") or "account_1")
        current_strategy = self._strategy_name_for_account(current_id)
        current_name = next(
            (str(account.get("name") or current_id) for account in getattr(self, "_paper_accounts", []) if str(account.get("id") or "") == current_id),
            current_id,
        )
        positions = list((stats or {}).get("positions") or [])
        held_syms = ", ".join(p.get("sym", "") for p in positions[:6] if p.get("sym"))
        if len(positions) > 6:
            held_syms += ", ..."
        if not held_syms:
            held_syms = "No holdings loaded"
        selected_id = str(getattr(self, "_selected_account_id", current_id) or current_id)
        selected_strategy = self._strategy_name_for_account(selected_id)
        selected_name = next(
            (str(account.get("name") or selected_id) for account in getattr(self, "_paper_accounts", []) if str(account.get("id") or "") == selected_id),
            selected_id,
        )
        mode_note = "Creating or activating an account swaps the full paper runtime"
        summary = Text.from_markup(
            f"[bold {WHITE}]Active[/] {current_name}   "
            f"[bold {WHITE}]Strategy[/] {current_strategy}   "
            f"[bold {WHITE}]Selected[/] {selected_name}   "
            f"[bold {WHITE}]Selected strategy[/] {selected_strategy}   "
            f"[bold {WHITE}]Holdings[/] {snapshot['holdings']}   "
            f"[bold {WHITE}]Trades[/] {snapshot['trades']}   "
            f"[bold {WHITE}]NAV rows[/] {snapshot['nav_rows']}   "
            f"[bold {WHITE}]NAV[/] {_npr_k(snapshot['nav'])}   "
            f"[bold {WHITE}]Cash[/] {_npr_k(snapshot['cash'])}\n"
            f"[#8aa0b5]Held symbols:[/] {held_syms}\n"
            f"[#708091]{mode_note}[/]"
        )
        try:
            self.query_one("#profile-summary", Static).update(summary)
        except Exception:
            return
        try:
            widget = self.query_one("#profile-inp-portfolio", Input)
            if not widget.value and profile.get("portfolio_path"):
                widget.value = str(profile.get("portfolio_path") or "")
        except Exception:
            pass
        try:
            target_nav = profile.get("target_nav")
            if target_nav:
                self.query_one("#profile-inp-target-nav", Input).value = f"{float(target_nav):,.2f}".replace(",", "")
        except Exception:
            pass
        try:
            self.query_one("#profile-shortcuts", Static).update(
                Text.from_markup(
                    f"[#708091]ACCOUNT KEYS[/]   "
                    f"[bold {AMBER}]N[/] NEW   "
                    f"[bold {AMBER}]A[/] ACTIVATE   "
                    f"[bold {AMBER}]V[/] SET NAV   "
                    f"[bold {AMBER}]W[/] WATCHLIST   "
                    f"[bold {AMBER}]H[/] HELP"
                )
            )
        except Exception:
            pass
        help_text = ""
        if self._account_help_visible:
            help_text = (
                "N  Create account from NAME + NAV + optional SEED file\n"
                "A  Activate selected account row\n"
                "V  Apply NAV field to active account\n"
                "W  Sync watchlist from active holdings\n"
                "H  Toggle command help\n"
                "Tip: select an account row first, then activate"
            )
        try:
            self.query_one("#account-help", Static).update(Text(help_text, style=LABEL))
        except Exception:
            pass
        self._populate_account_list()

    @work(thread=True)
    def _browse_seed_file(self) -> None:
        """Open a native macOS file picker and populate the seed path input."""
        import subprocess as _sp
        self.call_from_thread(self._set_status, "Opening file picker…")
        try:
            script = (
                'POSIX path of (choose file with prompt '
                '"Select portfolio CSV (paper_portfolio.csv or MeroShare CSV)" '
                'of type {"csv", "CSV"})'
            )
            result = _sp.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            chosen = result.stdout.strip()
            if not chosen:
                self.call_from_thread(self._set_status, "Browse cancelled")
                return
            # Populate the seed input and show a preview
            self.call_from_thread(
                lambda p=chosen: (
                    setattr(self.query_one("#profile-inp-portfolio", Input), "value", p)
                    or self._set_status(f"Seed file selected: {Path(p).name}")
                )
            )
        except FileNotFoundError:
            self.call_from_thread(
                self._set_status,
                "osascript not found — type the path manually into the SEED field",
            )
        except Exception as exc:
            self.call_from_thread(self._set_status, f"Browse failed: {exc}")

    def _account_create_from_form(self) -> str:
        raw_name = self.query_one("#profile-inp-account-name", Input).value
        raw_path = self.query_one("#profile-inp-portfolio", Input).value
        raw_nav = self.query_one("#profile-inp-target-nav", Input).value
        return self._create_paper_account(raw_name, raw_path, raw_nav)

    def _account_activate_selected(self) -> str:
        return self._activate_paper_account(self._selected_account_id)

    def _account_set_nav_from_form(self) -> str:
        raw_nav = self.query_one("#profile-inp-target-nav", Input).value
        msg = self._seed_manual_nav(raw_nav)
        self._populate_portfolio_and_risk()
        return msg

    def _account_sync_watchlist(self) -> str:
        held = self._sync_holdings_watchlist()
        self._populate_paper_profile_panel(self._stats)
        return f"Watchlist synced from {held} holdings"

    def _toggle_account_help(self) -> str:
        self._account_help_visible = not self._account_help_visible
        self._populate_paper_profile_panel(self._stats)
        return "Account help shown" if self._account_help_visible else "Account help hidden"

    def _backup_profile_targets(self, targets: list[Path]) -> Optional[Path]:
        existing = [Path(target) for target in targets if Path(target).exists()]
        if not existing:
            return None
        backup_dir = ensure_dir(PAPER_IMPORT_BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))
        for target in existing:
            shutil.copy2(target, backup_dir / target.name)
        return backup_dir

    def _persist_profile_paths(self, **updates) -> None:
        profile = _load_profile_config()
        profile.update({k: v for k, v in updates.items() if v is not None})
        _save_profile_config(profile)

    def _account_snapshot(self, account: dict) -> dict:
        account_id = str(account.get("id") or "")
        account_dir = _account_dir(account_id)
        snap = self._profile_runtime_snapshot(account_dir=account_dir)
        return {
            "id": account_id,
            "name": str(account.get("name") or account_id),
            "strategy": strategy_registry.strategy_name(str(account.get("strategy_id") or strategy_registry.default_strategy_for_account(account_id))),
            "holdings": snap["holdings"],
            "trades": snap["trades"],
            "nav": snap["nav"],
            "cash": snap["cash"],
            "updated_at": str(account.get("updated_at") or ""),
        }

    def _populate_account_list(self) -> None:
        try:
            dt = self.query_one("#dt-account-list", DataTable)
        except Exception:
            return
        dt.clear(columns=True)
        for label, key in [("ID", "id"), ("NAME", "name"), ("STRAT", "strategy"), ("HOLD", "hold"), ("NAV", "nav"), ("CASH", "cash"), ("STATUS", "status")]:
            dt.add_column(label, key=key)
        rows = []
        selected_index = 0
        for idx, account in enumerate(getattr(self, "_paper_accounts", []) or []):
            snap = self._account_snapshot(account)
            account_id = str(account.get("id") or "")
            status = "ACTIVE" if account_id == self._current_account_id else ("SELECTED" if account_id == self._selected_account_id else "")
            if account_id == self._selected_account_id:
                selected_index = idx
            rows.append(snap)
            dt.add_row(
                Text(account_id.replace("account_", "A"), style=DIM),
                Text(snap["name"], style=WHITE),
                Text(snap["strategy"], style=CYAN),
                Text(str(snap["holdings"]), style=WHITE),
                Text(_npr_k(snap["nav"]), style=AMBER),
                Text(_npr_k(snap["cash"]), style=WHITE),
                Text(status, style=f"bold {GAIN_HI}" if status == "ACTIVE" else CYAN if status == "SELECTED" else DIM),
            )
        if not rows:
            dt.add_row(_dim_text("—"), _dim_text("No accounts"), Text(""), Text(""), Text(""), Text(""), Text(""))
        else:
            try:
                dt.move_cursor(row=selected_index)
            except Exception:
                pass

    def _persist_active_account_snapshot(self) -> None:
        account_id = str(getattr(self, "_current_account_id", "") or "").strip()
        if not account_id:
            return
        target_dir = _account_dir(account_id)
        _blank_account_files(target_dir)
        for name, active_path in ACTIVE_ACCOUNT_FILES.items():
            _copy_file_if_exists(Path(active_path), target_dir / name)
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        now_stamp = datetime.now().isoformat(timespec="seconds")
        for account in accounts:
            if str(account.get("id") or "") == account_id:
                account["updated_at"] = now_stamp
        registry["accounts"] = accounts
        _save_accounts_registry(registry)
        self._paper_accounts = accounts

    def _activate_paper_account(self, account_id: str) -> str:
        target_id = str(account_id or "").strip()
        if not target_id:
            raise ValueError("Select an account to activate")
        if target_id == self._current_account_id:
            return f"{target_id} is already active"
        self._persist_active_account_snapshot()
        source_dir = _account_dir(target_id)
        _blank_account_files(source_dir)
        backup_dir = self._backup_profile_targets(list(ACTIVE_ACCOUNT_FILES.values()))
        for name, active_path in ACTIVE_ACCOUNT_FILES.items():
            source = source_dir / name
            target = Path(active_path)
            if source.exists():
                shutil.copy2(source, target)
        self._current_account_id = target_id
        self._selected_account_id = target_id
        # Switch active engine reference to the newly activated account
        self._trading_engine = self._account_engines.get(target_id)
        target_account = self._strategy_account_binding(target_id) or {}
        self._selected_strategy_id = str(target_account.get("strategy_id") or self._selected_strategy_id)
        self._apply_active_strategy_runtime()
        self._signals_table_cache_key = ""
        self._signals_table_cache_payload = None
        self._sync_agent_account_context_env()
        profile = _load_profile_config()
        profile["current_account_id"] = target_id
        _save_profile_config(profile)
        self._paper_watchlist = _load_watchlist()
        self._watchlist = list(self._paper_watchlist)
        self._load_paper_orders()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._populate_orders_tab()
        self._populate_watchlist()
        self._populate_paper_profile_panel(self._stats)
        self._populate_strategies_tab()
        self._load_signals_async(force=True)
        backup_note = f" | backup {backup_dir.name}" if backup_dir else ""
        active_name = next((str(a.get("name") or target_id) for a in self._paper_accounts if str(a.get("id") or "") == target_id), target_id)
        return f"Activated {active_name}{backup_note}"

    def _create_paper_account(self, raw_name: str, raw_portfolio_path: str, raw_target_nav: str) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("Enter an account name")
        portfolio_path = _coerce_dragdrop_path(raw_portfolio_path)
        token = str(raw_target_nav or "").strip().replace(",", "")
        if not token:
            raise ValueError("Enter a target NAV")
        target_nav = float(token)
        if target_nav <= 0:
            raise ValueError("Target NAV must be positive")
        if portfolio_path:
            if not portfolio_path.exists():
                raise ValueError("Portfolio CSV path not found")
            df = pd.read_csv(portfolio_path)
            portfolio_df = _normalize_import_portfolio(df)
        else:
            portfolio_df = pd.DataFrame(columns=PORTFOLIO_COLS)
        state, nav_log = _build_account_seed_state(portfolio_df, target_nav)
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        if len(accounts) >= MAX_ACCOUNTS:
            raise ValueError(f"Maximum {MAX_ACCOUNTS} accounts allowed")
        if any(str(account.get("name") or "").strip().lower() == name.lower() for account in accounts):
            raise ValueError(f"{name} already exists")
        account_id = _next_account_id(accounts)
        target_dir = _account_dir(account_id)
        _blank_account_files(target_dir)
        portfolio_df.reindex(columns=PORTFOLIO_COLS).to_csv(target_dir / "paper_portfolio.csv", index=False)
        portfolio_df.reindex(columns=PORTFOLIO_COLS).to_csv(target_dir / "tui_paper_portfolio.csv", index=False)
        pd.DataFrame(columns=TRADE_LOG_COLS).to_csv(target_dir / "paper_trade_log.csv", index=False)
        pd.DataFrame(columns=TRADE_LOG_COLS).to_csv(target_dir / "tui_paper_trade_log.csv", index=False)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(target_dir / "paper_nav_log.csv", index=False)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(target_dir / "tui_paper_nav_log.csv", index=False)
        save_runtime_state(str(target_dir / "paper_state.json"), dict(state))
        save_runtime_state(str(target_dir / "tui_paper_state.json"), dict(state))
        watchlist_entries = _build_holdings_watchlist_entries(portfolio_df)
        (target_dir / "watchlist.json").write_text(_json.dumps(_merge_watchlist_entries(watchlist_entries), indent=2))
        (target_dir / "tui_paper_orders.json").write_text("[]")
        (target_dir / "tui_paper_order_history.json").write_text("[]")
        now_stamp = datetime.now().isoformat(timespec="seconds")
        accounts.append({
            "id": account_id,
            "name": name,
            "strategy_id": strategy_registry.default_strategy_for_account(account_id),
            "created_at": now_stamp,
            "updated_at": now_stamp,
        })
        registry["accounts"] = accounts
        _save_accounts_registry(registry)
        self._paper_accounts = strategy_registry.ensure_account_strategy_ids(accounts)
        self._persist_profile_paths(portfolio_path=str(portfolio_path) if portfolio_path else "", target_nav=target_nav)
        self.query_one("#profile-inp-account-name", Input).value = f"Account {len(accounts) + 1}"
        self._selected_account_id = account_id
        # Start engine for the new account if multi-engine is running
        if self.trade_mode == "paper" and account_id not in getattr(self, "_account_engines", {}):
            _new_strategy = strategy_registry.load_strategy(strategy_registry.default_strategy_for_account(account_id))
            _new_config = dict((_new_strategy or {}).get("config") or {}) or dict(LONG_TERM_CONFIG)
            _new_adir = _account_dir(account_id)
            _new_engine = TUITradingEngine(
                capital=float(_new_config.get("initial_capital") or INITIAL_CAPITAL),
                signal_types=list(_new_config.get("signal_types") or list(LONG_TERM_CONFIG.get("signal_types") or [])),
                max_positions=int(_new_config.get("max_positions") or LONG_TERM_CONFIG.get("max_positions") or 5),
                holding_days=int(_new_config.get("holding_days") or LONG_TERM_CONFIG.get("holding_days") or 40),
                sector_limit=float(_new_config.get("sector_limit") or LONG_TERM_CONFIG.get("sector_limit") or 0.35),
                stop_loss_pct=float(_new_config.get("stop_loss_pct") or LONG_TERM_CONFIG.get("stop_loss_pct") or 0.08),
                trailing_stop_pct=float(_new_config.get("trailing_stop_pct") or LONG_TERM_CONFIG.get("trailing_stop_pct") or 0.10),
                hedge_enabled=bool(getattr(self, "_hedge_enabled", True)),
                portfolio_file=_new_adir / "tui_paper_portfolio.csv",
                trade_log_file=_new_adir / "tui_paper_trade_log.csv",
                nav_log_file=_new_adir / "tui_paper_nav_log.csv",
                state_file=_new_adir / "tui_paper_state.json",
                on_status=self._make_engine_status_cb(account_id),
                on_activity=self._make_engine_activity_cb(account_id),
                on_portfolio_changed=self._make_engine_portfolio_cb(account_id),
                on_agent_updated=self._make_engine_agent_cb(account_id),
            )
            self._account_engines[account_id] = _new_engine
            self._start_account_engine_worker(account_id)
        message = self._activate_paper_account(account_id)
        seed_note = f"{portfolio_df.shape[0]} seeded holdings" if not portfolio_df.empty else "cash-only account"
        return f"Created {name} | {seed_note} | {message}"

    def _save_current_account_snapshot(self) -> str:
        self._persist_active_account_snapshot()
        self._populate_paper_profile_panel(self._stats)
        active_name = next((str(a.get("name") or self._current_account_id) for a in self._paper_accounts if str(a.get("id") or "") == self._current_account_id), self._current_account_id)
        return f"Saved snapshot for {active_name}"

    def _delete_paper_account(self, account_id: str) -> str:
        """Stop engine, backup files, remove account from registry."""
        target_id = str(account_id or "").strip()
        if not target_id:
            raise ValueError("No account selected")
        registry = _load_accounts_registry()
        accounts = list(registry.get("accounts") or [])
        if len(accounts) <= 1:
            raise ValueError("Cannot delete the only account")
        target = next((a for a in accounts if str(a.get("id") or "") == target_id), None)
        if not target:
            raise ValueError(f"Account {target_id} not found")
        # Stop and remove the account's engine
        engine = self._account_engines.pop(target_id, None)
        if engine:
            try:
                engine.stop()
            except Exception:
                pass
        if self._trading_engine is engine:
            self._trading_engine = None
        # Backup account directory before deleting
        account_dir = _account_dir(target_id)
        if account_dir.exists():
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = ensure_dir(PAPER_IMPORT_BACKUP_DIR / f"deleted_{target_id}_{stamp}")
                shutil.copytree(account_dir, backup / target_id)
                shutil.rmtree(account_dir, ignore_errors=True)
            except Exception:
                pass
        # Remove from registry
        remaining = [a for a in accounts if str(a.get("id") or "") != target_id]
        registry["accounts"] = remaining
        _save_accounts_registry(registry)
        self._paper_accounts = strategy_registry.ensure_account_strategy_ids(remaining)
        name = str(target.get("name") or target_id)
        # If deleting the active account, activate the first remaining one
        if target_id == self._current_account_id and remaining:
            try:
                self._activate_paper_account(str(remaining[0].get("id") or ""))
            except Exception:
                pass
        else:
            self._populate_account_list()
        return f"Deleted '{name}' (backup saved)"

    def _copy_import_dataframe(
        self,
        source_path: Path,
        target_path: Path,
        *,
        normalizer,
        mirror_path: Optional[Path] = None,
    ) -> int:
        df = pd.read_csv(source_path)
        normalized = normalizer(df)
        ensure_dir(target_path.parent)
        normalized.to_csv(target_path, index=False)
        if mirror_path:
            ensure_dir(mirror_path.parent)
            normalized.to_csv(mirror_path, index=False)
        return len(normalized.index)

    def _effective_paper_watchlist(self) -> tuple[list[dict], int]:
        ltps = self.md.ltps() if hasattr(self, "md") else {}
        port = load_port()
        held_entries = _build_holdings_watchlist_entries(port, ltps)
        return _merge_watchlist_entries(held_entries, self._paper_watchlist), len(held_entries)

    def _sync_holdings_watchlist(self) -> int:
        port = load_port()
        held_entries = _build_holdings_watchlist_entries(port, self.md.ltps() if hasattr(self, "md") else {})
        merged = _merge_watchlist_entries(held_entries, self._paper_watchlist)
        self._paper_watchlist = merged
        _save_watchlist(self._paper_watchlist)
        self._watchlist = list(merged)
        self._populate_watchlist()
        self._persist_active_account_snapshot()
        self._populate_account_list()
        return len(held_entries)

    def _seed_manual_nav(self, raw_value: str) -> str:
        token = str(raw_value or "").strip().replace(",", "")
        if not token:
            raise ValueError("Enter a target NAV")
        target_nav = float(token)
        if target_nav <= 0:
            raise ValueError("Target NAV must be positive")
        stats = _compute_portfolio_stats(self.md)
        positions_value = float(stats.get("total_value") or 0.0)
        target_cash = round(target_nav - positions_value, 2)
        if target_cash < 0:
            raise ValueError(f"Target NAV is below current positions value {_npr_k(positions_value)}")

        state = load_runtime_state(str(PAPER_STATE_FILE))
        state["cash"] = target_cash
        state["daily_start_nav"] = target_nav
        save_runtime_state(str(PAPER_STATE_FILE), state)

        tui_state = load_runtime_state(str(TUI_PAPER_STATE_FILE))
        tui_state["cash"] = target_cash
        tui_state["daily_start_nav"] = target_nav
        save_runtime_state(str(TUI_PAPER_STATE_FILE), tui_state)

        nav_log = _load_nav_log()
        today = datetime.now().strftime("%Y-%m-%d")
        row = {
            "Date": today,
            "Cash": target_cash,
            "Positions_Value": round(positions_value, 2),
            "NAV": round(target_nav, 2),
            "Num_Positions": int(stats.get("n_positions") or 0),
        }
        if not nav_log.empty and str(nav_log.iloc[-1].get("Date", ""))[:10] == today:
            nav_log.iloc[-1] = row
        else:
            nav_log = pd.concat([nav_log, pd.DataFrame([row])], ignore_index=True)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(PAPER_NAV_LOG_FILE, index=False)
        nav_log.reindex(columns=NAV_LOG_COLS).to_csv(TUI_PAPER_NAV_LOG_FILE, index=False)

        self._persist_profile_paths(target_nav=target_nav)
        self._persist_active_account_snapshot()
        self._populate_account_list()
        return f"NAV set to {_npr_k(target_nav)} with cash {_npr_k(target_cash)}"

    def _import_profile_bundle(self, raw_path: str) -> str:
        path = _coerce_dragdrop_path(raw_path)
        if not path or not path.exists():
            raise ValueError("Import folder/file path not found")
        base_dir = path if path.is_dir() else path.parent
        portfolio = base_dir / "paper_portfolio.csv"
        trades = base_dir / "paper_trade_log.csv"
        nav = base_dir / "paper_nav_log.csv"
        state = base_dir / "paper_state.json"
        if not portfolio.exists() and path.is_file() and path.suffix.lower() == ".csv":
            if "portfolio" in path.name:
                portfolio = path
            elif "trade" in path.name:
                trades = path
            elif "nav" in path.name:
                nav = path
        if not any(p.exists() for p in (portfolio, trades, nav, state)):
            raise ValueError("No paper trading files found in that folder")
        backup_dir = self._backup_profile_targets(
            [
                PAPER_PORTFOLIO_FILE,
                PAPER_TRADE_LOG_FILE,
                PAPER_NAV_LOG_FILE,
                PAPER_STATE_FILE,
                TUI_PAPER_PORTFOLIO_FILE,
                TUI_PAPER_TRADE_LOG_FILE,
                TUI_PAPER_NAV_LOG_FILE,
                TUI_PAPER_STATE_FILE,
            ]
        )
        imported: list[str] = []
        if portfolio.exists():
            count = self._copy_import_dataframe(
                portfolio,
                PAPER_PORTFOLIO_FILE,
                normalizer=_normalize_import_portfolio,
                mirror_path=TUI_PAPER_PORTFOLIO_FILE,
            )
            imported.append(f"portfolio {count}")
        if trades.exists():
            count = self._copy_import_dataframe(
                trades,
                PAPER_TRADE_LOG_FILE,
                normalizer=_normalize_import_trade_log,
                mirror_path=TUI_PAPER_TRADE_LOG_FILE,
            )
            imported.append(f"trades {count}")
        if nav.exists():
            count = self._copy_import_dataframe(
                nav,
                PAPER_NAV_LOG_FILE,
                normalizer=_normalize_import_nav_log,
                mirror_path=TUI_PAPER_NAV_LOG_FILE,
            )
            imported.append(f"nav {count}")
        if state.exists():
            payload = load_runtime_state(str(state))
            if not isinstance(payload, dict):
                raise ValueError("paper_state.json is not valid JSON")
            save_runtime_state(str(PAPER_STATE_FILE), payload)
            save_runtime_state(str(TUI_PAPER_STATE_FILE), payload)
            imported.append("state")
        held_count = self._sync_holdings_watchlist()
        self._persist_profile_paths(
            folder_path=str(base_dir),
            portfolio_path=str(portfolio) if portfolio.exists() else None,
            trades_path=str(trades) if trades.exists() else None,
            nav_path=str(nav) if nav.exists() else None,
        )
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._populate_watchlist()
        self._populate_paper_profile_panel(self._stats)
        backup_note = f" | backup {backup_dir.name}" if backup_dir else ""
        return f"Imported {', '.join(imported)} | holdings synced {held_count}{backup_note}"

    def _import_profile_csv(self, kind: str, raw_path: str) -> str:
        path = _coerce_dragdrop_path(raw_path)
        if not path or not path.exists():
            raise ValueError(f"{kind.title()} CSV path not found")
        if path.is_dir():
            return self._import_profile_bundle(str(path))
        backup_dir = None
        if kind == "portfolio":
            backup_dir = self._backup_profile_targets([PAPER_PORTFOLIO_FILE, TUI_PAPER_PORTFOLIO_FILE])
            count = self._copy_import_dataframe(
                path,
                PAPER_PORTFOLIO_FILE,
                normalizer=_normalize_import_portfolio,
                mirror_path=TUI_PAPER_PORTFOLIO_FILE,
            )
            held_count = self._sync_holdings_watchlist()
            self._persist_profile_paths(portfolio_path=str(path))
            message = f"Imported portfolio {count} rows | holdings synced {held_count}"
        elif kind == "trades":
            backup_dir = self._backup_profile_targets([PAPER_TRADE_LOG_FILE, TUI_PAPER_TRADE_LOG_FILE])
            count = self._copy_import_dataframe(
                path,
                PAPER_TRADE_LOG_FILE,
                normalizer=_normalize_import_trade_log,
                mirror_path=TUI_PAPER_TRADE_LOG_FILE,
            )
            rebuilt_cash = calculate_cash_from_trade_log(INITIAL_CAPITAL, str(PAPER_TRADE_LOG_FILE))
            if rebuilt_cash is not None:
                state = load_runtime_state(str(PAPER_STATE_FILE))
                state["cash"] = float(rebuilt_cash)
                save_runtime_state(str(PAPER_STATE_FILE), state)
                tui_state = load_runtime_state(str(TUI_PAPER_STATE_FILE))
                tui_state["cash"] = float(rebuilt_cash)
                save_runtime_state(str(TUI_PAPER_STATE_FILE), tui_state)
            self._persist_profile_paths(trades_path=str(path))
            message = f"Imported trade log {count} rows"
        elif kind == "nav":
            backup_dir = self._backup_profile_targets([PAPER_NAV_LOG_FILE, TUI_PAPER_NAV_LOG_FILE])
            count = self._copy_import_dataframe(
                path,
                PAPER_NAV_LOG_FILE,
                normalizer=_normalize_import_nav_log,
                mirror_path=TUI_PAPER_NAV_LOG_FILE,
            )
            nav_log = _load_nav_log()
            if not nav_log.empty:
                try:
                    latest = nav_log.iloc[-1]
                    state = load_runtime_state(str(PAPER_STATE_FILE))
                    state["cash"] = float(latest.get("Cash") or state.get("cash") or 0.0)
                    state["daily_start_nav"] = float(latest.get("NAV") or state.get("daily_start_nav") or 0.0)
                    save_runtime_state(str(PAPER_STATE_FILE), state)
                    tui_state = load_runtime_state(str(TUI_PAPER_STATE_FILE))
                    tui_state["cash"] = state["cash"]
                    tui_state["daily_start_nav"] = state["daily_start_nav"]
                    save_runtime_state(str(TUI_PAPER_STATE_FILE), tui_state)
                except Exception:
                    pass
            self._persist_profile_paths(nav_path=str(path))
            message = f"Imported NAV log {count} rows"
        else:
            raise ValueError(f"Unsupported import kind: {kind}")
        self._populate_portfolio_and_risk()
        self._populate_trades_full()
        self._render_hedge_panel()
        self._populate_watchlist()
        self._populate_paper_profile_panel(self._stats)
        backup_note = f" | backup {backup_dir.name}" if backup_dir else ""
        return f"{message}{backup_note}"

    def _populate_portfolio_tab(self, s: dict):
        # NAV summary bar
        rc = _pnl_color(s["total_return"])
        gross_rc = _pnl_color(s.get("gross_return", s["total_return"]))
        alpha_c = _pnl_color(s["alpha"])
        mode_tag = self._display_nav_mode_tag()
        nav_parts = [
            mode_tag,
            f"[bold {AMBER}]NAV[/] [bold {WHITE}]{_npr_k(s['nav'])}[/]",
            f"[#888888]Cash[/] [bold {WHITE}]{_npr_k(s['cash'])}[/]",
            f"[#888888]Invested[/] [bold {WHITE}]{_npr_k(s['total_cost'])}[/]",
            f"[#888888]Day[/] [bold {_pnl_color(s.get('day_pnl', 0.0))}]{_npr_k(s.get('day_pnl', 0.0))}[/] "
            f"[{_pnl_color(s.get('day_ret', 0.0))}]{s.get('day_ret', 0.0):+.2f}%[/]",
            f"[#888888]Net[/] [bold {rc}]{s['total_return']:+.2f}%[/]",
            f"[#888888]Gross[/] [bold {gross_rc}]{s.get('gross_return', s['total_return']):+.2f}%[/]",
            f"[#888888]NEPSE[/] [{_pnl_color(s['nepse_ret'])}]{s['nepse_ret']:+.2f}%[/]",
            f"[#888888]Alpha[/] [bold {alpha_c}]{s['alpha']:+.2f}pts[/]",
            f"[#888888]MaxDD[/] [bold {LOSS_HI}]{s['max_dd']:.1f}%[/]",
            f"[#555555]Sig[/] [#888888]Showing full signal names[/]",
        ]
        self.query_one("#nav-summary", Static).update(
            Text.from_markup("   ".join(nav_parts)))

        # Holdings table
        dt = self.query_one("#dt-portfolio", DataTable)
        dt.clear(columns=True)
        for label, key in [("SYMBOL", "sym"), ("QTY", "qty"), ("ENTRY", "entry"),
                           ("LTP", "ltp"), ("DAY", "day"), ("DAY%", "day_rtn"),
                           ("P&L", "pnl"), ("RTN%", "rtn"),
                           ("DAYS", "days"), ("SIGNAL", "sig"), ("SECTOR", "sec")]:
            dt.add_column(label, key=key)
        for p in s["positions"]:
            dt.add_row(
                _sym_text(p["sym"]), Text(str(p["qty"]), style=WHITE),
                _dim_text(f"{p['entry']:.1f}"), _price_text(p["cur"]),
                _npr(p.get("day_pnl", 0.0)),
                _chg_text(p.get("day_ret", 0.0)),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                Text(str(p["days"]) + "d", style=YELLOW if p["days"] > 30 else LABEL),
                _dim_text(p["signal"]), _dim_text(p["sector"]),
            )
        if not s["positions"]:
            dt.add_row(_dim_text("No positions"), *[Text("")] * 10)
        self._populate_paper_profile_panel(s)

    def _populate_risk_tab(self, s: dict):
        # Risk summary bar
        parts = [
            f"[bold {AMBER}]RISK DASHBOARD[/]",
            f"[#888888]Positions[/] [bold {WHITE}]{s['n_positions']}[/]",
            f"[#888888]MaxDD[/] [bold {LOSS_HI}]{s['max_dd']:.1f}%[/]",
            f"[#888888]Peak NAV[/] [bold {WHITE}]{_npr_k(s['peak_nav'])}[/]",
            f"[#888888]Top3 Conc[/] [bold {YELLOW}]{s['top3_conc']:.1f}%[/]",
            f"[#888888]Realized[/] [{_pnl_color(s['realized'])}]{_npr_k(s['realized'])}[/]",
            f"[#888888]Unrealized[/] [{_pnl_color(s['unrealized'])}]{_npr_k(s['unrealized'])}[/]",
            f"[#888888]Age[/] [bold {WHITE}]{s['age_0_5']}≤5d {s['age_6_15']}≤15d {s['age_16']}>15d[/]",
        ]
        self.query_one("#risk-summary", Static).update(
            Text.from_markup("   ".join(parts)))

        # Concentration table (sectors + positions)
        dt = self.query_one("#dt-concentration", DataTable)
        dt.clear(columns=True)
        for label, key in [("TYPE", "type"), ("NAME", "name"),
                           ("VALUE", "val"), ("WEIGHT%", "wt")]:
            dt.add_column(label, key=key)

        tv = s["total_value"] if s["total_value"] > 0 else 1
        # Top positions by weight
        for p in s["positions"][:5]:
            wt = p["val"] / tv * 100
            dt.add_row(
                Text("POSITION", style=CYAN),
                _sym_text(p["sym"]),
                Text(_npr_k(p["val"]), style=WHITE),
                Text(f"{wt:.1f}%", style=f"bold {YELLOW}" if wt > 25 else WHITE),
            )
        # Sectors
        for sec, val in sorted(s["sector_exposure"].items(), key=lambda x: -x[1]):
            wt = val / tv * 100
            dt.add_row(
                Text("SECTOR", style=PURPLE),
                Text(sec, style=WHITE),
                Text(_npr_k(val), style=WHITE),
                Text(f"{wt:.1f}%", style=f"bold {LOSS_HI}" if wt > 35 else WHITE),
            )

        # Winners / Losers table
        dt2 = self.query_one("#dt-winloss", DataTable)
        dt2.clear(columns=True)
        for label, key in [("", "tag"), ("SYMBOL", "sym"), ("P&L", "pnl"),
                           ("RTN%", "rtn"), ("DAYS", "days")]:
            dt2.add_column(label, key=key)

        for p in sorted(s["winners"], key=lambda x: -x["pnl"])[:5]:
            dt2.add_row(
                Text("▲ WIN", style=f"bold {GAIN_HI}"), _sym_text(p["sym"]),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                _dim_text(f"{p['days']}d"),
            )
        for p in sorted(s["losers"], key=lambda x: x["pnl"])[:5]:
            dt2.add_row(
                Text("▼ LOSS", style=f"bold {LOSS_HI}"), _sym_text(p["sym"]),
                _npr(p["pnl"]), _chg_text(p["ret"]),
                _dim_text(f"{p['days']}d"),
            )
        if not s["winners"] and not s["losers"]:
            dt2.add_row(_dim_text("—"), _dim_text("No positions"), *[Text("")] * 3)

    # ── Signals workspace ─────────────────────────────────────────────────────

    def _populate_signals_workspace(self, force: bool = False) -> None:
        cache_key = self._data_version()
        if (
            not force
            and cache_key == self._signals_workspace_cache_key
            and self._signals_workspace_cache_payload is not None
        ):
            self._render_signals_workspace_payload(cache_key, self._signals_workspace_cache_payload)
            return
        self._set_signals_workspace_loading()
        self._load_signals_workspace_async(cache_key)

    def _set_signals_workspace_loading(self) -> None:
        bar = self.query_one("#screener-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   [#888888]Loading screener, calendar and sector view...[/]"
        ))
        self.query_one("#screener-list-title", Static).update("ACTIVE STOCKS — Loading...")
        self.query_one("#heatmap-content", Static).update(Text("  Loading sector data...", style=LABEL))

    @work(thread=True)
    def _load_signals_workspace_async(self, cache_key: str) -> None:
        try:
            calendar_cols = [
                ("SYMBOL", "sym"), ("BOOK CLOSE", "bc"), ("DAYS", "days"),
                ("CASH%", "cash"), ("BONUS%", "bonus"), ("BUY BY", "buy"),
            ]
            calendar_rows: list[list[Text]] = []
            now = datetime.now()
            if self.md.corp.empty:
                calendar_rows.append([_dim_text("—"), _dim_text("No upcoming events"), *[Text("")] * 4])
            else:
                for _, r in self.md.corp.iterrows():
                    bc = r["bookclose_date"]
                    days = (bc - now).days
                    cash = float(r.get("cash_dividend_pct") or 0)
                    bonus = float(r.get("bonus_share_pct") or 0)
                    buy_by = (bc - timedelta(days=5)).strftime("%Y-%m-%d")
                    uc = f"bold {YELLOW}" if days <= 7 else (YELLOW if days <= 14 else WHITE)
                    calendar_rows.append([
                        _sym_text(str(r["symbol"])),
                        Text(bc.strftime("%Y-%m-%d"), style=WHITE),
                        Text(f"{days}d", style=uc),
                        Text(f"{cash:.1f}%", style=f"bold {GAIN_HI}") if cash >= 5 else _dim_text("—"),
                        Text(f"{bonus:.1f}%", style=f"bold {GAIN_HI}") if bonus >= 10 else _dim_text("—"),
                        Text(buy_by, style=CYAN),
                    ])

            screener_cols = [
                ("SYMBOL", "sym"), ("SECTOR", "sec"), ("LTP", "ltp"),
                ("CHG%", "chg"), ("VOL", "vol"), ("VRAT", "vrat"),
                ("RANGE", "range"), ("TREND", "spark"),
            ]
            screener_rows: list[list[Text]] = []
            heatmap = Text("  Loading sector data...", style=LABEL)
            status_markup = (
                f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   [#888888]Loading screener, calendar and sector view...[/]"
            )
            list_title = "ACTIVE STOCKS — Loading..."

            conn = None
            all_stocks: list[dict] = []
            sector_data: dict[str, dict] = {}
            MIN_VOL = 1000
            try:
                conn = _db()
                latest_date = conn.execute(
                    "SELECT MAX(date) FROM stock_prices WHERE symbol != 'NEPSE'"
                ).fetchone()[0]
                prev_date = None
                if latest_date:
                    prev_date = conn.execute(
                        "SELECT MAX(date) FROM stock_prices WHERE date < ? AND symbol != 'NEPSE'",
                        (latest_date,)
                    ).fetchone()[0]

                if latest_date and prev_date:
                    today_rows = conn.execute(
                        "SELECT symbol, open, high, low, close, volume FROM stock_prices "
                        "WHERE date=? AND symbol != 'NEPSE'",
                        (latest_date,)
                    ).fetchall()
                    prev_map = {
                        r[0]: float(r[1])
                        for r in conn.execute(
                            "SELECT symbol, close FROM stock_prices WHERE date=?",
                            (prev_date,)
                        ).fetchall()
                    }
                    avg_vol_map = {
                        r[0]: float(r[1]) if r[1] else 0.0
                        for r in conn.execute(
                            "SELECT symbol, AVG(volume) FROM stock_prices "
                            "WHERE date > date(?, '-30 days') AND symbol != 'NEPSE' "
                            "GROUP BY symbol",
                            (latest_date,)
                        ).fetchall()
                    }

                    from backend.backtesting.simple_backtest import get_symbol_sector

                    for r in today_rows:
                        sym = r[0]
                        vol = int(r[5])
                        close = float(r[4])
                        prev = prev_map.get(sym, 0.0)
                        chg = (close - prev) / prev * 100 if prev > 0 else 0.0
                        sector = get_symbol_sector(sym) or "Other"
                        avg_v = avg_vol_map.get(sym, 0.0)
                        vol_ratio = vol / avg_v if avg_v > 0 else 0.0
                        stock = {
                            "sym": sym,
                            "sector": sector,
                            "ltp": close,
                            "chg": chg,
                            "vol": vol,
                            "vol_ratio": vol_ratio,
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                        }
                        if vol > 0:
                            sector_bucket = sector_data.setdefault(
                                sector,
                                {"total_chg": 0.0, "count": 0, "total_vol": 0, "stocks": []},
                            )
                            sector_bucket["total_chg"] += chg
                            sector_bucket["count"] += 1
                            sector_bucket["total_vol"] += vol
                            sector_bucket["stocks"].append(stock)
                        if vol >= MIN_VOL:
                            all_stocks.append(stock)

                    spark_data: dict[str, list[float]] = {}
                    top_syms = [s["sym"] for s in sorted(all_stocks, key=lambda x: -x["vol"])[:120]]
                    for sym in top_syms:
                        hist = conn.execute(
                            "SELECT close FROM stock_prices WHERE symbol=? "
                            "ORDER BY date DESC LIMIT 20",
                            (sym,),
                        ).fetchall()
                        if len(hist) >= 3:
                            spark_data[sym] = [float(r[0]) for r in reversed(hist)]

                    if sector_data:
                        sector_perf = []
                        total_vol_all = sum(d["total_vol"] for d in sector_data.values())
                        for sec, data in sector_data.items():
                            avg_chg = data["total_chg"] / data["count"] if data["count"] > 0 else 0.0
                            vol_wt = data["total_vol"] / total_vol_all * 100 if total_vol_all > 0 else 0.0
                            sector_perf.append((sec, avg_chg, data["count"], data["total_vol"], vol_wt))
                        sector_perf.sort(key=lambda x: -x[1])

                        heatmap = Text()
                        heatmap.append("\n", style="")
                        max_abs = max(abs(s[1]) for s in sector_perf) if sector_perf else 1
                        if max_abs == 0:
                            max_abs = 1
                        for sec, avg_chg, count, total_vol, vol_wt in sector_perf:
                            if avg_chg > 2:
                                fg = "#00ff7f"
                            elif avg_chg > 1:
                                fg = "#00cc60"
                            elif avg_chg > 0:
                                fg = "#66cc66"
                            elif avg_chg > -1:
                                fg = "#cc9933"
                            elif avg_chg > -3:
                                fg = "#cc4444"
                            else:
                                fg = "#ff4545"
                            n_filled = max(1, int(abs(avg_chg) / max_abs * 8))
                            blocks = "█" * n_filled + "░" * (8 - n_filled)
                            heatmap.append(f"  {blocks} ", style=fg)
                            heatmap.append(f"{avg_chg:+5.1f}%  ", style=f"bold {fg}")
                            heatmap.append(f"{sec}", style=WHITE)
                            heatmap.append(f"  {count}\n", style=DIM)
                        total_stocks = sum(d["count"] for d in sector_data.values())
                        n_up = sum(1 for s in all_stocks if s["chg"] > 0)
                        n_dn = sum(1 for s in all_stocks if s["chg"] < 0)
                        heatmap.append(f"\n  {total_stocks} stocks  {_vol(total_vol_all)} vol  ", style=LABEL)
                        heatmap.append(f"▲{n_up}", style=GAIN_HI)
                        heatmap.append(f" ▼{n_dn}\n", style=LOSS_HI)

                    all_stocks.sort(key=lambda x: -x["vol"])
                    for s in all_stocks[:100]:
                        spark = (
                            _render_sparkline(spark_data.get(s["sym"], []), width=12)
                            if s["sym"] in spark_data
                            else Text("", style=DIM)
                        )
                        vr = s["vol_ratio"]
                        if vr > 2.5:
                            vr_text = Text(f"{vr:.1f}x", style=f"bold {GAIN_HI}")
                        elif vr > 1.5:
                            vr_text = Text(f"{vr:.1f}x", style=YELLOW)
                        elif vr > 0:
                            vr_text = Text(f"{vr:.1f}x", style=DIM)
                        else:
                            vr_text = Text("—", style=DIM)
                        rng = s["high"] - s["low"]
                        rng_pct = rng / s["low"] * 100 if s["low"] > 0 else 0
                        rng_text = (
                            Text(f"{s['low']:.0f}-{s['high']:.0f}", style=DIM)
                            if rng_pct > 0
                            else Text("—", style=DIM)
                        )
                        sec_display = "—" if s["sector"] == "Other" else s["sector"][:14]
                        screener_rows.append([
                            _sym_text(s["sym"]),
                            _dim_text(sec_display),
                            _price_text(s["ltp"]),
                            _chg_text(s["chg"]),
                            _vol_text(s["vol"]),
                            vr_text,
                            rng_text,
                            spark,
                        ])

                    n_stocks = len(all_stocks)
                    n_up = sum(1 for s in all_stocks if s["chg"] > 0)
                    n_down = sum(1 for s in all_stocks if s["chg"] < 0)
                    n_sectors = len([s for s in sector_data if s != "Other"])
                    status_markup = (
                        f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   "
                        f"[#888888]Active[/] [bold {WHITE}]{n_stocks}[/] [#555555](vol≥1K)[/]   "
                        f"[{GAIN_HI}]▲{n_up}[/]  [{LOSS_HI}]▼{n_down}[/]   "
                        f"[#888888]{n_sectors} sectors[/]   "
                        f"[#555555]Signals + screener merged  │  ENTER Lookup  │  / Command[/]"
                    )
                    list_title = (
                        f"ACTIVE STOCKS — {n_stocks}  │  Vol≥1K  │  Sorted by volume  │  VRAT=vol/20d avg"
                    )
            finally:
                if conn:
                    conn.close()

            # ── Broker floor signals ─────────────────────────────────────────
            # (label, key, width) — explicit widths for perfect alignment
            broker_cols = [
                ("SYMBOL",   "bsym",    8),
                ("SIGNAL",   "btype",  14),
                ("BUY HHI",  "bhhi_b",  8),
                ("SLR HHI",  "bhhi_s",  8),
                ("CIRC",     "bcirc",   6),
                ("SELF%",    "bself",   6),
                ("SM SCORE", "bscore",  9),
                ("TRADES",   "btr",     7),
            ]
            broker_rows: list[list[Text]] = []
            top_broker_cols: list = []
            top_broker_rows: list[list[Text]] = []
            broker_title = "BROKER FLOOR SIGNALS — Run floorsheet scraper to enable"
            try:
                import sqlite3 as _sqlite3
                from backend.quant_pro.database import get_db_path as _get_db_path
                _bconn = _sqlite3.connect(str(_get_db_path()))
                _bconn.row_factory = _sqlite3.Row
                # Check if broker_signals_v2 table has any data
                try:
                    _latest_bdate = (_bconn.execute(
                        "SELECT MAX(as_of_date) FROM broker_signals_v2"
                    ).fetchone() or [None])[0]
                except Exception:
                    _latest_bdate = None
                if False:  # broker signal computation not included in public release
                    _brows = _bconn.execute("""
                        SELECT b.symbol,
                               b.hhi_buy, b.hhi_sell,
                               b.circular_score, b.pump_score,
                               COALESCE(b.self_trade_pct, b.pump_score) AS self_trade_pct,
                               b.smart_money_score, b.n_trades,
                               m.micro_score
                        FROM broker_signals_v2 b
                        LEFT JOIN broker_microstructure m
                          ON b.symbol = m.symbol AND m.as_of_date = b.as_of_date
                        WHERE b.as_of_date = ?
                          AND (b.circular_score > 0.15 OR b.pump_score > 0.05 OR b.smart_money_score > 0.08)
                        ORDER BY
                            CASE WHEN b.circular_score > 0.15 OR b.pump_score > 0.05 THEN 0 ELSE 1 END ASC,
                            b.circular_score DESC,
                            b.smart_money_score DESC
                        LIMIT 60
                    """, (_latest_bdate,)).fetchall()
                    n_circ = sum(1 for _r in _brows if float(_r["circular_score"] or 0) > 0.15)
                    n_sm   = sum(1 for _r in _brows if float(_r["smart_money_score"] or 0) > 0.08)
                    broker_title = (
                        f"BROKER FLOOR  {_latest_bdate}  |"
                        f"  {n_circ} circular/pump  ·  {n_sm} smart money  ·  {len(_brows)} total"
                    )
                    for _r in _brows:
                        circ  = float(_r["circular_score"]  or 0.0)
                        self_ = float(_r["self_trade_pct"]  or 0.0)  # self-trade fraction
                        pump  = float(_r["pump_score"]       or 0.0)  # composite pump score
                        sm    = float(_r["smart_money_score"] or 0.0)
                        hhi_b = float(_r["hhi_buy"]          or 0.0)
                        hhi_s = float(_r["hhi_sell"]         or 0.0)
                        trades = int(_r["n_trades"]          or 0)
                        # Microstructure-adjusted score
                        _ms = _r["micro_score"]
                        micro = float(_ms) if _ms is not None else 0.5
                        display_score = sm * (0.5 + 0.5 * micro)
                        # Signal classification — self_ is now direct self-trade evidence
                        if self_ > 0.20 and circ > 0.15:
                            sig_label, sig_style = "[!] WASH+SELF ", f"bold {LOSS_HI}"
                        elif self_ > 0.10:
                            sig_label, sig_style = "[!] SELF-TRADE", LOSS_HI
                        elif circ > 0.15:
                            sig_label, sig_style = "[!] CIRCULAR  ", LOSS_HI
                        elif display_score > 0.25:
                            sig_label, sig_style = "[+] SMART $   ", f"bold {GAIN_HI}"
                        else:
                            sig_label, sig_style = "[~] ACCUM     ", GAIN_HI
                        broker_rows.append([
                            _sym_text(str(_r["symbol"])),
                            Text(sig_label, style=sig_style),
                            Text(f"{hhi_b:5.2f}", style=GAIN_HI if hhi_b > 0.30 else WHITE),
                            Text(f"{hhi_s:5.2f}", style=LOSS_HI if hhi_s > 0.20 else GAIN_HI),
                            Text(f"{circ*100:4.0f}%", style=LOSS_HI if circ > 0.15 else WHITE),
                            Text(f"{self_*100:4.0f}%", style=LOSS_HI if self_ > 0.10 else WHITE),
                            Text(f"{display_score:7.3f}" if display_score > 0 else "      —",
                                 style=GAIN_HI if display_score > 0.15 else WHITE),
                            _dim_text(f"{trades:6d}"),
                        ])
                # Top Brokers panel — populated by floorsheet scraper (not in public release)
                top_broker_cols = []
                top_broker_rows = []
                _bconn.close()
            except Exception as _be:
                broker_rows.append([_dim_text("—"), _dim_text("Run floorsheet scraper to populate broker data"), *[Text("")] * 6])
                top_broker_cols = []
                top_broker_rows = []

            payload = {
                "calendar_cols": calendar_cols,
                "calendar_rows": calendar_rows,
                "screener_cols": screener_cols,
                "screener_rows": screener_rows,
                "heatmap": heatmap,
                "status_markup": status_markup,
                "list_title": list_title,
                "broker_cols": broker_cols,
                "broker_rows": broker_rows,
                "broker_title": broker_title,
                "top_broker_cols": top_broker_cols,
                "top_broker_rows": top_broker_rows,
            }
            self.call_from_thread(self._render_signals_workspace_payload, cache_key, payload)
        except Exception as e:
            self.call_from_thread(self._set_status, f"Signals workspace error: {e}")

    def _render_signals_workspace_payload(self, cache_key: str, payload: dict) -> None:
        self._signals_workspace_cache_key = cache_key
        self._signals_workspace_cache_payload = payload

        dt_calendar = self.query_one("#dt-calendar", DataTable)
        dt_calendar.clear(columns=True)
        for label, key in payload["calendar_cols"]:
            dt_calendar.add_column(label, key=key)
        for row in payload["calendar_rows"]:
            dt_calendar.add_row(*row)

        dt_screener = self.query_one("#dt-screener", DataTable)
        dt_screener.clear(columns=True)
        for label, key in payload["screener_cols"]:
            dt_screener.add_column(label, key=key)
        for row in payload["screener_rows"]:
            dt_screener.add_row(*row)

        self.query_one("#heatmap-content", Static).update(payload["heatmap"])
        self.query_one("#screener-status-bar", Static).update(Text.from_markup(payload["status_markup"]))
        self.query_one("#screener-list-title", Static).update(payload["list_title"])

        try:
            dt_broker = self.query_one("#dt-broker-floor", DataTable)
            dt_broker.clear(columns=True)
            for label, key, width in payload.get("broker_cols") or []:
                dt_broker.add_column(label, key=key, width=width)
            self._apply_broker_floor_filter()
            broker_title = payload.get("broker_title") or "BROKER FLOOR SIGNALS"
            self.query_one("#broker-floor-title", Static).update(broker_title)
        except Exception:
            pass

        try:
            dt_top = self.query_one("#dt-broker-top", DataTable)
            dt_top.clear(columns=True)
            top_cols = payload.get("top_broker_cols") or []
            top_rows = payload.get("top_broker_rows") or []
            for label, key, width in top_cols:
                dt_top.add_column(label, key=key, width=width)
            for row in top_rows:
                dt_top.add_row(*row)
            # Update title with latest date
            broker_title = payload.get("broker_title") or ""
            date_part = broker_title.split("|")[0].replace("BROKER FLOOR", "").strip() if "|" in broker_title else ""
            self.query_one("#broker-top-title", Static).update(
                f"TOP BROKERS BY VOLUME{('  ' + date_part) if date_part else ''}"
            )
        except Exception:
            pass

    def _apply_broker_floor_filter(self) -> None:
        """Re-populate broker floor table from cached payload using current filter."""
        payload = getattr(self, "_signals_workspace_cache_payload", None) or {}
        all_rows = list(payload.get("broker_rows") or [])
        f = getattr(self, "_broker_floor_filter", "all")
        if f == "circ":
            rows = [r for r in all_rows if str(r[1]) and ("CIRC" in str(r[1]) or "WASH" in str(r[1]))]
        elif f == "pump":
            rows = [r for r in all_rows if str(r[1]) and ("SELF" in str(r[1]) or "WASH" in str(r[1]))]
        elif f == "smart":
            rows = [r for r in all_rows if str(r[1]) and ("SMART" in str(r[1]) or "ACCUM" in str(r[1]))]
        else:
            rows = all_rows
        try:
            dt_broker = self.query_one("#dt-broker-floor", DataTable)
            dt_broker.clear()
            for row in rows:
                dt_broker.add_row(*row)
        except Exception:
            pass

    # ── Calendar tab ──────────────────────────────────────────────────────────

    def _populate_calendar(self):
        dt = self.query_one("#dt-calendar", DataTable)
        dt.clear(columns=True)
        for label, key in [("SYMBOL", "sym"), ("BOOK CLOSE", "bc"), ("DAYS", "days"),
                           ("CASH%", "cash"), ("BONUS%", "bonus"), ("BUY BY", "buy")]:
            dt.add_column(label, key=key)
        now = datetime.now()
        if self.md.corp.empty:
            dt.add_row(_dim_text("—"), _dim_text("No upcoming events"), *[Text("")] * 4)
        else:
            for _, r in self.md.corp.iterrows():
                bc = r["bookclose_date"]; days = (bc - now).days
                cash = float(r.get("cash_dividend_pct") or 0)
                bonus = float(r.get("bonus_share_pct") or 0)
                buy_by = (bc - timedelta(days=5)).strftime("%Y-%m-%d")
                uc = f"bold {YELLOW}" if days <= 7 else (YELLOW if days <= 14 else WHITE)
                dt.add_row(
                    _sym_text(str(r["symbol"])),
                    Text(bc.strftime("%Y-%m-%d"), style=WHITE),
                    Text(f"{days}d", style=uc),
                    Text(f"{cash:.1f}%", style=f"bold {GAIN_HI}") if cash >= 5 else _dim_text("—"),
                    Text(f"{bonus:.1f}%", style=f"bold {GAIN_HI}") if bonus >= 10 else _dim_text("—"),
                    Text(buy_by, style=CYAN))

    # ── Gold/Silver Hedge Panel ───────────────────────────────────────────────

    def _render_hedge_panel(self) -> None:
        """Render the compact 4-line hedge strip in the Portfolio tab."""
        from backend.quant_pro.gold_hedge import get_gold_regime
        from backend.quant_pro.data_scrapers.gold_silver_ingestion import get_latest_nepal_metals
        from backend.quant_pro.database import get_db_path as _get_db_path
        import math as _math

        SEP   = "  [#2a3038]│[/]  "
        GOLD  = "#FFCA28"
        SILV  = "#90CAF9"

        def _ok(v):
            return v is not None and not (isinstance(v, float) and _math.isnan(v))

        def _chg(pct, abs_v):
            if not _ok(pct):
                return "[#3a3f45]—[/]"
            col  = "#2ebd6e" if pct >= 0 else "#e05050"
            sign = "+" if pct >= 0 else ""
            ab   = f" {sign}{abs_v:,.0f}" if _ok(abs_v) else ""
            return f"[{col}]{sign}{pct:.2f}%{ab}[/]"

        try:
            db      = str(_get_db_path())
            regime  = get_gold_regime(db)
            metals  = get_latest_nepal_metals(db)
            g_tola  = metals.get("gold_npr_tola") or 0
            s_tola  = metals.get("silver_npr_tola") or 0
            g_chg   = metals.get("gold_chg_pct")
            s_chg   = metals.get("silver_chg_pct")
            g_abs   = metals.get("gold_chg_abs")
            s_abs   = metals.get("silver_chg_abs")

            rname   = regime.get("regime", "no_data")
            mom     = regime.get("momentum_20d", 0) * 100
            rcol    = {"risk_off": "#E05050", "neutral": "#D1980B", "risk_on": "#2ebd6e"}.get(rname, "#606870")
            ricon   = {"risk_off": "▲ RISK-OFF", "neutral": "◉ NEUTRAL", "risk_on": "▼ RISK-ON"}.get(rname, "NO DATA")

            # ── Line 2: prices ────────────────────────────────────────────────
            info = (
                f"  [{rcol}]{ricon}[/]  [{rcol}]{mom:+.1f}%[/]"
                f"{SEP}[bold {GOLD}]GOLD[/]  NPR [bold]{g_tola:,.0f}[/]/tola  {_chg(g_chg, g_abs)}"
                f"{SEP}[bold {SILV}]SILVER[/]  NPR [bold]{s_tola:,.0f}[/]/tola  {_chg(s_chg, s_abs)}"
            )
            self.query_one("#hedge-info-bar", Static).update(info)

            # ── Line 3: recommendation ────────────────────────────────────────
            capital = getattr(self._trading_engine, "capital", None) if self._trading_engine else None
            hedge_on = self._hedge_enabled
            if not hedge_on:
                rec = f"  [#3a3f45]Hedge monitoring disabled — click [bold]● HEDGE ON[/] to enable[/]"
            elif rname == "risk_off":
                cap_str = f"  NPR {capital * 0.10:,.0f}" if (capital and capital > 0) else ""
                g_t = (capital * 0.10 * 0.70 / g_tola) if (capital and g_tola) else 0
                s_t = (capital * 0.10 * 0.30 / s_tola) if (capital and s_tola) else 0
                t_str = f"  →  [{GOLD}]{g_t:.2f}t[/] gold + [{SILV}]{s_t:.2f}t[/] silver" if (capital and capital > 0) else ""
                rec = f"  [bold #E05050]▲ HEDGE RECOMMENDED[/]  Withhold 10%{cap_str}{t_str}"
            elif rname == "neutral":
                rec = f"  [{rcol}]◉ Monitor — neutral regime, consider partial gold buffer[/]"
            else:
                rec = f"  [{rcol}]▼ Risk-on — full capital deployable, no hedge needed[/]"
            self.query_one("#hedge-rec-bar", Static).update(rec)

            # ── Line 4: hedge trade history summary ───────────────────────────
            trades = list(getattr(self, "_hedge_trade_log", []) or [])
            if trades:
                last = trades[-1]
                metal = str(last.get("metal", "GOLD"))
                col   = GOLD if metal == "GOLD" else SILV
                total_g = sum(float(t.get("total", 0)) for t in trades if t.get("metal") == "GOLD")
                total_s = sum(float(t.get("total", 0)) for t in trades if t.get("metal") == "SILVER")
                g_part  = f"[{GOLD}]GOLD NPR {total_g:,.0f}[/]" if total_g else ""
                s_part  = f"[{SILV}]SILVER NPR {total_s:,.0f}[/]" if total_s else ""
                invested = "  ".join(p for p in [g_part, s_part] if p)
                trade_bar = (
                    f"  [#3a3f45]HEDGE TRADES[/]  {len(trades)} recorded{SEP}"
                    f"Last  [{col}]{last.get('date','')}  {metal}  {float(last.get('tola',0)):.2f}t[/]"
                    f"{SEP}Total invested  {invested}"
                )
            else:
                trade_bar = (
                    f"  [#2a3038]HEDGE TRADES[/]  "
                    f"[#3a3f45]None recorded — physical gold/silver purchases logged here when executed[/]"
                )
            self.query_one("#hedge-trade-bar", Static).update(trade_bar)

        except Exception as e:
            try:
                self.query_one("#hedge-info-bar", Static).update(
                    f"  [#555555]Hedge data unavailable — {e}[/]"
                )
                self.query_one("#hedge-rec-bar", Static).update("")
                self.query_one("#hedge-trade-bar", Static).update("")
            except Exception:
                pass

    # ── Trades tab (full history) ─────────────────────────────────────────────

    def _populate_trades_full(self):
        dt = self.query_one("#dt-trades-full", DataTable)
        dt.clear(columns=True)

        if self.trade_mode == "live" and not self.tms_service:
            # Live mode but TMS not connected yet
            for label, key in [("STATUS", "st")]:
                dt.add_column(label, key=key)
            dt.add_row(Text("Connecting to TMS — solve captcha in browser...", style=AMBER))
            self.query_one("#trades-title", Static).update("TMS TRADE BOOK  |  Connecting...")
            return

        if self.trade_mode == "live" and self.tms_service:
            # Live mode: show TMS broker trades
            for label, key in [("DATE", "dt"), ("ACTION", "act"), ("SYMBOL", "sym"),
                               ("QTY", "qty"), ("PRICE", "pr"), ("AMOUNT", "amt"),
                               ("STATUS", "st")]:
                dt.add_column(label, key=key)
            try:
                bundle = getattr(self, '_tms_bundle', None)
                if not bundle:
                    bundle = _merge_tms_bundle_with_cache(
                        self.tms_service.executor.fetch_monitor_bundle()
                    )
                    self._tms_bundle = bundle
                trades = bundle.get("trades_daily", {}).get("records", [])
                trades += bundle.get("trades_historic", {}).get("records", [])
                if trades:
                    for t in trades[:50]:
                        action = str(t.get("type", t.get("buy_sell", t.get("side", "")))).upper()
                        ac = GAIN_HI if "BUY" in action else (LOSS_HI if "SELL" in action else WHITE)
                        dt.add_row(
                            _dim_text(str(t.get("date", t.get("trade_date", "")))[:10]),
                            Text(action[:4], style=f"bold {ac}"),
                            _sym_text(str(t.get("symbol", t.get("script", "")))),
                            Text(str(t.get("quantity", t.get("qty", ""))), style=WHITE),
                            _price_text(float(t.get("rate", t.get("price", 0)) or 0)),
                            Text(str(t.get("amount", t.get("total", ""))), style=WHITE),
                            _dim_text(str(t.get("status", ""))[:12]),
                        )
                    self.query_one("#trades-title", Static).update(
                        f"TMS TRADE BOOK  |  {len(trades)} trades")
                else:
                    dt.add_row(_dim_text("No broker trades"), *[Text("")] * 6)
                    self.query_one("#trades-title", Static).update("TMS TRADE BOOK")
            except Exception as e:
                dt.add_row(_dim_text(f"TMS error: {e}"), *[Text("")] * 6)
            return

        # Paper mode: show local trade log
        for label, key in [("DATE", "dt"), ("ACTION", "act"), ("SYMBOL", "sym"),
                           ("SHARES", "sh"), ("PRICE", "pr"), ("FEES", "fees"),
                           ("P&L", "pnl"), ("RTN%", "rtn"), ("REASON", "rsn")]:
            dt.add_column(label, key=key)
        # Merge all trade log sources for the current account: engine (auto), paper (manual), tui (engine csv)
        _tl_sources: list[pd.DataFrame] = []
        if self._trading_engine:
            _etl = self._trading_engine.get_trade_log()
            if not _etl.empty:
                _tl_sources.append(_etl)
        _acc_tl_dir = _account_dir(self._current_account_id)
        for _fname in ("paper_trade_log.csv", "tui_paper_trade_log.csv"):
            _tl_path = _acc_tl_dir / _fname
            if _tl_path.exists():
                try:
                    _df = pd.read_csv(_tl_path)
                    if not _df.empty:
                        _tl_sources.append(_df)
                except Exception:
                    pass
        if _tl_sources:
            tl = pd.concat(_tl_sources, ignore_index=True)
            _dedup_keys = [c for c in ("Date", "Action", "Symbol", "Shares", "Price") if c in tl.columns]
            if _dedup_keys:
                tl = tl.drop_duplicates(subset=_dedup_keys, keep="first")
            if "Date" in tl.columns:
                tl = tl.sort_values("Date").reset_index(drop=True)
        else:
            tl = _load_trade_log()
        if not tl.empty:
            for _, r in tl.iloc[::-1].iterrows():
                action = str(r.get("Action", ""))
                ac = GAIN_HI if action == "BUY" else (LOSS_HI if action == "SELL" else WHITE)
                pnl_v = float(r.get("PnL", 0) or 0)
                pnl_pct = float(r.get("PnL_Pct", 0) or 0)
                dt.add_row(
                    _dim_text(str(r.get("Date", ""))[:10]),
                    Text(action, style=f"bold {ac}"),
                    _sym_text(str(r.get("Symbol", ""))),
                    Text(str(int(r.get("Shares", 0))), style=WHITE),
                    _price_text(float(r.get("Price", 0))),
                    _dim_text(f"{float(r.get('Fees', 0)):.0f}"),
                    Text(f"{_npr_k(pnl_v)}", style=_pnl_color(pnl_v)) if pnl_v else _dim_text("—"),
                    _chg_text(pnl_pct * 100.0) if pnl_pct else _dim_text("—"),
                    _dim_text(str(r.get("Reason", ""))[:16]),
                )
        else:
            dt.add_row(_dim_text("No trades yet"), *[Text("")] * 8)
        # Append hedge trades (gold/silver physical purchases) at the top
        hedge_trades = list(reversed(getattr(self, "_hedge_trade_log", []) or []))
        GOLD_COL   = "#FFCA28"
        SILVER_COL = "#90CAF9"
        for ht in hedge_trades:
            metal = str(ht.get("metal", "GOLD"))
            col = GOLD_COL if metal == "GOLD" else SILVER_COL
            tola = float(ht.get("tola", 0) or 0)
            price = float(ht.get("price", 0) or 0)
            total = float(ht.get("total", 0) or 0)
            dt.add_row(
                _dim_text(str(ht.get("date", ""))[:10]),
                Text("HEDGE", style=f"bold {col}"),
                Text(metal, style=f"bold {col}"),
                Text(f"{tola:.2f}t", style=WHITE),
                _price_text(price),
                _dim_text("—"),
                Text(f"{_npr_k(total)}", style=col),
                _dim_text("—"),
                _dim_text(str(ht.get("reason", ""))[:16]),
            )
        if not tl.empty:
            buys = (tl["Action"] == "BUY").sum()
            sells = (tl["Action"] == "SELL").sum()
            total_pnl = tl["PnL"].sum() if "PnL" in tl.columns else 0
            wins = ((tl["PnL"] > 0) & (tl["Action"] == "SELL")).sum() if "PnL" in tl.columns else 0
            losses = ((tl["PnL"] < 0) & (tl["Action"] == "SELL")).sum() if "PnL" in tl.columns else 0
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            hedge_count = len(hedge_trades)
            hedge_note = f"  +{hedge_count} hedge" if hedge_count else ""
            self.query_one("#trades-title", Static).update(
                f"TRADE HISTORY  |  {buys} buys  {sells} sells{hedge_note}  |  "
                f"Win rate: {wr:.0f}%  |  Total P&L: {_npr_k(total_pnl)}")

    # ── Lookup tab ────────────────────────────────────────────────────────────

    def _populate_lookup(self):
        sym = str(self.lookup_sym or "").strip().upper()
        if not sym:
            return
        tf = self.lookup_tf
        cache_key = (sym, tf, self._data_version())
        self._lookup_request_key = cache_key
        if cache_key in self._lookup_cache:
            self._render_lookup_payload(cache_key, self._lookup_cache[cache_key])
            return

        chart_w = self.query_one("#lookup-chart", Static)
        pane_width = max(60, int(getattr(chart_w.size, "width", 0) or (self.size.width - 6)))
        pane_height = max(14, int(getattr(chart_w.size, "height", 0) or 15))
        self._set_lookup_loading(sym, tf)
        self._load_lookup_async(sym, tf, cache_key[2], pane_width, pane_height)

    def _set_lookup_loading(self, sym: str, tf: str) -> None:
        tf_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
        self.query_one("#lookup-title", Static).update(
            f"LOOKUP: {sym}  —  {tf_labels.get(tf, 'Daily')}  —  Loading..."
        )
        self.query_one("#lookup-header", Static).update(Text("  Loading price and report data...", style=LABEL))
        self.query_one("#lookup-chart", Static).update(Text("  Rendering chart...", style=LABEL))
        self.query_one("#lookup-stats", Static).update(Text("  Loading statistics...", style=LABEL))
        self.query_one("#lookup-report", Static).update(Text("  Building financial report...", style=LABEL))
        self.query_one("#lookup-intel-title", Static).update(f"CORPORATE INTELLIGENCE — {sym}")
        self.query_one("#lookup-intel", Static).update(Text("  Loading intelligence...", style=LABEL))
        self.query_one("#lookup-fin-title", Static).update(f"QUARTERLY FINANCIALS — {sym}")
        self.query_one("#lookup-ca-title", Static).update(f"CORPORATE ACTIONS — {sym}")
        for table_id in ("#dt-lookup", "#dt-lookup-fin", "#dt-lookup-ca"):
            self.query_one(table_id, DataTable).clear(columns=True)

    @work(thread=True)
    def _load_lookup_async(
        self,
        sym: str,
        tf: str,
        data_version: str,
        pane_width: int,
        pane_height: int,
    ) -> None:
        cache_key = (sym, tf, data_version)
        try:
            min_sessions = 2 if tf in {"D", "I"} else {"W": 10, "M": 18, "Y": 24}.get(tf, 2)
            _ensure_lookup_history(sym, min_sessions=min_sessions)
            limits = {"D": 120, "W": 365, "M": 730, "Y": 2500, "I": 60}
            det = self.md.detail(sym, limit=limits.get(tf, 120))
            if not det:
                payload = {"found": False, "sym": sym}
                self.call_from_thread(self._render_lookup_payload, cache_key, payload)
                return

            lat = det["lat"]
            chg = det["chg"]
            h = det["h"]
            ltp = self.md.ltps().get(sym, float(lat["close"]))
            tf_labels = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly", "I": "Intraday"}
            chart_source = h
            stats_source = h
            count_label = "sessions"
            intraday_note = ""

            if tf == "I":
                try:
                    from backend.trading.live_trader import is_market_open, now_nst
                    market_open = bool(is_market_open())
                    nst_now = now_nst()
                except Exception:
                    market_open = True
                    nst_now = datetime.utcnow() + timedelta(hours=5, minutes=45)

                if market_open:
                    intraday_rows, intraday_session_date, intraday_snapshots = _load_intraday_ohlcv(
                        sym,
                        preferred_session_date=_nst_today_str(),
                    )
                    if not intraday_rows.empty:
                        chart_source = intraday_rows
                        stats_source = intraday_rows
                        count_label = "bars"
                        if intraday_session_date:
                            intraday_note = f"  ({intraday_session_date} · {intraday_snapshots} snapshots)"
                    else:
                        intraday_note = "  (no intraday snapshots — showing daily)"
                else:
                    intraday_note = (
                        f"  (market closed at {nst_now.strftime('%H:%M')} NST — showing daily)"
                    )

            n_sessions = len(chart_source)
            title = (
                f"LOOKUP: {sym}  —  {tf_labels.get(tf, 'Daily')}  "
                f"({n_sessions} {count_label})  —  "
                f"D/W/M/Y/I to switch  —  Press L to change{intraday_note}"
            )
            header = Text.assemble(
                (f"  {sym}  ", f"bold {CYAN}"),
                ("LTP ", LABEL), (f"{ltp:.1f} ", f"bold {WHITE}"),
                _chg_text(chg),
                (f"   O {lat['open']:.1f}  H {lat['high']:.1f}  "
                 f"L {lat['low']:.1f}  Vol {_vol(lat['volume'])}", LABEL),
            )

            try:
                chart_rows = _resample_ohlcv(chart_source, tf).sort_values("date").reset_index(drop=True)
                ideal_width = 18 + max(12, len(chart_rows)) * 3
                cw = max(60, min(pane_width - 2, ideal_width))
                ch_h = max(12, pane_height - 1)
                chart = _render_candlestick_chart(chart_source, width=cw, height=ch_h, timeframe=tf)
            except Exception as e:
                chart = Text(f"  Chart error: {e}", style=LABEL)

            h_display = _resample_ohlcv(stats_source, tf) if tf in ("W", "M", "Y") else stats_source
            h_asc = h_display.sort_values("date")
            closes = h_asc["close"].tolist()
            vols = h_asc["volume"].tolist()
            avg_vol = sum(vols) / len(vols) if vols else 0
            hi_30 = float(h_display["high"].max())
            lo_30 = float(h_display["low"].min())
            avg_close = float(h_display["close"].mean())
            volatility = float(h_display["close"].pct_change().std() * 100) if len(h_display) > 2 else 0
            rng_pct = (hi_30 - lo_30) / lo_30 * 100 if lo_30 else 0

            stats = Text()
            stats.append("  MARKET SNAPSHOT\n", style=f"bold {AMBER}")
            stats.append("  HIGH ", style=LABEL); stats.append(f"{hi_30:.1f}   ", style=GAIN_HI)
            stats.append("LOW ", style=LABEL); stats.append(f"{lo_30:.1f}   ", style=LOSS_HI)
            stats.append("AVG ", style=LABEL); stats.append(f"{avg_close:.1f}\n", style=WHITE)
            stats.append("  AVG VOL ", style=LABEL); stats.append(f"{_vol(avg_vol)}   ", style=CYAN)
            stats.append("VOL ", style=LABEL); stats.append(f"{volatility:.2f}%   ", style=YELLOW)
            stats.append("RANGE ", style=LABEL); stats.append(f"{rng_pct:.1f}%\n", style=WHITE)
            stats.append("  PRICE ", style=LABEL)
            stats.append_text(_render_sparkline(closes, width=26))
            stats.append("\n", style=WHITE)
            stats.append("  VOLUME", style=LABEL)
            stats.append_text(_render_sparkline(vols, width=26))

            fin_title = f"QUARTERLY FINANCIALS — {sym}"
            fin_cols: list[tuple[str, str]] = []
            fin_rows: list[list[Text]] = []
            try:
                fin_report = build_stock_report(sym, current_price=ltp)
                report = _render_stock_report(fin_report)
                fin_cols = [
                    ("PERIOD", "period"), ("REVENUE", "revenue"), ("NET PROFIT", "net_profit"),
                    ("EPS", "eps"), ("BVPS", "book_value"),
                ]
                rows = fin_report.get("financial_rows") or []
                if rows:
                    for row in rows:
                        fin_rows.append([
                            _dim_text(row.get("period", "—")),
                            Text(str(row.get("revenue", "—")), style=CYAN),
                            Text(str(row.get("net_profit", "—")), style=WHITE),
                            Text(str(row.get("eps", "—")), style=GAIN_HI),
                            Text(str(row.get("book_value", "—")), style=WHITE),
                        ])
                else:
                    fin_rows.append([
                        _dim_text("—"),
                        _dim_text("No cached financials"),
                        _dim_text("Run scraper"),
                        _dim_text("—"),
                        _dim_text("—"),
                    ])
            except Exception as e:
                fin_report = {}
                report = Text(f"  Financial report error: {e}", style=LABEL)
                fin_cols = [("STATUS", "status")]
                fin_rows = [[_dim_text("Financial report unavailable")]]

            intel_title = f"CORPORATE INTELLIGENCE — {sym}"
            intel = _render_lookup_intelligence(fin_report, sym)

            tf_col_label = {"D": "DATE", "W": "WEEK", "M": "MONTH", "Y": "YEAR", "I": "TIME"}
            ohlcv_title = f"OHLCV — {sym} ({tf_labels.get(tf, 'Daily')})"
            ohlcv_cols = [
                (tf_col_label.get(tf, "DATE"), "dt"), ("OPEN", "o"), ("HIGH", "h"),
                ("LOW", "l"), ("CLOSE", "c"), ("VOL", "vol"), ("CHG%", "chg"),
            ]
            ohlcv_rows: list[list[Text]] = []
            h_sorted = h_display.sort_values("date").reset_index(drop=True)
            chg_map: dict[str, Optional[float]] = {}
            for i in range(len(h_sorted)):
                date_key = str(h_sorted.iloc[i]["date"])[:10]
                if i == 0:
                    chg_map[date_key] = None
                else:
                    prev_close = float(h_sorted.iloc[i - 1]["close"])
                    curr_close = float(h_sorted.iloc[i]["close"])
                    chg_map[date_key] = (curr_close - prev_close) / prev_close * 100 if prev_close > 0 else None
            for _, r in h_display.iterrows():
                date_key = str(r["date"])[:10]
                ct = _chg_text(chg_map.get(date_key)) if chg_map.get(date_key) is not None else _dim_text("—")
                ohlcv_rows.append([
                    _dim_text(date_key),
                    _price_text(r["open"]),
                    _price_text(r["high"]),
                    _price_text(r["low"]),
                    _price_text(r["close"]),
                    _vol_text(r["volume"]),
                    ct,
                ])

            ca = det["ca"]
            ca_title = f"CORPORATE ACTIONS — {sym}" if not ca.empty else f"CORPORATE ACTIONS — {sym} — None"
            ca_cols = [("BOOK CLOSE", "bc"), ("CASH DIV%", "cash"), ("BONUS%", "bonus")]
            ca_rows: list[list[Text]] = []
            if not ca.empty:
                for _, r in ca.iterrows():
                    ca_rows.append([
                        Text(str(r["bookclose_date"])[:10], style=WHITE),
                        Text(f"{r['cash_dividend_pct']:.1f}%", style=GAIN_HI)
                        if pd.notna(r.get("cash_dividend_pct")) and r["cash_dividend_pct"]
                        else _dim_text("—"),
                        Text(f"{r['bonus_share_pct']:.1f}%", style=GAIN_HI)
                        if pd.notna(r.get("bonus_share_pct")) and r["bonus_share_pct"]
                        else _dim_text("—"),
                    ])

            payload = {
                "found": True,
                "title": title,
                "header": header,
                "chart": chart,
                "stats": stats,
                "report": report,
                "fin_title": fin_title,
                "fin_cols": fin_cols,
                "fin_rows": fin_rows,
                "intel_title": intel_title,
                "intel": intel,
                "ohlcv_title": ohlcv_title,
                "ohlcv_cols": ohlcv_cols,
                "ohlcv_rows": ohlcv_rows,
                "ca_title": ca_title,
                "ca_cols": ca_cols,
                "ca_rows": ca_rows,
            }
            self.call_from_thread(self._render_lookup_payload, cache_key, payload)
        except Exception as e:
            self.call_from_thread(
                self._render_lookup_payload,
                cache_key,
                {"found": False, "sym": sym, "error": str(e)},
            )

    def _render_lookup_payload(self, cache_key: tuple[str, str, str], payload: dict) -> None:
        if cache_key != self._lookup_request_key:
            return
        self._lookup_cache[cache_key] = payload
        while len(self._lookup_cache) > 12:
            oldest_key = next(iter(self._lookup_cache))
            self._lookup_cache.pop(oldest_key, None)

        title_w = self.query_one("#lookup-title", Static)
        header_w = self.query_one("#lookup-header", Static)
        chart_w = self.query_one("#lookup-chart", Static)
        stats_w = self.query_one("#lookup-stats", Static)
        report_w = self.query_one("#lookup-report", Static)
        summary_scroll = self.query_one("#lookup-summary-pane", VerticalScroll)
        intel_title = self.query_one("#lookup-intel-title", Static)
        intel_w = self.query_one("#lookup-intel", Static)
        dt = self.query_one("#dt-lookup", DataTable)
        fin_title = self.query_one("#lookup-fin-title", Static)
        fin_dt = self.query_one("#dt-lookup-fin", DataTable)
        ca_title = self.query_one("#lookup-ca-title", Static)
        ca_dt = self.query_one("#dt-lookup-ca", DataTable)

        if not payload.get("found"):
            sym = payload.get("sym", self.lookup_sym)
            title_w.update(f"LOOKUP: {sym} — NOT FOUND")
            header_w.update("")
            chart_w.update("")
            stats_w.update("")
            report_w.update(Text(f"  {payload.get('error', '')}", style=LABEL) if payload.get("error") else "")
            dt.clear(columns=True)
            fin_title.update("")
            fin_dt.clear(columns=True)
            ca_title.update("")
            ca_dt.clear(columns=True)
            intel_title.update("")
            intel_w.update("")
            return

        title_w.update(payload["title"])
        header_w.update(payload["header"])
        chart_w.update(payload["chart"])
        stats_w.update(payload["stats"])
        report_w.update(payload["report"])
        intel_title.update(payload["intel_title"])
        intel_w.update(payload["intel"])

        self.query_one("#lookup-ohlcv-title", Static).update(payload["ohlcv_title"])
        dt.clear(columns=True)
        for label, key in payload["ohlcv_cols"]:
            dt.add_column(label, key=key)
        for row in payload["ohlcv_rows"]:
            dt.add_row(*row)

        fin_title.update(payload["fin_title"])
        fin_dt.clear(columns=True)
        for label, key in payload["fin_cols"]:
            fin_dt.add_column(label, key=key)
        for row in payload["fin_rows"]:
            fin_dt.add_row(*row)

        ca_title.update(payload["ca_title"])
        ca_dt.clear(columns=True)
        if payload["ca_rows"]:
            for label, key in payload["ca_cols"]:
                ca_dt.add_column(label, key=key)
            for row in payload["ca_rows"]:
                ca_dt.add_row(*row)

        try:
            summary_scroll.scroll_home(animate=False)
            summary_scroll.focus()
        except Exception:
            pass


    # ── Command Palette ────────────────────────────────────────────────────────

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), callback=self._on_command)

    def _on_command(self, result: dict | None) -> None:
        if not result:
            return
        kind = result.get("kind", "")
        target = result.get("target", "")
        if kind == "tab":
            self.action_tab(target)
        elif kind == "action":
            if target == "buy":
                self.action_buy()
            elif target == "sell":
                self.action_sell()
            elif target == "refresh":
                self.action_refresh()
            elif target == "agent":
                self.action_run_agent()
        elif kind == "stock":
            self.lookup_sym = target
            self.action_tab("lookup")

    # ── Watchlist ─────────────────────────────────────────────────────────────

    def action_watchlist_add(self) -> None:
        if self.active_tab == "kalimati":
            entry = self._watchlist_entry_from_rates_selection()
            if not entry:
                self._set_status("Select a commodity, macro price, or forex row to add")
                return
            self._add_local_watchlist_entry(entry)
            return
        self.push_screen(WatchlistAddScreen(), callback=self._on_watchlist_add)

    def _on_watchlist_add(self, item: dict | str | None) -> None:
        if not item:
            return
        if isinstance(item, dict):
            normalized = _normalize_watchlist_entry(item)
            if not normalized:
                self._set_status("Unable to add watchlist item")
                return
            if str(normalized.get("kind") or "stock") == "stock" and self.trade_mode == "live" and self.tms_service:
                sym = str(normalized.get("symbol") or normalized.get("label") or "").upper()
                if sym:
                    self._watchlist_add_live(sym)
                return
            self._add_local_watchlist_entry(normalized)
            return
        sym = str(item).upper()
        if self.trade_mode == "live" and self.tms_service:
            self._watchlist_add_live(sym)
            return
        self._add_local_watchlist_entry(_stock_watchlist_entry(sym))

    def action_watchlist_remove(self) -> None:
        """Remove the currently selected row from watchlist."""
        if self.active_tab != "watchlist":
            return
        try:
            focused = getattr(self, "focused", None)
            focused_id = getattr(focused, "id", "") if focused is not None else ""
            if focused_id == "dt-watchlist-rates":
                dt_rates = self.query_one("#dt-watchlist-rates", DataTable)
                idx = dt_rates.cursor_row
                rows = getattr(self, "_watchlist_rates_rows", [])
                if idx is not None and 0 <= idx < len(rows) and rows[idx].get("tracked"):
                    label = str(rows[idx].get("label") or "")
                    kind = str(rows[idx].get("kind") or "")
                    self._remove_local_watchlist_entry({"key": f"{kind}:{label}", "label": label})
                return
            if focused_id == "dt-watchlist-commodities":
                dt_commodities = self.query_one("#dt-watchlist-commodities", DataTable)
                idx = dt_commodities.cursor_row
                rows = getattr(self, "_watchlist_commodity_rows", [])
                if idx is not None and 0 <= idx < len(rows) and rows[idx].get("tracked"):
                    label = str(rows[idx].get("label") or "")
                    self._remove_local_watchlist_entry({"key": f"commodity:{label}", "label": label})
                return
            dt = self.query_one("#dt-watchlist", DataTable)
            row_idx = dt.cursor_row
            stock_rows = getattr(self, "_watchlist_stock_rows", [])
            if row_idx is not None and 0 <= row_idx < len(stock_rows):
                entry = stock_rows[row_idx]
                if str(entry.get("kind") or "stock") == "stock" and self.trade_mode == "live" and self.tms_service:
                    sym = str(entry.get("symbol") or entry.get("label") or "").upper()
                    if sym:
                        self._watchlist_remove_live(sym)
                    return
                self._remove_local_watchlist_entry(entry)
        except Exception:
            pass

    def _add_local_watchlist_entry(self, entry: dict) -> None:
        normalized = _normalize_watchlist_entry(entry)
        if not normalized:
            self._set_status("Unable to add watchlist item")
            return
        existing = {_watchlist_entry_key(item) for item in self._paper_watchlist}
        key = _watchlist_entry_key(normalized)
        label = str(normalized.get("label") or normalized.get("symbol") or key)
        if key in existing:
            self._set_status(f"{label} already in watchlist")
            return
        self._paper_watchlist.append(normalized)
        _save_watchlist(self._paper_watchlist)
        if self.trade_mode != "live" or str(normalized.get("kind") or "stock") != "stock":
            self._watchlist = list(self._paper_watchlist)
        self._populate_watchlist()
        self._set_status(f"Added {label} to watchlist")

    def _remove_local_watchlist_entry(self, entry: dict) -> None:
        key = _watchlist_entry_key(entry)
        label = str(entry.get("label") or entry.get("symbol") or key)
        updated = [item for item in self._paper_watchlist if _watchlist_entry_key(item) != key]
        if len(updated) == len(self._paper_watchlist):
            self._set_status(f"{label} not found in local watchlist")
            return
        self._paper_watchlist = updated
        _save_watchlist(self._paper_watchlist)
        self._populate_watchlist()
        self._set_status(f"Removed {label} from watchlist")

    def _watchlist_entry_from_rates_selection(self) -> Optional[dict]:
        filtered_rows, filtered_indicators, filtered_forex = self._get_filtered_rates_payload()
        try:
            dt = self.query_one("#dt-kalimati", DataTable)
            row_idx = dt.cursor_row
            if row_idx is not None and 0 <= row_idx < len(filtered_rows):
                row = filtered_rows[row_idx]
                label = str(row.get("name_english") or "").strip()
                return {
                    "kind": "commodity",
                    "key": f"commodity:{label}",
                    "label": label,
                    "unit": str(row.get("unit") or ""),
                }
        except Exception:
            pass
        try:
            dt = self.query_one("#dt-macro", DataTable)
            row_idx = dt.cursor_row
            if row_idx is not None and 0 <= row_idx < len(filtered_indicators):
                row = filtered_indicators[row_idx]
                label = str(row.get("item") or "").strip()
                return {
                    "kind": "macro",
                    "key": f"macro:{label}",
                    "label": label,
                    "group": str(row.get("group") or ""),
                }
        except Exception:
            pass
        try:
            dt = self.query_one("#dt-forex", DataTable)
            row_idx = dt.cursor_row
            if row_idx is not None and 0 <= row_idx < len(filtered_forex):
                row = filtered_forex[row_idx]
                code = str(row.get("currency_code") or "").strip().upper()
                return {
                    "kind": "forex",
                    "key": f"forex:{code}",
                    "label": code,
                    "currency_name": str(row.get("currency_name") or ""),
                }
        except Exception:
            pass
        return None

    def _set_watchlist_from_tms_snapshot(self, snapshot: dict) -> bool:
        if not isinstance(snapshot, dict):
            return False
        raw_items = snapshot.get("items")
        if raw_items is None:
            return False
        symbols: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            sym = str((item or {}).get("symbol") or "").strip().upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            symbols.append(sym)
        self._live_watchlist = [_stock_watchlist_entry(sym) for sym in symbols]
        local_extras = [item for item in self._paper_watchlist if str(item.get("kind") or "stock") != "stock"]
        self._watchlist = list(self._live_watchlist) + local_extras
        bundle = dict(getattr(self, "_tms_bundle", None) or {})
        bundle["watchlist"] = snapshot
        self._tms_bundle = bundle
        try:
            save_tms_snapshot("tms_watchlist", dict(snapshot), status="ok")
        except Exception:
            pass
        return True

    @work(thread=True)
    def _watchlist_add_live(self, sym: str) -> None:
        self.app.call_from_thread(self._set_status, f"TMS monitor  |  Adding {sym} to broker watchlist...")
        try:
            result = build_tui_control_plane(self).sync_watchlist(action="add", symbol=sym)
            snapshot = dict(result.payload or {})
            self.app.call_from_thread(self._set_watchlist_from_tms_snapshot, snapshot)
            self.app.call_from_thread(self._populate_watchlist)
            self.app.call_from_thread(self._set_status, f"Added {sym} to TMS watchlist")
        except Exception as e:
            self.app.call_from_thread(self._set_status, self._summarize_tms_watchlist_error(e, "add"))

    @work(thread=True)
    def _watchlist_remove_live(self, sym: str) -> None:
        self.app.call_from_thread(self._set_status, f"TMS monitor  |  Removing {sym} from broker watchlist...")
        try:
            result = build_tui_control_plane(self).sync_watchlist(action="remove", symbol=sym)
            snapshot = dict(result.payload or {})
            self.app.call_from_thread(self._set_watchlist_from_tms_snapshot, snapshot)
            self.app.call_from_thread(self._populate_watchlist)
            self.app.call_from_thread(self._set_status, f"Removed {sym} from TMS watchlist")
        except Exception as e:
            self.app.call_from_thread(self._set_status, self._summarize_tms_watchlist_error(e, "remove"))

    @work(thread=True)
    def _refresh_watchlist_live(self, force: bool = False) -> None:
        if self.trade_mode != "live" or not self.tms_service:
            return
        if self._tms_watchlist_refresh_inflight:
            return
        now = time.monotonic()
        if not force and (now - float(getattr(self, "_last_tms_watchlist_fetch_at", 0.0) or 0.0)) < 90.0:
            return
        self._tms_watchlist_refresh_inflight = True
        try:
            self.app.call_from_thread(self._set_status, "TMS monitor  |  Syncing broker watchlist...")
            result = build_tui_control_plane(self).sync_watchlist(action="fetch")
            snapshot = dict(result.payload or {})
            self._last_tms_watchlist_fetch_at = time.monotonic()
            self.app.call_from_thread(self._set_watchlist_from_tms_snapshot, snapshot)
            self.app.call_from_thread(self._populate_watchlist)
            symbols = snapshot.get("symbols") or []
            self.app.call_from_thread(
                self._set_status,
                f"TMS monitor  |  Broker watchlist synced ({len(symbols)} symbols)",
            )
        except Exception as e:
            self._last_tms_watchlist_fetch_at = time.monotonic()
            self.app.call_from_thread(self._set_status, self._summarize_tms_watchlist_error(e, "sync"))
        finally:
            self._tms_watchlist_refresh_inflight = False

    def _populate_watchlist(self) -> None:
        """Populate watchlist DataTable with watched stocks and macro items."""
        dt = self.query_one("#dt-watchlist", DataTable)
        dt.clear(columns=True)
        for label, key in [("  #", "n"), ("ITEM", "sym"), ("VALUE", "ltp"),
                           ("CHG%", "chg"), ("OPEN", "open"), ("HIGH", "high"),
                           ("LOW", "low"), ("VOLUME", "vol"), ("TREND", "spark"),
                           ("SIGNAL", "sig")]:
            dt.add_column(label, key=key)

        watchlist_source = "LOCAL"
        held_count = 0
        if self.trade_mode == "live" and self.tms_service:
            if self._set_watchlist_from_tms_snapshot((self._tms_bundle or {}).get("watchlist")):
                watchlist_source = "TMS"
            elif self._live_watchlist:
                local_extras = [item for item in self._paper_watchlist if str(item.get("kind") or "stock") != "stock"]
                self._watchlist = list(self._live_watchlist) + local_extras
                watchlist_source = "CACHE"
            else:
                self._watchlist = [item for item in self._paper_watchlist if str(item.get("kind") or "stock") != "stock"]
                watchlist_source = "TMS"
        else:
            self._watchlist, held_count = self._effective_paper_watchlist()
            if held_count:
                watchlist_source = "LOCAL+HELD"

        if not self._watchlist:
            dt.add_row(_dim_text("—"), _dim_text("Press = to add stocks"),
                       *[Text("")] * 8)
            bar = self.query_one("#wl-status-bar", Static)
            bar.update(Text.from_markup(
                f"[bold {AMBER}]◆ WATCHLIST[/]   "
                f"[#888888]Tracking[/] [bold {WHITE}]0[/] [#888888]items[/]   "
                f"[#888888]Source[/] [bold {WHITE}]{watchlist_source}[/]   "
                f"[#555555]= Add  │  Sync holdings from Portfolio profile  │  - Remove  │  ENTER Open[/]"
            ))
            self._populate_watchlist_side_panels([], [], {}, {})
            return

        kalimati_rows = list(getattr(self, "_kalimati_rows", []) or [])
        macro_rates = dict(getattr(self, "_macro_rates", {}) or {})

        if any(str(item.get("kind") or "stock") != "stock" for item in self._watchlist):
            if not kalimati_rows:
                self._load_kalimati_async()
            if not macro_rates:
                self._load_macro_rates_async()

        ltps = self.md.ltps() if hasattr(self, 'md') else {}
        kalimati_by_name = {str(row.get("name_english") or ""): row for row in kalimati_rows}
        macro_by_label = {str(row.get("item") or ""): row for row in macro_rates.get("indicators", [])}
        forex_by_code = {str(row.get("currency_code") or "").upper(): row for row in macro_rates.get("forex_rows", [])}
        port = load_port()
        stock_entries = [item for item in self._watchlist if str(item.get("kind") or "stock") == "stock"]
        rates_entries = [item for item in self._watchlist if str(item.get("kind") or "") in ("forex", "macro")]
        commodity_entries = [item for item in self._watchlist if str(item.get("kind") or "") == "commodity"]
        self._watchlist_stock_rows = list(stock_entries)
        self.query_one("#wl-main-title", Static).update(f"STOCK WATCHLIST [{len(stock_entries)}]")

        # Build price data for each watchlist symbol
        conn = None
        try:
            conn = _db()
        except Exception:
            pass

        gainers = 0
        losers = 0

        def _watch_pct_text(pct: Optional[float]) -> Text:
            if pct is None:
                return _dim_text("—")
            return _chg_text(float(pct))

        def _watch_range_bar(mn: float, mx: float, avg: float, width: int = 15) -> Text:
            if mx <= mn:
                return Text("—", style=LABEL)
            pos = int((avg - mn) / (mx - mn) * width)
            pos = max(0, min(width - 1, pos))
            bar = "░" * pos + "█" + "░" * (width - pos - 1)
            color = GAIN if avg >= ((mn + mx) / 2) else LOSS
            return Text(bar, style=color)

        for i, entry in enumerate(stock_entries, 1):
            kind = str(entry.get("kind") or "stock")
            sym = str(entry.get("symbol") or entry.get("label") or "").upper()
            ltp = ltps.get(sym, 0) if kind == "stock" else 0
            chg_val = 0.0
            open_p = high_p = low_p = 0.0
            vol = 0
            sparkline = Text("—", style=LABEL)
            signal_text = Text("", style=DIM)

            if kind == "stock" and conn:
                try:
                    row = conn.execute(
                        "SELECT open, high, low, close, volume FROM stock_prices "
                        "WHERE symbol=? ORDER BY date DESC LIMIT 1", (sym,)).fetchone()
                    if row:
                        open_p, high_p, low_p = float(row[0]), float(row[1]), float(row[2])
                        if ltp <= 0:
                            ltp = float(row[3])
                        vol = int(row[4])
                    # Previous close for change %
                    rows2 = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 2", (sym,)).fetchall()
                    if len(rows2) >= 2:
                        prev = float(rows2[1][0])
                        if prev > 0:
                            chg_val = (ltp - prev) / prev * 100

                    # Sparkline from last 30 days
                    hist = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 30", (sym,)).fetchall()
                    if len(hist) >= 3:
                        closes = [float(r[0]) for r in reversed(hist)]
                        sparkline = _render_sparkline(closes, width=15)
                except Exception:
                    pass

            if kind == "stock":
                if not port.empty and sym in port["Symbol"].values:
                    signal_text = Text("● HELD", style=f"bold {CYAN}")
                if chg_val > 0:
                    gainers += 1
                elif chg_val < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    _sym_text(sym),
                    _price_text(ltp) if ltp > 0 else _dim_text("—"),
                    _chg_text(chg_val),
                    _price_text(open_p) if open_p > 0 else _dim_text("—"),
                    _price_text(high_p) if high_p > 0 else _dim_text("—"),
                    _price_text(low_p) if low_p > 0 else _dim_text("—"),
                    _vol_text(vol) if vol > 0 else _dim_text("—"),
                    sparkline,
                    signal_text,
                )
                continue

            label = str(entry.get("label") or "")
            if kind == "commodity":
                row = kalimati_by_name.get(label) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=WHITE),
                    Text(f"{float(row.get('avg') or 0):,.1f}", style=f"bold {AMBER}") if row else _dim_text("—"),
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(f"{float(row.get('min') or 0):,.1f}", style=DIM) if row else _dim_text("—"),
                    Text(f"{float(row.get('max') or 0):,.1f}", style=DIM) if row else _dim_text("—"),
                    Text(str(row.get("unit") or entry.get("unit") or "")[:8], style=LABEL),
                    Text("KALIMATI", style=DIM),
                    _watch_range_bar(float(row.get("min") or 0), float(row.get("max") or 0), float(row.get("avg") or 0), width=15) if row else Text("—", style=LABEL),
                    Text("● COMMODITY", style=f"bold {CYAN}"),
                )
                continue

            if kind == "macro":
                row = macro_by_label.get(label) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                value = float(row.get("value") or 0)
                unit = str(row.get("unit") or "")
                if unit.startswith("NPR"):
                    value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}") if value > 0 else _dim_text("—")
                else:
                    value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}") if value > 0 else _dim_text("—")
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=WHITE),
                    value_text,
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(unit[:10], style=LABEL) if unit else _dim_text("—"),
                    Text(str(row.get("group") or entry.get("group") or "")[:10], style=CYAN),
                    _dim_text("—"),
                    Text(str(row.get("source") or "")[:10], style=DIM),
                    Text("—", style=LABEL),
                    Text("● MACRO", style=f"bold {CYAN}"),
                )
                continue

            if kind == "forex":
                row = forex_by_code.get(label.upper()) or {}
                pct = row.get("change_pct")
                if pct is not None and float(pct) > 0:
                    gainers += 1
                elif pct is not None and float(pct) < 0:
                    losers += 1
                dt.add_row(
                    _dim_text(f"{i:2d}"),
                    Text(label[:18], style=f"bold {CYAN}"),
                    Text(f"{float(row.get('buy_rate') or 0):,.2f}", style=WHITE) if row else _dim_text("—"),
                    _watch_pct_text(float(pct)) if pct is not None else _dim_text("—"),
                    Text(f"{float(row.get('sell_rate') or 0):,.2f}", style=WHITE) if row else _dim_text("—"),
                    Text(str(row.get("currency_name") or entry.get("currency_name") or "")[:10], style=WHITE),
                    Text(str(row.get("unit") or 1), style=LABEL) if row else _dim_text("—"),
                    Text(str(row.get("source") or "NRB")[:10], style=DIM),
                    Text("—", style=LABEL),
                    Text("● FOREX", style=f"bold {CYAN}"),
                )

        if not stock_entries:
            dt.add_row(
                _dim_text("—"),
                _dim_text("No stock symbols"),
                *[Text("")] * 8
            )

        if conn:
            conn.close()

        self._populate_watchlist_side_panels(
            rates_entries,
            commodity_entries,
            macro_rates,
            {
                "kalimati_by_name": kalimati_by_name,
                "macro_by_label": macro_by_label,
                "forex_by_code": forex_by_code,
            },
        )

        # Status bar
        n = len(self._watchlist)
        bar = self.query_one("#wl-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ WATCHLIST[/]   "
            f"[#888888]Tracking[/] [bold {WHITE}]{n}[/] [#888888]items[/]   "
            f"[#888888]Stocks[/] [bold {WHITE}]{len(stock_entries)}[/]   "
            f"[#888888]Rates[/] [bold {CYAN}]{len(rates_entries)}[/]   "
            f"[#888888]Commodities[/] [bold {AMBER}]{len(commodity_entries)}[/]   "
            f"[#888888]Up[/] [bold {GAIN_HI}]{gainers}[/]   "
            f"[#888888]Down[/] [bold {LOSS_HI}]{losers}[/]   "
            f"[#888888]Source[/] [bold {WHITE}]{watchlist_source}[/]   "
            f"[#888888]Held[/] [bold {CYAN}]{held_count}[/]   "
            f"[#555555]= Add  │  Sync holdings from Portfolio profile  │  - Remove  │  ENTER Open[/]"
        ))

    def _populate_watchlist_side_panels(
        self,
        rates_entries: list[dict],
        commodity_entries: list[dict],
        macro_rates: dict,
        resolved_maps: dict,
    ) -> None:
        forex_by_code = dict(resolved_maps.get("forex_by_code") or {})
        macro_by_label = dict(resolved_maps.get("macro_by_label") or {})
        kalimati_by_name = dict(resolved_maps.get("kalimati_by_name") or {})

        if not forex_by_code and not macro_by_label and not macro_rates:
            self._load_macro_rates_async()
        if not kalimati_by_name:
            self._load_kalimati_async()

        dt_rates = self.query_one("#dt-watchlist-rates", DataTable)
        dt_rates.clear(columns=True)
        dt_rates.add_column("ITEM", key="item", width=16)
        dt_rates.add_column("VALUE", key="value", width=13)
        dt_rates.add_column("CHG%", key="chg", width=8)
        dt_rates.add_column("UNIT", key="unit", width=10)

        rate_rows: list[dict] = []
        if rates_entries:
            for entry in rates_entries:
                kind = str(entry.get("kind") or "")
                if kind == "forex":
                    row = forex_by_code.get(str(entry.get("label") or "").upper())
                    rate_rows.append({"kind": "forex", "tracked": True, "label": str(entry.get("label") or ""), "row": row})
                elif kind == "macro":
                    row = macro_by_label.get(str(entry.get("label") or ""))
                    rate_rows.append({"kind": "macro", "tracked": True, "label": str(entry.get("label") or ""), "row": row})
        self._watchlist_rates_rows = rate_rows

        def _pct_cell(pct: Optional[float]) -> Text:
            if pct is None:
                return _dim_text("—")
            return _chg_text(float(pct))

        if rate_rows:
            for item in rate_rows:
                row = item.get("row") or {}
                if item["kind"] == "forex":
                    value = float(row.get('buy_rate') or 0)
                    value_text = Text(f"{value:,.2f}", style=WHITE) if value > 0 else Text("Loading", style=DIM)
                    unit_text = Text(str(row.get("unit") or 1)[:10], style=LABEL)
                else:
                    value = float(row.get("value") or 0)
                    unit = str(row.get("unit") or "")
                    if value <= 0:
                        value_text = Text("Loading", style=DIM)
                    elif unit.startswith("NPR"):
                        value_text = Text(_format_compact_npr(value), style=f"bold {AMBER}")
                    else:
                        value_text = Text(f"{value:,.2f} {unit}".strip(), style=f"bold {AMBER}")
                    unit_text = Text(unit[:10], style=LABEL) if unit else _dim_text("—")
                label_style = f"bold {CYAN}" if item["kind"] == "forex" else WHITE
                dt_rates.add_row(
                    Text(str(item["label"])[:16], style=label_style),
                    value_text,
                    _pct_cell(row.get("change_pct")),
                    unit_text,
                )
        else:
            dt_rates.add_row(_dim_text("—"), _dim_text("No tracked rates"), Text(""), Text(""))
        self.query_one("#wl-rates-title", Static).update(f"FOREX & MACRO [{len(rate_rows)}]")

        dt_commodities = self.query_one("#dt-watchlist-commodities", DataTable)
        dt_commodities.clear(columns=True)
        dt_commodities.add_column("ITEM", key="item", width=18)
        dt_commodities.add_column("AVG", key="avg", width=10)
        dt_commodities.add_column("CHG%", key="chg", width=8)
        dt_commodities.add_column("UNIT", key="unit", width=8)

        commodity_rows: list[dict] = []
        if commodity_entries:
            for entry in commodity_entries:
                row = kalimati_by_name.get(str(entry.get("label") or ""))
                commodity_rows.append({"tracked": True, "label": str(entry.get("label") or ""), "row": row})
        self._watchlist_commodity_rows = commodity_rows

        if commodity_rows:
            for item in commodity_rows:
                row = item.get("row") or {}
                pct = row.get("change_pct")
                dt_commodities.add_row(
                    Text(str(item["label"])[:18], style=WHITE),
                    Text(f"{float(row.get('avg') or 0):,.1f}", style=f"bold {AMBER}") if float(row.get('avg') or 0) > 0 else Text("Loading", style=DIM),
                    _pct_cell(float(pct)) if pct is not None else _dim_text("—"),
                    Text(str(row.get("unit") or "")[:8], style=LABEL) if row else _dim_text("—"),
                )
        else:
            dt_commodities.add_row(_dim_text("—"), _dim_text("No tracked commodities"), Text(""), Text(""))
        self.query_one("#wl-commodities-title", Static).update(f"COMMODITIES [{len(commodity_rows)}]")

    # ── Screener ──────────────────────────────────────────────────────────────

    def _populate_screener(self) -> None:
        """Populate sector treemap and stock screener."""
        heatmap_w = self.query_one("#heatmap-content", Static)
        dt = self.query_one("#dt-screener", DataTable)
        dt.clear(columns=True)

        for label, key in [("SYMBOL", "sym"), ("SECTOR", "sec"), ("LTP", "ltp"),
                           ("CHG%", "chg"), ("VOL", "vol"), ("VRAT", "vrat"),
                           ("RANGE", "range"), ("TREND", "spark")]:
            dt.add_column(label, key=key)

        # Gather all stock data
        conn = None
        all_stocks = []
        sector_data = {}  # sector -> {total_chg, count, symbols, total_vol, stocks}
        MIN_VOL = 1000  # filter out illiquid noise
        try:
            conn = _db()
            latest_date = conn.execute(
                "SELECT MAX(date) FROM stock_prices WHERE symbol != 'NEPSE'"
            ).fetchone()[0]
            if not latest_date:
                heatmap_w.update(Text("No data", style=LABEL))
                return
            prev_date = conn.execute(
                "SELECT MAX(date) FROM stock_prices WHERE date < ? AND symbol != 'NEPSE'",
                (latest_date,)
            ).fetchone()[0]

            if latest_date and prev_date:
                today_rows = conn.execute(
                    "SELECT symbol, open, high, low, close, volume FROM stock_prices "
                    "WHERE date=? AND symbol != 'NEPSE'",
                    (latest_date,)
                ).fetchall()
                prev_map = {}
                for r in conn.execute(
                    "SELECT symbol, close FROM stock_prices WHERE date=?",
                    (prev_date,)
                ).fetchall():
                    prev_map[r[0]] = float(r[1])

                # Get 20-day avg volume for vol ratio
                avg_vol_map = {}
                for r in conn.execute(
                    "SELECT symbol, AVG(volume) FROM stock_prices "
                    "WHERE date > date(?, '-30 days') AND symbol != 'NEPSE' "
                    "GROUP BY symbol", (latest_date,)
                ).fetchall():
                    avg_vol_map[r[0]] = float(r[1]) if r[1] else 0

                from backend.backtesting.simple_backtest import get_symbol_sector

                for r in today_rows:
                    sym = r[0]
                    vol = int(r[5])
                    close = float(r[4])
                    prev = prev_map.get(sym, 0)
                    chg = (close - prev) / prev * 100 if prev > 0 else 0
                    sector = get_symbol_sector(sym) or "Other"
                    avg_v = avg_vol_map.get(sym, 0)
                    vol_ratio = vol / avg_v if avg_v > 0 else 0

                    stock = {
                        "sym": sym, "sector": sector, "ltp": close,
                        "chg": chg, "vol": vol, "vol_ratio": vol_ratio,
                        "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                    }
                    # Include all for sector stats, but only liquid ones in table
                    if vol > 0:
                        if sector not in sector_data:
                            sector_data[sector] = {"total_chg": 0, "count": 0,
                                                   "total_vol": 0, "stocks": []}
                        sector_data[sector]["total_chg"] += chg
                        sector_data[sector]["count"] += 1
                        sector_data[sector]["total_vol"] += vol
                        sector_data[sector]["stocks"].append(stock)
                    if vol >= MIN_VOL:
                        all_stocks.append(stock)

                # Get sparklines for all displayed stocks
                spark_data = {}
                top_syms = [s["sym"] for s in sorted(all_stocks, key=lambda x: -x["vol"])[:120]]
                for sym in top_syms:
                    hist = conn.execute(
                        "SELECT close FROM stock_prices WHERE symbol=? "
                        "ORDER BY date DESC LIMIT 20", (sym,)
                    ).fetchall()
                    if len(hist) >= 3:
                        spark_data[sym] = [float(r[0]) for r in reversed(hist)]
        except Exception:
            pass
        finally:
            if conn:
                conn.close()

        # ── Sector Performance ──
        if sector_data:
            sector_perf = []
            total_vol_all = sum(d["total_vol"] for d in sector_data.values())
            for sec, data in sector_data.items():
                avg_chg = data["total_chg"] / data["count"] if data["count"] > 0 else 0
                vol_wt = data["total_vol"] / total_vol_all * 100 if total_vol_all > 0 else 0
                sector_perf.append((sec, avg_chg, data["count"], data["total_vol"], vol_wt))
            sector_perf.sort(key=lambda x: -x[1])  # best performing first

            heatmap = Text()
            heatmap.append("\n", style="")

            # Colored block indicators: ■ = sector health at a glance
            max_abs = max(abs(s[1]) for s in sector_perf) if sector_perf else 1
            if max_abs == 0:
                max_abs = 1

            for sec, avg_chg, count, total_vol, vol_wt in sector_perf:
                # Pick color from smooth gradient
                if avg_chg > 2:
                    fg = "#00ff7f"
                elif avg_chg > 1:
                    fg = "#00cc60"
                elif avg_chg > 0:
                    fg = "#66cc66"
                elif avg_chg > -1:
                    fg = "#cc9933"
                elif avg_chg > -3:
                    fg = "#cc4444"
                else:
                    fg = "#ff4545"

                # Filled/empty blocks proportional to magnitude
                n_filled = max(1, int(abs(avg_chg) / max_abs * 8))
                blocks = "█" * n_filled + "░" * (8 - n_filled)

                heatmap.append(f"  {blocks} ", style=fg)
                heatmap.append(f"{avg_chg:+5.1f}%  ", style=f"bold {fg}")
                heatmap.append(f"{sec}", style=WHITE)
                heatmap.append(f"  {count}\n", style=DIM)

            # Summary line
            total_stocks = sum(d["count"] for d in sector_data.values())
            n_up = sum(1 for s in all_stocks if s["chg"] > 0)
            n_dn = sum(1 for s in all_stocks if s["chg"] < 0)
            heatmap.append(f"\n  {total_stocks} stocks  {_vol(total_vol_all)} vol  ", style=LABEL)
            heatmap.append(f"▲{n_up}", style=GAIN_HI)
            heatmap.append(f" ▼{n_dn}\n", style=LOSS_HI)
            heatmap_w.update(heatmap)
        else:
            heatmap_w.update(Text("  Loading sector data...", style=LABEL))

        # ── Stock list (liquid stocks only, sorted by volume) ──
        all_stocks.sort(key=lambda x: -x["vol"])
        for s in all_stocks[:100]:
            spark = _render_sparkline(spark_data.get(s["sym"], []), width=12) \
                if s["sym"] in spark_data else Text("", style=DIM)
            # Vol ratio indicator
            vr = s["vol_ratio"]
            if vr > 2.5:
                vr_text = Text(f"{vr:.1f}x", style=f"bold {GAIN_HI}")
            elif vr > 1.5:
                vr_text = Text(f"{vr:.1f}x", style=YELLOW)
            elif vr > 0:
                vr_text = Text(f"{vr:.1f}x", style=DIM)
            else:
                vr_text = Text("—", style=DIM)
            # Day range as compact string
            rng = s["high"] - s["low"]
            rng_pct = rng / s["low"] * 100 if s["low"] > 0 else 0
            rng_text = Text(f"{s['low']:.0f}-{s['high']:.0f}", style=DIM) if rng_pct > 0 \
                else Text("—", style=DIM)
            sec_display = s["sector"]
            if sec_display == "Other":
                sec_display = "—"
            dt.add_row(
                _sym_text(s["sym"]),
                _dim_text(sec_display[:14]),
                _price_text(s["ltp"]),
                _chg_text(s["chg"]),
                _vol_text(s["vol"]),
                vr_text,
                rng_text,
                spark,
            )

        # Status bar
        n_stocks = len(all_stocks)
        n_up = sum(1 for s in all_stocks if s["chg"] > 0)
        n_down = sum(1 for s in all_stocks if s["chg"] < 0)
        n_sectors = len([s for s in sector_data if s != "Other"])
        bar = self.query_one("#screener-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold {AMBER}]◆ SIGNALS WORKSPACE[/]   "
            f"[#888888]Active[/] [bold {WHITE}]{n_stocks}[/] [#555555](vol≥1K)[/]   "
            f"[{GAIN_HI}]▲{n_up}[/]  [{LOSS_HI}]▼{n_down}[/]   "
            f"[#888888]{n_sectors} sectors[/]   "
            f"[#555555]Signals + screener merged  │  ENTER Lookup  │  / Command[/]"
        ))
        self.query_one("#screener-list-title", Static).update(
            f"ACTIVE STOCKS — {n_stocks}  │  Vol≥1K  │  Sorted by volume  │  VRAT=vol/20d avg")


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    NepseDashboard().run()


if __name__ == "__main__":
    main()
