from __future__ import annotations

import copy
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.long_term import LONG_TERM_CONFIG
from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import ensure_dir, get_project_root

PROJECT_ROOT = get_project_root(__file__)
STRATEGY_REGISTRY_DIR = ensure_dir(PROJECT_ROOT / "data" / "strategy_registry")
BUILTIN_STRATEGY_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "builtin")
CUSTOM_STRATEGY_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "custom")
BACKTEST_RESULTS_DIR = ensure_dir(STRATEGY_REGISTRY_DIR / "backtests")
COMPARISON_LATEST_JSON = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.json"
COMPARISON_LATEST_CSV = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.csv"
COMPARISON_LATEST_PNG = BACKTEST_RESULTS_DIR / "registry_strategies_vs_nepse_latest.png"


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_c31_config() -> Dict[str, Any]:
    config = copy.deepcopy(LONG_TERM_CONFIG)
    config.setdefault("use_trailing_stop", True)
    config.setdefault("profit_target_pct", None)
    config.setdefault("event_exit_mode", False)
    return config


def _temp_forward_winner_config() -> Dict[str, Any]:
    return {
        "holding_days": 45,
        "max_positions": 5,
        "signal_types": ["quality", "quarterly_fundamental", "xsec_momentum"],
        "rebalance_frequency": 5,
        "stop_loss_pct": 0.06,
        "trailing_stop_pct": 0.15,
        "use_regime_filter": True,
        "sector_limit": 0.35,
        "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 2},
        "bear_threshold": -0.05,
        "initial_capital": float(LONG_TERM_CONFIG.get("initial_capital") or 1_000_000.0),
        "regime_sector_limits": {"bull": 0.5, "neutral": 0.35, "bear": 0.25},
        "profit_target_pct": None,
        "event_exit_mode": False,
        "use_trailing_stop": True,
    }


def _builtin_payloads() -> List[Dict[str, Any]]:
    return []


def ensure_builtin_strategies() -> None:
    for payload in _builtin_payloads():
        path = BUILTIN_STRATEGY_DIR / f"{payload['id']}.json"
        _json_write(path, payload)


def _load_strategy_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("id", path.stem)
    payload.setdefault("name", payload["id"])
    payload.setdefault("description", "")
    payload.setdefault("source", "custom" if path.parent == CUSTOM_STRATEGY_DIR else "builtin")
    payload.setdefault("editable", payload.get("source") != "builtin")
    payload.setdefault("runner_mode", "temp_patched")
    payload.setdefault("execution_mode", "paper_runtime")
    payload.setdefault("config", {})
    payload.setdefault("ranking_overlay", {"mode": "baseline"})
    payload["_path"] = str(path)
    return payload


def list_strategies() -> List[Dict[str, Any]]:
    ensure_builtin_strategies()
    rows: List[Dict[str, Any]] = []
    for base in (BUILTIN_STRATEGY_DIR, CUSTOM_STRATEGY_DIR):
        for path in sorted(base.glob("*.json")):
            payload = _load_strategy_file(path)
            if payload:
                rows.append(payload)
    rows.sort(key=lambda item: (0 if str(item.get("source")) == "builtin" else 1, str(item.get("name") or item.get("id") or "").lower()))
    return rows


def load_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    sid = str(strategy_id or "").strip()
    if not sid:
        return None
    ensure_builtin_strategies()
    for base in (CUSTOM_STRATEGY_DIR, BUILTIN_STRATEGY_DIR):
        path = base / f"{sid}.json"
        if path.exists():
            return _load_strategy_file(path)
    return None


def strategy_name(strategy_id: str) -> str:
    payload = load_strategy(strategy_id)
    if payload:
        return str(payload.get("name") or strategy_id)
    return str(strategy_id or "").strip()


def default_strategy_for_account(account_id: str) -> str:
    """Return the default strategy ID for an account. All accounts start with the C5 baseline."""
    return "default_c5"


def ensure_account_strategy_ids(accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ensure_builtin_strategies()
    updated: List[Dict[str, Any]] = []
    for account in list(accounts or []):
        row = dict(account or {})
        if not str(row.get("strategy_id") or "").strip():
            row["strategy_id"] = default_strategy_for_account(str(row.get("id") or ""))
        updated.append(row)
    return updated


def _slugify(token: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(token or "").strip().lower()).strip("_")
    return cleaned or "strategy"


def save_custom_strategy(payload: Dict[str, Any], *, strategy_id: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
    ensure_builtin_strategies()
    now = _timestamp()
    sid = _slugify(strategy_id or payload.get("id") or payload.get("name") or now)
    path = CUSTOM_STRATEGY_DIR / f"{sid}.json"
    existing = _load_strategy_file(path) if path.exists() else None
    if path.exists() and not overwrite:
        suffix = 2
        while (CUSTOM_STRATEGY_DIR / f"{sid}_{suffix}.json").exists():
            suffix += 1
        sid = f"{sid}_{suffix}"
        path = CUSTOM_STRATEGY_DIR / f"{sid}.json"
        existing = None
    record = {
        "id": sid,
        "name": str(payload.get("name") or sid),
        "description": str(payload.get("description") or "").strip(),
        "source": "custom",
        "editable": True,
        "runner_mode": str(payload.get("runner_mode") or "temp_patched"),
        "execution_mode": str(payload.get("execution_mode") or "paper_runtime"),
        "config": copy.deepcopy(payload.get("config") or {}),
        "ranking_overlay": copy.deepcopy(payload.get("ranking_overlay") or {"mode": "baseline"}),
        "created_at": str((existing or {}).get("created_at") or now),
        "updated_at": now,
        "notes": copy.deepcopy(payload.get("notes") or {}),
    }
    _json_write(path, record)
    return load_strategy(sid) or record


def _pct(value: float) -> float:
    return round(float(value) * 100.0, 4)


def _daily_nav_payload(result: Any) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for date_value, nav in list(getattr(result, "daily_nav", []) or []):
        if hasattr(date_value, "strftime"):
            date_out = date_value.strftime("%Y-%m-%d")
        else:
            date_out = str(date_value)[:10]
        rows.append([date_out, float(nav)])
    return rows


def _summarize_backtest_result(strategy_id: str, description: str, result: Any) -> Dict[str, Any]:
    final_nav = (
        float(result.daily_nav[-1][1])
        if getattr(result, "daily_nav", None)
        else float(getattr(result, "initial_capital", 0.0) or 0.0)
    )
    return {
        "id": str(strategy_id),
        "description": str(description),
        "total_return_pct": _pct(getattr(result, "total_return", 0.0)),
        "annualized_return_pct": _pct(getattr(result, "annualized_return", 0.0)),
        "volatility_pct": _pct(getattr(result, "volatility", 0.0)),
        "sharpe": round(float(getattr(result, "sharpe_ratio", 0.0) or 0.0), 4),
        "sortino": round(float(getattr(result, "sortino_ratio", 0.0) or 0.0), 4),
        "max_drawdown_pct": _pct(getattr(result, "max_drawdown", 0.0)),
        "max_drawdown_duration": int(getattr(result, "max_drawdown_duration", 0) or 0),
        "calmar": round(float(getattr(result, "calmar_ratio", 0.0) or 0.0), 4),
        "total_trades": int(getattr(result, "total_trades", 0) or 0),
        "win_rate_pct": _pct(getattr(result, "win_rate", 0.0)),
        "avg_win_pct": _pct(getattr(result, "avg_win", 0.0)),
        "avg_loss_pct": _pct(getattr(result, "avg_loss", 0.0)),
        "profit_factor": round(float(getattr(result, "profit_factor", 0.0) or 0.0), 4),
        "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0) or 0),
        "avg_holding_days": round(float(getattr(result, "avg_holding_days", 0.0) or 0.0), 2),
        "total_pnl": round(float(getattr(result, "total_pnl", 0.0) or 0.0), 2),
        "total_fees_paid": round(float(getattr(result, "total_fees_paid", 0.0) or 0.0), 2),
        "final_nav": round(final_nav, 2),
        "by_signal_type": getattr(result, "by_signal_type", lambda: {})(),
        "by_exit_reason": getattr(result, "by_exit_reason", lambda: {})(),
        "daily_nav": _daily_nav_payload(result),
    }


def _nepse_return(start_date: str, end_date: str) -> Dict[str, Any]:
    conn = sqlite3.connect(str(get_db_path()))
    try:
        rows = conn.execute(
            """
            SELECT date, close
            FROM stock_prices
            WHERE symbol = 'NEPSE'
              AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            (str(start_date), str(end_date)),
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < 2:
        return {
            "start": str(start_date),
            "end": str(end_date),
            "start_close": None,
            "end_close": None,
            "return_pct": 0.0,
        }

    start_row = rows[0]
    end_row = rows[-1]
    start_close = float(start_row[1])
    end_close = float(end_row[1])
    return_pct = ((end_close / start_close) - 1.0) * 100.0 if start_close > 0 else 0.0
    return {
        "start": str(start_row[0])[:10],
        "end": str(end_row[0])[:10],
        "start_close": round(start_close, 4),
        "end_close": round(end_close, 4),
        "return_pct": round(return_pct, 4),
    }


def run_strategy_backtest(strategy: Dict[str, Any], *, start_date: str, end_date: str, capital: float) -> Dict[str, Any]:
    from backend.backtesting.simple_backtest import run_backtest

    config = copy.deepcopy(strategy.get("config") or {})
    config["initial_capital"] = float(capital)
    result = run_backtest(start_date=start_date, end_date=end_date, **config)
    summary = _summarize_backtest_result(
        str(strategy.get("id") or "strategy"),
        str(strategy.get("description") or strategy.get("name") or "Strategy backtest"),
        result,
    )
    nepse = _nepse_return(start_date, end_date)
    summary["vs_nepse_pct_points"] = round(float(summary["total_return_pct"]) - float(nepse.get("return_pct") or 0.0), 4)

    payload = {
        "strategy": {
            "id": str(strategy.get("id") or ""),
            "name": str(strategy.get("name") or ""),
            "runner_mode": str(strategy.get("runner_mode") or "temp_patched"),
            "execution_mode": str(strategy.get("execution_mode") or "paper_runtime"),
            "config": copy.deepcopy(strategy.get("config") or {}),
            "ranking_overlay": copy.deepcopy(strategy.get("ranking_overlay") or {"mode": "baseline"}),
        },
        "window": {
            "start": str(start_date),
            "end": str(end_date),
            "capital": float(capital),
        },
        "nepse": nepse,
        "summary": summary,
        "generated_at": _timestamp(),
    }
    _json_write(BACKTEST_RESULTS_DIR / f"{strategy['id']}_latest.json", payload)
    return payload


def load_strategy_comparison_snapshot() -> Optional[Dict[str, Any]]:
    if not COMPARISON_LATEST_JSON.exists():
        return None
    try:
        payload = json.loads(COMPARISON_LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def comparison_metrics_for_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    snapshot = load_strategy_comparison_snapshot()
    if not snapshot:
        return None
    target = str(strategy_id or "").strip()
    for row in list(snapshot.get("strategies") or []):
        if str(row.get("id") or "") == target:
            metrics = dict(row.get("summary") or {})
            metrics["window"] = dict(snapshot.get("window") or {})
            metrics["nepse"] = dict(snapshot.get("nepse") or {})
            return metrics
    return None
