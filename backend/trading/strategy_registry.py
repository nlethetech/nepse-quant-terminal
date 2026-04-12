from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.quant_pro.paths import ensure_dir, get_project_root, get_runtime_dir

PROJECT_ROOT = get_project_root(__file__)
RUNTIME_DIR = ensure_dir(get_runtime_dir(__file__))
BUILTIN_STRATEGY_DIR = PROJECT_ROOT / "configs" / "strategies"
STRATEGY_RUNTIME_DIR = ensure_dir(RUNTIME_DIR / "strategy_registry")
BACKTEST_RESULTS_DIR = ensure_dir(STRATEGY_RUNTIME_DIR / "backtests")


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    payload.setdefault("source", "builtin")
    payload.setdefault("editable", False)
    payload.setdefault("config", {})
    payload["_path"] = str(path)
    return payload


def list_strategies() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(BUILTIN_STRATEGY_DIR.glob("*.json")):
        payload = _load_strategy_file(path)
        if payload:
            rows.append(payload)
    rows.sort(key=lambda item: str(item.get("name") or item.get("id") or "").lower())
    return rows


def load_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    sid = str(strategy_id or "").strip().lower()
    if not sid:
        return None
    path = BUILTIN_STRATEGY_DIR / f"{sid}.json"
    if not path.exists():
        return None
    return _load_strategy_file(path)


def strategy_name(strategy_id: str) -> str:
    payload = load_strategy(strategy_id)
    if payload:
        return str(payload.get("name") or strategy_id)
    return str(strategy_id or "").strip()


def default_strategy_for_account(account_id: str) -> str:
    return "sat06" if str(account_id or "").strip() == "account_2" else "c5"


def ensure_account_strategy_ids(accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    known = {str(item.get("id") or "") for item in list_strategies()}
    for account in list(accounts or []):
        row = dict(account or {})
        strategy_id = str(row.get("strategy_id") or "").strip().lower()
        if strategy_id not in known:
            row["strategy_id"] = default_strategy_for_account(str(row.get("id") or ""))
        updated.append(row)
    return updated


def _latest_backtest_path(strategy_id: str) -> Path:
    return BACKTEST_RESULTS_DIR / f"{str(strategy_id).strip().lower()}_latest.json"


def load_latest_backtest(strategy_id: str) -> Optional[Dict[str, Any]]:
    path = _latest_backtest_path(strategy_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_backtest_summary(strategy_id: str) -> Optional[Dict[str, Any]]:
    payload = load_latest_backtest(strategy_id)
    if not payload:
        return None
    summary = dict(payload.get("summary") or {})
    summary["window"] = dict(payload.get("window") or {})
    summary["nepse"] = dict(payload.get("nepse") or {})
    return summary


def run_strategy_backtest(strategy: Dict[str, Any], *, start_date: str, end_date: str, capital: float) -> Dict[str, Any]:
    from datetime import datetime as _dt

    from backend.backtesting.simple_backtest import run_backtest
    from backend.trading.live_trader import compute_nepse_benchmark_return

    config = copy.deepcopy(strategy.get("config") or {})
    config["initial_capital"] = float(capital)
    result = run_backtest(start_date=start_date, end_date=end_date, **config)

    start = _dt.strptime(start_date, "%Y-%m-%d").date()
    end = _dt.strptime(end_date, "%Y-%m-%d").date()
    nepse = compute_nepse_benchmark_return(start, end_date=end) or {"return_pct": 0.0}

    summary = {
        "total_return_pct": round(float(result.total_return) * 100.0, 4),
        "sharpe_ratio": round(float(result.sharpe_ratio), 4),
        "max_drawdown_pct": round(float(result.max_drawdown) * 100.0, 4),
        "trade_count": int(result.total_trades),
        "win_rate_pct": round(float(result.win_rate) * 100.0, 2),
        "avg_holding_days": round(
            sum(int((trade.exit_date - trade.entry_date).days) for trade in result.completed_trades) / max(1, len(result.completed_trades)),
            2,
        ),
        "final_nav": round(float(result.daily_nav[-1][1]) if result.daily_nav else float(capital), 2),
        "vs_nepse_pct_points": round((float(result.total_return) * 100.0) - float(nepse.get("return_pct") or 0.0), 4),
    }

    payload = {
        "strategy": {
            "id": str(strategy.get("id") or ""),
            "name": str(strategy.get("name") or ""),
            "config": copy.deepcopy(strategy.get("config") or {}),
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
    _latest_backtest_path(str(strategy.get("id") or "strategy")).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
