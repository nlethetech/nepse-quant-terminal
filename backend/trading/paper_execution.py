"""Shared paper execution service for account-scoped strategy autopilot.

This module is intentionally broker-free. It owns local paper order state,
fills, account ledgers, structured execution logs, and strategy run manifests.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    try:
        import msvcrt
    except ImportError:  # pragma: no cover
        msvcrt = None
else:
    msvcrt = None

import pandas as pd

from backend.backtesting.simple_backtest import get_symbol_sector
from backend.quant_pro.signal_ranking import blocked_signal_symbol_reason, canonicalize_signal_symbol
from backend.quant_pro.nepse_calendar import current_nepal_datetime
from backend.quant_pro.paths import ensure_dir, get_runtime_dir
from backend.trading.live_trader import (
    NAV_LOG_COLS,
    PORTFOLIO_COLS,
    TRADE_LOG_COLS,
    Position,
    TradeRecord,
    _realized_sell_breakdown,
    append_nav_log,
    append_trade_log,
    calculate_cash_from_trade_log,
    load_portfolio,
    load_runtime_state,
    load_trade_log_df,
    save_portfolio,
    save_runtime_state,
)
from validation.transaction_costs import TransactionCostModel as NepseFees


PAPER_ORDER_STATUSES = {"OPEN", "FILLED", "CANCELLED", "REJECTED"}


@dataclass
class PaperOrder:
    order_id: str
    account_id: str
    action: str
    symbol: str
    quantity: int
    limit_price: float
    slippage_pct: float
    status: str
    source: str
    reason: str
    strategy_id: str = ""
    run_id: str = ""
    quote_source: str = ""
    quote_age_seconds: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""
    risk_result: Dict[str, Any] = field(default_factory=dict)
    filled_qty: int = 0
    fill_price: float = 0.0
    day: str = ""

    def to_record(self) -> Dict[str, Any]:
        row = asdict(self)
        row["id"] = self.order_id
        row["qty"] = self.quantity
        row["price"] = self.limit_price
        row["trigger_price"] = self.limit_price
        return row

    @classmethod
    def from_record(cls, row: Dict[str, Any]) -> "PaperOrder":
        status = str(row.get("status") or "OPEN").upper()
        if status not in PAPER_ORDER_STATUSES:
            status = "OPEN"
        created = str(row.get("created_at") or "")
        return cls(
            order_id=str(row.get("order_id") or row.get("id") or uuid.uuid4().hex[:12]),
            account_id=str(row.get("account_id") or ""),
            action=str(row.get("action") or "").upper(),
            symbol=str(row.get("symbol") or "").upper(),
            quantity=int(float(row.get("quantity", row.get("qty", 0)) or 0)),
            limit_price=float(row.get("limit_price", row.get("price", 0.0)) or 0.0),
            slippage_pct=float(row.get("slippage_pct") or 0.0),
            status=status,
            source=str(row.get("source") or ""),
            reason=str(row.get("reason") or ""),
            strategy_id=str(row.get("strategy_id") or ""),
            run_id=str(row.get("run_id") or ""),
            quote_source=str(row.get("quote_source") or ""),
            quote_age_seconds=(
                float(row["quote_age_seconds"])
                if row.get("quote_age_seconds") not in (None, "")
                else None
            ),
            created_at=created,
            updated_at=str(row.get("updated_at") or created),
            risk_result=dict(row.get("risk_result") or {}),
            filled_qty=int(float(row.get("filled_qty") or 0)),
            fill_price=float(row.get("fill_price") or 0.0),
            day=str(row.get("day") or created[:10]),
        )


@dataclass
class PaperExecutionResult:
    ok: bool
    status: str
    message: str
    order: Optional[PaperOrder] = None
    filled_orders: List[PaperOrder] = field(default_factory=list)
    rejected_orders: List[PaperOrder] = field(default_factory=list)
    risk_result: Dict[str, Any] = field(default_factory=dict)


def _now_nst() -> datetime:
    return current_nepal_datetime()


def _now_stamp() -> str:
    return _now_nst().strftime("%Y-%m-%d %H:%M:%S")


def _lock_file(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:  # pragma: no cover - Windows only
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")


def _unlock_file(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover - Windows only
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError("No file-lock implementation available on this platform")


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload) if isinstance(payload, list) else []


def _write_json_locked(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def _ensure_csv(path: Path, columns: Iterable[str]) -> None:
    ensure_dir(path.parent)
    if not path.exists():
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def _trade_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    raw_date = row.get("Date") or ""
    try:
        date_text = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
    except Exception:
        date_text = str(raw_date)[:10]
    return (
        date_text,
        str(row.get("Action") or "").upper(),
        str(row.get("Symbol") or "").upper(),
        str(row.get("Shares") or ""),
        str(row.get("Price") or ""),
    )


def _quote_price_and_meta(symbol: str, payload: Any) -> tuple[float, str, Optional[float]]:
    if isinstance(payload, dict):
        raw_price = (
            payload.get("ltp")
            or payload.get("price")
            or payload.get("last_traded_price")
            or payload.get("close")
            or payload.get("last")
        )
        source = str(payload.get("source") or payload.get("quote_source") or "quote_snapshot")
        age = payload.get("age_seconds") or payload.get("quote_age_seconds")
        return float(raw_price or 0.0), source, (float(age) if age not in (None, "") else None)
    return float(payload or 0.0), "quote_snapshot", None


class PaperExecutionService:
    """Account-scoped local paper execution and ledger service."""

    def __init__(
        self,
        account_id: str = "account_1",
        *,
        account_dir: str | Path | None = None,
        initial_capital: float = 1_000_000.0,
        strategy_id: str = "",
        max_positions: int = 5,
        sector_limit: float = 0.35,
        max_order_notional: Optional[float] = None,
        max_quote_age_seconds: int = 900,
        max_daily_turnover_pct: float = 1.0,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.15,
    ) -> None:
        self.account_id = str(account_id or "account_1")
        self.account_dir = Path(account_dir) if account_dir else Path(get_runtime_dir(__file__)) / "accounts" / self.account_id
        ensure_dir(self.account_dir)
        self.initial_capital = float(initial_capital)
        self.strategy_id = str(strategy_id or "")
        self.max_positions = int(max_positions)
        self.sector_limit = float(sector_limit)
        self.max_order_notional = float(max_order_notional or (self.initial_capital / max(1, self.max_positions)))
        self.max_quote_age_seconds = int(max_quote_age_seconds)
        self.max_daily_turnover_pct = float(max_daily_turnover_pct)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)

        self.portfolio_file = self.account_dir / "paper_portfolio.csv"
        self.trade_log_file = self.account_dir / "paper_trade_log.csv"
        self.nav_log_file = self.account_dir / "paper_nav_log.csv"
        self.state_file = self.account_dir / "paper_state.json"
        self.orders_file = self.account_dir / "tui_paper_orders.json"
        self.order_history_file = self.account_dir / "tui_paper_order_history.json"
        self.structured_log_file = self.account_dir / "paper_execution.jsonl"
        self.manifest_dir = ensure_dir(self.account_dir / "strategy_runs")

        self._ensure_files()
        self._migrate_legacy_tui_files()

    def _ensure_files(self) -> None:
        _ensure_csv(self.portfolio_file, PORTFOLIO_COLS)
        _ensure_csv(self.trade_log_file, TRADE_LOG_COLS)
        _ensure_csv(self.nav_log_file, NAV_LOG_COLS)
        if not self.state_file.exists():
            save_runtime_state(
                str(self.state_file),
                {"cash": self.initial_capital, "daily_start_nav": self.initial_capital},
            )
        if not self.orders_file.exists():
            _write_json_locked(self.orders_file, [])
        if not self.order_history_file.exists():
            _write_json_locked(self.order_history_file, [])

    def _migrate_legacy_tui_files(self) -> None:
        legacy_portfolio = self.account_dir / "tui_paper_portfolio.csv"
        legacy_trade_log = self.account_dir / "tui_paper_trade_log.csv"
        legacy_state = self.account_dir / "tui_paper_state.json"

        if legacy_portfolio.exists() and not load_portfolio(str(self.portfolio_file)):
            legacy_positions = load_portfolio(str(legacy_portfolio))
            if legacy_positions:
                shutil.copy2(legacy_portfolio, self.portfolio_file)
                self._log_event("legacy_migration", {"file": legacy_portfolio.name, "rows": len(legacy_positions)})

        if legacy_trade_log.exists():
            existing = load_trade_log_df(str(self.trade_log_file))
            legacy = load_trade_log_df(str(legacy_trade_log))
            if not legacy.empty:
                existing_rows = existing.to_dict("records") if not existing.empty else []
                seen = {_trade_key(row) for row in existing_rows}
                new_rows = [row for row in legacy.to_dict("records") if _trade_key(row) not in seen]
                if new_rows:
                    with self.trade_log_file.open("a", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_COLS)
                        if self.trade_log_file.stat().st_size == 0:
                            writer.writeheader()
                        for row in new_rows:
                            writer.writerow({col: row.get(col, "") for col in TRADE_LOG_COLS})
                    self._log_event("legacy_migration", {"file": legacy_trade_log.name, "rows": len(new_rows)})

        state = load_runtime_state(str(self.state_file))
        if legacy_state.exists() and not isinstance(state.get("cash"), (int, float)):
            try:
                legacy_payload = json.loads(legacy_state.read_text(encoding="utf-8"))
                if isinstance(legacy_payload, dict):
                    save_runtime_state(str(self.state_file), legacy_payload)
                    self._log_event("legacy_migration", {"file": legacy_state.name, "rows": 1})
            except Exception:
                pass

    def _load_state(self) -> Dict[str, Any]:
        state = load_runtime_state(str(self.state_file))
        if not isinstance(state.get("cash"), (int, float)):
            rebuilt = calculate_cash_from_trade_log(self.initial_capital, str(self.trade_log_file))
            if rebuilt is not None:
                state["cash"] = rebuilt
            else:
                positions = load_portfolio(str(self.portfolio_file))
                state["cash"] = max(0.0, self.initial_capital - sum(pos.cost_basis for pos in positions.values()))
        state.setdefault("daily_start_nav", self.initial_capital)
        state.setdefault("manual_halt", False)
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        save_runtime_state(str(self.state_file), state)

    def _load_orders(self) -> List[PaperOrder]:
        return [PaperOrder.from_record(row) for row in _read_json_list(self.orders_file)]

    def _load_history(self) -> List[PaperOrder]:
        return [PaperOrder.from_record(row) for row in _read_json_list(self.order_history_file)]

    def _save_orders(self, orders: List[PaperOrder]) -> None:
        _write_json_locked(self.orders_file, [order.to_record() for order in orders])

    def _save_history(self, history: List[PaperOrder]) -> None:
        _write_json_locked(self.order_history_file, [order.to_record() for order in history])

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "account_id": self.account_id,
            **payload,
        }
        _append_jsonl(self.structured_log_file, row)

    def start_strategy_manifest(self, *, run_id: str, strategy_config: Dict[str, Any], signals: List[Dict[str, Any]]) -> None:
        payload = {
            "run_id": run_id,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "git_commit": self._git_commit(),
            "created_at": _now_stamp(),
            "data_date": _now_nst().strftime("%Y-%m-%d"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "strategy_config": strategy_config,
            "generated_signals": signals,
            "submitted_orders": [],
            "rejected_orders": [],
            "fills": [],
            "risk_blocks": [],
        }
        self._write_manifest(run_id, payload)
        self._log_event("strategy_run_started", {"run_id": run_id, "signal_count": len(signals)})

    def _append_manifest_event(self, run_id: str, key: str, payload: Dict[str, Any]) -> None:
        if not run_id:
            return
        path = self.manifest_dir / f"{run_id}.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            manifest = {}
        manifest.setdefault(key, [])
        if isinstance(manifest[key], list):
            manifest[key].append(payload)
        self._write_manifest(run_id, manifest)

    def _write_manifest(self, run_id: str, payload: Dict[str, Any]) -> None:
        _write_json_locked(self.manifest_dir / f"{run_id}.json", payload)

    def _git_commit(self) -> str:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
                text=True,
                capture_output=True,
                timeout=3,
            )
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except Exception:
            return ""

    def _has_same_day_trade(self, symbol: str, *, action: str) -> bool:
        trades = load_trade_log_df(str(self.trade_log_file))
        if trades.empty:
            return False
        today = _now_nst().date()
        mask = (
            (trades["Date"].dt.date == today)
            & (trades["Symbol"].astype(str).str.upper() == symbol)
            & (trades["Action"].astype(str).str.upper() == action)
        )
        return bool(mask.any())

    def _daily_turnover(self) -> float:
        trades = load_trade_log_df(str(self.trade_log_file))
        if trades.empty:
            return 0.0
        today = _now_nst().date()
        day_trades = trades[trades["Date"].dt.date == today]
        if day_trades.empty:
            return 0.0
        return float((day_trades["Shares"].astype(float) * day_trades["Price"].astype(float)).sum())

    def _risk_check(
        self,
        *,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float,
        quote_age_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        state = self._load_state()
        positions = load_portfolio(str(self.portfolio_file))
        nav = float(state.get("cash") or 0.0) + sum(pos.market_value for pos in positions.values())
        daily_start_nav = float(state.get("daily_start_nav") or nav or self.initial_capital)
        peak_nav = float(state.get("peak_nav") or max(daily_start_nav, nav, self.initial_capital))
        state["peak_nav"] = max(peak_nav, nav)
        self._save_state(state)

        if state.get("manual_halt"):
            return {"ok": False, "reason": "manual_halt"}
        symbol_block = blocked_signal_symbol_reason(symbol)
        if symbol_block:
            return {"ok": False, "reason": "blocked_symbol", "detail": symbol_block, "symbol": symbol}
        if quote_age_seconds is not None and quote_age_seconds > self.max_quote_age_seconds:
            return {"ok": False, "reason": "stale_quote", "quote_age_seconds": quote_age_seconds}
        if quantity <= 0 or limit_price <= 0:
            return {"ok": False, "reason": "invalid_order"}
        notional = float(quantity) * float(limit_price)
        if notional > self.max_order_notional * 1.05:
            return {"ok": False, "reason": "max_order_notional", "notional": notional}
        if self._daily_turnover() + notional > self.initial_capital * self.max_daily_turnover_pct:
            return {"ok": False, "reason": "max_daily_turnover"}
        if daily_start_nav > 0 and (nav - daily_start_nav) / daily_start_nav < -self.max_daily_loss_pct:
            return {"ok": False, "reason": "max_daily_loss"}
        if peak_nav > 0 and (peak_nav - nav) / peak_nav > self.max_drawdown_pct:
            return {"ok": False, "reason": "max_drawdown"}

        action = action.upper()
        if action == "BUY":
            if symbol in positions:
                return {"ok": False, "reason": "already_holding"}
            if len(positions) >= self.max_positions:
                return {"ok": False, "reason": "max_positions"}
            if self._has_same_day_trade(symbol, action="SELL"):
                return {"ok": False, "reason": "same_day_churn_buy_after_sell"}
            sector = get_symbol_sector(symbol) or "Other"
            sector_value = sum(
                pos.market_value for pos in positions.values()
                if (get_symbol_sector(pos.symbol) or "Other") == sector
            )
            if nav > 0 and (sector_value + notional) / nav > self.sector_limit:
                return {"ok": False, "reason": "max_sector", "sector": sector}
            fees = NepseFees.total_fees(quantity, limit_price)
            if notional + fees > float(state.get("cash") or 0.0):
                return {"ok": False, "reason": "insufficient_cash"}
        elif action == "SELL":
            pos = positions.get(symbol)
            if pos is None:
                return {"ok": False, "reason": "missing_position"}
            if int(quantity) > int(pos.shares):
                return {"ok": False, "reason": "oversell"}
            if str(pos.entry_date or "")[:10] == _now_nst().strftime("%Y-%m-%d") or self._has_same_day_trade(symbol, action="BUY"):
                return {"ok": False, "reason": "same_day_churn_sell_after_buy"}
        else:
            return {"ok": False, "reason": "unsupported_action"}
        return {"ok": True, "reason": "accepted"}

    def _visible_rejection(
        self,
        *,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float,
        slippage_pct: float,
        source: str,
        reason: str,
        strategy_id: str,
        run_id: str,
        risk_result: Dict[str, Any],
    ) -> PaperOrder:
        ts = _now_stamp()
        order = PaperOrder(
            order_id=uuid.uuid4().hex[:12],
            account_id=self.account_id,
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            slippage_pct=slippage_pct,
            status="REJECTED",
            source=source,
            reason=reason or str(risk_result.get("reason") or "rejected"),
            strategy_id=strategy_id,
            run_id=run_id,
            created_at=ts,
            updated_at=ts,
            risk_result=risk_result,
            day=ts[:10],
        )
        history = self._load_history()
        history.append(order)
        self._save_history(history)
        self._log_event("paper_order_rejected", {"run_id": run_id, "order": order.to_record()})
        self._append_manifest_event(run_id, "rejected_orders", order.to_record())
        self._append_manifest_event(run_id, "risk_blocks", {"order_id": order.order_id, **risk_result})
        return order

    def submit_order(
        self,
        account_id: str,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float,
        source: str,
        reason: str,
        strategy_id: Optional[str] = None,
        *,
        slippage_pct: float = 2.0,
        run_id: str = "",
    ) -> PaperExecutionResult:
        if str(account_id or self.account_id) != self.account_id:
            return PaperExecutionResult(False, "rejected", f"Unknown account {account_id}")

        action_text = str(action or "").upper()
        sym = canonicalize_signal_symbol(symbol)
        qty = int(quantity)
        price = float(limit_price)
        risk = self._risk_check(action=action_text, symbol=sym, quantity=qty, limit_price=price)
        if not risk.get("ok"):
            rejected = self._visible_rejection(
                action=action_text,
                symbol=sym,
                quantity=qty,
                limit_price=price,
                slippage_pct=slippage_pct,
                source=source,
                reason=reason,
                strategy_id=str(strategy_id or self.strategy_id),
                run_id=run_id,
                risk_result=risk,
            )
            message = str(risk.get("reason") or "rejected")
            if message == "blocked_symbol":
                message = f"Blocked symbol {sym}: {risk.get('detail') or 'non_tradeable'}"
            return PaperExecutionResult(False, "rejected", message, rejected, rejected_orders=[rejected], risk_result=risk)

        orders = self._load_orders()
        duplicate = any(
            order.status == "OPEN" and order.action == action_text and order.symbol == sym
            for order in orders
        )
        if duplicate:
            risk = {"ok": False, "reason": "duplicate_open_order"}
            rejected = self._visible_rejection(
                action=action_text,
                symbol=sym,
                quantity=qty,
                limit_price=price,
                slippage_pct=slippage_pct,
                source=source,
                reason=reason,
                strategy_id=str(strategy_id or self.strategy_id),
                run_id=run_id,
                risk_result=risk,
            )
            return PaperExecutionResult(False, "rejected", "Duplicate open order", rejected, rejected_orders=[rejected], risk_result=risk)

        ts = _now_stamp()
        order = PaperOrder(
            order_id=uuid.uuid4().hex[:12],
            account_id=self.account_id,
            action=action_text,
            symbol=sym,
            quantity=qty,
            limit_price=price,
            slippage_pct=slippage_pct,
            status="OPEN",
            source=source,
            reason=reason,
            strategy_id=str(strategy_id or self.strategy_id),
            run_id=run_id,
            created_at=ts,
            updated_at=ts,
            risk_result=risk,
            day=ts[:10],
        )
        orders.append(order)
        self._save_orders(orders)
        self._log_event("paper_order_submitted", {"run_id": run_id, "order": order.to_record()})
        self._append_manifest_event(run_id, "submitted_orders", order.to_record())
        return PaperExecutionResult(True, "submitted", "Order accepted", order, risk_result=risk)

    def match_open_orders(self, account_id: str, quote_snapshot: Dict[str, Any]) -> PaperExecutionResult:
        if str(account_id or self.account_id) != self.account_id:
            return PaperExecutionResult(False, "rejected", f"Unknown account {account_id}")

        orders = self._load_orders()
        history = self._load_history()
        keep: List[PaperOrder] = []
        filled: List[PaperOrder] = []
        rejected: List[PaperOrder] = []
        state = self._load_state()
        positions = load_portfolio(str(self.portfolio_file))
        today = _now_nst().strftime("%Y-%m-%d")

        for order in orders:
            if order.status != "OPEN":
                keep.append(order)
                continue
            payload = quote_snapshot.get(order.symbol) or quote_snapshot.get(order.symbol.upper())
            if payload is None:
                keep.append(order)
                continue
            ltp, quote_source, quote_age = _quote_price_and_meta(order.symbol, payload)
            if ltp <= 0:
                keep.append(order)
                continue
            slip = float(order.slippage_pct or 0.0) / 100.0
            matched = (
                (order.action == "BUY" and ltp <= order.limit_price * (1 + slip))
                or (order.action == "SELL" and ltp >= order.limit_price * (1 - slip))
            )
            if not matched:
                keep.append(order)
                continue

            risk = self._risk_check(
                action=order.action,
                symbol=order.symbol,
                quantity=order.quantity,
                limit_price=ltp,
                quote_age_seconds=quote_age,
            )
            order.quote_source = quote_source
            order.quote_age_seconds = quote_age
            order.updated_at = _now_stamp()
            order.risk_result = risk
            if not risk.get("ok"):
                order.status = "REJECTED"
                order.reason = order.reason or str(risk.get("reason") or "rejected")
                history.append(order)
                rejected.append(order)
                self._log_event("paper_order_rejected_at_match", {"run_id": order.run_id, "order": order.to_record()})
                self._append_manifest_event(order.run_id, "rejected_orders", order.to_record())
                self._append_manifest_event(order.run_id, "risk_blocks", {"order_id": order.order_id, **risk})
                continue

            if order.action == "BUY":
                fees = NepseFees.total_fees(order.quantity, ltp)
                state["cash"] = float(state.get("cash") or 0.0) - (order.quantity * ltp + fees)
                positions[order.symbol] = Position(
                    symbol=order.symbol,
                    shares=order.quantity,
                    entry_price=ltp,
                    entry_date=today,
                    buy_fees=fees,
                    signal_type=order.reason or order.source,
                    high_watermark=ltp,
                    last_ltp=ltp,
                    last_ltp_source=quote_source,
                    last_ltp_time_utc=datetime.now(timezone.utc).isoformat(),
                )
                append_trade_log(
                    TradeRecord(today, "BUY", order.symbol, order.quantity, ltp, fees, order.reason or order.source),
                    str(self.trade_log_file),
                )
            else:
                pos = positions.get(order.symbol)
                if pos is None:
                    order.status = "REJECTED"
                    order.reason = "missing_position"
                    history.append(order)
                    rejected.append(order)
                    continue
                sell = _realized_sell_breakdown(pos, ltp, exit_date_str=today)
                state["cash"] = float(state.get("cash") or 0.0) + float(sell["proceeds"])
                append_trade_log(
                    TradeRecord(
                        today,
                        "SELL",
                        order.symbol,
                        int(pos.shares),
                        ltp,
                        float(sell["total_sell_cost"]),
                        order.reason or order.source,
                        pnl=round(float(sell["net_pnl"]), 2),
                        pnl_pct=round(float(sell["pnl_pct"]), 4),
                    ),
                    str(self.trade_log_file),
                )
                positions.pop(order.symbol, None)

            order.status = "FILLED"
            order.filled_qty = order.quantity
            order.fill_price = ltp
            history.append(order)
            filled.append(order)
            self._log_event("paper_order_filled", {"run_id": order.run_id, "order": order.to_record()})
            self._append_manifest_event(order.run_id, "fills", order.to_record())

        save_portfolio(positions, str(self.portfolio_file))
        state["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        self._save_orders(keep)
        self._save_history(history)
        return PaperExecutionResult(bool(filled), "matched", f"Filled {len(filled)} order(s)", filled_orders=filled, rejected_orders=rejected)

    def cancel_order(self, account_id: str, order_id: str, reason: str) -> PaperExecutionResult:
        if str(account_id or self.account_id) != self.account_id:
            return PaperExecutionResult(False, "rejected", f"Unknown account {account_id}")
        orders = self._load_orders()
        history = self._load_history()
        keep: List[PaperOrder] = []
        target: Optional[PaperOrder] = None
        for order in orders:
            if order.order_id == order_id and order.status == "OPEN":
                order.status = "CANCELLED"
                order.reason = reason
                order.updated_at = _now_stamp()
                history.append(order)
                target = order
                continue
            keep.append(order)
        self._save_orders(keep)
        self._save_history(history)
        if target is None:
            return PaperExecutionResult(False, "not_found", "Order not found")
        self._log_event("paper_order_cancelled", {"order": target.to_record()})
        return PaperExecutionResult(True, "cancelled", "Order cancelled", target)

    def get_account_execution_state(self, account_id: str | None = None) -> Dict[str, Any]:
        if account_id and str(account_id) != self.account_id:
            raise ValueError(f"Unknown account {account_id}")
        state = self._load_state()
        positions = load_portfolio(str(self.portfolio_file))
        open_orders = [order.to_record() for order in self._load_orders() if order.status == "OPEN"]
        history = [order.to_record() for order in self._load_history()]
        nav = float(state.get("cash") or 0.0) + sum(pos.market_value for pos in positions.values())
        return {
            "account_id": self.account_id,
            "cash": float(state.get("cash") or 0.0),
            "nav": nav,
            "positions": positions,
            "open_orders": open_orders,
            "order_history": history,
            "paths": {
                "portfolio": str(self.portfolio_file),
                "trade_log": str(self.trade_log_file),
                "nav_log": str(self.nav_log_file),
                "state": str(self.state_file),
                "orders": str(self.orders_file),
                "order_history": str(self.order_history_file),
            },
        }

    def log_nav_snapshot(self) -> None:
        state = self._load_state()
        positions = load_portfolio(str(self.portfolio_file))
        positions_value = sum(pos.market_value for pos in positions.values())
        nav = float(state.get("cash") or 0.0) + positions_value
        append_nav_log(_now_nst().strftime("%Y-%m-%d"), float(state.get("cash") or 0.0), positions_value, nav, len(positions), str(self.nav_log_file))
