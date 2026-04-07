"""Unified command service for paper and live execution paths."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

from backend.quant_pro.tms_audit import (
    count_intents_for_day,
    find_recent_open_intent,
    load_execution_intent,
    save_execution_intent,
)
from backend.quant_pro.tms_models import (
    ExecutionAction,
    ExecutionIntent,
    ExecutionSource,
    ExecutionStatus,
    utc_now_iso,
)
from backend.quant_pro.vendor_api import fetch_latest_ltp

from .decision_journal import (
    create_approval_request,
    load_approval_request,
    record_agent_decision,
    record_policy_event,
    update_approval_request,
)
from .models import (
    AgentDecision,
    ApprovalRequest,
    ApprovalStatus,
    CommandResult,
    MarketSnapshot,
    PolicyDecision,
    PortfolioSnapshot,
    RiskSnapshot,
    TradingMode,
)
from .policy_engine import PolicyContext, PolicyEngine, compute_price_deviation_pct
from .read_models import (
    build_live_state_snapshot,
    build_market_snapshot_from_trader,
    build_portfolio_snapshot_from_trader,
    build_risk_snapshot_from_trader,
    build_signal_candidates,
)


class ControlPlaneCommandService:
    """Single entrypoint for agent, MCP, TUI, and Telegram actions."""

    def __init__(
        self,
        *,
        trader: Any | None = None,
        live_service: Any | None = None,
        mode: TradingMode = TradingMode.PAPER,
        paper_submitter: Optional[Callable[[str, str, int, float], tuple[bool, str, Dict[str, Any]]]] = None,
        watchlist_fetcher: Optional[Callable[[], Dict[str, Any]]] = None,
        watchlist_adder: Optional[Callable[[str], Dict[str, Any]]] = None,
        watchlist_remover: Optional[Callable[[str], Dict[str, Any]]] = None,
        halt_callback: Optional[Callable[[str, str], None]] = None,
        resume_callback: Optional[Callable[[], None]] = None,
        reconcile_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        service_label: str = "control_plane",
    ):
        self.trader = trader
        self.live_service = live_service
        self.mode = _normalize_mode(str(mode))
        self.paper_submitter = paper_submitter
        self.watchlist_fetcher = watchlist_fetcher
        self.watchlist_adder = watchlist_adder
        self.watchlist_remover = watchlist_remover
        self.halt_callback = halt_callback
        self.resume_callback = resume_callback
        self.reconcile_callback = reconcile_callback
        self.service_label = service_label
        self.policy_engine = PolicyEngine()

    def get_market_snapshot(self) -> Dict[str, Any]:
        if self.trader is None:
            return MarketSnapshot(
                as_of="",
                regime="unknown",
                market_open=False,
                signal_count=0,
                price_source="",
            ).to_record()
        self._ensure_signals_loaded()
        return build_market_snapshot_from_trader(self.trader, market_open=self._is_market_open()).to_record()

    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        if self.trader is None:
            return PortfolioSnapshot(cash=0.0, nav=0.0, capital=0.0, open_positions=0).to_record()
        return build_portfolio_snapshot_from_trader(self.trader).to_record()

    def get_signal_candidates(self) -> Dict[str, Any]:
        if self.trader is None:
            return {"signals": []}
        self._ensure_signals_loaded()
        market = build_market_snapshot_from_trader(self.trader, market_open=self._is_market_open())
        return {"signals": [asdict(item) for item in market.signals], "regime": market.regime}

    def get_risk_status(self) -> Dict[str, Any]:
        if self.trader is None:
            return RiskSnapshot(
                halt_level="none",
                freeze_reason="",
                max_positions=0,
                open_positions=0,
                cash=0.0,
            ).to_record()
        return build_risk_snapshot_from_trader(self.trader).to_record()

    def get_live_state(self) -> Dict[str, Any]:
        return build_live_state_snapshot()

    def review_trade_candidate(
        self,
        *,
        mode: str,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: Optional[float],
        thesis: str = "",
        catalysts: Optional[list[str]] = None,
        risk: Optional[list[str]] = None,
        confidence: float = 0.0,
        horizon: str = "",
        source_signals: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        target_order_ref: Optional[str] = None,
        operator_surface: str = "mcp",
        allow_auto_approval: bool = False,
    ) -> Dict[str, Any]:
        decision = AgentDecision(
            action=str(action).lower(),
            symbol=str(symbol).upper(),
            quantity=int(quantity),
            limit_price=float(limit_price) if limit_price is not None else None,
            thesis=thesis,
            catalysts=list(catalysts or []),
            risk=list(risk or []),
            confidence=float(confidence),
            horizon=horizon,
            source_signals=list(source_signals or []),
            metadata=dict(metadata or {}),
        )
        record_agent_decision(decision)
        target_mode = _normalize_mode(str(mode))
        verdict = self._evaluate_policy(
            target_mode,
            action=decision.action,
            symbol=decision.symbol,
            quantity=decision.quantity,
            limit_price=decision.limit_price,
            target_order_ref=target_order_ref,
            allow_auto_approval=allow_auto_approval,
        )
        record_policy_event(
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            action=decision.action,
            mode=str(target_mode),
            verdict=verdict,
            metadata={"operator_surface": operator_surface, "target_order_ref": target_order_ref},
        )
        response = {"decision": decision.to_record(), "verdict": verdict.to_record()}
        if verdict.requires_approval:
            response["approval_hint"] = {
                "operator_surface": operator_surface,
                "status": ApprovalStatus.PENDING,
            }
        return response

    def submit_paper_order(
        self,
        *,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float,
        slippage_pct: Optional[float] = None,
        thesis: str = "",
        confidence: float = 0.0,
        source_signals: Optional[list[str]] = None,
    ) -> CommandResult:
        mode = TradingMode.PAPER
        verdict = self._evaluate_policy(
            mode,
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
        )
        record_policy_event(
            decision_id=None,
            symbol=symbol,
            action=action,
            mode=str(mode),
            verdict=verdict,
            metadata={"slippage_pct": slippage_pct, "thesis": thesis, "confidence": confidence, "source_signals": list(source_signals or [])},
        )
        if not verdict.allowed:
            return CommandResult(False, "rejected", "; ".join(verdict.reasons), mode)
        if self.paper_submitter is not None:
            ok, msg, payload = self.paper_submitter(str(action).upper(), str(symbol).upper(), int(quantity), float(limit_price))
            return CommandResult(ok, "submitted" if ok else "failed", msg, mode, payload=payload)
        if self.trader is None:
            return CommandResult(False, "unsupported", "Paper execution unavailable", mode)
        return self._submit_paper_via_trader(str(action).lower(), str(symbol).upper(), int(quantity), float(limit_price))

    def create_live_intent(
        self,
        *,
        action: str,
        symbol: str,
        quantity: int = 0,
        limit_price: Optional[float] = None,
        target_order_ref: Optional[str] = None,
        mode: str = "live",
        source: str = "strategy_entry",
        reason: str = "",
        strategy_tag: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        operator_surface: str = "mcp",
    ) -> CommandResult:
        target_mode = _normalize_mode(str(mode))
        verdict = self._evaluate_policy(
            target_mode,
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            target_order_ref=target_order_ref,
            allow_auto_approval=(str(action).lower() == str(ExecutionAction.CANCEL)),
        )
        record_policy_event(
            decision_id=None,
            symbol=symbol,
            action=action,
            mode=str(target_mode),
            verdict=verdict,
            metadata={"operator_surface": operator_surface, "strategy_tag": strategy_tag, "reason": reason},
        )
        if not verdict.allowed:
            return CommandResult(False, "rejected", "; ".join(verdict.reasons), target_mode)

        intent = ExecutionIntent(
            action=ExecutionAction(str(action).lower()),
            symbol=str(symbol).upper(),
            quantity=int(quantity or 0),
            limit_price=float(limit_price) if limit_price is not None else None,
            source=ExecutionSource(str(source)),
            target_order_ref=target_order_ref,
            reason=reason,
            strategy_tag=strategy_tag,
            requires_confirmation=bool(verdict.requires_approval),
            status=ExecutionStatus.PENDING_CONFIRMATION if verdict.requires_approval else ExecutionStatus.QUEUED,
            metadata=dict(metadata or {}),
        )

        if target_mode == TradingMode.SHADOW_LIVE:
            intent.status = ExecutionStatus.PENDING_CONFIRMATION if verdict.requires_approval else ExecutionStatus.ACCEPTED
            save_execution_intent(intent)
            approval_request = self._maybe_create_approval_request(
                intent,
                verdict=verdict,
                operator_surface=operator_surface,
                summary=f"{intent.action.upper()} {intent.symbol} x{intent.quantity} @ {intent.limit_price or 0:.1f}",
            )
            status = "pending_confirmation" if verdict.requires_approval else "shadow_recorded"
            return CommandResult(
                True,
                status,
                "Shadow live intent recorded",
                target_mode,
                payload={"intent": intent.to_record()},
                intent_id=intent.intent_id,
                approval_request=approval_request,
            )

        if self.live_service is None:
            return CommandResult(False, "unavailable", "Live execution service unavailable", target_mode)

        ok, detail, created_intent, result = self.live_service.submit_intent(intent, wait=not verdict.requires_approval)
        approval_request = self._maybe_create_approval_request(
            created_intent,
            verdict=verdict,
            operator_surface=operator_surface,
            summary=f"{created_intent.action.upper()} {created_intent.symbol} x{created_intent.quantity} @ {created_intent.limit_price or 0:.1f}",
        )
        payload = {"intent": created_intent.to_record()}
        if result is not None:
            payload["result"] = result.to_record()
        status = "pending_confirmation" if verdict.requires_approval else detail
        return CommandResult(
            ok,
            status,
            detail,
            target_mode,
            payload=payload,
            intent_id=created_intent.intent_id,
            approval_request=approval_request,
        )

    def confirm_live_intent(self, intent_id: str, *, mode: str = "live") -> CommandResult:
        target_mode = _normalize_mode(str(mode))
        if target_mode == TradingMode.SHADOW_LIVE:
            update_approval_request(intent_id, status=ApprovalStatus.APPROVED, metadata={"confirmed_at": utc_now_iso()})
            approval = load_approval_request(intent_id)
            return CommandResult(
                True,
                "approved",
                "Shadow live approval recorded",
                target_mode,
                payload={},
                intent_id=intent_id,
                approval_request=approval,
            )
        if self.live_service is None:
            return CommandResult(False, "unavailable", "Live execution service unavailable", target_mode, intent_id=intent_id)
        result = self.live_service.confirm_intent(intent_id, wait=True)
        intent = load_execution_intent(intent_id)
        update_approval_request(intent_id, status=ApprovalStatus.APPROVED, metadata={"confirmed_at": utc_now_iso()})
        approval = load_approval_request(intent_id)
        payload: Dict[str, Any] = {}
        if intent is not None:
            payload["intent"] = intent.to_record()
        if result is not None:
            payload["result"] = result.to_record()
        return CommandResult(True, "confirmed", "Live intent confirmed", target_mode, payload=payload, intent_id=intent_id, approval_request=approval)

    def cancel_live_intent(self, order_ref: str, *, operator_surface: str = "mcp") -> CommandResult:
        return self.create_live_intent(
            action=str(ExecutionAction.CANCEL),
            symbol="ORDER",
            quantity=0,
            limit_price=None,
            target_order_ref=order_ref,
            mode=str(self.mode if self.mode != TradingMode.PAPER else TradingMode.LIVE),
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="cancel_live_order",
            operator_surface=operator_surface,
        )

    def modify_live_intent(
        self,
        order_ref: str,
        *,
        limit_price: float,
        quantity: Optional[int] = None,
        operator_surface: str = "mcp",
    ) -> CommandResult:
        return self.create_live_intent(
            action=str(ExecutionAction.MODIFY),
            symbol="ORDER",
            quantity=int(quantity or 0),
            limit_price=float(limit_price),
            target_order_ref=order_ref,
            mode=str(self.mode if self.mode != TradingMode.PAPER else TradingMode.LIVE),
            source=str(ExecutionSource.OWNER_MANUAL),
            reason="modify_live_order",
            operator_surface=operator_surface,
        )

    def reconcile_live_state(self) -> CommandResult:
        if self.reconcile_callback is not None:
            summary = dict(self.reconcile_callback() or {})
        elif self.live_service is not None:
            summary = dict(self.live_service.reconcile() or {})
        else:
            return CommandResult(False, "unavailable", "Live reconciliation unavailable", TradingMode.LIVE)
        return CommandResult(True, "ok", "Live state reconciled", TradingMode.LIVE, payload=summary)

    def halt_trading(self, *, level: str = "all", reason: str = "manual halt") -> CommandResult:
        if self.halt_callback is None:
            return CommandResult(False, "unavailable", "Halt control unavailable", self.mode)
        self.halt_callback(level, reason)
        return CommandResult(True, "halted", f"Trading halted ({level})", self.mode, payload={"level": level, "reason": reason})

    def resume_trading(self) -> CommandResult:
        if self.resume_callback is None:
            return CommandResult(False, "unavailable", "Resume control unavailable", self.mode)
        self.resume_callback()
        return CommandResult(True, "resumed", "Trading resumed", self.mode)

    def sync_watchlist(self, *, action: str = "fetch", symbol: Optional[str] = None) -> CommandResult:
        op = str(action).lower()
        if op == "fetch":
            if self.watchlist_fetcher is None:
                return CommandResult(False, "unavailable", "Watchlist sync unavailable", self.mode)
            snapshot = dict(self.watchlist_fetcher() or {})
            return CommandResult(True, "ok", "Watchlist synced", self.mode, payload=snapshot)
        if op == "add":
            if self.watchlist_adder is None or not symbol:
                return CommandResult(False, "invalid", "Watchlist add unavailable", self.mode)
            snapshot = dict(self.watchlist_adder(str(symbol).upper()) or {})
            return CommandResult(True, "ok", f"Added {str(symbol).upper()} to watchlist", self.mode, payload=snapshot)
        if op == "remove":
            if self.watchlist_remover is None or not symbol:
                return CommandResult(False, "invalid", "Watchlist remove unavailable", self.mode)
            snapshot = dict(self.watchlist_remover(str(symbol).upper()) or {})
            return CommandResult(True, "ok", f"Removed {str(symbol).upper()} from watchlist", self.mode, payload=snapshot)
        return CommandResult(False, "invalid", f"Unsupported watchlist action: {action}", self.mode)

    def _submit_paper_via_trader(self, action: str, symbol: str, quantity: int, limit_price: float) -> CommandResult:
        if self.trader is None:
            return CommandResult(False, "unsupported", "Paper trader unavailable", TradingMode.PAPER)
        with self.trader._state_lock:
            if action == str(ExecutionAction.BUY):
                ok, msg = self.trader.execute_manual_buy(symbol, quantity, limit_price)
            elif action == str(ExecutionAction.SELL):
                held = self.trader.positions.get(symbol)
                if held is None:
                    return CommandResult(False, "rejected", f"No position in {symbol}.", TradingMode.PAPER)
                if int(quantity) not in {0, int(held.shares)}:
                    return CommandResult(
                        False,
                        "rejected",
                        f"Paper trader currently supports full-position sells only for {symbol}.",
                        TradingMode.PAPER,
                    )
                ok, msg = self.trader.execute_manual_sell(symbol)
            else:
                return CommandResult(False, "unsupported", f"Unsupported paper action: {action}", TradingMode.PAPER)
        return CommandResult(ok, "submitted" if ok else "failed", msg, TradingMode.PAPER, payload={"symbol": symbol, "quantity": quantity, "limit_price": limit_price})

    def _maybe_create_approval_request(
        self,
        intent: ExecutionIntent,
        *,
        verdict,
        operator_surface: str,
        summary: str,
    ) -> Optional[ApprovalRequest]:
        if not verdict.requires_approval:
            return None
        approval = ApprovalRequest(
            intent_id=intent.intent_id,
            summary=summary,
            operator_surface=operator_surface,
            status=ApprovalStatus.PENDING,
            requested_at=utc_now_iso(),
            expires_at=(self._utc_plus(seconds=90)),
            metadata={"symbol": intent.symbol, "action": str(intent.action), "quantity": intent.quantity, "limit_price": intent.limit_price},
        )
        create_approval_request(approval)
        return approval

    def _ensure_signals_loaded(self) -> None:
        if self.trader is None:
            return
        if getattr(self.trader, "signals_today", None):
            return
        prices_df = getattr(self.trader, "prices_df", None)
        signal_types = list(getattr(self.trader, "signal_types", []) or [])
        if prices_df is None or not signal_types:
            return
        try:
            from backend.trading.live_trader import generate_signals
            signals, regime = generate_signals(prices_df, signal_types)
            self.trader.signals_today = list(signals)
            self.trader.regime = regime
            self.trader.num_signals_today = len(signals)
        except Exception:
            return

    def _evaluate_policy(
        self,
        mode: TradingMode,
        *,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: Optional[float],
        target_order_ref: Optional[str] = None,
        allow_auto_approval: bool = False,
    ):
        live_settings = getattr(self.trader, "live_settings", None)
        ltp = fetch_latest_ltp(symbol) if symbol and symbol != "ORDER" else None
        max_positions = int(getattr(self.trader, "max_positions", 0) or 0)
        positions = dict(getattr(self.trader, "positions", {}) or {})
        portfolio = {
            "cash": float(getattr(self.trader, "cash", 0.0) or 0.0),
            "positions": positions,
            "max_positions": max_positions,
        }
        ctx = PolicyContext(
            mode=mode,
            action=str(action).lower(),
            symbol=str(symbol).upper(),
            quantity=int(quantity or 0),
            limit_price=float(limit_price) if limit_price is not None else None,
            target_order_ref=target_order_ref,
            portfolio=portfolio,
            risk=self.get_risk_status(),
            live_enabled=bool(mode == TradingMode.SHADOW_LIVE or getattr(self.trader, "live_execution_enabled", self.live_service is not None)),
            market_open=self._is_market_open(),
            max_order_notional=float(getattr(live_settings, "max_order_notional", 0.0) or 0.0) or None,
            max_daily_orders=int(getattr(live_settings, "max_daily_orders", 0) or 0) or None,
            intents_today=count_intents_for_day(self._day_key()),
            duplicate_open_intent=find_recent_open_intent(str(symbol).upper(), within_seconds=int(getattr(live_settings, "symbol_cooldown_secs", 0) or 0)) is not None if symbol and symbol != "ORDER" else False,
            price_deviation_pct=compute_price_deviation_pct(limit_price, ltp),
            max_price_deviation_pct=float(getattr(live_settings, "max_price_deviation_pct", 0.0) or 0.0) or None,
            owner_confirm_required=bool(getattr(live_settings, "owner_confirm_required", True)),
            allow_auto_approval=allow_auto_approval,
        )
        return self.policy_engine.evaluate(ctx)

    def _day_key(self) -> str:
        if self.trader is not None:
            try:
                from backend.trading.live_trader import now_nst
                return now_nst().date().isoformat()
            except Exception:
                pass
        return utc_now_iso()[:10]

    def _is_market_open(self) -> bool:
        try:
            from backend.trading.live_trader import is_market_open
            return bool(is_market_open())
        except Exception:
            return False

    @staticmethod
    def _utc_plus(*, seconds: int) -> str:
        from datetime import datetime, timezone

        return (datetime.now(timezone.utc) + timedelta(seconds=int(seconds))).replace(microsecond=0).isoformat()


def build_live_trader_control_plane(trader: Any) -> ControlPlaneCommandService:
    cached = getattr(trader, "_control_plane_service", None)
    if cached is not None:
        cached.live_service = getattr(trader, "live_execution_service", None)
        return cached
    service = ControlPlaneCommandService(
        trader=trader,
        live_service=getattr(trader, "live_execution_service", None),
        mode=_normalize_mode(str(getattr(trader, "execution_mode", "paper") or "paper")),
        halt_callback=lambda level, reason: trader.kill_live(level=level, reason=reason),
        resume_callback=trader.resume_live,
        reconcile_callback=trader.reconcile_live_orders,
        service_label="live_trader",
    )
    setattr(trader, "_control_plane_service", service)
    return service


def build_tui_control_plane(dashboard: Any) -> ControlPlaneCommandService:
    cached = getattr(dashboard, "_control_plane_service", None)
    def _paper_submitter(action: str, symbol: str, qty: int, price: float):
        msg = dashboard._submit_paper_order(action, symbol, qty, price)
        return True, msg, {"symbol": symbol, "quantity": qty, "limit_price": price}

    if cached is None:
        service = ControlPlaneCommandService(
            live_service=getattr(dashboard, "tms_service", None),
            mode=_normalize_mode(str(getattr(dashboard, "trade_mode", "paper"))),
            paper_submitter=_paper_submitter,
            service_label="dashboard_tui",
        )
        setattr(dashboard, "_control_plane_service", service)
    else:
        service = cached

    service.live_service = getattr(dashboard, "tms_service", None)
    service.mode = _normalize_mode(str(getattr(dashboard, "trade_mode", "paper")))
    service.paper_submitter = _paper_submitter
    service.watchlist_fetcher = (lambda: dashboard.tms_service.executor.fetch_watchlist_snapshot()) if getattr(dashboard, "tms_service", None) else None
    service.watchlist_adder = (lambda symbol: dashboard.tms_service.executor.add_watchlist_symbol(symbol)) if getattr(dashboard, "tms_service", None) else None
    service.watchlist_remover = (lambda symbol: dashboard.tms_service.executor.remove_watchlist_symbol(symbol)) if getattr(dashboard, "tms_service", None) else None
    service.reconcile_callback = (lambda: dashboard.tms_service.reconcile()) if getattr(dashboard, "tms_service", None) else None
    setattr(dashboard, "_control_plane_service", service)
    return service


def build_env_live_trader_control_plane() -> ControlPlaneCommandService:
    from backend.quant_pro.config import DEFAULT_CAPITAL
    from backend.trading.live_trader import DEFAULT_PAPER_PORTFOLIO, LiveTrader

    paper_portfolio = str(os.environ.get("NEPSE_MCP_PAPER_PORTFOLIO", "")).strip() or str(DEFAULT_PAPER_PORTFOLIO)

    args = argparse.Namespace(
        capital=float(os.environ.get("NEPSE_MCP_CAPITAL", DEFAULT_CAPITAL)),
        signals=os.environ.get("NEPSE_MCP_SIGNALS", "volume,quality,low_vol"),
        refresh_secs=int(os.environ.get("NEPSE_MCP_REFRESH_SECS", "300")),
        no_telegram=True,
        dry_run=os.environ.get("NEPSE_MCP_DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"},
        continuous=False,
        headless=True,
        mode=os.environ.get("NEPSE_MCP_TRADING_MODE", "paper"),
        paper_portfolio=paper_portfolio,
    )
    trader = LiveTrader(args)
    if trader.live_execution_enabled:
        trader._start_live_execution_service()
    return build_live_trader_control_plane(trader)


def _normalize_mode(mode: str) -> TradingMode:
    raw = str(mode or "paper").strip().lower()
    if raw == "dual":
        return TradingMode.LIVE
    return TradingMode(raw)
