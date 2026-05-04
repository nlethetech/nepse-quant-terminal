"""Policy engine for control-plane actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from backend.quant_pro.tms_models import ExecutionAction

from .models import PolicyDecision, PolicyVerdict, TradingMode


@dataclass
class PolicyContext:
    mode: TradingMode
    action: str
    symbol: str
    quantity: int
    limit_price: Optional[float]
    target_order_ref: Optional[str] = None
    portfolio: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    live_enabled: bool = False
    market_open: bool = False
    max_order_notional: Optional[float] = None
    max_daily_orders: Optional[int] = None
    intents_today: int = 0
    duplicate_open_intent: bool = False
    price_deviation_pct: Optional[float] = None
    max_price_deviation_pct: Optional[float] = None
    owner_confirm_required: bool = True
    allow_auto_approval: bool = False


class PolicyEngine:
    """Pure policy checks before paper execution or live intent creation."""

    def evaluate(self, ctx: PolicyContext) -> PolicyVerdict:
        reasons = []
        machine = []
        action = str(ctx.action or "").lower()
        supported_actions = {str(item.value).lower() for item in ExecutionAction}
        buy_action = str(ExecutionAction.BUY.value).lower()
        sell_action = str(ExecutionAction.SELL.value).lower()
        modify_action = str(getattr(ExecutionAction, "MODIFY", "")).lower()
        cancel_action = str(getattr(ExecutionAction, "CANCEL", "")).lower()
        symbol = str(ctx.symbol or "").upper()
        qty = int(ctx.quantity or 0)
        price = float(ctx.limit_price or 0.0) if ctx.limit_price is not None else None
        portfolio = dict(ctx.portfolio or {})
        positions = dict(portfolio.get("positions") or {})
        cash = float(portfolio.get("cash") or 0.0)
        max_positions = int(portfolio.get("max_positions") or 0)
        open_positions = len(positions)

        def deny(code: str, detail: str) -> PolicyVerdict:
            machine.append({"code": code, "detail": detail})
            reasons.append(detail)
            return PolicyVerdict(
                decision=PolicyDecision.DENY,
                reasons=reasons,
                machine_reasons=machine,
                requires_approval=False,
                approved_mode=ctx.mode,
            )

        if action not in supported_actions:
            return deny("invalid_action", f"Unsupported action: {ctx.action}")

        if action in {buy_action, sell_action} and qty <= 0:
            return deny("invalid_qty", "Quantity must be positive")
        if modify_action and action == modify_action and qty < 0:
            return deny("invalid_qty", "Quantity cannot be negative")
        price_required_actions = {buy_action, sell_action}
        if modify_action:
            price_required_actions.add(modify_action)
        if action in price_required_actions and (price is None or price <= 0):
            return deny("invalid_price", "Explicit positive limit price required")
        order_ref_actions = {item for item in {cancel_action, modify_action} if item}
        if action in order_ref_actions and not ctx.target_order_ref:
            return deny("missing_order_ref", "Target order reference required")

        if ctx.mode != TradingMode.PAPER:
            return deny("paper_only", "Only paper trading is supported in this build")

        if ctx.mode == TradingMode.PAPER:
            if action == buy_action:
                if symbol in positions:
                    return deny("duplicate_holding", f"Already holding {symbol}")
                if max_positions and open_positions >= max_positions:
                    return deny("max_positions", "Max positions reached")
                notional = float(price or 0.0) * qty
                if cash and notional > cash:
                    return deny("cash", "Insufficient cash")
            elif action == sell_action:
                held = positions.get(symbol)
                if held is None:
                    return deny("missing_position", f"No position in {symbol}")
                held_qty = int(held.get("shares") or held.get("quantity") or 0)
                if held_qty and qty and qty > held_qty:
                    return deny("oversell", f"Requested {qty} shares but only {held_qty} available")
            return PolicyVerdict(
                decision=PolicyDecision.ALLOW,
                reasons=["paper_execution_allowed"],
                machine_reasons=[{"code": "paper_allow", "detail": "Paper execution permitted"}],
                approved_mode=TradingMode.PAPER,
            )

        return deny("paper_only", "Only paper trading is supported in this build")


def compute_price_deviation_pct(limit_price: Optional[float], ltp: Optional[float]) -> Optional[float]:
    if limit_price is None or ltp is None or float(limit_price) <= 0 or float(ltp) <= 0:
        return None
    return abs((float(limit_price) / float(ltp)) - 1.0) * 100.0
