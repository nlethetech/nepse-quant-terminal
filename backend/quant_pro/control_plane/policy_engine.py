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

        if action not in {str(item) for item in ExecutionAction}:
            return deny("invalid_action", f"Unsupported action: {ctx.action}")

        if action in {str(ExecutionAction.BUY), str(ExecutionAction.SELL)} and qty <= 0:
            return deny("invalid_qty", "Quantity must be positive")
        if action == str(ExecutionAction.MODIFY) and qty < 0:
            return deny("invalid_qty", "Quantity cannot be negative")
        if action in {str(ExecutionAction.BUY), str(ExecutionAction.SELL), str(ExecutionAction.MODIFY)} and (price is None or price <= 0):
            return deny("invalid_price", "Explicit positive limit price required")
        if action in {str(ExecutionAction.CANCEL), str(ExecutionAction.MODIFY)} and not ctx.target_order_ref:
            return deny("missing_order_ref", "Target order reference required")

        if ctx.mode == TradingMode.PAPER:
            if action == str(ExecutionAction.BUY):
                if symbol in positions:
                    return deny("duplicate_holding", f"Already holding {symbol}")
                if max_positions and open_positions >= max_positions:
                    return deny("max_positions", "Max positions reached")
                notional = float(price or 0.0) * qty
                if cash and notional > cash:
                    return deny("cash", "Insufficient cash")
            elif action == str(ExecutionAction.SELL):
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

        if not ctx.live_enabled:
            return deny("live_disabled", "Live execution is disabled")
        if action != str(ExecutionAction.CANCEL) and not ctx.market_open:
            return deny("market_closed", "Market is closed")
        if ctx.max_daily_orders is not None and ctx.max_daily_orders > 0 and ctx.intents_today >= ctx.max_daily_orders:
            return deny("daily_cap", "Daily live order cap reached")
        if ctx.duplicate_open_intent and action in {str(ExecutionAction.BUY), str(ExecutionAction.SELL)}:
            return deny("cooldown", f"Recent open live intent exists for {symbol}")
        if ctx.max_order_notional is not None and price is not None and (price * qty) > float(ctx.max_order_notional):
            return deny("max_notional", "Order exceeds max notional")
        if (
            ctx.price_deviation_pct is not None
            and ctx.max_price_deviation_pct is not None
            and float(ctx.price_deviation_pct) > float(ctx.max_price_deviation_pct)
        ):
            return deny(
                "price_band",
                f"Limit price deviates {float(ctx.price_deviation_pct):.2f}% from LTP",
            )

        if action == str(ExecutionAction.BUY):
            if symbol in positions:
                return deny("duplicate_holding", f"Already holding {symbol}")
            if max_positions and open_positions >= max_positions:
                return deny("max_positions", "Max positions reached")
            notional = float(price or 0.0) * qty
            if cash and notional > cash:
                return deny("cash", "Insufficient cash")
        elif action == str(ExecutionAction.SELL):
            held = positions.get(symbol)
            if held is None:
                return deny("missing_position", f"No position in {symbol}")

        auto_allowed = ctx.allow_auto_approval or action == str(ExecutionAction.CANCEL)
        requires_approval = not auto_allowed and action in {
            str(ExecutionAction.BUY),
            str(ExecutionAction.SELL),
            str(ExecutionAction.MODIFY),
        }
        decision = PolicyDecision.REQUIRE_APPROVAL if requires_approval else PolicyDecision.ALLOW
        reason_code = "live_approval_required" if requires_approval else "live_allow"
        reason_detail = "Live order queued pending owner approval" if requires_approval else "Live action permitted"
        return PolicyVerdict(
            decision=decision,
            reasons=[reason_detail],
            machine_reasons=[{"code": reason_code, "detail": reason_detail}],
            requires_approval=requires_approval,
            approved_mode=ctx.mode,
        )


def compute_price_deviation_pct(limit_price: Optional[float], ltp: Optional[float]) -> Optional[float]:
    if limit_price is None or ltp is None or float(limit_price) <= 0 or float(ltp) <= 0:
        return None
    return abs((float(limit_price) / float(ltp)) - 1.0) * 100.0
