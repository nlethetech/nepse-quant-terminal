from __future__ import annotations

from backend.quant_pro.control_plane.models import PolicyDecision, TradingMode
from backend.quant_pro.control_plane.policy_engine import PolicyContext, PolicyEngine


def test_paper_buy_denied_for_duplicate_holding():
    engine = PolicyEngine()
    verdict = engine.evaluate(
        PolicyContext(
            mode=TradingMode.PAPER,
            action="buy",
            symbol="NABIL",
            quantity=10,
            limit_price=500.0,
            portfolio={"cash": 100_000.0, "positions": {"NABIL": {"shares": 20}}, "max_positions": 5},
        )
    )

    assert verdict.decision == PolicyDecision.DENY
    assert verdict.machine_reasons[0]["code"] == "duplicate_holding"


def test_live_buy_denied_in_paper_only_build():
    engine = PolicyEngine()
    verdict = engine.evaluate(
        PolicyContext(
            mode=TradingMode.LIVE,
            action="buy",
            symbol="NABIL",
            quantity=10,
            limit_price=500.0,
            portfolio={"cash": 1_000_000.0, "positions": {}, "max_positions": 10},
            live_enabled=True,
            market_open=True,
            owner_confirm_required=True,
            max_order_notional=500_000.0,
            max_daily_orders=20,
            intents_today=0,
        )
    )

    assert verdict.decision == PolicyDecision.DENY
    assert verdict.requires_approval is False
    assert verdict.machine_reasons[0]["code"] == "paper_only"


def test_shadow_live_buy_denied_in_paper_only_build():
    engine = PolicyEngine()
    verdict = engine.evaluate(
        PolicyContext(
            mode=TradingMode.SHADOW_LIVE,
            action="buy",
            symbol="NABIL",
            quantity=10,
            limit_price=500.0,
            portfolio={"cash": 1_000_000.0, "positions": {}, "max_positions": 10},
            live_enabled=True,
            market_open=False,
        )
    )

    assert verdict.decision == PolicyDecision.DENY
    assert verdict.machine_reasons[0]["code"] == "paper_only"


def test_live_price_checks_are_not_reached_in_paper_only_build():
    engine = PolicyEngine()
    verdict = engine.evaluate(
        PolicyContext(
            mode=TradingMode.LIVE,
            action="buy",
            symbol="NABIL",
            quantity=10,
            limit_price=550.0,
            portfolio={"cash": 1_000_000.0, "positions": {}, "max_positions": 10},
            live_enabled=True,
            market_open=True,
            price_deviation_pct=10.0,
            max_price_deviation_pct=2.5,
        )
    )

    assert verdict.decision == PolicyDecision.DENY
    assert verdict.machine_reasons[0]["code"] == "paper_only"
