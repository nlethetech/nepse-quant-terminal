"""Public paper-only NEPSE agent architecture.

This is the cleaned version of the internal multi-agent workflow. It keeps the
governance shape from the full system while removing private strategies,
credentials, execution integrations, and any live execution path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentStage:
    name: str
    roles: tuple[str, ...]


DATA_SOURCES: tuple[str, ...] = (
    "market_ohlcv_liquidity_price_trends",
    "public_news_and_event_evidence",
    "corporate_actions_and_filings",
    "fundamentals_and_sector_context",
)

STRATEGY_GATES: tuple[str, ...] = (
    "walk_forward_validation",
    "benchmark_comparison",
    "overfit_check",
    "fee_and_slippage_stress",
    "liquidity_and_lot_size_check",
)

PIPELINE_STAGES: tuple[AgentStage, ...] = (
    AgentStage("Quant Strategy Engine", ("quant_analyst",)),
    AgentStage(
        "Research Team",
        (
            "fundamentals_analyst",
            "news_analyst",
            "flow_analyst",
            "macro_analyst",
            "sector_analyst",
            "bull_researcher",
            "bear_researcher",
            "research_manager",
        ),
    ),
    AgentStage("Trade + Risk Governance", ("trader", "risk_aggressive", "risk_neutral", "risk_conservative")),
    AgentStage("Decision", ("portfolio_manager",)),
    AgentStage("Learning Loop", ("reflector",)),
)

PUBLIC_EXECUTION_MODE = "paper"
LIVE_EXECUTION_AVAILABLE = False


def expected_registered_roles() -> set[str]:
    roles: set[str] = set()
    for stage in PIPELINE_STAGES:
        roles.update(stage.roles)
    return roles
