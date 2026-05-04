"""Clean paper-only multi-agent decision orchestrator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

from .debate import run_research_debate, run_risk_debate
from .evidence import build_public_evidence, evidence_quorum
from .schemas import PortfolioDecision, ResearchPlan, RiskVerdict, SignalEnvelope, TraderProposal
from .state import NepseAgentState


class NepseAgentGraph:
    """Sequential graph-compatible runtime for public paper decisions."""

    version = "public-paper-v1"

    def __init__(self, *, checkpoint_root: str | Path = "data/runtime/nepse_agents/checkpoints"):
        self.checkpoint_root = Path(checkpoint_root)

    def run(self, candidate: SignalEnvelope | dict) -> NepseAgentState:
        envelope = candidate if isinstance(candidate, SignalEnvelope) else SignalEnvelope.from_record(candidate)
        state = NepseAgentState(run_id=f"run_{uuid4().hex[:12]}", candidate=envelope)
        state.evidence = build_public_evidence(envelope)
        state.evidence_quorum = evidence_quorum(state.evidence)
        state.debate_log.extend(run_research_debate(envelope, state.evidence))
        state.research_plan = self._research_manager(state)
        state.trader_proposal = self._trader(state)
        state.debate_log.extend(run_risk_debate(envelope))
        state.risk_verdict = self._risk_committee(state)
        state.portfolio_decision = self._portfolio_manager(state)
        state.completed = True
        self._write_checkpoint(state)
        return state

    def _research_manager(self, state: NepseAgentState) -> ResearchPlan:
        c = state.candidate
        quorum_ok = bool(state.evidence_quorum.get("ok"))
        if not quorum_ok or c.score < 0.45:
            rating = "Skip"
        elif c.score >= 0.72 and c.confidence >= 0.62:
            rating = "Buy"
        elif c.score >= 0.60:
            rating = "Overweight"
        else:
            rating = "Hold"
        synthesis = (
            f"Research manager reviewed public evidence for {c.symbol}. Rating {rating} reflects score "
            f"{c.score:.2f}, confidence {c.confidence:.2f}, and evidence quorum={quorum_ok}."
        )
        return ResearchPlan(
            rating=rating,
            synthesis=synthesis,
            confidence=min(0.95, max(c.confidence, 0.30)),
            horizon_days=c.suggested_horizon_days,
            expected_move_pct=round((c.score - 0.50) * 20.0, 2),
        )

    def _trader(self, state: NepseAgentState) -> TraderProposal:
        c = state.candidate
        plan = state.research_plan
        action = "BUY" if plan and plan.rating in {"Buy", "Overweight"} else "HOLD"
        qty = _round_lot(c.suggested_quantity) if action == "BUY" else 0
        if c.suggested_limit_price <= 0:
            action = "HOLD"
            qty = 0
        return TraderProposal(
            action=action,
            symbol=c.symbol,
            quantity=qty,
            limit_price=max(c.suggested_limit_price, 1.0),
            horizon_days=c.suggested_horizon_days,
            rationale=f"Trader proposes {action} for paper mode only after research rating {plan.rating if plan else 'Skip'}.",
        )

    def _risk_committee(self, state: NepseAgentState) -> RiskVerdict:
        proposal = state.trader_proposal
        c = state.candidate
        blocked = False
        reason = ""
        if not bool(state.evidence_quorum.get("ok")):
            blocked = True
            reason = "evidence quorum failed"
        elif c.risks and c.confidence < 0.55:
            blocked = True
            reason = "explicit risks with low confidence"
        elif proposal is None or proposal.action != "BUY":
            blocked = True
            reason = "no buy proposal"
        final_qty = 0 if blocked else _round_lot(proposal.quantity)
        return RiskVerdict(
            final_quantity=final_qty,
            final_stop_pct=proposal.stop_loss_pct if proposal else 0.08,
            final_horizon_days=proposal.horizon_days if proposal else c.suggested_horizon_days,
            blocked=blocked,
            block_reason=reason,
            committee_notes={
                "aggressive": "paper-only approval allowed when quorum passes",
                "neutral": "quantity rounded to lot size",
                "conservative": "fails closed on missing evidence or explicit risk",
            },
        )

    def _portfolio_manager(self, state: NepseAgentState) -> PortfolioDecision:
        c = state.candidate
        risk = state.risk_verdict
        plan = state.research_plan
        evidence_ids = [item.packet_id for item in state.evidence]
        approved = bool(risk and not risk.blocked and risk.final_quantity > 0)
        action = "APPROVE" if approved else "HOLD" if plan and plan.rating == "Hold" else "REJECT"
        return PortfolioDecision(
            symbol=c.symbol,
            action=action,
            final_quantity=risk.final_quantity if risk else 0,
            final_limit_price=max(c.suggested_limit_price, 0.0),
            horizon_days=risk.final_horizon_days if risk else c.suggested_horizon_days,
            conviction=plan.confidence if plan else 0.0,
            thesis=(
                f"Final paper-only decision for {c.symbol}: {action}. "
                f"{plan.synthesis if plan else 'No research plan was produced.'}"
            ),
            evidence_packet_ids=evidence_ids,
            risks=list(c.risks or []),
            catalysts=list(c.catalysts or []),
            rating=plan.rating if plan else "Skip",
            evidence_gate_ok=bool(state.evidence_quorum.get("ok")),
            metadata={
                "graph_version": self.version,
                "risk_block_reason": risk.block_reason if risk else "",
                "execution_note": "paper-only; no live order path exists in this public build",
            },
        )

    def _write_checkpoint(self, state: NepseAgentState) -> None:
        root = self.checkpoint_root / time.strftime("%Y-%m-%d")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{state.run_id}_{state.candidate.symbol}.json"
        path.write_text(json.dumps(state.to_record(), indent=2, sort_keys=True))
        state.checkpoint_path = str(path)


def _round_lot(quantity: int, lot_size: int = 10) -> int:
    qty = max(0, int(quantity or 0))
    return (qty // lot_size) * lot_size


def run_paper_decision(candidate: dict) -> dict:
    """Run the public paper-only agent workflow and return a JSON-safe record."""
    state = NepseAgentGraph().run(candidate)
    return state.to_record()
