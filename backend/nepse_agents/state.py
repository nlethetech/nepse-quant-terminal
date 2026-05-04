"""Mutable state for one paper-only agent decision run."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import DebateTurn, EvidencePacket, PortfolioDecision, ResearchPlan, RiskVerdict, SignalEnvelope, TraderProposal


@dataclass
class NepseAgentState:
    run_id: str
    candidate: SignalEnvelope
    evidence: list[EvidencePacket] = field(default_factory=list)
    debate_log: list[DebateTurn] = field(default_factory=list)
    research_plan: ResearchPlan | None = None
    trader_proposal: TraderProposal | None = None
    risk_verdict: RiskVerdict | None = None
    portfolio_decision: PortfolioDecision | None = None
    evidence_quorum: dict[str, object] = field(default_factory=dict)
    checkpoint_path: str = ""
    completed: bool = False

    def to_record(self) -> dict:
        return {
            "run_id": self.run_id,
            "candidate": self.candidate.to_record(),
            "evidence": [item.to_record() for item in self.evidence],
            "debate_log": [item.to_record() for item in self.debate_log],
            "research_plan": self.research_plan.to_record() if self.research_plan else None,
            "trader_proposal": self.trader_proposal.to_record() if self.trader_proposal else None,
            "risk_verdict": self.risk_verdict.to_record() if self.risk_verdict else None,
            "portfolio_decision": self.portfolio_decision.to_record() if self.portfolio_decision else None,
            "evidence_quorum": dict(self.evidence_quorum),
            "checkpoint_path": self.checkpoint_path,
            "completed": self.completed,
        }
