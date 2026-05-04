from __future__ import annotations

from backend.nepse_agents.architecture import LIVE_EXECUTION_AVAILABLE, PUBLIC_EXECUTION_MODE, expected_registered_roles
from backend.nepse_agents.gates import decision_allows_paper_trade
from backend.nepse_agents.orchestrator import NepseAgentGraph, run_paper_decision


def _candidate(**overrides):
    payload = {
        "symbol": "NABIL",
        "sector": "Banking",
        "score": 0.76,
        "strength": 0.80,
        "confidence": 0.72,
        "suggested_quantity": 30,
        "suggested_limit_price": 500.0,
        "suggested_horizon_days": 20,
        "catalysts": ["public earnings momentum"],
        "risks": [],
        "evidence_refs": ["public-source:demo"],
    }
    payload.update(overrides)
    return payload


def test_public_agent_architecture_is_paper_only():
    assert PUBLIC_EXECUTION_MODE == "paper"
    assert LIVE_EXECUTION_AVAILABLE is False
    assert {"quant_analyst", "research_manager", "trader", "portfolio_manager"} <= expected_registered_roles()


def test_public_agent_graph_approves_only_with_evidence_quorum(tmp_path):
    state = NepseAgentGraph(checkpoint_root=tmp_path).run(_candidate())

    decision = state.portfolio_decision.to_record()
    assert decision["execution_mode"] == "paper"
    assert decision["action"] == "APPROVE"
    assert decision["final_quantity"] == 30
    assert decision["evidence_gate_ok"] is True
    assert state.checkpoint_path

    ok, reason = decision_allows_paper_trade(decision)
    assert ok is True
    assert "paper" in reason


def test_public_agent_graph_fails_closed_without_evidence(tmp_path):
    state = NepseAgentGraph(checkpoint_root=tmp_path).run(_candidate(evidence_refs=[], signal_id=""))

    decision = state.portfolio_decision.to_record()
    assert decision["execution_mode"] == "paper"
    assert decision["action"] in {"HOLD", "REJECT"}

    ok, reason = decision_allows_paper_trade(decision)
    assert ok is False
    assert reason


def test_run_paper_decision_returns_json_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_paper_decision(_candidate(symbol="ADBL"))

    assert result["candidate"]["symbol"] == "ADBL"
    assert result["portfolio_decision"]["execution_mode"] == "paper"
    assert result["completed"] is True
