"""Paper-only trade approval gate for multi-agent decisions."""

from __future__ import annotations

from typing import Any


def decision_allows_paper_trade(decision: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a PortfolioDecision record allows paper execution."""
    if str(decision.get("execution_mode") or "paper") != "paper":
        return False, "Only paper execution is supported in the public build"
    if str(decision.get("action") or "").upper() != "APPROVE":
        return False, f"Decision action is {decision.get('action') or 'missing'}"
    if not bool(decision.get("evidence_gate_ok")):
        return False, "Evidence gate failed"
    if int(decision.get("final_quantity") or 0) <= 0:
        return False, "Final quantity is zero"
    return True, "Approved for paper execution only"
