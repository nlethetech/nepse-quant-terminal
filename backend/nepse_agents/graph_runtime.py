"""Compatibility wrapper for the cleaned public agent graph."""

from __future__ import annotations

from .orchestrator import NepseAgentGraph


class LangGraphDecisionEngine(NepseAgentGraph):
    """Name-compatible engine; uses the public paper-only sequential graph."""

    version = "public-paper-v1"


def langgraph_runtime_available() -> bool:
    return False
