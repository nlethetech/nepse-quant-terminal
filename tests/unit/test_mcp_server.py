from __future__ import annotations

from backend.quant_pro.control_plane.models import CommandResult, TradingMode
from apps.mcp.server import build_tool_adapters


class DummyService:
    def get_market_snapshot(self):
        return {"market": "ok"}

    def get_portfolio_snapshot(self):
        return {"portfolio": "ok"}

    def get_signal_candidates(self):
        return {"signals": []}

    def get_risk_status(self):
        return {"risk": "ok"}

    def review_trade_candidate(self, **kwargs):
        return {"reviewed": kwargs["symbol"]}

    def submit_paper_order(self, **kwargs):
        return CommandResult(True, "submitted", "paper_ok", TradingMode.PAPER, payload=kwargs)

    def halt_trading(self, level="all", reason="mcp halt"):
        return CommandResult(True, "halted", "halt_ok", TradingMode.PAPER, payload={"level": level})

    def resume_trading(self):
        return CommandResult(True, "resumed", "resume_ok", TradingMode.PAPER)


def test_mcp_tool_adapters_route_to_service():
    tools = build_tool_adapters(DummyService())

    assert tools["get_market_snapshot"]() == {"market": "ok"}
    assert tools["review_trade_candidate"](mode="paper", action="buy", symbol="NABIL", quantity=10, limit_price=500.0) == {"reviewed": "NABIL"}

    paper = tools["submit_paper_order"](action="buy", symbol="NABIL", quantity=10, limit_price=500.0)
    assert paper["payload"]["symbol"] == "NABIL"
    assert "create_live_intent" not in tools
    assert "confirm_live_intent" not in tools
    assert "reconcile_live_state" not in tools


def test_mcp_tool_adapters_expose_paper_agent_graph(monkeypatch):
    monkeypatch.setattr(
        "apps.mcp.server.run_paper_decision",
        lambda candidate: {"candidate": candidate, "portfolio_decision": {"execution_mode": "paper"}},
        raising=False,
    )

    tools = build_tool_adapters(DummyService())
    result = tools["run_multi_agent_decision"]({"symbol": "NABIL"})

    assert result["candidate"]["symbol"] == "NABIL"
    assert result["portfolio_decision"]["execution_mode"] == "paper"


def test_mcp_tool_adapters_expose_nepalosint_tools(monkeypatch):
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_semantic_story_search",
        lambda query, hours=720, top_k=10, min_similarity=0.45: {"query": query, "results": [{"story_id": "s1"}]},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_unified_search",
        lambda query, limit=10, election_year=None: {"query": query, "categories": {"stories": {"items": [{"id": "s1"}], "total": 1}}},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.nepalosint_related_stories",
        lambda story_id, top_k=8, min_similarity=0.55, hours=8760: {"source_story_id": story_id, "similar_stories": [{"story_id": "s2"}]},
        raising=False,
    )

    tools = build_tool_adapters(DummyService())

    assert tools["semantic_story_search"](query="HRL")["results"][0]["story_id"] == "s1"
    assert tools["unified_osint_search"](query="HRL")["categories"]["stories"]["total"] == 1
    assert tools["related_story_search"](story_id="s1")["similar_stories"][0]["story_id"] == "s2"


def test_mcp_tool_adapters_expose_agent_switch_tools(monkeypatch):
    monkeypatch.setattr(
        "apps.mcp.server.load_active_agent_config",
        lambda: {"backend": "gemma4_mlx", "provider_label": "gemma4_mlx"},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.list_agent_backends",
        lambda: [{"id": "gemma4_mlx"}, {"id": "claude"}],
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.set_active_agent",
        lambda backend, **kwargs: {"backend": backend, **kwargs},
        raising=False,
    )
    monkeypatch.setattr(
        "apps.mcp.server.reload_agent_runtime",
        lambda: {"backend": "claude"},
        raising=False,
    )

    tools = build_tool_adapters(DummyService())

    assert tools["get_active_agent"]()["backend"] == "gemma4_mlx"
    assert len(tools["list_agent_backends"]()["backends"]) == 2
    assert tools["set_active_agent"](backend="claude")["backend"] == "claude"
    assert tools["reload_agent_runtime"]()["active_agent"]["backend"] == "claude"
