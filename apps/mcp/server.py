"""Optional MCP server exposing the paper-trading NEPSE control plane."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

from backend.agents.agent_analyst import (
    append_external_agent_chat_message,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
    publish_external_agent_analysis,
    reload_agent_runtime,
)
from backend.agents.runtime_config import (
    list_agent_backends,
    load_active_agent_config,
    set_active_agent,
)
from backend.quant_pro.nepalosint_client import (
    related_stories as nepalosint_related_stories,
    semantic_story_search as nepalosint_semantic_story_search,
    unified_search as nepalosint_unified_search,
)
from backend.quant_pro.control_plane.command_service import (
    ControlPlaneCommandService,
    build_env_live_trader_control_plane,
)


def build_tool_adapters(service: ControlPlaneCommandService) -> Dict[str, Callable[..., Dict[str, Any]]]:
    return {
        "get_market_snapshot": lambda: service.get_market_snapshot(),
        "get_portfolio_snapshot": lambda: service.get_portfolio_snapshot(),
        "get_signal_candidates": lambda: service.get_signal_candidates(),
        "get_risk_status": lambda: service.get_risk_status(),
        "semantic_story_search": lambda query, hours=720, top_k=10, min_similarity=0.45: nepalosint_semantic_story_search(
            query,
            hours=hours,
            top_k=top_k,
            min_similarity=min_similarity,
        ),
        "unified_osint_search": lambda query, limit=10, election_year=None: nepalosint_unified_search(
            query,
            limit=limit,
            election_year=election_year,
        ),
        "related_story_search": lambda story_id, top_k=8, min_similarity=0.55, hours=8760: nepalosint_related_stories(
            story_id,
            top_k=top_k,
            min_similarity=min_similarity,
            hours=hours,
        ),
        "review_trade_candidate": lambda **kwargs: service.review_trade_candidate(**kwargs),
        "submit_paper_order": lambda **kwargs: service.submit_paper_order(**kwargs).to_record(),
        "halt_trading": lambda level="all", reason="mcp halt": service.halt_trading(level=level, reason=reason).to_record(),
        "resume_trading": lambda: service.resume_trading().to_record(),
        "get_agent_tab_state": lambda: {
            "analysis": load_agent_analysis(),
            "history": load_agent_history(limit=20),
            "archive_count": len(load_agent_archive_history()),
            "active_agent": load_active_agent_config(),
        },
        "get_active_agent": lambda: load_active_agent_config(),
        "list_agent_backends": lambda: {
            "active_agent": load_active_agent_config(),
            "backends": list_agent_backends(),
        },
        "set_active_agent": lambda backend, model=None, provider_label=None, source_label=None, fallback_backend=None, trust_remote_code=None: set_active_agent(
            backend,
            model=model,
            provider_label=provider_label,
            source_label=source_label,
            fallback_backend=fallback_backend,
            trust_remote_code=trust_remote_code,
        ),
        "reload_agent_runtime": lambda: {
            "active_agent": reload_agent_runtime(),
        },
        "publish_agent_analysis": lambda analysis, source="mcp_external", provider="external": publish_external_agent_analysis(
            analysis,
            source=source,
            provider=provider,
        ),
        "append_agent_chat_message": lambda role, message, source="mcp_external", provider="external": {
            "history": append_external_agent_chat_message(
                role,
                message,
                source=source,
                provider=provider,
            )
        },
    }


def build_server(
    service: ControlPlaneCommandService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
):
    service = service or build_env_live_trader_control_plane()
    adapters = build_tool_adapters(service)
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The optional `mcp` package is required to run apps.mcp.server. "
            "Install it in the project environment, then rerun the server."
        ) from exc

    server = FastMCP(
        "nepse-control-plane",
        host=host,
        port=int(port),
        streamable_http_path=streamable_http_path,
    )

    @server.tool()
    def get_market_snapshot():
        return adapters["get_market_snapshot"]()

    @server.tool()
    def get_portfolio_snapshot():
        return adapters["get_portfolio_snapshot"]()

    @server.tool()
    def get_signal_candidates():
        return adapters["get_signal_candidates"]()

    @server.tool()
    def get_risk_status():
        return adapters["get_risk_status"]()

    @server.tool()
    def semantic_story_search(
        query: str,
        hours: int = 720,
        top_k: int = 10,
        min_similarity: float = 0.45,
    ):
        return adapters["semantic_story_search"](
            query=query,
            hours=hours,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    @server.tool()
    def unified_osint_search(
        query: str,
        limit: int = 10,
        election_year: int | None = None,
    ):
        return adapters["unified_osint_search"](
            query=query,
            limit=limit,
            election_year=election_year,
        )

    @server.tool()
    def related_story_search(
        story_id: str,
        top_k: int = 8,
        min_similarity: float = 0.55,
        hours: int = 8760,
    ):
        return adapters["related_story_search"](
            story_id=story_id,
            top_k=top_k,
            min_similarity=min_similarity,
            hours=hours,
        )

    @server.tool()
    def review_trade_candidate(
        mode: str,
        action: str,
        symbol: str,
        quantity: int,
        limit_price: float | None = None,
        thesis: str = "",
        catalysts: list[str] | None = None,
        risk: list[str] | None = None,
        confidence: float = 0.0,
        horizon: str = "",
        source_signals: list[str] | None = None,
        target_order_ref: str | None = None,
        operator_surface: str = "mcp",
    ):
        return adapters["review_trade_candidate"](
            mode=mode,
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            thesis=thesis,
            catalysts=catalysts,
            risk=risk,
            confidence=confidence,
            horizon=horizon,
            source_signals=source_signals,
            target_order_ref=target_order_ref,
            operator_surface=operator_surface,
        )

    @server.tool()
    def submit_paper_order(action: str, symbol: str, quantity: int, limit_price: float):
        return adapters["submit_paper_order"](
            action=action,
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
        )

    @server.tool()
    @server.tool()
    def halt_trading(level: str = "all", reason: str = "mcp halt"):
        return adapters["halt_trading"](level=level, reason=reason)

    @server.tool()
    def resume_trading():
        return adapters["resume_trading"]()

    @server.tool()
    def get_agent_tab_state():
        return adapters["get_agent_tab_state"]()

    @server.tool()
    def get_active_agent():
        return adapters["get_active_agent"]()

    @server.tool()
    def list_agent_backends():
        return adapters["list_agent_backends"]()

    @server.tool()
    def set_active_agent(
        backend: str,
        model: str | None = None,
        provider_label: str | None = None,
        source_label: str | None = None,
        fallback_backend: str | None = None,
        trust_remote_code: bool | None = None,
    ):
        return adapters["set_active_agent"](
            backend=backend,
            model=model,
            provider_label=provider_label,
            source_label=source_label,
            fallback_backend=fallback_backend,
            trust_remote_code=trust_remote_code,
        )

    @server.tool()
    def reload_agent_runtime():
        return adapters["reload_agent_runtime"]()

    @server.tool()
    def publish_agent_analysis(
        analysis: dict,
        source: str = "mcp_external",
        provider: str = "external",
    ):
        return adapters["publish_agent_analysis"](analysis=analysis, source=source, provider=provider)

    @server.tool()
    def append_agent_chat_message(
        role: str,
        message: str,
        source: str = "mcp_external",
        provider: str = "external",
    ):
        return adapters["append_agent_chat_message"](
            role=role,
            message=message,
            source=source,
            provider=provider,
        )

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NEPSE MCP control-plane server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to expose.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP-based transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP-based transports.")
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Path to mount the streamable HTTP endpoint.",
    )
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path used by the SSE transport.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
    )
    server.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
