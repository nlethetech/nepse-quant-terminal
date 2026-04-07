# MCP Agent Setup

The NEPSE control plane can now be exposed in two ways:

- `stdio` for desktop agents and CLIs that launch local subprocess MCP servers.
- `streamable-http` for tools that prefer a URL-based MCP endpoint.

## Recommended Modes

- Use `stdio` for Codex CLI and Claude Desktop when the agent can launch local commands.
- Use `streamable-http` for local agent frameworks, browser-based tools, or Ollama-powered orchestrators that accept an MCP URL.

## Launch Commands

### 1. Stdio

```bash
./scripts/mcp/run_stdio_server.sh
```

### 2. Streamable HTTP

```bash
./scripts/mcp/run_http_server.sh
```

Defaults:

- Host: `127.0.0.1`
- Port: `8765`
- Endpoint: `/mcp`
- Trading mode: `paper`
- Dry-run: `true`

Override example:

```bash
MCP_PORT=9000 NEPSE_MCP_TRADING_MODE=shadow_live ./scripts/mcp/run_http_server.sh
```

## Exposed Tools

- `get_market_snapshot`
- `get_portfolio_snapshot`
- `get_signal_candidates`
- `get_risk_status`
- `review_trade_candidate`
- `submit_paper_order`
- `create_live_intent`
- `confirm_live_intent`
- `cancel_live_intent`
- `modify_live_intent`
- `reconcile_live_state`
- `halt_trading`
- `resume_trading`
- `sync_watchlist`
- `get_agent_tab_state`
- `publish_agent_analysis`
- `append_agent_chat_message`

## Agent Tab Bridge

External MCP-driven agents can now publish directly into the TUI Agent tab runtime.

Recommended flow:

1. Read context from:
   - `get_market_snapshot`
   - `get_portfolio_snapshot`
   - `get_signal_candidates`
   - `get_risk_status`
   - `get_agent_tab_state`
2. Make decisions externally.
3. Push the result back with:
   - `publish_agent_analysis`
   - `append_agent_chat_message`

This keeps one shared agent surface in the TUI without giving external agents direct UI ownership.

Good default:

- Treat the agent tab as a shared operator view.
- Treat MCP as the execution and orchestration surface.

Less good default:

- Making a remote agent the only source of truth for all TUI state.

The bridge is a good idea. Full hard-coupling of the tab to one external agent is usually not.

## Codex As The Visible Agent

If you want Codex to become the visible agent in the TUI Agent tab, run:

```bash
make codex-agent
```

Safe live-style dry run:

```bash
make codex-agent-shadow
```

That command:

1. Starts the local MCP HTTP server if needed
2. Runs a Codex subprocess against the MCP endpoint
3. Publishes Codex analysis into the shared agent runtime files
4. Makes the TUI Agent tab show Codex as the current provider

## Codex CLI Example

```bash
codex mcp add nepse_control_plane -- ./scripts/mcp/run_stdio_server.sh
```

Or via HTTP:

```bash
./scripts/mcp/run_http_server.sh
codex mcp add nepse_control_plane_http --url http://127.0.0.1:8765/mcp
```

## Claude Desktop Example

Add this to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "nepse-control-plane": {
      "command": "<PROJECT_ROOT>/scripts/mcp/run_stdio_server.sh"
    }
  }
}
```

If Claude Desktop requires an explicit interpreter instead:

```json
{
  "mcpServers": {
    "nepse-control-plane": {
      "command": "python3",
      "args": [
        "-m",
        "apps.mcp.server",
        "--transport",
        "stdio"
      ],
      "cwd": "<PROJECT_ROOT>"
    }
  }
}
```

## Local/Ollama Agent Example

For agent frameworks that accept a remote MCP URL, point them at:

```text
http://127.0.0.1:8765/mcp
```

If the framework launches subprocess MCP servers instead, use:

```text
<PROJECT_ROOT>/scripts/mcp/run_stdio_server.sh
```

Repo templates:

- [claude_desktop_stdio.json](/configs/mcp/claude_desktop_stdio.json)
- [claude_desktop_http.json](/configs/mcp/claude_desktop_http.json)
- [codex_config.toml](/configs/mcp/codex_config.toml)
- [generic_http_client.json](/configs/mcp/generic_http_client.json)

## Safety Defaults

- `paper` mode is the default.
- `NEPSE_MCP_DRY_RUN=true` is the default.
- For broker-connected modes, keep `NEPSE_LIVE_OWNER_CONFIRM_REQUIRED=true`.
- Prefer `shadow_live` before `live`.

## Secrets

Do not store broker credentials in workspace plaintext files.

Preferred options:

- `NEPSE_TMS_USERNAME` and `NEPSE_TMS_PASSWORD` in the process environment
- `NEPSE_TMS_SECRET_FILE=/path/to/tms.env`

Example external secret file:

```bash
NEPSE_TMS_USERNAME=your_user
NEPSE_TMS_PASSWORD=your_password
```
