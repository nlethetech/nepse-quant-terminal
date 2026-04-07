# MCP Client Templates

These files are ready-to-copy client templates for the NEPSE control-plane MCP server.

## Files

- `claude_desktop_stdio.json`
  - Claude Desktop config using a local subprocess MCP server.
- `claude_desktop_http.json`
  - Claude Desktop config pointing at the local HTTP MCP endpoint.
- `codex_config.toml`
  - Codex CLI config snippet for stdio or HTTP registration.
- `generic_http_client.json`
  - Generic MCP-over-HTTP template for OpenAI-, Ollama-, or custom agent runtimes.

## Server Launch Options

### Stdio

```bash
make mcp-stdio
```

### HTTP

```bash
make mcp-http
```

Shadow-live HTTP:

```bash
make mcp-shadow
```

Default HTTP endpoint:

```text
http://127.0.0.1:8765/mcp
```

## Safety Defaults

- `paper` mode by default
- `NEPSE_MCP_DRY_RUN=true` by default
- Use `shadow_live` before `live`
- Keep `NEPSE_LIVE_OWNER_CONFIRM_REQUIRED=true` for broker-connected modes

## Secrets

Do not store broker credentials in workspace plaintext files.

Use either:

- `NEPSE_TMS_USERNAME` and `NEPSE_TMS_PASSWORD`
- `NEPSE_TMS_SECRET_FILE=/absolute/path/to/tms.env`
