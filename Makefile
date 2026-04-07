PYTHON ?= python3
MCP_HOST ?= 127.0.0.1
MCP_PORT ?= 8765
MCP_PATH ?= /mcp
NEPSE_MCP_TRADING_MODE ?= paper
NEPSE_MCP_DRY_RUN ?= true

.PHONY: mcp-stdio mcp-http mcp-shadow mcp-test codex-agent codex-agent-shadow gemma-agent gemma-agent-ask gemma-agent-install

mcp-stdio:
	NEPSE_MCP_TRADING_MODE=$(NEPSE_MCP_TRADING_MODE) \
	NEPSE_MCP_DRY_RUN=$(NEPSE_MCP_DRY_RUN) \
	./scripts/mcp/run_stdio_server.sh

mcp-http:
	MCP_HOST=$(MCP_HOST) \
	MCP_PORT=$(MCP_PORT) \
	MCP_PATH=$(MCP_PATH) \
	NEPSE_MCP_TRADING_MODE=$(NEPSE_MCP_TRADING_MODE) \
	NEPSE_MCP_DRY_RUN=$(NEPSE_MCP_DRY_RUN) \
	./scripts/mcp/run_http_server.sh

mcp-shadow:
	MCP_HOST=$(MCP_HOST) \
	MCP_PORT=$(MCP_PORT) \
	MCP_PATH=$(MCP_PATH) \
	NEPSE_MCP_TRADING_MODE=shadow_live \
	NEPSE_MCP_DRY_RUN=true \
	./scripts/mcp/run_http_server.sh

mcp-test:
	$(PYTHON) -m pytest tests/unit/test_mcp_server.py tests/unit/test_agent_bridge.py -q

codex-agent:
	NEPSE_MCP_TRADING_MODE=paper \
	NEPSE_MCP_DRY_RUN=true \
	$(PYTHON) scripts/agents/run_codex_agent.py --mode paper --dry-run true

codex-agent-shadow:
	NEPSE_MCP_TRADING_MODE=shadow_live \
	NEPSE_MCP_DRY_RUN=true \
	$(PYTHON) scripts/agents/run_codex_agent.py --mode shadow_live --dry-run true

gemma-agent-install:
	$(PYTHON) -m pip install -U mlx-vlm

gemma-agent:
	NEPSE_AGENT_BACKEND=gemma4_mlx \
	NEPSE_AGENT_FALLBACK_BACKEND=claude \
	$(PYTHON) scripts/agents/run_gemma_agent.py --force

gemma-agent-ask:
	@if [ -z "$(Q)" ]; then echo "Usage: make gemma-agent-ask Q='what is the market doing?'"; exit 1; fi
	NEPSE_AGENT_BACKEND=gemma4_mlx \
	NEPSE_AGENT_FALLBACK_BACKEND=claude \
	$(PYTHON) scripts/agents/run_gemma_agent.py --question "$(Q)"
