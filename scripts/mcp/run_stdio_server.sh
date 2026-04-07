#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export NEPSE_MCP_TRADING_MODE="${NEPSE_MCP_TRADING_MODE:-paper}"
export NEPSE_MCP_DRY_RUN="${NEPSE_MCP_DRY_RUN:-true}"

exec python3 -m apps.mcp.server --transport stdio "$@"
