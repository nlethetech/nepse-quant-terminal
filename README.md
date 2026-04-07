# NEPSE Quant Clean

Local-first NEPSE trading workstation with a Textual terminal, paper/live execution stack, MCP control plane, NepalOSINT search, and a built-in Gemma 4 research agent.

The system is designed around one idea: the quantitative engine generates and ranks candidates, the agent explains and challenges them with filings/news/macro context, and the control plane decides what can actually be executed in paper, shadow-live, or live mode.

## What This App Does

- Runs a NEPSE-focused terminal trading workspace in [apps/tui/dashboard_tui.py](/apps/tui/dashboard_tui.py)
- Maintains a paper portfolio, paper order book, NAV log, and trade history
- Supports broker-assisted live and shadow-live trading through TMS execution intents
- Exposes the trading stack over MCP so external agents can inspect state and trigger approved actions
- Uses a local Gemma 4 MLX agent as the default built-in analyst
- Enriches stock analysis with:
  - quarterly financial caches
  - NepalOSINT semantic story search
  - NepalOSINT unified news + social search
  - related-story expansion
  - NRB forex rates
  - Kalimati commodity context
- Provides operator surfaces through:
  - Textual TUI
  - Telegram bot
  - MCP server
  - CLI scripts for research, ingestion, validation, and agent runs

## Main Entry Points

```bash
# Textual terminal
python3 -m apps.tui.dashboard_tui

# Live/paper engine
python3 -m backend.trading.live_trader --mode paper

# Legacy classic dashboard
python3 -m apps.classic.dashboard

# MCP server
make mcp-http
make mcp-stdio

# Local Gemma 4 analyst
make gemma-agent
make gemma-agent-ask Q="What is the market status right now? Use Nepal time."

# Run whichever built-in agent is currently active
make active-agent
make active-agent-ask Q="What is the market status right now? Use Nepal time."

# Daily ingestion + signal workflow
./scripts/ops/daily_run.sh
```

## Architecture

```text
apps/
  tui/
    dashboard_tui.py               Textual terminal UI
    dashboard_tui.tcss             TUI styling
  mcp/
    server.py                      MCP server for agents and remote clients
  classic/
    dashboard.py                   Legacy dashboard and research helpers

backend/
  agents/
    agent_analyst.py               Gemma 4 / Claude analyst, shortlist review, chat
  trading/
    live_trader.py                 Live + paper engine, signal generation, exits
    paper_trade_tracker.py         Paper portfolio updater / risk checks
    tui_trading_engine.py          TUI-side paper execution helpers
  quant_pro/
    control_plane/                Unified command, policy, read models, journal
    data_scrapers/                Quarterly reports, earnings, remittance, rainfall
    realtime_market.py            Snapshot provider for fresh NEPSE quotes
    signal_ranking.py             Cross-signal merge and ranking logic
    nepalosint_client.py          Semantic / unified / related story client
    telegram_bot.py               Interactive Telegram control surface
    tms_executor.py               Browser-backed TMS execution + watchlist ops
    tms_session.py                Live-mode secret loading and startup validation
    tms_audit.py                  Audit DB persistence for intents, results, snapshots
    macro_signals.py              NRB policy overlay
    event_layer.py                Event adjustment context for ranking
    quarterly_fundamental.py      Filing-driven factor logic
    alpha_practical.py            Daily scanner components

configs/
  long_term.py                    Main strategy profile used by the live/TUI stack
  mcp/                            Ready-to-use MCP client configs

scripts/
  ingestion/                      Deterministic market data ingestion
  signals/                        Daily signal generator
  agents/                         Local Codex / Gemma agent launchers
  mcp/                            MCP launch wrappers
  ops/                            Daily workflow scripts
  validation/                     Backtest / leakage / robustness scripts

data/
  nepse_market_data.db            Main SQLite market database
  runtime/                        Live audit DB, paper state, agent runtime files
  quarterly_reports/              Cached quarterly report extracts
  financial_reports/              Raw/cached financial report artifacts

tests/
  unit/                           Core unit and regression tests
```

## How The System Thinks

### 1. Market Data Layer

The app pulls and caches NEPSE data in SQLite, then force-refreshes intraday snapshots when the TUI or agent needs fresh context.

Main sources used by the runtime:

- NEPSE quotes and market breadth via [realtime_market.py](/backend/quant_pro/realtime_market.py)
- Vendor/API helpers via [vendor_api.py](/backend/quant_pro/vendor_api.py)
- Quarterly report caches via [quarterly_reports.py](/backend/quant_pro/data_scrapers/quarterly_reports.py)
- NRB rates and policy overlays via [macro_signals.py](/backend/quant_pro/macro_signals.py)
- Kalimati commodities via [kalimati_market.py](/backend/market/kalimati_market.py)
- NepalOSINT news/social search via [nepalosint_client.py](/backend/quant_pro/nepalosint_client.py)

### 2. Signal Generation

The live/TUI trading stack uses [generate_signals() in live_trader.py](/backend/trading/live_trader.py) with the active profile from [long_term.py](/configs/long_term.py).

Current default signal families:

- `volume`
- `quality`
- `low_vol`
- `mean_reversion`
- `disposition` (CGO breakout / disposition effect)
- `lead_lag` (sector spillover)
- `52wk_high`

Signal generation flow:

1. Load historical prices from SQLite.
2. Compute market regime.
3. If the regime filter is enabled and regime is `bear`, skip or downgrade entries.
4. Run each enabled signal family and collect raw candidates.
5. Apply Amihud illiquidity tilt to penalize names that look good but are too hard to trade.
6. Apply regime and NRB policy confidence multipliers.
7. Sort by `strength * confidence`.

### 3. How Stocks Are Initially Filtered

The app does not jump straight from raw signals to final trades. It filters in stages.

#### Stage A: Raw candidate generation

Each signal family emits candidates with:

- `symbol`
- `signal_type`
- `strength`
- `confidence`
- `reasoning`

#### Stage B: Merge duplicate names across strategies

[signal_ranking.py](/backend/quant_pro/signal_ranking.py) merges multiple signals for the same stock and creates a combined candidate using:

- the strongest primary signal
- support count from additional confirming signals
- a merged reasoning summary
- a raw score based on base signal quality plus multi-signal support

#### Stage C: Ranking penalties and adjustments

The ranker then reduces scores when necessary:

- removes already-held symbols from fresh-entry ranking
- penalizes sectors that are already heavy in the portfolio
- penalizes repeated signal-type concentration
- applies event-layer adjustments from [event_layer.py](/backend/quant_pro/event_layer.py)

This is why the final shortlist is more conservative than just “top signal score.”

#### Stage D: Portfolio and regime constraints

Before execution, the trader/control plane applies practical constraints:

- max positions
- sector concentration
- available cash
- halt/freeze state
- daily intent caps
- duplicate holding checks
- market-hours checks
- live approval rules

### 4. Agent Overlay

The agent is not the alpha generator. It is the research/risk overlay on top of the quant shortlist.

The built-in analyst lives in [agent_analyst.py](/backend/agents/agent_analyst.py) and currently:

- takes the top 10 ranked symbols
- force-refreshes the latest market snapshot
- loads quarterly metrics
- runs NepalOSINT semantic + unified + related-story search for each name
- injects NRB forex and Kalimati commodity context
- evaluates each stock into `APPROVE`, `HOLD`, or `REJECT`
- writes the analysis into the shared runtime file used by the TUI and MCP

The default primary agent is local Gemma 4 on MLX:

- model family: Gemma 4
- runtime: `mlx-vlm`
- launcher: [run_gemma_agent.py](/scripts/agents/run_gemma_agent.py)

The analyst is instructed to:

- use Nepal time only
- prefer missing-evidence honesty over hallucination
- treat filings as accounting truth
- treat NepalOSINT as event evidence
- treat the quant stack as timing context
- avoid generic `REVIEW/0%` outputs

## Control Plane

The control plane is the canonical orchestration layer for agents, TUI, Telegram, and MCP.

Modules:

- [models.py](/backend/quant_pro/control_plane/models.py)
- [policy_engine.py](/backend/quant_pro/control_plane/policy_engine.py)
- [command_service.py](/backend/quant_pro/control_plane/command_service.py)
- [decision_journal.py](/backend/quant_pro/control_plane/decision_journal.py)
- [read_models.py](/backend/quant_pro/control_plane/read_models.py)

What it does:

- normalizes `paper`, `live`, and `shadow_live`
- provides typed snapshots for market, portfolio, risk, and live state
- gates actions with policy checks
- routes paper orders and live intents through one service
- journals agent decisions, policy verdicts, and approvals

## Execution Modes

### Paper

- immediate local execution
- writes to paper portfolio / paper trade logs
- default safe mode for development and agent experimentation

### Shadow Live

- creates live-like intents and approval requests
- does not actually send broker orders
- safest way to validate agent behavior against the live workflow

### Live

- uses TMS execution intents and broker automation
- requires valid live settings and startup validation
- owner confirmation is the intended default for live actions

## Operator Surfaces

### Textual TUI

[dashboard_tui.py](/apps/tui/dashboard_tui.py) is the main operator interface.

It includes:

- market overview
- portfolio and P&L
- signals and lookup
- agent tab
- manual orders
- watchlist with forex and commodity panels
- account / TMS monitoring

### Telegram

[telegram_bot.py](/backend/quant_pro/telegram_bot.py) is the remote operator surface.

It supports:

- portfolio/status views
- trade activity summaries
- manual order flows
- approval flows for execution intents

### MCP

[server.py](/apps/mcp/server.py) exposes the stack to external agents.

Key MCP tools:

- `get_market_snapshot`
- `get_portfolio_snapshot`
- `get_signal_candidates`
- `get_risk_status`
- `semantic_story_search`
- `unified_osint_search`
- `related_story_search`
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
- `get_active_agent`
- `list_agent_backends`
- `set_active_agent`
- `reload_agent_runtime`
- `publish_agent_analysis`
- `append_agent_chat_message`

This lets Codex, Claude, local tools, or other MCP clients inspect the system and operate it through the same policy path.

## Runtime Files and State

Important runtime artifacts:

- [data/nepse_market_data.db](/data/nepse_market_data.db)
  - main market database
- [data/runtime/nepse_live_audit.db](/data/runtime/nepse_live_audit.db)
  - live audit DB for execution intents, approvals, and journal records
- [data/runtime/agents/agent_analysis.json](/data/runtime/agents/agent_analysis.json)
  - latest agent analysis shown in the TUI
- [data/runtime/agents/agent_chat_history.json](/data/runtime/agents/agent_chat_history.json)
  - recent visible agent chat
- [data/runtime/agents/active_agent.json](/data/runtime/agents/active_agent.json)
  - shared built-in agent selection used by MCP, the TUI, and the generic active-agent runner
- [data/runtime/trading/](/data/runtime/trading)
  - paper portfolio, paper orders, trade log, NAV log, and account runtime state

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # if you use one locally
```

For Apple Silicon local Gemma:

```bash
make gemma-agent-install
make gemma-agent
```

The first Gemma 4 run downloads the model. Later runs use the local cache.

## Switching Agents

Built-in agent selection is now runtime-configured, not just environment-driven.

Source of truth:

- [data/runtime/agents/active_agent.json](/data/runtime/agents/active_agent.json)

This file is read by:

- the TUI when it runs local analysis or chat
- the generic runner `scripts/agents/run_active_agent.py`
- MCP tools that inspect or change the active built-in backend

Supported built-in presets:

- `gemma4_mlx`
- `gemma4_experimental`
- `claude`

### Switch through MCP

Use these MCP tools:

- `get_active_agent`
- `list_agent_backends`
- `set_active_agent`
- `reload_agent_runtime`

Typical flow:

1. Call `list_agent_backends`
2. Call `set_active_agent(backend="claude")` or `set_active_agent(backend="gemma4_mlx")`
3. Call `reload_agent_runtime`
4. Run a fresh agent analysis or ask a new question in the TUI

### Switch from the terminal

Use the generic active-agent runner if you want to execute whichever backend is currently selected:

```bash
make active-agent
make active-agent-ask Q="What is the market doing right now?"
```

Use the explicit Gemma runner if you want to force Gemma for one execution regardless of the runtime selection:

```bash
make gemma-agent
make gemma-agent-ask Q="What is the market status right now? Use Nepal time."
```

### Precedence Rules

Agent selection resolves in this order:

1. Per-process environment overrides like `NEPSE_AGENT_BACKEND`
2. Runtime config in `data/runtime/agents/active_agent.json`
3. Built-in default `gemma4_mlx`

That means MCP switching changes the shared default, while one-off scripts can still override it for a single process.

## Useful Commands

```bash
# Start the TUI
python3 -m apps.tui.dashboard_tui

# Start the live trader in paper mode
python3 -m backend.trading.live_trader --mode paper

# Run the full daily workflow
./scripts/ops/daily_run.sh

# Launch MCP over HTTP
make mcp-http

# Launch MCP in shadow-live mode
make mcp-shadow

# Run Gemma analysis
make gemma-agent

# Ask Gemma a direct question
make gemma-agent-ask Q="What is the market status right now? Use Nepal time."

# Run the currently active built-in agent
make active-agent

# Ask the currently active built-in agent
make active-agent-ask Q="How would NEPSE react after the latest political news?"

# Run the core MCP / agent / TUI regression set
pytest tests/unit/test_agent_bridge.py tests/unit/test_mcp_server.py tests/unit/test_dashboard_tui_intraday.py -q
```

## Key Environment Variables

Core execution:

- `NEPSE_LIVE_EXECUTION_MODE`
- `NEPSE_MCP_TRADING_MODE`
- `NEPSE_MCP_DRY_RUN`
- `NEPSE_DB_FILE`

Agent selection and switching:

- `NEPSE_AGENT_BACKEND`
- `NEPSE_AGENT_MODEL`
- `NEPSE_AGENT_FALLBACK_BACKEND`
- `NEPSE_AGENT_PROVIDER_LABEL`
- `NEPSE_AGENT_SOURCE_LABEL`
- `NEPSE_AGENT_TRUST_REMOTE_CODE`

These env vars override the shared runtime agent selection when they are set for a process.

Live broker credentials:

- `NEPSE_TMS_SECRET_FILE`
- `NEPSE_TMS_USERNAME`
- `NEPSE_TMS_PASSWORD`

NepalOSINT (set base URL if you have access):

- `NEPALOSINT_BASE_URL` — override the default API base (see [API Access](#nepalosint-api-access) below)

## NepalOSINT API Access

The NepalOSINT endpoints used in this project (`/embeddings/search`, `/search/unified`, `/stories/:id/related`) are **not publicly open**.

To request access, contact **[@nlethetech on X](https://x.com/nlethetech)**.

## What Is Stable vs In Progress

Relatively stable:

- paper portfolio flow
- TUI main surfaces
- control-plane command routing
- MCP transport and tool mapping
- NepalOSINT integration
- local Gemma 4 batch analysis and chat

More iterative / still evolving:

- live broker automation
- agent auto-entry thresholds
- signal-family revalidation for newer extensions
- visual polish of some TUI tabs
- multi-agent orchestration

## Roadmap

These are roadmap items, not claims that they are fully complete today.

### Near Term

- Make the agent more portfolio-aware:
  - sector crowding
  - position replacement logic
  - better trim/sell reasoning for held names
- Improve the shortlist so more names get true cross-confirmation instead of pure timing signals
- Tighten event weighting from NepalOSINT and filings so generic momentum names get downgraded earlier
- Expand shadow-live evaluation and reporting

### Mid Term

- More robust live reconciliation between broker state and internal state
- Better order slicing / execution quality controls
- Richer macro layer:
  - rates
  - commodities
  - remittance
  - rainfall / hydro-sensitive features
- Stronger agent memory and session continuity across MCP clients

### Longer Term

- True digital hedge-fund workflow:
  - multiple specialized agents
  - PM/risk/execution separation
  - formal approval chains
  - research and execution journals tied together
- broader alternative data layer
- stronger factor research and validation loops
- better multi-account / multi-operator support

## Testing

```bash
pytest tests/unit -q
python3 ci/run_quality_gates.py --config ci/quality_gates.toml
python3 -m scripts.validation.test_leakage
```

## Notes

- Nepal time is the canonical market clock across the agent and trading stack.
- The control plane is the intended write surface. TUI, Telegram, and MCP should converge there.
- Use `shadow_live` before `live` whenever you are evaluating new automation or agent behavior.
