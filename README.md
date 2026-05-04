#NEPSE Quant Terminal

A terminal-based quantitative trading dashboard for the Nepal Stock Exchange (NEPSE), built with [Textual](https://textual.textualize.io/). Runs entirely in your terminal — no browser, no electron, no cloud dependency.

**Paper trading only.** This terminal simulates trades locally. It does not connect to any broker API.

---

## What It Does

- **Paper Trading** — full paper portfolio with buy/sell order book, P&L tracking, NAV history, and multi-account support. Seed from your MeroShare holdings CSV or start blank.
- **Auto Trading Engine** — assigns a quantitative strategy to each account. The engine runs in the background, generates signals every 5 trading days, and manages entries/exits automatically (holding periods, stop losses, trailing stops, regime filters).
- **Backtesting** — walk-forward validated backtests on 6+ years of NEPSE price data. Ships with C5 baseline: **+88% OOS return, Sharpe 2.2** vs. NEPSE +27%.
- **Market Dashboard** — live quotes, 52-week highs/lows, top movers, sector heatmap, volume signals.
- **Portfolio Analytics** — unrealized/realized P&L, sector concentration, holding age buckets, max drawdown, alpha vs. NEPSE benchmark.
- **Gold Hedge Overlay** — tracks gold/silver regime (risk-on / neutral / risk-off) and adjusts capital deployment accordingly.
- **AI Agent** — on-demand analysis of your portfolio positions and signal shortlist. Defaults to a local Ollama model, with Gemma 4 MLX or Claude CLI available as optional backends.
- **Paper Agent Graph** — cleaned evidence-gated research, debate, risk, and portfolio decision workflow for paper execution only.
- **Strategy Builder** — create, backtest, and assign custom strategies. Each account runs its own strategy independently.
- **Statistical Validation** — walk-forward OOS testing, Monte Carlo, CSCV/PBO overfitting detection, deflated Sharpe ratio, random baseline percentile.
- **MeroShare Import** — seed any account directly from your MeroShare "My Shares Values.csv" export.

---

## Architecture

### Paper Agent Workflow

The public agent workflow is evidence-gated, checkpointed, and restricted to paper execution. The implementation in `backend/nepse_agents/` does not include live order routing, credentials, or execution integrations.

![NEPSE Quant Terminal paper agent architecture](docs/assets/nepse-agent-architecture.png)

```
┌─────────────────────────────────────────────────────┐
│                   Textual TUI                       │
│  dashboard_tui.py  ·  9 tabs  ·  keyboard-driven    │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
    ┌──────▼──────┐      ┌────────▼────────┐
    │  Market     │      │  Trading Engine │
    │  Data Layer │      │  (per account)  │
    │  nepse_data │      │  tui_trading_   │
    │  .db        │      │  engine.py      │
    └──────┬──────┘      └────────┬────────┘
           │                      │
    ┌──────▼──────────────────────▼────────┐
    │          Signal Engine               │
    │  simple_backtest.py                  │
    │  volume · quality · low_vol ·        │
    │  mean_reversion · xsec_momentum ·    │
    │  quarterly_fundamental · satellite   │
    └──────────────────────────────────────┘
```

### Components

| Component | File | What it does |
|---|---|---|
| TUI | `apps/tui/dashboard_tui.py` | All UI — 9 tabs, keyboard shortcuts, paper order book |
| Trading Engine | `backend/trading/tui_trading_engine.py` | Per-account auto-trading loop, regime filter, stop logic |
| Paper Trader | `backend/trading/paper_trader.py` | Manual buy/sell execution, portfolio persistence |
| Signal Engine | `backend/backtesting/simple_backtest.py` | All signal generation and backtest runner |
| Strategy Registry | `backend/trading/strategy_registry.py` | Load, save, assign strategies per account |
| Market Data | `backend/market/` | Price DB queries, quote scraping, 52wk calculations |
| Validation | `validation/` | Walk-forward, Monte Carlo, CSCV, DSR, random baseline |
| Gold Hedge | `backend/quant_pro/gold_hedge.py` | Gold/silver regime detection → capital deployment % |
| AI Agent | `backend/agents/agent_analyst.py` | Ollama-first portfolio and signal analysis |
| Paper Agent Graph | `backend/nepse_agents/` | Evidence-gated research/debate/risk/portfolio workflow; paper-only |
| NEPSE Calendar | `nepse_calendar.py` | Sun–Thu trading days, public holidays, trading day counter |

---

## How Paper Trading Works

Each account has its own directory under `data/runtime/accounts/account_N/` containing:

```
paper_portfolio.csv      # open positions
paper_trade_log.csv      # all executed trades
paper_nav_log.csv        # daily NAV history
paper_state.json         # cash balance + runtime state
tui_paper_*              # engine auto-trade files
watchlist.json           # symbols to track
```

**Manual trading** (Order tab) writes to `paper_portfolio.csv`.
**Auto-trading** (engine) writes to `tui_paper_trade_log.csv` and reconciles with the manual portfolio on display.

The Trade History tab merges both sources and deduplicates by `(Date, Action, Symbol, Shares, Price)`.

---

## How the Signal Engine Works

Signals are generated per trading date using price + fundamental data from `nepse_data.db`. Each signal scores symbols 0.0–1.0. Signals are combined with regime-dependent weights:

```
Bull market  → xsec_momentum weight ×1.1, all others ×1.0
Bear market  → capital preservation mode (fewer positions)
Neutral      → standard weights
```

Regime is detected via a 60-day rolling NEPSE return: bear below threshold, bull above 0, neutral in between.

The engine runs a **5-trading-day signal cycle** — signals fire every 5 days, not daily, avoiding overtrading and matching NEPSE's lower liquidity.

### Available Signals

| Signal | Logic |
|---|---|
| `volume` | Volume breakout above 20-day average with price confirmation |
| `quality` | ROE + debt-to-equity + earnings stability composite |
| `low_vol` | Low 60-day realized volatility with positive momentum |
| `mean_reversion` | RSI oversold + distance below 52-week high |
| `xsec_momentum` | Cross-sectional 6m-minus-1m momentum (skip last month) |
| `quarterly_fundamental` | EPS growth + revenue growth from quarterly filings |
| `satellite_hydro` | Hydropower generation signals from WECS rainfall data |

---

## How Backtesting Works

```python
from backend.backtesting.simple_backtest import run_backtest

results = run_backtest(
    signal_types=["volume", "quality", "low_vol", "mean_reversion",
                  "xsec_momentum", "quarterly_fundamental"],
    holding_days=40,
    max_positions=5,
    stop_loss_pct=0.12,
    trailing_stop_pct=0.15,
    use_regime_filter=True,
    initial_capital=1_000_000,
)
```

Walk-forward validation splits 6+ years of history into rolling train/test windows and stitches OOS equity:

```bash
python -m validation.run_all --fast
```

Outputs: OOS equity curve, Sharpe, max drawdown, CSCV/PBO score, deflated Sharpe ratio, random baseline percentile.

---

## How the Auto-Trading Engine Works

When the TUI starts, one `TUITradingEngine` per account starts in a background daemon thread. Each engine:

1. Loads its account's strategy config (signal types, holding days, stop params)
2. Every 5 trading days: generates signals → ranks → buys top N symbols up to `max_positions`
3. Every day: checks exits — trailing stop, stop loss, or holding period expiry
4. Writes trades to `tui_paper_trade_log.csv` for that account
5. Persists state so it survives TUI restarts

Capital deployment adjusts by the gold hedge regime:
- **Risk-off** → 90% of capital deployed
- **Neutral** → 97%
- **Risk-on** → 100%

---

## How Strategies Work

A strategy is a JSON config in `data/strategy_registry/`:

```json
{
  "id": "my_strategy",
  "name": "My Strategy",
  "config": {
    "signal_types": ["volume", "quality", "xsec_momentum"],
    "holding_days": 40,
    "max_positions": 5,
    "stop_loss_pct": 0.12,
    "trailing_stop_pct": 0.15,
    "use_regime_filter": true,
    "regime_max_positions": {"bull": 5, "neutral": 4, "bear": 1},
    "sector_limit": 0.35
  }
}
```

Create strategies in the **Strategies tab** → press **N NEW**, configure signals with the toggle buttons, set parameters, press **SAVE**. Assign to any account with **→ ACTIVE ACCT**.

---

## Adding Custom Signals

Implement a function in `backend/backtesting/simple_backtest.py`:

```python
def generate_my_signal_at_date(
    symbols: list[str],
    date: str,
    prices_df: pd.DataFrame,
) -> list[dict]:
    # Return list of {"symbol": str, "score": float 0-1, "reason": str}
    ...
```

Register it in the `SIGNAL_MAP` dict inside `run_backtest()` and add `"my_signal"` to any strategy's `signal_types`.

---

## Setup

### Requirements

- Python 3.10–3.13 (recommended: 3.12) — Python 3.14+ is **not yet supported** (numba and the nepse package both cap at `<3.14`)
- macOS or Linux (Windows: WSL recommended)

### Installation

```bash
git clone https://github.com/nlethetech/nepse-quant-terminal
cd nepse-quant-terminal
pip install -r requirements.txt
```

### Database

Run the setup script — it downloads the pre-built database (~13 MB) from the GitHub release automatically:

```bash
python setup_data.py
```

Takes under a minute. You get 456K rows of OHLCV history for all NEPSE symbols, quarterly earnings, corporate actions, and benchmark history — enough for the signal engine, backtests, and charts to work immediately.

> **Important:** The bundled database is a snapshot. For accurate signals and backtests you must keep it up to date with fresh scraped data. The pre-built DB covers history through the release date — anything after that requires running the scraper.

**Scrape fresh data yourself (recommended for production use):**
```bash
python setup_data.py --scrape           # full historical scrape from Merolagani (~30–60 min)
python setup_data.py --scrape --days 90 # last 90 days only (~5 min)
```

**Daily incremental update** — run this after market close each day to keep data current:
```bash
python scripts/ingestion/deterministic_daily_ingestion.py
```

> The signal engine (volume breakout, momentum, quarterly fundamental, etc.) relies on recent price history. Stale data = stale signals. Set up a daily cron job or run the ingestion script manually each evening.

### Run

```bash
python -m apps.tui.dashboard_tui
```

---

## AI Agent Setup

The Agents tab provides on-demand equity analysis of your signal shortlist and portfolio. Ollama is the default backend; Gemma 4 MLX and Claude CLI are optional.

### Option 1 — Ollama (recommended, any hardware)

Ollama runs any open-source model locally via a simple REST server. No Apple Silicon required.

**Step 1 — Install Ollama**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Step 2 — Pull a model**

Pick one based on your available RAM:

| Model | RAM | Command | Notes |
|---|---|---|---|
| `llama3` | 8 GB | `ollama pull llama3` | Good all-rounder |
| `mistral` | 8 GB | `ollama pull mistral` | Fast, sharp reasoning |
| `phi3` | 4 GB | `ollama pull phi3` | Runs on low-end hardware |
| `qwen2` | 8 GB | `ollama pull qwen2` | Strong on structured output |
| `llama3:70b` | 40 GB | `ollama pull llama3:70b` | Best quality, high-end only |

**Step 3 — Start the Ollama server**
```bash
ollama serve
# Runs at http://localhost:11434 by default
```

**Step 4 — Optional: change the default model**

The terminal defaults to:

```json
{
  "selected_preset": "ollama",
  "backend": "ollama",
  "model": "llama3",
  "ollama_host": "http://localhost:11434"
}
```

To use a different Ollama model, edit `data/runtime/agents/active_agent.json` after first run:
```json
{
  "selected_preset": "ollama",
  "backend": "ollama",
  "model": "mistral",
  "ollama_host": "http://localhost:11434"
}
```

Or switch from inside the TUI: **Agents tab → Agent Settings → enter model name**.

**Step 5 — Run the terminal**
```bash
python -m apps.tui.dashboard_tui
```

Open the Agents tab and hit **Analyze** — the agent will pull your current shortlist and return a structured bull/bear breakdown per stock.

---

### Option 2 — Gemma 4 MLX (Apple Silicon only)

Runs Gemma 4 directly in-process via MLX. No server needed — faster response on M-series chips.

```bash
pip install mlx-vlm
# Model (~3 GB) downloads automatically on first use
```

To use this backend, select the `gemma4_mlx` preset in the Agents tab or in `data/runtime/agents/active_agent.json`.

---

### Option 3 — Claude CLI

Claude is optional and never used as the default fallback.

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code   # or via brew

# Authenticate
claude login
```

Then set `active_agent.json` backend to `"claude"`, or switch from the TUI.

---

### Switching backends at runtime

All three backends are hot-swappable without restarting the terminal. In the TUI:

```
Agents tab → Agent Settings (gear icon) → select backend → Save
```

The setting persists to `data/runtime/agents/active_agent.json`.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `1`–`9` | Switch tabs |
| `R` | Refresh market data |
| `B` | Buy (paper) |
| `S` | Sell (paper) |
| `N` | New account |
| `A` | Activate account |
| `W` | Sync watchlist |
| `H` | Help / shortcuts |
| `Q` | Quit |

---

## Project Structure

```
nepse-quant-terminal/
├── apps/tui/
│   ├── dashboard_tui.py        # Main TUI application
│   └── dashboard_tui.tcss      # Textual CSS styles
├── backend/
│   ├── trading/
│   │   ├── paper_trader.py         # Manual paper order execution
│   │   ├── live_trader.py          # Portfolio persistence utilities
│   │   ├── tui_trading_engine.py   # Per-account auto-trading engine
│   │   └── strategy_registry.py    # Strategy load/save/assign
│   ├── backtesting/
│   │   └── simple_backtest.py      # Signal engine + backtest runner
│   ├── market/                     # Market data, quotes, scraping
│   ├── agents/                     # AI agent (Ollama / Gemma MLX / Claude)
│   └── quant_pro/
│       ├── gold_hedge.py           # Gold regime overlay
│       ├── satellite_data.py       # Hydropower signal data
│       ├── regime_detection.py     # Market regime classifier
│       └── paths.py                # Project path utilities
├── configs/
│   └── long_term.py            # Default strategy parameters
├── data/
│   └── strategy_registry/      # Strategy JSON configs
├── validation/                 # Statistical validation suite
│   ├── walk_forward.py
│   ├── monte_carlo.py
│   ├── cscv_pbo.py
│   ├── statistical_tests.py
│   └── run_all.py
├── nepse_calendar.py           # NEPSE trading calendar (Sun–Thu)
└── requirements.txt
```

---

## Notes

- **Paper trading only.** No broker API. All trades are simulated locally.
- NEPSE trades **Sunday–Thursday**. The calendar module handles all public holidays.
- Holding periods are in **trading days**, not calendar days. 40 trading days ≈ 8 NEPSE weeks.
- The backtest includes realistic transaction costs: SEBON levy, broker commission, DP charges.
- The gold hedge module uses Nepal Rastra Bank gold price data — no external API required.

---

## License

MIT
