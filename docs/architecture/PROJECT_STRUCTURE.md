# Project Structure

Core app/runtime modules are now grouped to reduce the top-level sprawl:

- `apps/classic/`
  - legacy terminal dashboard implementation
- `apps/tui/`
  - Textual TUI implementation and its stylesheet
- `backend/agents/`
  - agent analysis and chat logic
- `backend/trading/`
  - live trading, paper tracking, and TUI trading engine
- `backend/market/`
  - Kalimati and related market-data helpers
- `backend/quant_pro/`
  - strategy, research, data, and reporting package
- `scripts/`, `ci/`, `tests/`, `configs/`, `data/`, `models/`, `research/`, `validation/`
  - supporting automation, validation, and generated outputs

Primary entrypoints now live inside the package/script folders:

- `python -m apps.tui.dashboard_tui`
- `python -m apps.classic.dashboard`
- `python -m backend.trading.live_trader`
- `python -m backend.trading.paper_trade_tracker`
