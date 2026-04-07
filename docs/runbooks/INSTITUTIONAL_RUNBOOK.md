# Institutional Runbook

This repo now includes three institutional blocks:

1. Deterministic ingestion with run metadata + freshness SLA.
2. Portfolio/risk state machine with immutable trade ledger.
3. CI quality gates for data, leakage, and strategy drift.

## 1) Deterministic Ingestion

Run full-universe ingestion:

```bash
python3 -m scripts.ingestion.deterministic_daily_ingestion --source both --strict --max-staleness-days 14
```

Useful options:

- `--max-symbols 20` for controlled test runs.
- `--watchlist-id 1` for watchlist scoped ingestion.
- `--symbols-file symbols.txt` for explicit universe.
- `--backfill-days 5` overlap to capture vendor corrections.

Metadata tables:

- `ingestion_runs`
- `ingestion_run_symbols`

## 2) Portfolio / Risk Engine

Open a position:

```bash
python3 -m scripts.portfolio.institutional_portfolio_engine --action open --symbol NABIL --qty 100 --price 520
```

Open from generated orders:

```bash
python3 -m scripts.portfolio.institutional_portfolio_engine --action open-from-orders --orders-file buy_orders.csv --use-live-prices
```

Run risk check (dry-run):

```bash
python3 -m scripts.portfolio.institutional_portfolio_engine --action risk-check --use-live-prices
```

Run risk check and apply exits:

```bash
python3 -m scripts.portfolio.institutional_portfolio_engine --action risk-check --use-live-prices --apply
```

Close manually:

```bash
python3 -m scripts.portfolio.institutional_portfolio_engine --action close --position-id 1 --price 550 --reason MANUAL_EXIT
```

Immutable ledger:

- `trade_ledger` is append-only (DB triggers block `UPDATE`/`DELETE`).
- `portfolio_positions` maintains current state.

## 3) CI Quality Gates

Run locally:

```bash
python3 ci/run_quality_gates.py --config ci/quality_gates.toml
```

Gate categories:

- Data freshness + integrity.
- Ingestion recency/SLA.
- Leakage audit (`test_leakage.py`).
- Backtest regression thresholds.

GitHub Action workflow:

- `.github/workflows/quant_quality.yml`

