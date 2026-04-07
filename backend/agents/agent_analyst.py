"""
NEPSE Agent Analyst — local-first equity research overlay.

Uses a local Gemma 4 model on MLX as the primary agent, with optional Claude
fallback, to perform structured bull/bear analysis on algorithmic shortlists,
cross-referencing OSINT intelligence, quarterly financials, and market regime.

Two modes:
  1. analyze() — batch analysis of shortlisted stocks (one agent call)
  2. ask()     — interactive Q&A with full context injection
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from configs.long_term import LONG_TERM_CONFIG
from backend.agents.runtime_config import (
    DEFAULT_GEMMA4_MODEL as RUNTIME_DEFAULT_GEMMA4_MODEL,
    EXPERIMENTAL_GEMMA4_MODEL as RUNTIME_EXPERIMENTAL_GEMMA4_MODEL,
    load_active_agent_config,
)
from backend.quant_pro.nepse_calendar import current_nepal_datetime, get_market_schedule, market_session_phase
from backend.quant_pro.control_plane.models import AgentDecision
from backend.quant_pro.nepalosint_client import symbol_intelligence, unified_search
from backend.quant_pro.paths import ensure_dir, get_project_root, get_runtime_dir, migrate_legacy_path

RUNTIME_DIR = ensure_dir(get_runtime_dir(__file__))
AGENTS_RUNTIME_DIR = ensure_dir(RUNTIME_DIR / "agents")
PROJECT_ROOT = get_project_root(__file__)
ANALYSIS_FILE = migrate_legacy_path(AGENTS_RUNTIME_DIR / "agent_analysis.json", [PROJECT_ROOT / "agent_analysis.json"])
AGENT_HISTORY_FILE = migrate_legacy_path(
    AGENTS_RUNTIME_DIR / "agent_chat_history.json",
    [PROJECT_ROOT / "agent_chat_history.json"],
)
AGENT_ARCHIVE_FILE = migrate_legacy_path(
    AGENTS_RUNTIME_DIR / "agent_chat_archive.json",
    [PROJECT_ROOT / "agent_chat_archive.json"],
)
OSINT_BASE = "https://nepalosint.com/api/v1"
MAX_AGENT_HISTORY_ITEMS = 12
MAX_AGENT_ARCHIVE_ITEMS = 240
AGENT_SHORTLIST_LIMIT = 10
AGENT_OSINT_DECISION_HOURS = 24
SUPER_SIGNAL_MIN_SCORE = 0.75
SUPER_SIGNAL_MIN_STRENGTH = 1.0
SUPER_SIGNAL_MIN_CONFIDENCE = 0.75
SUPER_SIGNAL_MIN_CONVICTION = 0.9
ANALYSIS_CACHE_MAX_AGE_SECS = 900
DEFAULT_AGENT_BACKEND = "gemma4_mlx"
DEFAULT_GEMMA4_MLX_MODEL = RUNTIME_DEFAULT_GEMMA4_MODEL
DEFAULT_GEMMA4_EXPERIMENTAL_MODEL = RUNTIME_EXPERIMENTAL_GEMMA4_MODEL
DEFAULT_AGENT_MAX_TOKENS = 4000
DEFAULT_AGENT_CHAT_MAX_TOKENS = 320
DEFAULT_AGENT_TEMPERATURE = 0.15
DEFAULT_AGENT_TOP_P = 0.9

POSITIVE_OSINT_TERMS = {
    "profit", "earnings", "dividend", "bonus", "rights", "approved", "approval",
    "contract", "expansion", "growth", "surge", "record", "upgrade", "recovery",
    "bull", "buyback", "award", "license",
}
NEGATIVE_OSINT_TERMS = {
    "arrest", "detain", "detained", "fraud", "probe", "investigation", "penalty",
    "default", "loss", "selloff", "decline", "downgrade", "suspension", "crackdown",
    "fine", "liquidity", "npl", "corruption", "panic", "halt",
}
POSITIVE_EVENT_TERMS = {
    "release", "released", "resume", "approval", "approved", "reinstated",
    "relief", "settlement", "alliance", "support", "easing", "stability",
}
NEGATIVE_EVENT_TERMS = {
    "arrest", "detained", "detain", "crackdown", "investigation", "probe",
    "violence", "unrest", "protest", "ban", "sanction", "corruption",
    "fraud", "collapse", "pressure", "selloff",
}

_MLX_MODEL = None
_MLX_PROCESSOR = None
_MLX_MODEL_ID: str | None = None
_MLX_LOCK = threading.Lock()

# ── System prompt: defines the analyst's identity and framework ──────────────

SYSTEM_PROMPT = """You are a senior NEPSE equity research analyst. You combine quantitative signals with qualitative intelligence to make sharp, defensible trade decisions.

NEPSE MARKET STRUCTURE:
- ~370 stocks, T+2 settlement, retail-heavy price action, and strong policy sensitivity
- 80%+ retail participation — herding, panic, and FOMO are tradeable patterns
- Circuit breakers: ±10% daily limits. Stocks hitting circuit = momentum exhaustion signal
- NRB (central bank) directives move banking sector hard. NEA policy moves hydro.
- Dividend/bonus announcements cause 2-5 day overreactions, then revert
- Political instability → immediate selling pressure, but usually recovers in 3-5 sessions

YOUR ANALYTICAL FRAMEWORK:
For each stock, you must assess:
1. FUNDAMENTALS — Is the business actually making money? Revenue trend, margins, EPS
2. VALUATION — Is the price justified? P/E, P/BV relative to sector
3. CATALYST — What could move this stock in the next 1-4 weeks? News, dividends, policy
4. RISK — What could go wrong? Political, regulatory, sector-specific, liquidity

YOUR RULES:
- Never approve a stock just because the algorithm flagged it. The algo sees price patterns; you see context.
- If financials show declining revenue or shrinking equity, that's a red flag regardless of price action.
- If OSINT shows political risk or regulatory headwind for a sector, downgrade the whole sector.
- If a stock is trading above 3x book value with no earnings growth, that's speculation not investment.
- Cross-reference: do the numbers match the narrative? If news is bullish but profits are declining, trust the numbers.
- Be specific. "Risky" is not analysis. "NPL rose from 1.2% to 2.8% in one quarter while the bank expanded lending 40%" is analysis.
- You must not hallucinate. If the context does not contain evidence for a claim, say it is absent and lean HOLD/REJECT rather than inventing support.
- Interpret the data fields explicitly:
  * signal_score = overall ranking strength from the quant stack
  * signal_confidence = reliability of the setup
  * signal_strength = magnitude of the setup
  * red_flags = deterministic accounting/risk warnings from filings
  * story_count/social_count/related_count = NepalOSINT evidence depth
  * forex/metals/commodities = macro pressure context for importers, banks, insurers, and consumer sectors
- Treat NepalOSINT as event evidence, filings as accounting truth, and the quant stack as timing/context. Final decisions must reconcile all three.
- Sector lenses:
  * banks: deposits, credit growth, NPL, NRB policy, liquidity pressure
  * insurers/reinsurers: float yield, claims discipline, treaty growth, solvency perception
  * hydros: NEA policy, hydrology, power-import dynamics, project execution
  * finance/microfinance: funding cost, asset quality, regulation, retail credit stress
  * manufacturing/importers: USD/NPR and commodity cost pressure
- Macro interpretation:
  * rising USD/NPR = imported inflation and margin pressure for import-dependent names
  * rising gold = risk aversion / liquidity preference
  * fast food/vegetable commodity spikes = CPI pressure and consumer margin stress
  * if macro context is irrelevant to a stock, say it is low-impact instead of forcing a story
- Final decision standard:
  * APPROVE only when the setup has both timing and evidence support
  * HOLD when the setup is interesting but incomplete or already owned
  * REJECT when evidence conflicts with the signal or downside dominates

RESPONSE STYLE: Direct, evidence-based, no hedging language. State your view and back it with data."""


def _agent_backend() -> str:
    env_value = str(os.environ.get("NEPSE_AGENT_BACKEND", "") or "").strip().lower()
    if env_value:
        return env_value
    cfg = load_active_agent_config()
    return str(cfg.get("backend") or DEFAULT_AGENT_BACKEND).strip().lower()


def _agent_model_id() -> str:
    env_value = str(os.environ.get("NEPSE_AGENT_MODEL", "") or "").strip()
    if env_value:
        return env_value
    cfg = load_active_agent_config()
    return str(cfg.get("model") or DEFAULT_GEMMA4_MLX_MODEL).strip()


def _agent_provider_label() -> str:
    env_value = str(os.environ.get("NEPSE_AGENT_PROVIDER_LABEL", "") or "").strip()
    if env_value:
        return env_value
    cfg = load_active_agent_config()
    return str(cfg.get("provider_label") or _agent_backend()).strip()


def _agent_source_label() -> str:
    env_value = str(os.environ.get("NEPSE_AGENT_SOURCE_LABEL", "") or "").strip()
    if env_value:
        return env_value
    cfg = load_active_agent_config()
    return str(cfg.get("source_label") or _agent_backend()).strip()


def _agent_trust_remote_code() -> bool:
    raw = str(os.environ.get("NEPSE_AGENT_TRUST_REMOTE_CODE", "0") or "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        cfg = load_active_agent_config()
        return bool(cfg.get("trust_remote_code"))
    return False


def _agent_fallback_backend() -> str:
    env_value = str(os.environ.get("NEPSE_AGENT_FALLBACK_BACKEND", "") or "").strip().lower()
    if env_value:
        return env_value
    cfg = load_active_agent_config()
    return str(cfg.get("fallback_backend") or "claude").strip().lower()


def reload_agent_runtime() -> dict:
    global _MLX_MODEL, _MLX_PROCESSOR, _MLX_MODEL_ID
    with _MLX_LOCK:
        _MLX_MODEL = None
        _MLX_PROCESSOR = None
        _MLX_MODEL_ID = None
    return load_active_agent_config()


def _call_claude(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = DEFAULT_AGENT_MAX_TOKENS) -> str:
    """Call claude CLI with Sonnet model."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p",
        "--model", "sonnet",
        "--system-prompt", system,
        "--output-format", "text",
        "--no-session-persistence",
        "--disallowed-tools", "WebSearch", "WebFetch",
    ]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            env=env, timeout=180, cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            return f"ERROR: claude CLI failed: {result.stderr[:500]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: claude CLI timed out (180s)"
    except FileNotFoundError:
        return "ERROR: claude CLI not found"


def _load_gemma4_mlx():
    global _MLX_MODEL, _MLX_PROCESSOR, _MLX_MODEL_ID

    model_id = _agent_model_id()
    with _MLX_LOCK:
        if _MLX_MODEL is not None and _MLX_PROCESSOR is not None and _MLX_MODEL_ID == model_id:
            return _MLX_MODEL, _MLX_PROCESSOR

        from mlx_vlm import load

        model, processor = load(
            model_id,
            trust_remote_code=_agent_trust_remote_code(),
        )
        _MLX_MODEL = model
        _MLX_PROCESSOR = processor
        _MLX_MODEL_ID = model_id
        return model, processor


def _call_gemma4_mlx(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = DEFAULT_AGENT_MAX_TOKENS) -> str:
    """Run the primary local Gemma 4 agent on MLX."""
    try:
        from mlx_vlm import generate
    except Exception as exc:
        return f"ERROR: mlx-vlm runtime unavailable: {exc}"

    try:
        model, processor = _load_gemma4_mlx()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = f"{system}\n\n{prompt}".strip()
        apply_template = getattr(processor, "apply_chat_template", None)
        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                chat_prompt = apply_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                chat_prompt = apply_template(messages, tokenize=False)
        elif callable(tokenizer_apply_template):
            try:
                chat_prompt = tokenizer_apply_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                chat_prompt = tokenizer_apply_template(messages, tokenize=False)
        with _MLX_LOCK:
            output = generate(
                model,
                processor,
                prompt=chat_prompt,
                verbose=False,
                max_tokens=int(max_tokens or DEFAULT_AGENT_MAX_TOKENS),
                temperature=float(os.environ.get("NEPSE_AGENT_TEMPERATURE", DEFAULT_AGENT_TEMPERATURE)),
                top_p=float(os.environ.get("NEPSE_AGENT_TOP_P", DEFAULT_AGENT_TOP_P)),
            )
        text = getattr(output, "text", output)
        return str(text or "").strip()
    except Exception as exc:
        return f"ERROR: Gemma 4 MLX inference failed: {exc}"


def _call_primary_agent(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = DEFAULT_AGENT_MAX_TOKENS) -> str:
    backend = _agent_backend()
    if backend in {"gemma4_mlx", "gemma4", "mlx", "mlx_gemma4"}:
        response = _call_gemma4_mlx(prompt, system=system, max_tokens=max_tokens)
        if not str(response).startswith("ERROR:"):
            return response
        fallback = _agent_fallback_backend()
        if fallback == "claude":
            fallback_response = _call_claude(prompt, system=system, max_tokens=max_tokens)
            if not str(fallback_response).startswith("ERROR:"):
                return fallback_response
        return response
    return _call_claude(prompt, system=system, max_tokens=max_tokens)


def load_agent_analysis() -> dict:
    """Load latest agent analysis from runtime storage."""
    if not ANALYSIS_FILE.exists():
        return {}
    try:
        return json.loads(ANALYSIS_FILE.read_text())
    except Exception:
        return {}


def _read_chat_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    return list(payload or []) if isinstance(payload, list) else []


def _normalize_chat_items(items: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for raw in list(items or []):
        if not isinstance(raw, dict):
            continue
        if "q" in raw and "a" in raw:
            ts = float(raw.get("ts") or time.time())
            source = str(raw.get("source") or "local_claude")
            provider = str(raw.get("provider") or "claude")
            question = str(raw.get("q") or "").strip()
            answer = str(raw.get("a") or "").strip()
            if question:
                normalized.append(
                    {
                        "role": "YOU",
                        "message": question,
                        "source": source,
                        "provider": provider,
                        "ts": ts,
                    }
                )
            if answer:
                normalized.append(
                    {
                        "role": "AGENT",
                        "message": answer,
                        "source": source,
                        "provider": provider,
                        "ts": ts,
                    }
                )
            continue

        role = str(raw.get("role") or "").strip().upper()
        message = str(raw.get("message") or "").strip()
        if not role or not message:
            continue
        normalized.append(
            {
                "role": role,
                "message": message,
                "source": str(raw.get("source") or "unknown"),
                "provider": str(raw.get("provider") or "unknown"),
                "ts": float(raw.get("ts") or time.time()),
            }
        )
    return normalized


def _load_combined_chat_history() -> list[dict]:
    archived = _normalize_chat_items(_read_chat_items(AGENT_ARCHIVE_FILE))
    active = _normalize_chat_items(_read_chat_items(AGENT_HISTORY_FILE))
    return archived + active


def _ensure_chat_storage_shape() -> None:
    active_raw = _read_chat_items(AGENT_HISTORY_FILE)
    archived_raw = _read_chat_items(AGENT_ARCHIVE_FILE)
    combined = _normalize_chat_items(archived_raw) + _normalize_chat_items(active_raw)
    needs_migration = len(_normalize_chat_items(active_raw)) > MAX_AGENT_HISTORY_ITEMS
    if not needs_migration:
        for raw in archived_raw + active_raw:
            if not isinstance(raw, dict):
                needs_migration = True
                break
            if "q" in raw and "a" in raw:
                needs_migration = True
                break
            if raw.get("role") is None or raw.get("message") is None:
                needs_migration = True
                break
    if needs_migration:
        _persist_chat_history(combined)


def _persist_chat_history(items: list[dict] | None) -> tuple[list[dict], list[dict]]:
    normalized = _normalize_chat_items(items)
    archived = normalized[:-MAX_AGENT_HISTORY_ITEMS] if len(normalized) > MAX_AGENT_HISTORY_ITEMS else []
    active = normalized[-MAX_AGENT_HISTORY_ITEMS:]
    if MAX_AGENT_ARCHIVE_ITEMS > 0:
        archived = archived[-MAX_AGENT_ARCHIVE_ITEMS:]
    AGENT_ARCHIVE_FILE.write_text(json.dumps(archived, indent=2, default=str))
    AGENT_HISTORY_FILE.write_text(json.dumps(active, indent=2, default=str))
    return active, archived


def load_agent_history(limit: Optional[int] = None, *, include_archive: bool = False) -> list[dict]:
    """Load recent agent chat history from runtime storage."""
    _ensure_chat_storage_shape()
    if include_archive:
        items = _load_combined_chat_history()
    else:
        items = _normalize_chat_items(_read_chat_items(AGENT_HISTORY_FILE))
    if limit is not None and limit >= 0:
        return items[-int(limit):]
    return items


def load_agent_archive_history(limit: Optional[int] = None) -> list[dict]:
    """Load archived agent chat history hidden from the default TUI view."""
    _ensure_chat_storage_shape()
    items = _normalize_chat_items(_read_chat_items(AGENT_ARCHIVE_FILE))
    if limit is not None and limit >= 0:
        return items[-int(limit):]
    return items


def publish_external_agent_analysis(
    analysis: dict,
    *,
    source: str = "mcp_external",
    provider: str = "external",
) -> dict:
    """Publish external agent analysis into the same runtime file the TUI uses."""
    payload = dict(analysis or {})
    now_utc = datetime.now(timezone.utc)
    payload.setdefault("timestamp", time.time())
    payload.setdefault("context_date", now_utc.strftime("%Y-%m-%d"))
    meta = dict(payload.get("agent_runtime_meta") or {})
    meta.update(
        {
            "source": source,
            "provider": provider,
            "updated_at": now_utc.replace(microsecond=0).isoformat(),
        }
    )
    payload["source"] = source
    payload["provider"] = provider
    payload["agent_runtime_meta"] = meta
    ANALYSIS_FILE.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def append_external_agent_chat_message(
    role: str,
    message: str,
    *,
    source: str = "mcp_external",
    provider: str = "external",
) -> list[dict]:
    """Append a single external agent chat message for the TUI chat pane."""
    history = _load_combined_chat_history()
    history.append(
        {
            "role": str(role or "").upper(),
            "message": str(message or ""),
            "source": source,
            "provider": provider,
            "ts": time.time(),
        }
    )
    active, _ = _persist_chat_history(history)
    return active


# ── Pre-computed metrics (Python, zero Claude cost) ──────────────────────────

def _compute_stock_metrics(symbol: str, current_price: float = 0) -> dict:
    """Compute analytical metrics from cached quarterly data + current price.

    Returns a dict of pre-computed insights so Sonnet doesn't waste tokens
    doing basic arithmetic.
    """
    metrics = {"symbol": symbol}

    try:
        from backend.quant_pro.data_scrapers.quarterly_reports import get_cached_financials
        data = get_cached_financials(symbol)
        if not data or not data.get("reports"):
            return metrics

        reports = [r for r in data["reports"] if r.get("financials") and "error" not in r["financials"]]
        if not reports:
            return metrics

        latest = reports[0]["financials"]
        sector = latest.get("sector", "unknown")
        metrics["sector"] = sector

        inc = latest.get("income_statement", {})
        bs = latest.get("balance_sheet", {})
        ps = latest.get("per_share", {})
        ratios = latest.get("ratios", {})

        # Basic financials
        revenue = inc.get("total_revenue", 0)
        net_profit = inc.get("net_profit", 0)
        total_assets = bs.get("total_assets", 0)
        equity = bs.get("shareholders_equity", 0)
        total_liabilities = bs.get("total_liabilities", 0)
        share_capital = bs.get("share_capital", 0)
        eps = ps.get("eps", 0)
        book_value = ps.get("book_value", 0)

        metrics["revenue"] = revenue
        metrics["net_profit"] = net_profit
        metrics["total_assets"] = total_assets
        metrics["equity"] = equity

        # Profit margin
        if revenue > 0:
            metrics["profit_margin_pct"] = round(net_profit / revenue * 100, 1)

        # Debt-to-equity
        if equity > 0:
            metrics["debt_to_equity"] = round(total_liabilities / equity, 2)

        # Return on equity (annualized from quarterly)
        quarter = latest.get("quarter", "")
        q_num = int(quarter.replace("Q", "")) if quarter.startswith("Q") else 1
        if equity > 0 and net_profit > 0:
            annualized_profit = net_profit * (4 / q_num) if q_num > 0 else net_profit
            metrics["roe_pct"] = round(annualized_profit / equity * 100, 1)

        # P/E ratio
        if eps > 0 and current_price > 0:
            annualized_eps = eps * (4 / q_num) if q_num > 0 else eps
            metrics["pe_ratio"] = round(current_price / annualized_eps, 1)
            metrics["eps_annualized"] = round(annualized_eps, 2)
        elif eps > 0:
            metrics["eps"] = eps

        # P/BV ratio
        if book_value > 0 and current_price > 0:
            metrics["pbv_ratio"] = round(current_price / book_value, 2)
            metrics["book_value"] = round(book_value, 1)

        # Banking-specific
        if sector == "banking":
            if ratios.get("npl_pct"):
                metrics["npl_pct"] = ratios["npl_pct"]
            if ratios.get("capital_adequacy_pct"):
                metrics["car_pct"] = ratios["capital_adequacy_pct"]
            deposits = bs.get("total_deposits", 0)
            loans = bs.get("total_loans", 0)
            if deposits > 0 and loans > 0:
                metrics["cd_ratio"] = round(loans / deposits * 100, 1)

        # QoQ trends (if we have 2+ quarters)
        if len(reports) >= 2:
            prev = reports[1]["financials"]
            prev_inc = prev.get("income_statement", {})
            prev_rev = prev_inc.get("total_revenue", 0)
            prev_np = prev_inc.get("net_profit", 0)

            if prev_rev > 0 and revenue > 0:
                metrics["revenue_growth_qoq_pct"] = round((revenue - prev_rev) / prev_rev * 100, 1)
            if prev_np > 0 and net_profit > 0:
                metrics["profit_growth_qoq_pct"] = round((net_profit - prev_np) / prev_np * 100, 1)

            # Banking: NPL trend
            prev_ratios = prev.get("ratios", {})
            if ratios.get("npl_pct") and prev_ratios.get("npl_pct"):
                metrics["npl_trend"] = "rising" if ratios["npl_pct"] > prev_ratios["npl_pct"] else "falling"

        # Red flags (automatic detection)
        flags = []
        if equity > 0 and total_liabilities / equity > 10:
            flags.append("extreme leverage (D/E > 10x)")
        if revenue > 0 and net_profit < 0:
            flags.append("loss-making despite revenue")
        if metrics.get("pbv_ratio", 0) > 5:
            flags.append(f"trading at {metrics['pbv_ratio']}x book — speculative premium")
        if metrics.get("npl_pct", 0) > 5:
            flags.append(f"high NPL at {metrics['npl_pct']}%")
        if metrics.get("profit_growth_qoq_pct") and metrics["profit_growth_qoq_pct"] < -30:
            flags.append(f"profit collapsed {metrics['profit_growth_qoq_pct']}% QoQ")
        if flags:
            metrics["red_flags"] = flags

    except Exception:
        pass

    return metrics


def _format_metrics(m: dict) -> str:
    """Format computed metrics into a compact text block for the prompt."""
    if len(m) <= 1:  # only symbol
        return ""

    parts = [f"  {m['symbol']} ({m.get('sector', '?')}):"]

    # Financials
    if m.get("revenue"):
        parts.append(f"Rev={m['revenue']:,.0f}")
    if m.get("net_profit"):
        parts.append(f"NP={m['net_profit']:,.0f}")
    if m.get("profit_margin_pct") is not None:
        parts.append(f"Margin={m['profit_margin_pct']}%")

    # Valuation
    if m.get("pe_ratio"):
        parts.append(f"P/E={m['pe_ratio']}")
    if m.get("pbv_ratio"):
        parts.append(f"P/BV={m['pbv_ratio']}x")
    if m.get("eps_annualized"):
        parts.append(f"EPS(ann)={m['eps_annualized']}")
    if m.get("book_value"):
        parts.append(f"BV={m['book_value']}")

    # Returns
    if m.get("roe_pct"):
        parts.append(f"ROE={m['roe_pct']}%")
    if m.get("debt_to_equity") is not None:
        parts.append(f"D/E={m['debt_to_equity']}")

    # Banking
    if m.get("npl_pct"):
        parts.append(f"NPL={m['npl_pct']}%")
    if m.get("car_pct"):
        parts.append(f"CAR={m['car_pct']}%")
    if m.get("cd_ratio"):
        parts.append(f"CD={m['cd_ratio']}%")

    # Trends
    if m.get("revenue_growth_qoq_pct") is not None:
        parts.append(f"RevGrowthQoQ={m['revenue_growth_qoq_pct']}%")
    if m.get("profit_growth_qoq_pct") is not None:
        parts.append(f"ProfitGrowthQoQ={m['profit_growth_qoq_pct']}%")

    # Red flags
    if m.get("red_flags"):
        parts.append(f"⚠ FLAGS: {'; '.join(m['red_flags'])}")

    return "  ".join(parts)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _refresh_intraday_market_snapshot() -> dict:
    """Force-refresh the intraday snapshot and mirror it into stock_prices for agent reads."""
    info: dict = {}
    try:
        from backend.quant_pro.database import get_db_path
        from backend.quant_pro.realtime_market import get_market_data_provider
        from backend.trading.live_trader import now_nst

        snapshot = get_market_data_provider().fetch_snapshot(force=True)
        session_date = now_nst().strftime("%Y-%m-%d")
        adv = dec = unch = 0
        rows: list[tuple[str, str, float, float, float, float, int]] = []
        for sym, quote in dict(snapshot.quotes or {}).items():
            ltp = _safe_float(quote.get("last_traded_price") or quote.get("close_price"))
            if ltp <= 0:
                continue
            prev_close = _safe_float(quote.get("previous_close"))
            pct = quote.get("percentage_change")
            try:
                pct = float(pct) if pct is not None else None
            except (TypeError, ValueError):
                pct = None
            if pct is None and prev_close > 0:
                pct = ((ltp - prev_close) / prev_close) * 100.0
            if pct is not None:
                if pct > 0:
                    adv += 1
                elif pct < 0:
                    dec += 1
                else:
                    unch += 1
            rows.append(
                (
                    str(sym).upper(),
                    session_date,
                    ltp,
                    ltp,
                    ltp,
                    ltp,
                    int(_safe_float(quote.get("total_trade_quantity"), 0.0)),
                )
            )

        if rows:
            conn = sqlite3.connect(str(get_db_path()))
            conn.executemany(
                "INSERT OR REPLACE INTO stock_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()

        info = {
            "session_date": session_date,
            "fetched_at_utc": str(snapshot.fetched_at_utc or ""),
            "source": str(snapshot.endpoint or snapshot.source or ""),
            "market_status": str(snapshot.market_status or ""),
            "advancers": adv,
            "decliners": dec,
            "unchanged": unch,
            "quote_count": len(rows),
        }
    except Exception as exc:
        info = {"error": str(exc)}
    return info


def _current_nst_session_date() -> str:
    return _nepal_market_clock()["session_date"]


def _nepal_market_clock() -> dict:
    nst_now = current_nepal_datetime()
    market_phase = market_session_phase(nst_now)
    market_open = market_phase == "OPEN"
    return {
        "session_date": nst_now.strftime("%Y-%m-%d"),
        "current_time": nst_now.strftime("%Y-%m-%d %H:%M"),
        "weekday": nst_now.strftime("%A"),
        "time_only": nst_now.strftime("%H:%M"),
        "market_open": market_open,
        "market_phase": market_phase,
        "timezone": "NPT (UTC+05:45)",
    }


def _pick_context_value(primary: dict | None, key: str, fallback):
    if isinstance(primary, dict) and key in primary and primary.get(key) is not None:
        return primary.get(key)
    return fallback


def _analysis_cache_is_fresh(analysis: dict | None) -> bool:
    payload = dict(analysis or {})
    if not payload or not list(payload.get("stocks") or []):
        return False
    age = time.time() - float(payload.get("timestamp") or 0.0)
    if age > ANALYSIS_CACHE_MAX_AGE_SECS:
        return False
    return str(payload.get("context_date") or "") == _current_nst_session_date()


def _clip_text(value: object, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def _active_account_context() -> dict:
    account_id = str(os.environ.get("NEPSE_ACTIVE_ACCOUNT_ID", "") or "").strip() or "account_1"
    account_name = str(os.environ.get("NEPSE_ACTIVE_ACCOUNT_NAME", "") or "").strip() or account_id
    account_dir_raw = str(os.environ.get("NEPSE_ACTIVE_ACCOUNT_DIR", "") or "").strip()
    account_dir = Path(account_dir_raw) if account_dir_raw else None
    portfolio_path = None
    if account_dir is not None:
        portfolio_path = account_dir / "paper_portfolio.csv"
    active_portfolio_raw = str(os.environ.get("NEPSE_ACTIVE_PORTFOLIO_FILE", "") or "").strip()
    if active_portfolio_raw:
        portfolio_path = Path(active_portfolio_raw)
    return {
        "id": account_id,
        "name": account_name,
        "dir": account_dir,
        "portfolio_path": portfolio_path,
    }


def _load_active_portfolio() -> tuple[list[dict], dict]:
    meta = _active_account_context()
    portfolio_path = meta.get("portfolio_path")
    try:
        if isinstance(portfolio_path, Path) and portfolio_path.exists():
            port = pd.read_csv(portfolio_path)
        else:
            from apps.classic.dashboard import load_port

            port = load_port()
        if port.empty:
            return [], meta
        rows = [
            {
                "symbol": str(r["Symbol"]).upper(),
                "qty": int(r["Quantity"]),
                "entry": float(r["Buy_Price"]),
            }
            for _, r in port.iterrows()
            if str(r.get("Symbol") or "").strip()
        ]
        return rows, meta
    except Exception:
        return [], meta


def _osint_keyword_bias(*texts: object) -> float:
    score = 0.0
    for raw in texts:
        text = str(raw or "").lower()
        if not text:
            continue
        pos_hits = sum(1 for token in POSITIVE_OSINT_TERMS if token in text)
        neg_hits = sum(1 for token in NEGATIVE_OSINT_TERMS if token in text)
        score += min(0.03 * pos_hits, 0.12)
        score -= min(0.04 * neg_hits, 0.16)
    return max(-0.18, min(0.18, score))


def _summarize_symbol_intelligence(symbol: str, intel: dict | None) -> str:
    payload = dict(intel or {})
    stories = list(payload.get("story_items") or [])
    social = list(payload.get("social_items") or [])
    related = list(payload.get("related_items") or [])
    if not stories and not social:
        return f"{symbol}: no direct NepalOSINT story or social hit in the configured lookback."

    parts: list[str] = [
        f"{symbol}: {int(payload.get('story_count') or 0)} story hits, "
        f"{int(payload.get('social_count') or 0)} social hits, "
        f"{int(payload.get('related_count') or 0)} related stories.",
    ]
    if stories:
        lead = dict(stories[0] or {})
        parts.append(
            f"Lead story: {_clip_text(lead.get('title'), 110)} "
            f"[{lead.get('source_name') or '?'}] {(lead.get('published_at') or '')[:16]}"
        )
    elif payload.get("semantic"):
        semantic_results = list(dict(payload.get("semantic") or {}).get("results") or [])
        if semantic_results:
            lead = dict(semantic_results[0] or {})
            parts.append(
                f"Lead semantic match: {_clip_text(lead.get('title'), 110)} "
                f"[{lead.get('source_name') or '?'}] {(lead.get('published_at') or '')[:16]}"
            )
    if social:
        top_social = dict(social[0] or {})
        parts.append(
            f"Social: {_clip_text(top_social.get('text'), 110)} "
            f"[@{top_social.get('author_username') or '?'}]"
        )
    if related:
        rel = dict(related[0] or {})
        parts.append(
            f"Related: {_clip_text(rel.get('title'), 96)} "
            f"[{rel.get('source_name') or '?'}]"
        )
    return "\n    ".join(parts)


def _fetch_energy_quote(symbol: str, label: str) -> Optional[dict]:
    try:
        import requests

        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "5d", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        response.raise_for_status()
        payload = dict(response.json() or {})
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
        meta = dict(result.get("meta") or {})
        price = float(meta.get("regularMarketPrice") or 0.0)
        prev = meta.get("previousClose")
        if prev is None:
            prev = meta.get("chartPreviousClose")
        prev = float(prev or 0.0)
        change_pct = ((price - prev) / prev * 100) if prev > 0 else None
        return {
            "name": label,
            "value": price,
            "unit": str(meta.get("currency") or "USD"),
            "change_pct": change_pct,
            "source": "Yahoo Finance",
        }
    except Exception:
        return None


def _format_macro_context(
    macro_market: dict | None,
    *,
    max_fx: int = 6,
    max_commodities: int = 5,
    max_energy: int = 2,
) -> str:
    payload = dict(macro_market or {})
    forex_rows = list(payload.get("forex") or [])
    commodity_rows = list(payload.get("commodities") or [])
    energy_rows = list(payload.get("energy") or [])
    macro_lines: list[str] = []
    if forex_rows:
        macro_lines.append("FOREX / RATES CONTEXT:")
        for row in forex_rows[:max_fx]:
            macro_lines.append(
                f"  {row.get('code','?'):4s} buy={float(row.get('buy') or 0.0):,.2f} "
                f"sell={float(row.get('sell') or 0.0):,.2f} unit={row.get('unit') or 1}"
            )
    if energy_rows:
        macro_lines.append("ENERGY / CRUDE CONTEXT:")
        for row in energy_rows[:max_energy]:
            pct = row.get("change_pct")
            pct_text = f" chg={float(pct):+.2f}%" if pct is not None else ""
            macro_lines.append(
                f"  {str(row.get('name') or '')[:24]:24s} spot={float(row.get('value') or 0.0):,.2f} "
                f"{row.get('unit') or ''}{pct_text}"
            )
    if commodity_rows:
        macro_lines.append("COMMODITY CONTEXT (top movers):")
        for row in commodity_rows[:max_commodities]:
            macro_lines.append(
                f"  {str(row.get('name') or '')[:24]:24s} avg={float(row.get('avg') or 0.0):,.1f} "
                f"chg={float(row.get('change_pct') or 0.0):+.1f}% {row.get('unit') or ''}"
            )
    return "\n".join(macro_lines) + ("\n" if macro_lines else "")


def _fetch_macro_market_context() -> dict:
    payload: dict = {"forex": [], "commodities": [], "energy": []}

    try:
        import requests

        response = requests.get("https://www.nrb.org.np/api/forex/v1/rates", timeout=8)
        response.raise_for_status()
        data = dict(response.json() or {})
        rows: list[dict] = []
        wanted = {"USD", "EUR", "GBP", "INR", "CNY", "JPY"}
        for bucket in list(data.get("data", {}).get("payload", []) or []):
            for rate in list(bucket.get("rates") or []):
                code = str(rate.get("currency", {}).get("iso3") or "").upper()
                if code not in wanted:
                    continue
                rows.append(
                    {
                        "code": code,
                        "name": str(rate.get("currency", {}).get("name") or ""),
                        "buy": float(rate.get("buy") or 0.0),
                        "sell": float(rate.get("sell") or 0.0),
                        "unit": int(rate.get("unit") or 1),
                    }
                )
        payload["forex"] = rows
    except Exception as exc:
        payload["forex_error"] = str(exc)

    try:
        from backend.market.kalimati_market import get_kalimati_display_rows

        rows = list(get_kalimati_display_rows() or [])
        movers = sorted(
            rows,
            key=lambda row: abs(float(row.get("change_pct") or 0.0)),
            reverse=True,
        )[:5]
        payload["commodities"] = [
            {
                "name": str(row.get("name_english") or ""),
                "avg": float(row.get("avg") or 0.0),
                "change_pct": float(row.get("change_pct") or 0.0),
                "unit": str(row.get("unit") or ""),
            }
            for row in movers
            if str(row.get("name_english") or "").strip()
        ]
    except Exception as exc:
        payload["commodities_error"] = str(exc)

    try:
        energy_rows = []
        for symbol, label in (("CL=F", "WTI Crude"), ("BZ=F", "Brent Crude")):
            row = _fetch_energy_quote(symbol, label)
            if row:
                energy_rows.append(row)
        payload["energy"] = energy_rows
    except Exception as exc:
        payload["energy_error"] = str(exc)

    return payload


def _fallback_stock_decision(
    symbol: str,
    sig: dict,
    row: dict,
    *,
    metrics: dict | None,
    intel: dict | None,
    is_held: bool,
    preview_mode: bool,
) -> dict:
    merged = dict(row or {})
    metrics = dict(metrics or {})
    intel = dict(intel or {})
    if preview_mode:
        merged.setdefault("verdict", "REVIEW")
        merged.setdefault("conviction", 0.0)
        merged.setdefault("what_matters", str(sig.get("reasoning") or ""))
        merged.setdefault("reasoning", str(sig.get("reasoning") or ""))
        return merged

    existing_verdict = str(merged.get("verdict") or "").upper()
    existing_conviction = _clamp_conviction(merged.get("conviction"))
    if not is_held and existing_verdict == "HOLD":
        existing_verdict = "REJECT"
        merged["verdict"] = "REJECT"
    if (
        existing_verdict in {"APPROVE", "HOLD", "REJECT"}
        and existing_conviction >= 0.25
        and str(merged.get("reasoning") or merged.get("what_matters") or "").strip()
    ):
        merged["conviction"] = existing_conviction
        merged.setdefault("sector", str(metrics.get("sector") or ""))
        if not str(merged.get("what_matters") or "").strip():
            merged["what_matters"] = _clip_text(sig.get("reasoning"), 120)
        if not str(merged.get("bull_case") or "").strip():
            merged["bull_case"] = f"Signal score {float(sig.get('score') or 0.0):.2f} still keeps {symbol} on the front foot."
        if not str(merged.get("bear_case") or "").strip():
            merged["bear_case"] = "; ".join(list(metrics.get("red_flags") or [])[:2]) or "Execution still needs confirmation from price action."
        return merged

    score = float(sig.get("score") or 0.0)
    confidence = float(sig.get("confidence") or 0.0)
    strength = float(sig.get("strength") or 0.0)
    base_conviction = 0.22 + min(score, 1.0) * 0.28 + min(confidence, 1.0) * 0.24 + min(strength / 1.5, 1.0) * 0.16
    evidence_parts: list[str] = []

    revenue_growth = float(metrics.get("revenue_growth_qoq_pct") or 0.0)
    profit_growth = float(metrics.get("profit_growth_qoq_pct") or 0.0)
    margin = float(metrics.get("profit_margin_pct") or 0.0)
    pe_ratio = float(metrics.get("pe_ratio") or 0.0)
    pbv_ratio = float(metrics.get("pbv_ratio") or 0.0)
    roe = float(metrics.get("roe_pct") or 0.0)
    red_flags = list(metrics.get("red_flags") or [])
    npl_pct = float(metrics.get("npl_pct") or 0.0)

    if revenue_growth > 5:
        base_conviction += 0.05
        evidence_parts.append(f"revenue growth {revenue_growth:.1f}% QoQ")
    if profit_growth > 5:
        base_conviction += 0.07
        evidence_parts.append(f"profit growth {profit_growth:.1f}% QoQ")
    if margin > 15:
        base_conviction += 0.05
        evidence_parts.append(f"margin {margin:.1f}%")
    if 0 < pe_ratio <= 12:
        base_conviction += 0.04
        evidence_parts.append(f"P/E {pe_ratio:.1f}")
    if 0 < pbv_ratio <= 2.0:
        base_conviction += 0.03
        evidence_parts.append(f"P/BV {pbv_ratio:.2f}x")
    if roe >= 12:
        base_conviction += 0.04
        evidence_parts.append(f"ROE {roe:.1f}%")

    if margin and margin < 5:
        base_conviction -= 0.05
    if pe_ratio > 25:
        base_conviction -= 0.06
    if pbv_ratio > 4:
        base_conviction -= 0.07
    if npl_pct > 5:
        base_conviction -= 0.08
    if red_flags:
        base_conviction -= min(0.09 * len(red_flags), 0.27)

    story_items = list(intel.get("story_items") or [])
    social_items = list(intel.get("social_items") or [])
    top_story = dict(story_items[0] or {}) if story_items else {}
    top_social = dict(social_items[0] or {}) if social_items else {}
    osint_bias = _osint_keyword_bias(
        top_story.get("title"),
        top_story.get("summary"),
        top_social.get("text"),
    )
    base_conviction += osint_bias
    if story_items or social_items:
        base_conviction += min(0.01 * (len(story_items) + len(social_items)), 0.05)

    conviction = _clamp_conviction(max(0.15, min(0.95, base_conviction)))
    severe_negative = bool(red_flags) or osint_bias <= -0.08 or npl_pct > 5 or pe_ratio > 35 or pbv_ratio > 5

    if is_held:
        verdict = "REJECT" if severe_negative and conviction < 0.72 else "HOLD"
    else:
        if severe_negative or score < 0.35:
            verdict = "REJECT"
        elif conviction >= 0.70 and score >= 0.65 and osint_bias > -0.06:
            verdict = "APPROVE"
        else:
            verdict = "REJECT"

    lead_story = _clip_text(top_story.get("title"), 96) if top_story else ""
    if not evidence_parts and sig.get("reasoning"):
        evidence_parts.append(_clip_text(sig.get("reasoning"), 84))
    if lead_story:
        evidence_parts.append(f"OSINT lead: {lead_story}")

    bull_case = merged.get("bull_case") or (
        f"Quant setup is supported by {_clip_text(', '.join(evidence_parts), 110)}."
        if evidence_parts else
        f"Signal score {score:.2f} and confidence {confidence:.0%} keep {symbol} on the long radar."
    )
    bear_case = merged.get("bear_case") or (
        "; ".join(red_flags[:2]) if red_flags else
        (f"OSINT tone is negative around {_clip_text(top_story.get('title'), 80)}." if osint_bias < -0.04 and top_story else
         f"Signal strength {strength:.2f} still needs confirmation.")
    )
    what_matters = merged.get("what_matters") or (
        _clip_text(lead_story, 120) if lead_story else
        _clip_text(", ".join(evidence_parts), 120) if evidence_parts else
        _clip_text(sig.get("reasoning"), 120)
    )
    reasoning = merged.get("reasoning") or (
        f"{symbol} scores {score:.2f} with {confidence:.0%} signal confidence. "
        f"Financial read: {_clip_text(_format_metrics({'symbol': symbol, **metrics}), 180) or 'limited recent filing data'}. "
        f"NepalOSINT: {_clip_text(_summarize_symbol_intelligence(symbol, intel), 200)}."
    )

    merged.update(
        {
            "verdict": verdict,
            "conviction": conviction,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "what_matters": what_matters,
            "reasoning": reasoning,
            "sector": str(merged.get("sector") or metrics.get("sector") or ""),
        }
    )
    return merged


# ── Context gathering ────────────────────────────────────────────────────────

def _gather_context() -> dict:
    """Gather all context for the agent: signals, news, regime, portfolio."""
    context = {}
    fresh_market = _refresh_intraday_market_snapshot()
    if fresh_market:
        context["fresh_market"] = fresh_market
    context["macro_market"] = _fetch_macro_market_context()

    # 1. Algorithm signals + price data
    try:
        from backend.backtesting.simple_backtest import load_all_prices
        from apps.classic.dashboard import MD, _db
        from backend.trading.live_trader import generate_signals
        md = MD(top_n=10)
        conn = _db()
        prices_df = load_all_prices(conn)
        conn.close()
        sigs, regime = generate_signals(
            prices_df,
            list(LONG_TERM_CONFIG["signal_types"]),
            use_regime_filter=True,
        )
        regime_blocked = False
        if not sigs and str(regime).lower() == "bear":
            sigs, regime = generate_signals(
                prices_df,
                list(LONG_TERM_CONFIG["signal_types"]),
                use_regime_filter=False,
            )
            regime_blocked = True
        sigs = list(sigs or [])[:AGENT_SHORTLIST_LIMIT]

        context["signals"] = [
            {
                "symbol": str(s.get("symbol") or "").upper(),
                "type": str(s.get("signal_type") or s.get("type") or ""),
                "direction": "BUY",
                "strength": round(float(s.get("strength") or 0.0), 3),
                "confidence": round(float(s.get("confidence") or 0.0), 2),
                "score": round(float(s.get("score") or 0.0), 3),
                "reasoning": str(s.get("reasoning") or ""),
                "rank": idx + 1,
                "regime_blocked": regime_blocked,
            }
            for idx, s in enumerate(sigs)
        ]
        context["regime"] = regime
        context["session_date"] = str(_pick_context_value(fresh_market, "session_date", md.latest))
        context["shortlist_limit"] = AGENT_SHORTLIST_LIMIT

        # NEPSE index
        if len(md.nepse) >= 2:
            ni = md.nepse.iloc[0]["close"]
            np_ = md.nepse.iloc[1]["close"]
            context["nepse_index"] = round(ni, 1)
            context["nepse_change_pct"] = round((ni - np_) / np_ * 100, 2)

        # Breadth
        context["advancers"] = int(_pick_context_value(fresh_market, "advancers", md.adv))
        context["decliners"] = int(_pick_context_value(fresh_market, "decliners", md.dec))
        context["unchanged"] = int(_pick_context_value(fresh_market, "unchanged", md.unch))

        # Current prices for P/E, P/BV computation
        context["prices"] = md.ltps()
        context["signal_metrics"] = {
            str(s.get("symbol") or "").upper(): _compute_stock_metrics(
                str(s.get("symbol") or "").upper(),
                float((context["prices"] or {}).get(str(s.get("symbol") or "").upper()) or 0.0),
            )
            for s in context["signals"]
            if str(s.get("symbol") or "").strip()
        }

    except Exception as e:
        context["signals"] = []
        context["signal_error"] = str(e)
        context["signal_metrics"] = {}

    # 2. NepalOSINT semantic + unified + related-story search per shortlisted stock
    try:
        symbol_intel: dict[str, dict] = {}
        embeddings_context = []
        for s in context.get("signals", []):
            sym = str(s.get("symbol") or "").upper()
            if not sym or "::" in sym:
                continue
            intel = symbol_intelligence(
                sym,
                hours=720,
                top_k=6,
                min_similarity=0.45,
                base_url=OSINT_BASE,
            )
            symbol_intel[sym] = intel
            for item in list(dict(intel.get("semantic") or {}).get("results") or [])[:3]:
                title = str(item.get("title") or "")
                if title:
                    embeddings_context.append(
                        {
                            "symbol": sym,
                            "title": title[:150],
                            "source": item.get("source_name", ""),
                            "similarity": round(float(item.get("similarity") or 0.0), 3),
                            "date": (item.get("published_at") or "")[:16],
                        }
                    )
        context["symbol_intelligence"] = symbol_intel
        context["embeddings"] = embeddings_context
    except Exception as e:
        context["symbol_intelligence"] = {}
        context["embeddings"] = []
        context["embeddings_error"] = str(e)

    # 3. OSINT news feed (last 48h)
    try:
        import requests
        r = requests.get(
            f"{OSINT_BASE}/analytics/consolidated-stories",
            params={"limit": 30}, timeout=8)
        r.raise_for_status()
        stories = r.json()
        context["news"] = []
        for s in stories:
            headline = s.get("canonical_headline", "")
            summary = s.get("summary", "") or ""
            url = s.get("url", "") or ""
            text = ""
            if summary and not any(ord(c) > 127 for c in summary[:10]):
                text = summary[:200]
            elif headline and not any(ord(c) > 127 for c in headline[:10]):
                text = headline
            elif url:
                slug = url.rstrip("/").split("/")[-1]
                slug = re.sub(r'\.(html?|aspx|php)$', '', slug, flags=re.I)
                slug = slug.replace("-", " ").replace("_", " ")
                slug = re.sub(r'[\s]+\d[\d\s\.]*$', '', slug).strip()
                if slug and not slug.isdigit() and len(slug) > 5:
                    text = slug.title()
            if text:
                context["news"].append({
                    "type": s.get("story_type", ""),
                    "severity": s.get("severity", ""),
                    "source": s.get("source_name", ""),
                    "text": text[:200],
                    "time": s.get("first_reported_at", "")[:16],
                })
    except Exception as e:
        context["news"] = []
        context["news_error"] = str(e)

    # 4. Portfolio
    portfolio_rows, portfolio_meta = _load_active_portfolio()
    context["portfolio"] = portfolio_rows
    context["portfolio_account"] = portfolio_meta

    return context


def _clamp_conviction(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _derive_action_label(stock: dict, *, is_held: bool) -> str:
    verdict = str(stock.get("verdict") or "").upper()
    if verdict == "APPROVE":
        return "BUY"
    if verdict == "HOLD":
        return "HOLD" if is_held else "PASS"
    if verdict == "REJECT":
        return "SELL" if is_held else "PASS"
    return "REVIEW"


def _merge_agent_output_with_shortlist(parsed: dict, ctx: dict) -> dict:
    signal_rows = list(ctx.get("signals") or [])
    portfolio_symbols = {
        str(item.get("symbol") or "").upper()
        for item in list(ctx.get("portfolio") or [])
        if str(item.get("symbol") or "").strip()
    }
    metrics_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("signal_metrics") or {}).items()
    }
    intel_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("symbol_intelligence") or {}).items()
    }
    preview_mode = bool(parsed.get("_preview")) or (
        parsed.get("trade_today") is None
        and not str(parsed.get("market_view") or "").strip()
        and not list(parsed.get("stocks") or [])
    )
    parsed_rows = {
        str(item.get("symbol") or "").upper(): dict(item)
        for item in list(parsed.get("stocks") or [])
        if str(item.get("symbol") or "").strip()
    }
    merged_rows: list[dict] = []
    for rank, sig in enumerate(signal_rows, 1):
        symbol = str(sig.get("symbol") or "").upper()
        is_held = symbol in portfolio_symbols
        row = _fallback_stock_decision(
            symbol,
            sig,
            dict(parsed_rows.get(symbol) or {}),
            metrics=metrics_map.get(symbol),
            intel=intel_map.get(symbol),
            is_held=is_held,
            preview_mode=preview_mode,
        )
        conviction = _clamp_conviction(row.get("conviction"))
        normalized_verdict = str(row.get("verdict") or "REVIEW").upper()
        if not is_held and normalized_verdict == "HOLD":
            normalized_verdict = "REJECT"
        merged = {
            "symbol": symbol,
            "algo_signal": str(row.get("algo_signal") or sig.get("direction") or "BUY").upper(),
            "signal_type": str(sig.get("type") or row.get("signal_type") or ""),
            "signal_strength": float(sig.get("strength") or 0.0),
            "signal_confidence": float(sig.get("confidence") or 0.0),
            "signal_score": float(sig.get("score") or 0.0),
            "shortlist_rank": rank,
            "sector": str(row.get("sector") or ""),
            "verdict": normalized_verdict,
            "conviction": conviction,
            "bull_case": str(row.get("bull_case") or ""),
            "bear_case": str(row.get("bear_case") or ""),
            "what_matters": str(row.get("what_matters") or sig.get("reasoning") or ""),
            "reasoning": str(row.get("reasoning") or sig.get("reasoning") or ""),
            "last_price": float((ctx.get("prices") or {}).get(symbol) or 0.0),
            "is_held": is_held,
            "regime_blocked": bool(sig.get("regime_blocked")),
            "metrics": metrics_map.get(symbol, {}),
            "osint": intel_map.get(symbol, {}),
        }
        merged["action_label"] = _derive_action_label(merged, is_held=is_held)
        merged["auto_entry_candidate"] = bool(
            merged["action_label"] == "BUY"
            and not is_held
            and bool(parsed.get("trade_today", False))
            and merged["signal_score"] >= SUPER_SIGNAL_MIN_SCORE
            and merged["signal_strength"] >= SUPER_SIGNAL_MIN_STRENGTH
            and merged["signal_confidence"] >= SUPER_SIGNAL_MIN_CONFIDENCE
            and conviction >= SUPER_SIGNAL_MIN_CONVICTION
        )
        merged_rows.append(merged)

    enriched = dict(parsed)
    enriched["shortlist"] = signal_rows
    enriched["stocks"] = merged_rows
    enriched["super_signal_thresholds"] = {
        "score": SUPER_SIGNAL_MIN_SCORE,
        "strength": SUPER_SIGNAL_MIN_STRENGTH,
        "signal_confidence": SUPER_SIGNAL_MIN_CONFIDENCE,
        "agent_conviction": SUPER_SIGNAL_MIN_CONVICTION,
    }
    return enriched


def build_algo_shortlist_snapshot() -> dict:
    """Return the ranked algo shortlist without waiting for full agent analysis."""
    ctx = _gather_context()
    preview = {
        "_preview": True,
        "timestamp": time.time(),
        "context_date": ctx.get("session_date", ""),
        "regime": ctx.get("regime", "unknown"),
        "trade_today": None,
        "trade_today_reason": "",
        "market_view": "",
        "risks": [],
        "portfolio_note": "",
        "stocks": [],
    }
    return _merge_agent_output_with_shortlist(preview, ctx)


# ── Batch analysis ───────────────────────────────────────────────────────────

def analyze(force: bool = False) -> dict:
    """Run the agent analysis. Returns the analysis dict."""
    if not force and ANALYSIS_FILE.exists():
        cached = load_agent_analysis()
        if _analysis_cache_is_fresh(cached):
            return cached

    ctx = _gather_context()
    nepal_clock = _nepal_market_clock()
    prices = ctx.get("prices", {})
    metrics_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("signal_metrics") or {}).items()
    }
    intel_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("symbol_intelligence") or {}).items()
    }

    # ── Build structured context blocks ──

    # Signals + pre-computed metrics
    signals_text = ""
    metrics_text = ""
    if ctx.get("signals"):
        signals_text = "ALGORITHM SHORTLIST:\n"
        metrics_lines = ["PRE-COMPUTED METRICS (from latest quarterly filings):"]
        for s in ctx["signals"]:
            signals_text += (
                f"  {s['direction']:4s} {s['symbol']:10s} "
                f"strength={s['strength']:.3f} conf={s['confidence']:.2f} "
                f"type={s['type']}  reason: {s['reasoning'][:60]}\n"
            )
            m = metrics_map.get(str(s["symbol"]).upper(), {})
            formatted = _format_metrics(m)
            if formatted:
                if prices.get(s["symbol"], 0) > 0:
                    formatted += f"  CMP={prices.get(s['symbol'], 0):.1f}"
                metrics_lines.append(formatted)
        if len(metrics_lines) > 1:
            metrics_text = "\n".join(metrics_lines) + "\n"
    else:
        signals_text = "ALGORITHM SHORTLIST: No signals generated today.\n"

    news_text = ""
    if ctx.get("news"):
        news_text = "OSINT NEWS FEED (last 48h, from Nepal OSINT):\n"
        for n in ctx["news"][:20]:
            news_text += f"  [{n['severity']:6s}] [{n['type']:10s}] {n['text'][:150]}\n"

    portfolio_text = ""
    portfolio_meta = dict(ctx.get("portfolio_account") or {})
    if ctx.get("portfolio"):
        portfolio_text = (
            f"ACTIVE ACCOUNT HOLDINGS ({portfolio_meta.get('name') or portfolio_meta.get('id') or 'account'} / "
            f"{portfolio_meta.get('id') or 'account_1'}):\n"
        )
        for p in ctx["portfolio"]:
            cur = prices.get(p["symbol"], 0)
            pnl = ((cur - p["entry"]) / p["entry"] * 100) if cur and p["entry"] else 0
            portfolio_text += (
                f"  {p['symbol']:10s} qty={p['qty']} entry={p['entry']:.1f}"
                f"  CMP={cur:.1f}  P&L={pnl:+.1f}%\n"
            )
    else:
        portfolio_text = (
            f"ACTIVE ACCOUNT HOLDINGS ({portfolio_meta.get('name') or portfolio_meta.get('id') or 'account'} / "
            f"{portfolio_meta.get('id') or 'account_1'}): none currently held.\n"
        )

    embeddings_text = ""
    if ctx.get("embeddings"):
        embeddings_text = "OSINT VECTOR SEARCH (matched intelligence from 38K+ stories):\n"
        for e in ctx["embeddings"]:
            embeddings_text += (
                f"  [{e['symbol']:6s}] {e['source']:15s} {e['date']}  {e['title']}\n"
            )

    symbol_intelligence_text = ""
    if intel_map:
        lines = ["SYMBOL-SPECIFIC NEPALOSINT INTELLIGENCE:"]
        for s in ctx.get("signals", []):
            sym = str(s.get("symbol") or "").upper()
            if not sym:
                continue
            lines.append(f"  {_summarize_symbol_intelligence(sym, intel_map.get(sym))}")
        if len(lines) > 1:
            symbol_intelligence_text = "\n".join(lines) + "\n"

    macro_text = _format_macro_context(ctx.get("macro_market"))

    schedule = get_market_schedule()

    # ── The analysis prompt ──
    prompt = f"""MARKET CLOCK FACTS:
  Nepal weekday/date/time right now: {nepal_clock.get('weekday', 'unknown')}, {ctx.get('session_date', 'unknown')} {nepal_clock.get('time_only', 'unknown')} {nepal_clock.get('timezone', 'NPT')}
  NEPSE session right now: {nepal_clock.get('market_phase', 'UNKNOWN')}
  NEPSE trading week: {schedule.get('trading_week', 'unknown')}
  NEPSE special pre-open session: {schedule.get('special_preopen', 'unknown')}
  NEPSE pre-open session: {schedule.get('preopen', 'unknown')}
  NEPSE regular session: {schedule.get('regular', 'unknown')}
  If you mention the day, session status, or "today", you must use these exact facts and must not contradict them.

MARKET STATE:
  Date: {ctx.get('session_date', 'unknown')}
  Current Nepal Time: {nepal_clock.get('current_time', 'unknown')} {nepal_clock.get('timezone', 'NPT')}
  NEPSE Session: {nepal_clock.get('market_phase', 'UNKNOWN')} ({schedule.get('trading_week', 'unknown')}, regular {schedule.get('regular', 'unknown')})
  NEPSE: {ctx.get('nepse_index', 'N/A')} ({ctx.get('nepse_change_pct', 'N/A')}%)
  Regime: {ctx.get('regime', 'unknown')}
  Breadth: ▲{ctx.get('advancers', '?')} vs ▼{ctx.get('decliners', '?')}

{signals_text}
{metrics_text}
{portfolio_text}
{news_text}
{embeddings_text}
{symbol_intelligence_text}
{macro_text}

INSTRUCTIONS:

1. MARKET ASSESSMENT: One sharp paragraph. State whether this is pre-open, live session, post-close, or weekend first if relevant. What's the market actually doing today — not what the index says, but what breadth, volume, and news tell you? Is this a day to deploy capital or preserve it?

2. FOR EACH STOCK in the shortlist, analyze using this framework:
   - FUNDAMENTALS: What do the metrics tell you? Revenue trend, margins, valuation (P/E, P/BV). Are the numbers strong or is this a price-action-only play?
   - CATALYST: Any OSINT news, upcoming events, or sector developments that could move this stock?
   - RISK: What could go wrong? Sector headwinds, political risk, overvaluation, liquidity?
   - BULL vs BEAR: In 1 sentence each, the strongest argument for and against
   - VERDICT: APPROVE, REJECT, or HOLD with conviction 0-100%
   - WHAT MATTERS: One sentence — "What actually matters for this stock right now is..."
   - Review ALL shortlisted stocks in rank order. Do not skip any real stock.
   - HOLD is only valid for symbols already present in ACTIVE ACCOUNT HOLDINGS. If a stock is not currently held in the active account, the final stance must resolve to APPROVE or REJECT.

3. PORTFOLIO RISK CHECK: Are we overexposed to any sector? Should we trim anything?

Respond in this EXACT JSON format (no markdown, no code fences, just raw JSON):
{{
  "market_view": "your market assessment paragraph",
  "trade_today": true or false,
  "trade_today_reason": "concise reason",
  "risks": ["systemic risk 1", "systemic risk 2", "systemic risk 3"],
  "portfolio_note": "any concern about current holdings",
  "stocks": [
    {{
      "symbol": "SYMBOL",
      "algo_signal": "BUY or SELL",
      "sector": "sector name",
      "verdict": "APPROVE or REJECT or HOLD",
      "conviction": 0.0 to 1.0,
      "bull_case": "strongest reason to buy",
      "bear_case": "strongest reason NOT to buy",
      "what_matters": "the one thing that actually matters for this stock right now",
      "reasoning": "2-3 sentences with specific numbers and evidence"
    }}
  ]
}}

CRITICAL: Skip stocks with SECTOR:: prefix — those are index proxies, not tradeable.
Review every real stock in the top {ctx.get('shortlist_limit', AGENT_SHORTLIST_LIMIT)} shortlist. Use specific numbers from the metrics. Don't say "strong fundamentals" — say "P/E 8.2 on 42% margin with rising revenue."
Use Nepal time only. Treat all session, market-hours, and date references as NPT unless the prompt explicitly says otherwise.
REVIEW is forbidden in the final stock verdicts. Every real shortlisted stock must end as APPROVE, HOLD, or REJECT with conviction above 0.35.
"""

    raw = _call_primary_agent(prompt)

    # Parse JSON from response
    analysis = {"raw_response": raw, "timestamp": time.time(),
                "context_date": ctx.get("session_date", ""),
                "regime": ctx.get("regime", "unknown"),
                "fresh_market": dict(ctx.get("fresh_market") or {})}
    try:
        text = raw
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            analysis.update(parsed)
    except (json.JSONDecodeError, IndexError):
        analysis["parse_error"] = True

    analysis = _merge_agent_output_with_shortlist(analysis, ctx)

    return publish_external_agent_analysis(
        analysis,
        source=_agent_source_label(),
        provider=_agent_provider_label(),
    )


# ── Interactive Q&A ──────────────────────────────────────────────────────────

def _vector_search_for_question(question: str) -> str:
    """Search NepalOSINT using semantic, unified, and related-story endpoints."""
    results: list[str] = []
    intel = symbol_intelligence(
        question,
        hours=720,
        top_k=5,
        min_similarity=0.45,
        base_url=OSINT_BASE,
    )
    for item in list(intel.get("story_items") or [])[:3]:
        results.append(
            f"  [story] [{item.get('source_name', ''):15s}] "
            f"{(item.get('published_at') or '')[:16]}  {_clip_text(item.get('title'), 150)}"
        )
    for item in list(intel.get("social_items") or [])[:2]:
        results.append(
            f"  [social] [@{item.get('author_username', ''):15s}] "
            f"{(item.get('tweeted_at') or '')[:16]}  {_clip_text(item.get('text'), 150)}"
        )
    for item in list(intel.get("related_items") or [])[:2]:
        results.append(
            f"  [related] [{item.get('source_name', ''):15s}] "
            f"{(item.get('published_at') or '')[:16]}  {_clip_text(item.get('title'), 150)}"
        )
    if results:
        return "RELEVANT NEPALOSINT SEARCH RESULTS:\n" + "\n".join(results) + "\n"
    return ""


def _extract_symbol_from_question(question: str) -> Optional[str]:
    """Try to extract a stock symbol from the user's question."""
    # Look for uppercase 2-6 letter words that could be symbols
    words = question.upper().split()
    for word in words:
        clean = re.sub(r'[^A-Z]', '', word)
        if 2 <= len(clean) <= 8 and clean.isalpha():
            # Check if it's a known symbol
            try:
                from backend.quant_pro.data_scrapers.quarterly_reports import get_cached_financials
                if get_cached_financials(clean):
                    return clean
            except Exception:
                pass
            # Check if it's in the latest analysis
            if ANALYSIS_FILE.exists():
                try:
                    a = json.loads(ANALYSIS_FILE.read_text())
                    for s in a.get("stocks", []):
                        if s.get("symbol", "").upper() == clean:
                            return clean
                except Exception:
                    pass
    return None


def _question_is_time_sensitive(question: str) -> bool:
    text = str(question or "").lower()
    markers = (
        "today",
        "right now",
        "now",
        "current",
        "market status",
        "market open",
        "market closed",
        "session",
        "hours",
        "what time",
    )
    return any(marker in text for marker in markers)


def _question_is_directional_market_call(question: str) -> bool:
    text = str(question or "").lower()
    market_markers = (
        "how would nepse react",
        "how will nepse react",
        "market react",
        "upward pressure",
        "downward pressure",
        "market pressure",
        "what happens to nepse",
        "what would nepse do",
        "after the news",
        "political",
    )
    event_markers = POSITIVE_EVENT_TERMS | NEGATIVE_EVENT_TERMS | {"kp oli", "oli", "release"}
    return any(marker in text for marker in market_markers) or any(token in text for token in event_markers)


def _question_focus_query(question: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s]", " ", str(question or " "))
    text = re.sub(
        r"\b(how|would|will|could|should|what|happen|happens|after|the|news|of|to|nepse|market|react|reaction|be|on|if|do|does|did)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = " ".join(text.split())
    return text[:96].strip() or str(question or "").strip()[:96]


def _event_market_context(question: str) -> dict:
    query = _question_focus_query(question)
    unified = unified_search(query or question, limit=8, base_url=OSINT_BASE, timeout=8)
    categories = dict(unified.get("categories") or {})
    stories = list(dict(categories.get("stories") or {}).get("items") or [])
    social = list(dict(categories.get("social_signals") or {}).get("items") or [])

    lines: list[str] = []
    for item in stories[:4]:
        lines.append(
            f"  [story] [{str(item.get('source_name') or ''):15s}] "
            f"{str(item.get('published_at') or '')[:16]}  {_clip_text(item.get('title'), 140)}"
        )
    for item in social[:3]:
        lines.append(
            f"  [social] [@{str(item.get('author_username') or ''):15s}] "
            f"{str(item.get('tweeted_at') or '')[:16]}  {_clip_text(item.get('text'), 140)}"
        )

    text_blob = " ".join(
        [
            str(question or ""),
            *(str(item.get("title") or "") for item in stories),
            *(str(item.get("summary") or "") for item in stories),
            *(str(item.get("text") or "") for item in social),
            *(str(item.get("match_reason") or "") for item in stories),
        ]
    )
    bias = _osint_keyword_bias(text_blob)
    lowered = text_blob.lower()
    bias += min(0.05 * sum(1 for token in POSITIVE_EVENT_TERMS if token in lowered), 0.12)
    bias -= min(0.06 * sum(1 for token in NEGATIVE_EVENT_TERMS if token in lowered), 0.16)
    bias = max(-0.25, min(0.25, bias))
    return {
        "query": query or question,
        "stories": stories,
        "social": social,
        "bias": bias,
        "context_text": ("EVENT / POLITICAL OSINT CONTEXT:\n" + "\n".join(lines) + "\n") if lines else "",
    }


def _response_is_hedged_market_call(response: str) -> bool:
    text = " ".join(str(response or "").lower().split())
    if not text:
        return True
    if not text.startswith("base case:"):
        return True
    hedges = (
        "depends entirely",
        "could go either way",
        "could be both",
        "depends on the content",
        "depends on how",
        "both ways",
        "however, if",
        "if the release is perceived",
    )
    return any(token in text for token in hedges)


def _build_directional_market_answer(question: str, analysis: dict, nepal_clock: dict, event_ctx: dict) -> str:
    breadth_up = int((analysis.get("fresh_market") or {}).get("advancers") or analysis.get("advancers") or 0)
    breadth_down = int((analysis.get("fresh_market") or {}).get("decliners") or analysis.get("decliners") or 0)
    breadth_total = max(1, breadth_up + breadth_down)
    breadth_edge = (breadth_up - breadth_down) / breadth_total
    regime = str(analysis.get("regime") or "unknown").lower()
    regime_bias = 0.08 if regime == "bull" else -0.08 if regime == "bear" else 0.0
    event_bias = float(event_ctx.get("bias") or 0.0)
    question_text = str(question or "").lower()
    if any(token in question_text for token in POSITIVE_EVENT_TERMS):
        event_bias += 0.06
    if any(token in question_text for token in NEGATIVE_EVENT_TERMS):
        event_bias -= 0.08
    score = event_bias + (breadth_edge * 0.18) + regime_bias
    phase = str(nepal_clock.get("market_phase") or "UNKNOWN")

    if score >= 0.08:
        base_case = "upward pressure"
    elif score <= -0.08:
        base_case = "downward pressure"
    elif score > 0:
        base_case = "flat-to-upward pressure"
    else:
        base_case = "flat-to-downward pressure"

    horizon = "over the next 1-3 sessions"
    if phase == "OPEN":
        timing = "Because NEPSE is already live, I would expect the reaction to show up intraday first and then carry into the next 1-2 sessions if follow-through headlines confirm it."
    elif phase in {"PREOPEN", "PREMARKET"}:
        timing = "Because this is before the 11:00 NPT regular session, the reaction should hit the opening auction and first hour of cash trading."
    else:
        timing = "Because this is outside the regular 11:00-15:00 NPT session, the first clean read should come at the next open."

    stories = list(event_ctx.get("stories") or [])
    social = list(event_ctx.get("social") or [])
    lead_ref = ""
    if stories:
        lead_ref = _clip_text(stories[0].get("title"), 96)
    elif social:
        lead_ref = _clip_text(social[0].get("text"), 96)

    evidence_parts = [
        f"breadth is currently ▲{breadth_up} vs ▼{breadth_down}",
        f"the regime is {regime or 'unknown'}",
    ]
    if lead_ref:
        evidence_parts.append(f"OSINT lead is \"{lead_ref}\"")
    evidence = ", ".join(evidence_parts)

    invalidation = (
        "I would fade this call only if follow-up headlines shift into protests, arrests, or escalation."
        if "upward" in base_case
        else "I would only soften that view if follow-up headlines show de-escalation and breadth recovers fast after the open."
    )
    return (
        f"Base case: {base_case} {horizon}. "
        f"{timing} The reason is that {evidence}. {invalidation}"
    )


def ask(question: str) -> str:
    """Ask the agent a follow-up question with full context injection."""
    question = _clip_text(question, 900)
    directional_market_call = _question_is_directional_market_call(question)

    # Load latest analysis
    ctx_text = ""
    nepal_clock = _nepal_market_clock()
    schedule = get_market_schedule()
    analysis = load_agent_analysis()
    if not _analysis_cache_is_fresh(analysis):
        if not list(analysis.get("stocks") or []):
            try:
                analysis = build_algo_shortlist_snapshot()
            except Exception:
                analysis = load_agent_analysis()
    if analysis:
        try:
            a = analysis
            ctx_text = f"""Your latest analysis:
Market view: {a.get('market_view', 'N/A')}
Trade today: {a.get('trade_today', 'N/A')} — {a.get('trade_today_reason', '')}
Risks: {', '.join(a.get('risks', []))}
Fresh market: {json.dumps(a.get('fresh_market', {}), default=str)}

Stock verdicts:
"""
            for s in a.get("stocks", []):
                ctx_text += (
                    f"  {s['symbol']}: {s['verdict']} ({s.get('conviction', '?'):.0%}) "
                    f"— {s.get('what_matters', s.get('reasoning', ''))[:120]}\n"
                )
        except Exception:
            pass

    # Auto-inject metrics if question is about a specific stock
    stock_ctx = ""
    sym = _extract_symbol_from_question(question)
    if sym:
        try:
            from apps.classic.dashboard import MD
            md = MD(top_n=5)
            price = md.ltps().get(sym, 0)
        except Exception:
            price = 0
        m = _compute_stock_metrics(sym, price)
        formatted = _format_metrics(m)
        if formatted:
            stock_ctx = f"\nFINANCIAL DATA for {sym}:\n{formatted}\n"

    # Vector search for relevant OSINT context
    vector_ctx = _vector_search_for_question(question)
    event_ctx = _event_market_context(question) if directional_market_call else {}
    event_osint_ctx = str(event_ctx.get("context_text") or "")

    macro_ctx = _format_macro_context(_fetch_macro_market_context(), max_fx=4, max_commodities=3, max_energy=2)

    portfolio_rows, portfolio_meta = _load_active_portfolio()
    portfolio_ctx = (
        f"ACTIVE ACCOUNT: {portfolio_meta.get('name') or portfolio_meta.get('id') or 'account'} "
        f"({portfolio_meta.get('id') or 'account_1'})\n"
    )
    if portfolio_rows:
        holdings = ", ".join(
            f"{row['symbol']} x{int(row['qty'])}"
            for row in portfolio_rows[:8]
        )
        portfolio_ctx += f"ACTIVE HOLDINGS: {holdings}\n"
    else:
        portfolio_ctx += "ACTIVE HOLDINGS: none\n"

    # Chat history
    include_history = not _question_is_time_sensitive(question)
    history = load_agent_history(limit=4, include_archive=False) if include_history else []

    history_text = ""
    if history:
        history_text = "\nRecent conversation:\n"
        for h in history[-6:]:
            role = str(h.get("role") or "").upper()
            speaker = "User" if role == "YOU" else "You" if role == "AGENT" else role.title()
            history_text += f"{speaker}: {str(h.get('message') or '')[:220]}\n"
        history_text += "\n"

    prompt = f"""NON-NEGOTIABLE NEPAL MARKET FACTS:
- Nepal weekday/date/time right now: {nepal_clock.get('weekday', 'unknown')}, {nepal_clock.get('session_date', 'unknown')} {nepal_clock.get('time_only', 'unknown')} {nepal_clock.get('timezone', 'NPT')}
- NEPSE session right now: {nepal_clock.get('market_phase', 'UNKNOWN')}
- NEPSE trading week: {schedule.get('trading_week', 'unknown')}
- NEPSE special pre-open session: {schedule.get('special_preopen', 'unknown')}
- NEPSE pre-open session: {schedule.get('preopen', 'unknown')}
- NEPSE regular session: {schedule.get('regular', 'unknown')}
- If you mention today, now, market hours, or session status, you must use these exact facts and must not contradict them.

{ctx_text}
{portfolio_ctx}
{stock_ctx}
{vector_ctx}
{event_osint_ctx}
{macro_ctx}
{history_text}
User question: {question}

IMPORTANT:
- Keep your answer SHORT — 2-4 sentences max.
- Be direct and conversational, like a quick terminal chat reply.
- No headers, no bullet lists, no markdown formatting. Just plain text.
- If you must list things, use commas inline.
- Use specific numbers from the financial data, latest analysis, and vector search when relevant.
- If asked about a stock, reference its valuation, margins, growth, or news — don't give a generic answer.
- Use Nepal time only when talking about timing, session status, market hours, or "today".
- If the market is closed, say it is closed; if it is pre-open, make that explicit.
- Do not confuse accounts. ACTIVE ACCOUNT and ACTIVE HOLDINGS above are the source of truth for whether a stock is already owned.
- If a stock is already held in ACTIVE HOLDINGS, your stance can be BUY, HOLD, or SELL in plain language.
- If a stock is not held, do not recommend HOLD; the stance should resolve to BUY or PASS/AVOID.
- For political, policy, or NEPSE reaction questions, you must take a base-case directional stance: upward pressure, downward pressure, or flat-to-upward/downward bias.
- Do not answer with "it could go either way", "depends entirely", or similar hedging for those event-driven market questions.
- If OSINT evidence is thin, still give the most likely direction and say what would invalidate that call.
- If evidence is missing, say it is missing rather than inventing facts."""

    response = _call_primary_agent(prompt, max_tokens=DEFAULT_AGENT_CHAT_MAX_TOKENS)
    if directional_market_call and _response_is_hedged_market_call(response):
        response = _build_directional_market_answer(question, analysis, nepal_clock, event_ctx)

    if str(os.environ.get("NEPSE_AGENT_DISABLE_HISTORY", "0") or "0").strip().lower() not in {"1", "true", "yes", "on"}:
        history = _load_combined_chat_history()
        ts = time.time()
        source_label = _agent_source_label()
        provider_label = _agent_provider_label()
        history.extend(
            [
                {"role": "YOU", "message": question, "ts": ts, "source": source_label, "provider": provider_label},
                {"role": "AGENT", "message": response, "ts": ts, "source": source_label, "provider": provider_label},
            ]
        )
        _persist_chat_history(history)

    return response


# ── Trade approval gate ──────────────────────────────────────────────────────

def check_trade_approval(symbol: str, action: str) -> tuple[bool, str]:
    """Check if the agent approves a specific trade."""
    if not ANALYSIS_FILE.exists():
        return True, "No agent analysis available — trade allowed by default"

    try:
        a = json.loads(ANALYSIS_FILE.read_text())
        age = time.time() - a.get("timestamp", 0)
        if age > 7200:
            return True, "Agent analysis stale (>2h) — trade allowed by default"

        if not a.get("trade_today", True):
            return False, f"Agent says NO TRADING today: {a.get('trade_today_reason', 'no reason')}"

        for s in a.get("stocks", []):
            if s["symbol"].upper() == symbol.upper():
                verdict = s.get("verdict", "APPROVE").upper()
                what = s.get("what_matters", s.get("reasoning", ""))[:120]
                if verdict == "APPROVE":
                    return True, f"APPROVED: {what}"
                elif verdict == "REJECT":
                    return False, f"REJECTED: {what}"
                else:
                    return True, f"HOLD (allowing): {what}"

        return True, f"{symbol} not in agent's shortlist — trade allowed"

    except Exception as e:
        return True, f"Agent check error: {e}"


def build_agent_trade_decisions(force: bool = False) -> list[AgentDecision]:
    """Map the latest agent analysis into explicit operator decisions."""
    analysis = analyze(force=force)
    decisions: list[AgentDecision] = []
    for stock in list(analysis.get("stocks") or []):
        verdict = str(stock.get("verdict") or "").upper()
        if verdict not in {"APPROVE", "HOLD"}:
            continue
        confidence = float(stock.get("conviction") or 0.0)
        decisions.append(
            AgentDecision(
                action="buy",
                symbol=str(stock.get("symbol") or "").upper(),
                quantity=0,
                limit_price=None,
                thesis=str(stock.get("what_matters") or stock.get("reasoning") or ""),
                catalysts=[str(stock.get("bull_case") or "")] if stock.get("bull_case") else [],
                risk=[str(stock.get("bear_case") or "")] if stock.get("bear_case") else [],
                confidence=confidence,
                horizon="1-4 weeks",
                source_signals=[str(stock.get("algo_signal") or "BUY")],
                metadata={
                    "verdict": verdict,
                    "sector": str(stock.get("sector") or ""),
                    "raw_reasoning": str(stock.get("reasoning") or ""),
                },
            )
        )
    return decisions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ask":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What's your market view today?"
        print(ask(q))
    else:
        print("Running agent analysis...")
        result = analyze(force=True)
        print(json.dumps(result, indent=2, default=str))
