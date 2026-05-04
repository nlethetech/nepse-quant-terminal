"""
NEPSE Agent Analyst — local-first equity research overlay.

Uses local Ollama by default, with optional Gemma 4 MLX or Claude backends, to
perform structured bull/bear analysis on algorithmic shortlists, cross-referencing
OSINT intelligence, quarterly financials, and market regime.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from configs.long_term import LONG_TERM_CONFIG
from backend.agents.runtime_config import (
    DEFAULT_GEMMA4_MODEL as RUNTIME_DEFAULT_GEMMA4_MODEL,
    DEFAULT_OLLAMA_MODEL as RUNTIME_DEFAULT_OLLAMA_MODEL,
    EXPERIMENTAL_GEMMA4_MODEL as RUNTIME_EXPERIMENTAL_GEMMA4_MODEL,
    load_active_agent_config,
)
from backend.quant_pro.nepse_calendar import current_nepal_datetime, get_market_schedule, market_session_phase
from backend.quant_pro.control_plane.models import AgentDecision
from backend.quant_pro.nepalosint_client import (
    consolidated_stories,
    consolidated_stories_history,
    resolve_osint_base_url,
    semantic_story_search,
    symbol_intelligence,
    unified_search,
)
from backend.quant_pro.paths import ensure_dir, get_project_root, get_runtime_dir, migrate_legacy_path
from backend.quant_pro.signal_ranking import canonicalize_signal_symbol
from backend.trading import strategy_registry

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
AGENT_SIGNAL_SNAPSHOT_FILE = AGENTS_RUNTIME_DIR / "active_signals_snapshot.json"
MAX_AGENT_HISTORY_ITEMS = 12
MAX_AGENT_ARCHIVE_ITEMS = 240
AGENT_SHORTLIST_LIMIT = 10
AGENT_OSINT_DECISION_HOURS = 24
SUPER_SIGNAL_MIN_SCORE = 0.75
SUPER_SIGNAL_MIN_STRENGTH = 1.0
SUPER_SIGNAL_MIN_CONFIDENCE = 0.75
SUPER_SIGNAL_MIN_CONVICTION = 0.9
ANALYSIS_CACHE_MAX_AGE_SECS = 900
DEFAULT_AGENT_BACKEND = "ollama"
DEFAULT_GEMMA4_MLX_MODEL = RUNTIME_DEFAULT_GEMMA4_MODEL
DEFAULT_GEMMA4_EXPERIMENTAL_MODEL = RUNTIME_EXPERIMENTAL_GEMMA4_MODEL
DEFAULT_OLLAMA_MODEL = RUNTIME_DEFAULT_OLLAMA_MODEL
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
ANALYST_PATTERN_TAXONOMY = {
    "earnings_quality_mixed": "Profit improves but asset quality, fee income, or other quality indicators weaken.",
    "project_delay_execution_risk": "Construction, commissioning, or infrastructure slippage delays monetization.",
    "financing_dilution_stress": "Rights issue, refinancing, or working-capital stress raises dilution or funding risk.",
    "policy_support_with_lag": "Policy is directionally supportive but the financial impact arrives with a lag.",
    "policy_or_tax_rumor_conflict": "Rumor-driven sentiment moves ahead of official confirmation.",
    "input_cost_margin_pressure": "FX or commodity costs rise faster than the company can pass them through.",
    "transmission_or_curtailment_revenue_risk": "Grid, offtake, or curtailment issues reduce realizable revenue.",
}

_MLX_MODEL = None
_MLX_PROCESSOR = None
_MLX_MODEL_ID: str | None = None
_MLX_LOCK = threading.Lock()

# ── System prompt: defines the analyst's identity and framework ──────────────

SYSTEM_PROMPT = """You are a disciplined NEPSE equity research analyst. You combine quantitative signals with qualitative intelligence to make sharp, defensible analyst notes and trade reviews.

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
- Use only the supplied context. Do not invent filings, headlines, dates, prices, or policy facts that are not present.
- Separate observed facts from inference. Facts should stay source-grounded; interpretation should explain the mechanism.
- When the prompt includes source ids, cite them for factual claims and list only ids you actually used.
- When source ids are available, each factual bullet should carry the source id instead of leaving support implicit.
- If evidence is weak, missing, or conflicting, say that directly and lower confidence.
- If the supplied evidence includes Nepali, translate it into analyst-grade English while preserving entities, dates, percentages, units, and causal language.
- If translation nuance matters, call it out explicitly instead of smoothing it over.
- Ban generic filler. Avoid lines like "this may be positive" unless you explain the transmission channel.
- Mechanism standard: observed event -> channel -> likely effect on earnings, cash flow, funding cost, valuation, or positioning.
- If the prompt provides a pattern taxonomy, map the event to the best-fit pattern class and explain why.
- When the response is a note or research answer rather than raw JSON, prefer this order: Summary, Key facts, Translation notes, Why it matters, Mechanism, Pattern match, Likely relevance, Risks or counterpoints, Confidence, Missing information, Sources.
- If no taxonomy fit is defensible, say no_clear_match instead of forcing a pattern.
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

RESPONSE STYLE: Direct, evidence-based, concise. State your view, separate facts from inference, explain why it matters, and surface uncertainty instead of hiding it. Do not drift into generic finance-summary language."""


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
    backend = str(cfg.get("backend") or DEFAULT_AGENT_BACKEND).strip().lower()
    fallback = DEFAULT_OLLAMA_MODEL if backend == "ollama" else DEFAULT_GEMMA4_MLX_MODEL
    return str(cfg.get("model") or fallback).strip()


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
    return str(cfg.get("fallback_backend") or "").strip().lower()


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


def _call_ollama(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = DEFAULT_AGENT_MAX_TOKENS) -> str:
    """Call a local Ollama model via its REST API (http://localhost:11434)."""
    try:
        import urllib.request, json as _json
        cfg = load_active_agent_config()
        model = str(cfg.get("model") or DEFAULT_OLLAMA_MODEL).strip()
        host = str(cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
        payload = _json.dumps({
            "model": model,
            "prompt": f"<system>\n{system}\n</system>\n\n{prompt}",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = _json.loads(resp.read().decode())
        return str(data.get("response") or "").strip() or "ERROR: Ollama returned empty response"
    except Exception as exc:
        return f"ERROR: Ollama call failed: {exc}"


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
    if backend == "ollama":
        return _call_ollama(prompt, system=system, max_tokens=max_tokens)
    return _call_claude(prompt, system=system, max_tokens=max_tokens)


def load_agent_analysis() -> dict:
    """Load latest agent analysis from runtime storage."""
    if not ANALYSIS_FILE.exists():
        return {}
    try:
        return json.loads(ANALYSIS_FILE.read_text())
    except Exception:
        return {}


def load_agent_signal_snapshot() -> dict:
    if not AGENT_SIGNAL_SNAPSHOT_FILE.exists():
        return {}
    try:
        payload = json.loads(AGENT_SIGNAL_SNAPSHOT_FILE.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def publish_agent_signal_snapshot(snapshot: dict) -> dict:
    payload = dict(snapshot or {})
    payload.setdefault("saved_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    AGENT_SIGNAL_SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2, default=str))
    return payload


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
    account = _active_account_context()
    strategy = _active_strategy_context()
    payload.setdefault("timestamp", time.time())
    payload.setdefault("context_date", now_utc.strftime("%Y-%m-%d"))
    payload.setdefault("account_id", str(account.get("id") or "account_1"))
    payload.setdefault("account_name", str(account.get("name") or payload.get("account_id") or "account_1"))
    payload.setdefault("strategy_id", str(strategy.get("id") or "default_c5"))
    payload.setdefault("strategy_name", str(strategy.get("name") or payload.get("strategy_id") or "default_c5"))
    meta = dict(payload.get("agent_runtime_meta") or {})
    meta.update(
        {
            "source": source,
            "provider": provider,
            "updated_at": now_utc.replace(microsecond=0).isoformat(),
            "account_id": payload.get("account_id"),
            "account_name": payload.get("account_name"),
            "strategy_id": payload.get("strategy_id"),
            "strategy_name": payload.get("strategy_name"),
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
    if bool(payload.get("parse_error")):
        return False
    verdicts = {
        str(row.get("verdict") or "").upper()
        for row in list(payload.get("stocks") or [])
        if isinstance(row, dict)
    }
    verdicts.discard("")
    if verdicts and not (verdicts & {"APPROVE", "HOLD", "REJECT"}):
        return False
    age = time.time() - float(payload.get("timestamp") or 0.0)
    if age > ANALYSIS_CACHE_MAX_AGE_SECS:
        return False
    if str(payload.get("context_date") or "") != _current_nst_session_date():
        return False
    active_account = _active_account_context()
    if str(payload.get("account_id") or "") != str(active_account.get("id") or "account_1"):
        return False
    if payload.get("strategy_id") is not None:
        active_strategy = _active_strategy_context()
        return str(payload.get("strategy_id") or "") == str(active_strategy.get("id") or "")
    return True


def _clip_text(value: object, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def _agent_osint_base() -> str:
    return resolve_osint_base_url()


def _story_title(raw: dict) -> str:
    return str(
        raw.get("canonical_headline")
        or raw.get("headline")
        or raw.get("title")
        or raw.get("summary")
        or ""
    ).strip()


def _story_url(raw: dict) -> str:
    return str(raw.get("url") or raw.get("canonical_url") or raw.get("link") or "").strip()


def _story_time(raw: dict) -> str:
    return str(
        raw.get("first_reported_at")
        or raw.get("published_at")
        or raw.get("created_at")
        or raw.get("tweeted_at")
        or ""
    ).strip()[:16]


def _story_ref(raw: dict) -> tuple[str, str]:
    return (_story_title(raw), _story_url(raw))


def _normalize_semantic_story_item(raw: dict) -> dict:
    item = dict(raw or {})
    if not _story_title(item):
        item["title"] = item.get("headline") or item.get("canonical_headline") or item.get("text") or item.get("summary") or ""
    if not _story_url(item):
        item["url"] = item.get("canonical_url") or item.get("link") or ""
    if not _story_time(item):
        item["published_at"] = item.get("published_at_utc") or item.get("date") or item.get("created_at") or ""
    if not (item.get("source_name") or item.get("source")):
        item["source_name"] = item.get("source") or "NepalOSINT"
    return item


def _format_citable_story(item: dict) -> str:
    title = _clip_text(_story_title(item), 150)
    source = str(item.get("source_name") or item.get("source") or "Unknown source").strip()
    timestamp = _story_time(item)
    url = _story_url(item)
    parts = [part for part in (title, source, timestamp, url) if part]
    return " | ".join(parts)


def _format_story_source_ref(item: dict, ref_id: int) -> str:
    title = _clip_text(_story_title(item), 120)
    source = str(item.get("source_name") or item.get("source") or "Unknown source").strip()
    timestamp = _story_time(item)
    url = _story_url(item)
    parts = [part for part in (source, timestamp, url) if part]
    suffix = " | ".join(parts)
    return f"[{ref_id}] {title}" + (f" | {suffix}" if suffix else "")


def _detect_text_language(*texts: object) -> str:
    blob = " ".join(str(text or "") for text in texts)
    if not blob.strip():
        return "unknown"
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", blob))
    has_ascii_letters = bool(re.search(r"[A-Za-z]", blob))
    if has_devanagari and has_ascii_letters:
        return "mixed"
    if has_devanagari:
        return "ne"
    if has_ascii_letters:
        return "en"
    return "unknown"


def _analyst_pattern_taxonomy_text() -> str:
    lines = ["ANALYST PATTERN TAXONOMY:"]
    for key, description in ANALYST_PATTERN_TAXONOMY.items():
        lines.append(f"  - {key}: {description}")
    return "\n".join(lines) + "\n"


def _analysis_source_id(prefix: str, suffix: str) -> str:
    clean_prefix = re.sub(r"[^A-Z0-9]+", "_", str(prefix or "").upper()).strip("_")
    clean_suffix = re.sub(r"[^A-Z0-9]+", "_", str(suffix or "").upper()).strip("_")
    joined = "_".join(part for part in (clean_prefix, clean_suffix) if part)
    return joined[:48] or "SOURCE"


def _build_metric_source_packet(symbol: str, metrics: dict | None, last_price: float = 0.0) -> Optional[dict]:
    formatted = _format_metrics({"symbol": symbol, **dict(metrics or {})})
    if not formatted:
        return None
    text = formatted
    if last_price > 0:
        text = f"{text}  CMP={last_price:.1f}"
    return {
        "id": _analysis_source_id(symbol, "filing"),
        "label": "latest filing metrics",
        "language": "en",
        "text": text,
    }


def _build_story_source_packets(prefix: str, items: list[dict] | None, *, limit: int = 2) -> list[dict]:
    packets: list[dict] = []
    for idx, item in enumerate(list(items or [])[:limit], start=1):
        text = _format_citable_story(dict(item or {}))
        if not text:
            continue
        language = _detect_text_language(
            dict(item or {}).get("canonical_headline"),
            dict(item or {}).get("headline"),
            dict(item or {}).get("title"),
            dict(item or {}).get("summary"),
        )
        packets.append(
            {
                "id": _analysis_source_id(prefix, f"news{idx}"),
                "label": f"{prefix} news {idx}",
                "language": language,
                "text": text,
            }
        )
    return packets


def _build_analysis_source_packets(
    ctx: dict,
    metrics_map: dict[str, dict],
    intel_map: dict[str, dict],
    sector_intel_map: dict[str, dict],
) -> dict:
    packets: list[dict] = []
    by_symbol: dict[str, list[str]] = {}
    prices = dict(ctx.get("prices") or {})

    market_packet = {
        "id": "MARKET_STATE",
        "label": "market state",
        "text": (
            f"Date {ctx.get('session_date', 'unknown')} | NEPSE {ctx.get('nepse_index', 'N/A')} "
            f"({ctx.get('nepse_change_pct', 'N/A')}%) | Breadth ▲{ctx.get('advancers', '?')} "
            f"▼{ctx.get('decliners', '?')} | Regime {ctx.get('regime', 'unknown')}"
        ),
    }
    packets.append(market_packet)

    for signal in list(ctx.get("signals") or []):
        symbol = str(signal.get("symbol") or "").upper()
        if not symbol:
            continue
        ids: list[str] = []

        metric_packet = _build_metric_source_packet(
            symbol,
            metrics_map.get(symbol),
            float(prices.get(symbol) or 0.0),
        )
        if metric_packet:
            packets.append(metric_packet)
            ids.append(metric_packet["id"])

        symbol_story_packets = _build_story_source_packets(
            symbol,
            list(dict(intel_map.get(symbol) or {}).get("story_items") or []),
            limit=2,
        )
        for packet in symbol_story_packets:
            packets.append(packet)
            ids.append(packet["id"])

        sector_name = str(metrics_map.get(symbol, {}).get("sector") or "").strip()
        sector_key = _sector_key(sector_name)
        sector_packets = _build_story_source_packets(
            sector_key or symbol,
            list(dict(sector_intel_map.get(sector_key) or {}).get("story_items") or []),
            limit=1,
        )
        for packet in sector_packets:
            packets.append(packet)
            ids.append(packet["id"])

        by_symbol[symbol] = ids

    lines = ["CITABLE SOURCE PACKETS:"]
    for packet in packets:
        lines.append(
            f"  [{packet['id']}][{packet.get('language') or 'unknown'}] {packet['label']}: {packet['text']}"
        )
    return {"packets": packets, "by_symbol": by_symbol, "text": "\n".join(lines) + "\n"}


def _story_blob(raw: dict) -> str:
    return " ".join(
        str(
            raw.get(key) or ""
        ).strip()
        for key in (
            "canonical_headline",
            "headline",
            "title",
            "summary",
            "content",
            "story_type",
            "category",
            "source_name",
            "source",
        )
    ).strip().lower()


def _story_published_dt(raw: dict) -> Optional[datetime]:
    value = (
        raw.get("first_reported_at")
        or raw.get("published_at")
        or raw.get("created_at")
        or raw.get("tweeted_at")
        or ""
    )
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_news_date_window(question: str) -> dict:
    text = str(question or "")
    now = datetime.now(timezone.utc)
    lowered = text.lower()
    explicit_dates: list[datetime] = []

    for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        try:
            explicit_dates.append(datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc))
        except ValueError:
            pass
    for match in re.finditer(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,)?\s+(20\d{2})\b",
        text,
        flags=re.IGNORECASE,
    ):
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            explicit_dates.append(datetime(int(match.group(3)), months[match.group(1)[:3].lower()], int(match.group(2)), tzinfo=timezone.utc))
        except ValueError:
            pass

    if explicit_dates:
        start = min(explicit_dates).replace(hour=0, minute=0, second=0, microsecond=0)
        end = max(explicit_dates).replace(hour=23, minute=59, second=59, microsecond=0)
        hours = max(24, int((now - start).total_seconds() // 3600) + 48)
        return {"start": start, "end": end, "hours": hours, "label": f"{start.date().isoformat()} to {end.date().isoformat()}"}

    rel = re.search(r"\blast\s+(\d{1,3})\s*(h|hr|hrs|hour|hours)\b", lowered)
    if rel:
        hours = max(1, min(24 * 3650, int(rel.group(1))))
        start = now - timedelta(hours=hours)
        return {"start": start, "end": now, "hours": hours + 2, "label": f"last {hours} hours"}

    rel = re.search(r"\blast\s+(\d{1,3})\s*(d|day|days)\b", lowered)
    if rel:
        days = max(1, min(3650, int(rel.group(1))))
        start = now - timedelta(days=days)
        return {"start": start, "end": now, "hours": days * 24 + 24, "label": f"last {days} days"}

    if "last week" in lowered or "past week" in lowered:
        start = now - timedelta(days=7)
        return {"start": start, "end": now, "hours": 8 * 24, "label": "last 7 days"}
    if "last month" in lowered or "past month" in lowered:
        start = now - timedelta(days=31)
        return {"start": start, "end": now, "hours": 32 * 24, "label": "last 31 days"}
    if "recent" in lowered:
        start = now - timedelta(days=14)
        return {"start": start, "end": now, "hours": 15 * 24, "label": "recent 14 days"}

    return {"start": None, "end": None, "hours": 720, "label": ""}


def _story_in_date_window(item: dict, date_window: dict) -> bool:
    start = date_window.get("start")
    end = date_window.get("end")
    if start is None and end is None:
        return True
    published = _story_published_dt(item)
    if published is None:
        return False
    if start is not None and published < start:
        return False
    if end is not None and published > end:
        return False
    return True


def _date_window_is_explicit(date_window: dict) -> bool:
    return date_window.get("start") is not None and date_window.get("end") is not None


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


def _active_strategy_context() -> dict:
    strategy_id = str(os.environ.get("NEPSE_ACTIVE_STRATEGY_ID", "") or "").strip()
    strategy_name = str(os.environ.get("NEPSE_ACTIVE_STRATEGY_NAME", "") or "").strip()
    payload = strategy_registry.load_strategy(strategy_id) if strategy_id else None
    config = dict((payload or {}).get("config") or {})
    if not strategy_id:
        strategy_id = "default_c5"
    if not strategy_name:
        strategy_name = str((payload or {}).get("name") or strategy_registry.strategy_name(strategy_id) or strategy_id)
    if not config:
        config = dict(LONG_TERM_CONFIG)
    return {
        "id": strategy_id,
        "name": strategy_name,
        "config": config,
    }


def _matching_signal_snapshot() -> dict:
    snapshot = load_agent_signal_snapshot()
    if not snapshot:
        return {}
    active_account = _active_account_context()
    active_strategy = _active_strategy_context()
    if str(snapshot.get("account_id") or "") != str(active_account.get("id") or ""):
        return {}
    if str(snapshot.get("strategy_id") or "") != str(active_strategy.get("id") or ""):
        return {}
    if str(snapshot.get("context_date") or "") != _current_nst_session_date():
        return {}
    signals = list(snapshot.get("signals") or [])
    if not signals:
        return {}
    return snapshot


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


def _sector_key(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def _summarize_sector_intelligence(sector: str, intel: dict | None) -> str:
    payload = dict(intel or {})
    stories = list(payload.get("story_items") or [])
    social = list(payload.get("social_items") or [])
    if not stories and not social:
        return f"{sector}: no direct 24h NepalOSINT sector hit."
    parts = [
        f"{sector}: {int(payload.get('story_count') or 0)} story hits, {int(payload.get('social_count') or 0)} social hits in the last {AGENT_OSINT_DECISION_HOURS}h.",
    ]
    if stories:
        lead = dict(stories[0] or {})
        parts.append(
            f"Lead sector story: {_clip_text(lead.get('title'), 110)} "
            f"[{lead.get('source_name') or '?'}] {(lead.get('published_at') or '')[:16]}"
        )
    elif social:
        lead = dict(social[0] or {})
        parts.append(
            f"Lead sector social: {_clip_text(lead.get('text'), 110)} "
            f"[@{lead.get('author_username') or '?'}]"
        )
    return " ".join(parts)


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
    sector_intel: dict | None,
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

    sector_payload = dict(sector_intel or {})
    sector_story_items = list(sector_payload.get("story_items") or [])
    sector_social_items = list(sector_payload.get("social_items") or [])
    top_sector_story = dict(sector_story_items[0] or {}) if sector_story_items else {}
    top_sector_social = dict(sector_social_items[0] or {}) if sector_social_items else {}
    sector_bias = _osint_keyword_bias(
        top_sector_story.get("title"),
        top_sector_story.get("summary"),
        top_sector_social.get("text"),
    )
    base_conviction += sector_bias * 0.6
    if sector_story_items or sector_social_items:
        base_conviction += min(0.005 * (len(sector_story_items) + len(sector_social_items)), 0.03)

    conviction = _clamp_conviction(max(0.15, min(0.95, base_conviction)))
    severe_negative = bool(red_flags) or osint_bias <= -0.08 or sector_bias <= -0.08 or npl_pct > 5 or pe_ratio > 35 or pbv_ratio > 5

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
    sector_lead = _clip_text(top_sector_story.get("title"), 96) if top_sector_story else ""
    if not evidence_parts and sig.get("reasoning"):
        evidence_parts.append(_clip_text(sig.get("reasoning"), 84))
    if lead_story:
        evidence_parts.append(f"OSINT lead: {lead_story}")
    if sector_lead:
        evidence_parts.append(f"sector lead: {sector_lead}")

    bull_case = merged.get("bull_case") or (
        f"Quant setup is supported by {_clip_text(', '.join(evidence_parts), 110)}."
        if evidence_parts else
        f"Signal score {score:.2f} and confidence {confidence:.0%} keep {symbol} on the long radar."
    )
    bear_case = merged.get("bear_case") or (
        "; ".join(red_flags[:2]) if red_flags else
        (f"Sector tone is weak around {_clip_text(top_sector_story.get('title'), 80)}." if sector_bias < -0.04 and top_sector_story else
         f"OSINT tone is negative around {_clip_text(top_story.get('title'), 80)}." if osint_bias < -0.04 and top_story else
         f"Signal strength {strength:.2f} still needs confirmation.")
    )
    what_matters = merged.get("what_matters") or (
        _clip_text(f"Sector context: {sector_lead}", 120) if sector_lead and abs(sector_bias) >= max(abs(osint_bias), 0.05) else
        _clip_text(lead_story, 120) if lead_story else
        _clip_text(", ".join(evidence_parts), 120) if evidence_parts else
        _clip_text(sig.get("reasoning"), 120)
    )
    reasoning = merged.get("reasoning") or (
        f"{symbol} scores {score:.2f} with {confidence:.0%} signal confidence. "
        f"Financial read: {_clip_text(_format_metrics({'symbol': symbol, **metrics}), 180) or 'limited recent filing data'}. "
        f"NepalOSINT: {_clip_text(_summarize_symbol_intelligence(symbol, intel), 200)}. "
        f"Sector check: {_clip_text(_summarize_sector_intelligence(str(metrics.get('sector') or 'Unknown'), sector_payload), 180)}."
    )
    summary = merged.get("summary") or (
        f"{symbol} has a {'constructive' if verdict == 'APPROVE' else 'mixed' if verdict == 'HOLD' else 'fragile'} setup, "
        f"but the case depends on {_clip_text(what_matters, 90)}."
    )
    key_facts = list(merged.get("key_facts") or [])
    if not key_facts:
        if metrics.get("profit_growth_qoq_pct") is not None:
            key_facts.append(f"Profit growth QoQ is {float(metrics.get('profit_growth_qoq_pct') or 0.0):.1f}%")
        if metrics.get("revenue_growth_qoq_pct") is not None:
            key_facts.append(f"Revenue growth QoQ is {float(metrics.get('revenue_growth_qoq_pct') or 0.0):.1f}%")
        if metrics.get("profit_margin_pct") is not None:
            key_facts.append(f"Profit margin is {float(metrics.get('profit_margin_pct') or 0.0):.1f}%")
        if lead_story:
            key_facts.append(f"Lead NepalOSINT story: {lead_story}")
    why_it_matters = merged.get("why_it_matters") or (
        f"The observed setup matters because {_clip_text(what_matters, 110)} affects the near-term evidence needed to justify a position."
    )
    mechanism = merged.get("mechanism") or (
        f"Observed setup -> investor focus shifts to {_clip_text(what_matters, 80).lower()} -> near-term pricing reflects whether that evidence confirms or weakens the thesis."
    )
    language_detected = str(merged.get("language_detected") or "en")
    translation_quality_notes = list(merged.get("translation_quality_notes") or [])
    if not translation_quality_notes:
        translation_quality_notes = ["No material translation ambiguity flagged in the fallback path."]
    likely_impact = merged.get("likely_impact") or (
        f"{'Moderately positive' if verdict == 'APPROVE' else 'Neutral to mixed' if verdict == 'HOLD' else 'Negative'} for {symbol} over the next 1-4 weeks because "
        f"{_clip_text(why_it_matters, 120).lower()}"
    )
    likely_relevance = merged.get("likely_relevance") or likely_impact
    risks_counterpoints = list(merged.get("risks_counterpoints") or [])
    if not risks_counterpoints:
        risks_counterpoints = [bear_case]
        if red_flags:
            risks_counterpoints.extend(red_flags[:1])
    confidence_note = merged.get("confidence_note") or (
        f"{'High' if conviction >= 0.75 else 'Medium' if conviction >= 0.5 else 'Low'} because the setup has "
        f"{'multiple aligned signals' if conviction >= 0.75 else 'some support but still open questions' if conviction >= 0.5 else 'thin or conflicting evidence'}."
    )
    uncertainty_notes = list(merged.get("uncertainty_notes") or [])
    if not uncertainty_notes:
        uncertainty_notes = ["Fallback path used because the model did not return a fully structured multilingual analysis."]
    missing_information = list(merged.get("missing_information") or [])
    if not missing_information:
        missing_information.append("Fresh confirming evidence is limited if the model did not return explicit source-backed facts.")
    cited_sources = list(merged.get("cited_sources") or [])
    historical_pattern_class = str(merged.get("historical_pattern_class") or "no_clear_match")

    merged.update(
        {
            "verdict": verdict,
            "conviction": conviction,
            "language_detected": language_detected,
            "translation_quality_notes": translation_quality_notes[:3],
            "summary": summary,
            "key_facts": key_facts[:4],
            "why_it_matters": why_it_matters,
            "mechanism": mechanism,
            "historical_pattern_class": historical_pattern_class,
            "likely_impact": likely_impact,
            "likely_relevance": likely_relevance,
            "risks_counterpoints": risks_counterpoints[:3],
            "confidence_note": confidence_note,
            "uncertainty_notes": uncertainty_notes[:3],
            "missing_information": missing_information[:3],
            "cited_sources": cited_sources,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "what_matters": what_matters,
            "reasoning": reasoning,
            "sector": str(merged.get("sector") or metrics.get("sector") or ""),
        }
    )
    return merged


# ── Context gathering ────────────────────────────────────────────────────────

def _gather_context(*, preview_only: bool = False) -> dict:
    """Gather all context for the agent: signals, news, regime, portfolio."""
    context = {}
    fresh_market = _refresh_intraday_market_snapshot()
    if fresh_market:
        context["fresh_market"] = fresh_market
    context["macro_market"] = _fetch_macro_market_context()
    active_strategy = _active_strategy_context()
    strategy_config = dict(active_strategy.get("config") or {})
    signal_types = list(strategy_config.get("signal_types") or list(LONG_TERM_CONFIG["signal_types"]))
    use_regime_filter = bool(strategy_config.get("use_regime_filter", True))
    context["strategy_id"] = str(active_strategy.get("id") or "default_c5")
    context["strategy_name"] = str(active_strategy.get("name") or "Live C31")
    context["strategy_signal_types"] = list(signal_types)
    signal_snapshot = _matching_signal_snapshot()

    # 1. Algorithm signals + price data
    try:
        from apps.classic.dashboard import MD
        md = MD(top_n=10)
        regime_blocked = False
        if signal_snapshot:
            sigs = list(signal_snapshot.get("signals") or [])
            regime = str(signal_snapshot.get("regime") or "unknown")
        else:
            from backend.backtesting.simple_backtest import load_all_prices
            from apps.classic.dashboard import _db
            from backend.trading.live_trader import generate_signals

            conn = _db()
            prices_df = load_all_prices(conn)
            conn.close()
            sigs, regime = generate_signals(
                prices_df,
                signal_types,
                use_regime_filter=use_regime_filter,
            )
            if not sigs and str(regime).lower() == "bear":
                sigs, regime = generate_signals(
                    prices_df,
                    signal_types,
                    use_regime_filter=False,
                )
                regime_blocked = True
        sigs = list(sigs or [])[:AGENT_SHORTLIST_LIMIT]

        context["signals"] = [
            {
                "symbol": canonicalize_signal_symbol(s.get("symbol")),
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
        context["signals_source"] = "snapshot" if signal_snapshot else "generated"

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
            canonicalize_signal_symbol(s.get("symbol")): _compute_stock_metrics(
                canonicalize_signal_symbol(s.get("symbol")),
                float((context["prices"] or {}).get(canonicalize_signal_symbol(s.get("symbol"))) or 0.0),
            )
            for s in context["signals"]
            if str(s.get("symbol") or "").strip()
        }

    except Exception as e:
        context["signals"] = []
        context["signal_error"] = str(e)
        context["signal_metrics"] = {}

    portfolio_rows, portfolio_meta = _load_active_portfolio()
    context["portfolio"] = portfolio_rows
    context["portfolio_account"] = portfolio_meta

    if preview_only:
        context["symbol_intelligence"] = {}
        context["sector_intelligence"] = {}
        context["embeddings"] = []
        context["news"] = []
        return context

    # 2. NepalOSINT semantic + unified + related-story search per shortlisted stock
    try:
        symbol_intel: dict[str, dict] = {}
        sector_intel: dict[str, dict] = {}
        embeddings_context = []
        base_url = _agent_osint_base()
        signal_metrics = dict(context.get("signal_metrics") or {})
        symbols = [
            str(s.get("symbol") or "").upper()
            for s in context.get("signals", [])
            if str(s.get("symbol") or "").strip() and "::" not in str(s.get("symbol") or "")
        ]
        symbols = list(dict.fromkeys(symbols))
        sector_pairs: dict[str, str] = {}

        def _fetch_symbol_intel(sym: str) -> tuple[str, dict]:
            return (
                sym,
                symbol_intelligence(
                    sym,
                    hours=AGENT_OSINT_DECISION_HOURS,
                    top_k=4,
                    min_similarity=0.45,
                    base_url=base_url,
                ),
            )

        if symbols:
            with ThreadPoolExecutor(max_workers=min(6, len(symbols))) as executor:
                futures = {executor.submit(_fetch_symbol_intel, sym): sym for sym in symbols}
                for future in as_completed(futures):
                    sym, intel = future.result()
                    symbol_intel[sym] = intel
                    sector_name = str(dict(signal_metrics.get(sym) or {}).get("sector") or "").strip()
                    sector_key = _sector_key(sector_name)
                    if sector_name and sector_key and sector_key not in sector_pairs:
                        sector_pairs[sector_key] = sector_name
                    for item in list(dict(intel.get("semantic") or {}).get("results") or [])[:2]:
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

        def _fetch_sector_intel(item: tuple[str, str]) -> tuple[str, dict]:
            sector_key, sector_name = item
            return (
                sector_key,
                symbol_intelligence(
                    sector_name,
                    hours=AGENT_OSINT_DECISION_HOURS,
                    top_k=3,
                    min_similarity=0.45,
                    base_url=base_url,
                ),
            )

        if sector_pairs:
            with ThreadPoolExecutor(max_workers=min(4, len(sector_pairs))) as executor:
                futures = {executor.submit(_fetch_sector_intel, item): item[0] for item in sector_pairs.items()}
                for future in as_completed(futures):
                    sector_key, intel = future.result()
                    sector_intel[sector_key] = intel
        context["symbol_intelligence"] = symbol_intel
        context["sector_intelligence"] = sector_intel
        context["embeddings"] = embeddings_context
    except Exception as e:
        context["symbol_intelligence"] = {}
        context["sector_intelligence"] = {}
        context["embeddings"] = []
        context["embeddings_error"] = str(e)

    # 3. OSINT news feed (last 48h)
    try:
        stories = consolidated_stories(limit=30, base_url=_agent_osint_base(), timeout=8)
        context["news"] = []
        for s in stories:
            headline = _story_title(s)
            summary = s.get("summary", "") or ""
            url = _story_url(s)
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
                    "source": s.get("source_name", "") or s.get("source", ""),
                    "text": text[:200],
                    "time": _story_time(s),
                    "url": url,
                })
    except Exception as e:
        context["news"] = []
        context["news_error"] = str(e)

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
        canonicalize_signal_symbol(item.get("symbol"))
        for item in list(ctx.get("portfolio") or [])
        if str(item.get("symbol") or "").strip()
    }
    metrics_map = {
        canonicalize_signal_symbol(key): dict(value or {})
        for key, value in dict(ctx.get("signal_metrics") or {}).items()
    }
    intel_map = {
        canonicalize_signal_symbol(key): dict(value or {})
        for key, value in dict(ctx.get("symbol_intelligence") or {}).items()
    }
    sector_intel_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("sector_intelligence") or {}).items()
    }
    preview_mode = bool(parsed.get("_preview"))
    parsed_rows = {
        canonicalize_signal_symbol(item.get("symbol")): dict(item)
        for item in list(parsed.get("stocks") or [])
        if str(item.get("symbol") or "").strip()
    }
    merged_rows: list[dict] = []
    for rank, sig in enumerate(signal_rows, 1):
        symbol = canonicalize_signal_symbol(sig.get("symbol"))
        is_held = symbol in portfolio_symbols
        row = _fallback_stock_decision(
            symbol,
            sig,
            dict(parsed_rows.get(symbol) or {}),
            metrics=metrics_map.get(symbol),
            intel=intel_map.get(symbol),
            sector_intel=sector_intel_map.get(_sector_key(metrics_map.get(symbol, {}).get("sector"))),
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
            "language_detected": str(row.get("language_detected") or "unknown"),
            "translation_quality_notes": list(row.get("translation_quality_notes") or []),
            "summary": str(row.get("summary") or ""),
            "key_facts": list(row.get("key_facts") or []),
            "why_it_matters": str(row.get("why_it_matters") or ""),
            "mechanism": str(row.get("mechanism") or ""),
            "historical_pattern_class": str(row.get("historical_pattern_class") or ""),
            "likely_impact": str(row.get("likely_impact") or ""),
            "likely_relevance": str(row.get("likely_relevance") or ""),
            "risks_counterpoints": list(row.get("risks_counterpoints") or []),
            "confidence_note": str(row.get("confidence_note") or ""),
            "uncertainty_notes": list(row.get("uncertainty_notes") or []),
            "missing_information": list(row.get("missing_information") or []),
            "cited_sources": list(row.get("cited_sources") or []),
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
    if not preview_mode:
        if parsed.get("trade_today") is None:
            enriched["trade_today"] = any(
                str(row.get("verdict") or "").upper() == "APPROVE"
                for row in merged_rows
            )
        enriched.setdefault(
            "trade_today_reason",
            "Deterministic fallback used because the agent response was unavailable or unparsable.",
        )
        enriched.setdefault(
            "market_view",
            "Fallback decision path used. Shortlist and pre-computed metrics were analyzed without a usable model response.",
        )
    enriched["super_signal_thresholds"] = {
        "score": SUPER_SIGNAL_MIN_SCORE,
        "strength": SUPER_SIGNAL_MIN_STRENGTH,
        "signal_confidence": SUPER_SIGNAL_MIN_CONFIDENCE,
        "agent_conviction": SUPER_SIGNAL_MIN_CONVICTION,
    }
    return enriched


def _extract_agent_json_payload(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    if start < 0:
        return {}
    candidate = text[start:]
    end = candidate.rfind("}")
    if end >= 0:
        candidate = candidate[: end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    marker = '"stocks": ['
    marker_idx = candidate.find(marker)
    if marker_idx < 0:
        return {}

    header = candidate[:marker_idx] + '"stocks": []}'
    try:
        repaired = json.loads(header)
    except json.JSONDecodeError:
        return {}
    if not isinstance(repaired, dict):
        return {}

    decoder = json.JSONDecoder()
    stocks_blob = candidate[marker_idx + len(marker):]
    rows: list[dict] = []
    idx = 0
    while idx < len(stocks_blob):
        while idx < len(stocks_blob) and stocks_blob[idx] in " \r\n\t,":
            idx += 1
        if idx >= len(stocks_blob) or stocks_blob[idx] == "]":
            break
        try:
            item, next_idx = decoder.raw_decode(stocks_blob, idx)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            rows.append(item)
        idx = next_idx
    repaired["stocks"] = rows
    repaired["_partial_parse"] = True
    repaired["_partial_stock_count"] = len(rows)
    return repaired


def build_algo_shortlist_snapshot() -> dict:
    """Return the ranked algo shortlist without waiting for full agent analysis."""
    ctx = _gather_context(preview_only=True)
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

    ctx = _gather_context(preview_only=False)
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

    sector_intelligence_text = ""
    sector_intel_map = {
        str(key).upper(): dict(value or {})
        for key, value in dict(ctx.get("sector_intelligence") or {}).items()
    }
    source_packets = _build_analysis_source_packets(ctx, metrics_map, intel_map, sector_intel_map)
    if sector_intel_map:
        lines = [f"SECTOR NEPALOSINT INTELLIGENCE (last {AGENT_OSINT_DECISION_HOURS}h):"]
        seen: set[str] = set()
        for s in ctx.get("signals", []):
            sym = str(s.get("symbol") or "").upper()
            sector_name = str(metrics_map.get(sym, {}).get("sector") or "").strip()
            sector_key = _sector_key(sector_name)
            if not sector_name or not sector_key or sector_key in seen:
                continue
            seen.add(sector_key)
            lines.append(f"  {_summarize_sector_intelligence(sector_name, sector_intel_map.get(sector_key))}")
        if len(lines) > 1:
            sector_intelligence_text = "\n".join(lines) + "\n"

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
{sector_intelligence_text}
{source_packets.get('text', '')}
{_analyst_pattern_taxonomy_text()}
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
   - SUMMARY: one thesis line that distinguishes fact from interpretation
   - KEY FACTS: 2-4 bullets, each with explicit evidence and a cited source id when available
   - TRANSLATION NOTES: if any cited source is Nepali or mixed-language, state the detected language and preserve financially important wording in English
   - WHY IT MATTERS: explain the mechanism from observed fact to stock/sector impact
   - MECHANISM: explicitly map event -> channel -> effect on earnings, margin, funding cost, execution, or valuation
   - PATTERN MATCH: assign the best-fit class from ANALYST PATTERN TAXONOMY and justify it
   - LIKELY IMPACT: direction, horizon, and why the move may be muted or strong
   - RISKS / COUNTERPOINTS: at least 2 concrete invalidation paths or missing-evidence items
   - CONFIDENCE NOTE: Low / Medium / High with one sentence explaining the evidence quality
   - CITED SOURCES: list only source ids from CITABLE SOURCE PACKETS that you actually used
   - Review ALL shortlisted stocks in rank order. Do not skip any real stock.
   - HOLD is only valid for symbols already present in ACTIVE ACCOUNT HOLDINGS. If a stock is not currently held in the active account, the final stance must resolve to APPROVE or REJECT.
   - You must explicitly check the last {AGENT_OSINT_DECISION_HOURS}h of NepalOSINT symbol and sector news before deciding. If sector news is negative, that is a real headwind even when the stock-specific chart looks good.
   - If a factual claim is not supported by the prompt context or source packets, say the evidence is missing rather than inventing support.
   - Use the source packet ids for filing metrics, symbol news, and sector news whenever you state those facts.

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
      "language_detected": "en or ne or mixed or unknown",
      "translation_quality_notes": ["translation nuance note", "or state no material translation ambiguity"],
      "summary": "1 sentence thesis",
      "key_facts": ["fact with citation [SOURCE_ID]", "fact with citation [SOURCE_ID]"],
      "why_it_matters": "fact-to-mechanism explanation",
      "mechanism": "event -> channel -> effect",
      "historical_pattern_class": "best-fit taxonomy class or no_clear_match",
      "likely_impact": "direction, horizon, and reason",
      "likely_relevance": "concise stock, sector, or market relevance statement",
      "risks_counterpoints": ["risk or counterpoint", "risk or missing evidence"],
      "confidence_note": "Low or Medium or High: evidence quality explanation",
      "uncertainty_notes": ["uncertainty or conflict", "uncertainty or conflict"],
      "missing_information": ["what evidence is still missing", "what would change the view"],
      "cited_sources": ["SOURCE_ID_1", "SOURCE_ID_2"],
      "bull_case": "strongest reason to buy",
      "bear_case": "strongest reason NOT to buy",
      "what_matters": "the one thing that actually matters for this stock right now",
      "reasoning": "2-3 sentences with facts, mechanism, and uncertainty"
    }}
  ]
}}

CRITICAL: Skip stocks with SECTOR:: prefix — those are index proxies, not tradeable.
Review every real stock in the top {ctx.get('shortlist_limit', AGENT_SHORTLIST_LIMIT)} shortlist. Use specific numbers from the metrics. Don't say "strong fundamentals" — say "P/E 8.2 on 42% margin with rising revenue."
Use Nepal time only. Treat all session, market-hours, and date references as NPT unless the prompt explicitly says otherwise.
REVIEW is forbidden in the final stock verdicts. Every real shortlisted stock must end as APPROVE, HOLD, or REJECT with conviction above 0.35.
The Sources section is for machine-readable ids only. Never fabricate or guess a source id.
"""

    raw = _call_primary_agent(prompt)

    # Parse JSON from response
    analysis = {"raw_response": raw, "timestamp": time.time(),
                "context_date": ctx.get("session_date", ""),
                "regime": ctx.get("regime", "unknown"),
                "fresh_market": dict(ctx.get("fresh_market") or {})}
    parsed = _extract_agent_json_payload(raw)
    if parsed:
        analysis.update(parsed)
    else:
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
        base_url=_agent_osint_base(),
    )
    for item in list(intel.get("story_items") or [])[:3]:
        results.append(
            f"  [story] [{item.get('source_name', ''):15s}] "
            f"{(item.get('published_at') or '')[:16]}  {_clip_text(item.get('title'), 150)}"
            f"{'  ' + str(item.get('url') or '') if item.get('url') else ''}"
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


def _expanded_political_query(question: str) -> str:
    focus = _question_focus_query(question)
    terms = [focus] if focus else []
    text = str(question or "").lower()
    political_expansions = [
        "government",
        "cabinet",
        "parliament",
        "coalition",
        "prime minister",
        "minister",
        "resignation",
        "protest",
        "scandal",
        "corruption",
        "arrest",
        "probe",
        "policy decision",
    ]
    if "scandal" in text or "corruption" in text:
        political_expansions.extend(["anti corruption", "investigation", "ciaa"])
    if "government" in text or "cabinet" in text:
        political_expansions.extend(["ordinance", "decision", "ministry"])
    return " ".join(part for part in [*terms, *political_expansions] if part).strip()


def _question_is_news_request(question: str) -> bool:
    text = str(question or "").lower()
    markers = (
        "breaking news",
        "latest news",
        "recent news",
        "top news",
        "exact news",
        "political news",
        "politics news",
        "headlines",
        "headline",
        "cite",
        "cited",
        "citation",
        "source",
        "sources",
        "what is the news",
        "what's the news",
        "recent political development",
        "political development",
    )
    return any(marker in text for marker in markers) or _question_is_political_news_request(question)


def _question_is_political_news_request(question: str) -> bool:
    text = str(question or "").lower()
    markers = (
        "political development",
        "political news",
        "politics",
        "government",
        "cabinet",
        "parliament",
        "prime minister",
        "coalition",
        "resignation",
        "protest",
    )
    return any(marker in text for marker in markers)


def _question_requires_ranked_news_answer(question: str) -> bool:
    text = str(question or "").lower()
    if any(marker in text for marker in ("top news", "exact news", "top 5", "top five")):
        return True
    return bool(re.search(r"\b\d+\b\s+(?:top\s+)?news\b", text))


def _requested_news_count(question: str, default: int = 5) -> int:
    text = str(question or "").lower()
    match = re.search(r"\btop\s+(\d+)\b", text) or re.search(r"\bgive me\s+(\d+)\b", text)
    if match:
        try:
            return max(1, min(8, int(match.group(1))))
        except ValueError:
            pass
    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    for word, value in word_map.items():
        if f"top {word}" in text or f"give me {word}" in text:
            return value
    return default


def _news_relevance_score(item: dict, question: str) -> float:
    text = _story_blob(item)
    title = _story_title(item).lower()
    category = str(item.get("category") or item.get("story_type") or "").lower()
    question_text = str(question or "").lower()
    score = 0.0

    if _story_title(item):
        score += 1.0
    if _story_url(item):
        score += 0.2

    political_tokens = {
        "politic", "government", "cabinet", "parliament", "prime minister",
        "coalition", "ministry", "resignation", "protest", "election",
        "minister", "ordinance", "opposition", "ruling party", "manifesto",
        "policy decision", "corruption", "scandal", "probe", "investigation",
        "arrest", "ciaa",
    }
    hard_political_tokens = {
        "government", "cabinet", "parliament", "prime minister", "coalition",
        "minister", "resignation", "protest", "corruption", "scandal",
        "probe", "investigation", "arrest", "ciaa",
    }
    market_tokens = {
        "nrb", "liquidity", "interest rate", "policy", "bank", "tax",
        "budget", "regulation", "hydro", "nea", "power", "inflation",
    }
    market_recap_tokens = {
        "nepse", "market", "share", "stock", "stocks", "index", "points", "trading",
    }

    if _question_is_political_news_request(question_text):
        if any(token in text for token in political_tokens):
            score += 2.5
        else:
            score -= 0.6
        if any(token in title for token in hard_political_tokens):
            score += 2.2
        if any(token in category for token in ("politic", "government", "crime", "policy")):
            score += 1.4
        # Penalize market recap headlines that mention politics only indirectly.
        if any(token in title for token in market_recap_tokens) and not any(token in title for token in hard_political_tokens):
            score -= 2.8
        if "nepse" in title and "government" not in title and "cabinet" not in title and "parliament" not in title:
            score -= 1.6

    if "nepse" in question_text or "market" in question_text or "open" in question_text:
        if any(token in text for token in market_tokens):
            score += 1.4

    focus_query = _question_focus_query(question_text).lower()
    for token in [part for part in focus_query.split() if len(part) >= 4][:6]:
        if token in text:
            score += 0.4

    bias = abs(_osint_keyword_bias(text))
    score += bias * 8.0

    published_dt = _story_published_dt(item)
    if published_dt is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600.0)
        if age_hours <= 12:
            score += 1.2
        elif age_hours <= 24:
            score += 0.8
        elif age_hours <= 48:
            score += 0.4

    return score


def _rank_news_items_for_question(items: list[dict], question: str) -> list[dict]:
    return sorted(
        [dict(item or {}) for item in items if isinstance(item, dict) and (_story_title(item) or _story_url(item))],
        key=lambda item: (_news_relevance_score(item, question), _story_time(item)),
        reverse=True,
    )


def _story_has_direct_political_hit(item: dict) -> bool:
    text = _story_blob(item)
    return any(
        token in text
        for token in (
            "politic", "government", "cabinet", "parliament", "prime minister",
            "coalition", "resignation", "protest", "election",
        )
    )


def _latest_news_context_for_question(question: str) -> dict:
    if not _question_is_news_request(question):
        return {"items": [], "context_text": ""}

    base_url = _agent_osint_base()
    items: list[dict] = []
    seen_refs: set[tuple[str, str]] = set()
    query = _expanded_political_query(question) if _question_is_political_news_request(question) else _question_focus_query(question)
    date_window = _parse_news_date_window(question)

    if _date_window_is_explicit(date_window):
        start = date_window["start"].date().isoformat()
        end = date_window["end"].date().isoformat()
        history = consolidated_stories_history(
            start_date=start,
            end_date=end,
            limit=100,
            offset=0,
            category="economic" if ("nepse" in str(question).lower() or "market" in str(question).lower()) else None,
            base_url=base_url,
            timeout=8,
        )
        for item in list(history.get("items") or [])[:100]:
            ref = _story_ref(item)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            items.append(dict(item or {}))

    if query and len(query) >= 3:
        unified = unified_search(query, limit=8, base_url=base_url, timeout=8)
        categories = dict(unified.get("categories") or {})
        for item in list(dict(categories.get("stories") or {}).get("items") or [])[:8]:
            if not _story_in_date_window(item, date_window):
                continue
            ref = _story_ref(item)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            items.append(dict(item or {}))

        semantic = semantic_story_search(
            query,
            hours=int(date_window.get("hours") or 720),
            top_k=10,
            min_similarity=0.35,
            base_url=base_url,
            timeout=8,
        )
        for raw in list(semantic.get("results") or [])[:10]:
            item = _normalize_semantic_story_item(dict(raw or {}))
            if not _story_in_date_window(item, date_window):
                continue
            ref = _story_ref(item)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            items.append(item)

    if len(items) < 4:
        for item in consolidated_stories(limit=8, base_url=base_url, timeout=8):
            if not _story_in_date_window(item, date_window):
                continue
            ref = _story_ref(item)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            items.append(dict(item or {}))
            if len(items) >= 6:
                break

    if not items:
        return {"items": [], "context_text": ""}

    ranked_items = _rank_news_items_for_question(items, question)
    lines = ["LATEST NEPALOSINT NEWS RESULTS:"]
    if date_window.get("label"):
        lines.append(f"  Date window: {date_window['label']}")
    for item in ranked_items[:6]:
        lines.append(f"  [story] {_format_citable_story(item)}")
    return {
        "items": ranked_items[:6],
        "direct_political_hit": any(_story_has_direct_political_hit(item) for item in ranked_items[:6]),
        "date_window": {"label": date_window.get("label") or "", "hours": int(date_window.get("hours") or 720)},
        "context_text": "\n".join(lines) + "\n",
    }


def _event_market_context(question: str) -> dict:
    query = _question_focus_query(question)
    unified = unified_search(query or question, limit=8, base_url=_agent_osint_base(), timeout=8)
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


def _response_lacks_news_citations(response: str) -> bool:
    text = " ".join(str(response or "").split()).lower()
    if not text:
        return True
    if "http://" in text or "https://" in text:
        return False
    weak_markers = (
        "no breaking news cited",
        "no breaking news",
        "no cited news",
        "current data",
        "purely technical",
    )
    return any(marker in text for marker in weak_markers)


def _build_news_digest_answer(question: str, news_ctx: dict) -> str:
    items = list(news_ctx.get("items") or [])
    if not items:
        return "I do not have any current NepalOSINT story hits I can cite from the retrieved feed right now, so I will not invent headlines."

    cited = []
    for item in items[:4]:
        cited.append(_format_citable_story(item))

    intro = "Latest NepalOSINT headlines:"
    if "breaking" in str(question or "").lower():
        intro = "Latest NepalOSINT breaking headlines:"
    return f"{intro} " + "; ".join(cited) + "."


def _story_market_impact_summary(item: dict) -> str:
    text = _story_blob(item)
    bias = _osint_keyword_bias(text)
    if any(token in text for token in ("protest", "violence", "arrest", "coalition", "resignation", "politic")):
        driver = "political risk usually moves broad NEPSE sentiment quickly"
    elif any(token in text for token in ("nrb", "liquidity", "interest rate", "bank", "lending", "deposit")):
        driver = "banks are index-heavy and liquidity headlines move the open"
    elif any(token in text for token in ("cabinet", "government", "policy", "budget", "tax", "regulation")):
        driver = "policy headlines reprice sector sentiment at the open"
    elif any(token in text for token in ("hydro", "nea", "power")):
        driver = "hydro names are highly policy-sensitive on NEPSE"
    else:
        driver = "it can shift opening risk appetite if traders treat it as macro-relevant"

    if bias >= 0.04:
        return f"likely positive for NEPSE because {driver}"
    if bias <= -0.04:
        return f"likely negative for NEPSE because {driver}"
    return f"market-relevant but direction depends on follow-through because {driver}"


def _build_ranked_news_answer(question: str, news_ctx: dict, nepal_clock: dict) -> str:
    items = list(news_ctx.get("items") or [])
    if not items:
        return "I do not have any current NepalOSINT story hits I can cite from the retrieved feed right now, so I will not invent headlines."

    count = min(len(items), _requested_news_count(question, default=5 if _question_requires_ranked_news_answer(question) else 3))
    selected = items[:count]
    lines: list[str] = []
    date_label = str(dict(news_ctx.get("date_window") or {}).get("label") or "").strip()

    if _question_is_political_news_request(question):
        if news_ctx.get("direct_political_hit"):
            lead = selected[0]
            lines.append(
                f"Most relevant political development: {_clip_text(_story_title(lead), 150)} [1]. "
                f"NEPSE read-through: {_story_market_impact_summary(lead)}."
            )
        else:
            lines.append(
                "I do not see a direct political headline in the retrieved NepalOSINT feed right now. "
                f"The market-moving items to watch before the next {nepal_clock.get('market_phase', 'NEPSE')} session are:"
            )
    else:
        lines.append(f"Top {count} NepalOSINT items most likely to matter for the next NEPSE open:")
    if date_label:
        lines.append(f"Date window: {date_label}.")

    for idx, item in enumerate(selected, start=1):
        title = _clip_text(_story_title(item), 150)
        lines.append(f"{idx}. {title} [{idx}] — {_story_market_impact_summary(item)}.")
    lines.append("Sources:")
    for idx, item in enumerate(selected, start=1):
        lines.append(f"- {_format_story_source_ref(item, idx)}")
    return "\n".join(lines)


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
    news_request = _question_is_news_request(question)

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
    news_ctx = _latest_news_context_for_question(question)
    latest_news_ctx = str(news_ctx.get("context_text") or "")
    event_ctx = _event_market_context(question) if directional_market_call else {}
    event_osint_ctx = str(event_ctx.get("context_text") or "")

    deterministic_news_response = None
    if news_request and (_question_requires_ranked_news_answer(question) or _question_is_political_news_request(question)):
        deterministic_news_response = _build_ranked_news_answer(question, news_ctx, nepal_clock)

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
{latest_news_ctx}
{event_osint_ctx}
{_analyst_pattern_taxonomy_text()}
{macro_ctx}
{history_text}
User question: {question}

IMPORTANT:
- Keep your answer concise.
- For simple status questions, reply in 2-4 sentences max.
- For stock, sector, market-impact, or cited-news analysis questions, you may use compact labeled lines:
  Summary:
  Key facts:
  Translation notes:
  Why it matters:
  Mechanism:
  Pattern match:
  Likely relevance:
  Risks or counterpoints:
  Confidence:
  Missing information:
  Sources:
- Be direct and conversational, like a quick terminal chat reply.
- Use specific numbers from the financial data, latest analysis, and vector search when relevant.
- If asked about a stock, reference its valuation, margins, growth, or news — don't give a generic answer.
- Use Nepal time only when talking about timing, session status, market hours, or "today".
- If the market is closed, say it is closed; if it is pre-open, make that explicit.
- Do not confuse accounts. ACTIVE ACCOUNT and ACTIVE HOLDINGS above are the source of truth for whether a stock is already owned.
- If a stock is already held in ACTIVE HOLDINGS, your stance can be BUY, HOLD, or SELL in plain language.
- If a stock is not held, do not recommend HOLD; the stance should resolve to BUY or PASS/AVOID.
- For political, policy, or NEPSE reaction questions, you must take a base-case directional stance: upward pressure, downward pressure, or flat-to-upward/downward bias.
- Do not answer with "it could go either way", "depends entirely", or similar hedging for those event-driven market questions.
- If the user asks for breaking news, headlines, citations, or sources, answer from the NepalOSINT stories in the prompt and cite each item inline with source, timestamp, and URL.
- If the user explicitly asks for cited news, you may list 3-6 cited headlines in one compact paragraph instead of the normal 2-4 sentence cap.
- Separate observed facts from inference. If you make a factual news claim, cite the source inline. If evidence is missing or conflicting, say so explicitly.
- If the prompt evidence is Nepali or mixed-language, translate it into clean English and note any financially material nuance when relevant.
- If the news fits a repeated pattern from ANALYST PATTERN TAXONOMY, you may name that pattern briefly.
- If you say something is positive or negative, explain why it matters for the stock, sector, or market.
- If OSINT evidence is thin, still give the most likely direction and say what would invalidate that call.
- When source ids are present, each bullet under Key facts should cite the supporting id instead of relying on an uncited paragraph.
- If evidence is missing, say it is missing rather than inventing facts."""

    if deterministic_news_response is not None:
        response = deterministic_news_response
    else:
        response = _call_primary_agent(prompt, max_tokens=DEFAULT_AGENT_CHAT_MAX_TOKENS)
        if news_request and not directional_market_call and _response_lacks_news_citations(response):
            response = _build_news_digest_answer(question, news_ctx)
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
