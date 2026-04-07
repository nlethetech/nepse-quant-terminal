"""Deterministic stock financial report builder for the TUI lookup screen."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from backend.quant_pro.database import get_db_path
from backend.quant_pro.paths import get_project_root

PROJECT_ROOT = get_project_root(__file__)
QUARTERLY_REPORTS_DIR = PROJECT_ROOT / "data" / "quarterly_reports"
FINANCIAL_REPORTS_DIR = PROJECT_ROOT / "data" / "financial_reports"

COMPANY_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "RURU": {
        "company_name": "Ru Ru Jalbidhyut Pariyojana Limited",
        "board": [
            {"name": "Ashish Subedi", "role": "Chairman"},
            {"name": "Chandra Bahadur Pun", "role": "Director"},
            {"name": "Govinda Chalise", "role": "Independent Director"},
            {"name": "Sarita Shakya Pradhan", "role": "Director"},
            {"name": "Sagar Pathak", "role": "Director"},
            {"name": "Anjan Raj Poudyal", "role": "Director"},
            {"name": "Abinash Silwal", "role": "Director"},
        ],
        "officers": [
            {"name": "Kishor Prasad Ghimire", "role": "Company Secretary"},
        ],
    },
    "RSML": {
        "company_name": "Reliance Spinning Mills Limited",
        "board": [
            {"name": "Pawan Kumar Golyan", "role": "Chairman"},
            {"name": "Shashi Kanta Agrawal", "role": "Director"},
            {"name": "Akshay Golyan", "role": "Managing Director"},
        ],
        "officers": [],
    },
}

INTELLIGENCE_OVERRIDES: dict[str, dict[str, Any]] = {
    "RURU": {
        "headline": "Upper Hugdi Khola is a 5.0 MW run-of-river hydro asset in Gulmi with PPA-linked cash flow routed through Birbash substation.",
        "project": {
            "Project": "Upper Hugdi Khola Hydropower Project",
            "Type": "Run-of-river hydro",
            "Location": "Gulmi, Nepal",
            "COD (BS)": "2071-12-09",
            "Installed Capacity": "5.0 MW",
        },
        "resource_input": {
            "River": "Hugdi Khola",
            "Catchment": "120 sqkm",
            "Design Q": "3.75 m3/s",
            "Gross Head": "184.5 m",
            "Net Head": "160.0-171.17 m",
            "Design Energy": "28.26 GWh",
            "Wet / Dry": "22.77 / 5.49 GWh",
        },
        "power_flow": {
            "Evacuation": "Birbash substation, Gulmi",
            "Offtaker": "Nepal Electricity Authority",
            "Revenue Basis": "PPA",
            "Buyer Mix": "Single buyer / NEA",
        },
        "q2_provisional": {
            "Period": "FY2082/83 Q2 provisional",
            "Revenue": 91_138_828,
            "Gross Profit": 68_095_540,
            "PAT": 52_731_282,
            "Cash": 191_127_998,
            "Receivables": 102_233_581,
            "Inventory": 9_182_531,
            "Assets": 955_968_226,
            "Equity": 825_042_845,
            "Liabilities": 170_925_381,
        },
        "risk_flags": [
            "Hydrology and seasonal flow risk shape dry-season output.",
            "Single-buyer exposure sits with NEA under the current PPA chain.",
            "Billing and receivables need monitoring versus cash conversion.",
            "Grid evacuation depends on Birbash substation continuity.",
            "Tariff / PPA terms, royalties, and regulatory changes can move earnings.",
            "O&M availability remains critical for a small run-of-river asset.",
        ],
        "recent_watch": [
            "Jan 1, 2026: Nepal renewed 654 MW of Indian power imports for the dry season, with officials noting run-of-river generation drops sharply in winter. That keeps hydrology and seasonal dispatch risk live for small hydro producers.",
            "Mar 30, 2026: The government set a 180-day deadline to clear stalled PPAs and power-sector permits while shaping an export strategy. Policy, permit, and market-access rules are still moving rather than fully settled.",
            "Jun 16, 2025: RURU disclosed a partial divestment in Karnali Water Power, indicating active capital allocation and liquidity management outside the core asset.",
        ],
    },
    "RSML": {
        "headline": "Reliance Spinning Mills is an export-oriented yarn platform with twin Sunsari plants, imported fibre inputs, and India/Turkey-led market exposure.",
        "project": {
            "Company": "Reliance Spinning Mills Limited",
            "Sector": "Textiles / yarn manufacturing",
            "Type": "Export-oriented spinning mill",
            "Registered Office": "Kamaladi, Kathmandu, Nepal",
            "Plants": "Khanar, Sunsari; Duhabi, Sunsari",
            "Installed Capacity": "50,163 MT / year",
        },
        "resource_input": {
            "Core Inputs": "Polyester staple fibre, viscose, acrylic, cotton inputs, dyes and chemicals",
            "Source Countries": "India, China, Hong Kong, Indonesia, Malaysia, Thailand, Korea, Austria",
            "Procurement": "Letters of credit",
            "Working Capital": "Import-heavy raw material cycle",
            "Product Stack": "Polyester, viscose, acrylic, cotton, blended yarn, sewing thread, vortex, open-end yarn",
        },
        "power_flow": {
            "Primary Export Markets": "India, Turkey",
            "Other Markets": "UK, Vietnam, Canada, Kenya, Portugal, Spain, Brazil, Cambodia, Bangladesh",
            "Domestic Market": "Nepal",
            "Operating Units": "2 production units",
            "Energy Profile": "9.2 MW captive solar plus grid dependency",
        },
        "q2_provisional": {
            "Period": "FY2082/83 Q2 unaudited",
            "Revenue": 4_790_291_163.87,
            "Gross Profit": 599_390_082.07,
            "Operating Profit": 329_848_345.70,
            "PAT": 137_435_414.45,
            "Inventory": 2_094_770_425.25,
            "Receivables": 1_315_268_370.91,
            "Cash": 56_795_782.10,
            "Assets": 14_568_829_021.23,
            "Equity": 9_320_270_889.39,
            "Liabilities": 5_248_558_131.84,
        },
        "risk_flags": [
            "Imported raw materials leave margins exposed to fibre and shipping cost volatility.",
            "Foreign-exchange risk sits across both procurement and export settlement cycles.",
            "India and Turkey concentration creates channel and trade-access dependence.",
            "Working-capital intensity remains high with large inventory and receivables balances.",
            "Power availability and energy-cost changes can pressure conversion margins.",
            "Finance cost burden remains material against operating profit.",
            "Trade policy and market-access changes can alter export economics quickly.",
        ],
        "recent_watch": [
            "Dec 8, 2025: CARE Ratings Nepal kept RSML under credit watch with developing implications, so funding and balance-sheet execution still deserve monitoring after the public issue process.",
            "Dec 29, 2025: ICRA said Indian cotton spinners could see FY26 revenue fall 4-6% with margin contraction of 50-100 bps as tariff effects pass through the value chain, which matters because RSML sells into India-facing yarn markets.",
            "Apr 2, 2026: China Daily reported crude-linked polyester filament and staple-fibre costs rising while downstream textile demand stayed soft, which is a direct margin watch for yarn producers exposed to synthetic-fibre inputs.",
        ],
    }
}


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _quarter_number(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return int(digits)
    return 0


def _fiscal_sort_key(raw: str) -> tuple[int, int]:
    digits = [int(chunk) for chunk in "".join(ch if ch.isdigit() else " " for ch in str(raw)).split()]
    if not digits:
        return (0, 0)
    if len(digits) == 1:
        return (digits[0], 0)
    return (digits[0], digits[1])


def _format_fiscal_year(raw: str) -> str:
    digits = [chunk for chunk in "".join(ch if ch.isdigit() else " " for ch in str(raw)).split()]
    if len(digits) >= 2:
        left = digits[0][-4:] if len(digits[0]) >= 4 else digits[0]
        right = digits[1][-2:] if len(digits[1]) >= 2 else digits[1]
        return f"{left}/{right}"
    return str(raw) if raw else "—"


def _format_period(fiscal_year: str, quarter: Any) -> str:
    q_num = _quarter_number(quarter)
    q_label = f"Q{q_num}" if q_num else str(quarter or "—")
    return f"{_format_fiscal_year(fiscal_year)} {q_label}".strip()


def _compact_money(value: float | None) -> str:
    if value is None:
        return "—"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000_000:
        return f"{sign}NPR {abs_value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{sign}NPR {abs_value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{sign}NPR {abs_value / 1_000:.1f}K"
    return f"{sign}NPR {abs_value:.0f}"


def _format_number(value: float | None, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _format_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _latest_fundamentals_row(symbol: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fundamentals' LIMIT 1"
        ).fetchone()
        if exists is None:
            return {}
        row = conn.execute(
            """
            SELECT *
            FROM fundamentals
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return dict(row) if row else {}


def _company_profile(symbol: str) -> dict[str, Any]:
    profile = COMPANY_PROFILE_OVERRIDES.get(symbol.upper(), {})
    if not isinstance(profile, dict):
        return {}
    board = [dict(item) for item in (profile.get("board") or []) if isinstance(item, dict)]
    officers = [dict(item) for item in (profile.get("officers") or []) if isinstance(item, dict)]
    return {
        "company_name": str(profile.get("company_name") or "").strip(),
        "board": board,
        "officers": officers,
    }


def _build_intelligence_brief(symbol: str) -> dict[str, Any]:
    payload = INTELLIGENCE_OVERRIDES.get(symbol.upper(), {})
    if not isinstance(payload, dict) or not payload:
        return {}

    q2 = payload.get("q2_provisional") or {}
    revenue = _safe_float(q2.get("Revenue"))
    gross_profit = _safe_float(q2.get("Gross Profit"))
    pat = _safe_float(q2.get("PAT"))
    cash = _safe_float(q2.get("Cash"))
    receivables = _safe_float(q2.get("Receivables"))
    assets = _safe_float(q2.get("Assets"))
    equity = _safe_float(q2.get("Equity"))
    liabilities = _safe_float(q2.get("Liabilities"))

    commercial_rows: list[tuple[str, str]] = []
    for label, raw in [
        ("Period", q2.get("Period")),
        ("Revenue", _compact_money(revenue)),
        ("Gross Profit", _compact_money(gross_profit)),
        ("PAT", _compact_money(pat)),
        ("Cash", _compact_money(cash)),
        ("Receivables", _compact_money(receivables)),
        ("Assets", _compact_money(assets)),
        ("Equity", _compact_money(equity)),
    ]:
        if raw not in (None, "", "—"):
            commercial_rows.append((str(label), str(raw)))

    if revenue not in (None, 0):
        gross_margin = _safe_div(gross_profit, revenue)
        pat_margin = _safe_div(pat, revenue)
        receivable_load = _safe_div(receivables, revenue)
        if gross_margin is not None:
            commercial_rows.append(("Gross Margin", _format_pct(gross_margin * 100)))
        if pat_margin is not None:
            commercial_rows.append(("PAT Margin", _format_pct(pat_margin * 100)))
        if receivable_load is not None:
            commercial_rows.append(("Receivable Load", f"{receivable_load:.2f}x rev"))

    if equity not in (None, 0):
        leverage = _safe_div(liabilities, equity)
        if leverage is not None:
            commercial_rows.append(("Liability / Equity", f"{leverage:.2f}x"))

    sections: list[dict[str, Any]] = []
    for title, source in [
        ("Asset", payload.get("project") or {}),
        ("Water / Generation", payload.get("resource_input") or {}),
        ("Revenue Chain", payload.get("power_flow") or {}),
    ]:
        rows = [(str(k), str(v)) for k, v in source.items() if str(v).strip()]
        if rows:
            sections.append({"title": title, "rows": rows})

    if commercial_rows:
        sections.append({"title": "Commercial Pulse", "rows": commercial_rows})

    risks = [str(item).strip() for item in (payload.get("risk_flags") or []) if str(item).strip()]
    recent_watch = [str(item).strip() for item in (payload.get("recent_watch") or []) if str(item).strip()]
    if recent_watch:
        sections.append({"title": "Recent Watch", "bullets": recent_watch})
    if risks:
        sections.append({"title": "Risk Map", "bullets": risks})

    return {
        "headline": str(payload.get("headline") or "").strip(),
        "sections": sections,
    }


def _load_cached_report_rows(symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_file = QUARTERLY_REPORTS_DIR / f"{symbol}.json"
    if not cache_file.exists():
        return _load_new_cache_report_rows(symbol)

    try:
        cached = json.loads(cache_file.read_text())
    except Exception:
        return _load_new_cache_report_rows(symbol)

    rows: list[dict[str, Any]] = []
    latest_detail: dict[str, Any] = {}
    for report in cached.get("reports", []):
        financials = report.get("financials") or {}
        if not financials or "error" in financials:
            continue
        income = financials.get("income_statement") or {}
        balance = financials.get("balance_sheet") or {}
        per_share = financials.get("per_share") or {}
        ratios = financials.get("ratios") or {}
        row = {
            "source": "cache",
            "fiscal_year": financials.get("fiscal_year", ""),
            "quarter": _quarter_number(financials.get("quarter")),
            "period": _format_period(financials.get("fiscal_year", ""), financials.get("quarter")),
            "announcement_date": report.get("date", ""),
            "report_date": report.get("date", ""),
            "revenue": _safe_float(income.get("total_revenue")),
            "net_profit": _safe_float(income.get("net_profit")),
            "operating_profit": _safe_float(income.get("operating_profit")),
            "interest_income": _safe_float(income.get("interest_income")),
            "interest_expense": _safe_float(income.get("interest_expense")),
            "eps": _safe_float(per_share.get("eps")),
            "book_value": _safe_float(per_share.get("book_value")),
            "total_assets": _safe_float(balance.get("total_assets")),
            "total_liabilities": _safe_float(balance.get("total_liabilities")),
            "shareholders_equity": _safe_float(balance.get("shareholders_equity")),
            "share_capital": _safe_float(balance.get("share_capital")),
            "retained_earnings": _safe_float(balance.get("retained_earnings")),
            "total_deposits": _safe_float(balance.get("total_deposits")),
            "total_loans": _safe_float(balance.get("total_loans")),
            "npl_pct": _safe_float(ratios.get("npl_pct")),
            "capital_adequacy_pct": _safe_float(ratios.get("capital_adequacy_pct")),
            "cost_income_ratio": _safe_float(ratios.get("cost_income_ratio")),
            "sector": str(financials.get("sector", "")).strip().lower(),
            "notes": str(financials.get("notes", "")).strip(),
        }
        rows.append(row)
        if not latest_detail:
            latest_detail = row

    rows.sort(key=lambda row: (_fiscal_sort_key(row["fiscal_year"]), row["quarter"]), reverse=True)
    if rows and not latest_detail:
        latest_detail = rows[0]
    if rows:
        return rows, latest_detail
    return _load_new_cache_report_rows(symbol)


def _is_trusted_unified_report(report: dict[str, Any]) -> bool:
    extracted = report.get("extracted_json") or {}
    if not extracted or "error" in extracted:
        return False
    status = str(report.get("extraction_status") or "")
    quality = extracted.get("quality") or {}
    confidence = float(quality.get("confidence") or 0)
    review_flags = quality.get("review_flags") or []
    if status.startswith("local_"):
        return status == "local_ok" and confidence >= 0.70 and not review_flags
    return status == "ok"


def _load_new_cache_report_rows(symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_file = FINANCIAL_REPORTS_DIR / f"{symbol}.json"
    if not cache_file.exists():
        return [], {}

    try:
        cached = json.loads(cache_file.read_text())
    except Exception:
        return [], {}

    rows: list[dict[str, Any]] = []
    latest_detail: dict[str, Any] = {}
    for report in cached.get("reports", []):
        if report.get("report_type") != "quarterly":
            continue
        if not _is_trusted_unified_report(report):
            continue
        extracted = report.get("extracted_json") or {}
        income = extracted.get("income_statement") or {}
        balance = extracted.get("balance_sheet") or {}
        per_share = extracted.get("per_share") or {}
        ratios = extracted.get("ratios") or {}
        row = {
            "source": "financial_reports_cache",
            "fiscal_year": extracted.get("fiscal_year", ""),
            "quarter": _quarter_number(extracted.get("quarter")),
            "period": _format_period(extracted.get("fiscal_year", ""), extracted.get("quarter")),
            "announcement_date": report.get("announcement_date", ""),
            "report_date": report.get("announcement_date", ""),
            "revenue": _safe_float(income.get("total_revenue")),
            "net_profit": _safe_float(income.get("net_profit")),
            "operating_profit": _safe_float(income.get("operating_profit")),
            "interest_income": _safe_float(income.get("interest_income")),
            "interest_expense": _safe_float(income.get("interest_expense")),
            "eps": _safe_float(per_share.get("eps")),
            "book_value": _safe_float(per_share.get("book_value")),
            "total_assets": _safe_float(balance.get("total_assets")),
            "total_liabilities": _safe_float(balance.get("total_liabilities")),
            "shareholders_equity": _safe_float(balance.get("shareholders_equity")),
            "share_capital": _safe_float(balance.get("share_capital")),
            "retained_earnings": _safe_float(balance.get("retained_earnings")),
            "total_deposits": _safe_float(balance.get("total_deposits")),
            "total_loans": _safe_float(balance.get("total_loans")),
            "npl_pct": _safe_float(ratios.get("npl_pct")),
            "capital_adequacy_pct": _safe_float(ratios.get("capital_adequacy_pct")),
            "cost_income_ratio": _safe_float(ratios.get("cost_income_ratio")),
            "sector": str(extracted.get("sector", "")).strip().lower(),
            "notes": str(extracted.get("notes", "")).strip(),
        }
        rows.append(row)
        if not latest_detail:
            latest_detail = row

    rows.sort(key=lambda row: (_fiscal_sort_key(row["fiscal_year"]), row["quarter"]), reverse=True)
    if rows and not latest_detail:
        latest_detail = rows[0]
    return rows, latest_detail


def _load_db_quarter_rows(symbol: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='quarterly_earnings' LIMIT 1"
        ).fetchone()
        if exists is None:
            return []
        rows = conn.execute(
            """
            SELECT symbol, fiscal_year, quarter, eps, net_profit, revenue, book_value,
                   announcement_date, report_date, source
            FROM quarterly_earnings
            WHERE symbol = ?
            ORDER BY fiscal_year DESC, quarter DESC
            """,
            (symbol,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    result = []
    for row in rows:
        item = dict(row)
        result.append(
            {
                "source": item.get("source") or "quarterly_earnings",
                "fiscal_year": item.get("fiscal_year", ""),
                "quarter": _quarter_number(item.get("quarter")),
                "period": _format_period(item.get("fiscal_year", ""), item.get("quarter")),
                "announcement_date": item.get("announcement_date") or "",
                "report_date": item.get("report_date") or "",
                "revenue": _safe_float(item.get("revenue")),
                "net_profit": _safe_float(item.get("net_profit")),
                "eps": _safe_float(item.get("eps")),
                "book_value": _safe_float(item.get("book_value")),
            }
        )
    result.sort(key=lambda row: (_fiscal_sort_key(row["fiscal_year"]), row["quarter"]), reverse=True)
    return result


def _load_override_quarter_rows(symbol: str) -> list[dict[str, Any]]:
    payload = INTELLIGENCE_OVERRIDES.get(symbol.upper(), {})
    q2 = (payload or {}).get("q2_provisional") or {}
    if not isinstance(q2, dict) or not q2:
        return []
    period = str(q2.get("Period") or "").strip()
    quarter_match = re.search(r"\bQ([1-4])\b", period, flags=re.I)
    quarter = int(quarter_match.group(1)) if quarter_match else 2
    fiscal_year = ""
    fy_match = re.search(r"(\d{4}\s*/\s*\d{2,4})", period)
    if fy_match:
        fiscal_year = fy_match.group(1).replace(" ", "")
    elif period:
        fiscal_year = period.split()[0]
    return [
        {
            "source": "intelligence_override",
            "fiscal_year": fiscal_year,
            "quarter": quarter,
            "period": _format_period(fiscal_year, quarter) if fiscal_year else period or f"Q{quarter}",
            "announcement_date": "",
            "report_date": "",
            "revenue": _safe_float(q2.get("Revenue")),
            "net_profit": _safe_float(q2.get("PAT")),
            "operating_profit": _safe_float(q2.get("Operating Profit")),
            "eps": _safe_float(q2.get("EPS")),
            "book_value": _safe_float(q2.get("Book Value")),
            "total_assets": _safe_float(q2.get("Assets")),
            "total_liabilities": _safe_float(q2.get("Liabilities")),
            "shareholders_equity": _safe_float(q2.get("Equity")),
            "sector": "",
            "notes": str(payload.get("headline") or "").strip(),
        }
    ]


def _merge_quarter_rows(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in primary + secondary:
        key = (str(row.get("fiscal_year", "")), _quarter_number(row.get("quarter")))
        if key in seen:
            continue
        merged.append(row)
        seen.add(key)
    merged.sort(key=lambda row: (_fiscal_sort_key(row["fiscal_year"]), row["quarter"]), reverse=True)
    return merged


def _derive_metrics(latest: dict[str, Any], previous: dict[str, Any] | None, current_price: float | None, fundamentals: dict[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    revenue = _safe_float(latest.get("revenue"))
    net_profit = _safe_float(latest.get("net_profit"))
    operating_profit = _safe_float(latest.get("operating_profit"))
    equity = _safe_float(latest.get("shareholders_equity"))
    liabilities = _safe_float(latest.get("total_liabilities"))
    deposits = _safe_float(latest.get("total_deposits"))
    loans = _safe_float(latest.get("total_loans"))
    eps = _safe_float(latest.get("eps"))
    book_value = _safe_float(latest.get("book_value"))
    q_num = _quarter_number(latest.get("quarter")) or 1

    metrics["profit_margin_pct"] = _safe_div(net_profit, revenue)
    if metrics["profit_margin_pct"] is not None:
        metrics["profit_margin_pct"] *= 100

    metrics["operating_margin_pct"] = _safe_div(operating_profit, revenue)
    if metrics["operating_margin_pct"] is not None:
        metrics["operating_margin_pct"] *= 100

    metrics["debt_to_equity"] = _safe_div(liabilities, equity)
    metrics["loan_to_deposit_pct"] = _safe_div(loans, deposits)
    if metrics["loan_to_deposit_pct"] is not None:
        metrics["loan_to_deposit_pct"] *= 100

    annualized_eps = None
    if eps is not None and q_num > 0:
        annualized_eps = eps * (4 / q_num)
    metrics["eps_annualized"] = annualized_eps

    metrics["roe_pct"] = _safe_div(annualized_eps, book_value)
    if metrics["roe_pct"] is not None:
        metrics["roe_pct"] *= 100
    elif _safe_float(fundamentals.get("roe")) is not None:
        metrics["roe_pct"] = _safe_float(fundamentals.get("roe"))

    if current_price and annualized_eps and annualized_eps > 0:
        metrics["pe_ratio"] = current_price / annualized_eps
    else:
        metrics["pe_ratio"] = _safe_float(fundamentals.get("pe_ratio"))

    if current_price and book_value and book_value > 0:
        metrics["pb_ratio"] = current_price / book_value
    else:
        metrics["pb_ratio"] = _safe_float(fundamentals.get("pb_ratio"))

    prev_revenue = _safe_float(previous.get("revenue")) if previous else None
    prev_profit = _safe_float(previous.get("net_profit")) if previous else None
    prev_eps = _safe_float(previous.get("eps")) if previous else None
    prev_book = _safe_float(previous.get("book_value")) if previous else None
    prev_npl = _safe_float(previous.get("npl_pct")) if previous else None

    metrics["revenue_growth_qoq_pct"] = None
    if revenue is not None and prev_revenue not in (None, 0):
        metrics["revenue_growth_qoq_pct"] = (revenue - prev_revenue) / prev_revenue * 100

    metrics["profit_growth_qoq_pct"] = None
    if net_profit is not None and prev_profit not in (None, 0):
        metrics["profit_growth_qoq_pct"] = (net_profit - prev_profit) / prev_profit * 100

    metrics["eps_growth_qoq_pct"] = None
    if eps is not None and prev_eps not in (None, 0):
        metrics["eps_growth_qoq_pct"] = (eps - prev_eps) / prev_eps * 100

    metrics["book_value_growth_qoq_pct"] = None
    if book_value is not None and prev_book not in (None, 0):
        metrics["book_value_growth_qoq_pct"] = (book_value - prev_book) / prev_book * 100

    latest_npl = _safe_float(latest.get("npl_pct"))
    if latest_npl is not None and prev_npl is not None:
        metrics["npl_change_pct_pts"] = latest_npl - prev_npl
    else:
        metrics["npl_change_pct_pts"] = None

    return metrics


def _score_report(latest: dict[str, Any], metrics: dict[str, float | None], fundamentals: dict[str, Any]) -> tuple[int, str, list[str], list[str], list[str]]:
    score = 50
    positives: list[str] = []
    risks: list[str] = []
    monitors: list[str] = []

    net_profit = _safe_float(latest.get("net_profit"))
    revenue = _safe_float(latest.get("revenue"))
    retained_earnings = _safe_float(latest.get("retained_earnings"))
    pe_ratio = metrics.get("pe_ratio")
    pb_ratio = metrics.get("pb_ratio")
    roe_pct = metrics.get("roe_pct")
    dte = metrics.get("debt_to_equity")
    profit_margin = metrics.get("profit_margin_pct")
    rev_qoq = metrics.get("revenue_growth_qoq_pct")
    profit_qoq = metrics.get("profit_growth_qoq_pct")
    eps_qoq = metrics.get("eps_growth_qoq_pct")
    npl_pct = _safe_float(latest.get("npl_pct"))
    car_pct = _safe_float(latest.get("capital_adequacy_pct"))
    ld_ratio = metrics.get("loan_to_deposit_pct")
    sector = str(latest.get("sector") or fundamentals.get("sector") or "").lower()

    if net_profit is not None and net_profit > 0:
        score += 8
        positives.append(f"Latest quarter remained profitable at {_compact_money(net_profit)}.")
    elif net_profit is not None and net_profit < 0:
        score -= 10
        risks.append(f"Latest quarter was loss-making at {_compact_money(net_profit)}.")

    if revenue is not None and revenue > 0:
        positives.append(f"Revenue printed at {_compact_money(revenue)}.")

    if profit_margin is not None and profit_margin >= 20:
        score += 6
        positives.append(f"Profit margin is healthy at {_format_pct(profit_margin)}.")
    elif profit_margin is not None and profit_margin < 8:
        score -= 5
        risks.append(f"Profit margin is thin at {_format_pct(profit_margin)}.")

    if rev_qoq is not None and rev_qoq >= 10:
        score += 5
        positives.append(f"Revenue grew {_format_pct(rev_qoq)} QoQ.")
    elif rev_qoq is not None and rev_qoq <= -10:
        score -= 7
        risks.append(f"Revenue fell {_format_pct(rev_qoq)} QoQ.")

    if profit_qoq is not None and profit_qoq >= 12:
        score += 7
        positives.append(f"Net profit grew {_format_pct(profit_qoq)} QoQ.")
    elif profit_qoq is not None and profit_qoq <= -15:
        score -= 9
        risks.append(f"Net profit dropped {_format_pct(profit_qoq)} QoQ.")

    if eps_qoq is not None and eps_qoq >= 10:
        score += 4
        positives.append(f"EPS improved {_format_pct(eps_qoq)} versus the prior quarter.")
    elif eps_qoq is not None and eps_qoq <= -10:
        score -= 4
        risks.append(f"EPS contracted {_format_pct(eps_qoq)} versus the prior quarter.")

    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio <= 12:
            score += 7
            positives.append(f"Valuation is not stretched at P/E {_format_number(pe_ratio, 1)}.")
        elif pe_ratio <= 20:
            score += 2
            monitors.append(f"P/E {_format_number(pe_ratio, 1)} is acceptable but not obviously cheap.")
        elif pe_ratio > 30:
            score -= 7
            risks.append(f"Valuation looks expensive at P/E {_format_number(pe_ratio, 1)}.")

    if pb_ratio is not None and pb_ratio > 0:
        if pb_ratio <= 1.8:
            score += 4
            positives.append(f"P/BV {_format_number(pb_ratio, 2)}x leaves room for rerating.")
        elif pb_ratio > 4.0:
            score -= 6
            risks.append(f"P/BV {_format_number(pb_ratio, 2)}x is rich for current fundamentals.")

    if roe_pct is not None:
        if roe_pct >= 15:
            score += 6
            positives.append(f"Annualized ROE screens strong at {_format_pct(roe_pct)}.")
        elif roe_pct < 7:
            score -= 5
            risks.append(f"Annualized ROE is weak at {_format_pct(roe_pct)}.")

    if dte is not None:
        if sector not in {"banking", "microfinance"} and dte > 3:
            score -= 6
            risks.append(f"Leverage is elevated at debt/equity {_format_number(dte, 2)}x.")
        elif sector not in {"banking", "microfinance"} and dte < 1.2:
            score += 3
            positives.append(f"Balance sheet leverage is manageable at debt/equity {_format_number(dte, 2)}x.")

    if retained_earnings is not None and retained_earnings < 0:
        score -= 4
        risks.append(f"Retained earnings remain negative at {_compact_money(retained_earnings)}.")

    if sector in {"banking", "microfinance"}:
        if npl_pct is not None:
            if npl_pct <= 2:
                score += 5
                positives.append(f"Asset quality is solid with NPL {_format_pct(npl_pct)}.")
            elif npl_pct >= 5:
                score -= 8
                risks.append(f"NPL {_format_pct(npl_pct)} is elevated.")
        if car_pct is not None:
            if car_pct >= 12:
                score += 4
                positives.append(f"Capital adequacy is comfortable at {_format_pct(car_pct)}.")
            elif car_pct < 11:
                score -= 5
                risks.append(f"Capital adequacy at {_format_pct(car_pct)} needs monitoring.")
        if ld_ratio is not None and (ld_ratio < 70 or ld_ratio > 92):
            monitors.append(f"Loan/deposit ratio {_format_pct(ld_ratio)} sits away from the comfort zone.")

    if not monitors and not positives and not risks:
        monitors.append("Financial history is still sparse; scrape more quarters for a stronger view.")

    score = max(0, min(100, int(round(score))))
    if score >= 70:
        signal = "ACCUMULATE"
    elif score >= 52:
        signal = "WATCH"
    else:
        signal = "CAUTION"
    return score, signal, positives[:4], risks[:4], monitors[:4]


def _build_summary(symbol: str, signal: str, score: int, latest: dict[str, Any], metrics: dict[str, float | None], current_price: float | None) -> str:
    fragments = [f"{symbol} screens {signal.lower()} on the current cached financials (score {score}/100)."]

    revenue = _safe_float(latest.get("revenue"))
    net_profit = _safe_float(latest.get("net_profit"))
    if revenue is not None or net_profit is not None:
        fragments.append(
            f"Latest quarter {_format_period(latest.get('fiscal_year', ''), latest.get('quarter'))} shows revenue {_compact_money(revenue)} and net profit {_compact_money(net_profit)}."
        )

    pe_ratio = metrics.get("pe_ratio")
    pb_ratio = metrics.get("pb_ratio")
    valuation_bits = []
    if current_price:
        valuation_bits.append(f"price {current_price:.1f}")
    if pe_ratio is not None:
        valuation_bits.append(f"P/E {_format_number(pe_ratio, 1)}")
    if pb_ratio is not None:
        valuation_bits.append(f"P/BV {_format_number(pb_ratio, 2)}x")
    if valuation_bits:
        fragments.append("Valuation snapshot: " + ", ".join(valuation_bits) + ".")

    profit_qoq = metrics.get("profit_growth_qoq_pct")
    rev_qoq = metrics.get("revenue_growth_qoq_pct")
    trend_bits = []
    if rev_qoq is not None:
        trend_bits.append(f"revenue {_format_pct(rev_qoq)} QoQ")
    if profit_qoq is not None:
        trend_bits.append(f"profit {_format_pct(profit_qoq)} QoQ")
    if trend_bits:
        fragments.append("Momentum in the numbers: " + ", ".join(trend_bits) + ".")

    return " ".join(fragments)


def build_stock_report(symbol: str, current_price: float | None = None) -> dict[str, Any]:
    """Build a deterministic stock report from cached scraped data."""
    symbol = symbol.upper().strip()
    profile = _company_profile(symbol)
    cache_rows, latest_detail = _load_cached_report_rows(symbol)
    db_rows = _load_db_quarter_rows(symbol)
    override_rows = _load_override_quarter_rows(symbol)
    quarter_rows = _merge_quarter_rows(_merge_quarter_rows(cache_rows, db_rows), override_rows)
    latest = latest_detail or (quarter_rows[0] if quarter_rows else {})
    previous = quarter_rows[1] if len(quarter_rows) > 1 else None
    fundamentals = _latest_fundamentals_row(symbol)
    metrics = _derive_metrics(latest or {}, previous, current_price, fundamentals) if (latest or fundamentals) else {}

    eps_snapshot = _safe_float(latest.get("eps")) if latest else None
    if eps_snapshot is None:
        eps_snapshot = _safe_float(fundamentals.get("eps"))

    book_value_snapshot = _safe_float(latest.get("book_value")) if latest else None
    if book_value_snapshot is None:
        book_value_snapshot = _safe_float(fundamentals.get("book_value_per_share"))

    if latest:
        sector = str(latest.get("sector") or fundamentals.get("sector") or "").strip()
        score, signal, positives, risks, monitors = _score_report(latest, metrics, fundamentals)
        summary = _build_summary(symbol, signal, score, latest, metrics, current_price)
    elif fundamentals:
        sector = str(fundamentals.get("sector") or "").strip()
        score, signal = 0, "NO DATA"
        positives = []
        risks = []
        monitors = ["Quarterly financial cache is missing; showing latest stored valuation snapshot."]
        valuation_bits = []
        if current_price:
            valuation_bits.append(f"price {current_price:.1f}")
        if metrics.get("pe_ratio") is not None:
            valuation_bits.append(f"P/E {_format_number(metrics.get('pe_ratio'), 1)}")
        if metrics.get("pb_ratio") is not None:
            valuation_bits.append(f"P/BV {_format_number(metrics.get('pb_ratio'), 2)}x")
        summary = f"{symbol} has no cached quarterly report yet, but stored fundamentals are available for snapshot analysis."
        if valuation_bits:
            summary += " Valuation snapshot: " + ", ".join(valuation_bits) + "."
    else:
        sector = str(fundamentals.get("sector") or "").strip()
        score, signal = 0, "NO DATA"
        positives = []
        risks = []
        if symbol == "NEPSE":
            monitors = ["Index view uses price and breadth data only."]
            summary = "NEPSE is an index. Company financial statements do not apply."
        else:
            monitors = ["No cached quarterly report is available for this symbol yet."]
            summary = f"{symbol} is in price-only mode. No local company financials are loaded."

    snapshot = [
        ("Sector", sector.title() if sector else "—"),
        ("Quarter", _format_period(latest.get("fiscal_year", ""), latest.get("quarter")) if latest else "—"),
        ("Revenue", _compact_money(_safe_float(latest.get("revenue")) if latest else None)),
        ("Net Profit", _compact_money(_safe_float(latest.get("net_profit")) if latest else None)),
        ("EPS", _format_number(eps_snapshot, 2)),
        ("Book Value", _format_number(book_value_snapshot, 2)),
        ("P/E", _format_number(metrics.get("pe_ratio"), 1) if metrics else "—"),
        ("P/BV", f"{_format_number(metrics.get('pb_ratio'), 2)}x" if metrics and metrics.get("pb_ratio") is not None else "—"),
        ("Reported P/E", _format_number(_safe_float(fundamentals.get("pe_ratio")), 1)),
        ("Reported P/BV", f"{_format_number(_safe_float(fundamentals.get('pb_ratio')), 2)}x" if _safe_float(fundamentals.get("pb_ratio")) is not None else "—"),
        ("ROE", _format_pct(metrics.get("roe_pct")) if metrics else "—"),
        ("Margin", _format_pct(metrics.get("profit_margin_pct")) if metrics else "—"),
        ("Debt/Equity", f"{_format_number(metrics.get('debt_to_equity'), 2)}x" if metrics and metrics.get("debt_to_equity") is not None else "—"),
        ("Revenue QoQ", _format_pct(metrics.get("revenue_growth_qoq_pct")) if metrics else "—"),
        ("Profit QoQ", _format_pct(metrics.get("profit_growth_qoq_pct")) if metrics else "—"),
    ]

    if metrics.get("loan_to_deposit_pct") is not None:
        snapshot.append(("Loan/Deposit", _format_pct(metrics.get("loan_to_deposit_pct"))))
    if _safe_float(latest.get("npl_pct")) is not None:
        snapshot.append(("NPL", _format_pct(_safe_float(latest.get("npl_pct")))))
    if _safe_float(latest.get("capital_adequacy_pct")) is not None:
        snapshot.append(("CAR", _format_pct(_safe_float(latest.get("capital_adequacy_pct")))))

    financial_rows = []
    for row in quarter_rows[:4]:
        financial_rows.append(
            {
                "period": row["period"],
                "revenue": _compact_money(_safe_float(row.get("revenue"))),
                "net_profit": _compact_money(_safe_float(row.get("net_profit"))),
                "eps": _format_number(_safe_float(row.get("eps")), 2),
                "book_value": _format_number(_safe_float(row.get("book_value")), 2),
            }
        )

    return {
        "symbol": symbol,
        "company_name": profile.get("company_name") or "",
        "signal": signal,
        "score": score,
        "summary": summary,
        "snapshot": snapshot,
        "financial_rows": financial_rows,
        "positives": positives,
        "risks": risks,
        "monitors": monitors,
        "sector": sector,
        "latest_notes": str(latest.get("notes", "")).strip() if latest else "",
        "company_profile": profile,
        "intelligence": _build_intelligence_brief(symbol),
        "has_data": bool(latest or fundamentals),
    }
