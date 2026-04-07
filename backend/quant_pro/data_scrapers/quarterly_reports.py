#!/usr/bin/env python3
"""
Merolagani Quarterly Report Scraper

Scrapes quarterly financial statement images from Merolagani, extracts
structured financial data using Claude (Sonnet) vision, and caches results.

Flow:
  1. Fetch announcement list for a symbol from Merolagani
  2. Get the GIF image URL from each announcement detail page
  3. Download the GIF (financial statement image)
  4. Send to Claude Sonnet to extract structured financials (Nepali → English)
  5. Cache extracted data as JSON per symbol

Usage:
    python3 -m backend.quant_pro.data_scrapers.quarterly_reports --symbol NABIL
    python3 -m backend.quant_pro.data_scrapers.quarterly_reports --symbols NABIL JBBL SCB
    python3 -m backend.quant_pro.data_scrapers.quarterly_reports --from-signals
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from backend.quant_pro.paths import get_project_root

project_root = str(get_project_root(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

log = logging.getLogger(__name__)

MEROLAGANI_BASE = "https://merolagani.com"
IMAGE_BASE = "https://images.merolagani.com"
CACHE_DIR = Path(project_root) / "data" / "quarterly_reports"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How many quarters back to fetch (most recent)
MAX_QUARTERS = 4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _claude_extraction_enabled() -> bool:
    return os.environ.get("NEPSE_ENABLE_CLAUDE_EXTRACTION", "").strip().lower() in {"1", "true", "yes"}


def get_announcement_ids(symbol: str, max_reports: int = MAX_QUARTERS) -> list[dict]:
    """Fetch quarterly report announcement IDs for a symbol from Merolagani.

    Uses the Merolagani AJAX API to get all announcements for a symbol,
    then filters for quarterly financial statement announcements.
    """
    url = (f"{MEROLAGANI_BASE}/handlers/webrequesthandler.ashx"
           f"?type=get_announcements&symbol={symbol}&pageSize=50&pageNo=1")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        announcements = r.json()
        if not isinstance(announcements, list):
            log.warning(f"Unexpected response for {symbol}: {type(announcements)}")
            return []
    except Exception as e:
        log.warning(f"Failed to fetch announcements for {symbol}: {e}")
        return []

    results = []
    quarterly_keywords = [
        "provisional financial statement",
        "quarterly report",
        "unaudited financial",
    ]
    for ann in announcements:
        desc = ann.get("announcementDetail", "")
        # Filter for quarterly financial statements only
        if not any(kw in desc.lower() for kw in quarterly_keywords):
            continue
        results.append({
            "id": str(ann["announcementID"]),
            "description": desc[:200],
            "date": ann.get("announcementDateAD", "")[:10],
        })
        if len(results) >= max_reports:
            break

    return results


def get_image_url(announcement_id: str) -> str | None:
    """Extract the financial statement image URL from an announcement detail page."""
    url = f"{MEROLAGANI_BASE}/AnnouncementDetail.aspx?id={announcement_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"Failed to fetch announcement {announcement_id}: {e}")
        return None

    # Look for the GIF URL in the page source
    # Pattern: images.merolagani.com//Uploads/Repository/XXXXX.gif
    match = re.search(
        r'(https?://images\.merolagani\.com//Uploads/Repository/\d+\.(?:gif|png|jpg|pdf))',
        r.text
    )
    if match:
        return match.group(1)

    # Fallback: look for hidden input with PDF viewer value
    soup = BeautifulSoup(r.text, "html.parser")
    hidden = soup.find("input", {"id": re.compile(r"pdfviewer", re.I)})
    if hidden and hidden.get("value"):
        val = hidden["value"]
        if val.startswith("http"):
            return val
        return f"{IMAGE_BASE}/{val.lstrip('/')}"

    return None


def download_image(url: str, save_path: Path) -> bool:
    """Download a financial statement image."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        save_path.write_bytes(r.content)
        return True
    except Exception as e:
        log.warning(f"Failed to download {url}: {e}")
        return False


def extract_financials_with_claude(image_path: Path, description: str = "") -> dict:
    """
    Use Claude Haiku (via CLI) to extract structured financial data from
    a quarterly financial statement image (may be in Nepali or English).

    Passes the image file path so Claude can read it with its Read tool.
    """
    if not _claude_extraction_enabled():
        return {
            "error": "Claude extraction disabled. Set NEPSE_ENABLE_CLAUDE_EXTRACTION=true to allow it.",
        }

    abs_path = str(image_path.resolve())

    prompt = f"""Read the image at {abs_path} — it's a quarterly financial statement from Nepal.
Context: {description}

CRITICAL RULES:
- Identify the company's SECTOR first (banking, hydropower, insurance, microfinance, manufacturing, etc.)
- For BANKS/MICROFINANCE: interest_income and total_deposits are relevant
- For HYDROPOWER/MANUFACTURING/OTHER: interest_income should be 0 (revenue from sales is NOT interest income). total_deposits should be 0.
- "Revenue from Sale of Electricity" or "बिक्रीबाट आम्दानी" = total_revenue, NOT interest_income
- "Gross Profit" or "कुल नाफा" = operating_profit
- Look for EPS (प्रति शेयर आम्दानी) and book value (प्रति शेयर नेटवर्थ) — often near the bottom
- fiscal_year should be Nepali BS year format like "2082/83" or "081-082"

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "sector": "banking/hydropower/insurance/microfinance/manufacturing/other",
  "fiscal_year": "XXXX/XX",
  "quarter": "Q1/Q2/Q3/Q4",
  "balance_sheet": {{
    "total_assets": number,
    "total_liabilities": number,
    "shareholders_equity": number,
    "share_capital": number,
    "retained_earnings": number,
    "total_deposits": number,
    "total_loans": number
  }},
  "income_statement": {{
    "total_revenue": number,
    "operating_profit": number,
    "net_profit": number,
    "interest_income": number,
    "interest_expense": number
  }},
  "per_share": {{
    "eps": number,
    "book_value": number
  }},
  "ratios": {{
    "npl_pct": number,
    "capital_adequacy_pct": number,
    "cost_income_ratio": number
  }},
  "notes": "sector classification and key observations (in English)"
}}

Use 0 for fields not present or not applicable to the sector. Numbers in NPR, no commas."""

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude", "-p",
        "--model", "haiku",
        "--output-format", "text",
        "--no-session-persistence",
        "--allowedTools", "Read",
        "--add-dir", str(CACHE_DIR),
    ]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            env=env, timeout=120, cwd=project_root,
        )
        if result.returncode != 0:
            log.error(f"Claude CLI failed: {result.stderr[:300]}")
            return {"error": result.stderr[:300]}

        text = result.stdout.strip()
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"error": "No JSON in response", "raw": text[:500]}
    except subprocess.TimeoutExpired:
        return {"error": "Claude CLI timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except FileNotFoundError:
        return {"error": "Claude CLI not found"}


def scrape_quarterly_reports(
    symbol: str,
    max_quarters: int = MAX_QUARTERS,
    force: bool = False,
) -> dict:
    """
    Scrape and extract quarterly financial reports for a symbol.
    Returns cached data if available and not forced.
    """
    cache_file = CACHE_DIR / f"{symbol}.json"

    # Check cache
    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            age_hours = (time.time() - cached.get("scraped_at", 0)) / 3600
            if age_hours < 168:  # 1 week cache
                log.info(f"{symbol}: using cached data ({age_hours:.0f}h old)")
                return cached
        except Exception:
            pass

    log.info(f"{symbol}: fetching quarterly reports from Merolagani...")
    announcements = get_announcement_ids(symbol, max_quarters)
    if not announcements:
        log.warning(f"{symbol}: no quarterly reports found")
        return {"symbol": symbol, "reports": [], "error": "No reports found"}

    reports = []
    for ann in announcements:
        ann_id = ann["id"]
        log.info(f"  {symbol} announcement {ann_id}: {ann['description'][:60]}")

        # Get image URL
        img_url = get_image_url(ann_id)
        if not img_url:
            log.warning(f"  No image URL for announcement {ann_id}")
            reports.append({
                "announcement_id": ann_id,
                "description": ann["description"],
                "error": "No image URL found",
            })
            continue

        # Download image
        img_path = CACHE_DIR / f"{symbol}_{ann_id}.gif"
        if not img_path.exists():
            log.info(f"  Downloading {img_url}")
            if not download_image(img_url, img_path):
                reports.append({
                    "announcement_id": ann_id,
                    "description": ann["description"],
                    "error": "Download failed",
                })
                continue
        else:
            log.info(f"  Image already cached: {img_path.name}")

        # Extract financials with Claude
        log.info(f"  Extracting financials with Claude Sonnet...")
        financials = extract_financials_with_claude(img_path, ann.get("description", ""))
        reports.append({
            "announcement_id": ann_id,
            "description": ann["description"],
            "image_url": img_url,
            "financials": financials,
        })

        # Rate limit between API calls
        time.sleep(1)

    result = {
        "symbol": symbol,
        "scraped_at": time.time(),
        "report_count": len(reports),
        "reports": reports,
    }

    # Save cache
    cache_file.write_text(json.dumps(result, indent=2, default=str))
    log.info(f"{symbol}: saved {len(reports)} reports to {cache_file}")
    return result


def get_cached_financials(symbol: str) -> dict | None:
    """Get cached financial data for a symbol (no scraping)."""
    cache_file = CACHE_DIR / f"{symbol}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            return None
    # Fallback to the unified financial reports cache if quarterly-only cache
    # is not present. This keeps older consumers working after the broader
    # annual/quarterly scraper is introduced.
    unified_cache = Path(project_root) / "data" / "financial_reports" / f"{symbol}.json"
    if unified_cache.exists():
        try:
            unified = json.loads(unified_cache.read_text())
            reports = []
            for report in unified.get("reports", []):
                if report.get("report_type") != "quarterly":
                    continue
                status = str(report.get("extraction_status") or "")
                extracted = report.get("extracted_json")
                quality = (extracted or {}).get("quality") or {}
                confidence = float(quality.get("confidence") or 0)
                review_flags = quality.get("review_flags") or []
                if status.startswith("local_"):
                    if status != "local_ok" or confidence < 0.70 or review_flags:
                        continue
                elif status != "ok":
                    continue
                if extracted:
                    reports.append(
                        {
                            "announcement_id": report.get("announcement_id"),
                            "description": report.get("description", ""),
                            "image_url": report.get("file_url"),
                            "financials": extracted,
                        }
                    )
            if reports:
                return {
                    "symbol": symbol,
                    "scraped_at": unified.get("scraped_at_utc"),
                    "report_count": len(reports),
                    "reports": reports,
                }
        except Exception:
            return None
    return None


def get_financial_summary(symbol: str) -> str:
    """Get a text summary of cached financials for agent context."""
    data = get_cached_financials(symbol)
    if not data or not data.get("reports"):
        return f"No quarterly financial data available for {symbol}"

    # Get sector from first report
    sector = "unknown"
    for r in data["reports"]:
        fin = r.get("financials", {})
        if fin.get("sector"):
            sector = fin["sector"]
            break

    lines = [f"QUARTERLY FINANCIALS — {symbol} ({sector}, from Merolagani):"]
    for r in data["reports"]:
        fin = r.get("financials", {})
        if "error" in fin:
            continue
        fy = fin.get("fiscal_year", "?")
        q = fin.get("quarter", "?")
        inc = fin.get("income_statement", {})
        bs = fin.get("balance_sheet", {})
        ps = fin.get("per_share", {})
        ratios = fin.get("ratios", {})

        parts = [f"  {fy} {q}:"]
        if inc.get("net_profit"):
            parts.append(f"Net Profit={inc['net_profit']:,.0f}")
        if inc.get("total_revenue"):
            parts.append(f"Revenue={inc['total_revenue']:,.0f}")
        if ps.get("eps"):
            parts.append(f"EPS={ps['eps']:.2f}")
        if ps.get("book_value"):
            parts.append(f"BV={ps['book_value']:.1f}")
        if ratios.get("npl_pct"):
            parts.append(f"NPL={ratios['npl_pct']:.2f}%")
        if ratios.get("capital_adequacy_pct"):
            parts.append(f"CAR={ratios['capital_adequacy_pct']:.2f}%")
        if bs.get("total_assets"):
            parts.append(f"Assets={bs['total_assets']:,.0f}")

        notes = fin.get("notes", "")
        if notes:
            parts.append(f"Note: {notes[:80]}")

        lines.append("  ".join(parts))

    return "\n".join(lines) if len(lines) > 1 else f"No parseable financials for {symbol}"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape Merolagani quarterly reports")
    parser.add_argument("--symbol", help="Single symbol to scrape")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols to scrape")
    parser.add_argument("--from-signals", action="store_true",
                        help="Scrape reports for all symbols in current algo signals")
    parser.add_argument("--quarters", type=int, default=MAX_QUARTERS,
                        help=f"Max quarters to fetch (default: {MAX_QUARTERS})")
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    parser.add_argument("--summary", action="store_true",
                        help="Show cached financial summaries")
    args = parser.parse_args()

    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.from_signals:
        try:
            from backend.backtesting.simple_backtest import (
                load_all_prices,
                generate_volume_breakout_signals_at_date,
                generate_quality_signals_at_date,
                generate_low_volatility_signals_at_date,
                generate_mean_reversion_signals_at_date,
            )
            from apps.classic.dashboard import MD, _db
            md = MD(top_n=10)
            conn = _db()
            prices_df = load_all_prices(conn)
            conn.close()
            from datetime import datetime
            today = datetime.strptime(md.latest, "%Y-%m-%d")
            sigs = []
            sigs.extend(generate_volume_breakout_signals_at_date(prices_df, today))
            sigs.extend(generate_quality_signals_at_date(prices_df, today))
            sigs.extend(generate_low_volatility_signals_at_date(prices_df, today))
            sigs.extend(generate_mean_reversion_signals_at_date(prices_df, today))
            sigs = sorted(sigs, key=lambda x: x.strength, reverse=True)
            symbols = [s for s in dict.fromkeys(s.symbol for s in sigs)
                       if s.isalpha()][:20]
            # Also include portfolio holdings
            from apps.classic.dashboard import load_port
            port = load_port()
            if not port.empty:
                for sym in port["Symbol"].unique():
                    if sym not in symbols:
                        symbols.append(sym)
            log.info(f"From signals + portfolio: {len(symbols)} symbols")
        except Exception as e:
            log.error(f"Failed to get signals: {e}")
            sys.exit(1)

    if args.summary:
        for sym in symbols or [p.stem for p in CACHE_DIR.glob("*.json")]:
            print(get_financial_summary(sym))
            print()
        return

    if not symbols:
        parser.print_help()
        return

    for sym in symbols:
        result = scrape_quarterly_reports(sym, args.quarters, args.force)
        n_ok = sum(1 for r in result.get("reports", [])
                   if r.get("financials") and "error" not in r["financials"])
        print(f"{sym}: {n_ok}/{result.get('report_count', 0)} reports extracted")


if __name__ == "__main__":
    main()
