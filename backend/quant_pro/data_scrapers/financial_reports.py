#!/usr/bin/env python3
"""Unified NEPSE financial reports scraper.

Scrapes full report histories from MeroLagani for each symbol and classifies
them into quarterly, annual, and other financial filings. It stores metadata
in SQLite and caches per-symbol JSON plus downloaded source files.

This module is intentionally split from the older quarterly-only scraper so the
project can scrape the entire market without forcing expensive structured
extraction for every filing. Structured extraction remains optional.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from backend.quant_pro.database import get_db_path
from backend.quant_pro.data_scrapers.quarterly_reports import extract_financials_with_claude
from backend.quant_pro.local_financial_ocr import extract_financials_locally
from backend.quant_pro.paths import get_project_root

log = logging.getLogger(__name__)

MEROLAGANI_BASE = "https://merolagani.com"
IMAGE_BASE = "https://images.merolagani.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PROJECT_ROOT = get_project_root(__file__)
CACHE_DIR = PROJECT_ROOT / "data" / "financial_reports"
FILES_DIR = CACHE_DIR / "files"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REQUEST_SLEEP = float(os.environ.get("NEPSE_FIN_REPORT_REQUEST_SLEEP", "0.35"))
DEFAULT_SYMBOL_SLEEP = float(os.environ.get("NEPSE_FIN_REPORT_SYMBOL_SLEEP", "0.20"))


def _claude_extraction_enabled() -> bool:
    return os.environ.get("NEPSE_ENABLE_CLAUDE_EXTRACTION", "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ReportTypeConfig:
    name: str
    keywords: tuple[str, ...]


REPORT_TYPES: tuple[ReportTypeConfig, ...] = (
    ReportTypeConfig(
        "quarterly",
        (
            "quarterly report",
            "quarterly financial statement",
            "provisional financial statement",
            "unaudited financial statement",
            "first quarter",
            "second quarter",
            "third quarter",
            "fourth quarter",
            "1st quarterly report",
            "2nd quarterly report",
            "3rd quarterly report",
            "4th quarterly report",
            "q1",
            "q2",
            "q3",
            "q4",
        ),
    ),
    ReportTypeConfig(
        "annual",
        (
            "annual report",
            "annual financial statement",
            "audited financial statement",
            "annual financial statements",
        ),
    ),
    ReportTypeConfig(
        "other_financial",
        (
            "monthly report",
            "monthly financial statement",
            "half yearly report",
            "half-yearly report",
            "semi annual report",
            "financial highlights",
            "interim financial statement",
            "nine month report",
            "third month report",
            "annual budget",
        ),
    ),
)

QUARTER_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
}


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _sleep(seconds: float = DEFAULT_REQUEST_SLEEP) -> None:
    time.sleep(seconds)


def _normalize_fiscal_year(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^FY[:\s]*", "", raw, flags=re.I)

    match = re.search(r"(\d{4})[/-](\d{4})", raw)
    if match:
        return f"{match.group(1)[-3:]}-{match.group(2)[-3:]}"

    match = re.search(r"(\d{4})[/-](\d{2})", raw)
    if match:
        return f"{match.group(1)[-3:]}-0{match.group(2)}"

    match = re.search(r"(\d{3})[/-](\d{3})", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match = re.search(r"(\d{3})[/-](\d{2})", raw)
    if match:
        return f"{match.group(1)}-0{match.group(2)}"

    return raw


def classify_report_type(description: str) -> str | None:
    desc = (description or "").lower()
    for config in REPORT_TYPES:
        if any(keyword in desc for keyword in config.keywords):
            return config.name
    if "financial statement" in desc or "financial report" in desc:
        return "other_financial"
    return None


def parse_report_period(description: str) -> dict[str, Any]:
    desc = description or ""
    lower = desc.lower()
    result: dict[str, Any] = {"fiscal_year": None, "quarter": None, "period_label": None}

    fy_match = re.search(r"fiscal year\s+(\d{3,4}[/-]\d{2,4})", desc, re.I)
    if not fy_match:
        fy_match = re.search(r"\b(\d{3,4}[/-]\d{2,4})\b", desc)
    if fy_match:
        result["fiscal_year"] = _normalize_fiscal_year(fy_match.group(1))

    for token, quarter in QUARTER_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            result["quarter"] = quarter
            break

    if result["quarter"]:
        result["period_label"] = f"Q{result['quarter']}"
    elif "annual" in lower or "audited" in lower:
        result["period_label"] = "FY"
    elif "monthly" in lower:
        result["period_label"] = "MONTHLY"
    elif "half yearly" in lower or "half-yearly" in lower or "semi annual" in lower:
        result["period_label"] = "H1"
    elif "nine month" in lower:
        result["period_label"] = "9M"

    return result


def fetch_symbol_announcements(
    symbol: str,
    session: requests.Session | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Fetch a large announcement history for a symbol from MeroLagani."""
    session = session or _new_session()
    url = (
        f"{MEROLAGANI_BASE}/handlers/webrequesthandler.ashx"
        f"?type=get_announcements&symbol={symbol}&pageSize={page_size}&pageNo=1"
    )
    response = session.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected announcement payload for {symbol}: {type(payload)}")
    return payload


def resolve_report_file_url(
    announcement_id: str,
    session: requests.Session | None = None,
) -> str | None:
    """Resolve the downloadable image/pdf URL from an announcement detail page."""
    session = session or _new_session()
    url = f"{MEROLAGANI_BASE}/AnnouncementDetail.aspx?id={announcement_id}"
    response = session.get(url, timeout=20)
    response.raise_for_status()
    text = response.text

    match = re.search(
        r"(https?://images\.merolagani\.com//Uploads/Repository/\d+\.(?:gif|png|jpg|jpeg|pdf|webp))",
        text,
        re.I,
    )
    if match:
        return match.group(1)

    soup = BeautifulSoup(text, "html.parser")
    hidden = soup.find("input", {"id": re.compile(r"pdfviewer", re.I)})
    if hidden and hidden.get("value"):
        value = hidden["value"]
        if value.startswith("http"):
            return value
        return f"{IMAGE_BASE}/{value.lstrip('/')}"

    for element in soup.find_all(["a", "iframe", "embed", "img"], href=True) + soup.find_all(["iframe", "embed", "img"], src=True):
        candidate = element.get("href") or element.get("src")
        if not candidate:
            continue
        if "Uploads/Repository" not in candidate:
            continue
        if candidate.startswith("http"):
            return candidate
        if candidate.startswith("//"):
            return f"https:{candidate}"
        return f"{MEROLAGANI_BASE}/{candidate.lstrip('/')}"

    return None


def _extension_from_url(url: str | None) -> str:
    if not url:
        return ".bin"
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix else ".bin"


def download_report_file(
    file_url: str,
    target_path: Path,
    session: requests.Session | None = None,
) -> bool:
    session = session or _new_session()
    response = session.get(file_url, timeout=40)
    response.raise_for_status()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return True


def create_financial_reports_table(db_path: str | Path | None = None) -> None:
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS financial_reports (
            symbol TEXT NOT NULL,
            announcement_id TEXT NOT NULL,
            announcement_date TEXT,
            report_type TEXT NOT NULL,
            fiscal_year TEXT,
            fiscal_quarter INTEGER,
            period_label TEXT,
            description TEXT,
            file_url TEXT,
            local_path TEXT,
            file_ext TEXT,
            extraction_status TEXT,
            extracted_json TEXT,
            extracted_at_utc TEXT,
            source TEXT DEFAULT 'merolagani',
            PRIMARY KEY (symbol, announcement_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_financial_reports_symbol ON financial_reports(symbol)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_financial_reports_type ON financial_reports(report_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_financial_reports_date ON financial_reports(announcement_date)"
    )
    conn.commit()
    conn.close()


def upsert_financial_reports(
    rows: Iterable[dict[str, Any]],
    db_path: str | Path | None = None,
) -> int:
    db_path = db_path or get_db_path()
    create_financial_reports_table(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    inserted = 0
    for row in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO financial_reports (
                symbol, announcement_id, announcement_date, report_type, fiscal_year,
                fiscal_quarter, period_label, description, file_url, local_path, file_ext,
                extraction_status, extracted_json, extracted_at_utc, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("symbol"),
                row.get("announcement_id"),
                row.get("announcement_date"),
                row.get("report_type"),
                row.get("fiscal_year"),
                row.get("fiscal_quarter"),
                row.get("period_label"),
                row.get("description"),
                row.get("file_url"),
                row.get("local_path"),
                row.get("file_ext"),
                row.get("extraction_status"),
                json.dumps(row.get("extracted_json")) if row.get("extracted_json") is not None else None,
                row.get("extracted_at_utc"),
                row.get("source", "merolagani"),
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return inserted


def list_all_symbols(db_path: str | Path | None = None) -> list[str]:
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=30)
    rows = conn.execute(
        """
        SELECT DISTINCT symbol
        FROM stock_prices
        WHERE symbol NOT LIKE 'SECTOR::%' AND symbol != 'NEPSE'
        ORDER BY symbol
        """
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]


def _cache_file(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def _load_cached_symbol_reports(symbol: str) -> dict[str, Any] | None:
    path = _cache_file(symbol)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cached_symbol_reports(symbol: str, payload: dict[str, Any]) -> None:
    _cache_file(symbol).write_text(json.dumps(payload, indent=2, default=str))


def scrape_symbol_financial_reports(
    symbol: str,
    session: requests.Session | None = None,
    report_types: set[str] | None = None,
    download: bool = True,
    extract_local: bool = False,
    extract_structured: bool = False,
    extract_types: set[str] | None = None,
    force: bool = False,
    db_path: str | Path | None = None,
    request_sleep: float = DEFAULT_REQUEST_SLEEP,
) -> dict[str, Any]:
    """Scrape all financial reports for a symbol and persist metadata."""
    session = session or _new_session()
    report_types = report_types or {"quarterly", "annual", "other_financial"}
    extract_types = extract_types or {"quarterly", "annual", "other_financial"}
    symbol = symbol.upper().strip()

    if extract_structured and not _claude_extraction_enabled():
        raise RuntimeError(
            "Structured extraction is disabled. Set NEPSE_ENABLE_CLAUDE_EXTRACTION=true to allow Claude-backed extraction."
        )

    cached = _load_cached_symbol_reports(symbol)
    if cached and not force:
        requested = set(cached.get("report_types", []))
        cached_reports = [report for report in cached.get("reports", []) if report.get("report_type") in report_types]
        downloads_ready = (
            not download
            or all((not report.get("file_url")) or report.get("local_path") for report in cached_reports)
        )
        extraction_ready = (
            not extract_structured and not extract_local
            or all(
                report.get("report_type") not in extract_types
                or
                report.get("extracted_json") is not None
                or str(report.get("extraction_status") or "").startswith("extract_error")
                for report in cached_reports
            )
        )
        if report_types.issubset(requested) and downloads_ready and extraction_ready:
            return cached

    announcements = fetch_symbol_announcements(symbol, session=session, page_size=500)
    records: list[dict[str, Any]] = []
    run_stats = {
        "symbol": symbol,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_types": sorted(report_types),
        "announcement_count": len(announcements),
        "matched_reports": 0,
        "downloaded_reports": 0,
        "resolved_files": 0,
        "extracted_reports": 0,
        "reports": [],
    }

    seen_ids: set[str] = set()
    for announcement in announcements:
        announcement_id = str(announcement.get("announcementID") or "").strip()
        if not announcement_id or announcement_id in seen_ids:
            continue
        seen_ids.add(announcement_id)

        description = (announcement.get("announcementDetail") or "").strip()
        report_type = classify_report_type(description)
        if report_type not in report_types:
            continue

        parsed = parse_report_period(description)
        record: dict[str, Any] = {
            "symbol": symbol,
            "announcement_id": announcement_id,
            "announcement_date": (announcement.get("announcementDateAD") or "")[:10],
            "report_type": report_type,
            "fiscal_year": parsed.get("fiscal_year"),
            "fiscal_quarter": parsed.get("quarter"),
            "period_label": parsed.get("period_label"),
            "description": description,
            "file_url": None,
            "local_path": None,
            "file_ext": None,
            "extraction_status": None,
            "extracted_json": None,
            "extracted_at_utc": None,
            "source": "merolagani",
        }
        run_stats["matched_reports"] += 1

        try:
            file_url = resolve_report_file_url(announcement_id, session=session)
            record["file_url"] = file_url
            record["file_ext"] = _extension_from_url(file_url)
            if file_url:
                run_stats["resolved_files"] += 1
            _sleep(request_sleep)
        except Exception as exc:
            record["extraction_status"] = f"resolve_error: {exc}"
            records.append(record)
            continue

        file_path: Path | None = None
        if record["file_url"] and download:
            try:
                file_dir = FILES_DIR / symbol
                file_path = file_dir / f"{announcement_id}{record['file_ext']}"
                if force or not file_path.exists():
                    download_report_file(record["file_url"], file_path, session=session)
                    _sleep(request_sleep)
                record["local_path"] = str(file_path)
                run_stats["downloaded_reports"] += 1
            except Exception as exc:
                record["extraction_status"] = f"download_error: {exc}"

        if extract_local and report_type in extract_types and file_path and file_path.exists():
            try:
                extracted = extract_financials_locally(file_path, description)
                record["extracted_json"] = extracted
                record["extracted_at_utc"] = datetime.now(timezone.utc).isoformat()
                quality = extracted.get("quality") or {}
                status = "local_ok"
                if quality.get("confidence", 0) < 0.55 or quality.get("review_flags"):
                    status = "local_review"
                record["extraction_status"] = status
                run_stats["extracted_reports"] += 1
            except Exception as exc:
                record["extraction_status"] = f"local_extract_error: {exc}"
        elif extract_structured and report_type in extract_types and file_path and file_path.exists():
            try:
                extracted = extract_financials_with_claude(file_path, description)
                record["extracted_json"] = extracted
                record["extracted_at_utc"] = datetime.now(timezone.utc).isoformat()
                record["extraction_status"] = "ok" if "error" not in extracted else "extract_error"
                run_stats["extracted_reports"] += 1
            except Exception as exc:
                record["extraction_status"] = f"extract_error: {exc}"

        records.append(record)

    existing_reports = []
    existing_types: set[str] = set()
    if cached:
        existing_reports = [
            report
            for report in cached.get("reports", [])
            if report.get("report_type") not in report_types
        ]
        existing_types = set(cached.get("report_types", []))

    merged_reports = existing_reports + records
    merged_reports.sort(
        key=lambda report: (
            report.get("announcement_date") or "",
            str(report.get("announcement_id") or ""),
        ),
        reverse=True,
    )

    payload = {
        "symbol": symbol,
        "scraped_at_utc": run_stats["scraped_at_utc"],
        "report_types": sorted(existing_types | report_types),
        "extract_types": sorted(extract_types),
        "announcement_count": len(announcements),
        "matched_reports": len(merged_reports),
        "resolved_files": sum(1 for report in merged_reports if report.get("file_url")),
        "downloaded_reports": sum(1 for report in merged_reports if report.get("local_path")),
        "extracted_reports": sum(1 for report in merged_reports if report.get("extracted_json") is not None),
        "reports": merged_reports,
    }

    _save_cached_symbol_reports(symbol, payload)
    upsert_financial_reports(records, db_path=db_path)
    return payload


def scrape_market_financial_reports(
    symbols: list[str] | None = None,
    db_path: str | Path | None = None,
    report_types: set[str] | None = None,
    download: bool = True,
    extract_local: bool = False,
    extract_structured: bool = False,
    extract_types: set[str] | None = None,
    force: bool = False,
    max_symbols: int | None = None,
    request_sleep: float = DEFAULT_REQUEST_SLEEP,
    symbol_sleep: float = DEFAULT_SYMBOL_SLEEP,
) -> dict[str, Any]:
    """Scrape report histories for a symbol list or the full NEPSE universe."""
    db_path = db_path or get_db_path()
    symbols = symbols or list_all_symbols(db_path)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    extract_types = extract_types or {"quarterly", "annual", "other_financial"}

    session = _new_session()
    create_financial_reports_table(db_path)
    started = time.monotonic()

    aggregate = {
        "symbols_total": len(symbols),
        "symbols_processed": 0,
        "symbols_with_reports": 0,
        "matched_reports": 0,
        "resolved_files": 0,
        "downloaded_reports": 0,
        "extracted_reports": 0,
        "errors": 0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
    }

    for idx, symbol in enumerate(symbols, 1):
        elapsed = time.monotonic() - started
        avg_per_symbol = elapsed / max(1, aggregate["symbols_processed"])
        remaining = len(symbols) - aggregate["symbols_processed"]
        eta = avg_per_symbol * remaining if aggregate["symbols_processed"] else None
        if eta is not None:
            log.info("[%d/%d] Scraping %s reports | elapsed %.1fm | eta %.1fm", idx, len(symbols), symbol, elapsed / 60, eta / 60)
        else:
            log.info("[%d/%d] Scraping %s reports", idx, len(symbols), symbol)
        try:
            result = scrape_symbol_financial_reports(
                symbol,
                session=session,
                report_types=report_types,
                download=download,
                extract_local=extract_local,
                extract_structured=extract_structured,
                extract_types=extract_types,
                force=force,
                db_path=db_path,
                request_sleep=request_sleep,
            )
            aggregate["symbols_processed"] += 1
            if result.get("matched_reports", 0) > 0:
                aggregate["symbols_with_reports"] += 1
            for key in ("matched_reports", "resolved_files", "downloaded_reports", "extracted_reports"):
                aggregate[key] += int(result.get(key, 0) or 0)
            aggregate["elapsed_seconds"] = time.monotonic() - started
            if aggregate["symbols_processed"] > 0:
                avg_per_symbol = aggregate["elapsed_seconds"] / aggregate["symbols_processed"]
                aggregate["eta_seconds"] = avg_per_symbol * (len(symbols) - aggregate["symbols_processed"])
            _sleep(symbol_sleep)
        except Exception as exc:
            aggregate["errors"] += 1
            log.exception("Failed scraping %s reports: %s", symbol, exc)

    aggregate["elapsed_seconds"] = time.monotonic() - started
    aggregate["eta_seconds"] = 0.0
    return aggregate


def summarize_symbol_reports(symbol: str) -> dict[str, Any]:
    cached = _load_cached_symbol_reports(symbol) or {}
    counts: dict[str, int] = {}
    for report in cached.get("reports", []):
        report_type = report.get("report_type", "unknown")
        counts[report_type] = counts.get(report_type, 0) + 1
    return {
        "symbol": symbol.upper(),
        "counts": counts,
        "resolved_files": sum(1 for report in cached.get("reports", []) if report.get("file_url")),
        "downloaded_files": sum(1 for report in cached.get("reports", []) if report.get("local_path")),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Full NEPSE financial reports scraper")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all symbols from stock_prices")
    parser.add_argument(
        "--report-types",
        nargs="*",
        choices=["quarterly", "annual", "other_financial"],
        default=["quarterly", "annual", "other_financial"],
        help="Report types to scrape",
    )
    parser.add_argument("--no-download", action="store_true", help="Skip file downloads")
    parser.add_argument("--extract-local", action="store_true", help="Run local OCR extraction on downloaded files")
    parser.add_argument("--extract-structured", action="store_true", help="Run Claude extraction on downloaded files")
    parser.add_argument(
        "--extract-types",
        nargs="*",
        choices=["quarterly", "annual", "other_financial"],
        default=["quarterly", "annual", "other_financial"],
        help="Report types to send through structured extraction",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing cache")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit symbol count for a partial run")
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=DEFAULT_REQUEST_SLEEP,
        help="Seconds to sleep after each announcement detail or file request",
    )
    parser.add_argument(
        "--symbol-sleep",
        type=float,
        default=DEFAULT_SYMBOL_SLEEP,
        help="Seconds to sleep between symbols",
    )
    parser.add_argument("--summary", nargs="*", help="Show cache summary for specific symbols and exit")
    args = parser.parse_args()

    if args.summary:
        for symbol in args.summary:
            print(json.dumps(summarize_symbol_reports(symbol), indent=2))
        return

    if args.extract_structured and not _claude_extraction_enabled():
        raise SystemExit(
            "Structured extraction is disabled. Export NEPSE_ENABLE_CLAUDE_EXTRACTION=true if you explicitly want Claude extraction."
        )

    symbols = [symbol.upper() for symbol in args.symbols] if args.symbols else None
    if args.all and not symbols:
        symbols = list_all_symbols()

    stats = scrape_market_financial_reports(
        symbols=symbols,
        report_types=set(args.report_types),
        download=not args.no_download,
        extract_local=args.extract_local,
        extract_structured=args.extract_structured,
        extract_types=set(args.extract_types),
        force=args.force,
        max_symbols=args.max_symbols,
        request_sleep=args.request_sleep,
        symbol_sleep=args.symbol_sleep,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
