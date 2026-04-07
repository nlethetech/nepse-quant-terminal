"""Local TMS browser executor and queued live execution service."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from .tms_audit import (
    count_intents_for_day,
    find_recent_open_intent,
    init_live_audit_db,
    list_execution_intents,
    load_execution_intent,
    load_latest_live_orders,
    load_latest_live_positions,
    log_execution_attempt,
    mark_intent_notified,
    record_reconciliation_run,
    save_execution_intent,
    save_execution_result,
    save_live_orders,
    save_live_positions,
    update_execution_intent,
)
from .tms_models import (
    ExecutionAction,
    ExecutionIntent,
    ExecutionResult,
    ExecutionSource,
    ExecutionStatus,
    FillState,
    PositionSnapshot,
    SessionStatus,
    utc_now_iso,
)
from .tms_session import TMSSettings, human_pause, load_tms_settings
from .vendor_api import fetch_latest_ltp

logger = logging.getLogger(__name__)


def _extract_watchlist_items(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        symbol = str(
            row.get("symbol")
            or row.get("scrip")
            or row.get("script")
            or row.get("scrip_name")
            or row.get("security")
            or row.get("stock")
            or ""
        ).strip().upper()
        if not symbol or not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", symbol):
            continue
        if symbol in seen or symbol in {"TOTAL", "SCRIP"}:
            continue
        seen.add(symbol)
        items.append(
            {
                "symbol": symbol,
                "ltp": _safe_float(
                    row.get("ltp")
                    or row.get("last_traded_price")
                    or row.get("last_price")
                    or row.get("price")
                ),
                "change_pct": _safe_float(
                    row.get("change_pct")
                    or row.get("change_percent")
                    or row.get("change")
                ),
                "volume": _safe_int(
                    row.get("volume")
                    or row.get("qty")
                    or row.get("quantity")
                ),
                "raw": row,
            }
        )
    return items


def _import_playwright():
    try:
        from playwright.sync_api import Error, TimeoutError, sync_playwright
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Playwright is required for live TMS automation. Install `playwright` and browser binaries."
        ) from exc
    return sync_playwright, TimeoutError, Error


def _screenshot_file(base_dir: Path, prefix: str) -> Path:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix)).strip("_") or "tms"
    return base_dir / f"{safe_prefix}_{int(time.time())}.png"


def _normalize_fill_state(status_text: str) -> FillState:
    text = (status_text or "").strip().lower()
    if any(token in text for token in ("partial", "part fill", "partial fill")):
        return FillState.PARTIALLY_FILLED
    if any(token in text for token in ("filled", "executed", "complete", "matched")):
        return FillState.FILLED
    if any(token in text for token in ("cancel", "withdrawn")):
        return FillState.CANCELLED
    if any(token in text for token in ("reject", "failed", "invalid")):
        return FillState.REJECTED
    if any(token in text for token in ("pending", "open", "queue", "accepted", "submitted")):
        return FillState.PENDING
    return FillState.UNKNOWN


def _status_from_fill_state(fill_state: FillState, *, submitted: bool, action: ExecutionAction) -> ExecutionStatus:
    if action == ExecutionAction.MODIFY and fill_state == FillState.REJECTED:
        return ExecutionStatus.MODIFY_FAILED
    if fill_state == FillState.FILLED:
        return ExecutionStatus.FILLED
    if fill_state == FillState.PARTIALLY_FILLED:
        return ExecutionStatus.PARTIALLY_FILLED
    if fill_state == FillState.CANCELLED:
        return ExecutionStatus.CANCELLED
    if fill_state == FillState.REJECTED:
        return ExecutionStatus.SUBMIT_FAILED
    if submitted:
        return ExecutionStatus.SUBMITTED_PENDING
    return ExecutionStatus.SUBMIT_FAILED


def _safe_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(raw))
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_int(raw: Any) -> Optional[int]:
    value = _safe_float(raw)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


class TMSBrowserExecutor:
    """Playwright-backed browser executor for TMS workflows."""

    def __init__(self, settings: Optional[TMSSettings] = None):
        self.settings = settings or load_tms_settings()
        self._browser_lock = threading.RLock()
        self._shared_playwright = None
        self._shared_context = None
        self._shared_page = None
        self._shared_headless: Optional[bool] = None

    def _launch_context(self, *, headless: Optional[bool] = None):
        sync_playwright, _, _ = _import_playwright()
        playwright = sync_playwright().start()
        browser_name = "chromium"
        launcher = getattr(playwright, browser_name)
        kwargs: Dict[str, Any] = {
            "user_data_dir": str(self.settings.profile_dir),
            "headless": self.settings.headless if headless is None else bool(headless),
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ],
        }
        if self.settings.browser == "chrome":
            kwargs["channel"] = "chrome"
        try:
            context = launcher.launch_persistent_context(**kwargs)
        except Exception as exc:
            try:
                playwright.stop()
            except Exception:
                pass
            detail = " ".join(str(exc).split())
            lowered = detail.lower()
            if (
                "processsingleton" in lowered
                or "singletonlock" in lowered
                or "profile is already in use" in lowered
            ):
                raise RuntimeError("TMS browser profile is already in use") from exc
            raise
        return playwright, context

    def _open_page(self, context, preferred_page=None):
        page = preferred_page
        try:
            if page is not None and page.is_closed():
                page = None
        except Exception:
            page = None
        if page is None:
            for existing in context.pages:
                try:
                    host = urlparse(existing.url or "").netloc.lower()
                except Exception:
                    host = ""
                if "nepsetms.com.np" in host:
                    page = existing
                    break
        if page is None:
            page = context.new_page()
        page.goto(self.settings.base_url, wait_until="domcontentloaded", timeout=10000)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            # TMS keeps background requests open on the dashboard; DOM content is enough.
            pass
        return page

    def _resolved_headless(self, headless: Optional[bool] = None) -> bool:
        return self.settings.headless if headless is None else bool(headless)

    def _should_reuse_browser(self, headless: Optional[bool] = None) -> bool:
        return not self._resolved_headless(headless)

    def _close_shared_browser(self) -> None:
        try:
            if self._shared_context is not None:
                self._shared_context.close()
        except Exception:
            pass
        finally:
            self._shared_context = None
            self._shared_page = None
        try:
            if self._shared_playwright is not None:
                self._shared_playwright.stop()
        except Exception:
            pass
        finally:
            self._shared_playwright = None
            self._shared_headless = None

    def close(self) -> None:
        with self._browser_lock:
            self._close_shared_browser()

    def _acquire_browser(self, *, headless: Optional[bool] = None):
        resolved_headless = self._resolved_headless(headless)
        if not self._should_reuse_browser(resolved_headless):
            playwright, context = self._launch_context(headless=resolved_headless)
            page = self._open_page(context)
            return playwright, context, page, False

        if self._shared_context is None or self._shared_playwright is None or self._shared_headless != resolved_headless:
            self._close_shared_browser()
            self._shared_playwright, self._shared_context = self._launch_context(headless=resolved_headless)
            self._shared_headless = resolved_headless
            self._shared_page = None

        try:
            page = self._open_page(self._shared_context, preferred_page=self._shared_page)
        except Exception:
            self._close_shared_browser()
            self._shared_playwright, self._shared_context = self._launch_context(headless=resolved_headless)
            self._shared_headless = resolved_headless
            page = self._open_page(self._shared_context)
        self._shared_page = page
        return self._shared_playwright, self._shared_context, page, True

    def _run_with_page(self, callback: Callable[[Any], Any], *, headless: Optional[bool] = None):
        with self._browser_lock:
            playwright = None
            context = None
            shared = False
            try:
                playwright, context, page, shared = self._acquire_browser(headless=headless)
                return callback(page)
            finally:
                if not shared:
                    try:
                        if context is not None:
                            context.close()
                    finally:
                        if playwright is not None:
                            playwright.stop()

    def _try_locator(self, page, selector: str):
        locator = page.locator(selector).first
        locator.wait_for(state="attached", timeout=900)
        return locator

    def _first_locator(self, page, key: str):
        for selector in self.settings.selectors.get(key, []):
            try:
                return self._try_locator(page, selector)
            except Exception:
                continue
        return None

    def _maybe_click(self, page, key: str) -> bool:
        locator = self._first_locator(page, key)
        if locator is None:
            return False
        locator.click(timeout=2000)
        human_pause(self.settings)
        return True

    def _capture(self, page, prefix: str) -> str:
        target = _screenshot_file(self.settings.screenshot_dir, prefix)
        page.screenshot(path=str(target), full_page=True)
        return str(target)

    def _route_url(self, path: str) -> str:
        parsed = urlparse(self.settings.base_url)
        clean_path = str(path or "").strip()
        if clean_path.startswith("http://") or clean_path.startswith("https://"):
            return clean_path
        if not clean_path.startswith("/"):
            clean_path = f"/{clean_path}"
        return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", "", ""))

    def _goto(self, page, path: str):
        try:
            page.goto(self._route_url(path), wait_until="domcontentloaded", timeout=10000)
        except Exception:
            # Navigation can time out while the already-loaded SPA is still usable.
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        human_pause(self.settings, minimum=0.2, maximum=0.6)
        return page

    def _body_text(self, page) -> str:
        try:
            return page.locator("body").inner_text(timeout=3000)
        except Exception:
            return ""

    def _all_tables(self, page) -> List[List[List[str]]]:
        payload: List[List[List[str]]] = []
        try:
            tables = page.locator("table").all()
        except Exception:
            return payload
        for table in tables:
            try:
                rows = table.locator("tr").all()
            except Exception:
                continue
            table_rows: List[List[str]] = []
            for row in rows:
                try:
                    cells = [str(cell).strip() for cell in row.locator("th, td").all_inner_texts()]
                except Exception:
                    continue
                if any(cells):
                    table_rows.append(cells)
            if table_rows:
                payload.append(table_rows)
        return payload

    def _best_table(self, page, *, keywords: Iterable[str]) -> List[List[str]]:
        best: List[List[str]] = []
        best_score = -1
        lower_keywords = [str(item).strip().lower() for item in keywords if str(item).strip()]
        for table in self._all_tables(page):
            header_blob = " ".join(" ".join(row).lower() for row in table[:2])
            score = len(table)
            for keyword in lower_keywords:
                if keyword in header_blob:
                    score += 10
            if score > best_score:
                best = table
                best_score = score
        return best

    def _rows_to_records(self, rows: List[List[str]]) -> List[Dict[str, Any]]:
        if len(rows) < 2:
            return []
        headers = [_normalize_key(cell) or f"col_{idx + 1}" for idx, cell in enumerate(rows[0])]
        records: List[Dict[str, Any]] = []
        for row in rows[1:]:
            if not any(str(cell).strip() for cell in row):
                continue
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            payload: Dict[str, Any] = {}
            for idx, header in enumerate(headers):
                payload[str(header)] = str(padded[idx]).strip() if idx < len(padded) else ""
            records.append(payload)
        return records

    def _extract_numeric_label(self, text: str, *labels: str) -> Optional[float]:
        for label in labels:
            pattern = rf"{re.escape(label)}(?:\s*\(NPR\))?\s*[:\-]?\s*(?:NPR\.?\s*)?([0-9,]+(?:\.\d+)?)"
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return _safe_float(match.group(1))
        return None

    def _extract_string_label(self, text: str, *labels: str) -> Optional[str]:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[:\-]?\s*([^\n]+)"
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                value = str(match.group(1)).strip()
                if value:
                    return value
        return None

    def _has_no_records(self, text: str) -> bool:
        haystack = str(text or "").lower()
        return "no records available" in haystack or "no data available" in haystack

    def _build_health_payload(
        self,
        *,
        status: SessionStatus,
        selector_health: Dict[str, bool],
        last_sync_utc: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> Dict[str, Any]:
        pages_total = len(selector_health)
        pages_ok = sum(1 for ok in selector_health.values() if ok)
        selector_health_pct = round((pages_ok / pages_total) * 100.0, 1) if pages_total else 0.0
        return {
            "snapshot_time_utc": utc_now_iso(),
            "ready": bool(status.ready),
            "login_required": bool(status.login_required),
            "current_url": status.current_url,
            "detail": error_detail or status.detail,
            "dashboard_text": status.dashboard_text,
            "screenshot_path": status.screenshot_path,
            "selector_health": selector_health,
            "pages_ok": pages_ok,
            "pages_total": pages_total,
            "selector_health_pct": selector_health_pct,
            "last_sync_utc": last_sync_utc,
        }

    def _fetch_dashboard_account(self, page) -> Dict[str, Any]:
        self._goto(page, "/tms/client/dashboard")
        text = self._body_text(page)
        trade_summary = {
            "total_turnover": self._extract_numeric_label(text, "Total Turnover"),
            "traded_shares": _safe_int(self._extract_numeric_label(text, "Traded Shares")),
            "transactions": _safe_int(self._extract_numeric_label(text, "Transactions")),
            "scrips_traded": _safe_int(self._extract_numeric_label(text, "Scrips Traded")),
            "buy_count": _safe_int(self._extract_numeric_label(text, "Buy Count")),
            "sell_count": _safe_int(self._extract_numeric_label(text, "Sell Count")),
        }
        collateral_summary = {
            "collateral_amount": self._extract_numeric_label(text, "Collateral Amount"),
            "collateral_utilized": self._extract_numeric_label(text, "Collateral Utilized"),
            "collateral_available": self._extract_numeric_label(text, "Collateral Available"),
            "payable_amount": self._extract_numeric_label(text, "Payable Amount"),
            "receivable_amount": self._extract_numeric_label(text, "Receivable Amount"),
            "net_receivable_amount": self._extract_numeric_label(text, "Net Receivable Amount"),
            "net_payable_amount": self._extract_numeric_label(text, "Net Payable Amount"),
        }
        dp_holding_summary = {
            "last_sync": self._extract_string_label(text, "Last Sync"),
            "holdings_count": _safe_int(self._extract_numeric_label(text, "Total No. of Holdings")),
            "total_amount_cp": self._extract_numeric_label(text, "Total Amount as of CP"),
        }
        return {
            "snapshot_time_utc": utc_now_iso(),
            "page_url": page.url,
            "trade_summary": trade_summary,
            "collateral_summary": collateral_summary,
            "dp_holding_summary": dp_holding_summary,
            "body_excerpt": text[:2000],
        }

    def _fetch_funds_snapshot(self, page) -> Dict[str, Any]:
        self._goto(page, "/tms/me/gen-bank/manage-collateral")
        manage_text = self._body_text(page)
        tx_table = self._best_table(page, keywords=("transaction", "collateral", "date", "amount"))
        transactions = self._rows_to_records(tx_table)[:10]

        self._goto(page, "/tms/me/gen-bank/load-fund")
        load_text = self._body_text(page)
        self._goto(page, "/tms/me/gen-bank/fund-withdrawal")
        refund_text = self._body_text(page)
        self._goto(page, "/tms/me/gen-bank/net-settlement-info")
        settlement_text = self._body_text(page)

        return {
            "snapshot_time_utc": utc_now_iso(),
            "cash_collateral_amount": self._extract_numeric_label(manage_text, "Cash Collateral Amount"),
            "cheque_collateral_amount": self._extract_numeric_label(manage_text, "Cheque Collateral Amount"),
            "fund_transfer_amount": self._extract_numeric_label(manage_text, "Fund Transfer Amount"),
            "multiplication_factor": self._extract_numeric_label(manage_text, "Multiplication Factor"),
            "collateral_utilized": self._extract_numeric_label(manage_text, "Utilized Collateral", "Collateral Utilized"),
            "refund_request_amount": self._extract_numeric_label(manage_text, "Refund Request"),
            "collateral_total": self._extract_numeric_label(manage_text, "Total Collateral"),
            "collateral_available": self._extract_numeric_label(manage_text, "Available Collateral"),
            "non_cash_collateral": self._extract_numeric_label(manage_text, "Non Cash Collateral"),
            "top_up_amount": self._extract_numeric_label(manage_text, "Top up Amount"),
            "credit_for_sale": self._extract_numeric_label(manage_text, "Credit for Sale"),
            "available_trading_limit": self._extract_numeric_label(load_text, "Available Trading Limit"),
            "utilized_trading_limit": self._extract_numeric_label(load_text, "Utilized Trading Limit"),
            "total_trading_limit": self._extract_numeric_label(load_text, "Total Trading Limit"),
            "pending_refund_request": self._extract_numeric_label(refund_text, "Pending Refund Request"),
            "max_refund_allowed": self._extract_numeric_label(refund_text, "Max Refund Allowed"),
            "payable_amount": self._extract_numeric_label(settlement_text, "Net Payable Amount", "Payable Amount"),
            "receivable_amount": self._extract_numeric_label(settlement_text, "Net Receivable Amount", "Receivable Amount"),
            "recent_transactions": transactions,
        }

    def _fetch_holdings_snapshot(self, page) -> Dict[str, Any]:
        self._goto(page, "/tms/me/dp-holding")
        text = self._body_text(page)
        table = self._best_table(page, keywords=("symbol", "balance", "ltp", "value"))
        records = self._rows_to_records(table)
        items: List[Dict[str, Any]] = []
        for row in records:
            symbol = str(
                row.get("symbol")
                or row.get("scrip")
                or row.get("scrip_name")
                or row.get("isin")
                or ""
            ).strip().upper()
            if not symbol or symbol in {"NO", "TOTAL"}:
                continue
            item = {
                "symbol": symbol,
                "cds_total_balance": _safe_int(row.get("cds_total_balance") or row.get("total_balance")),
                "cds_free_balance": _safe_int(row.get("cds_free_balance") or row.get("free_balance")),
                "tms_balance": _safe_int(row.get("tms_balance") or row.get("balance")),
                "close_price": _safe_float(row.get("close_price")),
                "ltp": _safe_float(row.get("ltp")),
                "value_as_of_cp": _safe_float(row.get("value_as_of_cp") or row.get("value")),
                "value_as_of_ltp": _safe_float(row.get("value_as_of_ltp")),
                "raw": row,
            }
            items.append(item)
        if not items:
            for match in re.finditer(
                r"^\s*\d+\s+([A-Z]{2,10})\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
                text,
                flags=re.MULTILINE,
            ):
                items.append(
                    {
                        "symbol": match.group(1),
                        "cds_total_balance": _safe_int(match.group(2)),
                        "cds_free_balance": _safe_int(match.group(3)),
                        "tms_balance": _safe_int(match.group(4)),
                        "close_price": _safe_float(match.group(5)),
                        "value_as_of_cp": _safe_float(match.group(6)),
                        "ltp": _safe_float(match.group(7)),
                        "value_as_of_ltp": _safe_float(match.group(8)),
                        "raw": {"line": match.group(0).strip()},
                    }
                )
        total_amount_ltp = sum(float(item.get("value_as_of_ltp") or 0.0) for item in items)
        total_amount_cp = sum(float(item.get("value_as_of_cp") or 0.0) for item in items)
        return {
            "snapshot_time_utc": utc_now_iso(),
            "last_sync": self._extract_string_label(text, "Last Sync"),
            "count": len(items),
            "total_amount_cp": total_amount_cp,
            "total_amount_ltp": total_amount_ltp,
            "items": items,
            "no_records": self._has_no_records(text) and not items,
        }

    def _fetch_book_snapshot(self, page, *, path: str, book_type: str) -> Dict[str, Any]:
        self._goto(page, path)
        text = self._body_text(page)
        keywords = ("order", "status", "price", "qty", "quantity", "trade", "symbol", "scrip")
        table = self._best_table(page, keywords=keywords)
        records = self._rows_to_records(table)
        display_records: List[Dict[str, Any]] = []
        for row in records:
            lowered = " ".join(str(value) for value in row.values()).lower()
            if "no records available" in lowered or "no data available" in lowered:
                continue
            display_records.append(row)
        return {
            "snapshot_time_utc": utc_now_iso(),
            "page_url": page.url,
            "scope": "historic" if "history" in book_type else "daily",
            "book_type": book_type,
            "row_count": len(display_records),
            "no_records": self._has_no_records(text) or not display_records,
            "records": display_records[:50],
            "raw_rows": table[:20] if table else [],
        }

    def _open_member_market_watch(self, page) -> None:
        self._goto(page, "/tms/me/member-market-watch")
        if self._first_locator(page, "login_form") is not None:
            raise RuntimeError("Login required")
        page.wait_for_timeout(1200)
        if page.locator("table.market-watch-table").count() == 0:
            raise RuntimeError("Member market watch page not ready")

    def _watchlist_editor_descriptor(self, page) -> Dict[str, str]:
        payload = page.evaluate(
            """
            () => {
              const modals = Array.from(document.querySelectorAll('.modal'));
              for (const modal of modals) {
                const body = (modal.innerText || '').toUpperCase();
                if (!body.includes('SELECT SECURITIES')) continue;
                const modalId = modal.id || '';
                const trigger = modalId
                  ? document.querySelector(`button[data-target="#${modalId}"]`)
                  : null;
                return {
                  trigger_id: trigger?.id || '',
                  target: trigger?.getAttribute('data-target') || '',
                  modal_id: modalId,
                };
              }
              return null;
            }
            """
        )
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("Watchlist editor modal not found")
        return {
            "trigger_id": str(payload.get("trigger_id") or ""),
            "target": str(payload.get("target") or ""),
            "modal_id": str(payload.get("modal_id") or ""),
        }

    def _open_watchlist_editor(self, page) -> str:
        self._open_member_market_watch(page)
        descriptor = self._watchlist_editor_descriptor(page)
        modal_selector = descriptor["target"] or (f"#{descriptor['modal_id']}" if descriptor["modal_id"] else "")
        if not modal_selector:
            raise RuntimeError("Watchlist editor modal selector missing")
        page.evaluate(
            """({ triggerId, selector }) => {
              let btn = null;
              if (triggerId) btn = document.getElementById(triggerId);
              if (!btn && selector) btn = document.querySelector(`button[data-target="${selector}"]`);
              if (btn) btn.click();
            }""",
            {"triggerId": descriptor["trigger_id"], "selector": modal_selector},
        )
        page.wait_for_timeout(800)
        return modal_selector

    def _watchlist_table_rows(self, page) -> List[Dict[str, Any]]:
        self._open_member_market_watch(page)
        rows = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('table.market-watch-table tbody tr')).map((row) => {
              const cells = Array.from(row.querySelectorAll('td')).map((cell) => (cell.innerText || '').trim());
              const tooltip = (row.querySelector('.tooltipData')?.innerText || '').trim();
              return { cells, tooltip };
            })
            """
        )
        items: List[Dict[str, Any]] = []
        for row in rows or []:
            cells = list((row or {}).get("cells") or [])
            if not cells:
                continue
            symbol = str(cells[0] or "").split()[0].strip().upper()
            if not symbol or not re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", symbol):
                continue
            items.append(
                {
                    "symbol": symbol,
                    "name": str((row or {}).get("tooltip") or "").strip(),
                    "ltp": _safe_float(cells[1] if len(cells) > 1 else None),
                    "high": _safe_float(cells[2] if len(cells) > 2 else None),
                    "low": _safe_float(cells[3] if len(cells) > 3 else None),
                    "open": _safe_float(cells[4] if len(cells) > 4 else None),
                    "close": _safe_float(cells[5] if len(cells) > 5 else None),
                    "change_pct": _safe_float(cells[6] if len(cells) > 6 else None),
                    "raw": {"cells": cells},
                }
            )
        return items

    def _set_watchlist_symbols(self, page, modal_selector: str, desired_symbols: List[str]) -> None:
        payload = page.evaluate(
            """
            ({ modalSelector, desiredSymbols }) => {
              const root = modalSelector && modalSelector.startsWith('#')
                ? document.getElementById(modalSelector.slice(1))
                : document.querySelector(modalSelector);
              if (!root) return { ok: false, detail: 'modal not found' };

              const norm = (value) => (value || '').replace(/\\s+/g, ' ').trim().toUpperCase();
              const wanted = Array.from(new Set((desiredSymbols || []).map(norm).filter(Boolean)));
              const lists = Array.from(root.querySelectorAll('ul'));
              if (lists.length < 2) return { ok: false, detail: 'picklist lists not found' };

                      const left = lists[0];
                      const right = lists[1];
                      const buttons = Array.from(root.querySelectorAll('button'));
                      const moveRight = buttons.find((btn) => (btn.className || '').includes('point-right'));
                      const moveLeft = buttons.find((btn) => (btn.className || '').includes('point-left'));
                      if (!moveRight || !moveLeft) return { ok: false, detail: 'picklist move buttons not found' };

                      const moveOne = (listEl, symbol, directionButton) => {
                        const item = Array.from(listEl.querySelectorAll('li')).find((li) => norm(li.innerText) === symbol);
                        if (!item) return false;
                        item.click();
                        directionButton.click();
                        return true;
                      };

                      const removed = [];
                      const moved = [];
                      const missing = [];
                      const currentRight = () => Array.from(right.querySelectorAll('li')).map((li) => norm(li.innerText)).filter(Boolean);

                      for (const symbol of currentRight()) {
                        if (wanted.includes(symbol)) continue;
                        if (moveOne(right, symbol, moveLeft)) removed.push(symbol);
                      }

                      for (const symbol of wanted) {
                        if (currentRight().includes(symbol)) {
                          continue;
                        }
                        if (moveOne(left, symbol, moveRight)) moved.push(symbol);
                        else missing.push(symbol);
                      }

                      return {
                        ok: missing.length === 0,
                        moved,
                        removed,
                        missing,
                        rightSymbols: currentRight(),
                      };
                    }
            """,
            {"modalSelector": modal_selector, "desiredSymbols": desired_symbols},
        )
        if not isinstance(payload, dict) or not payload.get("ok"):
            detail = ""
            if isinstance(payload, dict):
                detail = str(payload.get("detail") or "")
                missing = list(payload.get("missing") or [])
                if missing:
                    detail = f"{detail}; missing={','.join(str(item) for item in missing)}".strip("; ")
            raise RuntimeError(detail or "Failed to stage TMS watchlist symbols")

    def _submit_watchlist_symbols(self, page, modal_selector: str) -> None:
        payload = page.evaluate(
            """
            (modalSelector) => {
              const root = modalSelector && modalSelector.startsWith('#')
                ? document.getElementById(modalSelector.slice(1))
                : document.querySelector(modalSelector);
              if (!root) return false;
              const button = Array.from(root.querySelectorAll('button')).find((item) => /update/i.test((item.innerText || '').trim()));
              if (!button) return false;
              button.click();
              return true;
            }
            """,
            modal_selector,
        )
        if not payload:
            raise RuntimeError("Watchlist update button not found")
        human_pause(self.settings, minimum=0.8, maximum=1.4)

    def _fetch_watchlist_snapshot(self, page) -> Dict[str, Any]:
        items = self._watchlist_table_rows(page)
        return {
            "snapshot_time_utc": utc_now_iso(),
            "page_url": page.url,
            "count": len(items),
            "items": items,
            "symbols": [str(item["symbol"]) for item in items],
            "no_records": not items,
            "raw_rows": [dict(item.get("raw") or {}) for item in items[:20]],
        }

    def _member_watchlist_row_index(self, page, symbol: str) -> int:
        self._open_member_market_watch(page)
        row_index = page.evaluate(
            """
            (targetSymbol) => {
              const rows = Array.from(document.querySelectorAll('table.market-watch-table tbody tr'));
              return rows.findIndex((row) => {
                const firstCell = row.querySelector('td');
                const text = ((firstCell && firstCell.innerText) || '').trim().toUpperCase();
                return text === String(targetSymbol || '').trim().toUpperCase();
              });
            }
            """,
            symbol,
        )
        if row_index is None:
            return -1
        return int(row_index)

    def _add_watchlist_symbol_via_member_page(self, page, symbol: str) -> None:
        self._open_member_market_watch(page)
        input_locator = page.locator("ng-select#companyNames input[role='combobox']").first
        input_locator.click()
        page.wait_for_timeout(250)
        input_locator.fill(symbol)
        option_index = -1
        for _ in range(8):
            page.wait_for_timeout(350)
            option_index_raw = page.evaluate(
                """
                (targetSymbol) => {
                  const prefix = String(targetSymbol || '').trim().toUpperCase();
                  const options = Array.from(document.querySelectorAll('.ng-dropdown-panel .ng-option'));
                  return options.findIndex((option) => {
                    const text = (option.innerText || '').trim().toUpperCase();
                    return text.startsWith(prefix + ' ') || text.startsWith(prefix + '(');
                  });
                }
                """,
                symbol,
            )
            option_index = -1 if option_index_raw is None else int(option_index_raw)
            if option_index >= 0:
                break
        if option_index < 0:
            raise RuntimeError(f"{symbol} not found in TMS symbol picker")
        page.locator(".ng-dropdown-panel .ng-option").nth(option_index).click()
        page.wait_for_timeout(250)

        add_button = page.locator("button.add-security-button").first
        if add_button.count() == 0:
            raise RuntimeError("TMS Add Security button not found")
        add_button.click()
        human_pause(self.settings, minimum=0.8, maximum=1.4)

    def _remove_watchlist_symbol_via_member_page(self, page, symbol: str) -> None:
        self._open_member_market_watch(page)
        row_locator = page.locator("table.market-watch-table tbody tr", has_text=symbol).first
        if row_locator.count() == 0:
            raise RuntimeError(f"{symbol} row not found in TMS watchlist")
        row_locator.click(button="right")
        page.wait_for_timeout(300)

        delete_item = page.locator("[role='menuitem']", has_text="Delete Security").first
        if delete_item.count() == 0:
            raise RuntimeError("TMS Delete Security menu item not found")
        delete_item.click()
        human_pause(self.settings, minimum=0.8, maximum=1.4)

    def fetch_watchlist_snapshot(self) -> Dict[str, Any]:
        def _callback(page):
            if self._first_locator(page, "login_form") is not None:
                raise RuntimeError("Login required")
            return self._fetch_watchlist_snapshot(page)

        return self._run_with_page(_callback)

    def add_watchlist_symbol(self, symbol: str) -> Dict[str, Any]:
        target_symbol = str(symbol or "").strip().upper()
        if not target_symbol:
            raise ValueError("Symbol is required")

        def _callback(page):
            if self._first_locator(page, "login_form") is not None:
                raise RuntimeError("Login required")
            before = self._fetch_watchlist_snapshot(page)
            current_symbols = [str(item).strip().upper() for item in (before.get("symbols") or []) if str(item).strip()]
            if target_symbol in set(current_symbols):
                return before
            self._add_watchlist_symbol_via_member_page(page, target_symbol)
            after = self._fetch_watchlist_snapshot(page)
            after_symbols = {str(item).strip().upper() for item in (after.get("symbols") or []) if str(item).strip()}
            if target_symbol not in after_symbols:
                raise RuntimeError(f"{target_symbol} did not appear in TMS watchlist")
            return after

        return self._run_with_page(_callback)

    def remove_watchlist_symbol(self, symbol: str) -> Dict[str, Any]:
        target_symbol = str(symbol or "").strip().upper()
        if not target_symbol:
            raise ValueError("Symbol is required")

        def _callback(page):
            if self._first_locator(page, "login_form") is not None:
                raise RuntimeError("Login required")
            before = self._fetch_watchlist_snapshot(page)
            current_symbols = [str(item).strip().upper() for item in (before.get("symbols") or []) if str(item).strip()]
            if target_symbol not in set(current_symbols):
                return before
            self._remove_watchlist_symbol_via_member_page(page, target_symbol)
            after = self._fetch_watchlist_snapshot(page)
            after_symbols = {str(item).strip().upper() for item in (after.get("symbols") or []) if str(item).strip()}
            if target_symbol in after_symbols:
                raise RuntimeError(f"{target_symbol} is still present in TMS watchlist")
            return after

        return self._run_with_page(_callback)

    def fetch_monitor_bundle(self) -> Dict[str, Any]:
        selector_health: Dict[str, bool] = {
            "dashboard": False,
            "watchlist": False,
            "funds": False,
            "holdings": False,
            "orders_daily": False,
            "orders_historic": False,
            "trades_daily": False,
            "trades_historic": False,
        }
        try:
            def _callback(page):
                screenshot = self._capture(page, "tms_monitor")
                login_form = self._first_locator(page, "login_form")
                if login_form is not None:
                    status = SessionStatus(
                        ready=False,
                        login_required=True,
                        current_url=page.url,
                        dashboard_text=self._body_text(page)[:250],
                        screenshot_path=screenshot,
                        detail="Login form detected",
                    )
                    return {"health": self._build_health_payload(status=status, selector_health=selector_health)}

                selector_health["dashboard"] = self._first_locator(page, "dashboard_ready") is not None
                account = self._fetch_dashboard_account(page)
                try:
                    watchlist = self._fetch_watchlist_snapshot(page)
                    selector_health["watchlist"] = True
                except Exception as exc:
                    watchlist = {
                        "snapshot_time_utc": utc_now_iso(),
                        "page_url": page.url,
                        "count": 0,
                        "items": [],
                        "symbols": [],
                        "no_records": True,
                        "detail": f"watchlist fetch failed: {exc}",
                    }
                funds = self._fetch_funds_snapshot(page)
                selector_health["funds"] = bool(funds)
                holdings = self._fetch_holdings_snapshot(page)
                selector_health["holdings"] = True
                orders_daily = self._fetch_book_snapshot(page, path="/tms/me/order-book-v3", book_type="orders_daily")
                selector_health["orders_daily"] = True
                orders_historic = self._fetch_book_snapshot(page, path="/tms/me/order-book-history", book_type="orders_historic")
                selector_health["orders_historic"] = True
                trades_daily = self._fetch_book_snapshot(page, path="/tms/me/trade-book", book_type="trades_daily")
                selector_health["trades_daily"] = True
                trades_historic = self._fetch_book_snapshot(page, path="/tms/me/trade-book-history", book_type="trades_historic")
                selector_health["trades_historic"] = True

                last_sync = holdings.get("last_sync") or account.get("dp_holding_summary", {}).get("last_sync")
                health = self._build_health_payload(
                    status=SessionStatus(
                        ready=True,
                        login_required=False,
                        current_url=page.url,
                        dashboard_text=account.get("body_excerpt", "")[:250],
                        screenshot_path=screenshot,
                        detail="TMS monitor bundle fetched",
                    ),
                    selector_health=selector_health,
                    last_sync_utc=last_sync,
                )
                return {
                    "health": health,
                    "account": account,
                    "watchlist": watchlist,
                    "funds": funds,
                    "holdings": holdings,
                    "orders_daily": orders_daily,
                    "orders_historic": orders_historic,
                    "trades_daily": trades_daily,
                    "trades_historic": trades_historic,
                }

            return self._run_with_page(_callback)
        except Exception as exc:
            status = SessionStatus(
                ready=False,
                login_required=False,
                current_url="",
                dashboard_text="",
                screenshot_path=None,
                detail=f"TMS monitor failed: {exc}",
            )
            return {"health": self._build_health_payload(status=status, selector_health=selector_health, error_detail=str(exc))}

    def session_status(self) -> SessionStatus:
        def _callback(page):
            screenshot = self._capture(page, "tms_session")
            login_form = self._first_locator(page, "login_form")
            if login_form is not None:
                return SessionStatus(
                    ready=False,
                    login_required=True,
                    current_url=page.url,
                    screenshot_path=screenshot,
                    detail="Login form detected",
                )
            ready = self._first_locator(page, "dashboard_ready") is not None
            text = page.locator("body").inner_text(timeout=2000)[:250]
            return SessionStatus(
                ready=bool(ready),
                login_required=not bool(ready),
                current_url=page.url,
                dashboard_text=text,
                screenshot_path=screenshot,
                detail="Dashboard ready" if ready else "Dashboard marker not found",
            )

        return self._run_with_page(_callback)

    def open_browser(self) -> None:
        playwright = None
        context = None
        try:
            playwright, context = self._launch_context(headless=False)
            page = self._open_page(context)
            logger.info("TMS browser opened at %s using profile %s", page.url, self.settings.profile_dir)
            print("TMS browser opened. Log in manually in the dedicated profile. Press Ctrl+C to close.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def auto_login(self, username: str, password: str) -> SessionStatus:
        """Automated TMS login: launch headless browser, fill credentials, return session status."""
        playwright = None
        context = None
        try:
            playwright, context = self._launch_context(headless=self.settings.headless)
            page = self._open_page(context)

            login_form = self._first_locator(page, "login_form")
            if login_form is None:
                # Already logged in (persistent session)
                ready = self._first_locator(page, "dashboard_ready") is not None
                return SessionStatus(
                    ready=bool(ready),
                    login_required=False,
                    current_url=page.url,
                    dashboard_text=self._body_text(page)[:250],
                    detail="Already authenticated (session persisted)",
                )

            # Fill username — look for username/client-id input
            username_selectors = [
                "input[name='username']",
                "input[name='clientId']",
                "input[placeholder*='Client']",
                "input[placeholder*='Username']",
                "input[type='text']",
            ]
            filled_user = False
            for sel in username_selectors:
                try:
                    loc = self._try_locator(page, sel)
                    loc.click(timeout=1500)
                    loc.fill(username)
                    filled_user = True
                    human_pause(self.settings)
                    break
                except Exception:
                    continue
            if not filled_user:
                return SessionStatus(
                    ready=False, login_required=True,
                    current_url=page.url,
                    detail="Could not find username field",
                )

            # Fill password
            password_selectors = [
                "input[name='password']",
                "input[type='password']",
            ]
            filled_pass = False
            for sel in password_selectors:
                try:
                    loc = self._try_locator(page, sel)
                    loc.click(timeout=1500)
                    loc.fill(password)
                    filled_pass = True
                    human_pause(self.settings)
                    break
                except Exception:
                    continue
            if not filled_pass:
                return SessionStatus(
                    ready=False, login_required=True,
                    current_url=page.url,
                    detail="Could not find password field",
                )

            # Click login/submit button
            login_btn_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "button:has-text('Login')",
                "button:has-text('Sign In')",
                "button:has-text('Log In')",
            ]
            clicked = False
            for sel in login_btn_selectors:
                try:
                    loc = self._try_locator(page, sel)
                    loc.click(timeout=2000)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                # Try pressing Enter as fallback
                page.keyboard.press("Enter")

            human_pause(self.settings, minimum=2.0, maximum=4.0)

            # Check result
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            error = self._first_locator(page, "error_text")
            if error is not None:
                err_text = error.inner_text(timeout=1000).strip()
                return SessionStatus(
                    ready=False, login_required=True,
                    current_url=page.url,
                    detail=f"Login failed: {err_text[:100]}",
                )

            login_still = self._first_locator(page, "login_form")
            if login_still is not None:
                return SessionStatus(
                    ready=False, login_required=True,
                    current_url=page.url,
                    detail="Login form still present after submit — check credentials",
                )

            ready = self._first_locator(page, "dashboard_ready") is not None
            return SessionStatus(
                ready=bool(ready),
                login_required=False,
                current_url=page.url,
                dashboard_text=self._body_text(page)[:250],
                detail="Login successful" if ready else "Login submitted, dashboard not confirmed",
            )
        except Exception as exc:
            return SessionStatus(
                ready=False, login_required=True,
                current_url="",
                detail=f"Auto-login error: {exc}",
            )
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def _prepare_order_page(self, page, action: ExecutionAction) -> None:
        self._maybe_click(page, "order_nav")
        if action == ExecutionAction.BUY:
            self._maybe_click(page, "buy_tab")
        elif action == ExecutionAction.SELL:
            self._maybe_click(page, "sell_tab")
        human_pause(self.settings)
        locator = self._first_locator(page, "order_form")
        if locator is None:
            raise RuntimeError("TMS order form not found")

    def _fill_text_input(self, page, key: str, value: str) -> None:
        locator = self._first_locator(page, key)
        if locator is None:
            raise RuntimeError(f"TMS field not found: {key}")
        locator.click(timeout=1500)
        try:
            locator.fill("")
        except Exception:
            pass
        locator.fill(str(value))
        human_pause(self.settings)

    def submit_order(self, intent: ExecutionIntent) -> ExecutionResult:
        playwright = None
        context = None
        submitted = False
        screenshot = None
        try:
            playwright, context = self._launch_context(headless=self.settings.headless)
            page = self._open_page(context)
            if self._first_locator(page, "login_form") is not None:
                screenshot = self._capture(page, f"{intent.intent_id}_login_required")
                return ExecutionResult(
                    intent_id=intent.intent_id,
                    status=ExecutionStatus.SUBMIT_FAILED,
                    submitted=False,
                    fill_state=FillState.UNKNOWN,
                    status_text="Session expired - relogin required",
                    screenshot_path=screenshot,
                    dom_error_text="login_required",
                )

            self._prepare_order_page(page, intent.action)
            self._fill_text_input(page, "symbol_input", intent.symbol)
            self._fill_text_input(page, "quantity_input", str(intent.quantity))
            self._fill_text_input(page, "price_input", f"{float(intent.limit_price or 0.0):.2f}")
            screenshot = self._capture(page, f"{intent.intent_id}_pre_submit")

            button_key = "submit_buy" if intent.action == ExecutionAction.BUY else "submit_sell"
            submit_btn = self._first_locator(page, button_key)
            if submit_btn is None:
                raise RuntimeError(f"TMS submit button not found for {intent.action}")
            submit_btn.click(timeout=2500)
            submitted = True
            human_pause(self.settings, minimum=0.6, maximum=1.4)
            self._maybe_click(page, "confirm_submit")
            human_pause(self.settings, minimum=0.8, maximum=1.6)

            success = self._first_locator(page, "success_text")
            error = self._first_locator(page, "error_text")
            screenshot = self._capture(page, f"{intent.intent_id}_post_submit")
            broker_order_ref = None
            status_text = ""
            if success is not None:
                status_text = success.inner_text(timeout=1000).strip()
            elif error is not None:
                status_text = error.inner_text(timeout=1000).strip()
            ref_locator = self._first_locator(page, "order_reference")
            if ref_locator is not None:
                try:
                    broker_order_ref = ref_locator.inner_text(timeout=1000).strip()
                except Exception:
                    broker_order_ref = None

            if not status_text:
                status_text = "Submitted to TMS" if submitted else "Submit failed"
            fill_state = _normalize_fill_state(status_text)
            uncertain = submitted and success is None and error is None and not broker_order_ref
            status = (
                ExecutionStatus.FROZEN
                if uncertain and self.settings.failsafe_freeze_on_uncertain_submit
                else _status_from_fill_state(fill_state, submitted=submitted, action=intent.action)
            )
            return ExecutionResult(
                intent_id=intent.intent_id,
                status=status,
                submitted=submitted,
                fill_state=fill_state if fill_state != FillState.UNKNOWN else FillState.PENDING,
                status_text=status_text or ("Uncertain submission state" if uncertain else "Submitted"),
                broker_order_ref=broker_order_ref,
                observed_price=float(intent.limit_price or 0.0),
                observed_qty=int(intent.quantity),
                screenshot_path=screenshot,
                dom_error_text=None if success is not None or broker_order_ref else (status_text or None),
                submitted_at=utc_now_iso() if submitted else None,
                uncertain_submission=uncertain,
                payload={"url": page.url},
            )
        except Exception as exc:
            return ExecutionResult(
                intent_id=intent.intent_id,
                status=ExecutionStatus.SUBMIT_FAILED,
                submitted=submitted,
                fill_state=FillState.UNKNOWN,
                status_text="TMS browser submission failed",
                screenshot_path=screenshot,
                dom_error_text=str(exc),
                submitted_at=utc_now_iso() if submitted else None,
                uncertain_submission=bool(submitted),
            )
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def _find_row_button(self, page, order_ref: str, button_text: str):
        try:
            row = page.locator(f"tr:has-text('{order_ref}')").first
            row.wait_for(state="attached", timeout=2000)
            return row.locator(f"button:has-text('{button_text}')").first
        except Exception:
            return None

    def cancel_order(self, intent: ExecutionIntent) -> ExecutionResult:
        return self._change_order(intent, button_text="Cancel", action_status=ExecutionStatus.CANCELLED)

    def modify_order(self, intent: ExecutionIntent) -> ExecutionResult:
        return self._change_order(intent, button_text="Modify", action_status=ExecutionStatus.SUBMITTED_PENDING)

    def _change_order(self, intent: ExecutionIntent, *, button_text: str, action_status: ExecutionStatus) -> ExecutionResult:
        playwright = None
        context = None
        screenshot = None
        try:
            playwright, context = self._launch_context(headless=self.settings.headless)
            page = self._open_page(context)
            if self._first_locator(page, "login_form") is not None:
                screenshot = self._capture(page, f"{intent.intent_id}_login_required")
                return ExecutionResult(
                    intent_id=intent.intent_id,
                    status=ExecutionStatus.SUBMIT_FAILED,
                    submitted=False,
                    fill_state=FillState.UNKNOWN,
                    status_text="Session expired - relogin required",
                    screenshot_path=screenshot,
                    dom_error_text="login_required",
                )
            if not self._maybe_click(page, "order_book_nav"):
                raise RuntimeError("TMS order book not found")
            target_ref = intent.target_order_ref or intent.broker_order_ref
            if not target_ref:
                raise RuntimeError("Target order reference missing")
            button = self._find_row_button(page, target_ref, button_text)
            if button is None:
                raise RuntimeError(f"{button_text} button not found for order {target_ref}")
            button.click(timeout=2000)
            human_pause(self.settings)
            if intent.action == ExecutionAction.MODIFY:
                if intent.quantity:
                    self._fill_text_input(page, "quantity_input", str(intent.quantity))
                if intent.limit_price is not None:
                    self._fill_text_input(page, "price_input", f"{float(intent.limit_price):.2f}")
            self._maybe_click(page, "confirm_submit")
            screenshot = self._capture(page, f"{intent.intent_id}_{intent.action}")
            status_text = f"{button_text} submitted"
            fill_state = FillState.CANCELLED if intent.action == ExecutionAction.CANCEL else FillState.PENDING
            return ExecutionResult(
                intent_id=intent.intent_id,
                status=action_status,
                submitted=True,
                fill_state=fill_state,
                status_text=status_text,
                broker_order_ref=target_ref,
                observed_price=float(intent.limit_price or 0.0) if intent.limit_price is not None else None,
                observed_qty=int(intent.quantity) if intent.quantity else None,
                screenshot_path=screenshot,
                submitted_at=utc_now_iso(),
            )
        except Exception as exc:
            return ExecutionResult(
                intent_id=intent.intent_id,
                status=ExecutionStatus.MODIFY_FAILED if intent.action == ExecutionAction.MODIFY else ExecutionStatus.SUBMIT_FAILED,
                submitted=False,
                fill_state=FillState.UNKNOWN,
                status_text=f"{button_text} failed",
                screenshot_path=screenshot,
                dom_error_text=str(exc),
            )
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def _extract_table_rows(self, page, table_key: str) -> List[List[str]]:
        locator = self._first_locator(page, table_key)
        if locator is None:
            return []
        rows = locator.locator("tr").all()
        payload: List[List[str]] = []
        for row in rows:
            try:
                cells = [cell.strip() for cell in row.locator("th, td").all_inner_texts()]
            except Exception:
                continue
            if cells:
                payload.append(cells)
        return payload

    def fetch_orders(self) -> List[Dict[str, Any]]:
        playwright = None
        context = None
        try:
            playwright, context = self._launch_context(headless=self.settings.headless)
            page = self._open_page(context)
            if self._first_locator(page, "login_form") is not None:
                return []
            if not self._maybe_click(page, "order_book_nav"):
                return []
            rows = self._extract_table_rows(page, "orders_table")
            snapshots: List[Dict[str, Any]] = []
            for cells in rows[1:]:
                text = " | ".join(cells)
                broker_order_ref = next((cell for cell in cells if re.search(r"\d{4,}", cell)), "")
                symbol = next((cell for cell in cells if re.fullmatch(r"[A-Z]{2,10}", cell.strip())), "")
                action = "sell" if "sell" in text.lower() else "buy" if "buy" in text.lower() else ""
                quantity = next((int(re.sub(r"[^\d]", "", cell)) for cell in cells if re.search(r"\d", cell)), 0)
                price = 0.0
                for cell in cells:
                    cleaned = re.sub(r"[^0-9.]", "", cell)
                    if cleaned.count(".") <= 1 and cleaned:
                        try:
                            candidate = float(cleaned)
                        except ValueError:
                            continue
                        if candidate > 1:
                            price = candidate
                            break
                status_text = text
                snapshots.append(
                    {
                        "broker_order_ref": broker_order_ref,
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "price": price,
                        "status_text": status_text,
                        "fill_state": str(_normalize_fill_state(status_text)),
                    }
                )
            return snapshots
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def fetch_positions(self) -> List[Dict[str, Any]]:
        playwright = None
        context = None
        try:
            playwright, context = self._launch_context(headless=self.settings.headless)
            page = self._open_page(context)
            if self._first_locator(page, "login_form") is not None:
                return []
            if not self._maybe_click(page, "positions_nav"):
                return []
            rows = self._extract_table_rows(page, "positions_table")
            snapshots: List[Dict[str, Any]] = []
            for cells in rows[1:]:
                symbol = next((cell for cell in cells if re.fullmatch(r"[A-Z]{2,10}", cell.strip())), "")
                numeric_values = []
                for cell in cells:
                    cleaned = re.sub(r"[^0-9.]", "", cell)
                    if cleaned and cleaned.count(".") <= 1:
                        try:
                            numeric_values.append(float(cleaned))
                        except ValueError:
                            pass
                quantity = int(numeric_values[0]) if numeric_values else 0
                avg_price = float(numeric_values[1]) if len(numeric_values) > 1 else 0.0
                market_value = float(numeric_values[-1]) if len(numeric_values) > 2 else None
                if symbol:
                    snapshots.append(
                        asdict(
                            PositionSnapshot(
                                symbol=symbol,
                                quantity=quantity,
                                average_price=avg_price,
                                market_value=market_value,
                                raw={"cells": cells},
                            )
                        )
                    )
            return snapshots
        finally:
            try:
                if context is not None:
                    context.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def reconcile(self) -> Dict[str, Any]:
        orders = self.fetch_orders()
        positions = self.fetch_positions()
        if orders:
            save_live_orders(orders, source="tms_reconcile")
        if positions:
            save_live_positions(positions, source="tms_reconcile")
        return {
            "orders": len(orders),
            "positions": len(positions),
            "timestamp_utc": utc_now_iso(),
        }


class LocalTMSExecutionService:
    """Queued local execution service shared by Telegram and strategy automation."""

    def __init__(
        self,
        settings: Optional[TMSSettings] = None,
        *,
        snapshot_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        result_callback: Optional[Callable[[ExecutionIntent, ExecutionResult, str], None]] = None,
    ):
        self.settings = settings or load_tms_settings()
        self.snapshot_provider = snapshot_provider
        self.result_callback = result_callback
        self.executor = TMSBrowserExecutor(self.settings)
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._events: Dict[str, threading.Event] = {}
        self._results: Dict[str, ExecutionResult] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.halt_level = "none"
        self.freeze_reason: Optional[str] = None
        self._last_session_status: Optional[SessionStatus] = None
        self._last_session_check = 0.0
        init_live_audit_db()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._worker, name="TMSExecutionWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._queue.put("")
        self.executor.close()

    def session_status(self, *, force: bool = False) -> SessionStatus:
        now = time.monotonic()
        if not force and self._last_session_status is not None and (now - self._last_session_check) < 30:
            return self._last_session_status
        status = self.executor.session_status()
        self._last_session_status = status
        self._last_session_check = now
        return status

    def set_halt(self, level: str, *, reason: str = "") -> None:
        with self._lock:
            self.halt_level = level
            self.freeze_reason = reason or self.freeze_reason

    def resume(self) -> None:
        with self._lock:
            self.halt_level = "none"
            self.freeze_reason = None

    def is_blocked(self, action: ExecutionAction) -> Optional[str]:
        with self._lock:
            level = self.halt_level
            reason = self.freeze_reason
        if level == "none":
            return None
        if level == "entries" and action == ExecutionAction.BUY:
            return reason or "Live entries halted"
        if level == "all" and action != ExecutionAction.CANCEL:
            return reason or "All live actions halted"
        return None

    def _snapshot(self) -> Dict[str, Any]:
        if self.snapshot_provider is None:
            return {}
        return dict(self.snapshot_provider() or {})

    def validate_intent(self, intent: ExecutionIntent) -> Tuple[bool, str]:
        if not self.settings.enabled or self.settings.mode not in {"live", "dual"}:
            return False, "Live execution is disabled"
        blocked = self.is_blocked(intent.action)
        if blocked:
            return False, blocked
        if intent.action in {ExecutionAction.BUY, ExecutionAction.SELL}:
            if intent.quantity <= 0:
                return False, "Quantity must be positive"
            if intent.limit_price is None or float(intent.limit_price) <= 0:
                return False, "Explicit limit price required"
        if intent.action == ExecutionAction.MODIFY:
            if intent.quantity < 0:
                return False, "Quantity cannot be negative"
            if intent.limit_price is None or float(intent.limit_price) <= 0:
                return False, "Explicit limit price required"
        if intent.action in {ExecutionAction.CANCEL, ExecutionAction.MODIFY} and not intent.target_order_ref:
            return False, "Target order reference required"

        try:
            from backend.trading.live_trader import is_market_open, now_nst
        except ImportError:
            is_market_open = lambda: True  # type: ignore[assignment]
            now_nst = lambda: None  # type: ignore[assignment]

        if intent.action != ExecutionAction.CANCEL and not is_market_open():
            return False, "Market is closed"
        current_nst = now_nst()
        day_key = current_nst.date().isoformat() if hasattr(current_nst, "date") else utc_now_iso()[:10]
        if count_intents_for_day(day_key) >= self.settings.max_daily_orders:
            return False, "Daily live order cap reached"
        duplicate = find_recent_open_intent(intent.symbol, within_seconds=self.settings.symbol_cooldown_secs)
        if duplicate and duplicate.intent_id != intent.intent_id and intent.action in {ExecutionAction.BUY, ExecutionAction.SELL}:
            return False, f"Recent open live intent exists for {intent.symbol}"
        ltp = fetch_latest_ltp(intent.symbol) if intent.symbol else None
        if intent.action in {ExecutionAction.BUY, ExecutionAction.SELL} and (ltp is None or ltp <= 0):
            return False, f"{intent.symbol} is not tradable right now"
        if ltp and intent.limit_price is not None:
            deviation_pct = abs((float(intent.limit_price) / float(ltp)) - 1.0) * 100.0
            if deviation_pct > self.settings.max_price_deviation_pct:
                return False, f"Limit price deviates {deviation_pct:.2f}% from LTP"
        if intent.limit_price is not None and (float(intent.limit_price) * int(intent.quantity)) > self.settings.max_order_notional:
            return False, "Order exceeds max notional"

        snapshot = self._snapshot()
        cash = float(snapshot.get("cash") or 0.0)
        positions = dict(snapshot.get("positions") or {})
        max_positions = int(snapshot.get("max_positions") or 0)
        if intent.action == ExecutionAction.BUY:
            notional = float(intent.limit_price or 0.0) * int(intent.quantity)
            if cash and notional > cash:
                return False, "Insufficient cash for live buy"
            if max_positions and len(positions) >= max_positions:
                return False, "Max positions reached"
            if intent.symbol in positions:
                return False, f"Already holding {intent.symbol}"

        status = self.session_status()
        if not status.ready:
            return False, "Session expired - relogin required"
        return True, "ok"

    def create_intent(self, intent: ExecutionIntent, *, validate: bool = True) -> Tuple[bool, str, ExecutionIntent]:
        if validate:
            ok, detail = self.validate_intent(intent)
            if not ok:
                intent.status = ExecutionStatus.REJECTED_PRETRADE
                intent.last_error = detail
                save_execution_intent(intent)
                log_execution_attempt(intent.intent_id, stage="validate", status="rejected_pretrade", detail=detail)
                return False, detail, intent
        save_execution_intent(intent)
        return True, "ok", intent

    def queue_intent(self, intent_id: str, *, wait: bool = False, timeout: float = 90.0) -> Optional[ExecutionResult]:
        event = threading.Event()
        self._events[intent_id] = event
        self._queue.put(intent_id)
        if not wait:
            return None
        event.wait(timeout=max(1.0, timeout))
        return self._results.get(intent_id)

    def confirm_intent(self, intent_id: str, *, wait: bool = True, timeout: float = 90.0) -> Optional[ExecutionResult]:
        intent = load_execution_intent(intent_id)
        if intent is None:
            return None
        intent.status = ExecutionStatus.QUEUED
        intent.confirmed_at = utc_now_iso()
        save_execution_intent(intent)
        return self.queue_intent(intent.intent_id, wait=wait, timeout=timeout)

    def submit_intent(self, intent: ExecutionIntent, *, wait: bool = False, timeout: float = 90.0) -> Tuple[bool, str, ExecutionIntent, Optional[ExecutionResult]]:
        ok, detail, intent = self.create_intent(intent)
        if not ok:
            return False, detail, intent, None
        if intent.requires_confirmation:
            update_execution_intent(intent.intent_id, status=str(ExecutionStatus.PENDING_CONFIRMATION))
            intent.status = ExecutionStatus.PENDING_CONFIRMATION
            return True, "pending_confirmation", intent, None
        update_execution_intent(intent.intent_id, status=str(ExecutionStatus.QUEUED))
        intent.status = ExecutionStatus.QUEUED
        result = self.queue_intent(intent.intent_id, wait=wait, timeout=timeout)
        return True, "queued", intent, result

    def _worker(self) -> None:
        while not self._stop:
            intent_id = self._queue.get()
            if self._stop:
                break
            if not intent_id:
                continue
            intent = load_execution_intent(intent_id)
            if intent is None:
                continue
            update_execution_intent(intent.intent_id, status=str(ExecutionStatus.SUBMITTING))
            log_execution_attempt(intent.intent_id, stage="queue", status="submitting", detail=f"{intent.action}:{intent.symbol}")
            result = self._execute(intent)
            self._results[intent.intent_id] = result
            event = self._events.get(intent.intent_id)
            if event is not None:
                event.set()
            if self.result_callback is not None:
                try:
                    self.result_callback(intent, result, "submit")
                except Exception as exc:
                    logger.warning("Result callback failed for %s: %s", intent.intent_id, exc)

    def _execute(self, intent: ExecutionIntent) -> ExecutionResult:
        if intent.action in {ExecutionAction.BUY, ExecutionAction.SELL}:
            result = self.executor.submit_order(intent)
        elif intent.action == ExecutionAction.CANCEL:
            result = self.executor.cancel_order(intent)
        else:
            result = self.executor.modify_order(intent)
        if result.uncertain_submission and self.settings.failsafe_freeze_on_uncertain_submit:
            self.set_halt("all", reason="Execution frozen after uncertain TMS submit state")
        save_execution_result(intent, result, source="tms_browser")
        return result

    def reconcile(self) -> Dict[str, Any]:
        summary = self.executor.reconcile()
        orders = load_latest_live_orders(limit=100)
        matched = 0
        for intent in list_execution_intents(
            statuses=[
                str(ExecutionStatus.SUBMITTED_PENDING),
                str(ExecutionStatus.ACCEPTED),
                str(ExecutionStatus.PARTIALLY_FILLED),
            ],
            limit=200,
        ):
            if not intent.broker_order_ref:
                continue
            latest = next((row for row in orders if row.get("broker_order_ref") == intent.broker_order_ref), None)
            if latest is None:
                continue
            fill_state = FillState(str(latest.get("fill_state") or FillState.UNKNOWN))
            status = _status_from_fill_state(fill_state, submitted=True, action=intent.action)
            updates: Dict[str, Any] = {
                "status": str(status),
            }
            if status in {ExecutionStatus.FILLED, ExecutionStatus.CANCELLED, ExecutionStatus.SUBMIT_FAILED, ExecutionStatus.PARTIALLY_FILLED}:
                updates["completed_at"] = utc_now_iso()
            update_execution_intent(intent.intent_id, **updates)
            matched += 1
            if self.result_callback is not None and status in {ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED, ExecutionStatus.CANCELLED}:
                result = ExecutionResult(
                    intent_id=intent.intent_id,
                    status=status,
                    submitted=True,
                    fill_state=fill_state,
                    status_text=str(latest.get("status_text") or status),
                    broker_order_ref=intent.broker_order_ref,
                    observed_price=float(latest.get("price") or (intent.limit_price or 0.0)),
                    observed_qty=int(latest.get("quantity") or intent.quantity),
                    completed_at=utc_now_iso(),
                    payload={"reconciled": True},
                )
                try:
                    self.result_callback(intent, result, "reconcile")
                except Exception as exc:
                    logger.warning("Reconcile callback failed for %s: %s", intent.intent_id, exc)
        summary["matched_intents"] = matched
        record_reconciliation_run("ok", summary)
        return summary


def format_result_lines(intent: ExecutionIntent, result: ExecutionResult) -> List[str]:
    lines = [
        f"Intent: {intent.intent_id}",
        f"Action: {intent.action.upper()} {intent.symbol}",
        f"Status: {result.status}",
        f"Detail: {result.status_text or '-'}",
    ]
    if intent.quantity:
        lines.append(f"Qty: {intent.quantity}")
    if intent.limit_price is not None:
        lines.append(f"Limit: NPR {intent.limit_price:,.2f}")
    if result.broker_order_ref:
        lines.append(f"Order Ref: {result.broker_order_ref}")
    if result.screenshot_path:
        lines.append(f"Screenshot: {result.screenshot_path}")
    if result.dom_error_text:
        lines.append(f"Error: {result.dom_error_text}")
    return lines


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Local TMS browser executor")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("session-status")
    sub.add_parser("open-browser")
    sub.add_parser("reconcile")
    sub.add_parser("watchlist-get")
    wl_add = sub.add_parser("watchlist-add")
    wl_add.add_argument("--symbol", required=True, type=str)
    wl_remove = sub.add_parser("watchlist-remove")
    wl_remove.add_argument("--symbol", required=True, type=str)
    dry = sub.add_parser("dry-run-order")
    dry.add_argument("--file", required=True, type=str)

    args = parser.parse_args()
    settings = load_tms_settings()
    service = LocalTMSExecutionService(settings)

    if args.command == "session-status":
        status = service.session_status(force=True)
        print(json.dumps(asdict(status), indent=2, default=str))
        return 0 if status.ready else 1
    if args.command == "open-browser":
        TMSBrowserExecutor(settings).open_browser()
        return 0
    if args.command == "reconcile":
        print(json.dumps(service.reconcile(), indent=2, default=str))
        return 0
    if args.command == "watchlist-get":
        print(json.dumps(TMSBrowserExecutor(settings).fetch_watchlist_snapshot(), indent=2, default=str))
        return 0
    if args.command == "watchlist-add":
        print(json.dumps(TMSBrowserExecutor(settings).add_watchlist_symbol(args.symbol), indent=2, default=str))
        return 0
    if args.command == "watchlist-remove":
        print(json.dumps(TMSBrowserExecutor(settings).remove_watchlist_symbol(args.symbol), indent=2, default=str))
        return 0
    if args.command == "dry-run-order":
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        intent = ExecutionIntent.from_record(payload)
        ok, detail = service.validate_intent(intent)
        print(json.dumps({"ok": ok, "detail": detail, "intent": intent.to_record()}, indent=2, default=str))
        return 0 if ok else 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
