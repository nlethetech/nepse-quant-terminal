"""Paper order lifecycle + order DataTables + order-ticket form widgets.

Ticket dialog callbacks, the NEPSE same-day rule, order create/cancel/match,
the daily/historic/trades order DataTables, and the order-ticket form helpers.
Live-order tab rendering stays here (reads self._tms_bundle).

State hand-off seam: _match_paper_orders reads worker-marshaled self.md, whose
writer is _do_refresh in TabRefreshMixin (lands the next step). Tests call
_match_paper_orders directly on a __new__ app with md pre-seeded, so it is green
independent of _do_refresh.
"""

import json as _json
import uuid
from pathlib import Path
from typing import Optional

from rich.markup import escape as _escape
from rich.text import Text
from textual import work
from textual.widgets import Button, DataTable, Input, Static

from apps.tui.screens.dialog import ModalDialog
from apps.tui.theme import AMBER, CYAN, DIM, GAIN, GAIN_HI, LOSS, LOSS_HI, WHITE, YELLOW
from backend.quant_pro.dashboard_data import exec_buy, exec_sell, load_port
from backend.agents.agent_analyst import check_trade_approval
from backend.trading.paper_execution import PaperExecutionService
from apps.tui.io.orders_io import (
    _build_sell_holdings_map,
    _paper_filled_orders_for_day,
    _resolve_sell_qty,
)
from apps.tui.io.website_book import describe_result
from apps.tui.io.persistence import (
    _account_dir,
    _account_initial_capital_from_files,
    _tms_health_flag,
    _write_json_locked,
)


class OrderBookMixin:
    def _on_buy(self, result: dict | None) -> None:
        if not result: return
        # Agent veto gate (reads cached analysis only — no Sonnet call)
        approved, reason = check_trade_approval(result["symbol"], "BUY")
        if not approved:
            self._set_status(f"AGENT BLOCKED BUY {result['symbol']}: {reason}")
            return
        if self.trade_mode == "live":
            self._submit_tms_order(result, "BUY", reason)
            return
        try:
            qty = int(str(result["shares"]).strip())
            if qty <= 0:
                raise ValueError
        except ValueError:
            self._set_status(f"Order: Invalid buy quantity for {result['symbol']}")
            return
        price = self._resolve_ticket_price(result["symbol"], result.get("price", ""))
        if price is None and self._website_book is not None and not str(result.get("price", "") or "").strip():
            price = 0.0  # a market order: the website prices it
        if price is None:
            self._set_status(f"Order: No price available for {result['symbol']}")
            return
        slippage = self._parse_slippage(result.get("slippage", ""))
        market = not str(result.get("price", "") or "").strip()
        msg = self._submit_paper_order("BUY", result["symbol"], qty, price, slippage, market=market)
        self._set_status(f"{msg}  |  Agent: {reason[:60]}")
        self.action_tab("orders")

    def _on_sell(self, result: dict | None) -> None:
        if not result: return
        # Agent veto gate
        approved, reason = check_trade_approval(result["symbol"], "SELL")
        if not approved:
            self._set_status(f"AGENT BLOCKED SELL {result['symbol']}: {reason}")
            return
        holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
        try:
            qty = _resolve_sell_qty(result["symbol"], result["shares"], holdings)
        except ValueError as exc:
            self._set_status(f"Order: {exc}")
            return
        price = self._resolve_ticket_price(result["symbol"], result.get("price", ""))
        if price is None and self._website_book is not None and not str(result.get("price", "") or "").strip():
            price = 0.0  # a market order: the website prices it
        if price is None:
            self._set_status(f"Order: No price available for {result['symbol']}")
            return
        slippage = self._parse_slippage(result.get("slippage", ""))
        market = not str(result.get("price", "") or "").strip()
        msg = self._submit_paper_order("SELL", result["symbol"], qty, price, slippage, market=market)
        self._set_status(f"{msg}  |  Agent: {reason[:60]}")
        self.action_tab("orders")

    def _load_paper_orders(self) -> None:
        """Load paper orders from JSON files (or the linked website book)."""
        book = self._website_book
        if book is not None:
            from backend.trading.live_trader import now_nst
            self._paper_orders, self._paper_order_history = book.order_records()
            self._paper_trades_today = _paper_filled_orders_for_day(
                self._paper_order_history,
                now_nst().strftime("%Y-%m-%d"),
            )
            return
        account_id = str(getattr(self, "__dict__", {}).get("_current_account_id", "") or "").strip()
        if account_id:
            account_dir = _account_dir(account_id)
            self.PAPER_ORDERS_FILE = account_dir / "tui_paper_orders.json"
            self.PAPER_ORDER_HISTORY_FILE = account_dir / "tui_paper_order_history.json"
        self._paper_orders = []
        self._paper_order_history = []
        if self.PAPER_ORDERS_FILE.exists():
            try:
                self._paper_orders = _json.loads(self.PAPER_ORDERS_FILE.read_text())
            except Exception:
                self._paper_orders = []
        if self.PAPER_ORDER_HISTORY_FILE.exists():
            try:
                self._paper_order_history = _json.loads(self.PAPER_ORDER_HISTORY_FILE.read_text())
            except Exception:
                self._paper_order_history = []
        from backend.trading.live_trader import now_nst
        self._paper_trades_today = _paper_filled_orders_for_day(
            self._paper_order_history,
            now_nst().strftime("%Y-%m-%d"),
        )

    def _save_paper_orders(self) -> None:
        """Persist paper orders to JSON files."""
        _write_json_locked(Path(self.PAPER_ORDERS_FILE), list(self._paper_orders))
        _write_json_locked(Path(self.PAPER_ORDER_HISTORY_FILE), list(self._paper_order_history))

    def _create_paper_order(self, action: str, symbol: str, qty: int, price: float, slippage: float = 2.0) -> dict:
        """Create a new paper order dict."""
        from backend.trading.live_trader import now_nst
        ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
        order = {
            "id": uuid.uuid4().hex[:12],
            "action": action,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "slippage_pct": slippage,
            "status": "OPEN",
            "filled_qty": 0,
            "fill_price": 0.0,
            "trigger_price": price,
            "created_at": ts,
            "updated_at": ts,
            "day": now_nst().strftime("%Y-%m-%d"),
            "source": "dashboard_tui",
            "reason": "",
        }
        return order

    def _paper_has_same_day_order(self, symbol: str, *, action: str) -> bool:
        from backend.trading.live_trader import now_nst
        sym = str(symbol or "").strip().upper()
        act = str(action or "").strip().upper()
        today = now_nst().strftime("%Y-%m-%d")
        for row in [*self._paper_orders, *self._paper_order_history]:
            if str(row.get("symbol") or "").strip().upper() != sym:
                continue
            if str(row.get("action") or "").strip().upper() != act:
                continue
            row_day = str(row.get("day") or "")[:10] or str(row.get("created_at") or "")[:10]
            if row_day != today:
                continue
            if str(row.get("status") or "").strip().upper() in {"OPEN", "FILLED"}:
                return True
        return False

    def _paper_position_opened_today(self, symbol: str) -> bool:
        from backend.trading.live_trader import now_nst
        sym = str(symbol or "").strip().upper()
        today = now_nst().strftime("%Y-%m-%d")
        port = load_port()
        if port.empty or "Symbol" not in port.columns:
            return False
        rows = port[port["Symbol"].astype(str).str.upper() == sym]
        if rows.empty:
            return False
        entry_dates = rows.get("Entry_Date")
        if entry_dates is None:
            return False
        return any(str(value)[:10] == today for value in entry_dates.tolist())

    def _paper_same_day_trade_block(self, action: str, symbol: str) -> Optional[str]:
        if self._website_book is not None:
            return None  # the website applies NEPSE's same-day rule to the shared book
        sym = str(symbol or "").strip().upper()
        act = str(action or "").strip().upper()
        if act == "BUY" and self._paper_has_same_day_order(sym, action="SELL"):
            return f"Rejected: NEPSE same-day rule blocks buying {sym} after a sell today."
        if act == "SELL":
            if self._paper_position_opened_today(sym) or self._paper_has_same_day_order(sym, action="BUY"):
                return f"Rejected: NEPSE same-day rule blocks selling {sym} on the same day as a buy."
        return None

    def _submit_paper_order(
        self, action: str, symbol: str, qty: int, price: float, slippage: float = 2.0, *, market: bool = False
    ) -> str:
        """Submit a new paper limit order.

        Linked to nepsequant.com: the order goes to the website's paper book
        instead. No price typed → a market order (fills at the last price while
        NEPSE is open); a typed price → a limit order at that price."""
        book = self._website_book
        if book is not None:
            if not book.can_trade:
                return (
                    "This key can only read your nepsequant.com paper book. To trade it from here, create a key "
                    "with \"Paper trading\" ticked at nepsequant.com/desk/settings and run ./nqt-connect login."
                )
            limit = None if market else float(price)
            self._send_website_order(action, symbol, int(qty), limit)
            kind = "at market" if limit is None else f"limit {limit:,.1f}"
            return f"Sending {action} {symbol} x{qty} {kind} to your nepsequant.com paper book…"
        self._load_paper_orders()
        block_reason = self._paper_same_day_trade_block(action, symbol)
        if block_reason:
            return block_reason
        account_id = str(getattr(self, "__dict__", {}).get("_current_account_id", "") or "account_1")
        account_dir = _account_dir(account_id)
        service = PaperExecutionService(
            account_id,
            account_dir=account_dir,
            initial_capital=_account_initial_capital_from_files(account_dir),
        )
        result = service.submit_order(
            account_id,
            action,
            symbol,
            qty,
            price,
            "dashboard_tui",
            "",
            slippage_pct=slippage,
        )
        self._load_paper_orders()
        self._populate_orders_tab()
        if not result.ok:
            return f"Rejected: {action} {symbol} x{qty} @ {price:,.1f} — {result.message}"
        order = result.order
        return f"Order {order.order_id}: {action} {symbol} x{qty} @ {price:,.1f} slip:{slippage:.1f}% — OPEN"

    @work(thread=True, group="website-book")
    def _send_website_order(self, action: str, symbol: str, qty: int, limit: Optional[float]) -> None:
        result = self._website_book.submit(action, symbol, qty, limit_price=limit)
        self.call_from_thread(self._after_website_change, describe_result(result, action, symbol, qty, limit))

    @work(thread=True, group="website-book")
    def _cancel_website_orders(self, order_id: Optional[str]) -> None:
        book = self._website_book
        if order_id:
            res = book.cancel(order_id)
            msg = f"Order {order_id[:8]} cancelled" if res.get("ok") else f"Cancel {order_id[:8]}: {res.get('message')}"
        else:
            done, problems = book.cancel_all()
            msg = f"Cancelled {done} open orders" + (f" ({len(problems)} could not be cancelled)" if problems else "")
        self.call_from_thread(self._after_website_change, msg)

    def _after_website_change(self, message: str) -> None:
        self._set_status(message)
        try:
            self._append_activity(message)
        except Exception:
            pass
        self._load_paper_orders()
        self._populate_orders_tab()
        self._populate_portfolio_and_risk()
        self._populate_trades_full()

    def _cancel_paper_order(self, order_id: str) -> str:
        """Cancel a specific paper order by ID."""
        if self._website_book is not None:
            self._cancel_website_orders(order_id)
            return f"Cancelling order {order_id[:8]} on your nepsequant.com paper book…"
        from backend.trading.live_trader import now_nst
        for o in self._paper_orders:
            if o["id"] == order_id and o["status"] == "OPEN":
                o["status"] = "CANCELLED"
                o["updated_at"] = now_nst().strftime("%Y-%m-%d %H:%M:%S")
                self._paper_order_history.append(o)
                self._paper_orders = [x for x in self._paper_orders if x["id"] != order_id]
                self._save_paper_orders()
                self._populate_orders_tab()
                return f"Order {order_id} cancelled"
        return f"Order {order_id} not found or already filled"

    def _cancel_all_paper_orders(self) -> str:
        """Cancel all open paper orders."""
        if self._website_book is not None:
            self._cancel_website_orders(None)
            return "Cancelling all open orders on your nepsequant.com paper book…"
        from backend.trading.live_trader import now_nst
        ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
        count = 0
        for o in self._paper_orders:
            if o["status"] == "OPEN":
                o["status"] = "CANCELLED"
                o["updated_at"] = ts
                self._paper_order_history.append(o)
                count += 1
        self._paper_orders = [x for x in self._paper_orders if x["status"] == "OPEN"]
        self._save_paper_orders()
        self._populate_orders_tab()
        return f"Cancelled {count} open orders"

    def _match_paper_orders(self) -> None:
        """Check open orders against current prices and fill if matched."""
        if self._website_book is not None:
            # The website fills the shared book; the refresh already fetched it.
            self._load_paper_orders()
            return
        from backend.trading.live_trader import now_nst
        if str(getattr(self, "__dict__", {}).get("_current_account_id", "") or "").strip():
            self._load_paper_orders()
        if not self._paper_orders:
            return
        # Get current LTPs from market data
        ltps: dict[str, float] = {}
        if hasattr(self, 'md') and not self.md.quotes.empty:
            for _, row in self.md.quotes.iterrows():
                sym = row.get("symbol", "")
                price = row.get("close", 0)
                if sym and price > 0:
                    ltps[sym] = float(price)
        # Also check from gainers/losers for broader coverage
        for df in [self.md.gainers, self.md.losers]:
            if not df.empty:
                for _, row in df.iterrows():
                    sym = row.get("symbol", "")
                    price = row.get("close", 0)
                    if sym and price > 0:
                        ltps[sym] = float(price)

        if not str(getattr(self, "__dict__", {}).get("_current_account_id", "") or "").strip():
            ts = now_nst().strftime("%Y-%m-%d %H:%M:%S")
            filled = []
            for o in self._paper_orders:
                if o["status"] != "OPEN":
                    continue
                sym = o["symbol"]
                if sym not in ltps:
                    continue
                ltp = ltps[sym]
                slip_pct = o.get("slippage_pct", 2.0) / 100.0
                matched = (
                    o["action"] == "BUY" and ltp <= o["price"] * (1 + slip_pct)
                ) or (
                    o["action"] == "SELL" and ltp >= o["price"] * (1 - slip_pct)
                )
                if not matched:
                    continue
                block_reason = self._paper_same_day_trade_block(o["action"], sym)
                if block_reason:
                    o["status"] = "CANCELLED"
                    o["updated_at"] = ts
                    o["reason"] = "same_day_rule"
                    filled.append(o)
                    status_msg = f"Order cancelled: {o['action']} {sym} — {block_reason}"
                    self._append_activity(f"ORDER CANCELLED: {o['action']} {sym} — {block_reason}")
                    self._set_status(status_msg)
                    continue
                o["status"] = "FILLED"
                o["filled_qty"] = o["qty"]
                o["fill_price"] = ltp
                o["updated_at"] = ts
                filled.append(o)
                msg = exec_buy(o["symbol"], str(o["qty"]), str(ltp)) if o["action"] == "BUY" else exec_sell(o["symbol"], str(o["qty"]), str(ltp))
                self._append_activity(f"ORDER FILLED: {o['action']} {sym} x{o['qty']} @ {ltp:,.1f} — {msg}")
            if filled:
                for o in filled:
                    self._paper_order_history.append(o)
                    self._paper_trades_today.append(o)
                self._paper_orders = [x for x in self._paper_orders if x["status"] == "OPEN"]
                self._save_paper_orders()
                self._populate_orders_tab()
                self._populate_portfolio_and_risk()
                self._populate_trades_full()
            self._render_hedge_panel()
            return

        account_id = str(getattr(self, "__dict__", {}).get("_current_account_id", "") or "account_1")
        account_dir = _account_dir(account_id)
        service = PaperExecutionService(
            account_id,
            account_dir=account_dir,
            initial_capital=_account_initial_capital_from_files(account_dir),
        )
        result = service.match_open_orders(
            account_id,
            {sym: {"ltp": ltp, "source": "dashboard_market_snapshot", "age_seconds": 0} for sym, ltp in ltps.items()},
        )

        for order in result.filled_orders:
            self._append_activity(
                f"ORDER FILLED: {order.action} {order.symbol} x{order.filled_qty} @ {order.fill_price:,.1f}"
            )
        for order in result.rejected_orders:
            self._append_activity(
                f"ORDER REJECTED: {order.action} {order.symbol} — {order.risk_result.get('reason', order.reason)}"
            )

        if result.filled_orders or result.rejected_orders:
            self._load_paper_orders()
            self._populate_orders_tab()
            self._populate_portfolio_and_risk()
            self._populate_trades_full()
        self._render_hedge_panel()

    def _populate_orders_tab(self) -> None:
        """Populate the Orders tab DataTables."""
        if self.trade_mode == "live":
            self._populate_orders_tab_live()
            return
        self._populate_orders_tab_paper()

    def _populate_orders_tab_paper(self) -> None:
        """Populate orders tab from paper order book."""
        self._load_paper_orders()
        # -- Daily order book (open orders) --
        dt_daily = self.query_one("#dt-orders-daily", DataTable)
        dt_daily.clear(columns=True)
        dt_daily.add_columns("ID", "ACTION", "SYMBOL", "QTY", "LIMIT", "SLIP%", "BAND", "STATUS", "TIME")
        from backend.trading.live_trader import now_nst
        today = now_nst().strftime("%Y-%m-%d")
        book = self._website_book
        daily = list(self._paper_orders) if book is not None else [o for o in self._paper_orders if o.get("day") == today]
        for o in daily:
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            status_style = YELLOW if o["status"] == "OPEN" else (GAIN if o["status"] == "FILLED" else LOSS)
            slip_pct = float(o.get("slippage_pct", 2.0) or 0.0)
            band_text = self._format_order_band(o["action"], float(o["price"]), slip_pct)
            dt_daily.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["qty"]), style=WHITE),
                Text(f'{o["price"]:,.1f}', style=AMBER),
                Text(f"{slip_pct:.1f}%", style=DIM),
                Text(band_text, style=CYAN),
                Text(o["status"], style=status_style),
                Text(o.get("created_at", "")[-8:], style=DIM),
            )
        if not daily:
            dt_daily.add_row(
                Text("—", style=DIM), Text("—", style=DIM), Text("No open orders", style=DIM),
                Text("—", style=DIM), Text("—", style=DIM), Text("—", style=DIM),
                Text("—", style=DIM), Text("—", style=DIM), Text("—", style=DIM),
            )

        # -- Historic order book (filled + cancelled) --
        dt_hist = self.query_one("#dt-orders-historic", DataTable)
        dt_hist.clear(columns=True)
        dt_hist.add_columns("ID", "ACTION", "SYMBOL", "QTY", "PRICE", "STATUS", "FILL PX", "TIME")
        for o in reversed(self._paper_order_history[-50:]):
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            if o["status"] == "FILLED":
                status_style = GAIN
            elif o["status"] == "CANCELLED":
                status_style = LOSS
            else:
                status_style = YELLOW
            fill_px = f'{o["fill_price"]:,.1f}' if o.get("fill_price") else "—"
            dt_hist.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["qty"]), style=WHITE),
                Text(f'{o["price"]:,.1f}', style=AMBER),
                Text(o["status"], style=status_style),
                Text(fill_px, style=WHITE),
                Text(str(o.get("created_at") or o.get("updated_at") or "")[-8:], style=DIM),
            )

        # -- Today's trades (filled orders) --
        dt_trades = self.query_one("#dt-orders-trades", DataTable)
        dt_trades.clear(columns=True)
        dt_trades.add_columns("ID", "ACTION", "SYMBOL", "QTY", "FILL PRICE", "VALUE", "TIME")
        for o in reversed(self._paper_trades_today):
            act_style = GAIN_HI if o["action"] == "BUY" else LOSS_HI
            value = o.get("fill_price", 0) * o.get("filled_qty", 0)
            dt_trades.add_row(
                Text(o["id"][:8], style=DIM),
                Text(o["action"], style=act_style),
                Text(o["symbol"], style=WHITE),
                Text(str(o["filled_qty"]), style=WHITE),
                Text(f'{o.get("fill_price", 0):,.1f}', style=AMBER),
                Text(f'{value:,.0f}', style=WHITE),
                Text(o.get("updated_at", "")[-8:], style=DIM),
            )

        # -- Update order status bar --
        open_count = len([o for o in self._paper_orders if o["status"] == "OPEN"])
        filled_today = len(self._paper_trades_today)
        bar = self.query_one("#order-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold #ffaf00]ORDER MANAGEMENT[/]  │  "
            f"[{GAIN}]Open: {open_count}[/]  │  "
            f"[{CYAN}]Filled today: {filled_today}[/]  │  "
            + (
                f"[#888888]Mode: PAPER  │  Book: nepsequant.com (shared with the website)"
                + ("" if book.can_trade else "  │  read-only key")
                + (f"  │  [{LOSS}]{_escape(book.error)}[/]" if book.error else "")
                + "[/]"
                if book is not None
                else "[#888888]Mode: PAPER  │  Place orders via the order book[/]"
            )
        ))

    def _populate_orders_tab_live(self) -> None:
        """Populate orders tab from TMS live order book."""
        bundle = self._tms_bundle or {}
        orders_daily = bundle.get("orders_daily", {})
        orders_historic = bundle.get("orders_historic", {})
        trades_daily = bundle.get("trades_daily", {})

        # -- Daily orders --
        dt_daily = self.query_one("#dt-orders-daily", DataTable)
        dt_daily.clear(columns=True)
        daily_records = orders_daily.get("records", [])
        if daily_records:
            cols = list(daily_records[0].keys())[:8]
            dt_daily.add_columns(*[c.upper() for c in cols])
            for rec in daily_records:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_daily.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_daily.add_columns("STATUS")
            dt_daily.add_row(Text("No daily orders", style=DIM))

        # -- Historic orders --
        dt_hist = self.query_one("#dt-orders-historic", DataTable)
        dt_hist.clear(columns=True)
        hist_records = orders_historic.get("records", [])
        if hist_records:
            cols = list(hist_records[0].keys())[:8]
            dt_hist.add_columns(*[c.upper() for c in cols])
            for rec in hist_records[:50]:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_hist.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_hist.add_columns("STATUS")
            dt_hist.add_row(Text("No historic orders", style=DIM))

        # -- Today's trades --
        dt_trades = self.query_one("#dt-orders-trades", DataTable)
        dt_trades.clear(columns=True)
        trade_records = trades_daily.get("records", [])
        if trade_records:
            cols = list(trade_records[0].keys())[:7]
            dt_trades.add_columns(*[c.upper() for c in cols])
            for rec in trade_records:
                vals = [str(rec.get(c, "")) for c in cols]
                dt_trades.add_row(*[Text(v, style=WHITE) for v in vals])
        else:
            dt_trades.add_columns("STATUS")
            dt_trades.add_row(Text("No trades today", style=DIM))

        # -- Status bar --
        health = bundle.get("health", {})
        session_ok = _tms_health_flag(health, "ready")
        open_count = len([r for r in daily_records if "open" in str(r).lower() or "pending" in str(r).lower()])
        bar = self.query_one("#order-status-bar", Static)
        bar.update(Text.from_markup(
            f"[bold #ffaf00]ORDER MANAGEMENT[/]  │  "
            f"[{GAIN if session_ok else LOSS}]TMS: {'CONNECTED' if session_ok else 'DISCONNECTED'}[/]  │  "
            f"[{CYAN}]Daily orders: {len(daily_records)}[/]  │  "
            f"[{YELLOW}]Open: {open_count}[/]  │  "
            f"[#888888]Mode: PAPER[/]"
        ))

    def _on_order_submit(self) -> None:
        """Handle order form submission."""
        try:
            sym = self.query_one("#order-inp-symbol", Input).value.strip().upper()
            qty_str = self.query_one("#order-inp-qty", Input).value.strip()
            price_str = self.query_one("#order-inp-price", Input).value.strip()
            slip_str = self.query_one("#order-inp-slippage", Input).value.strip()
        except Exception:
            return

        if not sym:
            self._set_status("Order: Symbol required")
            return
        action = self._order_action
        allow_all = action == "SELL" and qty_str.strip().lower() == "all"
        if not qty_str or (not allow_all and (not qty_str.isdigit() or int(qty_str) <= 0)):
            self._set_status("Order: Valid quantity required")
            return

        qty = int(qty_str) if not allow_all else 0
        # If no price, use last close from market data
        if not price_str:
            price = self._get_ltp_for_symbol(sym)
            if not price and self._website_book is not None:
                price = 0.0  # a market order: the website prices it
            elif not price:
                self._set_status(f"Order: No price available for {sym}")
                return
            else:
                self.query_one("#order-inp-price", Input).value = f"{price:.1f}"
        else:
            try:
                price = float(price_str.replace(",", ""))
            except ValueError:
                self._set_status("Order: Invalid price")
                return

        # Slippage %
        try:
            slippage = float(slip_str) if slip_str else 2.0
        except ValueError:
            slippage = 2.0

        if action == "SELL":
            holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
            try:
                qty = _resolve_sell_qty(sym, qty_str, holdings)
            except ValueError as exc:
                self._set_status(f"Order: {exc}")
                return

        if self.trade_mode == "live":
            self._submit_live_order(action, sym, qty, price)
        else:
            msg = self._submit_paper_order(action, sym, qty, price, slippage, market=not price_str)
            self._set_status(msg)

        # Clear inputs except slippage (reusable)
        self.query_one("#order-inp-symbol", Input).value = ""
        self.query_one("#order-inp-qty", Input).value = ""
        self.query_one("#order-inp-price", Input).value = ""

    def _get_ltp_for_symbol(self, sym: str) -> Optional[float]:
        """Get last traded price for symbol from market data."""
        if hasattr(self, 'md'):
            for df in [self.md.quotes, self.md.gainers, self.md.losers]:
                if not df.empty and "symbol" in df.columns:
                    match = df[df["symbol"] == sym]
                    if not match.empty:
                        return float(match.iloc[0].get("close", 0))
        return None

    def _refresh_order_action_buttons(self) -> None:
        """Apply active styling to BUY / SELL toggle without mutating labels."""
        try:
            buy_btn = self.query_one("#order-btn-buy", Button)
            sell_btn = self.query_one("#order-btn-sell", Button)
        except Exception:
            return
        if self._order_action == "BUY":
            buy_btn.add_class("order-action-active")
            sell_btn.remove_class("order-action-active")
        else:
            sell_btn.add_class("order-action-active")
            buy_btn.remove_class("order-action-active")

    def _preferred_order_symbol(self) -> str:
        if self.active_tab == "lookup" and self.lookup_sym:
            return str(self.lookup_sym).strip().upper()
        try:
            dt = self.query_one("#dt-portfolio", DataTable)
            row_index = dt.cursor_row
            if row_index is not None:
                row = dt.get_row_at(row_index)
                sym = str(row[0]).strip().upper()
                if sym and sym != "NO POSITIONS":
                    return sym
        except Exception:
            pass
        positions = self._stats.get("positions", []) if hasattr(self, "_stats") else []
        if positions:
            return str(positions[0].get("sym") or "").strip().upper()
        return ""

    def _preferred_sell_ticket_defaults(self) -> tuple[str, str]:
        sym = self._preferred_order_symbol()
        holdings = _build_sell_holdings_map(self._stats.get("positions", []) if hasattr(self, "_stats") else [])
        qty = holdings.get(sym) if sym else None
        if qty:
            return sym, str(qty)
        if holdings:
            fallback_sym = next(iter(holdings))
            return fallback_sym, str(holdings[fallback_sym])
        return "", ""

    def _preferred_order_price_text(self, sym: str) -> str:
        price = self._get_ltp_for_symbol(sym) if sym else None
        return f"{price:.1f}" if price else ""

    def _resolve_ticket_price(self, sym: str, raw_price: str) -> Optional[float]:
        if raw_price:
            try:
                return float(str(raw_price).replace(",", "").strip())
            except ValueError:
                return None
        return self._get_ltp_for_symbol(sym)

    def _parse_slippage(self, raw_value: str) -> float:
        try:
            return max(0.0, float(str(raw_value).strip() or "2.0"))
        except ValueError:
            return 2.0

    def _format_order_band(self, action: str, price: float, slippage_pct: float) -> str:
        slip = max(0.0, slippage_pct) / 100.0
        if action == "BUY":
            upper = price * (1 + slip)
            return f"≤ {upper:,.1f}"
        lower = price * (1 - slip)
        return f"≥ {lower:,.1f}"

    def action_buy(self) -> None:
        self.push_screen(
            ModalDialog(
                "buy",
                initial_symbol=self._preferred_order_symbol(),
                initial_price=self._preferred_order_price_text(self._preferred_order_symbol()),
            ),
            callback=self._on_buy,
        )

    def action_sell(self) -> None:
        symbol, shares = self._preferred_sell_ticket_defaults()
        self.push_screen(
            ModalDialog(
                "sell",
                initial_symbol=symbol,
                initial_shares=shares,
                initial_price=self._preferred_order_price_text(symbol),
                holdings_positions=self._stats.get("positions", []) if hasattr(self, "_stats") else [],
            ),
            callback=self._on_sell,
        )
