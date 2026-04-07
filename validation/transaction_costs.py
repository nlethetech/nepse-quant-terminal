"""
Unified NEPSE Transaction Cost Model.

Single source of truth for all NEPSE transaction costs, replacing the three
separate NepseFees implementations in simple_backtest.py, paper_trade_tracker.py,
and live_trader.py.

Fee schedule per SEBON regulations (updated Jan 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised breakdown of a single-leg transaction."""
    amount: float            # shares * price
    commission: float        # Broker commission (tiered)
    sebon_fee: float         # SEBON regulatory fee (0.015%)
    nepse_fee: float         # NEPSE fee (20% of broker commission)
    dp_charge: float         # DP charge (flat Rs 25)
    dp_name_transfer: float  # DP name transfer (Rs 5, sell side only)
    total: float             # Sum of all fees

    def __str__(self) -> str:
        lines = [
            f"  Trade amount:       NPR {self.amount:>14,.2f}",
            f"  Broker commission:  NPR {self.commission:>14,.2f}",
            f"  SEBON fee:          NPR {self.sebon_fee:>14,.2f}",
            f"  NEPSE fee:          NPR {self.nepse_fee:>14,.2f}",
            f"  DP charge:          NPR {self.dp_charge:>14,.2f}",
        ]
        if self.dp_name_transfer > 0:
            lines.append(
                f"  DP name transfer:   NPR {self.dp_name_transfer:>14,.2f}"
            )
        lines.append(f"  ─────────────────────────────────")
        lines.append(f"  Total fees:         NPR {self.total:>14,.2f}")
        lines.append(f"  Fee %:              {self.total / self.amount * 100:>14.4f}%"
                      if self.amount > 0 else "  Fee %:                    N/A")
        return "\n".join(lines)


@dataclass(frozen=True)
class RoundTripCost:
    """Complete round-trip cost including CGT."""
    buy: CostBreakdown
    sell: CostBreakdown
    cgt: float              # Capital gains tax
    gross_pnl: float        # (exit - entry) * shares
    net_pnl: float          # gross_pnl - buy.total - sell.total - cgt
    total_cost: float       # buy.total + sell.total + cgt
    cost_pct: float         # total_cost / buy.amount

    def __str__(self) -> str:
        lines = [
            "BUY SIDE:",
            str(self.buy),
            "",
            "SELL SIDE:",
            str(self.sell),
            "",
            f"  Capital gains tax:  NPR {self.cgt:>14,.2f}",
            f"  ─────────────────────────────────",
            f"  Gross P&L:          NPR {self.gross_pnl:>14,.2f}",
            f"  Total costs:        NPR {self.total_cost:>14,.2f}",
            f"  Net P&L:            NPR {self.net_pnl:>14,.2f}",
            f"  Round-trip cost %:  {self.cost_pct * 100:>14.4f}%",
        ]
        return "\n".join(lines)


class TransactionCostModel:
    """
    Single source of truth for all NEPSE transaction costs.

    Covers:
    - Tiered broker commission (6 tiers per SEBON schedule)
    - SEBON regulatory fee (0.015%)
    - NEPSE fee (20% of broker commission)
    - DP charge (Rs 25 per transaction)
    - DP name transfer (Rs 5 per script, sell side)
    - Capital gains tax (7.5% short-term, 5% long-term)
    - Dividend tax (5% on cash dividends)
    """

    # ── Broker commission tiers ──────────────────────────────────────────
    # (upper_bound, rate_or_flat, is_flat)
    BROKER_TIERS = [
        (2_500,       10.0,   True),    # Flat Rs 10
        (50_000,      0.0036, False),   # 0.36%
        (500_000,     0.0033, False),   # 0.33%
        (2_000_000,   0.0031, False),   # 0.31%
        (10_000_000,  0.0027, False),   # 0.27%
        (float("inf"), 0.0024, False),  # 0.24%
    ]

    # ── Regulatory fees ──────────────────────────────────────────────────
    SEBON_FEE_PCT = 0.00015        # 0.015%
    NEPSE_FEE_PCT = 0.20           # 20% of broker commission

    # ── Depository fees ──────────────────────────────────────────────────
    DP_CHARGE = 25.0               # Rs 25 per transaction
    DP_NAME_TRANSFER = 5.0         # Rs 5 per script (sell side only)

    # ── Tax rates ────────────────────────────────────────────────────────
    CGT_SHORT_TERM = 0.075         # 7.5% on gains if held < 365 days
    CGT_LONG_TERM = 0.05           # 5.0% on gains if held >= 365 days
    CGT_THRESHOLD_DAYS = 365
    DIVIDEND_TAX = 0.05            # 5% on cash dividends

    # ── Convenience shortcuts ────────────────────────────────────────────

    @classmethod
    def broker_commission(cls, amount: float) -> float:
        """Tiered broker commission per SEBON rules."""
        for upper, rate, is_flat in cls.BROKER_TIERS:
            if amount <= upper:
                return rate if is_flat else amount * rate
        # Should never reach here, but safety fallback
        return amount * 0.0024

    @classmethod
    def total_buy_cost(cls, shares: int, price: float) -> CostBreakdown:
        """Full cost breakdown for a buy transaction."""
        amount = shares * price
        commission = cls.broker_commission(amount)
        sebon = amount * cls.SEBON_FEE_PCT
        nepse_fee = commission * cls.NEPSE_FEE_PCT
        dp = cls.DP_CHARGE
        total = commission + sebon + nepse_fee + dp
        return CostBreakdown(
            amount=amount,
            commission=commission,
            sebon_fee=sebon,
            nepse_fee=nepse_fee,
            dp_charge=dp,
            dp_name_transfer=0.0,
            total=total,
        )

    @classmethod
    def total_sell_cost(
        cls,
        shares: int,
        price: float,
        holding_days: Optional[int] = None,
        entry_price: Optional[float] = None,
    ) -> CostBreakdown:
        """
        Full cost breakdown for a sell transaction.

        CGT is NOT included here — it is computed separately in round_trip_cost()
        because it depends on the full trade P&L.
        """
        amount = shares * price
        commission = cls.broker_commission(amount)
        sebon = amount * cls.SEBON_FEE_PCT
        nepse_fee = commission * cls.NEPSE_FEE_PCT
        dp = cls.DP_CHARGE
        dp_name = cls.DP_NAME_TRANSFER
        total = commission + sebon + nepse_fee + dp + dp_name
        return CostBreakdown(
            amount=amount,
            commission=commission,
            sebon_fee=sebon,
            nepse_fee=nepse_fee,
            dp_charge=dp,
            dp_name_transfer=dp_name,
            total=total,
        )

    @classmethod
    def capital_gains_tax(
        cls,
        gross_pnl: float,
        holding_days: int,
    ) -> float:
        """
        Capital gains tax on a completed trade.

        Returns 0 if the trade was a loss.
        """
        if gross_pnl <= 0:
            return 0.0
        rate = (
            cls.CGT_LONG_TERM
            if holding_days >= cls.CGT_THRESHOLD_DAYS
            else cls.CGT_SHORT_TERM
        )
        return gross_pnl * rate

    @classmethod
    def dividend_tax(cls, dividend_amount: float) -> float:
        """Tax on cash dividend income."""
        if dividend_amount <= 0:
            return 0.0
        return dividend_amount * cls.DIVIDEND_TAX

    @classmethod
    def round_trip_cost(
        cls,
        shares: int,
        entry_price: float,
        exit_price: float,
        holding_days: int = 0,
    ) -> RoundTripCost:
        """
        Complete round-trip cost including buy fees, sell fees, and CGT.
        """
        buy = cls.total_buy_cost(shares, entry_price)
        sell = cls.total_sell_cost(shares, exit_price)
        gross_pnl = (exit_price - entry_price) * shares
        cgt = cls.capital_gains_tax(gross_pnl, holding_days)
        total_cost = buy.total + sell.total + cgt
        net_pnl = gross_pnl - total_cost
        cost_pct = total_cost / buy.amount if buy.amount > 0 else 0.0
        return RoundTripCost(
            buy=buy,
            sell=sell,
            cgt=cgt,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_cost=total_cost,
            cost_pct=cost_pct,
        )

    @classmethod
    def total_fees(cls, shares: int, price: float, is_sell: bool = False) -> float:
        """
        Total single-leg fees including all SEBON-mandated components.

        Includes: broker commission + SEBON fee + NEPSE fee + DP charge
        Sell side adds: DP name transfer (Rs 5)
        """
        amount = shares * price
        commission = cls.broker_commission(amount)
        sebon = amount * cls.SEBON_FEE_PCT
        nepse_fee = commission * cls.NEPSE_FEE_PCT
        dp = cls.DP_CHARGE
        total = commission + sebon + nepse_fee + dp
        if is_sell:
            total += cls.DP_NAME_TRANSFER
        return total

    @classmethod
    def round_trip_pct(
        cls, shares: int, entry_price: float, exit_price: float
    ) -> float:
        """
        Round-trip cost as a fraction of entry value.

        Buy side: commission + SEBON + NEPSE fee + DP
        Sell side: commission + SEBON + NEPSE fee + DP + DP name transfer
        """
        entry_value = shares * entry_price
        if entry_value <= 0:
            return 0.0
        buy_fees = cls.total_fees(shares, entry_price, is_sell=False)
        sell_fees = cls.total_fees(shares, exit_price, is_sell=True)
        return (buy_fees + sell_fees) / entry_value

    # Alias for backward compatibility with paper_trade_tracker.NepseFees
    calculate_broker_commission = broker_commission

    @classmethod
    def calculate_total_fees(
        cls, qty: int, price: float, is_buy: bool = True
    ) -> dict:
        """
        Dict-based fee breakdown including all SEBON-mandated components.

        Buy side: commission + SEBON + NEPSE fee + DP
        Sell side: adds DP name transfer (Rs 5)
        """
        amount = qty * price
        commission = cls.broker_commission(amount)
        sebon = amount * cls.SEBON_FEE_PCT
        nepse_fee = commission * cls.NEPSE_FEE_PCT
        dp = cls.DP_CHARGE
        dp_name_transfer = 0.0 if is_buy else cls.DP_NAME_TRANSFER
        total = commission + sebon + nepse_fee + dp + dp_name_transfer
        return {
            "commission": commission,
            "sebon": sebon,
            "nepse_fee": nepse_fee,
            "dp_charge": dp,
            "dp_name_transfer": dp_name_transfer,
            "total": total,
        }

    @classmethod
    def calculate_tax(
        cls, net_gain: float, holding_period_days: int = 20
    ) -> float:
        """
        Backward-compatible: CGT calculation.

        Matches the old paper_trade_tracker.NepseFees.calculate_tax() signature.
        """
        return cls.capital_gains_tax(net_gain, holding_period_days)
