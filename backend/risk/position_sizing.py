"""
Position Sizing Module for NEPSE Paper Trading

Conservative position sizing with:
- Max 15% per position
- Max 35% per sector
- Kelly criterion capped at 25%
- 20% cash reserve
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from backend.quant_pro.sectors import SECTOR_GROUPS


@dataclass
class Position:
    """A sized position."""
    symbol: str
    shares: int
    weight: float
    value: float
    sector: str
    signal_type: str
    confidence: float


# NEPSE Transaction Costs
BROKER_COMMISSION_TIERS = [
    (2_500, 10.0),        # Flat Rs 10 for < 2,500
    (50_000, 0.0036),     # 0.36%
    (500_000, 0.0033),    # 0.33%
    (2_000_000, 0.0031),  # 0.31%
    (10_000_000, 0.0027), # 0.27%
    (float('inf'), 0.0024), # 0.24%
]
SEBON_FEE = 0.00015  # 0.015%
DP_CHARGE = 25.0     # Rs 25 per transaction


def calculate_transaction_cost(amount: float, is_buy: bool = True) -> float:
    """Calculate NEPSE transaction costs."""
    if amount <= 0:
        return 0.0

    # Broker commission (tiered)
    for threshold, rate in BROKER_COMMISSION_TIERS:
        if amount <= threshold:
            if isinstance(rate, float) and rate < 1:
                commission = amount * rate
            else:
                commission = rate
            break
    else:
        commission = amount * 0.0024

    # SEBON fee
    sebon = amount * SEBON_FEE

    # DP charge
    dp = DP_CHARGE

    return commission + sebon + dp


def get_symbol_sector(symbol: str) -> str:
    """Look up sector for a symbol."""
    for sector, symbols in SECTOR_GROUPS.items():
        if symbol in symbols:
            return sector
    return "Others"


def calculate_kelly_fraction(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    max_kelly: float = 0.25,
) -> float:
    """
    Calculate Kelly criterion position size.

    Kelly f* = (p * b - q) / b
    where:
        p = probability of win
        q = 1 - p
        b = win/loss ratio

    We use half-Kelly (capped at max_kelly) for safety.
    """
    if avg_loss == 0 or avg_win <= 0:
        return 0.0

    q = 1 - win_prob
    b = avg_win / avg_loss

    kelly = (win_prob * b - q) / b
    half_kelly = kelly / 2

    return max(0, min(half_kelly, max_kelly))


def size_positions(
    signals: List[Dict],
    capital: float,
    prices: Dict[str, float],
    max_positions: int = 7,
    max_single_pct: float = 0.15,
    max_sector_pct: float = 0.35,
    cash_reserve_pct: float = 0.20,
) -> List[Position]:
    """
    Convert signals to sized positions with risk limits.

    Args:
        signals: List of dicts with keys: symbol, signal_type, strength, confidence
        capital: Total capital in NPR
        prices: Dict of symbol -> current price
        max_positions: Max number of positions
        max_single_pct: Max allocation per position (default 15%)
        max_sector_pct: Max allocation per sector (default 35%)
        cash_reserve_pct: Minimum cash reserve (default 20%)

    Returns:
        List of Position objects
    """
    if not signals or capital <= 0:
        return []

    deployable = capital * (1 - cash_reserve_pct)

    # Sort signals by strength * confidence
    sorted_signals = sorted(
        signals,
        key=lambda s: s.get("strength", 0) * s.get("confidence", 0),
        reverse=True
    )

    positions: List[Position] = []
    sector_allocations: Dict[str, float] = {}
    total_allocated = 0.0

    for sig in sorted_signals:
        if len(positions) >= max_positions:
            break

        symbol = sig["symbol"]
        if symbol not in prices or prices[symbol] <= 0:
            continue

        price = prices[symbol]
        sector = get_symbol_sector(symbol)

        # Calculate raw weight from signal
        strength = sig.get("strength", 0.5)
        confidence = sig.get("confidence", 0.5)
        raw_weight = strength * confidence

        # Apply position limit
        weight = min(raw_weight, max_single_pct)

        # Check sector limit
        current_sector = sector_allocations.get(sector, 0.0)
        if current_sector + weight > max_sector_pct:
            weight = max(0, max_sector_pct - current_sector)

        if weight < 0.03:  # Skip if less than 3%
            continue

        # Check total limit
        if total_allocated + weight > 1.0:
            weight = max(0, 1.0 - total_allocated)

        if weight < 0.03:
            continue

        # Calculate position
        value = deployable * weight
        shares = int(value / price)

        if shares <= 0:
            continue

        # Actual values
        actual_value = shares * price
        actual_weight = actual_value / capital

        positions.append(Position(
            symbol=symbol,
            shares=shares,
            weight=actual_weight,
            value=actual_value,
            sector=sector,
            signal_type=sig.get("signal_type", "unknown"),
            confidence=confidence,
        ))

        sector_allocations[sector] = current_sector + actual_weight
        total_allocated += actual_weight

    return positions


def estimate_round_trip_cost(positions: List[Position]) -> float:
    """Estimate total round-trip transaction costs."""
    total = 0.0
    for pos in positions:
        buy_cost = calculate_transaction_cost(pos.value, is_buy=True)
        sell_cost = calculate_transaction_cost(pos.value, is_buy=False)
        total += buy_cost + sell_cost
    return total


def format_positions_for_csv(positions: List[Position]) -> str:
    """Format positions for CSV output."""
    lines = ["Symbol,Shares,Weight,Signal_Type,Confidence,Sector,Value"]
    for pos in positions:
        lines.append(
            f"{pos.symbol},{pos.shares},{pos.weight:.1%},"
            f"{pos.signal_type},{pos.confidence:.1%},{pos.sector},{pos.value:,.0f}"
        )
    return "\n".join(lines)


def should_rebalance(
    current_positions: dict,
    proposed_positions: dict,
    current_prices: dict,
    transaction_cost_rate: float = 0.006,
) -> bool:
    """
    Check if rebalancing utility exceeds transaction cost (No-Trade Zone, Model 9).

    Compares total turnover (sum of absolute weight changes) against
    the NO_TRADE_UTILITY_THRESHOLD plus estimated transaction costs.
    Only rebalance if the change is large enough to justify friction.

    NEPSE round-trip costs are approximately 0.5-1.2% depending on trade
    size, so the default transaction_cost_rate of 0.6% per leg is
    conservative.

    Args:
        current_positions: {symbol: weight} of current portfolio
        proposed_positions: {symbol: weight} of proposed portfolio
        current_prices: {symbol: price} for cost estimation (unused in
                        weight-based version but kept for API consistency)
        transaction_cost_rate: estimated cost per unit of turnover
                               (~0.6% per leg for NEPSE)

    Returns:
        True if rebalancing is justified (turnover exceeds no-trade threshold).
        False if the change is too small -- HOLD current positions.
    """
    from backend.quant_pro.config import NO_TRADE_UTILITY_THRESHOLD

    # Compute total turnover as sum of absolute weight changes
    all_symbols = set(list(current_positions.keys()) + list(proposed_positions.keys()))
    total_turnover = sum(
        abs(proposed_positions.get(s, 0.0) - current_positions.get(s, 0.0))
        for s in all_symbols
    )

    # Estimated transaction cost of the rebalance
    transaction_cost = total_turnover * transaction_cost_rate

    # Only rebalance if turnover is significant relative to threshold + cost
    return total_turnover > NO_TRADE_UTILITY_THRESHOLD + transaction_cost


__all__ = [
    "Position",
    "calculate_transaction_cost",
    "get_symbol_sector",
    "calculate_kelly_fraction",
    "size_positions",
    "estimate_round_trip_cost",
    "format_positions_for_csv",
    "should_rebalance",
]
