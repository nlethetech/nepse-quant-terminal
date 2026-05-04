"""
Alpha signal types and base dataclasses for NEPSE Quant.

This module defines the signal taxonomy used across the trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class SignalType(Enum):
    """Types of alpha signals supported by the system."""
    CORPORATE_ACTION = "corporate_action"
    FUNDAMENTAL = "fundamental"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    DIVIDEND = "dividend"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    QUARTERLY_FUNDAMENTAL = "quarterly_fundamental"
    XSEC_MOMENTUM = "xsec_momentum"
    ACCUMULATION = "accumulation"
    DISPOSITION = "disposition"
    RESIDUAL_MOMENTUM = "residual_momentum"
    LEAD_LAG = "lead_lag"
    ANCHORING_52WK = "anchoring_52wk"
    INFORMED_TRADING = "informed_trading"
    PAIRS_TRADE = "pairs_trade"
    EARNINGS_DRIFT = "earnings_drift"
    MACRO_REMITTANCE = "macro_remittance"
    SATELLITE_HYDRO = "satellite_hydro"
    NLP_SENTIMENT = "nlp_sentiment"
    SETTLEMENT_PRESSURE = "settlement_pressure"
    VALUE_BOUNCE = "value_bounce"


@dataclass
class AlphaSignal:
    """A single alpha signal for a symbol."""
    symbol: str
    signal_type: SignalType
    direction: int          # +1 long, 0 neutral
    strength: float         # 0 to 1
    confidence: float       # 0 to 1
    reasoning: str
    expires: Optional[datetime] = None
    target_exit_date: Optional[datetime] = None

    @property
    def score(self) -> float:
        return self.direction * self.strength * self.confidence


@dataclass
class FundamentalData:
    """Basic fundamental data for a symbol."""
    symbol: str
    sector: str = ""
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    nav: Optional[float] = None
    book_value: Optional[float] = None
    revenue_growth_qoq: Optional[float] = None
    profit_growth_qoq: Optional[float] = None
    eps_growth_qoq: Optional[float] = None
    book_value_growth_qoq: Optional[float] = None
    capital_adequacy_pct: Optional[float] = None
    npl_pct: Optional[float] = None
    cost_income_ratio: Optional[float] = None
    latest_net_profit: Optional[float] = None
    latest_revenue: Optional[float] = None
    data_source: str = ""


@dataclass
class CorporateAction:
    """Minimal corporate action record used by the public signal generator."""
    symbol: str
    action_type: str = ""
    announcement_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    value: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class FundamentalScanner:
    """Fundamental quality/value scanner for the public paper-trading build."""

    def __init__(self):
        self.fundamentals: Dict[str, FundamentalData] = {}

    def update_fundamentals(self, data: FundamentalData) -> None:
        self.fundamentals[str(data.symbol).upper()] = data

    def scan(self) -> List[AlphaSignal]:
        signals: List[AlphaSignal] = []
        for fd in self.fundamentals.values():
            symbol = str(fd.symbol).upper()
            if not symbol:
                continue
            if fd.latest_net_profit is not None and fd.latest_net_profit <= 0:
                continue
            if fd.capital_adequacy_pct is not None and fd.capital_adequacy_pct < 8.5:
                continue
            if fd.npl_pct is not None and fd.npl_pct > 6.0:
                continue

            score = 0.0
            reasons: List[str] = []
            if fd.pe_ratio is not None and fd.pe_ratio <= 12:
                score += 0.18
                reasons.append(f"P/E {fd.pe_ratio:.1f}")
            if fd.pb_ratio is not None and fd.pb_ratio <= 1.2:
                score += 0.16
                reasons.append(f"P/B {fd.pb_ratio:.2f}")
            if fd.roe is not None and fd.roe >= 0.14:
                score += 0.14
                reasons.append(f"ROE {fd.roe:.0%}")
            if fd.dividend_yield is not None and fd.dividend_yield >= 0.04:
                score += 0.08
                reasons.append(f"dividend {fd.dividend_yield:.0%}")
            if fd.eps_growth_qoq is not None and fd.eps_growth_qoq > 0:
                score += min(0.20, fd.eps_growth_qoq)
                reasons.append(f"EPS QoQ {fd.eps_growth_qoq:.0%}")
            if fd.revenue_growth_qoq is not None and fd.revenue_growth_qoq > 0:
                score += min(0.12, fd.revenue_growth_qoq)
                reasons.append(f"revenue QoQ {fd.revenue_growth_qoq:.0%}")
            if fd.profit_growth_qoq is not None and fd.profit_growth_qoq > 0:
                score += min(0.12, fd.profit_growth_qoq)
                reasons.append(f"profit QoQ {fd.profit_growth_qoq:.0%}")
            if fd.capital_adequacy_pct is not None:
                score += 0.08
                reasons.append(f"CAR {fd.capital_adequacy_pct:.1f}%")
            if fd.npl_pct is not None and fd.npl_pct <= 3.0:
                score += 0.06
                reasons.append(f"NPL {fd.npl_pct:.1f}%")

            if score < 0.45:
                continue
            confidence = min(0.95, 0.55 + score * 0.35)
            signals.append(
                AlphaSignal(
                    symbol=symbol,
                    signal_type=SignalType.FUNDAMENTAL,
                    direction=1,
                    strength=min(1.0, score),
                    confidence=confidence,
                    reasoning=", ".join(reasons) or "fundamental quality/value screen",
                )
            )
        return signals


class MomentumScanner:
    """Stub momentum scanner."""
    def calculate_signals(self, prices, volumes) -> List[AlphaSignal]:
        return []


class LiquidityScanner:
    """Stub liquidity scanner."""
    def calculate_signals(self, prices, volumes) -> List[AlphaSignal]:
        return []


class CorporateActionScanner:
    """Public-release corporate-action scanner stub."""

    def __init__(self):
        self.actions: List[CorporateAction] = []

    def add_action(self, action: CorporateAction) -> None:
        self.actions.append(action)

    def scan(self, as_of: Optional[datetime] = None) -> List[AlphaSignal]:
        return []


class AlphaAggregator:
    """Simple signal aggregator used by legacy scripts."""

    def aggregate(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        return list(signals)
