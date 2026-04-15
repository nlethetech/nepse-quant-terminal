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
    roe: Optional[float] = None
    eps: Optional[float] = None
    nav: Optional[float] = None


class FundamentalScanner:
    """Stub scanner — extend with your own data source."""
    def __init__(self):
        self.fundamentals: dict = {}

    def scan(self) -> List[AlphaSignal]:
        return []


class MomentumScanner:
    """Stub momentum scanner."""
    def calculate_signals(self, prices, volumes) -> List[AlphaSignal]:
        return []


class LiquidityScanner:
    """Stub liquidity scanner."""
    def calculate_signals(self, prices, volumes) -> List[AlphaSignal]:
        return []
