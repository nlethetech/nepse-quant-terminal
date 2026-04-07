from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    method: str
    info: str
    vol_series: Optional[pd.Series]
    probs: Optional[pd.DataFrame]


@dataclass
class FactorModelResult:
    probability: Optional[float]
    coefficients: Optional[pd.Series]
    latest_factors: Optional[pd.Series]
    contributions: Optional[pd.Series]
    history: Optional[pd.DataFrame]
    message: str
    accuracy_cv: Optional[float] = None
    accuracy_std: Optional[float] = None
    confusion: Optional[pd.DataFrame] = None
    feature_importance_df: Optional[pd.DataFrame] = None
    warnings: Optional[List[str]] = None
    component_probs: Optional[Dict[str, float]] = None
    component_confidence: Optional[Dict[str, float]] = None
    cv_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    disabled_reason: Optional[str] = None
    position_multiplier: Optional[float] = None
    sector_rank_snapshot: Optional[pd.Series] = None
    model_tier: str = "Unrated"
    trust_score: float = 1.0


@dataclass
class DownsideRiskResult:
    downside_vol_annual: float
    max_drawdown_pct: float
    max_drawdown_start: Optional[pd.Timestamp]
    max_drawdown_end: Optional[pd.Timestamp]
    current_drawdown_pct: float
    longest_duration_days: int
    drawdown_series: pd.Series
    equity_curve: pd.Series


@dataclass
class DecisionResult:
    decision: str
    rationale: List[str]


__all__ = [
    "RegimeResult",
    "FactorModelResult",
    "DownsideRiskResult",
    "DecisionResult",
]
