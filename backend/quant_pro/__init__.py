"""
NEPSE Quant Pro package.

This package modularizes the quantitative logic for the Streamlit dashboard.
Each submodule contains focused functionality that is imported by the thin
Streamlit front-end in app.py.
"""

from .config import PERIODS_PER_YEAR
from .types import (
    RegimeResult,
    FactorModelResult,
    DownsideRiskResult,
    DecisionResult,
)

__all__ = [
    "PERIODS_PER_YEAR",
    "RegimeResult",
    "FactorModelResult",
    "DownsideRiskResult",
    "DecisionResult",
]
