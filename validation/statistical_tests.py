"""
Statistical significance tests for strategy validation.

Implements:
- Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado (2012)
- Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014)
- Minimum Track Record Length (MinTRL)
- T-test on excess returns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


# ── Risk-free rate ───────────────────────────────────────────────────────
# 91-day NRB T-bill rate as of Jan 2026
R_F_ANNUAL = 0.0235
NEPSE_TRADING_DAYS = 240


def _sharpe_std_error(sr: float, n: int, skew: float, kurtosis: float) -> float:
    """
    Standard error of the Sharpe ratio estimator.

    From Lo (2002) and Bailey & Lopez de Prado (2012):
        SE(SR) = sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (n - 1))

    Parameters
    ----------
    sr : Observed annualised Sharpe ratio (we convert to per-period internally)
    n  : Number of return observations
    skew : Skewness of returns
    kurtosis : Excess kurtosis of returns
    """
    # Work in per-period units
    sr_pp = sr / math.sqrt(NEPSE_TRADING_DAYS)
    variance = (
        1.0
        - skew * sr_pp
        + ((kurtosis - 1) / 4.0) * sr_pp ** 2
    )
    if variance < 0:
        variance = 1.0  # Degenerate case fallback
    return math.sqrt(max(variance, 0) / max(n - 1, 1))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """
    Probabilistic Sharpe Ratio — P(true SR > benchmark SR).

    Bailey & Lopez de Prado (2012).

    Parameters
    ----------
    observed_sharpe : Annualised Sharpe ratio of the strategy
    benchmark_sharpe : Sharpe ratio of the benchmark (often 0)
    n_obs : Number of return observations
    skew : Skewness of strategy returns
    kurtosis : Excess kurtosis of strategy returns

    Returns
    -------
    PSR in [0, 1] — probability that the true SR exceeds the benchmark
    """
    if n_obs < 2:
        return 0.0
    se = _sharpe_std_error(observed_sharpe, n_obs, skew, kurtosis)
    if se <= 0:
        return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
    # Convert both to per-period
    obs_pp = observed_sharpe / math.sqrt(NEPSE_TRADING_DAYS)
    bench_pp = benchmark_sharpe / math.sqrt(NEPSE_TRADING_DAYS)
    z = (obs_pp - bench_pp) / se
    return float(sp_stats.norm.cdf(z))


def _expected_max_sharpe(n_trials: int, var_sharpe: float, n_obs: int) -> float:
    """
    Expected maximum Sharpe ratio from n_trials independent trials.

    E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    where γ ≈ 0.5772 (Euler–Mascheroni constant)

    Simplified: E[max] ≈ sqrt(V(SR)) * [(1-γ)*Z(1-1/N) + γ*Z(1-1/Ne)]
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler–Mascheroni
    std_sr = math.sqrt(max(var_sharpe, 1e-10))
    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return std_sr * ((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
    var_sharpe: Optional[float] = None,
) -> float:
    """
    Deflated Sharpe Ratio — adjusts PSR for multiple testing.

    Bailey & Lopez de Prado (2014).

    Parameters
    ----------
    observed_sharpe : Annualised Sharpe of the selected strategy
    n_trials : Number of strategy variants tested (backtest permutations)
    n_obs : Number of return observations
    skew : Skewness of returns
    kurtosis : Excess kurtosis of returns
    var_sharpe : Variance of Sharpe ratios across trials (estimated if None)

    Returns
    -------
    DSR in [0, 1]
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(
            observed_sharpe, 0.0, n_obs, skew, kurtosis
        )
    if var_sharpe is None:
        # Estimate from the SE formula: Var(SR) ≈ SE^2
        se = _sharpe_std_error(observed_sharpe, n_obs, skew, kurtosis)
        var_sharpe = se ** 2

    sr_star = _expected_max_sharpe(n_trials, var_sharpe, n_obs)
    # Convert sr_star back to annualised
    sr_star_ann = sr_star * math.sqrt(NEPSE_TRADING_DAYS)
    return probabilistic_sharpe_ratio(
        observed_sharpe, sr_star_ann, n_obs, skew, kurtosis
    )


def min_track_record_length(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """
    Minimum Track Record Length (MinTRL).

    Bailey & Lopez de Prado (2012) — minimum number of observations needed
    for the observed SR to be statistically significant at the given confidence.

    Returns
    -------
    Number of observations (trading days) needed.
    """
    z_alpha = sp_stats.norm.ppf(confidence)
    sr_pp = observed_sharpe / math.sqrt(NEPSE_TRADING_DAYS)
    bench_pp = benchmark_sharpe / math.sqrt(NEPSE_TRADING_DAYS)

    diff = sr_pp - bench_pp
    if diff <= 0:
        return float("inf")

    # MinTRL = 1 + (1 - skew*SR + (kurt-1)/4 * SR^2) * (z_alpha / diff)^2
    var_factor = 1.0 - skew * sr_pp + ((kurtosis - 1) / 4.0) * sr_pp ** 2
    if var_factor < 0:
        var_factor = 1.0
    n_star = 1.0 + var_factor * (z_alpha / diff) ** 2
    return n_star


def excess_return_ttest(
    daily_returns: np.ndarray,
    rf_daily: Optional[float] = None,
) -> dict:
    """
    One-sample t-test on daily excess returns vs zero.

    Parameters
    ----------
    daily_returns : Array of daily strategy returns
    rf_daily : Daily risk-free rate (default: R_F_ANNUAL / NEPSE_TRADING_DAYS)

    Returns
    -------
    Dict with t_stat, p_value, n_obs, significant (at 5%)
    """
    if rf_daily is None:
        rf_daily = R_F_ANNUAL / NEPSE_TRADING_DAYS

    excess = daily_returns - rf_daily
    n = len(excess)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "n_obs": n, "significant": False}

    t_stat, p_value = sp_stats.ttest_1samp(excess, 0.0)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_obs": n,
        "significant": p_value < 0.05,
    }


@dataclass
class StatisticalReport:
    """Full statistical significance report."""
    sharpe_ratio: float
    n_obs: int
    skew: float
    kurtosis: float
    psr: float
    dsr: float
    min_trl: float
    min_trl_years: float
    ttest: dict
    n_trials: int
    psr_pass: bool
    dsr_pass: bool
    ttest_pass: bool
    sufficient_track_record: bool


def full_statistical_report(
    daily_returns: np.ndarray,
    sharpe_ratio: float,
    n_trials: int = 1,
    psr_threshold: float = 0.90,
    dsr_threshold: float = 0.85,
    var_sharpe: Optional[float] = None,
) -> StatisticalReport:
    """
    Run all statistical tests and return a comprehensive report.

    Parameters
    ----------
    daily_returns : Array of daily strategy returns
    sharpe_ratio : Annualised Sharpe ratio
    n_trials : Number of strategy variants tried (for DSR)
    psr_threshold : Minimum PSR to pass (default 0.90)
    dsr_threshold : Minimum DSR to pass (default 0.85)
    var_sharpe : Variance of Sharpe ratios across trials (for DSR).
        If None, estimated from SE formula. Pass actual variance from
        parameter sweep results for more accurate DSR.
    """
    n_obs = len(daily_returns)
    skew = float(sp_stats.skew(daily_returns)) if n_obs > 2 else 0.0
    kurt = float(sp_stats.kurtosis(daily_returns, fisher=True)) if n_obs > 2 else 0.0
    # fisher=True gives excess kurtosis (normal = 0)
    # Our formulas use kurtosis where normal = 3, so add 3
    kurtosis_full = kurt + 3.0

    psr = probabilistic_sharpe_ratio(sharpe_ratio, 0.0, n_obs, skew, kurtosis_full)
    dsr = deflated_sharpe_ratio(
        sharpe_ratio, n_trials, n_obs, skew, kurtosis_full,
        var_sharpe=var_sharpe,
    )
    min_trl = min_track_record_length(sharpe_ratio, 0.0, skew, kurtosis_full)
    min_trl_years = min_trl / NEPSE_TRADING_DAYS
    ttest = excess_return_ttest(daily_returns)

    return StatisticalReport(
        sharpe_ratio=sharpe_ratio,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
        psr=psr,
        dsr=dsr,
        min_trl=min_trl,
        min_trl_years=min_trl_years,
        ttest=ttest,
        n_trials=n_trials,
        psr_pass=psr >= psr_threshold,
        dsr_pass=dsr >= dsr_threshold,
        ttest_pass=ttest["significant"],
        sufficient_track_record=n_obs >= min_trl,
    )
