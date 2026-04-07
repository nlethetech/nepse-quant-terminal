"""Conformal Value-at-Risk with regime weighting.

Model 12: Distribution-free VaR bounds using split conformal prediction.

Standard parametric VaR (Gaussian assumption) understates tail risk in
NEPSE because:
    1. NEPSE returns have fat tails (circuit breakers at +/-10%).
    2. Regime shifts (bull/crash) make the return distribution non-stationary.
    3. Low liquidity causes clustering of extreme returns.

Conformal VaR provides finite-sample coverage guarantees WITHOUT
distributional assumptions.  The key idea:
    1. Split returns into training + calibration sets.
    2. Fit a base quantile estimator on training data.
    3. Compute "nonconformity scores" on calibration data (how wrong
       is the base estimate?).
    4. Adjust the prediction by the calibration quantile of scores.

Result: P(loss <= VaR) >= 1-alpha, guaranteed.

Academic basis:
    * Romano, Patterson & Candes (2019) "Conformalized Quantile Regression"
    * Vovk, Gammerman & Shafer (2005) "Algorithmic Learning in a Random World"
    * Barber et al. (2023) "Conformal Prediction Beyond Exchangeability"

Integration points:
    * position_sizing.py: scale positions by conformal VaR (tighter = bigger)
    * institutional.py: dynamic stop placement using conformal VaR
    * validation/regime_stress.py: validate coverage across regimes
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Conformal VaR
# ---------------------------------------------------------------------------

class ConformalVaR:
    """Distribution-free VaR with conformal calibration.

    Parameters
    ----------
    alpha : float
        Tail probability (default 0.05 = 95th percentile VaR).
    window : int
        Rolling lookback window for fitting (default 252 = ~1 NEPSE year).
    cal_ratio : float
        Fraction of window used for calibration (default 0.30).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        window: int = 252,
        cal_ratio: float = 0.30,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if window < 30:
            raise ValueError(f"window must be >= 30, got {window}")
        if not 0.1 <= cal_ratio <= 0.5:
            raise ValueError(f"cal_ratio must be in [0.1, 0.5], got {cal_ratio}")

        self.alpha = alpha
        self.window = window
        self.cal_ratio = cal_ratio

    def fit_predict(self, returns: np.ndarray) -> float:
        """Compute conformal VaR for the next period.

        Uses split conformal prediction:
            1. Split returns into training / calibration sets.
            2. Fit a base quantile estimate on the training set (empirical
               quantile + EWMA volatility scaling).
            3. Compute nonconformity scores on the calibration set.
            4. Adjust the base prediction by the calibration quantile of scores.

        Parameters
        ----------
        returns : np.ndarray
            Historical return series (1D array), most recent last.
            Length should be >= self.window.

        Returns
        -------
        float
            Conformal VaR estimate (negative number representing the loss
            threshold at the alpha quantile).  A return below this value
            is a VaR violation.
        """
        returns = np.asarray(returns, dtype=float)

        if len(returns) < 30:
            # Fallback: simple empirical quantile with no conformal adjustment
            return float(np.quantile(returns, self.alpha))

        # Use the most recent `window` observations
        if len(returns) > self.window:
            returns = returns[-self.window:]

        n = len(returns)
        split = max(20, int(n * (1 - self.cal_ratio)))
        train = returns[:split]
        cal = returns[split:]

        if len(cal) < 5:
            return float(np.quantile(returns, self.alpha))

        # Base estimator: EWMA-scaled quantile
        # This adapts to changing volatility (important for NEPSE regimes)
        base_var = self._ewma_quantile(train, self.alpha)

        # Nonconformity scores: how much does each calibration observation
        # deviate from the base VaR estimate?
        # Score = (base_var - actual_return): large positive = VaR was too loose
        scores = base_var - cal

        # Conformal adjustment: find the (1-alpha) quantile of scores
        # This inflates the VaR to achieve the target coverage
        q_level = min(1.0, (1 - self.alpha) * (1 + 1 / len(cal)))
        adjustment = float(np.quantile(scores, q_level))

        # Adjusted VaR
        conformal_var = base_var - adjustment

        return float(conformal_var)

    def fit_predict_interval(
        self,
        returns: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute conformal prediction interval [lower, upper].

        Returns
        -------
        tuple of (lower_bound, upper_bound)
            Two-sided prediction interval for the next return.
        """
        returns = np.asarray(returns, dtype=float)

        if len(returns) < 30:
            lo = float(np.quantile(returns, self.alpha / 2))
            hi = float(np.quantile(returns, 1 - self.alpha / 2))
            return lo, hi

        if len(returns) > self.window:
            returns = returns[-self.window:]

        n = len(returns)
        split = max(20, int(n * (1 - self.cal_ratio)))
        train = returns[:split]
        cal = returns[split:]

        if len(cal) < 5:
            lo = float(np.quantile(returns, self.alpha / 2))
            hi = float(np.quantile(returns, 1 - self.alpha / 2))
            return lo, hi

        # Base interval from training set
        base_lo = self._ewma_quantile(train, self.alpha / 2)
        base_hi = self._ewma_quantile(train, 1 - self.alpha / 2)

        # Nonconformity scores: max deviation outside the interval
        scores = np.maximum(base_lo - cal, cal - base_hi)
        scores = np.maximum(scores, 0.0)  # only positive deviations

        q_level = min(1.0, (1 - self.alpha) * (1 + 1 / len(cal)))
        adjustment = float(np.quantile(scores, q_level))

        return float(base_lo - adjustment), float(base_hi + adjustment)

    @staticmethod
    def _ewma_quantile(
        returns: np.ndarray,
        quantile: float,
        halflife: int = 30,
    ) -> float:
        """EWMA-scaled empirical quantile estimator.

        Scales returns by recent volatility to adapt to regime changes,
        then computes the quantile and rescales back.
        """
        if len(returns) < 10:
            return float(np.quantile(returns, quantile))

        # EWMA volatility
        decay = np.exp(-np.log(2) / halflife)
        weights = decay ** np.arange(len(returns) - 1, -1, -1)
        weights /= weights.sum()

        # Weighted variance
        mean_ret = np.average(returns, weights=weights)
        var_ret = np.average((returns - mean_ret) ** 2, weights=weights)
        ewma_vol = np.sqrt(max(var_ret, 1e-12))

        # Scale returns to unit volatility
        standardized = returns / ewma_vol

        # Quantile of standardized returns
        q_std = np.quantile(standardized, quantile)

        # Rescale back
        return float(q_std * ewma_vol)

    def coverage_test(
        self,
        returns: np.ndarray,
        var_estimates: np.ndarray,
    ) -> Dict[str, float]:
        """Test empirical coverage of VaR estimates.

        Parameters
        ----------
        returns : np.ndarray
            Realised returns.
        var_estimates : np.ndarray
            Corresponding VaR estimates (same length as returns).

        Returns
        -------
        dict with keys:
            empirical_coverage : float
                Fraction of observations where return >= VaR.
            expected_coverage : float
                1 - alpha.
            violations : int
                Number of VaR breaches.
            total : int
                Total observations.
            kupiec_pvalue : float
                p-value from Kupiec (1995) proportion-of-failures test.
        """
        returns = np.asarray(returns, dtype=float)
        var_estimates = np.asarray(var_estimates, dtype=float)

        if len(returns) != len(var_estimates):
            raise ValueError("returns and var_estimates must have same length")

        n = len(returns)
        violations = int(np.sum(returns < var_estimates))
        empirical_coverage = 1.0 - violations / max(n, 1)

        # Kupiec (1995) likelihood ratio test
        p_hat = violations / max(n, 1)
        p_expected = self.alpha

        kupiec_pvalue = 1.0
        if 0 < p_hat < 1 and 0 < p_expected < 1 and n > 0:
            try:
                from scipy import stats

                lr_stat = -2 * (
                    violations * np.log(p_expected / p_hat)
                    + (n - violations) * np.log((1 - p_expected) / (1 - p_hat))
                )
                kupiec_pvalue = float(1 - stats.chi2.cdf(lr_stat, df=1))
            except (ImportError, ValueError, FloatingPointError):
                kupiec_pvalue = float("nan")

        return {
            "empirical_coverage": empirical_coverage,
            "expected_coverage": 1 - self.alpha,
            "violations": violations,
            "total": n,
            "kupiec_pvalue": kupiec_pvalue,
        }


# ---------------------------------------------------------------------------
# Regime-Weighted Conformal VaR
# ---------------------------------------------------------------------------

class RegimeWeightedConformalVaR(ConformalVaR):
    """Conformal VaR that uses regime probabilities for weighting.

    When the HMM provides P(bull), P(neutral), P(bear), this class
    computes separate VaR estimates per regime and blends them by the
    posterior probabilities.

    This accounts for the non-stationarity that standard conformal
    prediction struggles with during regime transitions.

    Parameters
    ----------
    alpha : float
        Tail probability.
    window : int
        Rolling lookback.
    regime_labels : np.ndarray or None
        Historical regime labels (0=bull, 1=neutral, 2=bear) aligned
        with the returns passed to fit_predict.
    """

    def fit_predict_regime(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
        current_regime_probs: Optional[np.ndarray] = None,
    ) -> float:
        """Regime-weighted conformal VaR.

        Parameters
        ----------
        returns : np.ndarray
            Historical returns.
        regime_labels : np.ndarray
            Integer regime labels for each return (same length).
            0=bull, 1=neutral, 2=bear.
        current_regime_probs : np.ndarray or None
            Current posterior probabilities [p_bull, p_neutral, p_bear].
            If None, falls back to standard (unweighted) conformal VaR.

        Returns
        -------
        float
            Regime-weighted conformal VaR.
        """
        returns = np.asarray(returns, dtype=float)
        regime_labels = np.asarray(regime_labels, dtype=int)

        if current_regime_probs is None or len(returns) < 60:
            return self.fit_predict(returns)

        unique_regimes = np.unique(regime_labels)
        regime_vars: Dict[int, float] = {}

        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_rets = returns[mask]

            if len(regime_rets) < 20:
                # Not enough data for this regime; use full sample
                regime_vars[regime] = self.fit_predict(returns)
            else:
                regime_vars[regime] = self.fit_predict(regime_rets)

        # Blend by current regime probabilities
        probs = np.asarray(current_regime_probs, dtype=float)
        blended_var = 0.0
        for i, regime in enumerate(sorted(regime_vars.keys())):
            if i < len(probs):
                blended_var += probs[i] * regime_vars[regime]
            else:
                # Extra regime not in probs; equal weight
                blended_var += (1 / len(unique_regimes)) * regime_vars[regime]

        return float(blended_var)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compute_conformal_var(
    returns: np.ndarray,
    alpha: float = 0.05,
    window: int = 252,
) -> float:
    """One-shot conformal VaR computation.

    Parameters
    ----------
    returns : np.ndarray
        Historical return series.
    alpha : float
        Tail probability (default 0.05).
    window : int
        Rolling lookback (default 252).

    Returns
    -------
    float
        Conformal VaR estimate.
    """
    return ConformalVaR(alpha=alpha, window=window).fit_predict(returns)


def compute_conformal_position_scale(
    returns: np.ndarray,
    alpha: float = 0.05,
    max_loss_pct: float = 0.02,
) -> float:
    """Compute position scaling factor based on conformal VaR.

    The idea: size positions so that the maximum expected loss (at the
    alpha level) does not exceed max_loss_pct of total capital.

    Parameters
    ----------
    returns : np.ndarray
        Historical return series for the asset.
    alpha : float
        VaR confidence level.
    max_loss_pct : float
        Maximum acceptable loss as fraction of capital (default 2%).

    Returns
    -------
    float
        Position scale factor in [0.0, 1.0].  Multiply by capital
        to get the position size.
    """
    cvar = compute_conformal_var(returns, alpha=alpha)

    if cvar >= 0:
        # VaR is positive (no expected loss at this level) -- full position
        return 1.0

    # VaR is negative: position_scale = max_loss_pct / |VaR|
    scale = max_loss_pct / abs(cvar)
    return min(scale, 1.0)


__all__ = [
    "ConformalVaR",
    "RegimeWeightedConformalVaR",
    "compute_conformal_var",
    "compute_conformal_position_scale",
]
