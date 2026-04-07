"""
Regime Detection Module for NEPSE Quant System

Implements two complementary models:

Model 10: HMMRegimeDetector
    - 3-State Gaussian HMM (bull / neutral / bear)
    - Uses market returns + rolling volatility as features
    - Outputs posterior state probabilities for graduated exposure

Model 11: BOCPDDetector
    - Bayesian Online Changepoint Detection (Adams & MacKay 2007)
    - Pure numpy implementation with Normal-Gamma conjugate prior
    - Detects when regime transitions occur in real-time
    - Can trigger HMM refit to save compute

Reference:
    Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
    gwgundersen/bocd on GitHub for the Normal-Gamma conjugate implementation
"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# Model 10: HMM Regime Detector (3-State Gaussian HMM)
# =============================================================================

class HMMRegimeDetector:
    """
    3-State Gaussian Hidden Markov Model for market regime detection.

    States are labeled by their mean return after fitting:
        - bull:    highest mean return state
        - neutral: middle mean return state
        - bear:    lowest mean return state

    Features used:
        - Daily market returns
        - Rolling 20-day realized volatility

    Usage:
        detector = HMMRegimeDetector(n_states=3, lookback=252, n_init=10)
        detector.fit(market_returns)
        result = detector.predict(market_returns)
        exposure = detector.get_exposure_multiplier(result["probabilities"])
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 252,
        n_init: int = 10,
        vol_window: int = 20,
    ):
        self.n_states = n_states
        self.lookback = lookback
        self.n_init = n_init
        self.vol_window = vol_window
        self.model = None
        self._state_map: Dict[int, str] = {}  # maps HMM state index -> regime name
        self._fitted = False

    def _prepare_features(
        self,
        market_returns: np.ndarray,
        market_volatility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prepare feature matrix for HMM.

        If market_volatility is not provided, compute rolling vol from returns.
        Returns an (n_samples, 2) array of [return, volatility].
        """
        returns = np.asarray(market_returns).flatten()

        if market_volatility is not None:
            vol = np.asarray(market_volatility).flatten()
        else:
            # Compute rolling realized volatility
            vol = np.full_like(returns, np.nan)
            for i in range(self.vol_window, len(returns)):
                vol[i] = np.std(returns[i - self.vol_window : i], ddof=1)

        # Stack features, drop rows with NaN
        features = np.column_stack([returns, vol])
        valid_mask = ~np.isnan(features).any(axis=1)
        return features[valid_mask]

    def fit(
        self,
        market_returns: np.ndarray,
        market_volatility: Optional[np.ndarray] = None,
    ) -> "HMMRegimeDetector":
        """
        Fit HMM on market-level features.

        Uses GaussianHMM with full covariance, multiple random restarts
        to avoid local optima. After fitting, labels states by mean return.

        Args:
            market_returns: 1D array of daily market returns
            market_volatility: Optional 1D array of rolling volatility.
                               If None, computed internally using vol_window.

        Returns:
            self (for chaining)
        """
        from hmmlearn.hmm import GaussianHMM

        features = self._prepare_features(market_returns, market_volatility)

        if len(features) < self.n_states * 10:
            logger.warning(
                "Insufficient data for HMM fitting: %d samples (need >= %d). "
                "Returning unfitted detector.",
                len(features),
                self.n_states * 10,
            )
            return self

        # Use only the last `lookback` observations for fitting
        if len(features) > self.lookback:
            features = features[-self.lookback :]

        best_model = None
        best_score = -np.inf

        for i in range(self.n_init):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=200,
                    tol=1e-4,
                    random_state=i * 42,
                    verbose=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(features)

                score = model.score(features)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.debug("HMM init %d failed: %s", i, e)
                continue

        if best_model is None:
            logger.warning("All HMM initializations failed. Detector not fitted.")
            return self

        self.model = best_model
        self._fitted = True

        # Label states by mean return (feature index 0 = returns)
        state_means = best_model.means_[:, 0]  # mean return per state
        sorted_indices = np.argsort(state_means)

        regime_names = ["bear", "neutral", "bull"]
        self._state_map = {
            int(sorted_indices[i]): regime_names[i]
            for i in range(self.n_states)
        }

        logger.info(
            "HMM fitted: states=%d, log-likelihood=%.2f, "
            "mean returns: bear=%.4f, neutral=%.4f, bull=%.4f",
            self.n_states,
            best_score,
            state_means[sorted_indices[0]],
            state_means[sorted_indices[1]],
            state_means[sorted_indices[2]],
        )

        return self

    def predict(
        self,
        market_returns: np.ndarray,
        market_volatility: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Predict current regime with probabilities.

        Args:
            market_returns: 1D array of daily market returns (most recent window)
            market_volatility: Optional 1D array of rolling volatility.

        Returns:
            dict with:
                "regime": "bull" | "neutral" | "bear"
                "probabilities": {"bull": float, "neutral": float, "bear": float}
                "confidence": float (probability of most likely state)
        """
        if not self._fitted or self.model is None:
            # Fallback: return neutral with equal probabilities
            return {
                "regime": "neutral",
                "probabilities": {"bull": 0.33, "neutral": 0.34, "bear": 0.33},
                "confidence": 0.34,
            }

        features = self._prepare_features(market_returns, market_volatility)

        if len(features) == 0:
            return {
                "regime": "neutral",
                "probabilities": {"bull": 0.33, "neutral": 0.34, "bear": 0.33},
                "confidence": 0.34,
            }

        try:
            # Get posterior state probabilities for the last observation
            posteriors = self.model.predict_proba(features)
            last_probs = posteriors[-1]

            # Map HMM state probabilities to regime names
            regime_probs = {}
            for state_idx, regime_name in self._state_map.items():
                regime_probs[regime_name] = float(last_probs[state_idx])

            # Most likely regime
            regime = max(regime_probs, key=regime_probs.get)
            confidence = regime_probs[regime]

            return {
                "regime": regime,
                "probabilities": regime_probs,
                "confidence": confidence,
            }
        except Exception as e:
            logger.warning("HMM prediction failed: %s", e)
            return {
                "regime": "neutral",
                "probabilities": {"bull": 0.33, "neutral": 0.34, "bear": 0.33},
                "confidence": 0.34,
            }

    def get_exposure_multiplier(self, regime_probs: Dict[str, float]) -> float:
        """
        Convert regime probabilities to exposure multiplier (0.0 to 1.0).

        Formula:
            exposure = p_bull * 1.0 + p_neutral * 0.5 + p_bear * 0.0

        This gives graduated exposure:
            - Pure bull: 1.0 (full exposure)
            - Pure neutral: 0.5 (half exposure)
            - Pure bear: 0.0 (no new entries)
            - Mixed: weighted blend

        Args:
            regime_probs: dict with keys "bull", "neutral", "bear" mapping to
                          probabilities that sum to ~1.0

        Returns:
            Exposure multiplier in [0.0, 1.0]
        """
        p_bull = regime_probs.get("bull", 0.0)
        p_neutral = regime_probs.get("neutral", 0.0)
        p_bear = regime_probs.get("bear", 0.0)

        exposure = p_bull * 1.0 + p_neutral * 0.5 + p_bear * 0.0
        return float(np.clip(exposure, 0.0, 1.0))


# =============================================================================
# Model 11: BOCPD Detector (Bayesian Online Changepoint Detection)
# =============================================================================

class BOCPDDetector:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Maintains a run-length posterior distribution P(r_t | x_{1:t}) that
    tracks how long the current regime/segment has lasted. A changepoint
    is detected when the probability of run length 0 (new segment starting)
    exceeds a threshold.

    Uses Normal-Gamma conjugate prior for Gaussian observations, which
    allows exact Bayesian updates without sampling.

    The hazard function H(r) = 1/lambda is constant (geometric prior on
    segment length), meaning segments are expected to last ~lambda steps.

    Usage:
        detector = BOCPDDetector(hazard_lambda=200)
        for obs in observations:
            cp_prob = detector.update(obs)
            if detector.detect(threshold=0.5):
                print("Changepoint detected!")
    """

    def __init__(
        self,
        hazard_lambda: float = 200.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        """
        Initialize BOCPD with Normal-Gamma conjugate prior.

        Args:
            hazard_lambda: Expected run length between changepoints.
                           Higher = fewer expected changepoints.
            mu0: Prior mean of the Normal distribution.
            kappa0: Prior precision scaling (number of pseudo-observations
                    for the mean estimate).
            alpha0: Prior shape for the Gamma (precision) distribution.
            beta0: Prior rate for the Gamma (precision) distribution.
        """
        self.hazard = 1.0 / hazard_lambda

        # Prior hyperparameters (stored for reset)
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Run-length posterior: P(r_t | x_{1:t})
        # Initially, run length is 0 with probability 1
        self.run_length_probs: np.ndarray = np.array([1.0])

        # Sufficient statistics for Normal-Gamma per run length
        # Each entry corresponds to the posterior given run length r
        self._mu: np.ndarray = np.array([mu0])
        self._kappa: np.ndarray = np.array([kappa0])
        self._alpha: np.ndarray = np.array([alpha0])
        self._beta: np.ndarray = np.array([beta0])

        self._changepoint_prob: float = 0.0
        self._t: int = 0

    def _student_t_logpdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Log-pdf of Student-t predictive distribution.

        The predictive distribution for a new observation under the
        Normal-Gamma conjugate is a Student-t with:
            df = 2 * alpha
            loc = mu
            scale = sqrt(beta * (kappa + 1) / (alpha * kappa))

        Args:
            x: observed value
            mu, kappa, alpha, beta: Normal-Gamma sufficient statistics
                                     (arrays, one per run length)

        Returns:
            Array of log-predictive-probabilities, one per run length.
        """
        df = 2.0 * alpha
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
        scale = np.sqrt(scale_sq)

        # Student-t log-pdf using the gamma function formulation
        z = (x - mu) / scale
        log_pdf = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - ((df + 1.0) / 2.0) * np.log(1.0 + z**2 / df)
        )
        return log_pdf

    def update(self, observation: float) -> float:
        """
        Process one observation and return changepoint probability.

        Implements the Adams-MacKay (2007) message-passing algorithm:
        1. Evaluate predictive probability P(x_t | r_{t-1}, params)
        2. Compute growth probabilities (no changepoint at this step)
        3. Compute changepoint probability (new segment starts)
        4. Normalize the run-length distribution
        5. Update sufficient statistics for each run length

        Args:
            observation: scalar observation (e.g., daily market return)

        Returns:
            Changepoint probability P(r_t = 0) -- the probability that
            a new segment is starting at this time step.
        """
        self._t += 1
        x = float(observation)

        # Step 1: Predictive probabilities P(x_t | r_{t-1}, params)
        log_pred = self._student_t_logpdf(
            x, self._mu, self._kappa, self._alpha, self._beta
        )
        pred = np.exp(log_pred)

        # Step 2: Growth probabilities (existing run continues)
        # P(r_t = r_{t-1}+1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) * P(x_t|r_{t-1}) * (1 - H)
        growth_probs = self.run_length_probs * pred * (1.0 - self.hazard)

        # Step 3: Changepoint probability (new segment starts)
        # P(r_t = 0, x_{1:t}) = sum over r_{t-1} of P(r_{t-1}) * P(x_t|r_{t-1}) * H
        cp_prob = np.sum(self.run_length_probs * pred * self.hazard)

        # Step 4: Assemble new run-length distribution and normalize
        new_run_length_probs = np.empty(len(growth_probs) + 1)
        new_run_length_probs[0] = cp_prob
        new_run_length_probs[1:] = growth_probs

        evidence = new_run_length_probs.sum()
        if evidence > 0:
            new_run_length_probs /= evidence

        self.run_length_probs = new_run_length_probs
        self._changepoint_prob = float(new_run_length_probs[0])

        # Step 5: Update sufficient statistics (Normal-Gamma posterior update)
        # For the new changepoint run length (r=0), reset to prior
        # For continuing run lengths, do Bayesian update
        new_mu = np.empty(len(self._mu) + 1)
        new_kappa = np.empty(len(self._kappa) + 1)
        new_alpha = np.empty(len(self._alpha) + 1)
        new_beta = np.empty(len(self._beta) + 1)

        # Run length 0: reset to prior
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0

        # Run lengths 1..T: update existing sufficient statistics
        new_kappa[1:] = self._kappa + 1.0
        new_mu[1:] = (self._kappa * self._mu + x) / new_kappa[1:]
        new_alpha[1:] = self._alpha + 0.5
        new_beta[1:] = (
            self._beta
            + 0.5 * self._kappa * (x - self._mu) ** 2 / new_kappa[1:]
        )

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        # Truncate very long run lengths to prevent unbounded memory growth
        max_run = max(500, self._t)
        if len(self.run_length_probs) > max_run:
            # Keep only top max_run entries, renormalize
            self.run_length_probs = self.run_length_probs[:max_run]
            total = self.run_length_probs.sum()
            if total > 0:
                self.run_length_probs /= total
            self._mu = self._mu[:max_run]
            self._kappa = self._kappa[:max_run]
            self._alpha = self._alpha[:max_run]
            self._beta = self._beta[:max_run]

        return self._changepoint_prob

    def detect(self, threshold: float = 0.5) -> bool:
        """
        Return True if a recent changepoint is likely.

        Uses the cumulative probability of short run lengths (r < 5) as
        the detection criterion. When a changepoint occurs, probability
        mass concentrates on very short run lengths (the new segment
        just started). This is more robust than checking P(r=0) alone,
        which under normalization equals the constant hazard rate.

        Args:
            threshold: probability threshold. A value of 0.5 means
                       "more than 50% probability that the current
                       segment started fewer than 5 steps ago."

        Returns:
            True if a changepoint is detected at the current step.
        """
        if len(self.run_length_probs) < 5:
            return False
        # Probability mass on short run lengths (segment just started)
        short_run_prob = self.run_length_probs[:5].sum()
        return short_run_prob > threshold

    @property
    def changepoint_probability(self) -> float:
        """Current changepoint probability P(r_t = 0)."""
        return self._changepoint_prob

    @property
    def expected_run_length(self) -> float:
        """Expected run length E[r_t] under the current posterior."""
        if len(self.run_length_probs) == 0:
            return 0.0
        lengths = np.arange(len(self.run_length_probs))
        return float(np.sum(lengths * self.run_length_probs))

    def reset(self):
        """Reset detector state to initial prior."""
        self.run_length_probs = np.array([1.0])
        self._mu = np.array([self.mu0])
        self._kappa = np.array([self.kappa0])
        self._alpha = np.array([self.alpha0])
        self._beta = np.array([self.beta0])
        self._changepoint_prob = 0.0
        self._t = 0


# =============================================================================
# Convenience functions
# =============================================================================

def detect_regime_from_prices(
    prices_series: pd.Series,
    n_states: int = 3,
    lookback: int = 252,
    n_init: int = 10,
) -> Dict:
    """
    Convenience function: fit HMM and predict regime from a price series.

    Args:
        prices_series: pandas Series of daily prices (e.g., NEPSE index)
        n_states: number of HMM states (default 3)
        lookback: training lookback in trading days (default 252)
        n_init: number of random restarts (default 10)

    Returns:
        Dict with regime, probabilities, confidence, and exposure_multiplier.
    """
    returns = prices_series.pct_change().dropna().values

    detector = HMMRegimeDetector(
        n_states=n_states,
        lookback=lookback,
        n_init=n_init,
    )
    detector.fit(returns)
    result = detector.predict(returns)
    result["exposure_multiplier"] = detector.get_exposure_multiplier(
        result["probabilities"]
    )
    return result


def run_bocpd_on_returns(
    returns: np.ndarray,
    hazard_lambda: float = 200.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BOCPD on a full returns array and return changepoint probabilities.

    Args:
        returns: 1D array of daily returns
        hazard_lambda: expected run length between changepoints
        threshold: changepoint detection threshold

    Returns:
        Tuple of:
            - cp_probs: array of changepoint probabilities per timestep
            - changepoints: boolean array indicating detected changepoints
    """
    detector = BOCPDDetector(hazard_lambda=hazard_lambda)
    cp_probs = np.zeros(len(returns))
    changepoints = np.zeros(len(returns), dtype=bool)

    for i, ret in enumerate(returns):
        cp_probs[i] = detector.update(ret)
        changepoints[i] = detector.detect(threshold)

    n_cp = changepoints.sum()
    logger.info(
        "BOCPD complete: %d observations, %d changepoints detected "
        "(threshold=%.2f, hazard_lambda=%.0f)",
        len(returns),
        n_cp,
        threshold,
        hazard_lambda,
    )
    return cp_probs, changepoints


__all__ = [
    "HMMRegimeDetector",
    "BOCPDDetector",
    "detect_regime_from_prices",
    "run_bocpd_on_returns",
]
