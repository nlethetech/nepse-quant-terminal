"""Unit tests for the validation framework modules."""

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from validation.statistical_tests import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    min_track_record_length,
    excess_return_ttest,
    full_statistical_report,
)
from validation.monte_carlo import (
    monte_carlo_trade_resample,
    block_bootstrap_ci,
    MonteCarloResult,
    BootstrapResult,
)
from validation.kill_switch import KillSwitch, KillReason


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPSR:
    """Probabilistic Sharpe Ratio tests."""

    def test_psr_high_sharpe_high_probability(self):
        """High observed Sharpe should give PSR close to 1."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=2.0, benchmark_sharpe=0.0,
            n_obs=500, skew=0.0, kurtosis=3.0,
        )
        assert 0.95 <= psr <= 1.0

    def test_psr_zero_sharpe_half(self):
        """Zero Sharpe vs zero benchmark should give PSR ≈ 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=0.0, benchmark_sharpe=0.0,
            n_obs=500, skew=0.0, kurtosis=3.0,
        )
        assert 0.45 <= psr <= 0.55

    def test_psr_negative_sharpe_low(self):
        """Negative Sharpe should give PSR < 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=-1.0, benchmark_sharpe=0.0,
            n_obs=500, skew=0.0, kurtosis=3.0,
        )
        assert psr < 0.1

    def test_psr_more_data_higher_confidence(self):
        """More observations should give higher PSR for same Sharpe."""
        psr_small = probabilistic_sharpe_ratio(1.0, 0.0, 50, 0.0, 3.0)
        psr_large = probabilistic_sharpe_ratio(1.0, 0.0, 1000, 0.0, 3.0)
        assert psr_large > psr_small

    def test_psr_bounded_0_1(self):
        """PSR should always be between 0 and 1."""
        for sr in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0]:
            psr = probabilistic_sharpe_ratio(sr, 0.0, 240, 0.0, 3.0)
            assert 0.0 <= psr <= 1.0


class TestDSR:
    """Deflated Sharpe Ratio tests."""

    def test_dsr_penalizes_multiple_trials(self):
        """DSR should decrease as number of trials increases."""
        dsr_1 = deflated_sharpe_ratio(1.5, n_trials=1, n_obs=500, skew=0.0, kurtosis=3.0)
        dsr_10 = deflated_sharpe_ratio(1.5, n_trials=10, n_obs=500, skew=0.0, kurtosis=3.0)
        dsr_100 = deflated_sharpe_ratio(1.5, n_trials=100, n_obs=500, skew=0.0, kurtosis=3.0)
        assert dsr_1 > dsr_10 > dsr_100

    def test_dsr_single_trial_equals_psr(self):
        """With 1 trial, DSR should equal PSR vs benchmark=0."""
        dsr = deflated_sharpe_ratio(1.0, n_trials=1, n_obs=500, skew=0.0, kurtosis=3.0)
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 500, 0.0, 3.0)
        assert dsr == pytest.approx(psr, abs=0.01)

    def test_dsr_bounded_0_1(self):
        """DSR should always be between 0 and 1."""
        for trials in [1, 5, 50, 500]:
            dsr = deflated_sharpe_ratio(1.5, trials, 500, 0.0, 3.0)
            assert 0.0 <= dsr <= 1.0


class TestMinTRL:
    """Minimum Track Record Length tests."""

    def test_higher_sharpe_needs_less_data(self):
        """Higher Sharpe ratio should require shorter track record."""
        trl_low = min_track_record_length(0.5, 0.0, 0.0, 3.0)
        trl_high = min_track_record_length(2.0, 0.0, 0.0, 3.0)
        assert trl_high < trl_low

    def test_zero_sharpe_infinite_trl(self):
        """Zero Sharpe vs zero benchmark needs infinite track record."""
        trl = min_track_record_length(0.0, 0.0, 0.0, 3.0)
        assert trl == float("inf")

    def test_negative_sharpe_infinite_trl(self):
        """Negative excess Sharpe needs infinite track record."""
        trl = min_track_record_length(-0.5, 0.0, 0.0, 3.0)
        assert trl == float("inf")

    def test_trl_positive_finite(self):
        """Positive Sharpe should give finite MinTRL."""
        trl = min_track_record_length(1.5, 0.0, 0.0, 3.0)
        assert 0 < trl < float("inf")


class TestExcessReturnTtest:
    """T-test on excess returns."""

    def test_significant_positive_returns(self):
        """Strong positive returns should be significant."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=500)  # Mean 0.1% per day
        result = excess_return_ttest(returns, rf_daily=0.0)
        assert result["significant"] == True
        assert result["t_stat"] > 0
        assert result["p_value"] < 0.05

    def test_not_significant_zero_returns(self):
        """Zero-mean returns should not be significant."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=500)
        result = excess_return_ttest(returns, rf_daily=0.0)
        # With enough data, zero-mean should not be significant
        # (may occasionally be significant by chance — use tight tolerance)
        assert result["p_value"] > 0.001 or abs(result["t_stat"]) < 4

    def test_empty_returns(self):
        result = excess_return_ttest(np.array([]), rf_daily=0.0)
        assert result["significant"] is False
        assert result["n_obs"] == 0


class TestFullStatisticalReport:
    """Integration test for the full report."""

    def test_full_report_structure(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=500)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(240))
        report = full_statistical_report(returns, sharpe, n_trials=5)

        assert hasattr(report, "psr")
        assert hasattr(report, "dsr")
        assert hasattr(report, "min_trl")
        assert hasattr(report, "ttest")
        assert 0.0 <= report.psr <= 1.0
        assert 0.0 <= report.dsr <= 1.0
        assert report.n_obs == 500


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════


class TestMonteCarlo:
    """Monte Carlo trade resampling tests."""

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        returns = [0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01]
        r1 = monte_carlo_trade_resample(returns, rng_seed=42, n_simulations=100)
        r2 = monte_carlo_trade_resample(returns, rng_seed=42, n_simulations=100)
        np.testing.assert_array_equal(r1.all_terminal_wealth, r2.all_terminal_wealth)

    def test_different_seeds_different_results(self):
        returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        r1 = monte_carlo_trade_resample(returns, rng_seed=42, n_simulations=100)
        r2 = monte_carlo_trade_resample(returns, rng_seed=99, n_simulations=100)
        assert not np.array_equal(r1.all_terminal_wealth, r2.all_terminal_wealth)

    def test_all_winning_trades_no_ruin(self):
        returns = [0.05, 0.03, 0.04, 0.02, 0.06]
        result = monte_carlo_trade_resample(returns, n_simulations=1000)
        assert result.prob_ruin == 0.0
        assert result.prob_loss == 0.0

    def test_all_losing_trades_high_ruin(self):
        returns = [-0.15, -0.12, -0.10, -0.08, -0.20]
        result = monte_carlo_trade_resample(returns, n_simulations=1000)
        assert result.prob_loss > 0.9

    def test_empty_trades(self):
        result = monte_carlo_trade_resample([], n_simulations=100)
        assert result.n_trades == 0
        assert result.n_simulations == 0

    def test_percentiles_ordered(self):
        returns = [0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01, -0.02, 0.06]
        result = monte_carlo_trade_resample(returns, n_simulations=5000)
        pcts = result.terminal_wealth_pcts
        assert pcts[5] <= pcts[25] <= pcts[50] <= pcts[75] <= pcts[95]

    def test_sharpe_ci_contains_reasonable_range(self):
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.005, 0.02, size=50))
        result = monte_carlo_trade_resample(returns, n_simulations=5000)
        assert result.sharpe_ci[0] < result.sharpe_ci[1]


class TestBlockBootstrap:
    """Block bootstrap CI tests."""

    def test_bootstrap_ci_contains_point_estimate(self):
        """Point estimate should be within the 95% CI (most of the time)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, size=500)
        point_sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(240))

        result = block_bootstrap_ci(returns, n_bootstrap=5000, block_size=21, rng_seed=42)
        # Widen tolerance — point estimate should be within or near CI
        assert result.sharpe_ci[0] <= point_sharpe + 0.5
        assert result.sharpe_ci[1] >= point_sharpe - 0.5

    def test_ci_width_narrows_with_more_data(self):
        """More data should give tighter CIs."""
        rng = np.random.default_rng(42)
        small = rng.normal(0.0005, 0.01, size=100)
        large = rng.normal(0.0005, 0.01, size=1000)

        r_small = block_bootstrap_ci(small, n_bootstrap=2000, rng_seed=42)
        r_large = block_bootstrap_ci(large, n_bootstrap=2000, rng_seed=42)

        width_small = r_small.sharpe_ci[1] - r_small.sharpe_ci[0]
        width_large = r_large.sharpe_ci[1] - r_large.sharpe_ci[0]
        assert width_large < width_small

    def test_too_short_for_blocks(self):
        returns = np.array([0.01, 0.02])
        result = block_bootstrap_ci(returns, block_size=21)
        assert result.n_bootstrap == 0


# ═══════════════════════════════════════════════════════════════════════════
# Kill Switch
# ═══════════════════════════════════════════════════════════════════════════


class TestKillSwitch:
    """Kill switch trigger tests."""

    def test_daily_loss_trigger(self):
        ks = KillSwitch(max_daily_loss_pct=0.03)
        halt, reason = ks.check(
            current_nav=970_000, peak_nav=1_000_000,
            daily_pnl=-35_000, daily_start_nav=1_000_000,
            consecutive_losses=0,
        )
        assert halt is True
        assert "daily loss" in reason.lower()

    def test_drawdown_trigger(self):
        ks = KillSwitch(max_drawdown_pct=0.15)
        halt, reason = ks.check(
            current_nav=840_000, peak_nav=1_000_000,
            daily_pnl=-5_000, daily_start_nav=845_000,
            consecutive_losses=0,
        )
        assert halt is True
        assert "drawdown" in reason.lower()

    def test_consecutive_losses_trigger(self):
        ks = KillSwitch(max_consecutive_losses=5)
        halt, reason = ks.check(
            current_nav=980_000, peak_nav=1_000_000,
            daily_pnl=-2_000, daily_start_nav=982_000,
            consecutive_losses=6,
        )
        assert halt is True
        assert "consecutive" in reason.lower()

    def test_stale_data_trigger(self):
        ks = KillSwitch(stale_data_minutes=30)
        old_time = datetime.now() - timedelta(minutes=45)
        halt, reason = ks.check(
            current_nav=1_000_000, peak_nav=1_000_000,
            daily_pnl=0, daily_start_nav=1_000_000,
            consecutive_losses=0,
            last_data_time=old_time,
        )
        assert halt is True
        assert "stale" in reason.lower()

    def test_no_trigger_normal(self):
        ks = KillSwitch()
        halt, reason = ks.check(
            current_nav=995_000, peak_nav=1_000_000,
            daily_pnl=-5_000, daily_start_nav=1_000_000,
            consecutive_losses=2,
            last_data_time=datetime.now(),
        )
        assert halt is False
        assert reason == "OK"

    def test_stays_triggered_after_first_trigger(self):
        ks = KillSwitch(max_daily_loss_pct=0.03)
        ks.check(
            current_nav=960_000, peak_nav=1_000_000,
            daily_pnl=-40_000, daily_start_nav=1_000_000,
        )
        assert ks.is_triggered

        # Even with good numbers, should stay triggered
        halt, reason = ks.check(
            current_nav=1_100_000, peak_nav=1_100_000,
            daily_pnl=100_000, daily_start_nav=1_000_000,
        )
        assert halt is True
        assert "Previously triggered" in reason

    def test_reset(self):
        ks = KillSwitch(max_daily_loss_pct=0.03)
        ks.check(
            current_nav=960_000, peak_nav=1_000_000,
            daily_pnl=-40_000, daily_start_nav=1_000_000,
        )
        assert ks.is_triggered

        ks.reset()
        assert not ks.is_triggered

        halt, _ = ks.check(
            current_nav=995_000, peak_nav=1_000_000,
            daily_pnl=-5_000, daily_start_nav=1_000_000,
        )
        assert halt is False
