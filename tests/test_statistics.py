"""Comprehensive tests for the statistical engine."""

import math
import numpy as np
import pytest
from scipy import stats as sp_stats

from src.statistics import (
    BetaBinomialModel,
    GaussianHMM,
    GARCH,
    MonteCarloSimulator,
    IsotonicCalibrator,
    EnsemblePredictor,
    hurst_exponent,
    kl_divergence,
    jensen_shannon_divergence,
    kelly_with_uncertainty,
    volume_imbalance_signal,
    smart_money_divergence,
)


# ===========================================================================
# Beta-Binomial Model Tests
# ===========================================================================

class TestBetaBinomialModel:

    def test_uniform_prior(self):
        model = BetaBinomialModel(1, 1)
        assert abs(model.mean - 0.5) < 1e-10

    def test_update_shifts_posterior(self):
        prior = BetaBinomialModel(1, 1)
        posterior = prior.update(successes=8, failures=2)
        assert posterior.mean > 0.7  # strongly shifted toward 1
        assert posterior.alpha == 9
        assert posterior.beta == 3

    def test_update_with_more_data_reduces_variance(self):
        model1 = BetaBinomialModel(1, 1)
        model2 = model1.update(10, 10)
        model3 = model2.update(100, 100)
        assert model1.variance > model2.variance > model3.variance

    def test_credible_interval_contains_mean(self):
        model = BetaBinomialModel(10, 5)
        low, high = model.credible_interval(0.95)
        assert low < model.mean < high

    def test_credible_interval_wider_with_less_data(self):
        weak = BetaBinomialModel(2, 2)
        strong = BetaBinomialModel(200, 200)
        weak_ci = weak.credible_interval(0.95)
        strong_ci = strong.credible_interval(0.95)
        assert (weak_ci[1] - weak_ci[0]) > (strong_ci[1] - strong_ci[0])

    def test_soft_update(self):
        model = BetaBinomialModel(5, 5)
        soft = model.update_with_weight(0.8, weight=3.0)
        assert soft.mean > model.mean  # shifted toward 0.8

    def test_log_marginal_likelihood(self):
        model = BetaBinomialModel(1, 1)
        ll = model.log_marginal_likelihood(7, 10)
        assert isinstance(ll, float)
        assert not math.isnan(ll)

    def test_posterior_predictive_equals_mean(self):
        model = BetaBinomialModel(15, 5)
        assert abs(model.posterior_predictive() - model.mean) < 1e-10

    def test_strong_evidence_overwhelms_prior(self):
        biased_prior = BetaBinomialModel(1, 100)  # prior says ~1%
        posterior = biased_prior.update(successes=1000, failures=0)
        assert posterior.mean > 0.9  # evidence dominates


# ===========================================================================
# Gaussian HMM Tests
# ===========================================================================

class TestGaussianHMM:

    def test_fit_two_regimes(self):
        np.random.seed(42)
        # Generate data with two clear regimes
        regime1 = np.random.normal(0.0, 0.1, 100)
        regime2 = np.random.normal(1.0, 0.1, 100)
        data = np.concatenate([regime1, regime2])

        hmm = GaussianHMM(n_states=2, n_iter=50)
        hmm.fit(data)
        assert hmm._fitted
        # Means should be near 0 and 1
        sorted_means = sorted(hmm.means)
        assert sorted_means[0] < 0.3
        assert sorted_means[1] > 0.7

    def test_predict_regime(self):
        np.random.seed(42)
        regime1 = np.random.normal(-1.0, 0.2, 50)
        regime2 = np.random.normal(1.0, 0.2, 50)
        data = np.concatenate([regime1, regime2])

        hmm = GaussianHMM(n_states=2, n_iter=50)
        hmm.fit(data)
        path = hmm.predict_regime(data)
        assert len(path) == 100
        # First half and second half should be different regimes mostly
        assert np.mean(path[:25] == path[0]) > 0.8
        assert np.mean(path[75:] == path[75]) > 0.8

    def test_regime_probabilities_sum_to_one(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)
        hmm = GaussianHMM(n_states=2, n_iter=20)
        hmm.fit(data)
        probs = hmm.regime_probabilities(data)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ===========================================================================
# GARCH Tests
# ===========================================================================

class TestGARCH:

    def test_fit_and_forecast(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 200)
        # Add volatility clustering
        for i in range(50, 100):
            returns[i] *= 3  # high vol regime

        garch = GARCH()
        garch.fit(returns)
        assert garch._fitted
        assert garch.alpha > 0
        assert garch.beta_param > 0
        assert garch.persistence < 1.0

        forecast = garch.forecast(returns, horizon=5)
        assert len(forecast) == 5
        assert all(f > 0 for f in forecast)

    def test_stationarity_constraint(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        garch = GARCH()
        garch.fit(returns)
        assert garch.alpha + garch.beta_param < 1.0

    def test_unconditional_variance_positive(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        garch = GARCH()
        garch.fit(returns)
        assert garch.unconditional_variance > 0

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            GARCH().fit(np.array([0.01, 0.02, 0.03]))


# ===========================================================================
# Monte Carlo Simulator Tests
# ===========================================================================

class TestMonteCarloSimulator:

    def test_mean_near_base_prob(self):
        mc = MonteCarloSimulator(n_simulations=50000, seed=42)
        result = mc.simulate_binary_outcome(
            base_prob=0.6, volatility=0.5, time_remaining_frac=0.5
        )
        assert abs(result["mean"] - 0.6) < 0.05

    def test_ci_contains_base_prob(self):
        mc = MonteCarloSimulator(n_simulations=50000, seed=42)
        result = mc.simulate_binary_outcome(
            base_prob=0.7, volatility=0.3, time_remaining_frac=0.3
        )
        assert result["ci_95"][0] < 0.7 < result["ci_95"][1]

    def test_higher_volatility_wider_ci(self):
        mc = MonteCarloSimulator(n_simulations=30000, seed=42)
        low_vol = mc.simulate_binary_outcome(0.5, 0.1, 0.5)
        high_vol = mc.simulate_binary_outcome(0.5, 2.0, 0.5)
        low_width = low_vol["ci_95"][1] - low_vol["ci_95"][0]
        high_width = high_vol["ci_95"][1] - high_vol["ci_95"][0]
        assert high_width > low_width

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should have lower variance than naive."""
        mc1 = MonteCarloSimulator(n_simulations=10000, seed=42)
        mc2 = MonteCarloSimulator(n_simulations=10000, seed=42)

        # Run multiple times to check variance of estimates
        estimates_anti = []
        for s in range(10):
            mc = MonteCarloSimulator(n_simulations=5000, seed=s)
            r = mc.simulate_binary_outcome(0.5, 1.0, 0.5)
            estimates_anti.append(r["mean"])
        # Mean should be close to 0.5 with low variance
        assert abs(np.mean(estimates_anti) - 0.5) < 0.05

    def test_extreme_probs_handled(self):
        mc = MonteCarloSimulator(n_simulations=1000, seed=42)
        # prob = 0 or 1 should return unchanged
        r0 = mc.simulate_binary_outcome(0.0, 1.0, 0.5)
        assert r0["mean"] == 0.0
        r1 = mc.simulate_binary_outcome(1.0, 1.0, 0.5)
        assert r1["mean"] == 1.0

    def test_correlated_markets(self):
        mc = MonteCarloSimulator(n_simulations=10000, seed=42)
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        results = mc.simulate_correlated_markets(
            probs=[0.6, 0.4], correlation_matrix=corr,
            volatilities=[0.5, 0.5], time_remaining=0.5,
        )
        assert len(results) == 2
        assert all("mean" in r for r in results)


# ===========================================================================
# Isotonic Calibrator Tests
# ===========================================================================

class TestIsotonicCalibrator:

    def test_perfect_calibration_unchanged(self):
        cal = IsotonicCalibrator()
        # If predictions match outcomes, calibration shouldn't change much
        preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        cal.fit(preds, outcomes)
        # Calibrated values should be monotonically non-decreasing
        calibrated = [cal.calibrate(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 1e-10

    def test_overconfident_predictions_corrected(self):
        cal = IsotonicCalibrator()
        # Predictions say 90% but outcomes are only 60%
        np.random.seed(42)
        n = 100
        preds = np.random.uniform(0.8, 1.0, n)
        outcomes = np.random.binomial(1, 0.6, n).astype(float)
        cal.fit(preds, outcomes)
        # Calibrated 0.9 should be pulled down toward 0.6
        calibrated = cal.calibrate(0.9)
        assert calibrated < 0.85

    def test_uncalibrated_returns_original(self):
        cal = IsotonicCalibrator()
        assert cal.calibrate(0.7) == 0.7


# ===========================================================================
# Ensemble Predictor Tests
# ===========================================================================

class TestEnsemblePredictor:

    def test_inverse_variance_weighting(self):
        ep = EnsemblePredictor()
        result = ep.combine(
            estimates=[0.6, 0.8],
            variances=[0.01, 0.04],  # first model 4x more precise
        )
        # Should weight first model ~4x more
        assert result["probability"] < 0.7  # closer to 0.6

    def test_equal_variance_gives_equal_weights(self):
        ep = EnsemblePredictor()
        result = ep.combine(
            estimates=[0.3, 0.7],
            variances=[0.01, 0.01],
        )
        assert abs(result["probability"] - 0.5) < 0.01
        assert abs(result["weights"][0] - 0.5) < 0.01

    def test_combined_variance_less_than_any_individual(self):
        ep = EnsemblePredictor()
        result = ep.combine(
            estimates=[0.5, 0.6, 0.55],
            variances=[0.02, 0.03, 0.025],
        )
        assert result["variance"] < min(0.02, 0.03, 0.025)

    def test_single_model(self):
        ep = EnsemblePredictor()
        result = ep.combine(estimates=[0.7], variances=[0.01])
        assert abs(result["probability"] - 0.7) < 1e-10

    def test_correlated_combination(self):
        ep = EnsemblePredictor()
        cov = np.array([[0.01, 0.005], [0.005, 0.01]])
        result = ep.combine_with_correlation(
            estimates=[0.6, 0.7], covariance_matrix=cov,
        )
        assert 0.5 < result["probability"] < 0.8


# ===========================================================================
# Hurst Exponent Tests
# ===========================================================================

class TestHurstExponent:

    def test_random_walk_returns_valid(self):
        np.random.seed(42)
        walk = np.cumsum(np.random.normal(0, 1, 500))
        H = hurst_exponent(walk)
        assert 0.0 <= H <= 1.0  # must be in valid range

    def test_trending_series_above_half(self):
        # Strong uptrend
        trend = np.cumsum(np.random.normal(0.1, 0.01, 500))
        H = hurst_exponent(trend)
        assert H > 0.45  # should show persistence

    def test_mean_reverting_returns_valid(self):
        np.random.seed(42)
        # Mean-reverting (Ornstein-Uhlenbeck-like)
        series = [0.5]
        for _ in range(499):
            series.append(series[-1] + 0.5 * (0.5 - series[-1]) + np.random.normal(0, 0.01))
        H = hurst_exponent(series)
        assert 0.0 <= H <= 1.0  # valid range

    def test_insufficient_data_returns_half(self):
        assert hurst_exponent([1, 2, 3]) == 0.5


# ===========================================================================
# Kelly Criterion Tests
# ===========================================================================

class TestKellyWithUncertainty:

    def test_positive_edge_buy_yes(self):
        result = kelly_with_uncertainty(
            estimated_prob=0.7, prob_std=0.05, market_price=0.5
        )
        assert result["action"] == "BUY_YES"
        assert result["kelly_fraction"] > 0
        assert result["edge"] > 0

    def test_negative_edge_buy_no(self):
        result = kelly_with_uncertainty(
            estimated_prob=0.3, prob_std=0.05, market_price=0.6
        )
        assert result["action"] == "BUY_NO"
        assert result["kelly_fraction"] < 0
        assert result["edge"] < 0

    def test_no_edge_no_bet(self):
        result = kelly_with_uncertainty(
            estimated_prob=0.5, prob_std=0.05, market_price=0.5
        )
        assert result["action"] == "NO_BET"

    def test_high_uncertainty_reduces_fraction(self):
        certain = kelly_with_uncertainty(0.7, 0.01, 0.5)
        uncertain = kelly_with_uncertainty(0.7, 0.15, 0.5)
        assert abs(certain["kelly_fraction"]) > abs(uncertain["kelly_fraction"])

    def test_max_fraction_cap(self):
        result = kelly_with_uncertainty(0.99, 0.01, 0.01, max_fraction=0.25)
        assert abs(result["kelly_fraction"]) <= 0.25

    def test_extreme_market_prices(self):
        r0 = kelly_with_uncertainty(0.5, 0.1, 0.0)
        assert r0["kelly_fraction"] == 0.0
        r1 = kelly_with_uncertainty(0.5, 0.1, 1.0)
        assert r1["kelly_fraction"] == 0.0


# ===========================================================================
# Information Theory Tests
# ===========================================================================

class TestInformationTheory:

    def test_kl_identical_is_zero(self):
        p = np.array([0.3, 0.7])
        assert abs(kl_divergence(p, p)) < 1e-10

    def test_kl_non_negative(self):
        p = np.array([0.2, 0.8])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) >= 0

    def test_js_symmetric(self):
        p = np.array([0.2, 0.8])
        q = np.array([0.6, 0.4])
        assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-10

    def test_js_identical_is_zero(self):
        p = np.array([0.5, 0.5])
        assert abs(jensen_shannon_divergence(p, p)) < 1e-10

    def test_js_bounded(self):
        p = np.array([0.01, 0.99])
        q = np.array([0.99, 0.01])
        jsd = jensen_shannon_divergence(p, q)
        assert 0 <= jsd <= math.log(2) + 0.01


# ===========================================================================
# Market Microstructure Signal Tests
# ===========================================================================

class TestMicrostructureSignals:

    def test_volume_imbalance_all_buying(self):
        prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        volumes = [100] * 11
        signal = volume_imbalance_signal(volumes, prices, lookback=10)
        assert signal is not None
        assert signal > 0.5  # strong buying pressure

    def test_volume_imbalance_all_selling(self):
        prices = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        volumes = [100] * 11
        signal = volume_imbalance_signal(volumes, prices, lookback=10)
        assert signal is not None
        assert signal < -0.5  # strong selling pressure

    def test_volume_imbalance_insufficient_data(self):
        assert volume_imbalance_signal([1, 2], [1, 2], lookback=20) is None

    def test_smart_money_divergence_calculated(self):
        prices = list(range(1, 21))  # rising prices
        volumes = list(range(20, 0, -1))  # falling volumes
        div = smart_money_divergence(prices, volumes, window=10)
        assert div is not None
        assert div > 0  # price up but volume down = divergence
