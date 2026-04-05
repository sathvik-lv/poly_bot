"""Advanced statistical models for prediction market edge detection.

Implements:
- Bayesian hierarchical updating with conjugate priors (Beta-Binomial)
- Hidden Markov Model for regime detection
- GARCH(1,1) volatility forecasting
- Copula-based dependency modeling
- Monte Carlo simulation with antithetic variates
- Isotonic regression for probability calibration
- Ensemble model with inverse-variance weighting
- Kelly criterion with uncertainty adjustment
- Hurst exponent for mean-reversion vs trend detection
- Information-theoretic divergence measures (KL, Jensen-Shannon)
"""

import math
import random
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


# ===========================================================================
# Beta-Binomial Bayesian Model (conjugate prior for binary outcomes)
# ===========================================================================

class BetaBinomialModel:
    """Bayesian model using Beta conjugate prior for binary prediction markets.

    More principled than naive Bayes — the Beta distribution is the natural
    conjugate prior for Bernoulli/Binomial likelihoods, giving closed-form
    posterior updates and proper uncertainty quantification.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize with prior Beta(alpha, beta). Default = uniform prior."""
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Bayesian credible interval (not frequentist CI)."""
        tail = (1 - confidence) / 2
        low = sp_stats.beta.ppf(tail, self.alpha, self.beta)
        high = sp_stats.beta.ppf(1 - tail, self.alpha, self.beta)
        return (float(low), float(high))

    def update(self, successes: int, failures: int) -> "BetaBinomialModel":
        """Posterior update: Beta(a + s, b + f)."""
        return BetaBinomialModel(self.alpha + successes, self.beta + failures)

    def update_with_weight(self, observation: float, weight: float = 1.0) -> "BetaBinomialModel":
        """Soft update with fractional evidence (e.g., from noisy signals)."""
        return BetaBinomialModel(
            self.alpha + observation * weight,
            self.beta + (1 - observation) * weight,
        )

    def log_marginal_likelihood(self, successes: int, total: int) -> float:
        """Log marginal likelihood P(data | model) for model comparison."""
        failures = total - successes
        return (
            betaln(self.alpha + successes, self.beta + failures)
            - betaln(self.alpha, self.beta)
        )

    def kl_divergence_from(self, other: "BetaBinomialModel") -> float:
        """KL(self || other) — how much info is lost using `other` to approx `self`."""
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        return (
            betaln(a2, b2) - betaln(a1, b1)
            + (a1 - a2) * (sp_stats.beta.mean(a1, b1) - sp_stats.beta.mean(a2, b2))
            + (b1 - b2) * (
                float(sp_stats.digamma(b1)) - float(sp_stats.digamma(a1 + b1))
            )
        )

    def posterior_predictive(self) -> float:
        """P(next outcome = 1) under posterior predictive."""
        return self.mean


# ===========================================================================
# Hidden Markov Model — Regime Detection
# ===========================================================================

class GaussianHMM:
    """2-state Gaussian HMM for detecting market regime shifts.

    States represent e.g. "trending" vs "mean-reverting" regimes,
    or "high volatility" vs "low volatility" periods.
    Uses Baum-Welch (EM) for parameter estimation.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 50, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.transition_matrix: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None
        self.initial_probs: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        """Fit HMM parameters using Baum-Welch (EM algorithm)."""
        obs = np.asarray(observations, dtype=np.float64).ravel()
        T = len(obs)
        K = self.n_states

        # Initialize parameters via K-means-style splitting
        sorted_obs = np.sort(obs)
        chunk = T // K
        self.means = np.array([sorted_obs[i * chunk:(i + 1) * chunk].mean() for i in range(K)])
        self.variances = np.full(K, np.var(obs) + 1e-6)
        self.transition_matrix = np.full((K, K), 1.0 / K)
        self.initial_probs = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            # E-step: forward-backward
            log_emission = self._log_emission_probs(obs)
            log_alpha = self._forward(log_emission)
            log_beta = self._backward(log_emission)

            ll = self._logsumexp(log_alpha[-1])
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # Posterior state probabilities
            log_gamma = log_alpha + log_beta
            log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Transition posteriors
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(self.transition_matrix[i, j] + 1e-300)
                            + log_emission[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                xi[t] = np.exp(xi[t] - self._logsumexp(xi[t].ravel()))

            # M-step
            self.initial_probs = gamma[0] / gamma[0].sum()
            for i in range(K):
                self.transition_matrix[i] = xi[:, i, :].sum(axis=0) + 1e-10
                self.transition_matrix[i] /= self.transition_matrix[i].sum()
                weight = gamma[:, i]
                total_weight = weight.sum() + 1e-10
                self.means[i] = (weight * obs).sum() / total_weight
                self.variances[i] = (weight * (obs - self.means[i]) ** 2).sum() / total_weight + 1e-6

        self._fitted = True
        return self

    def predict_regime(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi decoding — most likely state sequence."""
        obs = np.asarray(observations, dtype=np.float64).ravel()
        T = len(obs)
        K = self.n_states
        log_emission = self._log_emission_probs(obs)

        viterbi = np.zeros((T, K))
        backptr = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(self.initial_probs + 1e-300) + log_emission[0]
        for t in range(1, T):
            for j in range(K):
                scores = viterbi[t - 1] + np.log(self.transition_matrix[:, j] + 1e-300)
                backptr[t, j] = int(np.argmax(scores))
                viterbi[t, j] = scores[backptr[t, j]] + log_emission[t, j]

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(viterbi[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]
        return path

    def regime_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Smoothed posterior P(state | all observations)."""
        obs = np.asarray(observations, dtype=np.float64).ravel()
        log_emission = self._log_emission_probs(obs)
        log_alpha = self._forward(log_emission)
        log_beta = self._backward(log_emission)
        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def _log_emission_probs(self, obs: np.ndarray) -> np.ndarray:
        T = len(obs)
        K = self.n_states
        log_e = np.zeros((T, K))
        for k in range(K):
            log_e[:, k] = sp_stats.norm.logpdf(obs, loc=self.means[k], scale=np.sqrt(self.variances[k]))
        return log_e

    def _forward(self, log_emission: np.ndarray) -> np.ndarray:
        T, K = log_emission.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.initial_probs + 1e-300) + log_emission[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    self._logsumexp(log_alpha[t - 1] + np.log(self.transition_matrix[:, j] + 1e-300))
                    + log_emission[t, j]
                )
        return log_alpha

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        T, K = log_emission.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for j in range(K):
                log_beta[t, j] = self._logsumexp(
                    np.log(self.transition_matrix[j, :] + 1e-300) + log_emission[t + 1] + log_beta[t + 1]
                )
        return log_beta

    @staticmethod
    def _logsumexp(a, axis=None, keepdims=False):
        a = np.asarray(a)
        a_max = np.max(a, axis=axis, keepdims=True)
        result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims)) + a_max.squeeze(axis=axis) if not keepdims else np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims)) + a_max
        return result


# ===========================================================================
# GARCH(1,1) Volatility Forecasting
# ===========================================================================

class GARCH:
    """GARCH(1,1) model for volatility forecasting.

    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

    Stationarity requires: alpha + beta < 1
    """

    def __init__(self):
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta_param: float = 0.0
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "GARCH":
        """Fit GARCH(1,1) via maximum likelihood."""
        returns = np.asarray(returns, dtype=np.float64).ravel()
        T = len(returns)
        if T < 10:
            raise ValueError("Need at least 10 observations for GARCH")

        sample_var = np.var(returns)

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.zeros(T)
            sigma2[0] = sample_var
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns ** 2 / sigma2)
            return -ll

        # Initial guess
        x0 = [sample_var * 0.05, 0.1, 0.85]
        bounds = [(1e-8, None), (1e-8, 0.5), (0.5, 0.9999)]
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        self.omega, self.alpha, self.beta_param = result.x
        self._fitted = True
        return self

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast conditional variance for `horizon` steps ahead."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        returns = np.asarray(returns, dtype=np.float64).ravel()

        # Compute current conditional variance
        sigma2 = np.var(returns)
        for t in range(1, len(returns)):
            sigma2 = self.omega + self.alpha * returns[t] ** 2 + self.beta_param * sigma2

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            sigma2 = self.omega + (self.alpha + self.beta_param) * sigma2
            forecasts[h] = sigma2
        return forecasts

    @property
    def persistence(self) -> float:
        """alpha + beta: closer to 1 = more persistent volatility."""
        return self.alpha + self.beta_param

    @property
    def unconditional_variance(self) -> float:
        """Long-run variance: omega / (1 - alpha - beta)."""
        denom = 1 - self.alpha - self.beta_param
        if denom <= 0:
            return float("inf")
        return self.omega / denom


# ===========================================================================
# Monte Carlo Simulation with Variance Reduction
# ===========================================================================

class MonteCarloSimulator:
    """Monte Carlo engine with antithetic variates for variance reduction."""

    def __init__(self, n_simulations: int = 50000, seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)

    def simulate_binary_outcome(
        self,
        base_prob: float,
        volatility: float,
        time_remaining_frac: float,
        drift: float = 0.0,
    ) -> dict:
        """Simulate binary outcome probability evolution.

        Uses geometric Brownian motion on log-odds with antithetic variates.

        Args:
            base_prob: Current probability estimate
            volatility: Annualized volatility of probability
            time_remaining_frac: Fraction of time remaining (0-1)
            drift: Directional bias in probability movement

        Returns:
            Dict with simulated stats and confidence intervals
        """
        if base_prob <= 0 or base_prob >= 1:
            return {"mean": base_prob, "std": 0.0, "ci_95": (base_prob, base_prob)}

        # Log-odds transform for unconstrained simulation
        log_odds = math.log(base_prob / (1 - base_prob))

        n_half = self.n_simulations // 2
        dt = time_remaining_frac

        # Standard + antithetic variates
        z = self.rng.standard_normal(n_half)
        z_anti = -z  # antithetic
        z_all = np.concatenate([z, z_anti])

        # Simulate log-odds evolution
        simulated_log_odds = log_odds + drift * dt + volatility * math.sqrt(dt) * z_all

        # Transform back to probability
        simulated_probs = 1.0 / (1.0 + np.exp(-simulated_log_odds))

        mean_prob = float(np.mean(simulated_probs))
        std_prob = float(np.std(simulated_probs))
        ci_low = float(np.percentile(simulated_probs, 2.5))
        ci_high = float(np.percentile(simulated_probs, 97.5))

        # Probability of resolving Yes
        p_yes = float(np.mean(simulated_probs > 0.5))

        return {
            "mean": mean_prob,
            "std": std_prob,
            "ci_95": (ci_low, ci_high),
            "ci_99": (float(np.percentile(simulated_probs, 0.5)), float(np.percentile(simulated_probs, 99.5))),
            "p_resolve_yes": p_yes,
            "skewness": float(sp_stats.skew(simulated_probs)),
            "kurtosis": float(sp_stats.kurtosis(simulated_probs)),
            "var_95": float(np.percentile(simulated_probs, 5)),  # Value at Risk
        }

    def simulate_correlated_markets(
        self,
        probs: list[float],
        correlation_matrix: np.ndarray,
        volatilities: list[float],
        time_remaining: float,
    ) -> list[dict]:
        """Simulate multiple correlated markets using Cholesky decomposition."""
        n_markets = len(probs)
        L = np.linalg.cholesky(correlation_matrix)

        z_independent = self.rng.standard_normal((self.n_simulations, n_markets))
        z_correlated = z_independent @ L.T

        results = []
        for i in range(n_markets):
            if probs[i] <= 0 or probs[i] >= 1:
                results.append({"mean": probs[i], "std": 0.0})
                continue
            log_odds = math.log(probs[i] / (1 - probs[i]))
            sim_lo = log_odds + volatilities[i] * math.sqrt(time_remaining) * z_correlated[:, i]
            sim_p = 1.0 / (1.0 + np.exp(-sim_lo))
            results.append({
                "mean": float(np.mean(sim_p)),
                "std": float(np.std(sim_p)),
                "ci_95": (float(np.percentile(sim_p, 2.5)), float(np.percentile(sim_p, 97.5))),
            })
        return results


# ===========================================================================
# Probability Calibration (Isotonic Regression)
# ===========================================================================

class IsotonicCalibrator:
    """Calibrate raw probability estimates using isotonic regression.

    Markets are often miscalibrated — events with 70% market price
    don't happen 70% of the time. This corrects systematic bias.
    """

    def __init__(self):
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit calibration curve from historical predictions vs outcomes.

        Uses Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
        """
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
        actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

        order = np.argsort(predicted_probs)
        self._x = predicted_probs[order]
        y_sorted = actual_outcomes[order]

        # PAVA (Pool Adjacent Violators)
        self._y = self._pava(y_sorted)
        self._fitted = True
        return self

    def calibrate(self, prob: float) -> float:
        """Map a raw probability to a calibrated probability."""
        if not self._fitted:
            return prob
        idx = np.searchsorted(self._x, prob, side="right") - 1
        idx = max(0, min(idx, len(self._y) - 1))
        return float(self._y[idx])

    @staticmethod
    def _pava(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm."""
        n = len(y)
        result = y.copy().astype(float)
        blocks = [[i] for i in range(n)]

        changed = True
        while changed:
            changed = False
            i = 0
            new_blocks = []
            while i < len(blocks):
                if i + 1 < len(blocks):
                    mean_curr = np.mean(result[blocks[i]])
                    mean_next = np.mean(result[blocks[i + 1]])
                    if mean_curr > mean_next:
                        merged = blocks[i] + blocks[i + 1]
                        merged_mean = np.mean(result[merged])
                        for idx in merged:
                            result[idx] = merged_mean
                        new_blocks.append(merged)
                        i += 2
                        changed = True
                        continue
                new_blocks.append(blocks[i])
                i += 1
            blocks = new_blocks
        return result


# ===========================================================================
# Hurst Exponent — Mean Reversion vs Trend Detection
# ===========================================================================

def hurst_exponent(series: list[float] | np.ndarray) -> float:
    """Estimate Hurst exponent using R/S analysis.

    H < 0.5: mean-reverting (anti-persistent) — fade the move
    H = 0.5: random walk — no edge from price history
    H > 0.5: trending (persistent) — follow the move

    This is critical for knowing whether to use momentum or
    mean-reversion strategies on a given market.
    """
    series = np.asarray(series, dtype=np.float64)
    N = len(series)
    if N < 20:
        return 0.5  # insufficient data

    max_k = min(N // 2, 100)
    min_k = 8
    if max_k <= min_k:
        return 0.5

    lag_sizes = []
    rs_values = []

    for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)):
        n_blocks = N // k
        if n_blocks < 1:
            continue
        rs_block = []
        for b in range(n_blocks):
            block = series[b * k:(b + 1) * k]
            mean_block = np.mean(block)
            deviations = np.cumsum(block - mean_block)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)
        if rs_block:
            lag_sizes.append(k)
            rs_values.append(np.mean(rs_block))

    if len(lag_sizes) < 3:
        return 0.5

    log_lags = np.log(lag_sizes)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = sp_stats.linregress(log_lags, log_rs)
    return float(np.clip(slope, 0.0, 1.0))


# ===========================================================================
# Information-Theoretic Measures
# ===========================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) — information lost when Q approximates P."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Add small epsilon to avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — symmetric, bounded [0, ln(2)]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ===========================================================================
# Ensemble Model with Inverse-Variance Weighting
# ===========================================================================

class EnsemblePredictor:
    """Combine multiple probability estimates using inverse-variance weighting.

    This is the optimal linear combination under the assumption
    that individual estimates are unbiased with known variance.
    """

    def combine(
        self,
        estimates: list[float],
        variances: list[float],
    ) -> dict:
        """Combine estimates with inverse-variance weighting.

        Args:
            estimates: Probability estimates from different models
            variances: Variance (uncertainty) of each estimate

        Returns:
            Combined estimate with uncertainty
        """
        estimates = np.asarray(estimates, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)

        # Avoid division by zero
        variances = np.clip(variances, 1e-10, None)
        weights = 1.0 / variances
        weights /= weights.sum()

        combined_mean = float(np.dot(weights, estimates))
        combined_var = float(1.0 / np.sum(1.0 / variances))

        return {
            "probability": np.clip(combined_mean, 0.0, 1.0),
            "variance": combined_var,
            "std": math.sqrt(combined_var),
            "ci_95": (
                max(0, combined_mean - 1.96 * math.sqrt(combined_var)),
                min(1, combined_mean + 1.96 * math.sqrt(combined_var)),
            ),
            "weights": weights.tolist(),
            "n_models": len(estimates),
        }

    def combine_with_correlation(
        self,
        estimates: list[float],
        covariance_matrix: np.ndarray,
    ) -> dict:
        """GLS-style combination accounting for cross-model correlations."""
        estimates = np.asarray(estimates, dtype=np.float64)
        cov = np.asarray(covariance_matrix, dtype=np.float64)

        # Regularize covariance
        cov += np.eye(len(estimates)) * 1e-8

        cov_inv = np.linalg.inv(cov)
        ones = np.ones(len(estimates))
        denom = float(ones @ cov_inv @ ones)
        weights = (cov_inv @ ones) / denom

        combined_mean = float(weights @ estimates)
        combined_var = float(1.0 / denom)

        return {
            "probability": np.clip(combined_mean, 0.0, 1.0),
            "variance": combined_var,
            "std": math.sqrt(combined_var),
            "weights": weights.tolist(),
        }


# ===========================================================================
# Kelly Criterion with Uncertainty Adjustment
# ===========================================================================

def kelly_with_uncertainty(
    estimated_prob: float,
    prob_std: float,
    market_price: float,
    max_fraction: float = 0.25,
) -> dict:
    """Kelly criterion adjusted for estimation uncertainty.

    Standard Kelly assumes you know the true probability. In reality,
    your estimate has uncertainty. This penalizes Kelly fraction
    proportionally to your uncertainty, reducing overbetting risk.

    Also applies fractional Kelly (default: cap at 25% of bankroll).
    """
    if market_price <= 0 or market_price >= 1:
        return {"kelly_fraction": 0.0, "action": "NO_BET", "edge": 0.0}

    # Win payout ratio
    b = (1 - market_price) / market_price  # payout per dollar risked

    # Standard Kelly
    f_full = (estimated_prob * b - (1 - estimated_prob)) / b

    # Uncertainty penalty: reduce Kelly by ratio of uncertainty to edge
    edge = estimated_prob - market_price
    if abs(edge) < 1e-10:
        return {"kelly_fraction": 0.0, "action": "NO_BET", "edge": 0.0}

    # Shrink Kelly when uncertainty is high relative to edge
    uncertainty_ratio = prob_std / abs(edge) if abs(edge) > 0 else float("inf")
    shrinkage = 1.0 / (1.0 + uncertainty_ratio)
    f_adjusted = f_full * shrinkage

    # Cap at max_fraction
    f_final = float(np.clip(f_adjusted, -max_fraction, max_fraction))

    # Determine action
    if f_final > 0.005:
        action = "BUY_YES"
    elif f_final < -0.005:
        action = "BUY_NO"
    else:
        action = "NO_BET"

    # Expected value per dollar
    ev = estimated_prob * (1 - market_price) - (1 - estimated_prob) * market_price

    return {
        "kelly_fraction": round(f_final, 6),
        "kelly_full": round(f_full, 6),
        "shrinkage_factor": round(shrinkage, 4),
        "action": action,
        "edge": round(edge, 6),
        "expected_value": round(ev, 6),
        "uncertainty_ratio": round(uncertainty_ratio, 4),
    }


# ===========================================================================
# Market Microstructure Edge Signals
# ===========================================================================

def volume_imbalance_signal(
    volumes: list[float], prices: list[float], lookback: int = 20
) -> Optional[float]:
    """Detect volume-weighted directional imbalance.

    Returns signal in [-1, 1]:
      > 0 = buying pressure (bullish)
      < 0 = selling pressure (bearish)
    """
    if len(volumes) < lookback or len(prices) < lookback:
        return None

    volumes = np.asarray(volumes[-lookback:], dtype=np.float64)
    prices = np.asarray(prices[-lookback:], dtype=np.float64)

    price_changes = np.diff(prices)
    vol_slice = volumes[1:]  # align with changes

    up_vol = np.sum(vol_slice[price_changes > 0])
    down_vol = np.sum(vol_slice[price_changes < 0])
    total = up_vol + down_vol

    if total == 0:
        return 0.0
    return float((up_vol - down_vol) / total)


def smart_money_divergence(
    prices: list[float], volumes: list[float], window: int = 10
) -> Optional[float]:
    """Detect divergence between price trend and volume trend.

    When price rises but volume falls (or vice versa), the trend
    is weakening — "smart money" is exiting. Returns divergence score.
    """
    if len(prices) < window + 1 or len(volumes) < window + 1:
        return None

    recent_prices = prices[-window:]
    recent_volumes = volumes[-window:]

    # Price trend (linear regression slope)
    x = np.arange(window)
    price_slope, _, _, _, _ = sp_stats.linregress(x, recent_prices)
    vol_slope, _, _, _, _ = sp_stats.linregress(x, recent_volumes)

    # Normalize slopes
    price_norm = price_slope / (np.std(recent_prices) + 1e-10)
    vol_norm = vol_slope / (np.std(recent_volumes) + 1e-10)

    # Divergence = difference in direction/magnitude
    return float(price_norm - vol_norm)
