"""Polymarket Prediction Engine — Maximum Accuracy Ensemble.

Combines multiple independent signal sources with advanced statistical
models to produce calibrated probability estimates with edge detection.

Architecture:
    1. Market Microstructure Analyzer — bid/ask dynamics, volume patterns
    2. Statistical Time Series Models — HMM regime, GARCH vol, Hurst exponent
    3. Multi-Source Data Fusion — crypto, fear/greed, economic indicators
    4. AI Semantic Analyzer — OpenRouter LLM for question understanding
    5. Bayesian Ensemble — inverse-variance weighted combination
    6. Calibration Layer — isotonic regression to debias
    7. Kelly Sizing — uncertainty-adjusted position sizing

Each sub-model produces (estimate, variance). The ensemble optimally
combines them, then calibration corrects systematic bias, and Kelly
determines position size given your edge and uncertainty.
"""

import math
import os
import json
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from scipy import stats as sp_stats

from src.market_client import MarketClient
from src.data_sources import CryptoDataSource, FearGreedIndex, EconomicDataSource, NewsDataSource
from src.statistics import (
    BetaBinomialModel,
    GaussianHMM,
    GARCH,
    MonteCarloSimulator,
    IsotonicCalibrator,
    EnsemblePredictor,
    hurst_exponent,
    jensen_shannon_divergence,
    kelly_with_uncertainty,
    volume_imbalance_signal,
    smart_money_divergence,
)


# ===========================================================================
# Sub-Model: Market Microstructure Signal
# ===========================================================================

class MicrostructureModel:
    """Extract edge signals from market microstructure data."""

    def analyze(self, market: dict) -> dict:
        """Analyze bid/ask, spread, volume for directional signals.

        Returns:
            estimate: probability bias from microstructure
            variance: uncertainty of the signal
            signals: dict of individual signal values
        """
        bid = market.get("best_bid")
        ask = market.get("best_ask")
        last = market.get("last_trade_price")
        spread = market.get("spread")
        volume_24h = market.get("volume_24h", 0) or 0
        liquidity = market.get("liquidity", 0) or 0

        signals = {}
        adjustments = []

        # 1. Bid-ask midpoint vs last trade — directional pressure
        if bid is not None and ask is not None:
            midpoint = (bid + ask) / 2
            if last is not None and midpoint > 0:
                trade_vs_mid = (last - midpoint) / midpoint
                signals["trade_vs_midpoint"] = trade_vs_mid
                # If last trade > midpoint, buying pressure
                adjustments.append(trade_vs_mid * 0.3)

        # 2. Spread analysis — tight spread = efficient pricing
        if spread is not None and ask is not None and ask > 0:
            spread_pct = spread / ask
            signals["spread_pct"] = spread_pct
            # Wide spread = more uncertainty, reduce confidence
            signals["market_efficiency"] = 1.0 - min(spread_pct * 10, 1.0)

        # 3. Volume/liquidity ratio — high vol relative to liquidity = conviction
        if liquidity > 0:
            vol_liq_ratio = volume_24h / liquidity
            signals["volume_liquidity_ratio"] = vol_liq_ratio
            # High ratio = strong conviction in current price
            signals["conviction"] = min(vol_liq_ratio / 10, 1.0)

        # Base estimate from last trade price
        base = last if last is not None else (bid + ask) / 2 if bid and ask else 0.5
        adjustment = sum(adjustments) if adjustments else 0
        estimate = np.clip(base + adjustment, 0.01, 0.99)

        # Variance inversely proportional to market efficiency
        efficiency = signals.get("market_efficiency", 0.5)
        conviction = signals.get("conviction", 0.5)
        variance = 0.05 * (1 - efficiency * 0.5) * (1 - conviction * 0.3)

        return {
            "estimate": float(estimate),
            "variance": float(max(variance, 0.001)),
            "signals": signals,
            "model": "microstructure",
        }


# ===========================================================================
# Sub-Model: Time Series Statistical Model
# ===========================================================================

class TimeSeriesModel:
    """HMM regime detection + GARCH volatility + Hurst mean-reversion."""

    def __init__(self):
        self.hmm = GaussianHMM(n_states=2, n_iter=30)
        self.garch = GARCH()
        self.mc = MonteCarloSimulator(n_simulations=20000, seed=42)

    def analyze(self, price_history: list[float], current_price: float, time_remaining_frac: float) -> dict:
        """Full time-series analysis.

        Args:
            price_history: Historical prices (oldest first)
            current_price: Current market price
            time_remaining_frac: Fraction of time remaining before resolution

        Returns:
            estimate, variance, and detailed signals
        """
        if len(price_history) < 20:
            return {
                "estimate": current_price,
                "variance": 0.05,
                "signals": {"insufficient_data": True},
                "model": "time_series",
            }

        prices = np.array(price_history, dtype=np.float64)
        returns = np.diff(np.log(np.clip(prices, 0.001, 0.999)))

        signals = {}

        # 1. Hurst exponent — is this market trending or mean-reverting?
        H = hurst_exponent(prices)
        signals["hurst_exponent"] = H
        signals["regime_type"] = "trending" if H > 0.55 else "mean_reverting" if H < 0.45 else "random_walk"

        # 2. HMM regime detection
        try:
            self.hmm.fit(returns)
            regime_probs = self.hmm.regime_probabilities(returns)
            current_regime = int(np.argmax(regime_probs[-1]))
            signals["current_regime"] = current_regime
            signals["regime_confidence"] = float(regime_probs[-1].max())
            # Regime means tell us directional bias
            regime_mean = float(self.hmm.means[current_regime])
            signals["regime_drift"] = regime_mean
        except Exception:
            regime_mean = 0.0
            signals["hmm_failed"] = True

        # 3. GARCH volatility forecast
        try:
            self.garch.fit(returns)
            vol_forecast = self.garch.forecast(returns, horizon=5)
            signals["garch_vol_1step"] = float(np.sqrt(vol_forecast[0]))
            signals["garch_vol_5step"] = float(np.sqrt(vol_forecast[-1]))
            signals["garch_persistence"] = self.garch.persistence
            annualized_vol = float(np.sqrt(vol_forecast[0] * 252))
        except Exception:
            annualized_vol = float(np.std(returns) * np.sqrt(252))
            signals["garch_failed"] = True

        # 4. Monte Carlo simulation for probability evolution
        drift = regime_mean * 252 if abs(regime_mean) > 1e-6 else 0.0
        mc_result = self.mc.simulate_binary_outcome(
            base_prob=current_price,
            volatility=annualized_vol,
            time_remaining_frac=time_remaining_frac,
            drift=drift,
        )
        signals["mc_mean"] = mc_result["mean"]
        signals["mc_std"] = mc_result["std"]
        signals["mc_ci_95"] = mc_result["ci_95"]
        signals["mc_p_resolve_yes"] = mc_result["p_resolve_yes"]
        signals["mc_skewness"] = mc_result["skewness"]

        # 5. Mean reversion signal (if Hurst < 0.5)
        if H < 0.5:
            # Price tends to revert to long-run mean
            long_run_mean = float(np.mean(prices[-50:] if len(prices) >= 50 else prices))
            reversion_pull = (long_run_mean - current_price) * (0.5 - H) * 2
            signals["mean_reversion_pull"] = reversion_pull
            estimate = current_price + reversion_pull * 0.3
        elif H > 0.55:
            # Trending — project recent momentum
            recent_momentum = float(np.mean(returns[-10:])) * 10
            signals["momentum_signal"] = recent_momentum
            estimate = current_price + recent_momentum * 0.2
        else:
            estimate = mc_result["mean"]

        estimate = float(np.clip(estimate, 0.01, 0.99))
        variance = mc_result["std"] ** 2

        return {
            "estimate": estimate,
            "variance": max(variance, 0.001),
            "signals": signals,
            "model": "time_series",
        }


# ===========================================================================
# Sub-Model: External Data Fusion
# ===========================================================================

class ExternalDataModel:
    """Fuse external data sources into probability adjustments."""

    def __init__(self):
        self.crypto = CryptoDataSource()
        self.fear_greed = FearGreedIndex()
        self.econ = EconomicDataSource()

    def analyze(self, market: dict, market_keywords: list[str]) -> dict:
        """Pull external data and compute probability adjustments.

        Args:
            market: Parsed market data
            market_keywords: Keywords extracted from market question
        """
        signals = {}
        adjustments = []

        # 1. Fear & Greed Index — contrarian signal
        try:
            fg = self.fear_greed.get_current()
            if fg:
                fg_value = fg["value"]
                signals["fear_greed_value"] = fg_value
                signals["fear_greed_class"] = fg["classification"]

                # Extreme fear = contrarian bullish, extreme greed = contrarian bearish
                if fg_value < 20:
                    adjustments.append(("fear_greed_contrarian", 0.05, 0.01))
                elif fg_value > 80:
                    adjustments.append(("fear_greed_contrarian", -0.05, 0.01))
                else:
                    adjustments.append(("fear_greed_neutral", 0, 0.02))
        except Exception:
            signals["fear_greed_failed"] = True

        # 2. Crypto market data (if market is crypto-related)
        crypto_keywords = {"bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "xrp"}
        is_crypto = bool(crypto_keywords & set(k.lower() for k in market_keywords))

        if is_crypto:
            try:
                # Map keywords to coin IDs
                coin_map = {
                    "bitcoin": "bitcoin", "btc": "bitcoin",
                    "ethereum": "ethereum", "eth": "ethereum",
                    "solana": "solana", "sol": "solana",
                    "xrp": "ripple",
                }
                coin_id = "bitcoin"
                for kw in market_keywords:
                    if kw.lower() in coin_map:
                        coin_id = coin_map[kw.lower()]
                        break

                market_data = self.crypto.get_market_data(coin_id)
                if market_data:
                    signals["crypto_price"] = market_data.get("current_price")
                    signals["crypto_24h_change"] = market_data.get("price_change_24h_pct")
                    signals["crypto_7d_change"] = market_data.get("price_change_7d_pct")
                    signals["crypto_30d_change"] = market_data.get("price_change_30d_pct")

                    # Strong recent moves increase confidence in directional markets
                    change_24h = market_data.get("price_change_24h_pct", 0) or 0
                    if abs(change_24h) > 5:
                        direction = 1 if change_24h > 0 else -1
                        adjustments.append(("crypto_momentum", direction * 0.03, 0.015))

                # Price history for volatility context
                history = self.crypto.get_price_history(coin_id, days=30)
                if len(history) > 10:
                    hist_prices = [p[1] for p in history]
                    returns = np.diff(np.log(hist_prices))
                    signals["crypto_realized_vol_30d"] = float(np.std(returns) * np.sqrt(365))
            except Exception:
                signals["crypto_data_failed"] = True

        # Combine adjustments
        total_adjustment = sum(a[1] for a in adjustments)
        total_variance = sum(a[2] for a in adjustments) if adjustments else 0.03

        # Base estimate is market price adjusted by external signals
        prices = market.get("outcome_prices", {})
        base = prices.get("Yes", 0.5) if "Yes" in prices else 0.5
        estimate = float(np.clip(base + total_adjustment, 0.01, 0.99))

        return {
            "estimate": estimate,
            "variance": max(total_variance, 0.005),
            "signals": signals,
            "adjustments": [(a[0], a[1]) for a in adjustments],
            "model": "external_data",
        }


# ===========================================================================
# Sub-Model: AI Semantic Analysis (OpenRouter)
# ===========================================================================

class AISemanticModel:
    """Use LLM via OpenRouter for semantic understanding of market questions.

    Analyzes the question text, resolution criteria, and context
    to produce a probability estimate grounded in reasoning.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: str = "google/gemini-2.0-flash-exp:free"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.enabled = bool(self.api_key)

    def analyze(self, market: dict) -> dict:
        """Ask LLM to estimate probability for a market question.

        Uses structured prompting with calibration instructions
        to get the most accurate probability estimate possible.
        """
        if not self.enabled:
            return {
                "estimate": None,
                "variance": None,
                "signals": {"ai_disabled": True},
                "model": "ai_semantic",
            }

        question = market.get("question", "")
        current_price = market.get("outcome_prices", {}).get("Yes", 0.5)

        prompt = f"""You are an expert superforecaster and prediction market analyst.
Your calibration is excellent — when you say 70%, events happen 70% of the time.

MARKET QUESTION: {question}

CURRENT MARKET PRICE: {current_price} (this is the market's implied probability)

TODAY'S DATE: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

END DATE: {market.get('end_date', 'Unknown')}

Analyze this prediction market question. Consider:
1. Base rates for similar events
2. Current geopolitical/economic context
3. Time remaining until resolution
4. Whether the market price seems efficient or mispriced

Respond with ONLY a JSON object (no other text):
{{
    "probability": <your estimate 0.01-0.99>,
    "confidence": <how confident you are in your estimate, 0.1-1.0>,
    "reasoning": "<brief 1-2 sentence reasoning>",
    "edge_vs_market": <your probability minus market price>
}}"""

        try:
            resp = requests.post(
                self.OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle potential markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            estimate = float(parsed["probability"])
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            edge = parsed.get("edge_vs_market", estimate - current_price)

            # Variance inversely proportional to confidence
            variance = 0.05 * (1 - confidence * 0.8)

            return {
                "estimate": np.clip(estimate, 0.01, 0.99),
                "variance": max(variance, 0.005),
                "signals": {
                    "ai_probability": estimate,
                    "ai_confidence": confidence,
                    "ai_reasoning": reasoning,
                    "ai_edge": edge,
                    "ai_model": self.model,
                },
                "model": "ai_semantic",
            }
        except Exception as e:
            return {
                "estimate": None,
                "variance": None,
                "signals": {"ai_error": str(e)},
                "model": "ai_semantic",
            }


# ===========================================================================
# Master Prediction Engine
# ===========================================================================

class PredictionEngine:
    """Master ensemble prediction engine.

    Pipeline:
        1. Fetch & parse market data
        2. Run all sub-models in parallel-ready fashion
        3. Ensemble combine with inverse-variance weighting
        4. Apply calibration correction
        5. Monte Carlo for final uncertainty quantification
        6. Kelly criterion for position sizing

    The engine produces a PredictionResult with:
        - Calibrated probability estimate
        - Confidence interval
        - Edge vs market
        - Recommended action and size
        - Full signal breakdown for transparency
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        ai_model: str = "google/gemini-2.0-flash-exp:free",
        calibration_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.market_client = MarketClient()
        self.microstructure = MicrostructureModel()
        self.time_series = TimeSeriesModel()
        self.external_data = ExternalDataModel()
        self.ai_model = AISemanticModel(api_key=openrouter_api_key, model=ai_model)
        self.ensemble = EnsemblePredictor()
        self.mc = MonteCarloSimulator(n_simulations=30000, seed=None)

        # Calibration layer
        self.calibrator = IsotonicCalibrator()
        if calibration_data is not None:
            preds, outcomes = calibration_data
            self.calibrator.fit(preds, outcomes)
            self._calibrated = True
        else:
            self._calibrated = False

    def predict(
        self,
        market_id: Optional[str] = None,
        market_data: Optional[dict] = None,
        price_history: Optional[list[float]] = None,
        time_remaining_frac: float = 0.5,
    ) -> dict:
        """Generate a full prediction for a market.

        Args:
            market_id: Polymarket market ID (will fetch if market_data not provided)
            market_data: Pre-parsed market data dict
            price_history: Historical price series (if available)
            time_remaining_frac: Fraction of time remaining (0=expired, 1=just opened)

        Returns:
            Comprehensive prediction result dict
        """
        # Step 1: Get market data
        if market_data is None and market_id is not None:
            raw = self.market_client.get_market(market_id)
            market_data = MarketClient.parse_market(raw)
        elif market_data is None:
            raise ValueError("Must provide market_id or market_data")

        current_price = market_data.get("outcome_prices", {}).get("Yes", 0.5)
        if current_price is None:
            current_price = 0.5

        # Extract keywords from question for data source targeting
        question = market_data.get("question", "")
        keywords = self._extract_keywords(question)

        # Step 2: Run all sub-models
        sub_results = {}

        # 2a. Microstructure analysis
        micro_result = self.microstructure.analyze(market_data)
        sub_results["microstructure"] = micro_result

        # 2b. Time series analysis (if we have price history)
        if price_history and len(price_history) >= 10:
            ts_result = self.time_series.analyze(price_history, current_price, time_remaining_frac)
            sub_results["time_series"] = ts_result

        # 2c. External data fusion
        ext_result = self.external_data.analyze(market_data, keywords)
        sub_results["external_data"] = ext_result

        # 2d. AI semantic analysis
        ai_result = self.ai_model.analyze(market_data)
        if ai_result["estimate"] is not None:
            sub_results["ai_semantic"] = ai_result

        # Step 3: Ensemble combination
        estimates = []
        variances = []
        model_names = []

        for name, result in sub_results.items():
            if result.get("estimate") is not None and result.get("variance") is not None:
                estimates.append(result["estimate"])
                variances.append(result["variance"])
                model_names.append(name)

        if not estimates:
            # Fallback: just use market price
            ensemble_result = {
                "probability": current_price,
                "variance": 0.05,
                "std": math.sqrt(0.05),
                "weights": [],
                "n_models": 0,
            }
        else:
            ensemble_result = self.ensemble.combine(estimates, variances)

        raw_estimate = ensemble_result["probability"]
        estimate_std = ensemble_result["std"]

        # Step 4: Calibration
        if self._calibrated:
            calibrated_estimate = self.calibrator.calibrate(raw_estimate)
        else:
            calibrated_estimate = raw_estimate

        # Step 5: Final Monte Carlo for uncertainty
        final_mc = self.mc.simulate_binary_outcome(
            base_prob=calibrated_estimate,
            volatility=estimate_std * 4,  # Scale uncertainty for simulation
            time_remaining_frac=time_remaining_frac,
        )

        # Step 6: Kelly sizing
        kelly_result = kelly_with_uncertainty(
            estimated_prob=calibrated_estimate,
            prob_std=estimate_std,
            market_price=current_price,
        )

        # Step 7: Compute edge metrics
        edge = calibrated_estimate - current_price
        ev_per_dollar = calibrated_estimate * (1 - current_price) - (1 - calibrated_estimate) * current_price

        # Confidence assessment
        model_agreement = self._assess_agreement(estimates)

        return {
            "market": {
                "id": market_data.get("id"),
                "question": question,
                "current_price": current_price,
                "spread": market_data.get("spread"),
                "volume_24h": market_data.get("volume_24h"),
                "liquidity": market_data.get("liquidity"),
            },
            "prediction": {
                "probability": round(calibrated_estimate, 4),
                "raw_probability": round(raw_estimate, 4),
                "std": round(estimate_std, 4),
                "ci_95": (
                    round(max(0, calibrated_estimate - 1.96 * estimate_std), 4),
                    round(min(1, calibrated_estimate + 1.96 * estimate_std), 4),
                ),
                "monte_carlo": {
                    "mean": round(final_mc["mean"], 4),
                    "ci_95": tuple(round(x, 4) for x in final_mc["ci_95"]),
                    "p_resolve_yes": round(final_mc["p_resolve_yes"], 4),
                    "skewness": round(final_mc["skewness"], 4),
                },
            },
            "edge": {
                "edge": round(edge, 4),
                "ev_per_dollar": round(ev_per_dollar, 4),
                "edge_confidence": "HIGH" if abs(edge) > 2 * estimate_std else "MEDIUM" if abs(edge) > estimate_std else "LOW",
            },
            "sizing": kelly_result,
            "ensemble": {
                "n_models": len(estimates),
                "model_names": model_names,
                "weights": [round(w, 4) for w in ensemble_result.get("weights", [])],
                "model_agreement": round(model_agreement, 4),
            },
            "sub_models": sub_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def scan_markets(self, limit: int = 50, min_edge: float = 0.03) -> list[dict]:
        """Scan active markets and return those with detected edge.

        Args:
            limit: Number of markets to scan
            min_edge: Minimum absolute edge to include

        Returns:
            List of predictions sorted by absolute edge (descending)
        """
        markets = self.market_client.get_markets(limit=limit)
        results = []

        for raw in markets:
            parsed = MarketClient.parse_market(raw)
            if not parsed.get("active") or parsed.get("closed"):
                continue

            try:
                prediction = self.predict(market_data=parsed, time_remaining_frac=0.5)
                if abs(prediction["edge"]["edge"]) >= min_edge:
                    results.append(prediction)
            except Exception:
                continue

        # Sort by absolute edge descending
        results.sort(key=lambda x: abs(x["edge"]["edge"]), reverse=True)
        return results

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract meaningful keywords from market question."""
        stop_words = {
            "will", "the", "be", "to", "in", "of", "a", "an", "is", "by",
            "on", "at", "for", "or", "and", "this", "that", "it", "as",
            "with", "from", "has", "have", "been", "was", "are", "were",
            "do", "does", "did", "not", "but", "if", "than", "more",
            "before", "after", "above", "below", "between", "up", "down",
            "yes", "no", "what", "when", "where", "who", "which", "how",
        }
        words = question.replace("?", "").replace(",", "").replace("'", "").split()
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    @staticmethod
    def _assess_agreement(estimates: list[float]) -> float:
        """Measure how much sub-models agree (0 = total disagreement, 1 = unanimous)."""
        if len(estimates) < 2:
            return 1.0
        std = float(np.std(estimates))
        # Map std to agreement score (std=0 -> 1.0, std=0.3 -> ~0)
        return float(np.clip(1.0 - std * 3.3, 0.0, 1.0))
