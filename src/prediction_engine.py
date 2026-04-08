"""Polymarket Prediction Engine — Maximum Accuracy Ensemble.

Combines multiple independent signal sources with advanced statistical
models to produce calibrated probability estimates with edge detection.

Architecture:
    1. Market Microstructure Analyzer — bid/ask dynamics, volume patterns
    2. Statistical Time Series Models — HMM regime, GARCH vol, Hurst exponent
    3. Multi-Source Data Fusion — crypto, fear/greed, economic indicators
    4. AI Semantic Analyzer — OpenRouter LLM with RAG news context
    5. CLOB Orderbook Analyzer — real depth, imbalance, whale detection
    6. Arbitrage Detector — complete-set mispricing (from ent0n29/polybot)
    7. Bayesian Ensemble — inverse-variance weighted combination
    8. Calibration Layer — isotonic regression to debias
    9. Kelly Sizing — uncertainty-adjusted position sizing
    10. Strategy Adapter — proven calendar spread risk management

Integrations from reference repos:
    - ent0n29/polybot: Complete-set arbitrage, orderbook analysis
    - Polymarket/agents: RAG-style news context for AI predictions
    - Polymarket/agent-skills: CLOB API patterns, market data queries
    - discountry/polymarket-trading-bot: Gamma API, 15-min market patterns
    - ATM_ITM_MIXED_CALENDAR_V1: 1/3 Kelly, regime allocation, exit hierarchy

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
from src.data_sources import (
    CryptoDataSource, FearGreedIndex, EconomicDataSource, NewsDataSource,
    VolatilityIndexSource, CommoditiesSource, ForexSource, StockIndexSource, GNewsSource,
)
from src.clob_client import ClobClient, ArbitrageDetector, OrderbookAnalyzer
from src.news_rag import NewsRAGEngine
from src.strategy_adapter import StrategyAdapter
from src.statistics import (
    BetaBinomialModel,
    GaussianHMM,
    GARCH,
    AnalyticalUncertainty,
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
    """Extract edge signals from market microstructure data.

    Key insight from backtest: Polymarket prices are directionally correct —
    markets at 0.7 tend to resolve Yes, markets at 0.3 resolve No.
    The edge comes from detecting when the market UNDERESTIMATES its own
    conviction (tight spread + high volume = price should be MORE extreme).

    DO NOT mean-revert toward 0.5 — that was proven wrong (-57% ROI).
    Instead: amplify the market's own direction when conviction signals confirm.
    """

    def analyze(self, market: dict) -> dict:
        bid = market.get("best_bid")
        ask = market.get("best_ask")
        last = market.get("last_trade_price")
        spread = market.get("spread")
        volume_24h = market.get("volume_24h", 0) or 0
        liquidity = market.get("liquidity", 0) or 0

        signals = {}
        directional_signals = []

        base = last if last is not None else (bid + ask) / 2 if bid and ask else 0.5

        # Market direction: which side of 0.5 is the price on?
        direction = base - 0.5  # positive = leaning Yes, negative = leaning No
        abs_direction = abs(direction)

        # 1. Buying pressure from trade position in spread
        if bid is not None and ask is not None and ask > bid:
            midpoint = (bid + ask) / 2
            spread_width = ask - bid

            if last is not None and spread_width > 0:
                position_in_spread = (last - bid) / spread_width
                buying_pressure = position_in_spread - 0.5
                signals["buying_pressure"] = float(buying_pressure)
                # Buying pressure CONFIRMS market direction → amplify
                if buying_pressure * direction > 0:  # same direction
                    directional_signals.append(buying_pressure * 0.06)

            spread_pct = spread_width / midpoint if midpoint > 0 else 0
            signals["spread_pct"] = float(spread_pct)
            signals["market_efficiency"] = float(1.0 - min(spread_pct * 5, 1.0))

        # 2. Volume conviction — high volume CONFIRMS market direction
        if volume_24h > 0:
            log_vol = math.log10(max(volume_24h, 1))
            signals["log_volume"] = float(log_vol)

            if log_vol > 4 and abs_direction > 0.1:
                # High volume ($10k+) + clear direction = market is right, push further
                vol_boost = direction * min(log_vol / 50, 0.06)
                signals["volume_confirmation"] = float(vol_boost)
                directional_signals.append(vol_boost)

            if log_vol < 2.5 and abs_direction > 0.15:
                # Very low volume + clear direction = less confidence, slight dampen
                dampen = -direction * 0.03
                signals["low_volume_dampen"] = float(dampen)
                directional_signals.append(dampen)

        # 3. Spread tightness confirms price accuracy
        if spread is not None:
            if spread < 0.03 and abs_direction > 0.1:
                # Tight spread = market agrees on price → push direction further
                tight_boost = direction * 0.04
                signals["tight_spread_confirm"] = float(tight_boost)
                directional_signals.append(tight_boost)
            elif spread > 0.06 and abs_direction > 0.2:
                # Wide spread + strong direction = uncertainty, slight dampen
                wide_dampen = -direction * 0.02
                signals["wide_spread_dampen"] = float(wide_dampen)
                directional_signals.append(wide_dampen)

        # Build estimate — cap adjustment at ±2% (backtest on 5000 markets showed
        # larger adjustments systematically lose money. Microstructure can detect
        # small inefficiencies but NOT large mispricings.)
        total_adjustment = sum(directional_signals) if directional_signals else 0
        total_adjustment = max(-0.02, min(0.02, total_adjustment))
        estimate = float(np.clip(base + total_adjustment, 0.01, 0.99))

        # Variance: set HIGH relative to other models. Backtest proved microstructure
        # alone cannot beat market price (Brier 0.1781 vs market 0.1665).
        # High variance = low weight in ensemble = defers to better models.
        efficiency = signals.get("market_efficiency", 0.5)
        variance = 0.08 * (1.3 - efficiency * 0.3)

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
        self.uncertainty = AnalyticalUncertainty()

    def analyze(self, price_history: list[float], current_price: float, time_remaining_frac: float) -> dict:
        """Full time-series analysis — deterministic, data-driven.

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

        # 4. Analytical uncertainty estimation (deterministic, data-driven)
        drift = regime_mean * 252 if abs(regime_mean) > 1e-6 else 0.0
        uncertainty_result = self.uncertainty.estimate_binary_outcome(
            base_prob=current_price,
            volatility=annualized_vol,
            time_remaining_frac=time_remaining_frac,
            drift=drift,
        )
        signals["uncertainty_mean"] = uncertainty_result["mean"]
        signals["uncertainty_std"] = uncertainty_result["std"]
        signals["uncertainty_ci_95"] = uncertainty_result["ci_95"]
        signals["uncertainty_p_resolve_yes"] = uncertainty_result["p_resolve_yes"]
        signals["uncertainty_skewness"] = uncertainty_result["skewness"]

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
            estimate = uncertainty_result["mean"]

        estimate = float(np.clip(estimate, 0.01, 0.99))
        variance = uncertainty_result["std"] ** 2

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
    """Fuse external data sources into probability adjustments.

    Data-driven only — no random noise, pure signal extraction.

    Sources:
        - Fear & Greed Index (contrarian regime)
        - VIX / realized volatility (regime classification)
        - Crypto market data (for crypto markets)
        - Oil & gold (macro regime, inflation proxy)
        - DXY dollar strength (global macro)
        - Stock indices (market breadth, sentiment)
        - Google News headlines (event-driven)
    """

    def __init__(self, backtest_mode: bool = False):
        self.backtest_mode = backtest_mode
        if not backtest_mode:
            self.crypto = CryptoDataSource()
            self.fear_greed = FearGreedIndex()
            self.econ = EconomicDataSource()
            self.vix = VolatilityIndexSource()
            self.commodities = CommoditiesSource()
            self.forex = ForexSource()
            self.indices = StockIndexSource()
            self.news = GNewsSource()

    def analyze(self, market: dict, market_keywords: list[str]) -> dict:
        """Pull external data and compute probability adjustments.

        Pure data-driven: every adjustment is grounded in real market data.
        In backtest_mode, skips live API calls (they timeout on historical data)
        and uses only market metadata signals.
        """
        signals = {}
        adjustments = []
        kw_lower = set(k.lower() for k in market_keywords)

        if self.backtest_mode:
            # Skip to market metadata signals (section 8) — no live APIs
            return self._metadata_signals(market, signals, adjustments)

        # 1. Fear & Greed Index — contrarian signal
        try:
            fg = self.fear_greed.get_current()
            if fg:
                fg_value = fg["value"]
                signals["fear_greed_value"] = fg_value
                signals["fear_greed_class"] = fg["classification"]
                if fg_value < 20:
                    adjustments.append(("fear_greed_contrarian", 0.02, 0.02))
                elif fg_value > 80:
                    adjustments.append(("fear_greed_contrarian", -0.02, 0.02))
                else:
                    adjustments.append(("fear_greed_neutral", 0, 0.03))
        except Exception:
            signals["fear_greed_failed"] = True

        # 2. VIX / Realized Volatility — regime signal (like calendar strategy VIX gate)
        try:
            vix_data = self.vix.get_vix()
            if vix_data:
                signals["vix_current"] = vix_data["current"]
                signals["vix_regime"] = vix_data["regime"]
                signals["vix_percentile"] = vix_data["percentile"]
                signals["vix_mean_30d"] = vix_data["mean_30d"]
                # High VIX = more uncertainty = widen confidence intervals
                if vix_data["current"] > 25:
                    adjustments.append(("vix_high_uncertainty", 0, 0.03))
                elif vix_data["current"] < 14:
                    adjustments.append(("vix_low_complacency", 0, 0.005))
        except Exception:
            signals["vix_failed"] = True

        # 3. Crypto market data (if market is crypto-related)
        crypto_keywords = {"bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "xrp"}
        is_crypto = bool(crypto_keywords & kw_lower)

        if is_crypto:
            try:
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

                    change_24h = market_data.get("price_change_24h_pct", 0) or 0
                    if abs(change_24h) > 5:
                        direction = 1 if change_24h > 0 else -1
                        adjustments.append(("crypto_momentum", direction * 0.03, 0.015))

                history = self.crypto.get_price_history(coin_id, days=30)
                if len(history) > 10:
                    hist_prices = [p[1] for p in history]
                    returns = np.diff(np.log(hist_prices))
                    signals["crypto_realized_vol_30d"] = float(np.std(returns) * np.sqrt(365))
            except Exception:
                signals["crypto_data_failed"] = True

        # 4. Oil & Gold — macro regime indicators (from calendar strategy)
        macro_terms = {"oil", "energy", "inflation", "economy", "fed", "rate",
                       "gold", "commodity", "gas", "opec", "recession", "gdp",
                       "trade", "tariff", "war", "sanctions"}
        is_macro = bool(macro_terms & kw_lower)

        if is_macro:
            try:
                oil = self.commodities.get_commodity("crude_oil")
                if oil and oil.get("change_5d_pct") is not None:
                    signals["oil_price"] = oil["price"]
                    signals["oil_change_5d"] = oil["change_5d_pct"]
                    signals["oil_volatility"] = oil.get("volatility_30d")
                    # Sharp oil moves affect geopolitical markets
                    if abs(oil["change_5d_pct"]) > 5:
                        adjustments.append(("oil_shock", 0, 0.02))
            except Exception:
                pass

            try:
                gold = self.commodities.get_commodity("gold")
                if gold and gold.get("change_5d_pct") is not None:
                    signals["gold_price"] = gold["price"]
                    signals["gold_change_5d"] = gold["change_5d_pct"]
                    # Gold surge = flight to safety = risk-off
                    if gold["change_5d_pct"] > 3:
                        adjustments.append(("gold_risk_off", 0, 0.015))
            except Exception:
                pass

        # 5. DXY Dollar strength — global macro signal
        try:
            dxy = self.forex.get_dxy()
            if dxy:
                signals["dxy"] = dxy["dxy"]
                signals["dxy_trend"] = dxy["trend"]
                signals["dxy_change_5d"] = dxy.get("change_5d_pct")
        except Exception:
            pass

        # 6. Stock market breadth — overall sentiment
        stock_terms = {"stock", "market", "nasdaq", "sp500", "dow", "crash",
                       "rally", "bull", "bear", "correction", "ipo"}
        if stock_terms & kw_lower:
            try:
                breadth = self.indices.get_market_breadth()
                if breadth:
                    signals["market_breadth"] = breadth["breadth"]
                    signals["market_avg_5d"] = breadth["avg_change_5d"]
                    signals["market_dispersion"] = breadth["dispersion"]
                    # Strong directional breadth is a signal
                    if breadth["breadth"] == "bullish" and breadth["avg_change_5d"] > 2:
                        adjustments.append(("market_bullish_breadth", 0.02, 0.01))
                    elif breadth["breadth"] == "bearish" and breadth["avg_change_5d"] < -2:
                        adjustments.append(("market_bearish_breadth", -0.02, 0.01))
            except Exception:
                pass

        # 7. Google News headlines — event-driven context
        question = market.get("question", "")
        if question and len(market_keywords) >= 2:
            try:
                headlines = self.news.search_headlines(" ".join(market_keywords[:3]), max_results=5)
                if headlines:
                    signals["n_news_headlines"] = len(headlines)
                    signals["news_sources"] = [h.get("source", "") for h in headlines[:3]]
            except Exception:
                pass

        # 8. Market metadata signals (always available)
        return self._metadata_signals(market, signals, adjustments)

    def _metadata_signals(self, market: dict, signals: dict, adjustments: list) -> dict:
        """Market metadata signals — works offline, no API needed.

        Key principle: market prices are directionally correct.
        High volume/liquidity = more efficient = trust market price more.
        Low volume = less info = higher uncertainty, NOT reversion to 0.5.
        """
        volume = market.get("volume") or market.get("volume_24h") or 0
        liquidity = market.get("liquidity") or 0
        current_spread = market.get("spread") or 0

        if volume > 0:
            signals["market_volume"] = volume
            log_vol = math.log10(max(volume, 1))
            if log_vol < 3:
                signals["volume_signal"] = "very_low"
                # Low volume = more uncertainty, but don't mean-revert
                adjustments.append(("low_volume_uncertainty", 0, 0.04))
            elif log_vol > 5:
                signals["volume_signal"] = "high"
                adjustments.append(("high_volume_accuracy", 0, 0.01))

        if liquidity > 0:
            signals["market_liquidity"] = liquidity
            if liquidity < 500:
                signals["liquidity_signal"] = "thin"
                # Thin liquidity = wider uncertainty, not skepticism of direction
                adjustments.append(("thin_liquidity_uncertainty", 0, 0.03))
            elif liquidity > 10000:
                signals["liquidity_signal"] = "deep"

        if current_spread > 0.05:
            signals["wide_spread"] = True
            adjustments.append(("spread_uncertainty", 0, 0.02))

        total_adjustment = sum(a[1] for a in adjustments)
        total_variance = sum(a[2] for a in adjustments) if adjustments else 0.05

        prices = market.get("outcome_prices", {})
        base = prices.get("Yes", 0.5) if "Yes" in prices else 0.5
        estimate = float(np.clip(base + total_adjustment, 0.01, 0.99))

        return {
            "estimate": estimate,
            "variance": max(total_variance, 0.03),
            "signals": signals,
            "adjustments": [(a[0], a[1]) for a in adjustments],
            "model": "external_data",
        }


# ===========================================================================
# Sub-Model: CLOB Orderbook Analysis (from ent0n29/polybot patterns)
# ===========================================================================

class OrderbookModel:
    """Extract edge signals from real CLOB orderbook data.

    IMPORTANT: Polymarket CLOB orderbooks are dominated by extreme limit
    orders (bids at 0.001, asks at 0.999) with huge sizes. The simple
    midpoint/VWAP is useless — it returns ~0.5 or worse for every market.

    Instead, we anchor on the market price and only extract DIRECTIONAL
    signal from imbalance and near-price depth.
    """

    def __init__(self):
        self.clob = ClobClient()
        self.analyzer = OrderbookAnalyzer(self.clob)

    def analyze(self, token_id: Optional[str] = None, market_price: float = 0.5) -> dict:
        """Analyze CLOB orderbook for directional signals.

        Args:
            token_id: CLOB token ID for the Yes outcome
            market_price: Current market price (anchor — we adjust from here)
        """
        if not token_id:
            return {
                "estimate": None, "variance": None,
                "signals": {"no_token_id": True}, "model": "orderbook",
            }

        try:
            analysis = self.analyzer.full_analysis(token_id)
            if not analysis:
                return {
                    "estimate": None, "variance": None,
                    "signals": {"fetch_failed": True}, "model": "orderbook",
                }

            signals = {
                "midpoint": analysis.get("midpoint"),
                "weighted_mid": analysis.get("weighted_mid"),
                "spread": analysis.get("spread"),
                "imbalance": analysis.get("imbalance"),
                "bid_depth": analysis.get("bid_depth"),
                "ask_depth": analysis.get("ask_depth"),
                "liquidity_score": analysis.get("liquidity_score"),
                "n_whale_bids": len(analysis.get("whale_bids", [])),
                "n_whale_asks": len(analysis.get("whale_asks", [])),
            }

            # Anchor on market price — only adjust based on orderbook signals
            adjustment = 0.0

            # 1. Imbalance: buy pressure → nudge up, sell pressure → nudge down
            imbalance = analysis.get("imbalance", 0) or 0
            adjustment += imbalance * 0.03  # max ~3% from imbalance

            # 2. Near-price depth ratio (only orders within 10% of market price)
            ob = self.clob.get_orderbook(token_id)
            if ob:
                near_bid_vol = sum(
                    float(b.get("size", 0)) for b in ob.get("bids", [])
                    if abs(float(b.get("price", 0)) - market_price) < 0.10
                )
                near_ask_vol = sum(
                    float(a.get("size", 0)) for a in ob.get("asks", [])
                    if abs(float(a.get("price", 0)) - market_price) < 0.10
                )
                near_total = near_bid_vol + near_ask_vol
                if near_total > 100:  # meaningful near-price liquidity
                    near_imbalance = (near_bid_vol - near_ask_vol) / near_total
                    adjustment += near_imbalance * 0.03
                    signals["near_bid_vol"] = near_bid_vol
                    signals["near_ask_vol"] = near_ask_vol
                    signals["near_imbalance"] = round(near_imbalance, 4)

            # 3. Whale activity: more whale bids than asks → bullish
            n_whale_bids = len(analysis.get("whale_bids", []))
            n_whale_asks = len(analysis.get("whale_asks", []))
            if n_whale_bids + n_whale_asks > 0:
                whale_signal = (n_whale_bids - n_whale_asks) / (n_whale_bids + n_whale_asks)
                adjustment += whale_signal * 0.02

            # Cap total adjustment at ±5%
            adjustment = float(np.clip(adjustment, -0.05, 0.05))
            estimate = float(np.clip(market_price + adjustment, 0.01, 0.99))
            signals["adjustment"] = round(adjustment, 4)

            # High variance — orderbook signal is weak directional info
            variance = 0.06

            return {
                "estimate": estimate,
                "variance": variance,
                "signals": signals,
                "model": "orderbook",
            }
        except Exception as e:
            return {
                "estimate": None, "variance": None,
                "signals": {"error": str(e)}, "model": "orderbook",
            }


# ===========================================================================
# Sub-Model: AI Semantic Analysis (OpenRouter + RAG Context)
# ===========================================================================

class AISemanticModel:
    """Use LLM via OpenRouter for semantic understanding of market questions.

    Rotates through ALL free models with automatic fallback on rate limits.
    25 free models available — if one returns 429, try the next immediately.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # All free models ranked by capability (best first)
    FREE_MODELS = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-120b:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "qwen/qwen3.6-plus:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "minimax/minimax-m2.5:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "google/gemma-3-27b-it:free",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-coder:free",
        "z-ai/glm-4.5-air:free",
        "stepfun/step-3.5-flash:free",
        "arcee-ai/trinity-large-preview:free",
        "google/gemma-3-12b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "arcee-ai/trinity-mini:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-3n-e4b-it:free",
        "google/gemma-3n-e2b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "liquid/lfm-2.5-1.2b-instruct:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "nvidia/nemotron-3-super-120b-a12b:free"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.enabled = bool(self.api_key)
        self.rag = NewsRAGEngine()
        self._model_index = 0  # rotate through models

    def _next_model(self) -> str:
        """Get next model in rotation."""
        model = self.FREE_MODELS[self._model_index % len(self.FREE_MODELS)]
        self._model_index += 1
        return model

    def _call_model(self, prompt: str, model: str) -> Optional[dict]:
        """Call a single model. Returns parsed response or None on failure."""
        try:
            resp = requests.post(
                self.OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 600,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                return None  # rate limited, try next
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip() and msg.get("reasoning"):
                content = msg["reasoning"]
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Extract JSON from content (handle models that add extra text)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            parsed = json.loads(content)
            parsed["_model_used"] = model
            return parsed
        except Exception:
            return None

    def analyze(self, market: dict) -> dict:
        """Ask LLMs to estimate probability — rotates through all free models.

        Tries up to 5 models if rate-limited, uses first successful response.
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

        # RAG: Gather relevant news context
        try:
            rag_context = self.rag.gather_context(question, max_items=8)
            context_text = rag_context.get("context_text", "")
        except Exception:
            context_text = ""

        prompt = f"""You are an expert superforecaster and prediction market analyst.
Your calibration is excellent -- when you say 70%, events happen 70% of the time.

MARKET QUESTION: {question}

CURRENT MARKET PRICE: {current_price} (this is the market's implied probability)

{context_text}

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

        # Try up to 8 models with rotation
        parsed = None
        models_tried = []
        for _ in range(8):
            model = self._next_model()
            models_tried.append(model.split("/")[-1].split(":")[0])
            parsed = self._call_model(prompt, model)
            if parsed is not None:
                break

        if parsed is None:
            return {
                "estimate": None,
                "variance": None,
                "signals": {"ai_error": f"All models failed: {','.join(models_tried)}"},
                "model": "ai_semantic",
            }

        try:
            estimate = float(parsed["probability"])
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            edge = parsed.get("edge_vs_market", estimate - current_price)
            model_used = parsed.get("_model_used", "unknown")

            variance = 0.05 * (1 - confidence * 0.8)

            return {
                "estimate": np.clip(estimate, 0.01, 0.99),
                "variance": max(variance, 0.005),
                "signals": {
                    "ai_probability": estimate,
                    "ai_confidence": confidence,
                    "ai_reasoning": reasoning,
                    "ai_edge": edge,
                    "ai_model": model_used,
                    "models_tried": len(models_tried),
                },
                "model": "ai_semantic",
            }
        except (KeyError, ValueError, TypeError) as e:
            return {
                "estimate": None,
                "variance": None,
                "signals": {"ai_error": f"Parse failed: {e}", "raw": str(parsed)[:200]},
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
        5. Analytical uncertainty quantification (deterministic)
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
        ai_model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        calibration_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
        total_equity: float = 10000.0,
        backtest_mode: bool = False,
    ):
        self.backtest_mode = backtest_mode
        self.market_client = MarketClient()
        self.microstructure = MicrostructureModel()
        self.time_series = TimeSeriesModel()
        self.external_data = ExternalDataModel(backtest_mode=backtest_mode)
        self.orderbook_model = OrderbookModel()
        self.ai_model = AISemanticModel(api_key=openrouter_api_key, model=ai_model)
        self.arbitrage = ArbitrageDetector()
        self.ensemble = EnsemblePredictor()
        self.uncertainty = AnalyticalUncertainty()

        # Strategy adapter — proven calendar spread risk management
        # Provides: 1/3 Kelly, regime allocation, entry gates, exit hierarchy
        self.strategy = StrategyAdapter(total_equity=total_equity)

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
        token_id: Optional[str] = None,
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
        # Auto-load from snapshot history if not provided
        if not price_history:
            try:
                from scripts.price_snapshots import get_price_history
                market_id = market_data.get("id")
                if market_id:
                    price_history = get_price_history(market_id)
            except Exception:
                pass
        if price_history and len(price_history) >= 10:
            ts_result = self.time_series.analyze(price_history, current_price, time_remaining_frac)
            sub_results["time_series"] = ts_result

        # 2c. External data fusion
        ext_result = self.external_data.analyze(market_data, keywords)
        sub_results["external_data"] = ext_result

        # 2d. CLOB orderbook analysis (from ent0n29/polybot + agent-skills)
        if token_id:
            ob_result = self.orderbook_model.analyze(token_id, market_price=current_price)
            if ob_result["estimate"] is not None:
                sub_results["orderbook"] = ob_result

        # 2e. AI semantic analysis (with RAG context from Polymarket/agents)
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

        # Step 4: Calibration DISABLED — was overfitting (trained on same data
        # it predicted, producing fake edges). Raw ensemble is more honest.
        calibrated_estimate = raw_estimate

        # Step 5: Analytical uncertainty quantification (deterministic)
        final_uncertainty = self.uncertainty.estimate_binary_outcome(
            base_prob=calibrated_estimate,
            volatility=estimate_std * 4,  # Scale uncertainty
            time_remaining_frac=time_remaining_frac,
        )

        # Step 6: Kelly sizing (raw)
        kelly_result = kelly_with_uncertainty(
            estimated_prob=calibrated_estimate,
            prob_std=estimate_std,
            market_price=current_price,
        )

        # Step 7: Compute edge metrics
        raw_edge = calibrated_estimate - current_price

        # Edge must exceed spread to be real — otherwise it's just noise
        market_spread = market_data.get("spread") or 0.02
        edge_after_spread = abs(raw_edge) - market_spread
        if edge_after_spread <= 0:
            # Edge doesn't survive the spread → report zero effective edge
            edge = 0.0
        else:
            edge = raw_edge

        ev_per_dollar = calibrated_estimate * (1 - current_price) - (1 - calibrated_estimate) * current_price

        # Confidence assessment — edge must also exceed model uncertainty
        model_agreement = self._assess_agreement(estimates)
        edge_confidence = "HIGH" if abs(edge) > 2 * estimate_std else "MEDIUM" if abs(edge) > estimate_std else "LOW"

        # Build base prediction result
        prediction_result = {
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
                "uncertainty": {
                    "mean": round(final_uncertainty["mean"], 4),
                    "ci_95": tuple(round(x, 4) for x in final_uncertainty["ci_95"]),
                    "p_resolve_yes": round(final_uncertainty["p_resolve_yes"], 4),
                    "skewness": round(final_uncertainty["skewness"], 4),
                },
            },
            "edge": {
                "edge": round(edge, 4),
                "ev_per_dollar": round(ev_per_dollar, 4),
                "edge_confidence": edge_confidence,
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

        # Step 8: Strategy adapter — entry evaluation with proven risk management
        # Applies: 1/3 Kelly, regime-based allocation, entry filter gates
        try:
            entry_eval = self.strategy.evaluate_entry(market_data, prediction_result)
            prediction_result["strategy"] = {
                "should_enter": entry_eval["should_enter"],
                "regime": entry_eval["regime"],
                "filters": entry_eval["filters"],
                "sizing_adjusted": entry_eval["sizing"],
                "is_contrarian": entry_eval.get("is_contrarian", False),
            }
        except Exception as e:
            prediction_result["strategy"] = {
                "should_enter": None,
                "error": str(e),
            }

        return prediction_result

    def scan_markets(self, limit: int = 50, min_edge: float = 0.03) -> list[dict]:
        """Scan active markets and return those with detected edge.

        Now includes strategy adapter evaluation:
        - Updates regime from Fear & Greed Index
        - Applies entry filter gates
        - Uses 1/3 Kelly with regime-aware sizing

        Args:
            limit: Number of markets to scan
            min_edge: Minimum absolute edge to include

        Returns:
            List of predictions sorted by absolute edge (descending)
        """
        # Update regime before scanning (like VIX check on Monday)
        # Use actual VIX if available, fall back to Fear&Greed
        try:
            vix_data = self.external_data.vix.get_vix()
            if vix_data and vix_data.get("current"):
                # Map VIX to Fear&Greed-like scale (VIX 10=greed 90, VIX 40=fear 10)
                vix_val = vix_data["current"]
                fg_equivalent = max(5, min(95, 100 - (vix_val - 10) * 3))
                self.strategy.allocator.update_regime(fg_equivalent)
            else:
                fg = self.external_data.fear_greed.get_current()
                if fg:
                    self.strategy.allocator.update_regime(fg["value"])
        except Exception:
            pass

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

    def update_regime(self, fear_greed_value: float):
        """Manually update market regime (called periodically)."""
        self.strategy.allocator.update_regime(fear_greed_value)

    def evaluate_exit(self, market_id: str, current_edge: float,
                      current_price: float, confidence: str,
                      time_to_expiry_frac: float,
                      imbalance: Optional[float] = None) -> dict:
        """Evaluate exit signals for an open position.

        Uses priority-based signal hierarchy:
            IMBALANCE_FLIP > STALL_DECAY > TIME_EXIT > EXPIRY_CLOSE > EDGE_EVAPORATED
        """
        return self.strategy.evaluate_exit(
            market_id=market_id,
            current_edge=current_edge,
            current_price=current_price,
            confidence=confidence,
            time_to_expiry_frac=time_to_expiry_frac,
            imbalance=imbalance,
        )

    def get_strategy_status(self) -> dict:
        """Get full strategy adapter status."""
        return self.strategy.get_status()

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
