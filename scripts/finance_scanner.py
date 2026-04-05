"""Financial Market Scanner — Targets econ/finance markets on Polymarket.

Uses real financial data (oil, yields, VIX, BTC, gold, Fed funds) to
compute fair probabilities for financial/economic prediction markets,
then compares against Polymarket odds to find mispricing.

Data sources (all free, multi-source fallback):
    - Twelve Data: Oil (CL), Gold (XAU/USD), VIX, indices — free tier
    - FRED: VIX (VIXCLS), Oil (DCOILWTICO), yields (DGS10, DGS2) — free
    - Binance: BTC/ETH with 24h stats + funding rate — free
    - metals.dev: Gold/silver spot — free
    - CoinGecko: Crypto fallback
    - Fear & Greed Index: Market sentiment
    - OpenRouter AI: Semantic analysis with financial context

This is where the edge comes from: hard financial data + AI reasoning
vs crowd-priced prediction markets.
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests

from src.market_client import MarketClient
from src.prediction_engine import PredictionEngine

GAMMA_API = "https://gamma-api.polymarket.com"


# =========================================================
# Financial Data Fetcher — Multi-Source (no Yahoo)
# =========================================================

class FinancialData:
    """Pull real-time financial data from free APIs — deep fallback chain.

    Sources:
    - Binance: BTC, ETH, crypto (free, no key)
    - Twelve Data: Stocks, forex, commodities (free: 800 req/day)
    - Financial Modeling Prep: Stocks, commodities, forex (free: 250 req/day)
    - Alpha Vantage: Stocks, forex (free: 25 req/day — last resort)
    - FRED (St. Louis Fed): Yields, VIX, S&P 500, oil (free, no key)
    - metals.dev: Gold/silver (free, no key)
    - CoinGecko: Crypto backup
    - Alternative.me: Fear & Greed

    Optional env vars for premium tiers:
    - FMP_API_KEY: Financial Modeling Prep (get free at financialmodelingprep.com)
    - ALPHA_VANTAGE_KEY: Alpha Vantage (get free at alphavantage.co)
    - MARKETSTACK_KEY: Marketstack (get free at marketstack.com)
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.cache = {}

    def _fetch_with_fallbacks(self, key: str, fetchers: list) -> dict:
        """Try multiple sources, return first success."""
        if key in self.cache:
            return self.cache[key]
        for fetcher in fetchers:
            try:
                result = fetcher()
                if result and result.get("price") is not None:
                    result["symbol"] = key
                    self.cache[key] = result
                    return result
            except Exception:
                continue
        return {"error": "all sources failed", "symbol": key}

    # --- Oil ---
    def get_oil(self) -> dict:
        return self._fetch_with_fallbacks("oil", [
            self._oil_from_twelvedata,
            self._oil_from_fmp,
            self._oil_from_fred,
            self._oil_from_metals_api,
        ])

    def _oil_from_fmp(self) -> dict:
        fmp_key = os.environ.get("FMP_API_KEY", "demo")
        r = self.session.get(
            "https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={"symbol": "CLUSD", "apikey": fmp_key}, timeout=10,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        entries = data[:30] if isinstance(data, list) else data.get("historical", [])[:30]
        closes = [float(h["close"]) for h in reversed(entries) if h.get("close")]
        return self._build_data(closes) if closes else {}

    def _oil_from_fred(self) -> dict:
        r = self.session.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": "DCOILWTICO", "api_key": "DEMO_KEY",
            "file_type": "json", "sort_order": "desc", "limit": 30,
        }, timeout=10)
        if r.status_code != 200:
            return {}
        obs = r.json().get("observations", [])
        closes = [float(o["value"]) for o in reversed(obs)
                  if o.get("value") and o["value"] != "."]
        return self._build_data(closes) if closes else {}

    def _oil_from_twelvedata(self) -> dict:
        r = self.session.get("https://api.twelvedata.com/time_series", params={
            "symbol": "CL", "interval": "1day", "outputsize": 30,
            "format": "JSON",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        values = r.json().get("values", [])
        if not values:
            return {}
        closes = [float(v["close"]) for v in reversed(values) if v.get("close")]
        return self._build_data(closes)

    def _oil_from_metals_api(self) -> dict:
        # metals-api.com has crude oil
        r = self.session.get("https://metals-api.com/api/latest", params={
            "base": "USD", "symbols": "CRUDE",
        }, timeout=10)
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            if "CRUDE" in rates:
                return {"price": 1.0 / rates["CRUDE"]}
        return {}

    # --- Gold ---
    def get_gold(self) -> dict:
        return self._fetch_with_fallbacks("gold", [
            self._gold_from_metals_dev,
            self._gold_from_twelvedata,
            self._gold_from_fmp,
        ])

    def _gold_from_fmp(self) -> dict:
        fmp_key = os.environ.get("FMP_API_KEY", "demo")
        r = self.session.get(
            "https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={"symbol": "GCUSD", "apikey": fmp_key}, timeout=10,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        entries = data[:30] if isinstance(data, list) else data.get("historical", [])[:30]
        closes = [float(h["close"]) for h in reversed(entries) if h.get("close")]
        return self._build_data(closes) if closes else {}

    def _gold_from_metals_dev(self) -> dict:
        r = self.session.get("https://api.metals.dev/v1/latest", params={
            "api_key": "demo", "currency": "USD", "unit": "toz",
        }, timeout=10)
        if r.status_code == 200:
            metals = r.json().get("metals", {})
            gold_price = metals.get("gold")
            if gold_price:
                return {"price": float(gold_price)}
        return {}

    def _gold_from_twelvedata(self) -> dict:
        r = self.session.get("https://api.twelvedata.com/time_series", params={
            "symbol": "XAU/USD", "interval": "1day", "outputsize": 30,
            "format": "JSON",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        values = r.json().get("values", [])
        if not values:
            return {}
        closes = [float(v["close"]) for v in reversed(values) if v.get("close")]
        return self._build_data(closes)

    # --- VIX ---
    def get_vix(self) -> dict:
        return self._fetch_with_fallbacks("vix", [
            self._vix_from_fred,
            self._vix_from_twelvedata,
        ])

    def _vix_from_fred(self) -> dict:
        """FRED (St. Louis Fed) — official VIX data, free, no key."""
        r = self.session.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": "VIXCLS",
            "api_key": "DEMO_KEY",  # FRED allows demo key for basic access
            "file_type": "json",
            "sort_order": "desc",
            "limit": 30,
        }, timeout=10)
        if r.status_code != 200:
            return {}
        obs = r.json().get("observations", [])
        closes = [float(o["value"]) for o in reversed(obs)
                  if o.get("value") and o["value"] != "."]
        return self._build_data(closes) if closes else {}

    def _vix_from_twelvedata(self) -> dict:
        r = self.session.get("https://api.twelvedata.com/time_series", params={
            "symbol": "VIX", "interval": "1day", "outputsize": 30,
            "format": "JSON",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        values = r.json().get("values", [])
        if not values:
            return {}
        closes = [float(v["close"]) for v in reversed(values) if v.get("close")]
        return self._build_data(closes)

    # --- Treasury Yields ---
    def get_10y_yield(self) -> dict:
        return self._fetch_with_fallbacks("yield_10y", [
            lambda: self._yield_from_fred("DGS10"),
            lambda: self._yield_from_twelvedata("US10Y"),
        ])

    def get_2y_yield(self) -> dict:
        return self._fetch_with_fallbacks("yield_2y", [
            lambda: self._yield_from_fred("DGS2"),
            lambda: self._yield_from_twelvedata("US02Y"),
        ])

    def _yield_from_fred(self, series_id: str) -> dict:
        r = self.session.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": series_id,
            "api_key": "DEMO_KEY",
            "file_type": "json",
            "sort_order": "desc",
            "limit": 30,
        }, timeout=10)
        if r.status_code != 200:
            return {}
        obs = r.json().get("observations", [])
        closes = [float(o["value"]) for o in reversed(obs)
                  if o.get("value") and o["value"] != "."]
        return self._build_data(closes) if closes else {}

    def _yield_from_twelvedata(self, symbol: str) -> dict:
        r = self.session.get("https://api.twelvedata.com/time_series", params={
            "symbol": symbol, "interval": "1day", "outputsize": 30, "format": "JSON",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        values = r.json().get("values", [])
        if not values:
            return {}
        closes = [float(v["close"]) for v in reversed(values) if v.get("close")]
        return self._build_data(closes)

    # --- S&P 500 ---
    def get_sp500(self) -> dict:
        return self._fetch_with_fallbacks("sp500", [
            self._sp500_from_twelvedata,
            self._sp500_from_fmp,
            self._sp500_from_av,
            self._sp500_from_fred,
        ])

    def _sp500_from_twelvedata(self) -> dict:
        r = self.session.get("https://api.twelvedata.com/time_series", params={
            "symbol": "SPX", "interval": "1day", "outputsize": 30, "format": "JSON",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        values = r.json().get("values", [])
        if not values:
            return {}
        closes = [float(v["close"]) for v in reversed(values) if v.get("close")]
        return self._build_data(closes)

    def _sp500_from_fmp(self) -> dict:
        """Financial Modeling Prep (250 calls/day free)."""
        fmp_key = os.environ.get("FMP_API_KEY", "demo")
        r = self.session.get(
            "https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={"symbol": "SPY", "apikey": fmp_key}, timeout=10,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        entries = data[:30] if isinstance(data, list) else data.get("historical", [])[:30]
        closes = [float(h["close"]) for h in reversed(entries) if h.get("close")]
        return self._build_data(closes) if closes else {}

    def _sp500_from_av(self) -> dict:
        """Alpha Vantage SPY ETF as S&P proxy (25 calls/day free)."""
        av_key = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
        r = self.session.get("https://www.alphavantage.co/query", params={
            "function": "TIME_SERIES_DAILY", "symbol": "SPY",
            "outputsize": "compact", "apikey": av_key,
        }, timeout=10)
        if r.status_code != 200:
            return {}
        ts = r.json().get("Time Series (Daily)", {})
        sorted_dates = sorted(ts.keys())[-30:]
        closes = [float(ts[d]["4. close"]) for d in sorted_dates]
        return self._build_data(closes) if closes else {}

    def _sp500_from_fred(self) -> dict:
        """FRED S&P 500 index (free)."""
        r = self.session.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": "SP500", "api_key": "DEMO_KEY",
            "file_type": "json", "sort_order": "desc", "limit": 30,
        }, timeout=10)
        if r.status_code != 200:
            return {}
        obs = r.json().get("observations", [])
        closes = [float(o["value"]) for o in reversed(obs)
                  if o.get("value") and o["value"] != "."]
        return self._build_data(closes) if closes else {}

    # --- BTC ---
    def get_btc(self) -> dict:
        """Bitcoin from Binance (primary) + CoinGecko (backup)."""
        return self._fetch_with_fallbacks("btc", [
            self._btc_from_binance,
            self._btc_from_coingecko,
        ])

    def _btc_from_binance(self) -> dict:
        # Current price
        r = self.session.get("https://api.binance.com/api/v3/ticker/price",
                             params={"symbol": "BTCUSDT"}, timeout=8)
        if r.status_code != 200:
            return {}
        price = float(r.json()["price"])

        # 30-day klines for history
        r2 = self.session.get("https://api.binance.com/api/v3/klines",
                              params={"symbol": "BTCUSDT", "interval": "1d", "limit": 30},
                              timeout=10)
        closes = [price]
        if r2.status_code == 200:
            closes = [float(k[4]) for k in r2.json()]

        # 24h stats
        r3 = self.session.get("https://api.binance.com/api/v3/ticker/24hr",
                              params={"symbol": "BTCUSDT"}, timeout=8)
        data = self._build_data(closes)
        if r3.status_code == 200:
            stats = r3.json()
            data["high_24h"] = float(stats.get("highPrice", 0))
            data["low_24h"] = float(stats.get("lowPrice", 0))
            data["change_24h"] = float(stats.get("priceChangePercent", 0))
            data["volume_24h"] = float(stats.get("quoteVolume", 0))

        # Funding rate for sentiment
        try:
            r4 = self.session.get("https://fapi.binance.com/fapi/v1/fundingRate",
                                  params={"symbol": "BTCUSDT", "limit": 1}, timeout=8)
            if r4.status_code == 200 and r4.json():
                data["funding_rate"] = float(r4.json()[0]["fundingRate"])
        except Exception:
            pass

        return data

    def _btc_from_coingecko(self) -> dict:
        r = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin", params={
            "localization": "false", "tickers": "false", "community_data": "false",
            "developer_data": "false",
        }, timeout=10)
        if r.status_code != 200:
            return {}
        market = r.json().get("market_data", {})
        return {
            "price": market.get("current_price", {}).get("usd"),
            "change_24h": market.get("price_change_percentage_24h"),
            "change_7d": market.get("price_change_percentage_7d"),
            "change_30d": market.get("price_change_percentage_30d"),
            "ath": market.get("ath", {}).get("usd"),
            "high_24h": market.get("high_24h", {}).get("usd"),
            "low_24h": market.get("low_24h", {}).get("usd"),
        }

    # --- Fear & Greed ---
    def get_fear_greed(self) -> dict:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            d = r.json().get("data", [{}])[0]
            return {"value": int(d.get("value", 50)), "classification": d.get("value_classification", "")}
        except Exception:
            return {"value": 50, "classification": "neutral"}

    # --- Helper ---
    def _build_data(self, closes: list[float]) -> dict:
        """Build standard data dict from a list of closing prices."""
        if not closes:
            return {}
        data = {
            "price": closes[-1],
            "prev_close": closes[-2] if len(closes) >= 2 else None,
            "closes": closes,
        }
        if len(closes) >= 2:
            data["change_1d_pct"] = (closes[-1] - closes[-2]) / closes[-2] * 100
            data["min_30d"] = min(closes)
            data["max_30d"] = max(closes)
            data["mean_30d"] = float(np.mean(closes))
        if len(closes) > 5:
            data["volatility"] = float(np.std(np.diff(np.log(closes))) * np.sqrt(252))
        return data

    def get_all(self) -> dict:
        """Fetch all financial data."""
        return {
            "oil": self.get_oil(),
            "gold": self.get_gold(),
            "vix": self.get_vix(),
            "yield_10y": self.get_10y_yield(),
            "yield_tbill": self.get_2y_yield(),
            "sp500": self.get_sp500(),
            "btc": self.get_btc(),
            "fear_greed": self.get_fear_greed(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =========================================================
# Market Type Classifiers + Probability Estimators
# =========================================================

FINANCE_KEYWORDS = {
    "fed_rate": ["fed", "interest rate", "fomc", "federal reserve", "rate cut", "rate hike", "bps"],
    "oil": ["oil", "wti", "crude", "brent", "opec"],
    "crypto": ["bitcoin", "btc ", "ethereum", "eth ", "solana", "dip to $", "price of bitcoin", "price of ethereum"],
    "inflation": ["inflation", "cpi", "consumer price", "pce"],
    "recession": ["recession", "gdp", "economic contraction"],
    "gold": ["gold", "precious metal"],
    "stocks": ["stock", "s&p", "nasdaq", "dow", "market crash", "correction"],
    "tariff": ["tariff", "trade war", "trade deal", "sanctions"],
    "treasury": ["treasury", "bond", "yield", "debt ceiling", "deficit"],
}


def classify_market(question: str) -> list[str]:
    """Classify a market question into financial categories."""
    q_lower = question.lower()
    categories = []
    for cat, keywords in FINANCE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            categories.append(cat)
    return categories


def estimate_fed_rate(question: str, fin_data: dict) -> dict:
    """Estimate probability for Fed rate decision markets.

    Uses yield curve, VIX, T-bill rates, and market pricing to
    estimate likelihood of rate changes.
    """
    q_lower = question.lower()
    vix = fin_data.get("vix", {}).get("price", 20)
    tbill = fin_data.get("yield_tbill", {}).get("price", 4.0)
    y10 = fin_data.get("yield_10y", {}).get("price", 4.3)
    fg = fin_data.get("fear_greed", {}).get("value", 50)

    # Parse what the market is asking
    is_cut = "decrease" in q_lower or "cut" in q_lower
    is_hike = "increase" in q_lower or "hike" in q_lower
    is_no_change = "no change" in q_lower

    # Extract basis points
    bps = 25
    if "50" in q_lower:
        bps = 50
    elif "75" in q_lower:
        bps = 75

    signals = {
        "vix": vix,
        "tbill_rate": tbill,
        "yield_10y": y10,
        "fear_greed": fg,
        "yield_curve_spread": y10 - tbill if y10 and tbill else None,
    }

    # Current Fed Funds rate is ~4.25-4.50% (as of early 2026)
    # T-bill rate near 3.6% implies market expects some easing
    # VIX at 24 = elevated uncertainty

    if is_no_change:
        # High VIX + extreme fear = Fed might cut, not hold
        # Low VIX + normal conditions = likely hold
        if vix > 25 and fg < 25:
            prob = 0.70  # Uncertainty suggests possible action
        elif vix < 18:
            prob = 0.92  # Calm market = Fed holds
        else:
            prob = 0.85  # Base case: Fed likely holds
        signals["logic"] = "no_change_base"

    elif is_cut:
        # T-bill well below Fed Funds = market pricing some cuts
        rate_signal = max(0, (4.3 - tbill) / 4.3) if tbill else 0.1
        fear_signal = max(0, (50 - fg) / 100) if fg < 50 else 0

        if bps >= 50:
            # 50+ bps cut requires crisis
            prob = 0.02 + rate_signal * 0.08 + fear_signal * 0.05
            if vix > 30:
                prob += 0.05
            signals["logic"] = "large_cut_crisis_needed"
        else:
            # 25 bps cut
            prob = 0.05 + rate_signal * 0.15 + fear_signal * 0.10
            if vix > 25:
                prob += 0.05
            signals["logic"] = "small_cut_data_dependent"

    elif is_hike:
        # Rate hike in current environment very unlikely unless inflation spikes
        prob = 0.02
        if vix < 15 and fg > 70:
            prob = 0.05  # Very calm + greedy might mean inflation risk
        signals["logic"] = "hike_very_unlikely"

    else:
        prob = 0.5
        signals["logic"] = "unclassified_fed"

    return {
        "probability": float(np.clip(prob, 0.01, 0.99)),
        "confidence": 0.6,
        "signals": signals,
        "category": "fed_rate",
    }


def estimate_oil_price(question: str, fin_data: dict) -> dict:
    """Estimate probability for oil price threshold markets.

    Uses current price, volatility, and trend to estimate
    probability of hitting a target price level.
    """
    oil = fin_data.get("oil", {})
    current = oil.get("price", 70)
    vol = oil.get("volatility")
    closes = oil.get("closes", [])
    vix = fin_data.get("vix", {}).get("price", 20)

    q_lower = question.lower()

    # Extract target price
    target = None
    for word in question.split():
        word_clean = word.replace("$", "").replace(",", "").replace("?", "")
        try:
            val = float(word_clean)
            if 30 < val < 500:  # reasonable oil price range
                target = val
        except ValueError:
            pass

    if target is None:
        return {"probability": 0.5, "confidence": 0.3, "signals": {"error": "no_target_found"}, "category": "oil"}

    is_high = "high" in q_lower or "above" in q_lower or "hit" in q_lower
    is_low = "low" in q_lower or "below" in q_lower or "dip" in q_lower

    # Extract timeframe (days remaining in month)
    now = datetime.now(timezone.utc)
    days_left = max(1, 30 - now.day)  # rough estimate for "in April"

    # Use log-normal model for oil price movement
    daily_vol = (vol / np.sqrt(252)) if vol else 0.025  # ~2.5% daily default

    # Adjust vol for recent regime: if last 3 days show big moves, inflate vol
    if closes and len(closes) >= 3:
        recent_returns = np.abs(np.diff(np.log(closes[-5:])))
        recent_daily_vol = float(np.mean(recent_returns)) if len(recent_returns) > 0 else daily_vol
        # Use max of historical and recent vol (regime break detection)
        if recent_daily_vol > daily_vol * 1.5:
            daily_vol = recent_daily_vol
            signals["vol_regime"] = "elevated"

    drift = 0  # assume no drift (random walk)

    if is_high or (not is_low):
        # P(price >= target)
        log_ratio = math.log(target / current) if current > 0 else 0
        z = (log_ratio - drift * days_left) / (daily_vol * math.sqrt(days_left))
        from scipy.stats import norm
        prob = float(1 - norm.cdf(z))
    else:
        # P(price <= target)
        log_ratio = math.log(target / current) if current > 0 else 0
        z = (log_ratio - drift * days_left) / (daily_vol * math.sqrt(days_left))
        from scipy.stats import norm
        prob = float(norm.cdf(z))

    # Adjust for recent trend
    if closes and len(closes) >= 3:
        recent_trend = (closes[-1] - closes[-3]) / closes[-3]
        if is_high and recent_trend > 0.05:
            prob = min(0.95, prob * 1.15)  # Strong uptrend increases high target prob
        elif is_high and recent_trend < -0.05:
            prob = max(0.02, prob * 0.85)  # Downtrend decreases high target prob

    signals = {
        "current_price": current,
        "target": target,
        "direction": "HIGH" if is_high else "LOW",
        "days_left": days_left,
        "daily_vol": round(daily_vol, 4),
        "annual_vol": round(vol, 4) if vol else None,
        "z_score": round(z, 2) if 'z' in dir() else None,
        "recent_closes": [round(c, 2) for c in closes[-5:]] if closes else [],
        "vix": vix,
    }

    return {
        "probability": float(np.clip(prob, 0.01, 0.99)),
        "confidence": 0.65,
        "signals": signals,
        "category": "oil",
    }


def estimate_btc_price(question: str, fin_data: dict) -> dict:
    """Estimate probability for Bitcoin price threshold markets."""
    btc = fin_data.get("btc", {})
    current = btc.get("price", 60000)
    change_24h = btc.get("change_24h", 0) or 0
    change_7d = btc.get("change_7d", 0) or 0
    high_24h = btc.get("high_24h", current)
    low_24h = btc.get("low_24h", current)

    q_lower = question.lower()

    # Extract target price
    target = None
    for word in question.split():
        word_clean = word.replace("$", "").replace(",", "").replace("?", "")
        try:
            val = float(word_clean)
            if 1000 < val < 500000:
                target = val
        except ValueError:
            pass

    if target is None:
        # Handle "Up or Down" type markets
        if "up or down" in q_lower or "up" in q_lower:
            # Simple: 50/50 adjusted for recent momentum
            prob = 0.5
            if change_24h and change_24h > 1:
                prob = 0.55
            elif change_24h and change_24h < -1:
                prob = 0.45
            return {"probability": prob, "confidence": 0.35,
                    "signals": {"type": "direction", "change_24h": change_24h}, "category": "crypto"}
        return {"probability": 0.5, "confidence": 0.3, "signals": {"error": "no_target_found"}, "category": "crypto"}

    is_above = "above" in q_lower or "hit" in q_lower
    is_dip = "dip" in q_lower or "below" in q_lower or "drop" in q_lower

    # BTC daily vol ~3-5%
    daily_vol = 0.035
    now = datetime.now(timezone.utc)
    days_left = max(1, 30 - now.day)

    from scipy.stats import norm

    if is_above:
        if current >= target:
            prob = 0.95  # Already above
        else:
            log_ratio = math.log(target / current)
            z = log_ratio / (daily_vol * math.sqrt(days_left))
            # For "will be above on date X" (point-in-time, not touch)
            prob = float(1 - norm.cdf(z))
    elif is_dip:
        if current <= target:
            prob = 0.90  # Already below
        else:
            # "Will BTC dip to X" means touch X at any point — use barrier option math
            log_ratio = math.log(target / current)
            # Approximate: P(min price <= target) over N days
            z = abs(log_ratio) / (daily_vol * math.sqrt(days_left))
            prob = float(2 * norm.cdf(-z))  # reflection principle approximation
    else:
        prob = 0.5

    # Trend adjustment
    if change_7d and abs(change_7d) > 5:
        direction = 1 if change_7d > 0 else -1
        if is_above and direction > 0:
            prob = min(0.95, prob * 1.1)
        elif is_dip and direction < 0:
            prob = min(0.95, prob * 1.1)

    signals = {
        "current_price": current,
        "target": target,
        "direction": "ABOVE" if is_above else "DIP" if is_dip else "UNKNOWN",
        "change_24h": round(change_24h, 2),
        "change_7d": round(change_7d, 2),
        "daily_vol": daily_vol,
        "days_left": days_left,
        "high_24h": high_24h,
        "low_24h": low_24h,
    }

    return {
        "probability": float(np.clip(prob, 0.01, 0.99)),
        "confidence": 0.55,
        "signals": signals,
        "category": "crypto",
    }


# =========================================================
# AI Financial Analyst
# =========================================================

AI_MODELS = [
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
    "nvidia/nemotron-nano-9b-v2:free",
    "arcee-ai/trinity-mini:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]
_ai_model_idx = 0


def ai_financial_analysis(question: str, fin_data: dict, quant_estimate: dict) -> dict:
    """Use AI models with full financial context — rotates through all 25 free models."""
    global _ai_model_idx
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"estimate": None, "reasoning": "No API key"}

    oil = fin_data.get("oil", {})
    btc = fin_data.get("btc", {})
    vix = fin_data.get("vix", {})
    y10 = fin_data.get("yield_10y", {})
    gold = fin_data.get("gold", {})
    fg = fin_data.get("fear_greed", {})
    sp = fin_data.get("sp500", {})

    context = f"""CURRENT FINANCIAL DATA (as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}):
- WTI Crude Oil: ${oil.get('price', '?')} (recent: {[round(c,1) for c in (oil.get('closes') or [])[-5:]]})
- Gold: ${gold.get('price', '?')}
- VIX: {vix.get('price', '?')} (30d range: {vix.get('min_30d','?')}-{vix.get('max_30d','?')})
- 10Y Treasury Yield: {y10.get('price', '?')}%
- S&P 500: {sp.get('price', '?')}
- Bitcoin: ${btc.get('price', '?')} (24h: {btc.get('change_24h','?')}%, 7d: {btc.get('change_7d','?')}%)
- Fear & Greed Index: {fg.get('value', '?')} ({fg.get('classification', '?')})

QUANTITATIVE MODEL ESTIMATE: {quant_estimate.get('probability', '?')} (based on {quant_estimate.get('category', '?')} model)
KEY SIGNALS: {json.dumps(quant_estimate.get('signals', {}), indent=2, default=str)[:500]}"""

    prompt = f"""You are an expert financial analyst and superforecaster specializing in
macro economics, monetary policy, commodities, and crypto markets.

MARKET QUESTION: {question}
TODAY: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

{context}

Based on the hard financial data above, estimate the probability of this outcome.
Consider: current price levels, volatility regime, trend direction, macro backdrop,
and any structural factors that make this more or less likely.

Respond with ONLY a JSON object:
{{"probability": <0.01-0.99>, "confidence": <0.1-1.0>, "reasoning": "<2-3 sentences>"}}"""

    # Try up to 8 models with rotation
    for _ in range(8):
        model = AI_MODELS[_ai_model_idx % len(AI_MODELS)]
        _ai_model_idx += 1
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                continue  # rate limited, try next model
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            parsed = json.loads(content)
            return {
                "estimate": float(parsed["probability"]),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": parsed.get("reasoning", ""),
                "model_used": model.split("/")[-1].split(":")[0],
            }
        except Exception:
            continue

    return {"estimate": None, "reasoning": "All 8 models failed (rate limited)"}


# =========================================================
# Main Scanner
# =========================================================

def main():
    print("=" * 70)
    print("  FINANCIAL MARKET SCANNER")
    print("=" * 70)

    # Step 1: Pull all financial data
    print("\n  Fetching financial data...")
    fin = FinancialData()
    data = fin.get_all()

    oil_price = data["oil"].get("price", "?")
    vix_val = data["vix"].get("price", "?")
    y10_val = data["yield_10y"].get("price", "?")
    btc_price = data["btc"].get("price", "?")
    gold_price = data["gold"].get("price", "?")
    fg_val = data["fear_greed"].get("value", "?")
    fg_class = data["fear_greed"].get("classification", "?")

    print(f"    Oil:     ${oil_price}")
    print(f"    Gold:    ${gold_price}")
    print(f"    VIX:     {vix_val}")
    print(f"    10Y:     {y10_val}%")
    print(f"    BTC:     ${btc_price}")
    print(f"    F&G:     {fg_val} ({fg_class})")

    # Step 2: Find financial markets on Polymarket
    print("\n  Scanning Polymarket for financial markets...")
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_markets = []
    for offset in [0, 100, 200]:
        resp = session.get(f"{GAMMA_API}/markets", params={
            "limit": 100, "active": True, "closed": False,
            "order": "volume24hr", "ascending": False, "offset": offset,
        }, timeout=15)
        batch = resp.json()
        if not batch:
            break
        all_markets.extend(batch)

    # Filter to financial markets with tradeable prices
    finance_markets = []
    for m in all_markets:
        q = m.get("question", "")
        cats = classify_market(q)
        if not cats:
            continue
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        p = float(prices[0]) if prices else 0.0
        if p < 0.03 or p > 0.97:
            continue
        vol = float(m.get("volume", 0) or 0)
        finance_markets.append({
            "question": q,
            "price": p,
            "volume": vol,
            "categories": cats,
            "id": m.get("id"),
            "end_date": m.get("endDate") or m.get("end_date"),
            "raw": m,
        })

    finance_markets.sort(key=lambda x: -x["volume"])
    print(f"  Found {len(finance_markets)} financial markets\n")

    # Step 3: Analyze each market
    results = []
    for i, fm in enumerate(finance_markets):
        q = fm["question"]
        market_price = fm["price"]
        cats = fm["categories"]

        print(f"  [{i+1:>2}] {q[:65]}")
        print(f"       Market: {market_price:.3f} | Vol: ${fm['volume']:,.0f} | Cat: {','.join(cats)}")

        # Quantitative estimate based on category
        quant = {"probability": 0.5, "confidence": 0.3, "signals": {}, "category": "unknown"}
        if "fed_rate" in cats:
            quant = estimate_fed_rate(q, data)
        elif "oil" in cats:
            quant = estimate_oil_price(q, data)
        elif "crypto" in cats:
            quant = estimate_btc_price(q, data)

        # AI analysis with financial context
        ai = ai_financial_analysis(q, data, quant)

        # Combine: weight quant model + AI
        if ai.get("estimate") is not None:
            # Inverse-variance weighted
            q_var = 0.04 * (1 - quant["confidence"])
            a_var = 0.04 * (1 - ai["confidence"])
            q_w = (1/q_var) / (1/q_var + 1/a_var)
            a_w = (1/a_var) / (1/q_var + 1/a_var)
            combined = q_w * quant["probability"] + a_w * ai["estimate"]
            n_models = 2
            model_str = f"quant({q_w:.0%})+AI({a_w:.0%})"
        else:
            combined = quant["probability"]
            n_models = 1
            model_str = "quant_only"

        edge = combined - market_price

        # Apply spread filter
        spread = 0.02  # assume 2% spread
        if abs(edge) < spread:
            real_edge = 0.0
        else:
            real_edge = edge

        result = {
            "question": q,
            "market_price": market_price,
            "our_estimate": round(combined, 4),
            "quant_estimate": round(quant["probability"], 4),
            "ai_estimate": ai.get("estimate"),
            "ai_reasoning": ai.get("reasoning", ""),
            "edge": round(edge, 4),
            "real_edge": round(real_edge, 4),
            "categories": cats,
            "models": model_str,
            "n_models": n_models,
            "volume": fm["volume"],
            "signals": quant.get("signals", {}),
        }
        results.append(result)

        conf_tag = "HIGH" if abs(real_edge) > 0.05 else "MED" if abs(real_edge) > 0.02 else "LOW"
        marker = " <<<" if abs(real_edge) > 0.03 else ""
        print(f"       Estimate: {combined:.3f} | Edge: {edge:+.3f} [{conf_tag}] | {model_str}")
        if ai.get("reasoning"):
            print(f"       AI: {ai['reasoning'][:80]}")
        print(f"       {marker}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINANCIAL SCAN RESULTS")
    print(f"{'='*70}")

    with_edge = [r for r in results if abs(r["real_edge"]) > 0.02]
    print(f"  Markets scanned: {len(results)}")
    print(f"  With real edge > 2%: {len(with_edge)}")

    if with_edge:
        with_edge.sort(key=lambda x: -abs(x["real_edge"]))
        print(f"\n  TOP OPPORTUNITIES:")
        for j, r in enumerate(with_edge[:5]):
            action = "BUY_YES" if r["edge"] > 0 else "BUY_NO"
            print(f"    {j+1}. {r['question'][:55]}")
            print(f"       {action} | Mkt={r['market_price']:.3f} Est={r['our_estimate']:.3f} Edge={r['edge']:+.3f}")
            if r["ai_reasoning"]:
                print(f"       AI: {r['ai_reasoning'][:75]}")

    # Save
    out_path = "data/finance_scan_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "financial_data": {k: v for k, v in data.items() if k != "timestamp"},
            "n_markets": len(results),
            "n_with_edge": len(with_edge),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
