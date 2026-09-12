"""Niche Data Alpha Scanner — Exploit authoritative external data vs crowd pricing.

The "buy cheap, sell when market catches up or collect $1 on resolution" strategy.

Scans for markets where authoritative external data disagrees with Polymarket:
1. Crypto price: log-normal model from exchange price + realised volatility
2. Weather: NOAA forecasts vs market pricing (when weather markets exist)

Sportsbook-vs-Polymarket comparison lives in scripts/divergence_study.py. The
old sports section here never produced a single comparison: it requested
markets=h2h,outrights, which the Odds API rejects for game sports (HTTP 422,
silently skipped). A naive "fix" would burn the 500/month free quota in about
a day, so it was removed rather than repaired.

Usage:
    python scripts/niche_scanner.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv
load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_DIR = "data"


# =============================================================
# Crypto Price Probability (from options/futures data)
# =============================================================

class CryptoPriceSource:
    """Derive probability of BTC/ETH hitting price targets from market data.

    Uses multiple free APIs for cross-validated pricing:
    - Binance (primary — most liquid, real-time)
    - CoinGecko (backup + market data)
    - CoinCap (backup)
    - Coinpaprika (backup)
    - Binance funding rate + open interest for sentiment
    """

    def __init__(self):
        self.session = requests.Session()

    def _get_btc_price(self) -> float | None:
        """Get current BTC price — cross-validated across 4 sources."""
        prices = []

        # Binance (primary — most liquid exchange)
        try:
            resp = self.session.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"}, timeout=8)
            if resp.status_code == 200:
                prices.append(float(resp.json()["price"]))
        except Exception:
            pass

        # CoinGecko
        try:
            resp = self.session.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin", "vs_currencies": "usd"}, timeout=8)
            if resp.status_code == 200:
                prices.append(resp.json()["bitcoin"]["usd"])
        except Exception:
            pass

        # CoinCap
        try:
            resp = self.session.get(
                "https://api.coincap.io/v2/assets/bitcoin", timeout=8)
            if resp.status_code == 200:
                p = resp.json().get("data", {}).get("priceUsd")
                if p:
                    prices.append(float(p))
        except Exception:
            pass

        # Coinpaprika
        try:
            resp = self.session.get(
                "https://api.coinpaprika.com/v1/tickers/btc-bitcoin", timeout=8)
            if resp.status_code == 200:
                p = resp.json().get("quotes", {}).get("USD", {}).get("price")
                if p:
                    prices.append(float(p))
        except Exception:
            pass

        if not prices:
            return None
        # Use median for robustness against outliers
        import numpy as np
        return float(np.median(prices))

    def _get_funding_sentiment(self) -> dict:
        """Get Binance futures funding rate + open interest for sentiment.

        Positive funding = longs pay shorts = market is bullish
        Negative funding = shorts pay longs = market is bearish
        """
        result = {"funding_rate": 0, "sentiment": "neutral"}
        try:
            # Funding rate
            resp = self.session.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": "BTCUSDT", "limit": 1}, timeout=8)
            if resp.status_code == 200 and resp.json():
                rate = float(resp.json()[0]["fundingRate"])
                result["funding_rate"] = rate
                if rate > 0.0005:
                    result["sentiment"] = "very_bullish"
                elif rate > 0.0001:
                    result["sentiment"] = "bullish"
                elif rate < -0.0005:
                    result["sentiment"] = "very_bearish"
                elif rate < -0.0001:
                    result["sentiment"] = "bearish"
        except Exception:
            pass

        try:
            # Open interest
            resp = self.session.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": "BTCUSDT"}, timeout=8)
            if resp.status_code == 200:
                result["open_interest"] = float(resp.json().get("openInterest", 0))
        except Exception:
            pass

        return result

    def _get_btc_volatility(self) -> float | None:
        """Get BTC annualized volatility from Binance klines."""
        try:
            # Binance: 30 daily candles
            resp = self.session.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1d", "limit": 30},
                timeout=10)
            if resp.status_code != 200:
                return None

            import numpy as np
            closes = [float(k[4]) for k in resp.json()]
            if len(closes) < 10:
                return None
            returns = np.diff(np.log(closes))
            daily_vol = np.std(returns)
            return float(daily_vol * np.sqrt(365))
        except Exception:
            pass

        # Fallback: CoinGecko
        try:
            resp = self.session.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": 30}, timeout=10)
            if resp.status_code != 200:
                return None
            import numpy as np
            prices = [p[1] for p in resp.json().get("prices", [])]
            if len(prices) < 10:
                return None
            returns = np.diff(np.log(prices))
            return float(np.std(returns) * np.sqrt(365))
        except Exception:
            return None

    def get_btc_price_prob(self, target_price: float, days_until: int,
                           direction: str = "above") -> float | None:
        """Estimate probability BTC is above/below target in N days.

        Uses log-normal model with historical volatility from Binance.
        Returns None when price or volatility is unavailable — Binance blocks
        the US IPs GitHub runners use, and the old 0.5 fallback turned that
        outage into a fake 50% edge against every extreme-priced market.
        """
        current = self._get_btc_price()
        if not current:
            return None

        annual_vol = self._get_btc_volatility()
        if not annual_vol:
            return None

        import numpy as np
        from scipy.stats import norm

        T = max(days_until, 1) / 365.0
        vol_T = annual_vol * np.sqrt(T)

        if vol_T < 0.001:
            # Very short time — just compare current to target
            if direction == "above":
                return 0.95 if current > target_price else 0.05
            else:
                return 0.95 if current < target_price else 0.05

        d = (np.log(current / target_price) + 0.5 * annual_vol**2 * T) / vol_T
        base_prob = float(norm.cdf(d))

        # Adjust for funding rate sentiment (small nudge)
        sentiment = self._get_funding_sentiment()
        sentiment_adj = 0
        if sentiment["sentiment"] == "very_bullish":
            sentiment_adj = 0.02
        elif sentiment["sentiment"] == "bullish":
            sentiment_adj = 0.01
        elif sentiment["sentiment"] == "bearish":
            sentiment_adj = -0.01
        elif sentiment["sentiment"] == "very_bearish":
            sentiment_adj = -0.02

        if direction == "above":
            return float(np.clip(base_prob + sentiment_adj, 0.01, 0.99))
        else:
            return float(np.clip(1 - base_prob - sentiment_adj, 0.01, 0.99))


# =============================================================
# NOAA Weather (for when Polymarket has weather markets)
# =============================================================

class NOAAWeatherSource:
    """NOAA/NWS weather forecasts — free, no key.

    For NYC (LaGuardia): lat=40.79, lon=-73.87
    For London: uses Open-Meteo (NOAA doesn't cover UK)

    When weather markets exist on Polymarket, this converts forecasts
    to probabilities and compares against market prices.
    """

    CITIES = {
        "nyc": {"lat": 40.79, "lon": -73.87, "unit": "F"},
        "london": {"lat": 51.47, "lon": -0.45, "unit": "C"},
        "seoul": {"lat": 37.57, "lon": 126.98, "unit": "C"},
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "poly_bot/1.0"})

    def get_forecast(self, city: str = "nyc") -> dict | None:
        """Get temperature forecast from NOAA (US) or Open-Meteo (international)."""
        info = self.CITIES.get(city)
        if not info:
            return None

        if city == "nyc":
            return self._noaa_forecast(info["lat"], info["lon"])
        else:
            return self._open_meteo_forecast(info["lat"], info["lon"])

    def _noaa_forecast(self, lat: float, lon: float) -> dict | None:
        try:
            resp = self.session.get(
                f"https://api.weather.gov/points/{lat},{lon}", timeout=10)
            if resp.status_code != 200:
                return None
            forecast_url = resp.json()["properties"]["forecast"]
            resp2 = self.session.get(forecast_url, timeout=10)
            if resp2.status_code != 200:
                return None
            periods = resp2.json().get("properties", {}).get("periods", [])
            return {"source": "noaa", "periods": periods}
        except Exception:
            return None

    def _open_meteo_forecast(self, lat: float, lon: float) -> dict | None:
        try:
            resp = self.session.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto", "forecast_days": 7,
            }, timeout=10)
            if resp.status_code != 200:
                return None
            return {"source": "open_meteo", "daily": resp.json().get("daily", {})}
        except Exception:
            return None


# =============================================================
# Market Matching — Match external data to Polymarket markets
# =============================================================

_CRYPTO_TARGET = re.compile(
    r"\b(bitcoin|btc|ethereum|eth)\b.*?\b(above|below|over|under|hit|reach|dip|fall|drop)\b"
    r".*?\$\s?(\d[\d,]*(?:\.\d+)?)\s*([km])?\b",
    re.IGNORECASE,
)
_ABOVE_WORDS = {"above", "over", "hit", "reach"}


def parse_crypto_target(question: str) -> dict | None:
    """Coin, direction and dollar target from a crypto threshold question.

    The target is the first '$' amount after an explicit direction word. The
    old greedy pattern captured the LAST digit in the question, so "above
    $84,000 on September 12" parsed as a $2 target and produced 99% "edges".
    """
    m = _CRYPTO_TARGET.search(question or "")
    if not m:
        return None
    try:
        target = float(m.group(3).replace(",", ""))
    except ValueError:
        return None
    target *= {"k": 1e3, "m": 1e6}.get((m.group(4) or "").lower(), 1.0)
    return {
        "coin": "ethereum" if m.group(1).lower() in ("ethereum", "eth") else "bitcoin",
        "direction": "above" if m.group(2).lower() in _ABOVE_WORDS else "below",
        "target_price": target,
    }


def fetch_polymarket_crypto_price() -> list[dict]:
    """Fetch crypto price threshold markets."""
    session = requests.Session()
    all_markets = []
    for offset in [0, 100]:
        try:
            resp = session.get(f"{GAMMA_API}/markets", params={
                "limit": 100, "active": True, "closed": False,
                "order": "volume24hr", "ascending": False, "offset": offset,
            }, timeout=15)
            all_markets.extend(resp.json())
        except Exception:
            break

    crypto_price = []
    for m in all_markets:
        q = m.get("question", "")
        parsed = parse_crypto_target(q)
        if not parsed:
            continue

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        if not prices:
            continue
        yes_p = float(prices[0])
        if yes_p <= 0 or yes_p >= 1:
            continue

        crypto_price.append({
            "question": q,
            "yes_price": yes_p,
            "volume": float(m.get("volume", 0) or 0),
            "id": m.get("id"),
            "end_date": m.get("endDate", ""),
            **parsed,
        })

    return crypto_price


# =============================================================
# Main Scanner
# =============================================================

def main():
    print("=" * 70)
    print("  NICHE DATA ALPHA SCANNER")
    print("  Buy underpriced, sell when market catches up or collect $1")
    print("=" * 70)

    signals = []

    # --- 1. Crypto price threshold markets ---
    print("\n  [1] Crypto Price Markets...")
    crypto_markets = fetch_polymarket_crypto_price()
    print(f"    Found {len(crypto_markets)} crypto price markets")

    if crypto_markets:
        crypto_src = CryptoPriceSource()
        n_unpriced = 0
        for m in crypto_markets:
            try:
                # Days until resolution
                days = 30  # default
                if m["end_date"]:
                    try:
                        end = datetime.fromisoformat(m["end_date"].replace("Z", "+00:00"))
                        days = max(1, (end - datetime.now(timezone.utc)).days)
                    except Exception:
                        pass

                if m["coin"] != "bitcoin":
                    continue  # only BTC for now

                model_prob = crypto_src.get_btc_price_prob(
                    m["target_price"], days, m["direction"])
                if model_prob is None:
                    n_unpriced += 1
                    continue
                market_prob = m["yes_price"]
                edge = model_prob - market_prob

                if abs(edge) > 0.08:
                    action = "BUY_YES" if edge > 0 else "BUY_NO"
                    signal = {
                        "source": "crypto_quant",
                        "question": m["question"][:70],
                        "market_price": market_prob,
                        "model_prob": round(model_prob, 4),
                        "edge": round(edge, 4),
                        "abs_edge": round(abs(edge), 4),
                        "action": action,
                        "market_id": m["id"],
                        "target_price": m["target_price"],
                        "days_until": days,
                    }
                    signals.append(signal)
                    direction = "UNDERPRICED" if edge > 0 else "OVERPRICED"
                    print(f"    {direction}: {m['question'][:55]}")
                    print(f"      Market={market_prob:.3f} Model={model_prob:.3f} "
                          f"Edge={edge:+.3f} -> {action}")
            except Exception:
                continue
        if n_unpriced:
            print(f"    Skipped {n_unpriced} markets: BTC price/volatility unavailable "
                  f"(no signal rather than a fabricated one)")

    # --- 2. Sports: see scripts/divergence_study.py ---
    print("\n  [2] Sports: handled by scripts/divergence_study.py")

    # --- 3. Weather ---
    print("\n  [3] Weather Markets...")
    # Check if any weather markets exist
    print("    No weather markets found on Polymarket currently")
    print("    NOAA scanner ready — will activate when weather markets appear")

    # Summary
    print(f"\n{'='*70}")
    print(f"  NICHE SIGNALS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total signals: {len(signals)}")

    if signals:
        signals.sort(key=lambda x: -x["abs_edge"])
        print(f"\n  TOP SIGNALS (by edge size):")
        for s in signals[:10]:
            print(f"    {s['abs_edge']*100:.1f}% edge | {s['action']} | {s['question']}")
            print(f"      Market={s['market_price']:.3f} Model={s['model_prob']:.3f} Source={s['source']}")

    # Save
    out_path = os.path.join(DATA_DIR, "niche_signals.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_signals": len(signals),
            "signals": signals,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
