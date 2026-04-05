"""Comprehensive free API data sources for prediction market enrichment.

All APIs require NO authentication or API key.
Sources: CoinGecko, CoinCap, Coinpaprika, Fear & Greed Index,
         Wikipedia, US Treasury, Open-Meteo weather, GeoNames,
         USGS Earthquakes, WHO disease data, exchangerate.host
"""

import requests
from datetime import datetime, timezone
from typing import Optional


# ===========================================================================
# Crypto Data Sources (3 independent sources for cross-validation)
# ===========================================================================

class CryptoDataSource:
    """Primary crypto data: CoinGecko (free, no key)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_price(self, coin_id: str = "bitcoin", vs_currency: str = "usd") -> Optional[float]:
        resp = self.session.get(
            f"{self.BASE_URL}/simple/price",
            params={"ids": coin_id, "vs_currencies": vs_currency},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get(coin_id, {}).get(vs_currency)

    def get_price_history(
        self, coin_id: str = "bitcoin", days: int = 30, vs_currency: str = "usd"
    ) -> list[tuple[float, float]]:
        """Returns list of (timestamp_ms, price) tuples."""
        resp = self.session.get(
            f"{self.BASE_URL}/coins/{coin_id}/market_chart",
            params={"vs_currency": vs_currency, "days": days},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("prices", [])

    def get_market_data(self, coin_id: str = "bitcoin") -> dict:
        resp = self.session.get(f"{self.BASE_URL}/coins/{coin_id}", timeout=15)
        if resp.status_code != 200:
            return {}
        data = resp.json().get("market_data", {})
        return {
            "current_price": data.get("current_price", {}).get("usd"),
            "market_cap": data.get("market_cap", {}).get("usd"),
            "total_volume": data.get("total_volume", {}).get("usd"),
            "price_change_24h_pct": data.get("price_change_percentage_24h"),
            "price_change_7d_pct": data.get("price_change_percentage_7d"),
            "price_change_30d_pct": data.get("price_change_percentage_30d"),
            "ath": data.get("ath", {}).get("usd"),
            "ath_change_pct": data.get("ath_change_percentage", {}).get("usd"),
            "circulating_supply": data.get("circulating_supply"),
            "max_supply": data.get("max_supply"),
        }

    def get_trending(self) -> list[dict]:
        """Get trending coins — signals market attention/momentum."""
        resp = self.session.get(f"{self.BASE_URL}/search/trending", timeout=10)
        if resp.status_code != 200:
            return []
        return [
            {"name": c["item"]["name"], "symbol": c["item"]["symbol"], "market_cap_rank": c["item"].get("market_cap_rank")}
            for c in resp.json().get("coins", [])
        ]

    def get_global_data(self) -> dict:
        """Global crypto market overview."""
        resp = self.session.get(f"{self.BASE_URL}/global", timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json().get("data", {})
        return {
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptos": data.get("active_cryptocurrencies"),
            "market_cap_change_24h_pct": data.get("market_cap_change_percentage_24h_usd"),
        }


class CoinCapSource:
    """Secondary crypto data: CoinCap (free, no key). Cross-validation."""

    BASE_URL = "https://api.coincap.io/v2"

    def __init__(self):
        self.session = requests.Session()

    def get_asset(self, asset_id: str = "bitcoin") -> Optional[dict]:
        resp = self.session.get(f"{self.BASE_URL}/assets/{asset_id}", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", {})
        return {
            "price_usd": float(data["priceUsd"]) if data.get("priceUsd") else None,
            "market_cap": float(data["marketCapUsd"]) if data.get("marketCapUsd") else None,
            "volume_24h": float(data["volumeUsd24Hr"]) if data.get("volumeUsd24Hr") else None,
            "change_24h_pct": float(data["changePercent24Hr"]) if data.get("changePercent24Hr") else None,
            "vwap_24h": float(data["vwap24Hr"]) if data.get("vwap24Hr") else None,
            "supply": float(data["supply"]) if data.get("supply") else None,
        }

    def get_asset_history(self, asset_id: str = "bitcoin", interval: str = "d1") -> list[dict]:
        """Get historical prices. interval: m1, m5, m15, m30, h1, h2, h6, h12, d1"""
        resp = self.session.get(
            f"{self.BASE_URL}/assets/{asset_id}/history",
            params={"interval": interval},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        return [
            {"price": float(d["priceUsd"]), "time": d["time"]}
            for d in resp.json().get("data", [])
            if d.get("priceUsd")
        ]


class CoinpaprikaSource:
    """Tertiary crypto data: Coinpaprika (free, no key). Cross-validation."""

    BASE_URL = "https://api.coinpaprika.com/v1"

    def __init__(self):
        self.session = requests.Session()

    def get_ticker(self, coin_id: str = "btc-bitcoin") -> Optional[dict]:
        resp = self.session.get(f"{self.BASE_URL}/tickers/{coin_id}", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        quotes = data.get("quotes", {}).get("USD", {})
        return {
            "price": quotes.get("price"),
            "volume_24h": quotes.get("volume_24h"),
            "market_cap": quotes.get("market_cap"),
            "pct_change_1h": quotes.get("percent_change_1h"),
            "pct_change_24h": quotes.get("percent_change_24h"),
            "pct_change_7d": quotes.get("percent_change_7d"),
            "pct_change_30d": quotes.get("percent_change_30d"),
            "ath_price": quotes.get("ath_price"),
            "pct_from_ath": quotes.get("percent_from_price_ath"),
        }

    def get_global(self) -> Optional[dict]:
        resp = self.session.get(f"{self.BASE_URL}/global", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "market_cap_usd": data.get("market_cap_usd"),
            "volume_24h_usd": data.get("volume_24h_usd"),
            "btc_dominance": data.get("bitcoin_dominance_percentage"),
            "cryptocurrencies_number": data.get("cryptocurrencies_number"),
            "market_cap_change_24h": data.get("market_cap_ath_value"),
        }


# ===========================================================================
# Fear & Greed Index
# ===========================================================================

class FearGreedIndex:
    """Crypto Fear & Greed Index — free, no key."""

    URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self.session = requests.Session()

    def get_current(self) -> Optional[dict]:
        resp = self.session.get(self.URL, params={"limit": 1}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        entry = data[0]
        return {
            "value": int(entry.get("value", 0)),
            "classification": entry.get("value_classification", ""),
            "timestamp": entry.get("timestamp"),
        }

    def get_history(self, days: int = 30) -> list[dict]:
        resp = self.session.get(self.URL, params={"limit": days}, timeout=10)
        if resp.status_code != 200:
            return []
        return [
            {
                "value": int(d.get("value", 0)),
                "classification": d.get("value_classification", ""),
                "timestamp": d.get("timestamp"),
            }
            for d in resp.json().get("data", [])
        ]


# ===========================================================================
# News & Knowledge Sources
# ===========================================================================

class NewsDataSource:
    """Multi-source news and knowledge fetcher."""

    WIKI_API = "https://en.wikipedia.org/w/api.php"

    def __init__(self):
        self.session = requests.Session()

    def get_wikipedia_summary(self, topic: str) -> Optional[str]:
        resp = self.session.get(
            self.WIKI_API,
            params={
                "action": "query", "titles": topic, "prop": "extracts",
                "exintro": True, "explaintext": True, "format": "json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract")
        return None

    def search_wikipedia(self, query: str, limit: int = 5) -> list[str]:
        resp = self.session.get(
            self.WIKI_API,
            params={"action": "opensearch", "search": query, "limit": limit, "format": "json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data[1] if len(data) > 1 else []

    def get_wikipedia_current_events(self) -> Optional[str]:
        """Get current events portal — real-time world events."""
        today = datetime.now(timezone.utc)
        title = f"Portal:Current_events/{today.strftime('%Y_%B_%d')}"
        return self.get_wikipedia_summary(title)


class HackerNewsSource:
    """Hacker News API — free, no key. Tech/startup sentiment."""

    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self):
        self.session = requests.Session()

    def get_top_stories(self, limit: int = 30) -> list[dict]:
        resp = self.session.get(f"{self.BASE_URL}/topstories.json", timeout=10)
        if resp.status_code != 200:
            return []
        story_ids = resp.json()[:limit]
        stories = []
        for sid in story_ids[:limit]:
            s = self.session.get(f"{self.BASE_URL}/item/{sid}.json", timeout=5)
            if s.status_code == 200:
                data = s.json()
                if data:
                    stories.append({
                        "title": data.get("title", ""),
                        "score": data.get("score", 0),
                        "url": data.get("url", ""),
                        "descendants": data.get("descendants", 0),
                    })
        return stories

    def search_stories(self, query: str, limit: int = 10) -> list[dict]:
        """Search HN via Algolia API (free, no key)."""
        resp = self.session.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "hitsPerPage": limit},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return [
            {
                "title": h.get("title", ""),
                "points": h.get("points", 0),
                "num_comments": h.get("num_comments", 0),
                "url": h.get("url", ""),
                "created_at": h.get("created_at", ""),
            }
            for h in resp.json().get("hits", [])
        ]


# ===========================================================================
# Economic & Government Data
# ===========================================================================

class EconomicDataSource:
    """Free economic indicators from public government APIs."""

    def __init__(self):
        self.session = requests.Session()

    def get_us_treasury_rates(self) -> Optional[dict]:
        resp = self.session.get(
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
            "/v2/accounting/od/avg_interest_rates",
            params={"sort": "-record_date", "page[size]": 10},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        return {
            "record_date": data[0].get("record_date"),
            "avg_interest_rate": data[0].get("avg_interest_rate_amt"),
            "security_desc": data[0].get("security_desc"),
        }

    def get_us_debt(self) -> Optional[dict]:
        """US national debt — useful for fiscal policy markets."""
        resp = self.session.get(
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
            "/v2/accounting/od/debt_to_penny",
            params={"sort": "-record_date", "page[size]": 1},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        return {
            "record_date": data[0].get("record_date"),
            "total_debt": data[0].get("tot_pub_debt_out_amt"),
        }

    def get_exchange_rates(self, base: str = "USD") -> Optional[dict]:
        """Exchange rates from exchangerate.host (free, no key)."""
        resp = self.session.get(
            f"https://api.exchangerate.host/latest",
            params={"base": base},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "base": data.get("base"),
            "date": data.get("date"),
            "rates": data.get("rates", {}),
        }


# ===========================================================================
# Weather Data (for weather-dependent markets)
# ===========================================================================

class WeatherSource:
    """Open-Meteo weather API — completely free, no key."""

    BASE_URL = "https://api.open-meteo.com/v1"

    def __init__(self):
        self.session = requests.Session()

    def get_forecast(self, latitude: float, longitude: float, days: int = 7) -> Optional[dict]:
        resp = self.session.get(
            f"{self.BASE_URL}/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("daily", {})

    def get_historical(self, latitude: float, longitude: float, start_date: str, end_date: str) -> Optional[dict]:
        resp = self.session.get(
            f"{self.BASE_URL}/archive",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("daily", {})


# ===========================================================================
# Natural Disaster / Geophysical Events
# ===========================================================================

class USGSEarthquakeSource:
    """USGS Earthquake API — free, no key. For geophysical event markets."""

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"

    def __init__(self):
        self.session = requests.Session()

    def get_recent(self, min_magnitude: float = 4.5, limit: int = 20) -> list[dict]:
        resp = self.session.get(
            f"{self.BASE_URL}/query",
            params={
                "format": "geojson",
                "minmagnitude": min_magnitude,
                "limit": limit,
                "orderby": "time",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return [
            {
                "magnitude": f["properties"].get("mag"),
                "place": f["properties"].get("place"),
                "time": f["properties"].get("time"),
                "tsunami": f["properties"].get("tsunami"),
                "type": f["properties"].get("type"),
            }
            for f in resp.json().get("features", [])
        ]


# ===========================================================================
# Volatility Index (VIX Proxy) — from CBOE via Yahoo Finance
# ===========================================================================

class VolatilityIndexSource:
    """VIX and market volatility data — key regime signal.

    Uses Yahoo Finance v8 API (free, no key) for VIX index data,
    plus derived volatility from S&P 500 recent moves.
    Adapted from calendar spread strategy's VIX-based capital allocation.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_vix(self) -> Optional[dict]:
        """Get current VIX level and recent history."""
        try:
            resp = self.session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
                params={"interval": "1d", "range": "1mo"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("chart", {}).get("result", [{}])[0]
            meta = data.get("meta", {})
            closes = data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if not closes:
                return None
            import numpy as np
            return {
                "current": meta.get("regularMarketPrice", closes[-1]),
                "previous_close": meta.get("previousClose"),
                "mean_30d": float(np.mean(closes)),
                "std_30d": float(np.std(closes)),
                "min_30d": float(np.min(closes)),
                "max_30d": float(np.max(closes)),
                "percentile": float(np.searchsorted(np.sort(closes), closes[-1]) / len(closes)),
                "regime": self._classify_vix(closes[-1]),
            }
        except Exception:
            return None

    def get_sp500_volatility(self) -> Optional[dict]:
        """Get S&P 500 realized volatility as secondary VIX signal."""
        try:
            resp = self.session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
                params={"interval": "1d", "range": "3mo"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("chart", {}).get("result", [{}])[0]
            closes = data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if len(closes) < 20:
                return None
            import numpy as np
            prices = np.array(closes)
            returns = np.diff(np.log(prices))
            return {
                "realized_vol_20d": float(np.std(returns[-20:]) * np.sqrt(252) * 100),
                "realized_vol_60d": float(np.std(returns[-60:]) * np.sqrt(252) * 100) if len(returns) >= 60 else None,
                "sp500_last": float(closes[-1]),
                "sp500_change_1d": float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else None,
                "sp500_change_5d": float((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else None,
                "sp500_change_20d": float((closes[-1] - closes[-20]) / closes[-20] * 100) if len(closes) >= 20 else None,
            }
        except Exception:
            return None

    @staticmethod
    def _classify_vix(vix_value: float) -> str:
        if vix_value < 14:
            return "low_vol"
        elif vix_value < 20:
            return "normal"
        elif vix_value < 25:
            return "elevated"
        elif vix_value < 35:
            return "high"
        else:
            return "extreme"


# ===========================================================================
# Commodities Data (Oil, Gold, etc.)
# ===========================================================================

class CommoditiesSource:
    """Oil, gold, and commodity prices via Yahoo Finance (free, no key).

    Oil and gold prices affect geopolitical, economic, and inflation markets.
    Adapted from calendar spread's use of macro indicators for regime detection.
    """

    SYMBOLS = {
        "crude_oil": "CL=F",         # WTI Crude Oil Futures
        "brent_oil": "BZ=F",         # Brent Crude Oil Futures
        "gold": "GC=F",              # Gold Futures
        "silver": "SI=F",            # Silver Futures
        "natural_gas": "NG=F",       # Natural Gas Futures
        "copper": "HG=F",            # Copper Futures
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_commodity(self, name: str = "crude_oil") -> Optional[dict]:
        """Get current price and recent history for a commodity."""
        symbol = self.SYMBOLS.get(name, name)
        try:
            resp = self.session.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1d", "range": "1mo"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("chart", {}).get("result", [{}])[0]
            meta = data.get("meta", {})
            closes = data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if not closes:
                return None
            import numpy as np
            return {
                "commodity": name,
                "price": meta.get("regularMarketPrice", closes[-1]),
                "previous_close": meta.get("previousClose"),
                "change_1d_pct": float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else None,
                "change_5d_pct": float((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else None,
                "change_20d_pct": float((closes[-1] - closes[-20]) / closes[-20] * 100) if len(closes) >= 20 else None,
                "high_30d": float(np.max(closes)),
                "low_30d": float(np.min(closes)),
                "volatility_30d": float(np.std(np.diff(np.log(np.array(closes) + 1e-10))) * np.sqrt(252) * 100),
            }
        except Exception:
            return None

    def get_all_commodities(self) -> dict:
        """Get prices for all tracked commodities."""
        results = {}
        for name in self.SYMBOLS:
            data = self.get_commodity(name)
            if data:
                results[name] = data
        return results

    def get_oil_gold_ratio(self) -> Optional[float]:
        """Oil/Gold ratio — macro regime indicator."""
        oil = self.get_commodity("crude_oil")
        gold = self.get_commodity("gold")
        if oil and gold and oil.get("price") and gold.get("price"):
            return round(oil["price"] / gold["price"], 6)
        return None


# ===========================================================================
# Forex / Currency Data
# ===========================================================================

class ForexSource:
    """Currency exchange rates and DXY (Dollar Index) proxy.

    DXY strength affects global markets, commodity prices, and
    emerging market prediction markets.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_dxy(self) -> Optional[dict]:
        """Get US Dollar Index (DXY) — key macro regime signal."""
        try:
            resp = self.session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB",
                params={"interval": "1d", "range": "1mo"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("chart", {}).get("result", [{}])[0]
            meta = data.get("meta", {})
            closes = data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if not closes:
                return None
            import numpy as np
            return {
                "dxy": meta.get("regularMarketPrice", closes[-1]),
                "previous_close": meta.get("previousClose"),
                "change_1d_pct": float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else None,
                "change_5d_pct": float((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else None,
                "mean_30d": float(np.mean(closes)),
                "trend": "strengthening" if closes[-1] > np.mean(closes) else "weakening",
            }
        except Exception:
            return None

    def get_major_pairs(self) -> dict:
        """Get major forex pairs vs USD."""
        pairs = {
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "USD/JPY": "JPY=X",
            "USD/CNY": "CNY=X",
            "USD/INR": "INR=X",
        }
        results = {}
        for name, symbol in pairs.items():
            try:
                resp = self.session.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"interval": "1d", "range": "5d"},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                meta = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                prev = meta.get("previousClose")
                if price:
                    results[name] = {
                        "rate": price,
                        "change_pct": round((price - prev) / prev * 100, 4) if prev else None,
                    }
            except Exception:
                continue
        return results


# ===========================================================================
# Stock Market Indices
# ===========================================================================

class StockIndexSource:
    """Major stock indices — regime and sentiment signals.

    Adapted from calendar spread strategy's use of NIFTY/BANKNIFTY
    as underlying signals. Here we use global indices.
    """

    INDICES = {
        "sp500": "%5EGSPC",
        "nasdaq": "%5EIXIC",
        "dow": "%5EDJI",
        "russell2000": "%5ERUT",
        "nifty50": "%5ENSEI",
        "ftse100": "%5EFTSE",
        "dax": "%5EGDAXI",
        "nikkei225": "%5EN225",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_index(self, name: str = "sp500") -> Optional[dict]:
        """Get current index level and recent performance."""
        symbol = self.INDICES.get(name, name)
        try:
            resp = self.session.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1d", "range": "1mo"},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("chart", {}).get("result", [{}])[0]
            meta = data.get("meta", {})
            closes = data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if not closes:
                return None
            import numpy as np
            return {
                "index": name,
                "price": meta.get("regularMarketPrice", closes[-1]),
                "previous_close": meta.get("previousClose"),
                "change_1d_pct": float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else None,
                "change_5d_pct": float((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else None,
                "change_20d_pct": float((closes[-1] - closes[-20]) / closes[-20] * 100) if len(closes) >= 20 else None,
                "above_20d_avg": closes[-1] > float(np.mean(closes)),
            }
        except Exception:
            return None

    def get_market_breadth(self) -> Optional[dict]:
        """Compare multiple indices to gauge broad market health."""
        results = {}
        for name in ["sp500", "nasdaq", "dow", "russell2000"]:
            data = self.get_index(name)
            if data and data.get("change_5d_pct") is not None:
                results[name] = data["change_5d_pct"]

        if len(results) < 2:
            return None

        import numpy as np
        changes = list(results.values())
        return {
            "indices": results,
            "avg_change_5d": float(np.mean(changes)),
            "breadth": "bullish" if all(c > 0 for c in changes) else "bearish" if all(c < 0 for c in changes) else "mixed",
            "dispersion": float(np.std(changes)),
        }


# ===========================================================================
# News Sentiment Sources (additional)
# ===========================================================================

class GNewsSource:
    """Google News RSS — free, no key. Headline sentiment analysis."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def search_headlines(self, query: str, max_results: int = 10) -> list[dict]:
        """Search Google News for headlines related to a query."""
        try:
            import xml.etree.ElementTree as ET
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            results = []
            for item in items[:max_results]:
                title = item.find("title")
                pub_date = item.find("pubDate")
                source = item.find("source")
                results.append({
                    "title": title.text if title is not None else "",
                    "published": pub_date.text if pub_date is not None else "",
                    "source": source.text if source is not None else "",
                })
            return results
        except Exception:
            return []


class EventRegistrySource:
    """NewsAPI.org-style event tracking via free Currents API."""

    BASE_URL = "https://api.currentsapi.services/v1"

    def __init__(self):
        self.session = requests.Session()

    def search_news(self, query: str, language: str = "en") -> list[dict]:
        """Search recent news. Note: free tier has limits."""
        try:
            # Currents API free endpoint (no key for limited access)
            resp = self.session.get(
                f"{self.BASE_URL}/search",
                params={"keywords": query, "language": language, "type": 1},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            return [
                {
                    "title": n.get("title", ""),
                    "description": n.get("description", "")[:200],
                    "published": n.get("published", ""),
                    "category": n.get("category", []),
                }
                for n in resp.json().get("news", [])[:10]
            ]
        except Exception:
            return []


# ===========================================================================
# Cross-Validation Engine
# ===========================================================================

class DataCrossValidator:
    """Cross-validate data across multiple independent sources.

    When 3 sources agree on a price, confidence is high.
    When they diverge, something unusual is happening.
    """

    def __init__(self):
        self.coingecko = CryptoDataSource()
        self.coincap = CoinCapSource()
        self.coinpaprika = CoinpaprikaSource()

    # Mapping from common names to each API's ID format
    COIN_IDS = {
        "bitcoin": {"coingecko": "bitcoin", "coincap": "bitcoin", "coinpaprika": "btc-bitcoin"},
        "ethereum": {"coingecko": "ethereum", "coincap": "ethereum", "coinpaprika": "eth-ethereum"},
        "solana": {"coingecko": "solana", "coincap": "solana", "coinpaprika": "sol-solana"},
        "xrp": {"coingecko": "ripple", "coincap": "xrp", "coinpaprika": "xrp-xrp"},
        "dogecoin": {"coingecko": "dogecoin", "coincap": "dogecoin", "coinpaprika": "doge-dogecoin"},
    }

    def get_validated_price(self, coin: str = "bitcoin") -> dict:
        """Get price from 3 sources and compute consensus."""
        ids = self.COIN_IDS.get(coin, {
            "coingecko": coin, "coincap": coin, "coinpaprika": f"{coin[:3]}-{coin}",
        })
        prices = []
        sources = {}

        # Source 1: CoinGecko
        try:
            p = self.coingecko.get_price(ids["coingecko"])
            if p is not None:
                prices.append(p)
                sources["coingecko"] = p
        except Exception:
            pass

        # Source 2: CoinCap
        try:
            data = self.coincap.get_asset(ids["coincap"])
            if data and data.get("price_usd"):
                prices.append(data["price_usd"])
                sources["coincap"] = data["price_usd"]
        except Exception:
            pass

        # Source 3: Coinpaprika
        try:
            data = self.coinpaprika.get_ticker(ids["coinpaprika"])
            if data and data.get("price"):
                prices.append(data["price"])
                sources["coinpaprika"] = data["price"]
        except Exception:
            pass

        if not prices:
            return {"consensus_price": None, "n_sources": 0, "agreement": 0}

        import numpy as np
        prices_arr = np.array(prices)
        mean_price = float(np.mean(prices_arr))
        std_price = float(np.std(prices_arr))
        cv = std_price / mean_price if mean_price > 0 else 0  # coefficient of variation

        return {
            "consensus_price": mean_price,
            "median_price": float(np.median(prices_arr)),
            "std": std_price,
            "coefficient_of_variation": cv,
            "n_sources": len(prices),
            "sources": sources,
            "agreement": 1.0 - min(cv * 100, 1.0),  # high agreement when CV is low
        }

    def get_validated_market_data(self, coin: str = "bitcoin") -> dict:
        """Get comprehensive market data with cross-validation."""
        price_data = self.get_validated_price(coin)
        ids = self.COIN_IDS.get(coin, {
            "coingecko": coin, "coincap": coin, "coinpaprika": f"{coin[:3]}-{coin}",
        })

        # Collect 24h changes from all sources
        changes_24h = []
        try:
            gecko_data = self.coingecko.get_market_data(ids["coingecko"])
            if gecko_data.get("price_change_24h_pct") is not None:
                changes_24h.append(gecko_data["price_change_24h_pct"])
        except Exception:
            pass

        try:
            cap_data = self.coincap.get_asset(ids["coincap"])
            if cap_data and cap_data.get("change_24h_pct") is not None:
                changes_24h.append(cap_data["change_24h_pct"])
        except Exception:
            pass

        try:
            paprika_data = self.coinpaprika.get_ticker(ids["coinpaprika"])
            if paprika_data and paprika_data.get("pct_change_24h") is not None:
                changes_24h.append(paprika_data["pct_change_24h"])
        except Exception:
            pass

        import numpy as np
        consensus_change = float(np.mean(changes_24h)) if changes_24h else None

        return {
            **price_data,
            "consensus_change_24h_pct": consensus_change,
            "n_change_sources": len(changes_24h),
        }
