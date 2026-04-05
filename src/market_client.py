"""Polymarket Gamma API client for fetching market data."""

import json
import requests
from typing import Optional


GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class MarketClient:
    """Fetches market data from Polymarket's free Gamma API."""

    def __init__(self, base_url: str = GAMMA_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_markets(
        self,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[dict]:
        params = {
            "limit": limit,
            "active": active,
            "closed": closed,
            "order": order,
            "ascending": ascending,
        }
        resp = self.session.get(f"{self.base_url}/markets", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_market(self, market_id: str) -> dict:
        resp = self.session.get(f"{self.base_url}/markets/{market_id}")
        resp.raise_for_status()
        return resp.json()

    def get_events(self, limit: int = 50) -> list[dict]:
        resp = self.session.get(
            f"{self.base_url}/events", params={"limit": limit, "active": True}
        )
        resp.raise_for_status()
        return resp.json()

    def search_markets(self, query: str, limit: int = 20) -> list[dict]:
        resp = self.session.get(
            f"{self.base_url}/markets",
            params={"limit": limit, "active": True, "closed": False, "tag": query},
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def parse_market(raw: dict) -> dict:
        """Extract key fields from a raw market response."""
        outcomes = raw.get("outcomes", [])
        prices = raw.get("outcomePrices", [])
        # API returns these as JSON strings — parse if needed
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = []
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except (json.JSONDecodeError, TypeError):
                prices = []
        outcome_prices = {}
        for i, outcome in enumerate(outcomes):
            try:
                outcome_prices[outcome] = float(prices[i]) if i < len(prices) else None
            except (ValueError, TypeError):
                outcome_prices[outcome] = None

        return {
            "id": raw.get("id"),
            "question": raw.get("question", ""),
            "slug": raw.get("slug", ""),
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "volume": _safe_float(raw.get("volume")),
            "volume_24h": _safe_float(raw.get("volume24hr")),
            "volume_1w": _safe_float(raw.get("volume1wk")),
            "liquidity": _safe_float(raw.get("liquidity")),
            "last_trade_price": _safe_float(raw.get("lastTradePrice")),
            "best_bid": _safe_float(raw.get("bestBid")),
            "best_ask": _safe_float(raw.get("bestAsk")),
            "spread": _compute_spread(raw),
            "end_date": raw.get("endDate"),
            "active": raw.get("active", False),
            "closed": raw.get("closed", False),
        }


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _compute_spread(raw: dict) -> Optional[float]:
    bid = _safe_float(raw.get("bestBid"))
    ask = _safe_float(raw.get("bestAsk"))
    if bid is not None and ask is not None:
        return round(ask - bid, 6)
    return None
