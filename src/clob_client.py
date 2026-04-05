"""Polymarket CLOB API Client — Real orderbook data.

The CLOB (Central Limit Order Book) API provides:
- Real-time orderbook depth (bids/asks)
- Token-level pricing
- Market metadata with token IDs
- Trade history

Based on patterns from: Polymarket/agents, discountry/polymarket-trading-bot

No authentication required for read-only market data.
"""

import requests
from typing import Optional


CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


class ClobClient:
    """Read-only CLOB API client for orderbook and pricing data."""

    def __init__(self, base_url: str = CLOB_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_markets(self, limit: int = 100, next_cursor: str = "") -> dict:
        """Get CLOB markets with token IDs and pricing."""
        params = {"limit": limit}
        if next_cursor:
            params["next_cursor"] = next_cursor
        resp = self.session.get(f"{self.base_url}/markets", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_market(self, condition_id: str) -> Optional[dict]:
        """Get a specific CLOB market by condition ID."""
        resp = self.session.get(f"{self.base_url}/markets/{condition_id}", timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()

    def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Get full orderbook (bids + asks) for a token.

        Returns:
            {
                "bids": [{"price": "0.55", "size": "100"}, ...],
                "asks": [{"price": "0.56", "size": "50"}, ...],
            }
        """
        resp = self.session.get(
            f"{self.base_url}/book",
            params={"token_id": token_id},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        return resp.json()

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        resp = self.session.get(
            f"{self.base_url}/midpoint",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        mid = data.get("mid")
        return float(mid) if mid is not None else None

    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        """Get best price for a token on a given side."""
        resp = self.session.get(
            f"{self.base_url}/price",
            params={"token_id": token_id, "side": side},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = data.get("price")
        return float(price) if price is not None else None

    def get_spread(self, token_id: str) -> Optional[dict]:
        """Get bid-ask spread for a token."""
        resp = self.session.get(
            f"{self.base_url}/spread",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json()

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price."""
        resp = self.session.get(
            f"{self.base_url}/last-trade-price",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = data.get("price")
        return float(price) if price is not None else None

    def parse_orderbook(self, orderbook: dict) -> dict:
        """Parse raw orderbook into structured analysis."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        bid_prices = [float(b["price"]) for b in bids if b.get("price")]
        ask_prices = [float(a["price"]) for a in asks if a.get("price")]
        bid_sizes = [float(b["size"]) for b in bids if b.get("size")]
        ask_sizes = [float(a["size"]) for a in asks if a.get("size")]

        best_bid = max(bid_prices) if bid_prices else None
        best_ask = min(ask_prices) if ask_prices else None

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(best_ask - best_bid, 6) if best_bid and best_ask else None,
            "midpoint": round((best_bid + best_ask) / 2, 6) if best_bid and best_ask else None,
            "bid_depth": sum(bid_sizes),
            "ask_depth": sum(ask_sizes),
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "imbalance": self._compute_imbalance(bid_sizes, ask_sizes),
            "weighted_mid": self._vwap_mid(bid_prices, bid_sizes, ask_prices, ask_sizes),
        }

    @staticmethod
    def _compute_imbalance(bid_sizes: list[float], ask_sizes: list[float]) -> Optional[float]:
        """Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
        > 0 = buy pressure, < 0 = sell pressure."""
        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        total = total_bid + total_ask
        if total == 0:
            return None
        return round((total_bid - total_ask) / total, 6)

    @staticmethod
    def _vwap_mid(
        bid_prices: list[float], bid_sizes: list[float],
        ask_prices: list[float], ask_sizes: list[float],
    ) -> Optional[float]:
        """Volume-weighted midpoint — better than simple midpoint."""
        if not bid_prices or not ask_prices:
            return None
        # Use top 5 levels
        bid_p = bid_prices[:5]
        bid_s = bid_sizes[:5]
        ask_p = ask_prices[:5]
        ask_s = ask_sizes[:5]

        bid_vwap = sum(p * s for p, s in zip(bid_p, bid_s))
        ask_vwap = sum(p * s for p, s in zip(ask_p, ask_s))
        total_vol = sum(bid_s) + sum(ask_s)

        if total_vol == 0:
            return None
        return round((bid_vwap + ask_vwap) / total_vol, 6)


class ArbitrageDetector:
    """Complete-set arbitrage detection (from ent0n29/polybot pattern).

    In prediction markets, a "complete set" of outcomes should sum to $1.
    If YES + NO prices < $1, you can buy both and guarantee profit.
    If YES + NO prices > $1, the market is inefficient in the other direction.

    This also works for multi-outcome markets (e.g., "Who wins the election?"
    where all candidate prices should sum to ~$1).
    """

    def __init__(self, clob: Optional[ClobClient] = None):
        self.clob = clob or ClobClient()

    def check_binary_arbitrage(self, yes_token_id: str, no_token_id: str) -> dict:
        """Check for arbitrage in a binary (Yes/No) market.

        If best_ask(YES) + best_ask(NO) < 1.0, buying both = guaranteed profit.
        """
        yes_price = self.clob.get_price(yes_token_id, side="buy")
        no_price = self.clob.get_price(no_token_id, side="buy")

        if yes_price is None or no_price is None:
            return {"arbitrage": False, "error": "Could not fetch prices"}

        total_cost = yes_price + no_price
        profit_per_dollar = 1.0 - total_cost  # payout is always $1

        return {
            "arbitrage": profit_per_dollar > 0.001,  # > 0.1% to cover fees
            "yes_price": yes_price,
            "no_price": no_price,
            "total_cost": round(total_cost, 6),
            "profit_per_share": round(profit_per_dollar, 6),
            "profit_pct": round(profit_per_dollar * 100, 4) if total_cost > 0 else 0,
            "strategy": "BUY_COMPLETE_SET" if profit_per_dollar > 0.001 else "NONE",
        }

    def check_multi_outcome_arbitrage(self, token_ids: list[str]) -> dict:
        """Check for arbitrage in a multi-outcome market.

        Sum of all outcome prices should = $1. If < $1, buy all = profit.
        """
        prices = []
        for tid in token_ids:
            p = self.clob.get_price(tid, side="buy")
            if p is None:
                return {"arbitrage": False, "error": f"Could not fetch price for {tid}"}
            prices.append(p)

        total_cost = sum(prices)
        profit = 1.0 - total_cost

        return {
            "arbitrage": profit > 0.001,
            "prices": prices,
            "total_cost": round(total_cost, 6),
            "profit_per_share": round(profit, 6),
            "n_outcomes": len(token_ids),
            "strategy": "BUY_ALL_OUTCOMES" if profit > 0.001 else "NONE",
        }

    def scan_for_arbitrage(self, limit: int = 50) -> list[dict]:
        """Scan CLOB markets for arbitrage opportunities."""
        opportunities = []
        data = self.clob.get_markets(limit=limit)
        markets = data.get("data", []) if isinstance(data, dict) else data

        for market in markets:
            tokens = market.get("tokens", [])
            if len(tokens) < 2:
                continue

            if not market.get("active") or market.get("closed"):
                continue

            token_ids = [t["token_id"] for t in tokens]

            if len(tokens) == 2:
                result = self.check_binary_arbitrage(token_ids[0], token_ids[1])
            else:
                result = self.check_multi_outcome_arbitrage(token_ids)

            if result.get("arbitrage"):
                result["question"] = market.get("question", "")
                result["condition_id"] = market.get("condition_id", "")
                opportunities.append(result)

        opportunities.sort(key=lambda x: x.get("profit_per_share", 0), reverse=True)
        return opportunities


class OrderbookAnalyzer:
    """Deep orderbook analysis for edge detection.

    Extracts signals from order flow, depth, and microstructure
    that aren't visible from just the top-of-book prices.
    """

    def __init__(self, clob: Optional[ClobClient] = None):
        self.clob = clob or ClobClient()

    def full_analysis(self, token_id: str) -> Optional[dict]:
        """Run complete orderbook analysis for a token."""
        ob = self.clob.get_orderbook(token_id)
        if not ob:
            return None

        parsed = self.clob.parse_orderbook(ob)

        # Additional depth analysis
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        # Depth at various levels
        depth_analysis = self._depth_at_levels(bids, asks)

        # Large order detection (whale spotting)
        whale_bids = self._detect_large_orders(bids)
        whale_asks = self._detect_large_orders(asks)

        # Price impact estimation
        impact = self._estimate_price_impact(bids, asks, trade_size=100)

        return {
            **parsed,
            "depth_analysis": depth_analysis,
            "whale_bids": whale_bids,
            "whale_asks": whale_asks,
            "price_impact_100": impact,
            "liquidity_score": self._liquidity_score(parsed),
        }

    @staticmethod
    def _depth_at_levels(bids: list, asks: list) -> dict:
        """Calculate cumulative depth at 1%, 2%, 5% from midpoint."""
        if not bids or not asks:
            return {}

        bid_prices = [float(b["price"]) for b in bids if b.get("price")]
        ask_prices = [float(a["price"]) for a in asks if a.get("price")]
        if not bid_prices or not ask_prices:
            return {}

        mid = (max(bid_prices) + min(ask_prices)) / 2

        result = {}
        for pct in [0.01, 0.02, 0.05]:
            bid_depth = sum(
                float(b.get("size", 0)) for b in bids
                if float(b.get("price", 0)) >= mid * (1 - pct)
            )
            ask_depth = sum(
                float(a.get("size", 0)) for a in asks
                if float(a.get("price", 0)) <= mid * (1 + pct)
            )
            result[f"bid_depth_{int(pct*100)}pct"] = round(bid_depth, 2)
            result[f"ask_depth_{int(pct*100)}pct"] = round(ask_depth, 2)

        return result

    @staticmethod
    def _detect_large_orders(orders: list, percentile: float = 95) -> list[dict]:
        """Detect unusually large orders (potential whale activity)."""
        if not orders:
            return []
        sizes = [float(o.get("size", 0)) for o in orders]
        if not sizes:
            return []

        import numpy as np
        threshold = float(np.percentile(sizes, percentile))
        return [
            {"price": float(o["price"]), "size": float(o["size"])}
            for o in orders
            if float(o.get("size", 0)) >= threshold and threshold > 0
        ]

    @staticmethod
    def _estimate_price_impact(bids: list, asks: list, trade_size: float) -> dict:
        """Estimate price impact of a trade of given size."""
        def walk_book(orders, size, ascending=True):
            sorted_orders = sorted(orders, key=lambda x: float(x.get("price", 0)), reverse=not ascending)
            filled = 0
            total_cost = 0
            for o in sorted_orders:
                price = float(o.get("price", 0))
                available = float(o.get("size", 0))
                fill = min(available, size - filled)
                total_cost += fill * price
                filled += fill
                if filled >= size:
                    break
            avg_price = total_cost / filled if filled > 0 else None
            return {"filled": filled, "avg_price": round(avg_price, 6) if avg_price else None}

        buy_impact = walk_book(asks, trade_size, ascending=True)
        sell_impact = walk_book(bids, trade_size, ascending=False)

        return {"buy": buy_impact, "sell": sell_impact}

    @staticmethod
    def _liquidity_score(parsed: dict) -> float:
        """Score 0-100 for how liquid this market is."""
        score = 0
        spread = parsed.get("spread")
        if spread is not None:
            if spread < 0.01:
                score += 40
            elif spread < 0.03:
                score += 25
            elif spread < 0.05:
                score += 10

        bid_depth = parsed.get("bid_depth", 0)
        ask_depth = parsed.get("ask_depth", 0)
        total_depth = bid_depth + ask_depth
        if total_depth > 10000:
            score += 30
        elif total_depth > 1000:
            score += 20
        elif total_depth > 100:
            score += 10

        levels = (parsed.get("bid_levels", 0) + parsed.get("ask_levels", 0))
        if levels > 20:
            score += 30
        elif levels > 10:
            score += 20
        elif levels > 5:
            score += 10

        return min(score, 100)
