"""RAG-style News Context Engine (inspired by Polymarket/agents).

Gathers relevant news/context for a market question from multiple
free sources, builds a context string, and feeds it to the AI model
for better-informed probability estimates.

Sources (all free, no auth):
- Hacker News (Algolia search API)
- Wikipedia (summary + search)
- Polymarket Data API (related markets)
- CoinGecko (trending + news signals)

Unlike the Polymarket/agents approach (which uses Chroma vector DB),
we use a simpler but effective keyword-based retrieval that doesn't
require any infrastructure — pure API calls.
"""

import requests
from datetime import datetime, timezone
from typing import Optional


class NewsRAGEngine:
    """Retrieve and rank news context relevant to a market question."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def gather_context(self, question: str, max_items: int = 10) -> dict:
        """Gather all relevant context for a market question.

        Returns a structured context dict and a formatted text summary
        suitable for injection into an LLM prompt.
        """
        keywords = self._extract_search_terms(question)
        query = " ".join(keywords[:5])

        sources = {}

        # 1. Hacker News (tech, crypto, politics — broad coverage)
        hn_results = self._search_hacker_news(query, limit=5)
        if hn_results:
            sources["hacker_news"] = hn_results

        # 2. Wikipedia context
        wiki_summary = self._get_wikipedia_context(keywords)
        if wiki_summary:
            sources["wikipedia"] = wiki_summary

        # 3. Related Polymarket markets (what does the crowd think?)
        related = self._get_related_markets(query)
        if related:
            sources["related_markets"] = related

        # 4. Crypto-specific context if relevant
        if self._is_crypto_related(keywords):
            crypto_ctx = self._get_crypto_context(keywords)
            if crypto_ctx:
                sources["crypto"] = crypto_ctx

        # Build formatted context string
        context_text = self._format_context(question, sources)

        return {
            "query": query,
            "keywords": keywords,
            "sources": sources,
            "n_items": sum(
                len(v) if isinstance(v, list) else 1
                for v in sources.values()
            ),
            "context_text": context_text,
        }

    def _search_hacker_news(self, query: str, limit: int = 5) -> list[dict]:
        """Search Hacker News via Algolia (free, no key)."""
        try:
            resp = self.session.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={"query": query, "hitsPerPage": limit, "tags": "story"},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            return [
                {
                    "title": h.get("title", ""),
                    "points": h.get("points", 0),
                    "num_comments": h.get("num_comments", 0),
                    "created_at": h.get("created_at", ""),
                    "url": h.get("url", ""),
                }
                for h in resp.json().get("hits", [])
                if h.get("title")
            ]
        except Exception:
            return []

    def _get_wikipedia_context(self, keywords: list[str]) -> list[dict]:
        """Get Wikipedia summaries for key entities in the question."""
        results = []
        # Try the most specific keywords first
        for kw in keywords[:3]:
            try:
                resp = self.session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query", "titles": kw, "prop": "extracts",
                        "exintro": True, "explaintext": True, "exsentences": 3,
                        "format": "json",
                    },
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract and len(extract) > 50:
                        results.append({
                            "topic": page.get("title", kw),
                            "summary": extract[:500],
                        })
                        break
            except Exception:
                continue
        return results

    def _get_related_markets(self, query: str) -> list[dict]:
        """Find related Polymarket markets for crowd wisdom context."""
        try:
            resp = self.session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": 5, "active": True, "closed": False},
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            # Simple keyword matching (could be improved with embeddings)
            query_words = set(query.lower().split())
            results = []
            for m in resp.json():
                q = m.get("question", "")
                q_words = set(q.lower().split())
                overlap = len(query_words & q_words)
                if overlap >= 2:
                    import json
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except Exception:
                            prices = []
                    yes_price = float(prices[0]) if prices else None
                    results.append({
                        "question": q[:100],
                        "yes_price": yes_price,
                        "volume": m.get("volume"),
                        "relevance": overlap,
                    })
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:3]
        except Exception:
            return []

    def _get_crypto_context(self, keywords: list[str]) -> dict:
        """Get crypto market context if question is crypto-related."""
        coin_map = {
            "bitcoin": "bitcoin", "btc": "bitcoin",
            "ethereum": "ethereum", "eth": "ethereum",
            "solana": "solana", "sol": "solana",
            "xrp": "ripple", "dogecoin": "dogecoin",
        }
        coin_id = None
        for kw in keywords:
            if kw.lower() in coin_map:
                coin_id = coin_map[kw.lower()]
                break

        if not coin_id:
            coin_id = "bitcoin"  # default for generic crypto questions

        try:
            resp = self.session.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                timeout=15,
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
            md = data.get("market_data", {})
            return {
                "coin": coin_id,
                "price": md.get("current_price", {}).get("usd"),
                "change_24h": md.get("price_change_percentage_24h"),
                "change_7d": md.get("price_change_percentage_7d"),
                "change_30d": md.get("price_change_percentage_30d"),
                "market_cap": md.get("market_cap", {}).get("usd"),
                "ath": md.get("ath", {}).get("usd"),
                "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
                "description": data.get("description", {}).get("en", "")[:300],
            }
        except Exception:
            return {}

    def _format_context(self, question: str, sources: dict) -> str:
        """Format gathered context into a string for LLM injection."""
        lines = [f"CONTEXT for: {question}", f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", ""]

        # News
        hn = sources.get("hacker_news", [])
        if hn:
            lines.append("RECENT NEWS:")
            for item in hn:
                lines.append(f"  - {item['title']} (points: {item['points']}, comments: {item['num_comments']})")
            lines.append("")

        # Wikipedia
        wiki = sources.get("wikipedia", [])
        if wiki:
            lines.append("BACKGROUND:")
            for item in wiki:
                lines.append(f"  {item['topic']}: {item['summary'][:200]}")
            lines.append("")

        # Related markets
        related = sources.get("related_markets", [])
        if related:
            lines.append("RELATED POLYMARKET MARKETS:")
            for m in related:
                price_str = f"{m['yes_price']:.1%}" if m.get("yes_price") is not None else "?"
                lines.append(f"  - {m['question']} (Yes: {price_str})")
            lines.append("")

        # Crypto
        crypto = sources.get("crypto", {})
        if crypto and crypto.get("price"):
            lines.append(f"CRYPTO DATA ({crypto['coin']}):")
            lines.append(f"  Price: ${crypto['price']:,.2f}")
            if crypto.get("change_24h") is not None:
                lines.append(f"  24h: {crypto['change_24h']:+.1f}%")
            if crypto.get("change_7d") is not None:
                lines.append(f"  7d: {crypto['change_7d']:+.1f}%")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _extract_search_terms(question: str) -> list[str]:
        """Extract meaningful search terms from a question."""
        stop_words = {
            "will", "the", "be", "to", "in", "of", "a", "an", "is", "by",
            "on", "at", "for", "or", "and", "this", "that", "it", "as",
            "with", "from", "has", "have", "been", "was", "are", "were",
            "do", "does", "did", "not", "but", "if", "than", "more",
            "before", "after", "above", "below", "between", "up", "down",
            "yes", "no", "what", "when", "where", "who", "which", "how",
            "win", "price", "above", "below",
        }
        words = question.replace("?", "").replace(",", "").replace("'", "").split()
        return [w for w in words if w.lower() not in stop_words and len(w) > 2]

    @staticmethod
    def _is_crypto_related(keywords: list[str]) -> bool:
        crypto_terms = {
            "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
            "xrp", "dogecoin", "doge", "blockchain", "defi", "nft",
        }
        return bool(crypto_terms & {k.lower() for k in keywords})
