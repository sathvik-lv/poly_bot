"""Cross-Platform Odds Comparison — Find mispricing between platforms.

Compares prediction market odds across:
1. Polymarket (primary)
2. Manifold Markets (free public API, play money but smart crowd)
3. Metaculus (needs API key, best calibrated forecasters)
4. PredictIt (needs account, real money US markets)

When the same event is priced differently across platforms,
that's a mispricing signal — one platform is wrong.

Usage:
    python scripts/cross_platform.py              # full comparison
    python scripts/cross_platform.py --manifold   # Manifold only
    python scripts/cross_platform.py --metaculus   # Metaculus only
"""

import json
import os
import sys
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests

DATA_DIR = "data"
GAMMA_API = "https://gamma-api.polymarket.com"


# =========================================================
# Platform Fetchers
# =========================================================

def fetch_polymarket(n: int = 200) -> list[dict]:
    """Fetch active Polymarket markets."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    all_markets = []
    for offset in [0, 100]:
        resp = session.get(f"{GAMMA_API}/markets", params={
            "limit": 100, "active": True, "closed": False,
            "order": "volume24hr", "ascending": False, "offset": offset,
        }, timeout=15)
        batch = resp.json()
        if not batch:
            break
        all_markets.extend(batch)

    results = []
    for m in all_markets[:n]:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        if not prices:
            continue
        results.append({
            "question": m.get("question", ""),
            "probability": float(prices[0]),
            "volume": float(m.get("volume", 0) or 0),
            "platform": "polymarket",
            "id": m.get("id"),
        })
    return results


def fetch_manifold(search_terms: list[str] = None) -> list[dict]:
    """Fetch Manifold Markets data (free public API)."""
    if search_terms is None:
        search_terms = [
            "fed rate", "interest rate", "inflation", "bitcoin",
            "trump", "president 2028", "recession", "oil",
            "world cup", "iran", "china taiwan", "ukraine",
            "AI", "OpenAI", "election", "GDP", "unemployment",
        ]

    results = []
    seen_ids = set()

    for term in search_terms:
        try:
            resp = requests.get("https://api.manifold.markets/v0/search-markets", params={
                "term": term, "limit": 20, "sort": "liquidity",
                "filter": "open",
            }, timeout=15)
            if resp.status_code != 200:
                continue

            for m in resp.json():
                mid = m.get("id")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)

                prob = m.get("probability")
                if prob is None or prob <= 0 or prob >= 1:
                    continue

                results.append({
                    "question": m.get("question", ""),
                    "probability": prob,
                    "volume": m.get("volume", 0) or 0,
                    "liquidity": m.get("totalLiquidity", 0) or 0,
                    "platform": "manifold",
                    "id": mid,
                    "url": m.get("url", ""),
                })
        except Exception:
            continue

    return results


def fetch_metaculus() -> list[dict]:
    """Fetch Metaculus predictions via their v2 Posts API.

    Requires API token (set METACULUS_API_TOKEN in .env).
    API docs: https://www.metaculus.com/api/posts/
    """
    token = os.environ.get("METACULUS_API_TOKEN")
    if not token:
        print("    (no METACULUS_API_TOKEN set — skipping)")
        return []

    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {token}",
    }

    results = []

    # Search terms to find markets that overlap with Polymarket
    search_terms = [
        "iran", "china taiwan", "trump", "fed interest rate",
        "bitcoin", "oil", "recession", "ukraine russia",
        "world cup", "AI", "election 2028", "inflation",
        "nuclear", "GDP", "tariff",
    ]

    seen_ids = set()

    n_checked = 0
    n_restricted = 0

    for term in search_terms:
        try:
            resp = requests.get("https://www.metaculus.com/api/posts/", params={
                "search": term,
                "limit": 20,
                "forecast_type": "binary",
                "statuses": "open",
                "order_by": "-activity",
            }, headers=headers, timeout=15)

            if resp.status_code == 403:
                print(f"    Metaculus 403 — token may lack access")
                return results
            if resp.status_code != 200:
                continue

            data = resp.json()
            posts = data.get("results", [])

            for post in posts:
                post_id = post.get("id")
                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                # Extract question from post
                question_data = post.get("question")
                if not question_data:
                    # Could be a group — check for sub-questions
                    group = post.get("group_of_questions", {})
                    for sub_q in group.get("questions", []):
                        n_checked += 1
                        before = len(results)
                        _extract_metaculus_question(sub_q, post, results, seen_ids)
                        if len(results) == before:
                            n_restricted += 1
                    continue

                n_checked += 1
                before = len(results)
                _extract_metaculus_question(question_data, post, results, seen_ids)
                if len(results) == before:
                    n_restricted += 1

        except Exception:
            continue

    if n_checked > 0 and len(results) == 0:
        print(f"    Metaculus: {n_checked} questions found but {n_restricted} have restricted predictions")
        print(f"    -> Email api-requests@metaculus.com to request 'Bot Benchmarking Access Tier'")

    return results


def _extract_metaculus_question(question_data: dict, post: dict,
                                results: list, seen_ids: set):
    """Extract a single Metaculus question into our format."""
    q_type = question_data.get("type", "")
    if q_type != "binary":
        return

    q_id = question_data.get("id")
    if q_id in seen_ids:
        return
    seen_ids.add(q_id)

    # Get community prediction
    aggregations = question_data.get("aggregations", {})
    recency = aggregations.get("recency_weighted", {})
    latest = recency.get("latest")
    if not latest:
        # Try community prediction
        community = aggregations.get("metaculus_prediction", {})
        latest = community.get("latest")
    if not latest:
        return

    # Binary questions: centers[0] is the probability
    centers = latest.get("centers", [])
    if not centers:
        return

    prob = centers[0]
    if prob is None or prob <= 0 or prob >= 1:
        return

    title = post.get("title", "") or question_data.get("title", "")
    post_id = post.get("id")

    results.append({
        "question": title,
        "probability": float(prob),
        "n_forecasters": question_data.get("nr_forecasters", 0),
        "platform": "metaculus",
        "id": q_id,
        "url": f"https://www.metaculus.com/questions/{post_id}/",
    })


def fetch_predictit() -> list[dict]:
    """Fetch PredictIt markets (free public API)."""
    results = []
    try:
        resp = requests.get("https://www.predictit.org/api/marketdata/all/", timeout=15)
        if resp.status_code != 200:
            return results

        for market in resp.json().get("markets", []):
            for contract in market.get("contracts", []):
                last_price = contract.get("lastTradePrice")
                if last_price is None or last_price <= 0 or last_price >= 1:
                    continue
                results.append({
                    "question": f"{market.get('name', '')}: {contract.get('name', '')}",
                    "probability": float(last_price),
                    "volume": contract.get("totalSharesTraded", 0) or 0,
                    "platform": "predictit",
                    "id": contract.get("id"),
                })
    except Exception:
        pass

    return results


# =========================================================
# Matching Engine
# =========================================================

STOP_WORDS = {"will", "the", "be", "by", "in", "on", "of", "a", "an", "to", "and",
               "or", "is", "it", "at", "for", "this", "that", "with", "from", "before",
               "after", "has", "have", "been", "does", "do", "what", "when", "where"}

# Generic event words — shared across many unrelated markets
GENERIC_WORDS = {"win", "presidential", "election", "president", "price", "above",
                 "below", "hit", "reach", "market", "world", "cup", "nomination",
                 "nominee", "candidate", "rate", "rates", "interest", "change"}


def normalize_question(q: str) -> str:
    """Normalize question text for matching."""
    q = q.lower().strip()
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()


def extract_keywords(q: str) -> set[str]:
    """Extract meaningful keywords from a question."""
    words = normalize_question(q).split()
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def extract_specific_keywords(q: str) -> set[str]:
    """Extract specific/distinguishing keywords (names, countries, years, numbers).

    These are the words that actually distinguish one market from another.
    "Trump 2028 presidential election" vs "Newsom 2028 presidential election"
    differ on Trump/Newsom, not on presidential/election.
    """
    words = normalize_question(q).split()
    specific = set()
    for w in words:
        if w in STOP_WORDS or len(w) <= 2:
            continue
        if w in GENERIC_WORDS:
            continue
        specific.add(w)
    return specific


def similarity(q1: str, q2: str) -> float:
    """Compute similarity requiring shared specific keywords.

    Markets about the same event must share specific identifiers
    (names, countries, dates), not just generic event words.
    """
    kw1 = extract_keywords(q1)
    kw2 = extract_keywords(q2)
    sp1 = extract_specific_keywords(q1)
    sp2 = extract_specific_keywords(q2)

    if not kw1 or not kw2:
        return 0.0

    # Must share at least 2 SPECIFIC keywords (not just "presidential election")
    specific_overlap = sp1 & sp2
    if len(specific_overlap) < 2:
        return 0.0

    # Full keyword overlap
    overlap = kw1 & kw2
    jaccard = len(overlap) / len(kw1 | kw2)

    # Sequence similarity
    n1 = normalize_question(q1)
    n2 = normalize_question(q2)
    seq_score = SequenceMatcher(None, n1, n2).ratio()

    # Combined: 40% keyword overlap + 60% sequence match
    return 0.4 * jaccard + 0.6 * seq_score


def find_matches(poly_markets: list[dict], other_markets: list[dict],
                 threshold: float = 0.55) -> list[dict]:
    """Find matching markets across platforms.

    Uses strict matching: requires shared specific keywords + high similarity.
    Each other-platform market can only match ONE Polymarket market (best match wins).
    """
    # First pass: find best Poly match for each other-platform market
    # This prevents 20 different FIFA WC country markets all matching one Manifold market
    candidates = []

    for pm in poly_markets:
        for om in other_markets:
            score = similarity(pm["question"], om["question"])
            if score >= threshold:
                candidates.append((score, pm, om))

    # Sort by score descending
    candidates.sort(key=lambda x: -x[0])

    # Deduplicate: each other-market ID matched to at most one poly market (best score)
    used_other = set()
    used_poly = set()
    matches = []

    for score, pm, om in candidates:
        other_key = (om["platform"], om.get("id"))
        poly_key = pm.get("id")

        if other_key in used_other or poly_key in used_poly:
            continue
        used_other.add(other_key)
        used_poly.add(poly_key)

        poly_prob = pm["probability"]
        other_prob = om["probability"]
        gap = poly_prob - other_prob

        matches.append({
            "polymarket_question": pm["question"][:80],
            "other_question": om["question"][:80],
            "polymarket_prob": round(poly_prob, 4),
            "other_prob": round(other_prob, 4),
            "gap": round(gap, 4),
            "abs_gap": round(abs(gap), 4),
            "similarity": round(score, 3),
            "other_platform": om["platform"],
            "poly_volume": pm.get("volume", 0),
            "other_volume": om.get("volume", 0),
            "poly_id": pm.get("id"),
            "other_id": om.get("id"),
            "other_url": om.get("url", ""),
            "signal": "POLY_HIGH" if gap > 0.05 else "POLY_LOW" if gap < -0.05 else "ALIGNED",
        })

    return sorted(matches, key=lambda x: -x["abs_gap"])


# =========================================================
# Main
# =========================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Platform Odds Comparison")
    parser.add_argument("--manifold", action="store_true", help="Manifold only")
    parser.add_argument("--metaculus", action="store_true", help="Metaculus only")
    parser.add_argument("--predictit", action="store_true", help="PredictIt only")
    args = parser.parse_args()

    print("=" * 70)
    print("  CROSS-PLATFORM ODDS COMPARISON")
    print("=" * 70)

    # Fetch Polymarket
    print("\n  Fetching Polymarket...")
    poly = fetch_polymarket()
    print(f"  Polymarket: {len(poly)} markets")

    all_matches = []

    # Manifold
    if not args.metaculus and not args.predictit:
        print("\n  Fetching Manifold Markets...")
        manifold = fetch_manifold()
        print(f"  Manifold: {len(manifold)} markets")
        if manifold:
            matches = find_matches(poly, manifold)
            all_matches.extend(matches)
            mismatch = [m for m in matches if m["abs_gap"] > 0.05]
            print(f"  Matched: {len(matches)} | Mispriced (>5%): {len(mismatch)}")

    # Metaculus
    if not args.manifold and not args.predictit:
        print("\n  Fetching Metaculus...")
        metaculus = fetch_metaculus()
        print(f"  Metaculus: {len(metaculus)} markets")
        if metaculus:
            matches = find_matches(poly, metaculus)
            all_matches.extend(matches)
            mismatch = [m for m in matches if m["abs_gap"] > 0.05]
            print(f"  Matched: {len(matches)} | Mispriced (>5%): {len(mismatch)}")

    # PredictIt
    if not args.manifold and not args.metaculus:
        print("\n  Fetching PredictIt...")
        predictit = fetch_predictit()
        print(f"  PredictIt: {len(predictit)} markets")
        if predictit:
            matches = find_matches(poly, predictit)
            all_matches.extend(matches)
            mismatch = [m for m in matches if m["abs_gap"] > 0.05]
            print(f"  Matched: {len(matches)} | Mispriced (>5%): {len(mismatch)}")

    # Results
    all_matches.sort(key=lambda x: -x["abs_gap"])

    print(f"\n{'='*70}")
    print(f"  CROSS-PLATFORM COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"  Total matched markets: {len(all_matches)}")

    mispricings = [m for m in all_matches if m["abs_gap"] > 0.05]
    aligned = [m for m in all_matches if m["abs_gap"] <= 0.05]

    print(f"  Mispriced (>5% gap): {len(mispricings)}")
    print(f"  Aligned (<5% gap): {len(aligned)}")

    if mispricings:
        print(f"\n  MISPRICINGS (Polymarket vs other platforms):")
        for m in mispricings[:15]:
            direction = "POLY HIGHER" if m["gap"] > 0 else "POLY LOWER"
            print(f"\n    Poly: {m['polymarket_question']}")
            print(f"    {m['other_platform'].upper()}: {m['other_question']}")
            print(f"    Poly={m['polymarket_prob']:.3f} vs {m['other_platform']}={m['other_prob']:.3f} "
                  f"Gap={m['gap']:+.3f} ({direction}) "
                  f"[match={m['similarity']:.0%}]")

    if aligned:
        print(f"\n  ALIGNED MARKETS (confirming efficient pricing):")
        for m in aligned[:10]:
            print(f"    {m['polymarket_question'][:50]}")
            print(f"      Poly={m['polymarket_prob']:.3f} {m['other_platform']}={m['other_prob']:.3f} "
                  f"Gap={m['gap']:+.3f}")

    # Save
    out_path = os.path.join(DATA_DIR, "cross_platform.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_poly": len(poly),
            "n_matches": len(all_matches),
            "n_mispricings": len(mispricings),
            "matches": all_matches,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
