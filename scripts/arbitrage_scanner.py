"""Internal Polymarket Arbitrage Scanner.

Finds mispricing within Polymarket itself:
1. Multi-outcome markets where all outcome prices should sum to ~$1
   (FIFA World Cup winner, US President, etc.)
   If sum < $1 → buy all outcomes = guaranteed profit
   If sum > $1 → overpriced outcomes = sell opportunity

2. Related binary markets with logical constraints
   (Fed cuts 25bps + Fed cuts 50bps + Fed holds should sum to ~100%)

3. Yes/No complement check
   (Yes price + No price should = $1, if not = free money)

No prediction skill needed — pure structural arbitrage.
"""

import json
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_DIR = "data"


def fetch_all_markets(n_pages: int = 5) -> list[dict]:
    """Fetch active markets."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    all_markets = []
    for offset in range(0, n_pages * 100, 100):
        resp = session.get(f"{GAMMA_API}/markets", params={
            "limit": 100, "active": True, "closed": False,
            "order": "volume24hr", "ascending": False, "offset": offset,
        }, timeout=15)
        batch = resp.json()
        if not batch:
            break
        all_markets.extend(batch)
    return all_markets


def check_yes_no_complement(markets: list[dict]) -> list[dict]:
    """Check if Yes + No prices = $1 for each market.

    On Polymarket, buying Yes + No for the same market should cost $1.
    If Yes + No < $1, buy both = guaranteed profit.
    If Yes + No > $1, something is wrong (rare).
    """
    opportunities = []
    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue

        if not prices or len(prices) < 2:
            continue

        yes_p = float(prices[0])
        no_p = float(prices[1])
        total = yes_p + no_p

        if total <= 0:
            continue

        gap = abs(total - 1.0)
        if gap > 0.01:  # More than 1% mispricing
            profit_pct = (1.0 - total) / total * 100 if total < 1.0 else 0
            opportunities.append({
                "type": "yes_no_complement",
                "question": m.get("question", "")[:70],
                "yes_price": yes_p,
                "no_price": no_p,
                "total": round(total, 4),
                "gap": round(gap, 4),
                "direction": "BUY_BOTH" if total < 1.0 else "OVERPRICED",
                "profit_pct": round(profit_pct, 2),
                "volume": float(m.get("volume", 0) or 0),
                "id": m.get("id"),
            })

    return sorted(opportunities, key=lambda x: -x["gap"])


def find_related_groups(markets: list[dict]) -> dict[str, list[dict]]:
    """Group markets that are logically related (same event, different outcomes).

    Examples:
    - "Will Spain win FIFA WC?" + "Will France win?" + ... → should sum to ~100%
    - "Fed cuts 25bps" + "Fed cuts 50bps" + "Fed holds" → should sum to ~100%
    """
    # Group by common phrases
    groups = defaultdict(list)

    # Known multi-outcome event patterns
    patterns = [
        ("FIFA World Cup", "win the 2026 FIFA World Cup"),
        ("2028 President", "win the 2028 US Presidential"),
        ("2028 Dem Primary", "win the 2028 Democratic"),
        ("2028 GOP Primary", "win the 2028 Republican"),
        ("NCAA Tournament", "win the 2026 NCAA"),
        ("NBA Finals", "win the 2026 NBA Finals"),
        ("NBA MVP", "win the 2025-2026 NBA MVP"),
        ("Hungary PM", "Prime Minister of Hungary"),
        ("Fed April", "after the April 2026 meeting"),
        ("Fed June", "after the June 2026 meeting"),
    ]

    for m in markets:
        q = m.get("question", "")
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

        for group_name, pattern in patterns:
            if pattern.lower() in q.lower():
                groups[group_name].append({
                    "question": q[:70],
                    "yes_price": yes_p,
                    "volume": float(m.get("volume", 0) or 0),
                    "id": m.get("id"),
                })
                break

    return dict(groups)


def analyze_group(name: str, markets: list[dict]) -> dict:
    """Analyze a group of related markets for arbitrage."""
    if len(markets) < 2:
        return {
            "group": name,
            "n_markets": len(markets),
            "total_yes_price": round(sum(m["yes_price"] for m in markets), 4),
            "overpriced_by": 0.0,
            "total_volume": sum(m["volume"] for m in markets),
            "markets": markets,
            "arbitrage": False,
            "strategy": "NONE",
            "expected_profit_pct": 0,
        }

    total_yes = sum(m["yes_price"] for m in markets)
    total_volume = sum(m["volume"] for m in markets)

    # For mutually exclusive outcomes, all Yes prices should sum to ~1.0
    overpriced = total_yes - 1.0

    result = {
        "group": name,
        "n_markets": len(markets),
        "total_yes_price": round(total_yes, 4),
        "overpriced_by": round(overpriced, 4),
        "total_volume": total_volume,
        "markets": sorted(markets, key=lambda x: -x["yes_price"]),
    }

    if abs(overpriced) > 0.02:  # More than 2% mispricing
        result["arbitrage"] = True
        if overpriced > 0:
            # Total > 1.0: market is overpriced
            # Strategy: sell (BUY_NO on) the most overpriced outcomes
            result["strategy"] = "SELL_OVERPRICED"
            result["expected_profit_pct"] = round(overpriced / total_yes * 100, 2)
        else:
            # Total < 1.0: market is underpriced
            # Strategy: buy all outcomes for less than $1, guaranteed $1 payout
            result["strategy"] = "BUY_ALL"
            result["expected_profit_pct"] = round(abs(overpriced) / total_yes * 100, 2)
    else:
        result["arbitrage"] = False
        result["strategy"] = "NONE"
        result["expected_profit_pct"] = 0

    return result


def check_fed_arbitrage(markets: list[dict]) -> list[dict]:
    """Special check for Fed rate decision markets.

    Fed outcomes are mutually exclusive:
    - Cut 50+ bps
    - Cut 25 bps
    - No change
    - Hike 25+ bps
    These MUST sum to ~100%.
    """
    fed_groups = defaultdict(list)

    for m in markets:
        q = (m.get("question") or "").lower()
        if "fed" not in q or ("interest rate" not in q and "bps" not in q):
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

        # Group by meeting date
        if "april" in q:
            meeting = "April 2026"
        elif "june" in q:
            meeting = "June 2026"
        elif "july" in q:
            meeting = "July 2026"
        else:
            meeting = "Unknown"

        fed_groups[meeting].append({
            "question": m.get("question", "")[:70],
            "yes_price": yes_p,
            "volume": float(m.get("volume", 0) or 0),
            "id": m.get("id"),
        })

    results = []
    for meeting, group in fed_groups.items():
        if len(group) >= 2:
            analysis = analyze_group(f"Fed {meeting}", group)
            results.append(analysis)

    return results


def main():
    print("=" * 70)
    print("  POLYMARKET INTERNAL ARBITRAGE SCANNER")
    print("=" * 70)

    print("\n  Fetching all active markets...")
    markets = fetch_all_markets(n_pages=5)
    print(f"  Loaded {len(markets)} markets")

    # 1. Yes/No complement check
    print(f"\n--- CHECK 1: Yes/No Complement (Yes + No should = $1) ---")
    complement_opps = check_yes_no_complement(markets)
    if complement_opps:
        for opp in complement_opps[:10]:
            print(f"  {opp['question']}")
            print(f"    Yes={opp['yes_price']:.3f} No={opp['no_price']:.3f} "
                  f"Total={opp['total']:.4f} Gap={opp['gap']:.4f} → {opp['direction']}")
    else:
        print("  No complement arbitrage found (market is efficient here)")

    # 2. Multi-outcome groups
    print(f"\n--- CHECK 2: Multi-Outcome Groups (all should sum to ~$1) ---")
    groups = find_related_groups(markets)

    arb_found = []
    for name, group_markets in sorted(groups.items(), key=lambda x: -len(x[1])):
        analysis = analyze_group(name, group_markets)

        print(f"\n  {name} ({analysis['n_markets']} outcomes):")
        print(f"    Total Yes price: {analysis['total_yes_price']:.4f} "
              f"(should be ~1.0, off by {analysis['overpriced_by']:+.4f})")

        for m in analysis.get("markets", [])[:8]:
            print(f"      {m['yes_price']:.3f} | {m['question']}")

        if analysis["n_markets"] > 8:
            print(f"      ... and {analysis['n_markets'] - 8} more")

        if analysis["arbitrage"]:
            print(f"    >>> ARBITRAGE: {analysis['strategy']} — "
                  f"expected profit {analysis['expected_profit_pct']:.1f}%")
            arb_found.append(analysis)
        else:
            print(f"    No arbitrage (within 2% of fair)")

    # 3. Fed rate specific
    print(f"\n--- CHECK 3: Fed Rate Decision Arbitrage ---")
    fed_results = check_fed_arbitrage(markets)
    for fr in fed_results:
        print(f"\n  {fr['group']} ({fr['n_markets']} outcomes):")
        print(f"    Total: {fr['total_yes_price']:.4f} (off by {fr['overpriced_by']:+.4f})")
        for m in fr.get("markets", []):
            print(f"      {m['yes_price']:.3f} | {m['question']}")
        if fr["arbitrage"]:
            print(f"    >>> ARBITRAGE: {fr['strategy']} — "
                  f"expected profit {fr['expected_profit_pct']:.1f}%")
            arb_found.append(fr)

    # Summary
    print(f"\n{'='*70}")
    print(f"  ARBITRAGE SUMMARY")
    print(f"{'='*70}")
    print(f"  Groups scanned: {len(groups) + len(fed_results)}")
    print(f"  Arbitrage opportunities: {len(arb_found)}")

    if arb_found:
        arb_found.sort(key=lambda x: -x["expected_profit_pct"])
        print(f"\n  OPPORTUNITIES (by profit %):")
        for a in arb_found:
            print(f"    {a['group']}: {a['strategy']} "
                  f"({a['expected_profit_pct']:.1f}% profit, "
                  f"{a['n_markets']} outcomes, "
                  f"sum={a['total_yes_price']:.4f})")
    else:
        print("  No arbitrage found — market is currently efficient")
        print("  Run again later, mispricings appear during high-activity periods")

    # Save
    out_path = os.path.join(DATA_DIR, "arbitrage_scan.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complement_opportunities": complement_opps[:20],
            "group_analyses": [analyze_group(n, g) for n, g in groups.items()],
            "fed_analyses": fed_results,
            "arbitrage_found": arb_found,
        }, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
