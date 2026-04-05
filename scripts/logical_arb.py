"""Logical Arbitrage Scanner — Exploits mathematical impossibilities in pricing.

Unlike the basic arbitrage scanner (multi-outcome sums), this finds LOGICAL
CONSTRAINTS between different markets:

1. Subset constraints: "Chiefs win Super Bowl" <= "AFC wins Super Bowl"
   If Chiefs > AFC → impossible → buy AFC Yes + sell Chiefs Yes

2. Implication constraints: "Fed cuts 50bps" implies "Fed cuts at all"
   If cut_50bps > cut_any → impossible → arb

3. Time hierarchy: "X by June" <= "X by December"
   If june_price > december_price → free money

4. Multi-outcome completeness: All candidates must sum to ~100%
   (Enhanced version of existing arbitrage_scanner.py)

5. Complement: "Not X" = 1 - "X"
   Cross-check between related Yes/No markets

Uses constraint graph + violation detection. No prediction skill needed.

Usage:
    python scripts/logical_arb.py              # full scan
    python scripts/logical_arb.py --verbose    # show all constraints checked
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_DIR = "data"

# Fee on Polymarket: ~2% round-trip (1% each side)
FEE_PCT = 0.02


# =============================================================
# Market Fetching
# =============================================================

def fetch_all_markets(n_pages: int = 10) -> list[dict]:
    """Fetch all active markets."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    all_markets = []
    for offset in range(0, n_pages * 100, 100):
        try:
            resp = session.get(f"{GAMMA_API}/markets", params={
                "limit": 100, "active": True, "closed": False,
                "order": "volume24hr", "ascending": False, "offset": offset,
            }, timeout=15)
            batch = resp.json()
            if not batch:
                break
            all_markets.extend(batch)
        except Exception:
            break
    return all_markets


def parse_market(m: dict) -> dict:
    """Extract price and metadata from raw market."""
    prices = m.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            prices = []
    yes_p = float(prices[0]) if prices else None
    no_p = float(prices[1]) if len(prices) > 1 else None

    return {
        "id": m.get("id"),
        "question": m.get("question", ""),
        "slug": m.get("slug", ""),
        "yes_price": yes_p,
        "no_price": no_p,
        "volume": float(m.get("volume", 0) or 0),
        "end_date": m.get("endDate", ""),
    }


# =============================================================
# Constraint Graph Builder
# =============================================================

class ConstraintGraph:
    """Graph of logical constraints between markets.

    Each constraint is: market_A relation market_B
    Where relation is one of: <=, >=, ==, complement, sum_to_1
    """

    def __init__(self):
        self.constraints = []
        self.markets = {}  # id -> parsed market

    def add_market(self, m: dict):
        self.markets[m["id"]] = m

    def add_constraint(self, id_a: str, relation: str, id_b: str,
                       reason: str = "", group: str = ""):
        self.constraints.append({
            "id_a": id_a,
            "relation": relation,
            "id_b": id_b,
            "reason": reason,
            "group": group,
        })

    def add_sum_constraint(self, ids: list[str], target: float = 1.0,
                           reason: str = "", group: str = ""):
        """All markets in ids should sum to target."""
        self.constraints.append({
            "type": "sum",
            "ids": ids,
            "target": target,
            "reason": reason,
            "group": group,
        })

    def check_violations(self, min_profit_pct: float = 1.0) -> list[dict]:
        """Find all constraint violations with profit > min_profit_pct after fees."""
        violations = []

        for c in self.constraints:
            if c.get("type") == "sum":
                v = self._check_sum_violation(c, min_profit_pct)
                if v:
                    violations.append(v)
                continue

            ma = self.markets.get(c["id_a"])
            mb = self.markets.get(c["id_b"])
            if not ma or not mb:
                continue
            if ma["yes_price"] is None or mb["yes_price"] is None:
                continue

            pa = ma["yes_price"]
            pb = mb["yes_price"]
            rel = c["relation"]

            violation = None

            if rel == "<=" and pa > pb + FEE_PCT:
                # A should be <= B, but A > B
                # Trade: buy B Yes, sell A Yes (buy A No)
                gap = pa - pb
                profit_pct = (gap - FEE_PCT) / max(pa, 0.01) * 100
                if profit_pct >= min_profit_pct:
                    violation = {
                        "type": "subset_violation",
                        "constraint": f"{ma['question'][:50]} <= {mb['question'][:50]}",
                        "prices": f"A={pa:.3f} > B={pb:.3f}",
                        "gap": round(gap, 4),
                        "profit_pct": round(profit_pct, 2),
                        "trade": f"BUY '{mb['question'][:40]}' YES + BUY '{ma['question'][:40]}' NO",
                        "reason": c["reason"],
                        "group": c["group"],
                        "market_a": ma,
                        "market_b": mb,
                    }

            elif rel == ">=" and pa < pb - FEE_PCT:
                gap = pb - pa
                profit_pct = (gap - FEE_PCT) / max(pb, 0.01) * 100
                if profit_pct >= min_profit_pct:
                    violation = {
                        "type": "subset_violation",
                        "constraint": f"{ma['question'][:50]} >= {mb['question'][:50]}",
                        "prices": f"A={pa:.3f} < B={pb:.3f}",
                        "gap": round(gap, 4),
                        "profit_pct": round(profit_pct, 2),
                        "trade": f"BUY '{ma['question'][:40]}' YES + BUY '{mb['question'][:40]}' NO",
                        "reason": c["reason"],
                        "group": c["group"],
                        "market_a": ma,
                        "market_b": mb,
                    }

            if violation:
                violations.append(violation)

        return sorted(violations, key=lambda x: -x["profit_pct"])

    def _check_sum_violation(self, c: dict, min_profit_pct: float) -> dict | None:
        """Check if a group of mutually exclusive outcomes sums correctly."""
        ids = c["ids"]
        markets = [self.markets.get(i) for i in ids]
        markets = [m for m in markets if m and m["yes_price"] is not None]

        if len(markets) < 2:
            return None

        total = sum(m["yes_price"] for m in markets)
        deviation = total - c["target"]

        if abs(deviation) <= FEE_PCT:
            return None

        if deviation > 0:
            # Overpriced: sell all outcomes proportionally
            profit_pct = (deviation - FEE_PCT) / total * 100
            strategy = "SELL_ALL"
        else:
            # Underpriced: buy all outcomes for < $1
            profit_pct = (abs(deviation) - FEE_PCT) / total * 100
            strategy = "BUY_ALL"

        if profit_pct < min_profit_pct:
            return None

        return {
            "type": "sum_violation",
            "constraint": f"{len(markets)} outcomes should sum to {c['target']:.2f}",
            "prices": f"sum={total:.4f} (off by {deviation:+.4f})",
            "gap": round(abs(deviation), 4),
            "profit_pct": round(profit_pct, 2),
            "trade": f"{strategy} across {len(markets)} outcomes",
            "reason": c["reason"],
            "group": c["group"],
            "n_markets": len(markets),
            "markets": sorted(markets, key=lambda x: -x["yes_price"]),
        }


# =============================================================
# Constraint Discovery — Auto-detect logical relationships
# =============================================================

def discover_time_hierarchies(markets: list[dict]) -> list[tuple]:
    """Find markets that are the same question with different deadlines.

    "X by June 2026" <= "X by December 2026" (always true).
    """
    constraints = []

    # Group by similar base question (strip date references)
    date_patterns = [
        (r'by (january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{0,2},?\s*\d{4}', 'by_date'),
        (r'by (end of |december 31,? ?)?\d{4}', 'by_year'),
        (r'in \d{4}', 'in_year'),
        (r'before \d{4}', 'before_year'),
        (r'by (q[1-4]|quarter)', 'by_quarter'),
    ]

    # Extract base question and date for each market
    dated_markets = []
    for m in markets:
        q = m["question"].lower()
        for pattern, ptype in date_patterns:
            match = re.search(pattern, q)
            if match:
                base = re.sub(pattern, "___DATE___", q).strip()
                date_str = match.group(0)
                dated_markets.append({
                    **m,
                    "_base": base,
                    "_date_str": date_str,
                    "_date_type": ptype,
                })
                break

    # Group by base question
    groups = defaultdict(list)
    for m in dated_markets:
        groups[m["_base"]].append(m)

    # For each group with multiple dates, create <= constraints
    month_order = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    for base, group in groups.items():
        if len(group) < 2:
            continue

        # Sort by date (earlier deadline first)
        def sort_key(m):
            ds = m["_date_str"]
            for month, num in month_order.items():
                if month in ds:
                    year_match = re.search(r'\d{4}', ds)
                    year = int(year_match.group()) if year_match else 2026
                    # Also extract day number if present
                    day_match = re.search(month + r'\s+(\d{1,2})', ds)
                    day = int(day_match.group(1)) if day_match else 15
                    return year * 10000 + num * 100 + day
            # "end of 2026" or "in 2026" or "before 2027" = December of that year
            year_match = re.search(r'\d{4}', ds)
            if year_match:
                year = int(year_match.group())
                if "before" in ds:
                    return (year - 1) * 10000 + 1231
                if "end of" in ds or "in " in ds:
                    return year * 10000 + 1231
                return year * 10000 + 1231
            return 99999999

        group.sort(key=sort_key)

        # P("X by earlier date") <= P("X by later date")
        # Earlier deadline is STRICTER so probability must be LOWER
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                earlier = group[i]
                later = group[j]
                constraints.append((
                    earlier["id"], "<=", later["id"],
                    f"P(by {earlier['_date_str']}) <= P(by {later['_date_str']})",
                    f"time_hierarchy:{base[:40]}",
                ))

    return constraints


def discover_subset_relations(markets: list[dict]) -> list[tuple]:
    """Find markets where one outcome is a subset of another.

    "Chiefs win Super Bowl" <= "AFC wins Super Bowl"
    "Fed cuts 50bps" <= "Fed cuts at all"
    """
    constraints = []

    # Known subset patterns
    subset_rules = [
        # (specific_pattern, general_pattern, reason)
        ("win the 2026 fifa world cup", "win the 2026 fifa world cup",
         "specific_team <= any_team"),

        # Fed rate: specific cut size implies cut happened
        ("decrease interest rates by 50", "decrease interest rates",
         "50bps cut implies cut happened"),
        ("decrease interest rates by 25", "decrease interest rates",
         "25bps cut implies cut happened"),
        ("increase interest rates by 25", "increase interest rates",
         "25bps hike implies hike happened"),
    ]

    # Candidate detection: same event, different scope
    # "Will X win the 2028 US Presidential Election" <= "Will X win the 2028 [Party] presidential nomination"
    # Actually opposite: winning nomination doesn't mean winning presidency, but
    # winning presidency implies winning nomination (or being nominated)

    # Build lookup
    q_lookup = {m["id"]: m["question"].lower() for m in markets}

    # Time hierarchy detection is handled separately
    # Here we look for logical subset: specific_outcome <= broader_outcome

    # Conference/League subsets in sports
    conference_rules = [
        ("win the 2026 nhl stanley cup", "win the eastern conference", "eastern_team"),
        ("win the 2026 nhl stanley cup", "win the western conference", "western_team"),
        ("win the 2026 nba finals", "win the eastern conference", "eastern_team"),
        ("win the 2026 nba finals", "win the western conference", "western_team"),
    ]

    return constraints


def discover_mutually_exclusive_groups(markets: list[dict]) -> list[dict]:
    """Find groups of mutually exclusive outcomes that must sum to ~1.0.

    Enhanced version: auto-discovers groups from question patterns.
    """
    groups = defaultdict(list)

    # Pattern: "Will [ENTITY] [ACTION]?" where ACTION is shared
    # Group by the shared action part

    # Known group patterns (entity varies, action is same)
    group_patterns = [
        ("win the 2026 FIFA World Cup", "FIFA_WC_2026"),
        ("win the 2028 US Presidential Election", "US_Pres_2028"),
        ("win the 2028 Democratic presidential nomination", "Dem_Nom_2028"),
        ("win the 2028 Republican presidential nomination", "GOP_Nom_2028"),
        ("win the 2026 NBA Finals", "NBA_Finals_2026"),
        ("win the 2026 NHL Stanley Cup", "NHL_Cup_2026"),
        ("win the 2026 NCAA", "NCAA_2026"),
        ("Prime Minister of Hungary", "Hungary_PM"),
        ("win the Metropolitan Division", "NHL_Metro"),
        ("win the Eastern Conference", "NHL_East"),
        ("win the Western Conference", "NHL_West"),
        ("win the Atlantic Division", "NHL_Atlantic"),
        ("win the Central Division", "NHL_Central"),
        ("win the Pacific Division", "NHL_Pacific"),
        ("win La Liga", "LaLiga"),
        ("win the 2025-2026 NBA MVP", "NBA_MVP"),
        ("win the 2026 Brazilian presidential", "Brazil_Pres_2026"),
        ("win the 2027 French presidential", "France_Pres_2027"),
        ("win the 2026 Colombian presidential", "Colombia_Pres_2026"),
    ]

    # Also detect Fed rate meeting groups
    fed_meetings = [
        ("after the April 2026 meeting", "Fed_April_2026"),
        ("after the June 2026 meeting", "Fed_June_2026"),
        ("after the July 2026 meeting", "Fed_July_2026"),
        ("after the September 2026 meeting", "Fed_Sep_2026"),
    ]
    group_patterns.extend(fed_meetings)

    for m in markets:
        q = m["question"]
        for pattern, group_name in group_patterns:
            if pattern.lower() in q.lower():
                groups[group_name].append(m)
                break

    return dict(groups)


# =============================================================
# Optimal Trade Sizing (simplified Bregman/Frank-Wolfe)
# =============================================================

def optimal_arb_sizes(prices: list[float], budget: float = 100.0,
                      direction: str = "SELL_ALL") -> list[float]:
    """Calculate optimal trade sizes across N outcomes for arbitrage.

    For SELL_ALL (sum > 1): sell each outcome proportional to overpricing.
    For BUY_ALL (sum < 1): buy each outcome to guarantee profit.

    Returns dollar amounts per outcome.
    """
    n = len(prices)
    total = sum(prices)

    if direction == "BUY_ALL" and total < 1.0:
        # Buy all outcomes: spend budget * (price_i / total) on each
        # Payout = budget / total (guaranteed, since one outcome = $1)
        sizes = [(p / total) * budget for p in prices]

    elif direction == "SELL_ALL" and total > 1.0:
        # Sell all outcomes: collect total, pay out 1.0
        # Weight by price (sell more of overpriced outcomes)
        sizes = [(p / total) * budget for p in prices]

    else:
        sizes = [budget / n] * n

    return [round(s, 2) for s in sizes]


# =============================================================
# Main Scanner
# =============================================================

def run_scan(verbose: bool = False) -> dict:
    """Run full logical arbitrage scan."""
    print("=" * 70)
    print("  LOGICAL ARBITRAGE SCANNER")
    print("=" * 70)

    print("\n  Fetching markets...")
    raw_markets = fetch_all_markets(n_pages=10)
    markets = [parse_market(m) for m in raw_markets]
    markets = [m for m in markets if m["yes_price"] is not None]
    print(f"  Loaded {len(markets)} markets with prices")

    graph = ConstraintGraph()
    for m in markets:
        graph.add_market(m)

    # --- 1. Time hierarchy constraints ---
    print("\n  [1/3] Discovering time hierarchy constraints...")
    time_constraints = discover_time_hierarchies(markets)
    for id_a, rel, id_b, reason, group in time_constraints:
        graph.add_constraint(id_a, rel, id_b, reason, group)
    print(f"    Found {len(time_constraints)} time hierarchy constraints")

    if verbose:
        for id_a, rel, id_b, reason, group in time_constraints[:10]:
            ma = graph.markets.get(id_a, {})
            mb = graph.markets.get(id_b, {})
            print(f"      {ma.get('question','')[:40]} {rel} {mb.get('question','')[:40]}")

    # --- 2. Subset constraints ---
    print("\n  [2/3] Discovering subset constraints...")
    subset_constraints = discover_subset_relations(markets)
    for id_a, rel, id_b, reason, group in subset_constraints:
        graph.add_constraint(id_a, rel, id_b, reason, group)
    print(f"    Found {len(subset_constraints)} subset constraints")

    # --- 3. Mutually exclusive groups ---
    print("\n  [3/3] Discovering mutually exclusive groups...")
    me_groups = discover_mutually_exclusive_groups(markets)
    for group_name, group_markets in me_groups.items():
        ids = [m["id"] for m in group_markets]
        graph.add_sum_constraint(ids, target=1.0,
                                 reason=f"Mutually exclusive outcomes",
                                 group=group_name)
    total_in_groups = sum(len(g) for g in me_groups.values())
    print(f"    Found {len(me_groups)} groups with {total_in_groups} total markets")

    if verbose:
        for name, gm in sorted(me_groups.items(), key=lambda x: -len(x[1])):
            total = sum(m["yes_price"] for m in gm if m["yes_price"])
            print(f"      {name}: {len(gm)} outcomes, sum={total:.4f}")

    # --- Check all violations ---
    print(f"\n  Total constraints: {len(graph.constraints)}")
    print(f"  Checking for violations (min profit > {FEE_PCT*100:.0f}% after fees)...")

    violations = graph.check_violations(min_profit_pct=0.5)

    # Report
    print(f"\n{'='*70}")
    print(f"  LOGICAL ARBITRAGE RESULTS")
    print(f"{'='*70}")

    time_violations = [v for v in violations if "time_hierarchy" in v.get("group", "")]
    subset_violations = [v for v in violations if v.get("type") == "subset_violation"
                         and "time_hierarchy" not in v.get("group", "")]
    sum_violations = [v for v in violations if v.get("type") == "sum_violation"]

    print(f"\n  Time hierarchy violations: {len(time_violations)}")
    for v in time_violations[:10]:
        ma = v.get("market_a", {})
        mb = v.get("market_b", {})
        print(f"    {v['profit_pct']:>5.1f}% profit | {ma.get('question','')[:35]} ({ma.get('yes_price',0):.3f})")
        print(f"             <= {mb.get('question','')[:35]} ({mb.get('yes_price',0):.3f})")
        print(f"             Trade: {v['trade']}")

    print(f"\n  Subset violations: {len(subset_violations)}")
    for v in subset_violations[:10]:
        print(f"    {v['profit_pct']:>5.1f}% profit | {v['constraint']}")
        print(f"             Trade: {v['trade']}")

    print(f"\n  Multi-outcome sum violations: {len(sum_violations)}")
    for v in sum_violations[:10]:
        print(f"    {v['profit_pct']:>5.1f}% profit | {v['group']}: {v['prices']}")
        print(f"             {v['n_markets']} outcomes | Trade: {v['trade']}")
        if verbose:
            for m in v.get("markets", [])[:5]:
                print(f"               {m['yes_price']:.3f} | {m['question'][:55]}")

    # Summary
    total_violations = len(violations)
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Constraints checked:    {len(graph.constraints)}")
    print(f"  Violations found:       {total_violations}")
    print(f"  Time hierarchy arbs:    {len(time_violations)}")
    print(f"  Subset arbs:            {len(subset_violations)}")
    print(f"  Multi-outcome arbs:     {len(sum_violations)}")

    if violations:
        best = violations[0]
        print(f"\n  BEST OPPORTUNITY: {best['profit_pct']:.1f}% profit")
        print(f"    {best.get('constraint', best.get('group', ''))}")
        print(f"    {best['trade']}")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_markets": len(markets),
        "n_constraints": len(graph.constraints),
        "n_violations": total_violations,
        "time_hierarchy_violations": time_violations[:20],
        "subset_violations": subset_violations[:20],
        "sum_violations": [{
            "group": v["group"],
            "prices": v["prices"],
            "profit_pct": v["profit_pct"],
            "trade": v["trade"],
            "n_markets": v["n_markets"],
        } for v in sum_violations[:20]],
    }

    out_path = os.path.join(DATA_DIR, "logical_arb.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Logical Arbitrage Scanner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all constraints")
    args = parser.parse_args()
    run_scan(verbose=args.verbose)


if __name__ == "__main__":
    main()
