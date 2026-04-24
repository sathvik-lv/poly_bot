"""Live market scanner — runs ALL models on real active markets.

This is the honest test: real prices, real orderbooks, real data sources.
No backtest tricks.
"""

import json
import os
import sys
import time
import math
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests

from src.market_client import MarketClient
from src.prediction_engine import PredictionEngine


def parse_token_ids(raw_market: dict) -> list[str]:
    """Extract CLOB token IDs from raw market data."""
    clob = raw_market.get("clobTokenIds", "[]")
    if isinstance(clob, str):
        try:
            return json.loads(clob)
        except json.JSONDecodeError:
            return []
    return clob if isinstance(clob, list) else []


def compute_time_remaining(end_date_str: str) -> float:
    """Fraction of time remaining (0=expired, 1=just opened)."""
    if not end_date_str:
        return 0.5
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        start = now - (end - now)  # rough estimate
        total = (end - start).total_seconds()
        remaining = (end - now).total_seconds()
        if total <= 0:
            return 0.01
        return max(0.01, min(1.0, remaining / total))
    except Exception:
        return 0.5


def main():
    print("=" * 70)
    print("  LIVE MARKET SCANNER — ALL MODELS ACTIVE")
    print("=" * 70)

    # Live engine — NOT backtest mode
    engine = PredictionEngine(backtest_mode=False)
    client = MarketClient()

    # Check which models are active
    has_ai = engine.ai_model.enabled
    print(f"\n  Models:")
    print(f"    Microstructure:  ACTIVE")
    print(f"    External Data:   ACTIVE (25 sources)")
    print(f"    Time Series:     ACTIVE (if price history available)")
    print(f"    AI Semantic:     {'ACTIVE (' + engine.ai_model.model + ')' if has_ai else 'DISABLED (set OPENROUTER_API_KEY)'}")
    print(f"    Orderbook:       ACTIVE (CLOB)")

    # Fetch active markets sorted by 24h volume
    print(f"\n  Fetching top active markets by volume...")
    raw_markets = client.session.get(
        f"{client.base_url}/markets",
        params={"limit": 100, "active": True, "closed": False,
                "order": "volume24hr", "ascending": False},
        timeout=15,
    ).json()

    # Filter: must have reasonable price (not already resolved)
    candidates = []
    for raw in raw_markets:
        parsed = MarketClient.parse_market(raw)
        price = parsed.get("outcome_prices", {}).get("Yes")
        if price is None:
            continue
        if price < 0.05 or price > 0.95:
            continue  # Already near-resolved
        if not parsed.get("active"):
            continue

        token_ids = parse_token_ids(raw)
        parsed["_token_id"] = token_ids[0] if token_ids else None
        parsed["_raw"] = raw
        candidates.append(parsed)

    print(f"  Found {len(candidates)} markets with tradeable prices (0.05-0.95)")
    print(f"  Scanning top 30...\n")

    results = []
    for i, market in enumerate(candidates[:30]):
        t0 = time.time()
        try:
            token_id = market.get("_token_id")
            time_rem = compute_time_remaining(market.get("end_date"))

            prediction = engine.predict(
                market_data=market,
                time_remaining_frac=time_rem,
                token_id=token_id,
            )

            edge = prediction["edge"]["edge"]
            prob = prediction["prediction"]["probability"]
            price = prediction["market"]["current_price"]
            confidence = prediction["edge"]["edge_confidence"]
            n_models = prediction["ensemble"]["n_models"]
            model_names = prediction["ensemble"]["model_names"]

            # Strategy evaluation
            strat = prediction.get("strategy", {})
            should_enter = strat.get("should_enter", None)
            regime = strat.get("regime", "?")

            elapsed = time.time() - t0
            q = market.get("question", "")[:55]

            if abs(edge) >= 0.02:
                marker = " <<<" if should_enter else ""
                print(f"  [{i+1:>2}] {q}")
                print(f"       Price={price:.3f} Pred={prob:.3f} Edge={edge:+.3f} [{confidence}] "
                      f"Models={n_models}({','.join(model_names)}) "
                      f"Enter={should_enter} {elapsed:.1f}s{marker}")
                results.append(prediction)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i+1:>2}] ERROR: {str(e)[:60]} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SCAN RESULTS")
    print(f"{'='*70}")
    print(f"  Markets scanned: {min(30, len(candidates))}")
    print(f"  With edge >= 2%: {len(results)}")

    if results:
        edges = [abs(r["edge"]["edge"]) for r in results]
        models_used = set()
        for r in results:
            models_used.update(r["ensemble"]["model_names"])

        enter_count = sum(1 for r in results if r.get("strategy", {}).get("should_enter"))
        print(f"  Strategy says ENTER: {enter_count}/{len(results)}")
        print(f"  Models active: {', '.join(sorted(models_used))}")
        print(f"  Avg edge: {np.mean(edges):.3f}")
        print(f"  Max edge: {np.max(edges):.3f}")

        # Top 5 opportunities
        results.sort(key=lambda x: abs(x["edge"]["edge"]), reverse=True)
        print(f"\n  TOP OPPORTUNITIES:")
        for j, r in enumerate(results[:5]):
            q = r["market"]["question"][:50]
            e = r["edge"]["edge"]
            p = r["prediction"]["probability"]
            mp = r["market"]["current_price"]
            action = r["sizing"]["action"]
            kelly = r["sizing"]["kelly_fraction"]
            enter = r.get("strategy", {}).get("should_enter", "?")
            print(f"    {j+1}. {q}")
            print(f"       {action} | Price={mp:.3f} Pred={p:.3f} Edge={e:+.3f} Kelly={kelly:.1%} Enter={enter}")

    # Save results
    out_path = "data/live_scan_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_scanned": min(30, len(candidates)),
            "n_with_edge": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
