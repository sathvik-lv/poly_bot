"""Live Prediction Tracker — Records predictions, checks resolutions, measures real edge.

This is the honest scorecard. It:
1. Scans active markets and records predictions with per-model breakdown
2. Checks previously predicted markets for resolution
3. Computes real accuracy metrics: Brier, per-model edge hit, ROI
4. Saves everything persistently so accuracy builds over time

Run periodically (e.g. every 6-12 hours) to build up a track record.

Usage:
    python scripts/live_tracker.py              # scan + resolve + report
    python scripts/live_tracker.py --scan       # scan new markets only
    python scripts/live_tracker.py --resolve    # check resolutions only
    python scripts/live_tracker.py --report     # show accuracy report only
"""

import argparse
import json
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

DATA_DIR = "data"
TRACKER_FILE = os.path.join(DATA_DIR, "live_predictions.jsonl")
GAMMA_API = "https://gamma-api.polymarket.com"


def parse_token_ids(raw_market: dict) -> list[str]:
    clob = raw_market.get("clobTokenIds", "[]")
    if isinstance(clob, str):
        try:
            return json.loads(clob)
        except json.JSONDecodeError:
            return []
    return clob if isinstance(clob, list) else []


def compute_time_remaining(end_date_str: str) -> float:
    if not end_date_str:
        return 0.5
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        start = now - (end - now)
        total = (end - start).total_seconds()
        remaining = (end - now).total_seconds()
        if total <= 0:
            return 0.01
        return max(0.01, min(1.0, remaining / total))
    except Exception:
        return 0.5


def load_tracked() -> list[dict]:
    if not os.path.exists(TRACKER_FILE):
        return []
    records = []
    with open(TRACKER_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_tracked(records: list[dict]):
    with open(TRACKER_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def append_tracked(record: dict):
    with open(TRACKER_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# =========================================================
# SCAN: Record predictions on active markets
# =========================================================

def scan_markets(n_markets: int = 20):
    """Scan top active markets and record predictions."""
    print(f"\n  SCANNING {n_markets} markets...")
    engine = PredictionEngine(backtest_mode=False)
    client = MarketClient()

    # Get already-predicted market IDs to avoid duplicates
    existing = load_tracked()
    predicted_ids = {r["market_id"] for r in existing if not r.get("resolved")}
    print(f"  Already tracking {len(predicted_ids)} unresolved predictions")

    raw_markets = client.session.get(
        f"{client.base_url}/markets",
        params={"limit": 60, "active": True, "closed": False,
                "order": "volume24hr", "ascending": False},
        timeout=15,
    ).json()

    candidates = []
    for raw in raw_markets:
        parsed = MarketClient.parse_market(raw)
        price = parsed.get("outcome_prices", {}).get("Yes")
        if price is None or price < 0.05 or price > 0.95:
            continue
        if not parsed.get("active"):
            continue
        if parsed.get("id") in predicted_ids:
            continue

        token_ids = parse_token_ids(raw)
        parsed["_token_id"] = token_ids[0] if token_ids else None
        candidates.append(parsed)

    print(f"  Found {len(candidates)} new tradeable markets")
    new_count = 0

    for i, market in enumerate(candidates[:n_markets]):
        try:
            t0 = time.time()
            token_id = market.get("_token_id")
            time_rem = compute_time_remaining(market.get("end_date"))

            prediction = engine.predict(
                market_data=market,
                time_remaining_frac=time_rem,
                token_id=token_id,
            )

            # Build tracking record with full per-model detail
            record = {
                "pred_id": f"live_{int(time.time()*1000)}_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_id": market.get("id"),
                "question": market.get("question", ""),
                "market_price": prediction["market"]["current_price"],
                "predicted_prob": prediction["prediction"]["probability"],
                "edge": prediction["edge"]["edge"],
                "edge_confidence": prediction["edge"]["edge_confidence"],
                "action": prediction["sizing"]["action"],
                "kelly": prediction["sizing"]["kelly_fraction"],
                "n_models": prediction["ensemble"]["n_models"],
                "model_names": prediction["ensemble"]["model_names"],
                "model_weights": prediction["ensemble"]["weights"],
                "sub_model_estimates": {
                    name: {
                        "estimate": result.get("estimate"),
                        "variance": result.get("variance"),
                    }
                    for name, result in prediction.get("sub_models", {}).items()
                    if result.get("estimate") is not None
                },
                "ai_reasoning": prediction.get("sub_models", {}).get(
                    "ai_semantic", {}
                ).get("signals", {}).get("ai_reasoning"),
                "strategy_enter": prediction.get("strategy", {}).get("should_enter"),
                "resolved": False,
                "outcome": None,
            }

            append_tracked(record)
            new_count += 1
            elapsed = time.time() - t0

            q = market.get("question", "")[:50]
            edge = prediction["edge"]["edge"]
            prob = prediction["prediction"]["probability"]
            price = prediction["market"]["current_price"]
            models = ",".join(prediction["ensemble"]["model_names"])
            print(f"  [{i+1:>2}] {q}")
            print(f"       Price={price:.3f} Pred={prob:.3f} Edge={edge:+.3f} Models={models} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1:>2}] ERROR: {str(e)[:60]}")

    print(f"\n  Recorded {new_count} new predictions (total tracking: {len(predicted_ids) + new_count})")


# =========================================================
# RESOLVE: Check outcomes of previously predicted markets
# =========================================================

def resolve_predictions():
    """Check if any tracked markets have resolved."""
    records = load_tracked()
    unresolved = [r for r in records if not r.get("resolved")]

    if not unresolved:
        print("\n  No unresolved predictions to check.")
        return 0

    print(f"\n  Checking {len(unresolved)} unresolved predictions...")
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    resolved_count = 0
    for r in unresolved:
        mid = r.get("market_id")
        if not mid:
            continue
        try:
            resp = session.get(f"{GAMMA_API}/markets/{mid}", timeout=10)
            if resp.status_code != 200:
                continue
            market = resp.json()

            # Check if resolved
            if not market.get("closed"):
                continue

            # Get outcome
            outcome_str = market.get("outcome")
            if outcome_str == "Yes":
                outcome = 1.0
            elif outcome_str == "No":
                outcome = 0.0
            else:
                # Try outcomePrices
                prices = market.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices and len(prices) >= 1:
                    yes_price = float(prices[0])
                    if yes_price > 0.95:
                        outcome = 1.0
                    elif yes_price < 0.05:
                        outcome = 0.0
                    else:
                        continue  # Not clearly resolved
                else:
                    continue

            r["resolved"] = True
            r["outcome"] = outcome
            r["resolved_at"] = datetime.now(timezone.utc).isoformat()
            resolved_count += 1

            edge = r.get("edge", 0)
            correct = (edge > 0 and outcome == 1) or (edge < 0 and outcome == 0)
            q = r.get("question", "")[:50]
            print(f"  RESOLVED: {q}")
            print(f"    Price={r['market_price']:.3f} Pred={r['predicted_prob']:.3f} "
                  f"Edge={edge:+.3f} Outcome={'YES' if outcome==1 else 'NO'} "
                  f"{'CORRECT' if correct else 'WRONG'}")

        except Exception:
            continue

    save_tracked(records)
    print(f"\n  Resolved {resolved_count} predictions")
    return resolved_count


# =========================================================
# REPORT: Accuracy metrics
# =========================================================

def report():
    """Generate accuracy report from tracked predictions."""
    records = load_tracked()
    resolved = [r for r in records if r.get("resolved")]
    unresolved = [r for r in records if not r.get("resolved")]

    print(f"\n{'='*70}")
    print(f"  LIVE PREDICTION ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"  Total predictions:  {len(records)}")
    print(f"  Resolved:           {len(resolved)}")
    print(f"  Unresolved:         {len(unresolved)}")

    if len(resolved) < 3:
        print(f"\n  Need at least 3 resolved predictions for metrics.")
        print(f"  Keep running --scan periodically, markets will resolve over time.")

        # Show unresolved predictions
        if unresolved:
            print(f"\n  PENDING PREDICTIONS:")
            for r in unresolved[:15]:
                q = r.get("question", "")[:50]
                edge = r.get("edge", 0)
                models = ",".join(r.get("model_names", []))
                ai = r.get("ai_reasoning", "")
                print(f"    {q}")
                print(f"      Price={r['market_price']:.3f} Pred={r['predicted_prob']:.3f} "
                      f"Edge={edge:+.3f} Models={models}")
                if ai:
                    print(f"      AI: {ai[:80]}")
        return

    # === ENSEMBLE METRICS ===
    preds = np.array([r["predicted_prob"] for r in resolved])
    outcomes = np.array([r["outcome"] for r in resolved])
    prices = np.array([r["market_price"] for r in resolved])
    edges = np.array([r.get("edge", 0) for r in resolved])

    brier = float(np.mean((preds - outcomes) ** 2))
    market_brier = float(np.mean((prices - outcomes) ** 2))

    # Edge hit rate
    edge_mask = np.abs(edges) > 0.01
    if edge_mask.sum() > 0:
        edge_correct = ((edges[edge_mask] > 0) & (outcomes[edge_mask] == 1)) | \
                       ((edges[edge_mask] < 0) & (outcomes[edge_mask] == 0))
        edge_hit_rate = float(edge_correct.mean())
    else:
        edge_hit_rate = 0

    # ROI
    profit = 0.0
    wagered = 0.0
    for r in resolved:
        e = r.get("edge", 0)
        o = r["outcome"]
        mp = r["market_price"]
        if abs(e) > 0.01:
            if e > 0:
                wagered += mp
                profit += (1 - mp) if o == 1 else -mp
            else:
                wagered += (1 - mp)
                profit += mp if o == 0 else -(1 - mp)

    roi = profit / max(wagered, 0.01)

    print(f"\n  ENSEMBLE ACCURACY:")
    print(f"    Brier Score (ours): {brier:.4f}")
    print(f"    Brier Score (mkt):  {market_brier:.4f}")
    print(f"    Brier Edge:         {market_brier - brier:+.4f} {'(BETTER than market)' if brier < market_brier else '(WORSE than market)'}")
    print(f"    Edge Hit Rate:      {edge_hit_rate:.1%}")
    print(f"    ROI:                {roi:+.1%}")
    print(f"    Profit:             ${profit:+.2f} (on ${wagered:.2f} wagered)")

    # === PER-MODEL BREAKDOWN ===
    model_data = {}
    for r in resolved:
        outcome = r["outcome"]
        mp = r["market_price"]
        for name, info in r.get("sub_model_estimates", {}).items():
            est = info.get("estimate") if isinstance(info, dict) else info
            if est is not None:
                if name not in model_data:
                    model_data[name] = {"est": [], "out": [], "mp": []}
                model_data[name]["est"].append(est)
                model_data[name]["out"].append(outcome)
                model_data[name]["mp"].append(mp)

    if model_data:
        print(f"\n  PER-MODEL ACCURACY:")
        print(f"  {'Model':<20} {'N':>4} {'Brier':>7} {'EdgeHit':>8} {'ROI':>7}")
        print(f"  {'-'*20} {'-'*4} {'-'*7} {'-'*8} {'-'*7}")

        for name in sorted(model_data.keys()):
            d = model_data[name]
            ests = np.array(d["est"])
            outs = np.array(d["out"])
            mps = np.array(d["mp"])

            m_brier = float(np.mean((ests - outs) ** 2))

            m_edges = ests - mps
            m_mask = np.abs(m_edges) > 0.01
            if m_mask.sum() > 0:
                m_correct = ((m_edges[m_mask] > 0) & (outs[m_mask] == 1)) | \
                            ((m_edges[m_mask] < 0) & (outs[m_mask] == 0))
                m_ehr = float(m_correct.mean())
            else:
                m_ehr = 0

            m_profit = 0.0
            m_wagered = 0.0
            for e, o, mp in zip(m_edges, outs, mps):
                if abs(e) > 0.01:
                    if e > 0:
                        m_wagered += mp
                        m_profit += (1 - mp) if o == 1 else -mp
                    else:
                        m_wagered += (1 - mp)
                        m_profit += mp if o == 0 else -(1 - mp)
            m_roi = m_profit / max(m_wagered, 0.01)

            print(f"  {name:<20} {len(d['est']):>4} {m_brier:>7.4f} {m_ehr:>7.1%} {m_roi:>+6.1%}")

    # Show recent resolved
    print(f"\n  RECENT RESOLUTIONS:")
    for r in resolved[-10:]:
        q = r.get("question", "")[:45]
        e = r.get("edge", 0)
        o = r["outcome"]
        correct = (e > 0 and o == 1) or (e < 0 and o == 0)
        tag = "OK" if correct else "XX"
        print(f"    [{tag}] {q} Edge={e:+.3f} -> {'YES' if o==1 else 'NO'}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Live Prediction Tracker")
    parser.add_argument("--scan", action="store_true", help="Scan new markets only")
    parser.add_argument("--resolve", action="store_true", help="Check resolutions only")
    parser.add_argument("--report", action="store_true", help="Show accuracy report only")
    parser.add_argument("-n", type=int, default=20, help="Number of markets to scan")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.scan:
        scan_markets(args.n)
    elif args.resolve:
        resolve_predictions()
    elif args.report:
        report()
    else:
        # Full cycle: scan + resolve + report
        scan_markets(args.n)
        resolve_predictions()
        report()


if __name__ == "__main__":
    main()
