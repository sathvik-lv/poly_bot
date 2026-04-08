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

# Category classification
CATEGORY_RULES = [
    ("fed_rate",    ["fed ", "interest rate", "fomc", "federal reserve", "bps"]),
    ("crypto",      ["bitcoin", "btc ", "ethereum", "eth ", "crypto", "solana"]),
    ("oil_energy",  ["oil", "crude", "wti ", "brent", "opec", "energy"]),
    ("macro",       ["gdp", "inflation", "cpi ", "recession", "unemployment", "tariff"]),
    ("geopolitics", ["invade", "invasion", "war ", "military", "regime", "sanctions",
                     "iran", "taiwan", "ukraine", "russia"]),
    ("elections",   ["election", "president", "governor", "senate", "nominee", "primary",
                     "democrat", "republican"]),
    ("sports",      ["fifa", "world cup", "nba ", "nfl ", "mlb ", "f1 ", "champion",
                     "tournament", "lakers", "celtics", "warriors", "yankees", "dodgers",
                     "match", "winner", "beat", "vs.", "points", "goals", "ufc", "nhl ",
                     "counter-strike", "league of legends", "dota", "esports"]),
    ("tech_ai",     ["openai", "chatgpt", "artificial intelligence", " ai ", "google",
                     "apple", "tesla"]),
]


def classify_market(question: str) -> str:
    q = question.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in q:
                return category
    return "other"


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
            question_text = market.get("question", "")
            record = {
                "pred_id": f"live_{int(time.time()*1000)}_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_id": market.get("id"),
                "question": question_text,
                "category": classify_market(question_text),
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
    """Generate accuracy report — per-model, per-category, per-market detail."""
    records = load_tracked()
    resolved = [r for r in records if r.get("resolved")]
    unresolved = [r for r in records if not r.get("resolved")]

    print(f"\n{'='*70}")
    print(f"  LIVE PREDICTION ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"  Total predictions:  {len(records)}")
    print(f"  Resolved:           {len(resolved)}")
    print(f"  Unresolved:         {len(unresolved)}")

    if not resolved:
        print(f"\n  No resolved predictions yet.")
        if unresolved:
            print(f"\n  PENDING ({len(unresolved)}):")
            for r in unresolved[:15]:
                q = r.get("question", "")[:50]
                cat = r.get("category", classify_market(r.get("question", "")))
                models = ",".join(r.get("model_names", []))
                print(f"    [{cat:<12}] {q}")
                print(f"      Price={r['market_price']:.3f} Pred={r['predicted_prob']:.3f} Models={models}")
        return

    # === ENSEMBLE METRICS ===
    preds = np.array([r["predicted_prob"] for r in resolved])
    outcomes = np.array([r["outcome"] for r in resolved])
    prices = np.array([r["market_price"] for r in resolved])

    brier = float(np.mean((preds - outcomes) ** 2))
    market_brier = float(np.mean((prices - outcomes) ** 2))

    print(f"\n  ENSEMBLE:")
    print(f"    Brier (ours): {brier:.4f}  Brier (mkt): {market_brier:.4f}  "
          f"Delta: {market_brier - brier:+.4f} {'BETTER' if brier < market_brier else 'WORSE'}")

    # ================================================================
    # PER-MODEL, PER-MARKET DETAIL (the full ledger)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PER-MODEL TRADE LEDGER")
    print(f"{'='*70}")

    # Collect all model names
    all_models = set()
    for r in resolved:
        for name in r.get("sub_model_estimates", {}).keys():
            all_models.add(name)

    for model in sorted(all_models):
        print(f"\n  --- {model.upper()} ---")
        print(f"  {'Market':<40} {'Cat':<12} {'Mkt':>5} {'Est':>5} {'Edge':>6} {'Out':>4} {'P&L':>7} {'Result':>6}")
        print(f"  {'-'*40} {'-'*12} {'-'*5} {'-'*5} {'-'*6} {'-'*4} {'-'*7} {'-'*6}")

        model_pnl = 0.0
        model_trades = 0
        model_wins = 0
        cat_pnl = {}  # category -> total P&L for this model

        for r in resolved:
            info = r.get("sub_model_estimates", {}).get(model)
            if info is None:
                continue

            est = info.get("estimate") if isinstance(info, dict) else info
            if est is None:
                continue

            mp = r["market_price"]
            outcome = r["outcome"]
            cat = r.get("category", classify_market(r.get("question", "")))
            q = r.get("question", "")[:40]

            edge = est - mp
            # Simulate $1 trade if model had edge > 1%
            if abs(edge) >= 0.01:
                model_trades += 1
                if edge > 0:  # model says BUY YES
                    pnl = (1 - mp) if outcome == 1 else -mp
                else:  # model says BUY NO
                    pnl = mp if outcome == 0 else -(1 - mp)
                result = "WIN" if pnl > 0 else "LOSS"
                if pnl > 0:
                    model_wins += 1
            else:
                pnl = 0.0
                result = "SKIP"

            model_pnl += pnl
            cat_pnl[cat] = cat_pnl.get(cat, 0) + pnl

            out_str = "YES" if outcome == 1 else "NO"
            print(f"  {q:<40} {cat:<12} {mp:>5.3f} {est:>5.3f} {edge:>+6.3f} {out_str:>4} ${pnl:>+6.2f} {result:>6}")

        # Model summary
        win_rate = model_wins / model_trades if model_trades > 0 else 0
        print(f"  {'':40} {'':12} {'':5} {'':5} {'':6} {'':4} {'-------':>7}")
        print(f"  {'TOTAL':40} {'':12} {'':5} {'':5} {'':6} {'':4} ${model_pnl:>+6.2f}  "
              f"{model_trades} trades, {win_rate:.0%} win")

        # Per-category subtotals for this model
        if cat_pnl:
            print(f"\n  By category:")
            for cat in sorted(cat_pnl.keys()):
                tag = "+" if cat_pnl[cat] > 0 else ""
                print(f"    {cat:<15} ${cat_pnl[cat]:>+.2f}")

    # ================================================================
    # PER-MODEL SUMMARY TABLE
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'N':>4} {'Brier':>7} {'MktBrier':>8} {'Beat?':>6} {'Trades':>6} {'WinR':>5} {'P&L':>8}")
    print(f"  {'-'*20} {'-'*4} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*5} {'-'*8}")

    for model in sorted(all_models):
        ests, outs, mps = [], [], []
        for r in resolved:
            info = r.get("sub_model_estimates", {}).get(model)
            if info is None:
                continue
            est = info.get("estimate") if isinstance(info, dict) else info
            if est is None:
                continue
            ests.append(est)
            outs.append(r["outcome"])
            mps.append(r["market_price"])

        if not ests:
            continue

        ests_a = np.array(ests)
        outs_a = np.array(outs)
        mps_a = np.array(mps)

        m_brier = float(np.mean((ests_a - outs_a) ** 2))
        mkt_brier = float(np.mean((mps_a - outs_a) ** 2))
        beat = "YES" if m_brier < mkt_brier - 0.001 else "no"

        edges = ests_a - mps_a
        trades, wins, pnl = 0, 0, 0.0
        for e, o, mp in zip(edges, outs_a, mps_a):
            if abs(e) < 0.01:
                continue
            trades += 1
            if e > 0:
                p = (1 - mp) if o == 1 else -mp
            else:
                p = mp if o == 0 else -(1 - mp)
            pnl += p
            if p > 0:
                wins += 1
        wr = f"{wins/trades:.0%}" if trades > 0 else "---"

        print(f"  {model:<20} {len(ests):>4} {m_brier:>7.4f} {mkt_brier:>8.4f} {beat:>6} {trades:>6} {wr:>5} ${pnl:>+7.2f}")

    # ================================================================
    # PER-CATEGORY SUMMARY
    # ================================================================
    from collections import defaultdict
    cat_results = defaultdict(lambda: {"ens": [], "out": [], "mp": []})
    for r in resolved:
        cat = r.get("category", classify_market(r.get("question", "")))
        cat_results[cat]["ens"].append(r["predicted_prob"])
        cat_results[cat]["out"].append(r["outcome"])
        cat_results[cat]["mp"].append(r["market_price"])

    if cat_results:
        print(f"\n  PER-CATEGORY:")
        print(f"  {'Category':<15} {'N':>4} {'Our Brier':>10} {'Mkt Brier':>10} {'Beat?':>6}")
        print(f"  {'-'*15} {'-'*4} {'-'*10} {'-'*10} {'-'*6}")
        for cat in sorted(cat_results.keys()):
            d = cat_results[cat]
            ens = np.array(d["ens"])
            outs = np.array(d["out"])
            mps = np.array(d["mp"])
            our_b = float(np.mean((ens - outs) ** 2))
            mkt_b = float(np.mean((mps - outs) ** 2))
            beat = "YES" if our_b < mkt_b - 0.001 else "no"
            print(f"  {cat:<15} {len(d['ens']):>4} {our_b:>10.4f} {mkt_b:>10.4f} {beat:>6}")

    # Show pending
    if unresolved:
        print(f"\n  PENDING ({len(unresolved)} markets waiting to resolve):")
        for r in unresolved[:10]:
            q = r.get("question", "")[:45]
            cat = r.get("category", classify_market(r.get("question", "")))
            models = ",".join(r.get("model_names", []))
            print(f"    [{cat:<12}] {q}  Models={models}")


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
