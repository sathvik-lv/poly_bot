"""Rolling Backtest — Periodic incremental accuracy tracker.

Instead of batch-testing 5000 markets at once, this fetches a small batch
of recently-resolved markets each cycle, runs them through the prediction
engine, and accumulates results in a rolling accuracy file.

Run every 6 hours (via daily_cycle.py or GitHub Actions) to get continuous
feedback on model performance as new markets resolve.

How it works:
1. Load already-tested market IDs from rolling_accuracy.json
2. Fetch recently-resolved markets from Gamma API (skip already-tested)
3. Run each through microstructure + external_data + orderbook + ensemble
4. Append results to rolling_accuracy.json
5. Print rolling accuracy report (Brier, per-model, per-category, P&L)

Usage:
    python scripts/rolling_backtest.py              # fetch + test + report
    python scripts/rolling_backtest.py --batch 100  # test up to 100 new markets
    python scripts/rolling_backtest.py --report     # report only (no new fetching)
    python scripts/rolling_backtest.py --reset      # clear history and start fresh
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv()

from src.market_client import MarketClient
from src.prediction_engine import (
    MicrostructureModel,
    ExternalDataModel,
    OrderbookModel,
    EnsemblePredictor,
)

DATA_DIR = "data"
ROLLING_FILE = os.path.join(DATA_DIR, "rolling_accuracy.json")
GAMMA_API = "https://gamma-api.polymarket.com"

# =========================================================
# Category Classification
# =========================================================

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
                     "counter-strike", "league of legends", "dota", "esports", "map ",
                     "round ", "bo3", "bo5"]),
    ("tech_ai",     ["openai", "chatgpt", "artificial intelligence", " ai ", "google",
                     "apple", "tesla"]),
    ("weather",     ["temperature", "weather", "rain", "snow", "hurricane"]),
    ("tweets",      ["tweet", "post on x", "elon musk post", "posts from"]),
]


def classify_market(question: str) -> str:
    q = question.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in q:
                return category
    return "other"


# =========================================================
# Rolling State Management
# =========================================================

def load_rolling() -> dict:
    """Load accumulated rolling backtest results."""
    if not os.path.exists(ROLLING_FILE):
        return {"tested_ids": [], "results": [], "last_run": None, "total_cycles": 0}
    try:
        with open(ROLLING_FILE) as f:
            data = json.load(f)
        # Ensure all fields exist
        data.setdefault("tested_ids", [])
        data.setdefault("results", [])
        data.setdefault("last_run", None)
        data.setdefault("total_cycles", 0)
        return data
    except (json.JSONDecodeError, KeyError):
        return {"tested_ids": [], "results": [], "last_run": None, "total_cycles": 0}


def save_rolling(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ROLLING_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =========================================================
# Fetch Recently-Resolved Markets (incremental)
# =========================================================

def fetch_new_resolved(already_tested: set, batch_limit: int = 100) -> list:
    """Fetch resolved markets we haven't tested yet.

    Scans recent closed markets ordered by volume. Stops after finding
    batch_limit new markets or checking 2000 API results (whichever first).
    """
    print(f"  Fetching new resolved markets (already tested: {len(already_tested)})...")
    new_markets = []
    offset = 0
    api_batch = 100
    max_checked = max(batch_limit * 20, 2000)  # scan deep enough to fill the batch
    checked = 0
    skip_reasons = defaultdict(int)

    while len(new_markets) < batch_limit and checked < max_checked:
        try:
            resp = requests.get(f"{GAMMA_API}/markets", params={
                "closed": "true",
                "limit": api_batch,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }, timeout=20)

            if resp.status_code != 200:
                time.sleep(2)
                offset += api_batch
                checked += api_batch
                continue

            batch = resp.json()
            if not batch:
                break

            for m in batch:
                checked += 1
                mid = m.get("id") or m.get("condition_id", "")

                # Skip already tested
                if mid in already_tested:
                    skip_reasons["already_tested"] += 1
                    continue

                # Parse outcome prices
                prices_raw = m.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    try:
                        prices = json.loads(prices_raw)
                    except (json.JSONDecodeError, TypeError):
                        skip_reasons["bad_prices"] += 1
                        continue
                else:
                    prices = prices_raw

                try:
                    float_prices = [float(p) for p in prices]
                except (ValueError, TypeError):
                    skip_reasons["unparseable_prices"] += 1
                    continue

                if not float_prices or max(float_prices) < 0.95:
                    skip_reasons["not_clearly_resolved"] += 1
                    continue

                if len(float_prices) != 2:
                    skip_reasons["not_binary"] += 1
                    continue

                # Parse outcomes
                outcomes_raw = m.get("outcomes", "[]")
                if isinstance(outcomes_raw, str):
                    try:
                        outcomes = json.loads(outcomes_raw)
                    except (json.JSONDecodeError, TypeError):
                        skip_reasons["bad_outcomes"] += 1
                        continue
                else:
                    outcomes = outcomes_raw

                if len(outcomes) < 2:
                    skip_reasons["too_few_outcomes"] += 1
                    continue

                winning_idx = float_prices.index(max(float_prices))
                actual_outcome = 1.0 if winning_idx == 0 else 0.0

                # Get pre-resolution price
                ltp = m.get("lastTradePrice")
                if not ltp:
                    skip_reasons["no_last_trade_price"] += 1
                    continue

                try:
                    pre_price = float(ltp)
                except (ValueError, TypeError):
                    skip_reasons["bad_ltp"] += 1
                    continue

                if pre_price < 0.05 or pre_price > 0.95:
                    skip_reasons["extreme_price"] += 1
                    continue

                parsed = MarketClient.parse_market(m)
                parsed["outcome_prices"] = {"Yes": pre_price, "No": 1.0 - pre_price}
                parsed["last_trade_price"] = pre_price

                new_markets.append({
                    "id": mid,
                    "parsed": parsed,
                    "pre_price": pre_price,
                    "actual_outcome": actual_outcome,
                    "question": m.get("question", ""),
                    "category": classify_market(m.get("question", "")),
                    "volume": float(m.get("volume", 0) or 0),
                    "token_ids": m.get("clobTokenIds", []),
                })

                if len(new_markets) >= batch_limit:
                    break

            offset += api_batch
            time.sleep(0.05)

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            time.sleep(2)
            offset += api_batch
            checked += api_batch

    print(f"  Found {len(new_markets)} new markets (checked {checked})")
    if skip_reasons:
        top = sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]
        for reason, count in top:
            print(f"    {reason}: {count}")
    return new_markets


# =========================================================
# Run Models on New Markets
# =========================================================

def test_batch(markets: list) -> list:
    """Run prediction models against a batch of resolved markets."""
    micro = MicrostructureModel()
    external = ExternalDataModel(backtest_mode=True)
    orderbook = OrderbookModel()
    ensemble = EnsemblePredictor()

    results = []
    t0 = time.time()

    for i, market in enumerate(markets):
        parsed = market["parsed"]
        actual = market["actual_outcome"]
        pre_price = market["pre_price"]
        question = market["question"]

        sub_results = {}

        # Microstructure
        try:
            micro_r = micro.analyze(parsed)
            if micro_r.get("estimate") is not None:
                sub_results["microstructure"] = micro_r
        except Exception:
            pass

        # External data
        try:
            keywords = [w for w in question.lower().split() if len(w) > 3][:5]
            ext_r = external.analyze(parsed, keywords)
            if ext_r.get("estimate") is not None:
                sub_results["external_data"] = ext_r
        except Exception:
            pass

        # Orderbook
        token_ids = market.get("token_ids", [])
        if token_ids:
            try:
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                ob_r = orderbook.analyze(token_ids[0], market_price=pre_price)
                if ob_r.get("estimate") is not None:
                    sub_results["orderbook"] = ob_r
            except Exception:
                pass

        # Ensemble
        estimates, variances, model_names = [], [], []
        for name, result in sub_results.items():
            if result.get("estimate") is not None and result.get("variance") is not None:
                estimates.append(result["estimate"])
                variances.append(result["variance"])
                model_names.append(name)

        if estimates:
            ens_r = ensemble.combine(estimates, variances)
            ens_prob = max(0.01, min(0.99, ens_r["probability"]))
        else:
            ens_prob = pre_price

        edge = ens_prob - pre_price

        result = {
            "id": market["id"],
            "question": question[:80],
            "category": market["category"],
            "pre_price": round(pre_price, 4),
            "ensemble_pred": round(ens_prob, 4),
            "actual": actual,
            "edge": round(edge, 4),
            "models_used": model_names,
            "model_estimates": {
                n: round(sub_results[n]["estimate"], 4) for n in model_names
            },
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    {i+1}/{len(markets)} ({rate:.0f}/sec)")

    elapsed = time.time() - t0
    print(f"  Tested {len(results)} markets in {elapsed:.1f}s")
    return results


# =========================================================
# Accuracy Report (rolling cumulative)
# =========================================================

def report(data: dict):
    """Print rolling accuracy report from accumulated results."""
    results = data.get("results", [])
    n = len(results)

    print(f"\n{'='*70}")
    print(f"  ROLLING BACKTEST ACCURACY")
    print(f"  {n} markets tested | {data.get('total_cycles', 0)} cycles")
    print(f"  Last run: {data.get('last_run', 'never')}")
    print(f"{'='*70}")

    if n < 5:
        print(f"  Need at least 5 tested markets for metrics. Run without --report.")
        return

    preds = np.array([r["ensemble_pred"] for r in results])
    actuals = np.array([r["actual"] for r in results])
    prices = np.array([r["pre_price"] for r in results])
    edges = np.array([r["edge"] for r in results])

    brier_ours = float(np.mean((preds - actuals) ** 2))
    brier_mkt = float(np.mean((prices - actuals) ** 2))

    print(f"\n  BRIER SCORES (lower = better, 0.25 = coin flip):")
    print(f"    Our ensemble: {brier_ours:.4f}")
    print(f"    Market price: {brier_mkt:.4f}")
    delta = brier_mkt - brier_ours
    tag = "BETTER" if delta > 0 else "WORSE"
    print(f"    Delta:        {delta:+.4f} ({tag} than market)")

    # --- Per-model breakdown ---
    model_data = defaultdict(lambda: {"est": [], "out": [], "mp": []})
    for r in results:
        for name, est in r.get("model_estimates", {}).items():
            model_data[name]["est"].append(est)
            model_data[name]["out"].append(r["actual"])
            model_data[name]["mp"].append(r["pre_price"])

    if model_data:
        print(f"\n  PER-MODEL ACCURACY:")
        print(f"    {'Model':<20} {'N':>5} {'Brier':>7} {'vs Mkt':>8} {'Verdict':>8}")
        print(f"    {'-'*20} {'-'*5} {'-'*7} {'-'*8} {'-'*8}")

        for name in sorted(model_data.keys()):
            d = model_data[name]
            ests = np.array(d["est"])
            outs = np.array(d["out"])
            mps = np.array(d["mp"])
            m_brier = float(np.mean((ests - outs) ** 2))
            mkt_brier = float(np.mean((mps - outs) ** 2))
            diff = m_brier - mkt_brier
            verdict = "GOOD" if diff < -0.005 else "NEUTRAL" if diff < 0.005 else "BAD"
            print(f"    {name:<20} {len(d['est']):>5} {m_brier:>7.4f} {diff:>+8.4f} {verdict:>8}")

    # --- Per-category breakdown ---
    cat_data = defaultdict(lambda: {"ens": [], "out": [], "mp": []})
    for r in results:
        cat = r.get("category", "other")
        cat_data[cat]["ens"].append(r["ensemble_pred"])
        cat_data[cat]["out"].append(r["actual"])
        cat_data[cat]["mp"].append(r["pre_price"])

    if cat_data:
        print(f"\n  PER-CATEGORY BREAKDOWN:")
        print(f"    {'Category':<15} {'N':>5} {'Our Brier':>10} {'Mkt Brier':>10} {'Beat?':>6}")
        print(f"    {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*6}")

        for cat in sorted(cat_data.keys()):
            d = cat_data[cat]
            ens = np.array(d["ens"])
            outs = np.array(d["out"])
            mps = np.array(d["mp"])
            our_b = float(np.mean((ens - outs) ** 2))
            mkt_b = float(np.mean((mps - outs) ** 2))
            beat = "YES" if our_b < mkt_b - 0.001 else "no"
            print(f"    {cat:<15} {len(d['ens']):>5} {our_b:>10.4f} {mkt_b:>10.4f} {beat:>6}")

    # --- Calibration ---
    print(f"\n  CALIBRATION:")
    print(f"    {'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'N':>6}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")

    for lo in range(0, 100, 10):
        hi = lo + 10
        mask = (preds >= lo / 100) & (preds < hi / 100)
        if mask.sum() == 0:
            continue
        avg_pred = float(preds[mask].mean())
        avg_actual = float(actuals[mask].mean())
        gap = avg_pred - avg_actual
        flag = " !!!" if abs(gap) > 0.15 else ""
        print(f"    {lo}-{hi}%{'':<6} {avg_pred:>10.3f} {avg_actual:>10.3f} {gap:>8.3f} {mask.sum():>6}{flag}")

    # --- P&L simulation ---
    for threshold in [0.02, 0.05, 0.10]:
        trades, wins, pnl = 0, 0, 0.0
        for r in results:
            e = r["edge"]
            a = r["actual"]
            mp = r["pre_price"]
            if abs(e) < threshold:
                continue
            trades += 1
            if e > 0:
                profit = (1 - mp) if a == 1 else -mp
            else:
                profit = mp if a == 0 else -(1 - mp)
            pnl += profit
            if profit > 0:
                wins += 1

        if trades > 0:
            print(f"\n  P&L (edge > {threshold:.0%}, $1/trade):")
            print(f"    Trades: {trades}  Win: {wins/trades:.1%}  "
                  f"Total: ${pnl:+.2f}  Avg: ${pnl/trades:+.4f}")

    # --- Trend: last 5 cycles ---
    cycle_markers = set()
    for r in results:
        tested = r.get("tested_at", "")
        if tested:
            cycle_markers.add(tested[:13])  # group by date+hour

    if len(cycle_markers) > 1:
        sorted_cycles = sorted(cycle_markers)
        print(f"\n  CYCLES: {len(sorted_cycles)} data collection points")
        # Show growth of sample
        cumulative = 0
        for cycle_key in sorted_cycles[-10:]:
            batch_n = sum(1 for r in results if r.get("tested_at", "")[:13] == cycle_key)
            cumulative += batch_n
            print(f"    {cycle_key}: +{batch_n} markets (cumulative: {cumulative})")

    print(f"\n{'='*70}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Rolling Backtest - Periodic accuracy tracker")
    parser.add_argument("--batch", type=int, default=75,
                        help="Max new markets to test per cycle (default 75)")
    parser.add_argument("--report", action="store_true",
                        help="Show report only, no new testing")
    parser.add_argument("--reset", action="store_true",
                        help="Clear history and start fresh")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.reset:
        save_rolling({"tested_ids": [], "results": [], "last_run": None, "total_cycles": 0})
        print("  Rolling backtest history cleared.")
        return

    data = load_rolling()

    if args.report:
        report(data)
        return

    # --- Fetch & test new markets ---
    print(f"\n{'='*70}")
    print(f"  ROLLING BACKTEST CYCLE")
    print(f"  Previously tested: {len(data['tested_ids'])} markets")
    print(f"  Fetching up to {args.batch} new resolved markets...")
    print(f"{'='*70}")

    already_tested = set(data["tested_ids"])
    new_markets = fetch_new_resolved(already_tested, batch_limit=args.batch)

    if not new_markets:
        print("  No new resolved markets found this cycle.")
        report(data)
        return

    # Run models
    print(f"\n  Running models on {len(new_markets)} new markets...")
    new_results = test_batch(new_markets)

    # Append to rolling state
    for r in new_results:
        data["tested_ids"].append(r["id"])
        data["results"].append(r)

    data["last_run"] = datetime.now(timezone.utc).isoformat()
    data["total_cycles"] = data.get("total_cycles", 0) + 1

    save_rolling(data)
    print(f"\n  Saved: {len(data['results'])} total results ({len(new_results)} new this cycle)")

    # Report
    report(data)


if __name__ == "__main__":
    main()
