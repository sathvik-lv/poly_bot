"""Honest Backtest — Run prediction models against 5000+ resolved Polymarket markets.

NO manipulation. NO peeking at outcomes. For each resolved market:
1. Use lastTradePrice as pre-resolution market price (filter 0.05-0.95)
2. Run each sub-model independently (microstructure, external, orderbook)
3. Compare prediction vs actual outcome
4. Report REAL per-model Brier scores, per-category edge, P&L simulation

If a model is garbage, this will show it clearly.

Usage:
    python scripts/honest_backtest.py                # full backtest (5000 markets)
    python scripts/honest_backtest.py --quick 500    # quick test
"""

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

# =========================================================
# Category Classification (same as paper_trader)
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
# Fetch Resolved Markets
# =========================================================

GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_resolved_markets(target_count: int = 5000) -> list:
    """Fetch resolved binary markets with honest pre-resolution prices.

    HONEST RULES:
    - Only use lastTradePrice as pre-resolution price proxy
    - Filter out prices < 0.05 or > 0.95 (already basically resolved, no edge to test)
    - Resolution determined by outcomePrices ([1,0] = first won, [0,1] = second won)
    - Skip markets where resolution is ambiguous
    """
    print(f"  Fetching up to {target_count} resolved markets...")
    all_markets = []
    offset = 0
    batch_size = 100
    skipped_reasons = defaultdict(int)

    while len(all_markets) < target_count:
        try:
            resp = requests.get(f"{GAMMA_API}/markets", params={
                "closed": "true",
                "limit": batch_size,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }, timeout=20)

            if resp.status_code != 200:
                print(f"    API error {resp.status_code} at offset {offset}")
                time.sleep(2)
                offset += batch_size
                continue

            batch = resp.json()
            if not batch:
                print(f"    No more markets at offset {offset}")
                break

            for m in batch:
                # --- Parse outcome prices to determine resolution ---
                prices_raw = m.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    try:
                        prices = json.loads(prices_raw)
                    except (json.JSONDecodeError, TypeError):
                        skipped_reasons["bad_prices"] += 1
                        continue
                else:
                    prices = prices_raw

                try:
                    float_prices = [float(p) for p in prices]
                except (ValueError, TypeError):
                    skipped_reasons["unparseable_prices"] += 1
                    continue

                # Must be clearly resolved (one outcome near 1.0)
                if not float_prices or max(float_prices) < 0.95:
                    skipped_reasons["not_clearly_resolved"] += 1
                    continue

                if len(float_prices) != 2:
                    skipped_reasons["not_binary"] += 1
                    continue

                # --- Parse outcomes ---
                outcomes_raw = m.get("outcomes", "[]")
                if isinstance(outcomes_raw, str):
                    try:
                        outcomes = json.loads(outcomes_raw)
                    except (json.JSONDecodeError, TypeError):
                        skipped_reasons["bad_outcomes"] += 1
                        continue
                else:
                    outcomes = outcomes_raw

                if len(outcomes) < 2:
                    skipped_reasons["too_few_outcomes"] += 1
                    continue

                # Resolution: which outcome won?
                winning_idx = float_prices.index(max(float_prices))
                # For standard Yes/No: idx 0 = Yes won, idx 1 = No won
                actual_outcome = 1.0 if winning_idx == 0 else 0.0

                # --- Get pre-resolution price (HONEST) ---
                # lastTradePrice is the last trade before the market closed
                ltp = m.get("lastTradePrice")
                if not ltp:
                    skipped_reasons["no_last_trade_price"] += 1
                    continue

                try:
                    pre_price = float(ltp)
                except (ValueError, TypeError):
                    skipped_reasons["bad_ltp"] += 1
                    continue

                # Filter: skip if price was already extreme (no real prediction to make)
                if pre_price < 0.05 or pre_price > 0.95:
                    skipped_reasons["extreme_price"] += 1
                    continue

                # --- Build parsed market data (what the engine would see) ---
                parsed = MarketClient.parse_market(m)
                # Override with pre-resolution price (not the post-resolution 0/1)
                parsed["outcome_prices"] = {"Yes": pre_price, "No": 1.0 - pre_price}
                parsed["last_trade_price"] = pre_price

                all_markets.append({
                    "parsed": parsed,
                    "pre_price": pre_price,
                    "actual_outcome": actual_outcome,
                    "question": m.get("question", ""),
                    "category": classify_market(m.get("question", "")),
                    "volume": float(m.get("volume", 0) or 0),
                    "token_ids": m.get("clobTokenIds", []),
                    "end_date": m.get("endDate", ""),
                })

            offset += batch_size

            if len(all_markets) % 500 == 0 and all_markets:
                total_skipped = sum(skipped_reasons.values())
                print(f"    ...{len(all_markets)} collected (checked {offset}, skipped {total_skipped})")

            time.sleep(0.05)

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            time.sleep(2)
            offset += batch_size

    total_skipped = sum(skipped_reasons.values())
    print(f"  Collected {len(all_markets)} honest markets (skipped {total_skipped})")
    print(f"  Skip reasons:")
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    return all_markets


# =========================================================
# Backtest Engine
# =========================================================

class BacktestEngine:
    """Run models against resolved markets. No cheating."""

    def __init__(self):
        self.micro = MicrostructureModel()
        self.external = ExternalDataModel(backtest_mode=True)
        self.orderbook = OrderbookModel()
        self.ensemble = EnsemblePredictor()

        # Results tracking
        self.results = []
        self.model_predictions = defaultdict(list)  # model -> [(pred, actual)]
        self.category_results = defaultdict(list)    # category -> [(ens_pred, actual, edge, pre_price)]
        self.ensemble_predictions = []               # [(pred, actual)]

    def run_market(self, market: dict) -> dict:
        """Run all fast models against one resolved market."""
        parsed = market["parsed"]
        actual = market["actual_outcome"]
        pre_price = market["pre_price"]
        question = market["question"]

        sub_results = {}

        # 1. Microstructure (uses volume, spread, bid/ask)
        try:
            micro_r = self.micro.analyze(parsed)
            if micro_r.get("estimate") is not None:
                sub_results["microstructure"] = micro_r
                self.model_predictions["microstructure"].append(
                    (micro_r["estimate"], actual))
        except Exception:
            pass

        # 2. External data (Fear & Greed, macro indicators)
        try:
            keywords = self._extract_keywords(question)
            ext_r = self.external.analyze(parsed, keywords)
            if ext_r.get("estimate") is not None:
                sub_results["external_data"] = ext_r
                self.model_predictions["external_data"].append(
                    (ext_r["estimate"], actual))
        except Exception:
            pass

        # 3. Orderbook (CLOB data — may fail for old resolved markets)
        token_ids = market.get("token_ids", [])
        if token_ids:
            try:
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                ob_r = self.orderbook.analyze(token_ids[0], market_price=pre_price)
                if ob_r.get("estimate") is not None:
                    sub_results["orderbook"] = ob_r
                    self.model_predictions["orderbook"].append(
                        (ob_r["estimate"], actual))
            except Exception:
                pass

        # 4. Ensemble combine
        estimates = []
        variances = []
        model_names = []
        for name, result in sub_results.items():
            if result.get("estimate") is not None and result.get("variance") is not None:
                estimates.append(result["estimate"])
                variances.append(result["variance"])
                model_names.append(name)

        if estimates:
            ens_r = self.ensemble.combine(estimates, variances)
            ens_prob = ens_r["probability"]
        else:
            ens_prob = pre_price  # fallback

        ens_prob = max(0.01, min(0.99, ens_prob))

        self.ensemble_predictions.append((ens_prob, actual))

        edge = ens_prob - pre_price
        category = market["category"]
        self.category_results[category].append((ens_prob, actual, edge, pre_price))

        result = {
            "question": question[:80],
            "category": category,
            "pre_price": pre_price,
            "ensemble_pred": round(ens_prob, 4),
            "actual": actual,
            "edge": round(edge, 4),
            "models_used": model_names,
            "model_estimates": {n: round(sub_results[n]["estimate"], 4) for n in model_names},
        }
        self.results.append(result)
        return result

    def _extract_keywords(self, question: str) -> list:
        q = question.lower()
        keywords = []
        keyword_map = {
            "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana"],
            "oil": ["oil", "crude", "wti", "brent", "opec"],
            "gold": ["gold"],
            "fed": ["fed", "interest rate", "fomc", "rate cut", "rate hike"],
            "inflation": ["inflation", "cpi", "consumer price"],
            "recession": ["recession", "gdp"],
        }
        for cat, kws in keyword_map.items():
            for kw in kws:
                if kw in q:
                    keywords.append(cat)
                    break
        return keywords


# =========================================================
# Metrics
# =========================================================

def brier_score(predictions: list) -> float:
    """Brier score: mean squared error. Lower = better. 0 = perfect, 0.25 = random."""
    if not predictions:
        return 1.0
    return float(np.mean([(p - a) ** 2 for p, a in predictions]))


def calibration_table(predictions: list, n_bins: int = 10) -> list:
    if not predictions:
        return []
    bins = defaultdict(list)
    for pred, actual in predictions:
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, actual))

    table = []
    for i in range(n_bins):
        if i not in bins or not bins[i]:
            continue
        preds = [p for p, a in bins[i]]
        actuals = [a for p, a in bins[i]]
        table.append({
            "bin": f"{i*10}-{(i+1)*10}%",
            "mean_pred": round(np.mean(preds), 3),
            "actual_rate": round(np.mean(actuals), 3),
            "count": len(bins[i]),
            "gap": round(abs(np.mean(preds) - np.mean(actuals)), 3),
        })
    return table


def simulate_trading(data_with_price: list, edge_threshold: float = 0.02) -> dict:
    """Simulate $1 trades where our edge > threshold. HONEST P&L."""
    buy_yes_pnl = []
    buy_no_pnl = []

    for ens_prob, actual, edge, pre_price in data_with_price:
        if edge > edge_threshold:
            # We think YES is underpriced -> BUY YES at pre_price
            # Profit = $1 - pre_price if YES wins, -pre_price if NO wins
            pnl = (1.0 - pre_price) if actual == 1.0 else -pre_price
            buy_yes_pnl.append({"pnl": pnl, "edge": edge})
        elif edge < -edge_threshold:
            # We think NO is underpriced -> BUY NO at (1 - pre_price)
            no_price = 1.0 - pre_price
            pnl = (1.0 - no_price) if actual == 0.0 else -no_price
            buy_no_pnl.append({"pnl": pnl, "edge": edge})

    all_pnl = [t["pnl"] for t in buy_yes_pnl + buy_no_pnl]

    result = {
        "trades": len(all_pnl),
        "buy_yes": len(buy_yes_pnl),
        "buy_no": len(buy_no_pnl),
    }

    if all_pnl:
        result["total_pnl"] = round(sum(all_pnl), 2)
        result["avg_pnl"] = round(np.mean(all_pnl), 4)
        result["win_rate"] = round(sum(1 for p in all_pnl if p > 0) / len(all_pnl), 3)
        result["avg_win"] = round(np.mean([p for p in all_pnl if p > 0]), 4) if any(p > 0 for p in all_pnl) else 0
        result["avg_loss"] = round(np.mean([p for p in all_pnl if p < 0]), 4) if any(p < 0 for p in all_pnl) else 0
        result["sharpe"] = round(np.mean(all_pnl) / (np.std(all_pnl) + 1e-10) * np.sqrt(len(all_pnl)), 2)

    # Edge bucket analysis
    edge_buckets = {
        "2-5%":   (0.02, 0.05),
        "5-10%":  (0.05, 0.10),
        "10-20%": (0.10, 0.20),
        "20%+":   (0.20, 1.0),
    }
    result["edge_buckets"] = {}
    for name, (lo, hi) in edge_buckets.items():
        bucket = [t for t in buy_yes_pnl + buy_no_pnl if lo <= abs(t["edge"]) < hi]
        if bucket:
            bpnl = [t["pnl"] for t in bucket]
            result["edge_buckets"][name] = {
                "trades": len(bucket),
                "win_rate": round(sum(1 for p in bpnl if p > 0) / len(bpnl), 3),
                "avg_pnl": round(np.mean(bpnl), 4),
                "total_pnl": round(sum(bpnl), 2),
            }

    return result


# =========================================================
# Report
# =========================================================

def print_report(engine: BacktestEngine):
    n = len(engine.ensemble_predictions)

    print("\n" + "=" * 70)
    print("  HONEST BACKTEST RESULTS")
    print(f"  {n} resolved markets | No data peeking | Real accuracy")
    print("=" * 70)

    # --- Brier Scores ---
    ens_brier = brier_score(engine.ensemble_predictions)
    mkt_brier = brier_score([(r["pre_price"], r["actual"]) for r in engine.results])
    naive_brier = 0.25  # always predicting 0.5

    print(f"\n  BRIER SCORES (lower = better, 0.25 = random coin flip):")
    print(f"    Our ensemble:  {ens_brier:.4f}")
    print(f"    Market price:  {mkt_brier:.4f}")
    print(f"    Naive (0.5):   {naive_brier:.4f}")

    if ens_brier < mkt_brier:
        improvement = (mkt_brier - ens_brier) / mkt_brier * 100
        print(f"    -> WE BEAT THE MARKET by {improvement:.1f}%")
    else:
        degradation = (ens_brier - mkt_brier) / mkt_brier * 100
        print(f"    -> Market beats us by {degradation:.1f}% (our models add noise)")

    # --- Per-Model Brier Scores ---
    print(f"\n  PER-MODEL ACCURACY:")
    print(f"    {'Model':<20} {'Brier':>8} {'N':>6} {'vs Market':>10} {'Verdict':>10}")
    print(f"    {'-'*20} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
    for model in sorted(engine.model_predictions.keys()):
        preds = engine.model_predictions[model]
        if preds:
            mb = brier_score(preds)
            diff = mb - mkt_brier
            verdict = "GOOD" if diff < -0.005 else "NEUTRAL" if abs(diff) < 0.005 else "BAD"
            print(f"    {model:<20} {mb:>8.4f} {len(preds):>6} {diff:>+10.4f} {verdict:>10}")

    # --- Per-Category ---
    print(f"\n  PER-CATEGORY BREAKDOWN:")
    print(f"    {'Category':<15} {'N':>5} {'Our Brier':>10} {'Mkt Brier':>10} {'Beat Mkt?':>10} {'Avg|Edge|':>10}")
    print(f"    {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cat in sorted(engine.category_results.keys()):
        cat_data = engine.category_results[cat]
        if len(cat_data) < 5:
            continue
        our = brier_score([(p, a) for p, a, e, pp in cat_data])
        mkt = brier_score([(pp, a) for p, a, e, pp in cat_data])
        avg_edge = np.mean([abs(e) for p, a, e, pp in cat_data])
        beat = "YES" if our < mkt else "no"
        print(f"    {cat:<15} {len(cat_data):>5} {our:>10.4f} {mkt:>10.4f} {beat:>10} {avg_edge:>10.4f}")

    # --- Calibration ---
    print(f"\n  CALIBRATION (are our probabilities honest?):")
    cal = calibration_table(engine.ensemble_predictions)
    print(f"    {'Bin':<10} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'N':>6}")
    print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
    for c in cal:
        gap_flag = " !!!" if c["gap"] > 0.10 else ""
        print(f"    {c['bin']:<10} {c['mean_pred']:>10.3f} {c['actual_rate']:>10.3f} {c['gap']:>8.3f} {c['count']:>6}{gap_flag}")

    # --- P&L Simulation ---
    all_data = [(p, a, e, pp)
                for cat_data in engine.category_results.values()
                for p, a, e, pp in cat_data]

    for threshold in [0.02, 0.05, 0.10]:
        pnl = simulate_trading(all_data, edge_threshold=threshold)
        print(f"\n  P&L SIMULATION (edge > {threshold:.0%}, $1/trade):")
        print(f"    Trades taken: {pnl['trades']} (BUY_YES={pnl['buy_yes']}, BUY_NO={pnl['buy_no']})")
        if pnl.get("total_pnl") is not None:
            print(f"    Total P&L:    ${pnl['total_pnl']:+.2f}")
            print(f"    Avg P&L:      ${pnl['avg_pnl']:+.4f}")
            print(f"    Win rate:     {pnl['win_rate']:.1%}")
            print(f"    Avg win:      ${pnl.get('avg_win', 0):+.4f}")
            print(f"    Avg loss:     ${pnl.get('avg_loss', 0):+.4f}")
            print(f"    Sharpe:       {pnl.get('sharpe', 0):.2f}")

            if pnl.get("edge_buckets"):
                print(f"\n    Edge Buckets:")
                print(f"    {'Bucket':<10} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Total':>9}")
                print(f"    {'-'*10} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
                for bname, bd in pnl["edge_buckets"].items():
                    print(f"    {bname:<10} {bd['trades']:>7} {bd['win_rate']:>7.1%} ${bd['avg_pnl']:>+8.4f} ${bd['total_pnl']:>+8.2f}")
        else:
            print(f"    No trades met threshold")

    # --- Per-Category P&L ---
    print(f"\n  PER-CATEGORY P&L (edge > 2%, $1/trade):")
    print(f"    {'Category':<15} {'Trades':>7} {'Win%':>7} {'Total P&L':>10} {'Avg P&L':>9}")
    print(f"    {'-'*15} {'-'*7} {'-'*7} {'-'*10} {'-'*9}")
    for cat in sorted(engine.category_results.keys()):
        cat_data = engine.category_results[cat]
        pnl = simulate_trading(cat_data, edge_threshold=0.02)
        if pnl["trades"] > 0:
            print(f"    {cat:<15} {pnl['trades']:>7} {pnl.get('win_rate',0):>7.1%} ${pnl.get('total_pnl',0):>+9.2f} ${pnl.get('avg_pnl',0):>+8.4f}")

    print("\n" + "=" * 70)

    return {
        "n_markets": n,
        "ensemble_brier": ens_brier,
        "market_brier": mkt_brier,
    }


# =========================================================
# Main
# =========================================================

# =========================================================
# Cross-Platform Backtest
# =========================================================

STOP_WORDS = {"will", "the", "be", "by", "in", "on", "of", "a", "an", "to", "and",
              "or", "is", "it", "at", "for", "this", "that", "with", "from", "before",
              "after", "has", "have", "been", "does", "do", "what", "when", "where"}

GENERIC_WORDS = {"win", "presidential", "election", "president", "price", "above",
                 "below", "hit", "reach", "market", "world", "cup", "nomination",
                 "nominee", "candidate", "rate", "rates", "interest", "change"}

import re

def extract_specific_keywords(q: str) -> set:
    q = re.sub(r'[^\w\s]', '', q.lower().strip())
    words = q.split()
    return {w for w in words if w not in STOP_WORDS and w not in GENERIC_WORDS and len(w) > 2}


def match_score(q1: str, q2: str) -> float:
    kw1 = extract_specific_keywords(q1)
    kw2 = extract_specific_keywords(q2)
    if not kw1 or not kw2:
        return 0
    overlap = kw1 & kw2
    if len(overlap) < 2:
        return 0
    return len(overlap) / max(len(kw1), len(kw2))


def fetch_manifold_resolved(search_terms: list, limit_per_term: int = 200) -> list:
    """Fetch resolved Manifold markets with pre-resolution probabilities."""
    session = requests.Session()
    all_markets = {}

    for term in search_terms:
        try:
            r = session.get("https://api.manifold.markets/v0/search-markets", params={
                "term": term, "filter": "resolved", "sort": "liquidity",
                "limit": limit_per_term,
            }, timeout=15)
            if r.status_code != 200:
                continue
            for m in r.json():
                mid = m.get("id")
                if mid and mid not in all_markets:
                    res = m.get("resolution")
                    if res in ("YES", "NO"):
                        # Get pre-resolution probability from bets
                        prob = m.get("probability", 0.5)
                        # probability field on resolved markets is post-resolution
                        # We need the last bet's probBefore as pre-resolution price
                        all_markets[mid] = {
                            "id": mid,
                            "question": m.get("question", ""),
                            "resolution": 1.0 if res == "YES" else 0.0,
                            "post_prob": prob,
                            "volume": m.get("volume", 0),
                        }
        except Exception:
            continue
        time.sleep(0.2)

    # Now fetch pre-resolution probabilities from bets
    print(f"    Fetching pre-resolution prices for {len(all_markets)} Manifold markets...")
    fetched = 0
    for mid, market in list(all_markets.items()):
        try:
            r = session.get(f"https://api.manifold.markets/v0/bets", params={
                "contractId": mid, "limit": 5, "order": "desc",
            }, timeout=10)
            if r.status_code == 200:
                bets = r.json()
                # Find last bet with probBefore (pre-resolution state)
                for bet in bets:
                    pb = bet.get("probBefore")
                    if pb is not None and 0.05 <= pb <= 0.95:
                        market["pre_prob"] = pb
                        fetched += 1
                        break
        except Exception:
            pass
        time.sleep(0.05)

    # Filter to only those with pre-resolution probabilities
    result = [m for m in all_markets.values() if "pre_prob" in m]
    print(f"    Got pre-resolution prices for {fetched} markets, usable: {len(result)}")
    return result


def run_cross_platform_backtest():
    """Match resolved Polymarket vs Manifold markets. Test if divergence = profit."""
    print("=" * 70)
    print("  CROSS-PLATFORM BACKTEST")
    print("  Does Manifold vs Polymarket divergence predict outcomes?")
    print("=" * 70)

    # Step 1: Fetch resolved Polymarket markets
    print("\n  [1/3] Fetching resolved Polymarket markets...")
    poly_markets = fetch_resolved_markets(target_count=2000)
    print(f"  Polymarket: {len(poly_markets)} resolved markets")

    # Step 2: Fetch resolved Manifold markets (search for topics that overlap)
    print("\n  [2/3] Fetching resolved Manifold markets...")
    search_terms = [
        "bitcoin", "ethereum", "trump", "biden", "fed rate",
        "inflation", "recession", "iran", "ukraine", "taiwan",
        "oil", "gold", "election", "AI", "openai", "world cup",
        "fifa", "nba", "super bowl", "elon musk", "spacex",
        "china", "russia", "tariff", "congress", "supreme court",
    ]
    manifold_markets = fetch_manifold_resolved(search_terms)
    print(f"  Manifold: {len(manifold_markets)} resolved markets with pre-res prices")

    # Step 3: Match
    print("\n  [3/3] Matching markets across platforms...")
    matches = []
    used_manifold = set()

    for pm in poly_markets:
        pq = pm["question"]
        best_match = None
        best_score = 0

        for mm in manifold_markets:
            if mm["id"] in used_manifold:
                continue
            score = match_score(pq, mm["question"])
            if score > best_score:
                best_score = score
                best_match = mm

        if best_match and best_score >= 0.3:
            used_manifold.add(best_match["id"])
            matches.append({
                "poly_question": pq[:80],
                "manifold_question": best_match["question"][:80],
                "match_score": best_score,
                "poly_price": pm["pre_price"],
                "manifold_price": best_match["pre_prob"],
                "actual_outcome": pm["actual_outcome"],
                "divergence": best_match["pre_prob"] - pm["pre_price"],
                "category": pm["category"],
            })

    print(f"  Matched: {len(matches)} markets across platforms")

    if len(matches) < 5:
        print("  Not enough matches for meaningful analysis!")
        return

    # --- Report ---
    print("\n" + "=" * 70)
    print(f"  CROSS-PLATFORM RESULTS ({len(matches)} matched markets)")
    print("=" * 70)

    # Brier comparison
    poly_brier = brier_score([(m["poly_price"], m["actual_outcome"]) for m in matches])
    manifold_brier = brier_score([(m["manifold_price"], m["actual_outcome"]) for m in matches])
    # Combined: average of the two platforms
    combined_brier = brier_score([((m["poly_price"] + m["manifold_price"]) / 2, m["actual_outcome"]) for m in matches])
    # Divergence-weighted: trust the platform that's more extreme
    def divergence_weighted(m):
        poly_conf = abs(m["poly_price"] - 0.5)
        mani_conf = abs(m["manifold_price"] - 0.5)
        if poly_conf + mani_conf == 0:
            return 0.5
        w = mani_conf / (poly_conf + mani_conf)
        return m["poly_price"] * (1 - w) + m["manifold_price"] * w
    divw_brier = brier_score([(divergence_weighted(m), m["actual_outcome"]) for m in matches])

    print(f"\n  BRIER SCORES (lower = better):")
    print(f"    Polymarket alone:   {poly_brier:.4f}")
    print(f"    Manifold alone:     {manifold_brier:.4f}")
    print(f"    Simple average:     {combined_brier:.4f}")
    print(f"    Confidence-weighted:{divw_brier:.4f}")

    winner = "Polymarket" if poly_brier < manifold_brier else "Manifold"
    print(f"    -> {winner} is more accurate on matched markets")
    if combined_brier < min(poly_brier, manifold_brier):
        print(f"    -> COMBINING PLATFORMS BEATS BOTH (this is the edge!)")

    # --- Divergence Analysis ---
    print(f"\n  DIVERGENCE ANALYSIS (when platforms disagree):")
    div_buckets = {
        "agree (<5%)": (0, 0.05),
        "small (5-10%)": (0.05, 0.10),
        "medium (10-20%)": (0.10, 0.20),
        "large (20%+)": (0.20, 1.0),
    }
    print(f"    {'Divergence':<20} {'N':>5} {'Poly Brier':>11} {'Mani Brier':>11} {'Avg Brier':>10} {'Who wins':>10}")
    print(f"    {'-'*20} {'-'*5} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for name, (lo, hi) in div_buckets.items():
        bucket = [m for m in matches if lo <= abs(m["divergence"]) < hi]
        if bucket:
            pb = brier_score([(m["poly_price"], m["actual_outcome"]) for m in bucket])
            mb = brier_score([(m["manifold_price"], m["actual_outcome"]) for m in bucket])
            ab = brier_score([((m["poly_price"]+m["manifold_price"])/2, m["actual_outcome"]) for m in bucket])
            who = "Poly" if pb < mb else "Manifold"
            print(f"    {name:<20} {len(bucket):>5} {pb:>11.4f} {mb:>11.4f} {ab:>10.4f} {who:>10}")

    # --- P&L: Trade Polymarket when Manifold disagrees ---
    print(f"\n  P&L: USE MANIFOLD TO TRADE POLYMARKET")
    print(f"  Strategy: When Manifold > Poly by X%, BUY YES on Poly")
    print(f"            When Manifold < Poly by X%, BUY NO on Poly")

    for threshold in [0.05, 0.10, 0.15, 0.20]:
        trades = []
        for m in matches:
            div = m["divergence"]  # manifold - poly
            if div > threshold:
                # Manifold thinks YES more likely -> BUY YES on Poly
                pnl = (1.0 - m["poly_price"]) if m["actual_outcome"] == 1.0 else -m["poly_price"]
                trades.append(pnl)
            elif div < -threshold:
                # Manifold thinks NO more likely -> BUY NO on Poly
                no_price = 1.0 - m["poly_price"]
                pnl = (1.0 - no_price) if m["actual_outcome"] == 0.0 else -no_price
                trades.append(pnl)

        if trades:
            print(f"\n    Threshold: {threshold:.0%} divergence")
            print(f"      Trades:    {len(trades)}")
            print(f"      Total P&L: ${sum(trades):+.2f}")
            print(f"      Avg P&L:   ${np.mean(trades):+.4f}")
            print(f"      Win rate:  {sum(1 for p in trades if p > 0)/len(trades):.1%}")
        else:
            print(f"\n    Threshold: {threshold:.0%} - no trades")

    # --- Show top matches ---
    print(f"\n  TOP MATCHES (by divergence):")
    sorted_matches = sorted(matches, key=lambda m: abs(m["divergence"]), reverse=True)
    for m in sorted_matches[:10]:
        poly_right = abs(m["poly_price"] - m["actual_outcome"]) < abs(m["manifold_price"] - m["actual_outcome"])
        winner = "POLY" if poly_right else "MANI"
        print(f"    [{winner}] Poly={m['poly_price']:.2f} Mani={m['manifold_price']:.2f} Actual={m['actual_outcome']:.0f} Div={m['divergence']:+.2f}")
        print(f"           {m['poly_question']}")

    print("\n" + "=" * 70)

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_matched": len(matches),
        "poly_brier": poly_brier,
        "manifold_brier": manifold_brier,
        "combined_brier": combined_brier,
        "matches": matches[:50],
    }
    with open("data/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to data/backtest_results.json")


# =========================================================
# Main
# =========================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", type=int, default=None)
    parser.add_argument("--xplatform", action="store_true", help="Cross-platform backtest")
    args = parser.parse_args()

    if args.xplatform:
        run_cross_platform_backtest()
        return

    target = args.quick or 5000

    print("=" * 70)
    print("  HONEST BACKTEST")
    print(f"  Target: {target} resolved markets")
    print("  Rules: lastTradePrice only, no post-resolution data, no cheating")
    print("=" * 70)

    # Step 1: Fetch
    t0 = time.time()
    markets = fetch_resolved_markets(target_count=target)
    print(f"  Fetch time: {time.time() - t0:.1f}s")

    if len(markets) < 10:
        print("  Not enough markets with honest pre-resolution prices!")
        return

    # Step 2: Run backtest
    print(f"\n  Running {len(markets)} markets through prediction models...")
    print(f"  Models: microstructure, external_data, orderbook")
    print(f"  (Skipping AI semantic -- too slow for {len(markets)} markets)")
    print(f"  (Skipping time_series -- needs 10+ price snapshots per market)")

    engine = BacktestEngine()
    t0 = time.time()

    for i, market in enumerate(markets):
        try:
            engine.run_market(market)
        except Exception:
            pass

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(markets) - i - 1) / rate if rate > 0 else 0
            ens_so_far = brier_score(engine.ensemble_predictions)
            print(f"    {i+1}/{len(markets)} ({rate:.0f}/sec, ETA {eta:.0f}s) Brier so far: {ens_so_far:.4f}")

    run_time = time.time() - t0
    print(f"  Done: {run_time:.1f}s ({len(markets)/max(run_time,0.1):.0f} markets/sec)")

    # Step 3: Report
    summary = print_report(engine)

    # Step 4: Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_markets": len(markets),
        "ensemble_brier": summary["ensemble_brier"],
        "market_brier": summary["market_brier"],
        "model_counts": {m: len(p) for m, p in engine.model_predictions.items()},
        "category_counts": {c: len(d) for c, d in engine.category_results.items()},
    }
    os.makedirs("data", exist_ok=True)
    with open("data/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to data/backtest_results.json")


if __name__ == "__main__":
    main()
