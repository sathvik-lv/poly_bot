"""Paper Trader — Simulates real trading with fake money.

Runs the full pipeline: scan markets → predict → check entry gates →
place paper trades → track P&L → resolve when markets close.

No real money. No API keys needed. Proves the system before going live.

Usage:
    python scripts/paper_trader.py                # scan + trade + resolve + report
    python scripts/paper_trader.py --scan         # scan and place new paper trades
    python scripts/paper_trader.py --resolve      # check resolutions
    python scripts/paper_trader.py --report       # show P&L report
    python scripts/paper_trader.py --equity 5000  # start with $5000 (default $10000)
"""

import argparse
import json
import math
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
from src.strategy_adapter import StrategyAdapter, MarketRegime

DATA_DIR = "data"
PAPER_FILE = os.path.join(DATA_DIR, "paper_trades.json")
ANALYTICS_FILE = os.path.join(DATA_DIR, "weight_analytics.json")
GAMMA_API = "https://gamma-api.polymarket.com"

# Category-tiered Kelly multipliers (HIGH/MEDIUM/SKIP).
# Derived from grid-sweep per-category analysis: HIGH categories have
# ret/DD >= 5 and WR >= 55%, MEDIUM has ret/DD >= 1.5 and WR >= 50%,
# everything else has insufficient data or negative ret/DD -> SKIP.
# Gated behind ENABLE_CATEGORY_TIERS=1; defaults to old behavior (1x for all).
CATEGORY_TIER_MULT = {
    # HIGH tier (n>=30, WR>=55%, ret/DD>=5): bet full Kelly
    "sports":       1.0,
    "niche_sports": 1.0,
    # MEDIUM tier (n>=30, WR>=50%, ret/DD>=1.5): half Kelly
    "other":        0.5,
    # SKIP tier (n<20 OR WR<50% OR ret/DD<1.5): zero capital, no entry
    "crypto":       0.0,
    "geopolitics":  0.0,
    "elections":    0.0,
    "tech_ai":      0.0,
    "fed_rate":     0.0,
    "oil_energy":   0.0,
    "macro":        0.0,
}
DEFAULT_TIER_MULT = 0.0  # any unknown category is SKIPped under tier mode

# Market category classification keywords.
# Order matters — first match wins. niche_sports comes BEFORE sports so the
# proven-winning patterns (tennis tournaments, esports, micro-prop bets) get
# their own bucket instead of being lumped into the inefficient mainstream
# "sports" category. n=104, WR 71.2% on this niche subset (was hidden in "other").
CATEGORY_RULES = [
    ("fed_rate",     ["fed ", "interest rate", "fomc", "federal reserve", "bps"]),
    ("crypto",       ["bitcoin", "btc ", "ethereum", "eth ", "crypto", "solana"]),
    ("oil_energy",   ["oil", "crude", "wti ", "brent", "opec", "energy"]),
    ("macro",        ["gdp", "inflation", "cpi ", "recession", "unemployment", "tariff"]),
    ("geopolitics",  ["invade", "invasion", "war ", "military", "regime", "sanctions", "iran", "taiwan", "ukraine", "russia"]),
    ("elections",    ["election", "president", "governor", "senate", "nominee", "primary", "democrat", "republican"]),
    ("niche_sports", [
        # Tennis tournaments / qualifiers
        "atp ", "wta ", "rolex monte carlo", "upper austria", "bmw open",
        "barcelona open", "porsche tennis", "indian wells", "miami open",
        "madrid open", "italian open", "wimbledon", "us open tennis",
        "qualification", "qualifying", "ladies linz",
        # Cricket leagues
        "ipl", "indian premier league", "t20", "test match", "odi ",
        # Esports
        "counter-strike", "csgo", "cs:go", "cs2", "overwatch", "league of legends",
        " dota", "valorant", "starcraft", "rocket league", "esports",
        # Micro-prop bet structures (very profitable subset of "other")
        "over/under", "o/u ", "total kills", "set handicap", "spread:",
        "games total", "round robin", "handicap:",
    ]),
    ("sports",       ["fifa", "world cup", "nba ", "nfl ", "mlb ", "nhl ", "f1 ", "champion", "tournament", "ufc"]),
    ("tech_ai",      ["openai", "chatgpt", "artificial intelligence", " ai ", "google", "apple", "tesla"]),
]


def classify_market(question: str) -> str:
    """Classify a market question into a category."""
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
        remaining = (end - now).total_seconds()
        total = max(remaining * 2, 86400)
        return max(0.01, min(1.0, remaining / total))
    except Exception:
        return 0.5


def load_state() -> dict:
    if os.path.exists(PAPER_FILE):
        with open(PAPER_FILE) as f:
            return json.load(f)
    return {
        "initial_equity": 10000.0,
        "equity": 10000.0,
        "cash": 10000.0,
        "open_positions": [],
        "closed_positions": [],
        "total_trades": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_scan": None,
    }


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PAPER_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =========================================================
# SCAN + TRADE
# =========================================================

def scan_and_trade(state: dict, n_markets: int = 30):
    """Scan markets, run predictions, place paper trades."""
    print(f"\n  PAPER TRADER — Scanning {n_markets} markets")
    print(f"  Cash: ${state['cash']:.2f} | Open positions: {len(state['open_positions'])}")

    engine = PredictionEngine(backtest_mode=False, total_equity=state["equity"])
    client = MarketClient()
    strategy = StrategyAdapter(total_equity=state["equity"])

    # Update regime from Fear & Greed
    try:
        fg_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        fg_val = int(fg_resp.json().get("data", [{}])[0].get("value", 50))
        strategy.allocator.update_regime(fg_val)
        print(f"  Regime: {strategy.allocator.regime.regime_label} (F&G={fg_val})")
    except Exception:
        fg_val = 50

    # Get open position market IDs to avoid doubling
    open_ids = {p["market_id"] for p in state["open_positions"]}

    # Fetch markets from multiple pools, but ONLY keep markets resolving
    # within 14 days — fast feedback is critical for weight analytics.
    MAX_DAYS = 14
    seen_ids = set()
    raw_markets = []
    now = datetime.now(timezone.utc)

    def passes_end_date_filter(m):
        """Return days_left if market resolves within MAX_DAYS, else None."""
        end_str = m.get("endDate", "")
        if not end_str:
            return None
        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            days_left = (end_dt - now).total_seconds() / 86400
            if 0.5 < days_left <= MAX_DAYS:
                return round(days_left, 1)
        except Exception:
            pass
        return None

    # Pool 1: Top volume markets (filtered to ≤14 days)
    try:
        vol_markets = client.session.get(
            f"{client.base_url}/markets",
            params={"limit": 100, "active": True, "closed": False,
                    "order": "volume24hr", "ascending": False},
            timeout=15,
        ).json()
        for m in vol_markets:
            mid = m.get("id")
            days_left = passes_end_date_filter(m)
            if mid and mid not in seen_ids and days_left is not None:
                seen_ids.add(mid)
                m["_days_left"] = days_left
                raw_markets.append(m)
    except Exception:
        pass

    # Pool 2: Recently created markets (filtered to ≤14 days)
    try:
        recent_markets = client.session.get(
            f"{client.base_url}/markets",
            params={"limit": 100, "active": True, "closed": False,
                    "order": "startDate", "ascending": False},
            timeout=15,
        ).json()
        for m in recent_markets:
            mid = m.get("id")
            days_left = passes_end_date_filter(m)
            if mid and mid not in seen_ids and days_left is not None:
                seen_ids.add(mid)
                m["_days_left"] = days_left
                raw_markets.append(m)
    except Exception:
        pass

    # Pool 3: End-date sorted (soonest first)
    try:
        ending_soon = client.session.get(
            f"{client.base_url}/markets",
            params={"limit": 100, "active": True, "closed": False,
                    "order": "endDate", "ascending": True},
            timeout=15,
        ).json()
        for m in ending_soon:
            mid = m.get("id")
            days_left = passes_end_date_filter(m)
            if mid and mid not in seen_ids and days_left is not None:
                seen_ids.add(mid)
                m["_days_left"] = days_left
                raw_markets.append(m)
    except Exception:
        pass

    print(f"  Fetched {len(raw_markets)} markets resolving within {MAX_DAYS} days")

    candidates = []
    n_short_candidates = 0
    for raw in raw_markets:
        parsed = MarketClient.parse_market(raw)
        price = parsed.get("outcome_prices", {}).get("Yes")
        # Fall back to first outcome price for non-Yes/No markets (Up/Down, etc.)
        if price is None:
            op = parsed.get("outcome_prices", {})
            first_vals = [v for v in op.values() if v is not None]
            price = first_vals[0] if first_vals else None
        if price is None or price < 0.05 or price > 0.95:
            continue
        if not parsed.get("active"):
            continue
        if parsed.get("id") in open_ids:
            continue
        token_ids = parse_token_ids(raw)
        parsed["_token_id"] = token_ids[0] if token_ids else None
        parsed["_raw"] = raw
        if raw.get("_days_left"):
            parsed["_days_left"] = raw["_days_left"]
            n_short_candidates += 1
        candidates.append(parsed)

    # Sort: short-dated first (faster feedback), then by volume
    candidates.sort(key=lambda x: (
        0 if x.get("_days_left") else 1,  # short-dated first
        x.get("_days_left", 999),          # sooner ending first
    ))

    print(f"  Candidates: {len(candidates)} ({n_short_candidates} short-dated) "
          f"(excluding {len(open_ids)} already open)\n")

    new_trades = 0
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

            # Run strategy evaluation
            strat = strategy.evaluate_entry(market, prediction)
            elapsed = time.time() - t0

            edge = prediction["edge"]["edge"]
            prob = prediction["prediction"]["probability"]
            price = prediction["market"]["current_price"]
            models = ",".join(prediction["ensemble"]["model_names"])

            if not strat["should_enter"]:
                if abs(edge) >= 0.03:
                    q = market.get("question", "")[:50]
                    failed = strat["filters"].get("failed_filters", [])
                    print(f"  [{i+1:>2}] SKIP {q}")
                    print(f"       Edge={edge:+.3f} but failed: {failed}")
                continue

            # PLACE PAPER TRADE
            sizing = strat["sizing"]
            trade_amount = sizing["size_dollars"]

            # Category-tier multiplier (gated). HIGH=1.0, MEDIUM=0.5, SKIP=0.0.
            # Backed by grid-sweep per-category ret/DD analysis.
            category = classify_market(market.get("question", ""))
            tier_mult = 1.0
            tier_applied = "ungated"
            if os.environ.get("ENABLE_CATEGORY_TIERS", "0") == "1":
                tier_mult = CATEGORY_TIER_MULT.get(category, DEFAULT_TIER_MULT)
                if tier_mult >= 1.0:
                    tier_applied = "HIGH"
                elif tier_mult >= 0.4:
                    tier_applied = "MEDIUM"
                else:
                    tier_applied = "SKIP"
                if tier_mult <= 0:
                    q = market.get("question", "")[:50]
                    print(f"  [{i+1:>2}] SKIP-TIER [{category:<12}] {q}  Edge={edge:+.3f}")
                    continue
                trade_amount = trade_amount * tier_mult

            if trade_amount <= 0 or trade_amount > state["cash"]:
                continue

            action = sizing["action"]
            days_left = market.get("_days_left")
            q = market.get("question", "")[:55]

            # Extract per-model estimates for attribution
            model_estimates = {}
            for mname, mresult in prediction.get("sub_models", {}).items():
                est = mresult.get("estimate")
                if est is not None:
                    model_estimates[mname] = round(float(est), 4)

            position = {
                "trade_id": f"paper_{int(time.time()*1000)}_{new_trades}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_id": market.get("id"),
                "question": market.get("question", ""),
                "category": category,
                "tier_applied": tier_applied,
                "tier_mult": tier_mult,
                "action": action,
                "entry_price": price,
                "predicted_prob": prob,
                "edge": edge,
                "abs_edge": round(abs(edge), 4),
                "edge_confidence": prediction["edge"]["edge_confidence"],
                "amount": trade_amount,
                "kelly": sizing.get("kelly_adjusted", 0),
                "regime": strat["regime"],
                "models": prediction["ensemble"]["model_names"],
                "model_estimates": model_estimates,
                "n_models": prediction["ensemble"]["n_models"],
                "days_left": days_left,
                "ai_reasoning": prediction.get("sub_models", {}).get(
                    "ai_semantic", {}).get("signals", {}).get("ai_reasoning"),
            }

            state["open_positions"].append(position)
            state["cash"] -= trade_amount
            state["total_trades"] += 1
            new_trades += 1

            dl_tag = f" | {days_left:.0f}d left" if days_left else ""
            print(f"  [{i+1:>2}] TRADE {q}")
            print(f"       {action} ${trade_amount:.2f} | Price={price:.3f} Pred={prob:.3f} "
                  f"Edge={edge:+.3f} | {models} ({elapsed:.1f}s){dl_tag}")
            if position.get("ai_reasoning"):
                print(f"       AI: {position['ai_reasoning'][:70]}".encode("ascii", "replace").decode("ascii"))

        except Exception as e:
            print(f"  [{i+1:>2}] ERROR: {str(e)[:60]}")

    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\n  New trades: {new_trades}")
    print(f"  Cash remaining: ${state['cash']:.2f}")
    print(f"  Open positions: {len(state['open_positions'])}")


# =========================================================
# RESOLVE
# =========================================================

def early_exit_open_positions(state: dict):
    """Re-predict each open position; close it if entry edge has decayed.

    Disabled by default. Set ENABLE_EARLY_EXIT=1 to turn on.
    Tunable via EXIT_DECAY_THRESHOLD (default 0.5) and EXIT_SPREAD_COST
    (default 0.02 — 2% slippage on close).

    Backed by data: scripts/v2_exit_sim.py shows ~+18% PnL improvement
    on the existing ledger at decay=0.5, spread=0.02.
    """
    if os.environ.get("ENABLE_EARLY_EXIT", "0") != "1":
        return
    open_pos = state["open_positions"]
    if not open_pos:
        return

    decay_threshold = float(os.environ.get("EXIT_DECAY_THRESHOLD", "0.5"))
    spread_cost = float(os.environ.get("EXIT_SPREAD_COST", "0.02"))

    # Lazy imports — keeps the resolver fast when feature is off
    from src.exit_simulator import per_dollar_pnl, should_exit
    from src.market_client import MarketClient
    from src.prediction_engine import PredictionEngine

    print(f"\n  EARLY EXIT CHECK: {len(open_pos)} open, "
          f"decay_threshold={decay_threshold}, spread_cost={spread_cost}")

    engine = PredictionEngine(backtest_mode=False, total_equity=state["equity"])
    client = MarketClient()

    still_open = []
    for pos in open_pos:
        mid = pos.get("market_id")
        if not mid or pos.get("entry_edge") is None and pos.get("edge") is None:
            still_open.append(pos)
            continue
        try:
            resp = client.session.get(f"{client.base_url}/markets/{mid}", timeout=10)
            if resp.status_code != 200:
                still_open.append(pos)
                continue
            market = resp.json()
            if market.get("closed"):
                # Will be handled by resolve_positions
                still_open.append(pos)
                continue
            parsed = MarketClient.parse_market(market)
            time_rem = compute_time_remaining(parsed.get("end_date"))
            tids = parse_token_ids(market)
            parsed["_token_id"] = tids[0] if tids else None
            prediction = engine.predict(
                market_data=parsed,
                time_remaining_frac=time_rem,
                token_id=parsed.get("_token_id"),
            )
            new_edge = prediction["edge"]["edge"]
            current_price = prediction["market"]["current_price"]
            entry_edge = float(pos.get("edge") or pos.get("entry_edge") or 0.0)

            if not should_exit(entry_edge, new_edge, decay_threshold):
                still_open.append(pos)
                continue

            # Trigger early exit
            action = pos["action"]
            entry_price = pos["entry_price"]
            amount = pos["amount"]
            pnl_per_dollar = per_dollar_pnl(action, entry_price, current_price) - spread_cost
            pnl = round(amount * pnl_per_dollar, 2)
            payout = round(amount + pnl, 2)
            pos["outcome"] = None  # not resolved yet
            pos["pnl"] = pnl
            pos["payout"] = payout
            pos["resolved_at"] = datetime.now(timezone.utc).isoformat()
            pos["result"] = "EXIT_WIN" if pnl > 0 else "EXIT_LOSS" if pnl < 0 else "EXIT_FLAT"
            pos["closed_reason"] = "edge_decay"
            pos["exit_price"] = round(current_price, 4)
            pos["entry_edge"] = entry_edge
            pos["exit_edge"] = round(float(new_edge), 4)
            state["closed_positions"].append(pos)
            state["cash"] += payout
            tag = "EXIT-WIN" if pnl > 0 else "EXIT-LOSS"
            q = pos.get("question", "")[:50]
            print(f"  [{tag}] {q}")
            print(f"       entry_edge={entry_edge:+.3f} -> new_edge={new_edge:+.3f}  "
                  f"price {entry_price:.3f}->{current_price:.3f}  P&L=${pnl:+.2f}")
        except Exception as e:
            still_open.append(pos)
            print(f"  [SKIP] {mid}: {str(e)[:60]}")

    state["open_positions"] = still_open
    state["equity"] = state["cash"] + sum(p["amount"] for p in still_open)
    save_state(state)


def resolve_positions(state: dict):
    """Check if any open positions have resolved."""
    # Run early-exit check before checking for natural resolution
    early_exit_open_positions(state)

    open_pos = state["open_positions"]
    if not open_pos:
        print("\n  No open positions to resolve.")
        return

    print(f"\n  Checking {len(open_pos)} open positions...")
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    still_open = []
    for pos in open_pos:
        mid = pos.get("market_id")
        if not mid:
            still_open.append(pos)
            continue

        try:
            resp = session.get(f"{GAMMA_API}/markets/{mid}", timeout=10)
            if resp.status_code != 200:
                still_open.append(pos)
                continue

            market = resp.json()
            if not market.get("closed"):
                still_open.append(pos)
                continue

            # Resolve
            outcome_str = market.get("outcome")
            if outcome_str == "Yes":
                outcome = 1.0
            elif outcome_str == "No":
                outcome = 0.0
            else:
                prices = market.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices and float(prices[0]) > 0.95:
                    outcome = 1.0
                elif prices and float(prices[0]) < 0.05:
                    outcome = 0.0
                else:
                    still_open.append(pos)
                    continue

            # Calculate P&L
            action = pos["action"]
            entry_price = pos["entry_price"]
            amount = pos["amount"]

            if action == "BUY_YES":
                shares = amount / entry_price
                payout = shares * outcome
                pnl = payout - amount
            elif action == "BUY_NO":
                shares = amount / (1 - entry_price)
                payout = shares * (1 - outcome)
                pnl = payout - amount
            else:
                pnl = 0
                payout = amount

            pos["outcome"] = outcome
            pos["pnl"] = round(pnl, 2)
            pos["payout"] = round(payout, 2)
            pos["resolved_at"] = datetime.now(timezone.utc).isoformat()
            pos["result"] = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAK_EVEN"

            state["closed_positions"].append(pos)
            state["cash"] += payout

            q = pos.get("question", "")[:50]
            tag = "WIN" if pnl > 0 else "LOSS"
            print(f"  [{tag}] {q}")
            print(f"       {action} ${amount:.2f} -> Outcome={'YES' if outcome==1 else 'NO'} "
                  f"P&L=${pnl:+.2f}")

        except Exception:
            still_open.append(pos)

    state["open_positions"] = still_open
    state["equity"] = state["cash"] + sum(p["amount"] for p in still_open)
    save_state(state)

    resolved = len(open_pos) - len(still_open)
    print(f"\n  Resolved: {resolved} | Still open: {len(still_open)}")


# =========================================================
# REPORT
# =========================================================

def compute_weight_analytics(closed: list[dict]) -> dict:
    """Compute analytics needed for capital weight allocation.

    Returns dict with:
    - model_pnl: Historical P&L by model
    - category_pnl: Accuracy by market category
    - edge_calibration: Predicted edge vs actual return by edge bucket
    """
    if not closed:
        return {"model_pnl": {}, "category_pnl": {}, "edge_calibration": {}}

    # 1. Historical P&L by model — which models make money
    model_pnl = {}
    for p in closed:
        trade_pnl = p.get("pnl", 0)
        trade_amount = p.get("amount", 1)
        trade_roi = trade_pnl / max(trade_amount, 0.01)

        for m in p.get("models", []):
            if m not in model_pnl:
                model_pnl[m] = {"total_pnl": 0, "trades": 0, "wins": 0,
                                "total_wagered": 0, "avg_edge": 0, "edges": []}
            model_pnl[m]["total_pnl"] += trade_pnl
            model_pnl[m]["trades"] += 1
            model_pnl[m]["total_wagered"] += trade_amount
            if trade_pnl > 0:
                model_pnl[m]["wins"] += 1
            model_pnl[m]["edges"].append(abs(p.get("edge", 0)))

        # Per-model estimate accuracy (how close was each model to the outcome)
        outcome = p.get("outcome")
        if outcome is not None:
            for mname, est in p.get("model_estimates", {}).items():
                if mname not in model_pnl:
                    model_pnl[mname] = {"total_pnl": 0, "trades": 0, "wins": 0,
                                        "total_wagered": 0, "avg_edge": 0, "edges": [],
                                        "brier_scores": []}
                if "brier_scores" not in model_pnl[mname]:
                    model_pnl[mname]["brier_scores"] = []
                model_pnl[mname]["brier_scores"].append((est - outcome) ** 2)

    for m in model_pnl:
        d = model_pnl[m]
        d["roi"] = round(d["total_pnl"] / max(d["total_wagered"], 0.01) * 100, 2)
        d["win_rate"] = round(d["wins"] / max(d["trades"], 1) * 100, 1)
        d["avg_edge"] = round(float(np.mean(d["edges"])) if d["edges"] else 0, 4)
        if d.get("brier_scores"):
            d["brier"] = round(float(np.mean(d["brier_scores"])), 4)
        d.pop("edges", None)
        d.pop("brier_scores", None)
        d["total_pnl"] = round(d["total_pnl"], 2)
        d["total_wagered"] = round(d["total_wagered"], 2)

    # 2. Accuracy by market category
    category_pnl = {}
    for p in closed:
        cat = p.get("category", "other")
        if cat not in category_pnl:
            category_pnl[cat] = {"total_pnl": 0, "trades": 0, "wins": 0,
                                 "total_wagered": 0, "brier_scores": []}
        category_pnl[cat]["total_pnl"] += p.get("pnl", 0)
        category_pnl[cat]["trades"] += 1
        category_pnl[cat]["total_wagered"] += p.get("amount", 0)
        if p.get("pnl", 0) > 0:
            category_pnl[cat]["wins"] += 1
        outcome = p.get("outcome")
        if outcome is not None:
            pred = p.get("predicted_prob", 0.5)
            category_pnl[cat]["brier_scores"].append((pred - outcome) ** 2)

    for cat in category_pnl:
        d = category_pnl[cat]
        d["roi"] = round(d["total_pnl"] / max(d["total_wagered"], 0.01) * 100, 2)
        d["win_rate"] = round(d["wins"] / max(d["trades"], 1) * 100, 1)
        if d["brier_scores"]:
            d["brier"] = round(float(np.mean(d["brier_scores"])), 4)
        d.pop("brier_scores", None)
        d["total_pnl"] = round(d["total_pnl"], 2)
        d["total_wagered"] = round(d["total_wagered"], 2)

    # 3. Edge calibration — when we say X% edge, do we actually make X%?
    # Bucket trades by predicted edge size
    edge_buckets = {
        "tiny_1-2%": (0.01, 0.02),
        "small_2-5%": (0.02, 0.05),
        "medium_5-10%": (0.05, 0.10),
        "large_10-20%": (0.10, 0.20),
        "huge_20%+": (0.20, 1.00),
    }
    edge_calibration = {}
    for bucket_name, (lo, hi) in edge_buckets.items():
        bucket_trades = [p for p in closed if lo <= abs(p.get("edge", 0)) < hi]
        if not bucket_trades:
            continue
        predicted_edges = [abs(p.get("edge", 0)) for p in bucket_trades]
        actual_rois = [p.get("pnl", 0) / max(p.get("amount", 1), 0.01) for p in bucket_trades]
        edge_calibration[bucket_name] = {
            "trades": len(bucket_trades),
            "avg_predicted_edge": round(float(np.mean(predicted_edges)) * 100, 2),
            "avg_actual_roi": round(float(np.mean(actual_rois)) * 100, 2),
            "calibration_ratio": round(
                float(np.mean(actual_rois)) / max(float(np.mean(predicted_edges)), 0.001), 2),
            "win_rate": round(sum(1 for p in bucket_trades if p.get("pnl", 0) > 0)
                              / len(bucket_trades) * 100, 1),
            "total_pnl": round(sum(p.get("pnl", 0) for p in bucket_trades), 2),
        }

    return {
        "model_pnl": model_pnl,
        "category_pnl": category_pnl,
        "edge_calibration": edge_calibration,
    }


def save_analytics(analytics: dict):
    """Save weight analytics to file for capital allocation decisions."""
    analytics["updated_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ANALYTICS_FILE, "w") as f:
        json.dump(analytics, f, indent=2, default=str)


def report(state: dict):
    """Full P&L report with weight allocation analytics."""
    closed = state["closed_positions"]
    open_pos = state["open_positions"]

    deployed = sum(p["amount"] for p in open_pos)
    equity = state["cash"] + deployed

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING REPORT")
    print(f"{'='*70}")
    print(f"  Started:          {state.get('started_at', '?')[:19]}")
    print(f"  Initial equity:   ${state['initial_equity']:,.2f}")
    print(f"  Current equity:   ${equity:,.2f}")
    print(f"  Cash:             ${state['cash']:,.2f}")
    print(f"  Deployed:         ${deployed:,.2f}")
    print(f"  Total return:     {(equity/state['initial_equity']-1)*100:+.2f}%")
    print(f"  Total trades:     {state['total_trades']}")
    print(f"  Open positions:   {len(open_pos)}")
    print(f"  Closed positions: {len(closed)}")

    if closed:
        # Defensive .get(): one ancient record from early dev had no pnl key.
        wins = [p for p in closed if p.get("pnl", 0) > 0]
        losses = [p for p in closed if p.get("pnl", 0) < 0]
        total_pnl = sum(p.get("pnl", 0) for p in closed)
        total_wagered = sum(p.get("amount", 0) for p in closed)

        print(f"\n  CLOSED TRADES:")
        print(f"    Wins:           {len(wins)}")
        print(f"    Losses:         {len(losses)}")
        print(f"    Win rate:       {len(wins)/len(closed)*100:.1f}%")
        print(f"    Total P&L:      ${total_pnl:+.2f}")
        print(f"    ROI:            {total_pnl/max(total_wagered,0.01)*100:+.1f}%")

        if wins:
            print(f"    Avg win:        ${np.mean([p['pnl'] for p in wins]):+.2f}")
        if losses:
            print(f"    Avg loss:       ${np.mean([p['pnl'] for p in losses]):+.2f}")

        # ===== WEIGHT ALLOCATION ANALYTICS =====
        analytics = compute_weight_analytics(closed)

        # --- 1. Historical P&L by Model ---
        model_pnl = analytics["model_pnl"]
        if model_pnl:
            print(f"\n  MODEL P&L (for weight allocation):")
            print(f"    {'Model':<20} {'P&L':>8} {'ROI':>7} {'WR':>5} {'Trades':>6} {'Brier':>7}")
            print(f"    {'-'*55}")
            for m in sorted(model_pnl, key=lambda x: -model_pnl[x]["total_pnl"]):
                d = model_pnl[m]
                brier_str = f"{d['brier']:.4f}" if "brier" in d else "  n/a"
                print(f"    {m:<20} ${d['total_pnl']:>+7.2f} {d['roi']:>+6.1f}% "
                      f"{d['win_rate']:>4.0f}% {d['trades']:>5} {brier_str:>7}")

        # --- 2. Accuracy by Market Category ---
        cat_pnl = analytics["category_pnl"]
        if cat_pnl:
            print(f"\n  CATEGORY P&L (where do we have edge?):")
            print(f"    {'Category':<15} {'P&L':>8} {'ROI':>7} {'WR':>5} {'Trades':>6} {'Brier':>7}")
            print(f"    {'-'*55}")
            for cat in sorted(cat_pnl, key=lambda x: -cat_pnl[x]["total_pnl"]):
                d = cat_pnl[cat]
                brier_str = f"{d['brier']:.4f}" if "brier" in d else "  n/a"
                print(f"    {cat:<15} ${d['total_pnl']:>+7.2f} {d['roi']:>+6.1f}% "
                      f"{d['win_rate']:>4.0f}% {d['trades']:>5} {brier_str:>7}")

        # --- 3. Edge Calibration ---
        edge_cal = analytics["edge_calibration"]
        if edge_cal:
            print(f"\n  EDGE CALIBRATION (predicted edge vs actual ROI):")
            print(f"    {'Edge Bucket':<15} {'Pred':>6} {'Actual':>7} {'Ratio':>6} {'WR':>5} {'Trades':>6} {'P&L':>8}")
            print(f"    {'-'*60}")
            for bucket in sorted(edge_cal.keys()):
                d = edge_cal[bucket]
                label = bucket.replace("_", " ")
                print(f"    {label:<15} {d['avg_predicted_edge']:>5.1f}% {d['avg_actual_roi']:>+6.1f}% "
                      f"{d['calibration_ratio']:>5.2f}x {d['win_rate']:>4.0f}% "
                      f"{d['trades']:>5} ${d['total_pnl']:>+7.2f}")

            # Summary insight
            total_ratio = sum(d["calibration_ratio"] * d["trades"] for d in edge_cal.values())
            total_n = sum(d["trades"] for d in edge_cal.values())
            if total_n > 0:
                avg_ratio = total_ratio / total_n
                if avg_ratio > 1.1:
                    print(f"\n    >>> Edges UNDER-predicted — system is better than it thinks")
                elif avg_ratio < 0.5:
                    print(f"\n    >>> Edges OVER-predicted — system thinks it has more edge than it does")
                    print(f"        Consider tightening entry gates or reducing position sizes")
                else:
                    print(f"\n    >>> Edge calibration looks reasonable (avg ratio={avg_ratio:.2f}x)")

        # Save analytics for weight allocation
        save_analytics(analytics)
        print(f"\n  Analytics saved to {ANALYTICS_FILE}")

        # Recent trades
        print(f"\n  RECENT CLOSED TRADES:")
        for p in closed[-10:]:
            q = p.get("question", "")[:40]
            tag = "WIN" if p["pnl"] > 0 else "LOSS"
            cat = p.get("category", "?")[:8]
            print(f"    [{tag:>4}] [{cat:<8}] {q} | {p['action']} ${p['amount']:.2f} -> ${p['pnl']:+.2f}")

    else:
        print("\n  No closed trades yet — analytics need resolved trades.")
        print("  Run --resolve after markets close to start collecting data.")

    if open_pos:
        print(f"\n  OPEN POSITIONS:")
        for p in open_pos:
            q = p.get("question", "")[:40]
            cat = p.get("category", classify_market(p.get("question", "")))
            models = ",".join(p.get("models", []))
            print(f"    [{cat:<10}] {q}")
            print(f"      {p['action']} ${p['amount']:.2f} | Edge={p['edge']:+.3f} | {models}")
            if p.get("ai_reasoning"):
                print(f"      AI: {p['ai_reasoning'][:65]}".encode("ascii", "replace").decode("ascii"))


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Trader")
    parser.add_argument("--scan", action="store_true", help="Scan and place paper trades")
    parser.add_argument("--resolve", action="store_true", help="Check resolutions")
    parser.add_argument("--report", action="store_true", help="Show P&L report")
    parser.add_argument("--equity", type=float, default=None, help="Reset with new equity")
    parser.add_argument("-n", type=int, default=30, help="Markets to scan")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    state = load_state()

    if args.equity is not None:
        state = {
            "initial_equity": args.equity,
            "equity": args.equity,
            "cash": args.equity,
            "open_positions": [],
            "closed_positions": [],
            "total_trades": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_scan": None,
        }
        save_state(state)
        print(f"  Reset paper account with ${args.equity:,.2f}")

    if args.scan:
        scan_and_trade(state, args.n)
    elif args.resolve:
        resolve_positions(state)
    elif args.report:
        report(state)
    else:
        # Full cycle
        scan_and_trade(state, args.n)
        resolve_positions(state)
        report(state)


if __name__ == "__main__":
    main()
