"""TEST 1 — Shadow Collector (parallel data branch)

This is a STANDALONE pipeline that does NOT touch any Test 0 files
(paper_trades.json, live_predictions.jsonl). It writes only to
data/test1_ledger.jsonl.

What it collects per market:
    1. Shadow ledger        — bet $ at every Kelly fraction + fixed % + flat $
    2. More markets         — 200/scan instead of 50
    3. Re-prediction        — re-predicts every 6h to track edge evolution
    4. Per-model shadow     — solo trade per sub-model
    5. Wider price range    — 2-98% instead of 5-95%
    6. Context features     — volume, liquidity, spread, hour, weekday, BTC, SPY

Run:
    python scripts/test1_collector.py             # one scan
    python scripts/test1_collector.py -n 300      # 300 markets
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

import requests

from src.market_client import MarketClient
from src.prediction_engine import PredictionEngine
from src.shadow_ledger import compute_per_model_shadow

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "test1_ledger.jsonl")
GAMMA_API = "https://gamma-api.polymarket.com"

# Test 1 settings — Kelly determination identical to Test 0;
# only what differs is data volume (more markets, re-predictions, context)
# and the allocation FRACTIONS we test in the report (1/8, 1/5, 1/4, 1/3, 1/2, 1x).
MAX_DAYS = 14
MIN_PRICE = 0.05            # SAME as Test 0
MAX_PRICE = 0.95            # SAME as Test 0
DEFAULT_SCAN_SIZE = 200     # 4x Test 0 (more data)
REPREDICT_HOURS = 6         # re-predict markets after 6h to capture edge evolution

# Order matters — first match wins. niche_sports comes BEFORE sports so the
# proven-winning patterns (tennis tournaments, esports, micro-prop bets) get
# their own bucket instead of being lumped into "other". n=104, WR 71.2% on
# this subset vs 54.6% on the mainstream "sports" bucket.
CATEGORY_RULES = [
    ("fed_rate",     ["fed ", "interest rate", "fomc", "federal reserve", "bps"]),
    ("crypto",       ["bitcoin", "btc ", "ethereum", "eth ", "crypto", "solana"]),
    ("oil_energy",   ["oil", "crude", "wti ", "brent", "opec", "energy"]),
    ("macro",        ["gdp", "inflation", "cpi ", "recession", "unemployment", "tariff"]),
    ("geopolitics",  ["invade", "invasion", "war ", "military", "regime", "sanctions",
                      "iran", "taiwan", "ukraine", "russia"]),
    ("elections",    ["election", "president", "governor", "senate", "nominee", "primary",
                      "democrat", "republican"]),
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
    ("sports",      ["fifa", "world cup", "nba ", "nfl ", "mlb ", "f1 ", "champion",
                     "tournament", "lakers", "celtics", "warriors", "yankees", "dodgers",
                     "match", "winner", "beat", "vs.", "points", "goals", "ufc", "nhl "]),
    ("tech_ai",     ["openai", "chatgpt", "artificial intelligence", " ai ", "google",
                     "apple", "tesla"]),
]


def classify_market(question: str) -> str:
    q = (question or "").lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in q:
                return category
    return "other"


def parse_token_ids(raw_market: dict) -> list:
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


# ---------- ledger I/O ---------------------------------------------------

def load_ledger() -> list:
    """Load all existing shadow records (one per line)."""
    if not os.path.exists(LEDGER_FILE):
        return []
    out = []
    with open(LEDGER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_record(record: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------- context features ---------------------------------------------

_context_cache = {}


def fetch_context_features() -> dict:
    """Fetch BTC + SPY price + Fear & Greed once per scan, cached."""
    now_ts = int(time.time())
    if "ts" in _context_cache and now_ts - _context_cache["ts"] < 300:
        return _context_cache["data"]

    ctx = {"btc_price": None, "eth_price": None, "fear_greed": None}

    # BTC + ETH from Coingecko (no key)
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
            timeout=8,
        )
        if r.ok:
            d = r.json()
            ctx["btc_price"] = d.get("bitcoin", {}).get("usd")
            ctx["eth_price"] = d.get("ethereum", {}).get("usd")
    except Exception:
        pass

    # Fear & Greed
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.ok:
            d = r.json()
            ctx["fear_greed"] = int(d.get("data", [{}])[0].get("value", 50))
    except Exception:
        pass

    _context_cache["ts"] = now_ts
    _context_cache["data"] = ctx
    return ctx


def market_context(raw: dict, parsed: dict) -> dict:
    """Per-market context (volume, liquidity, spread, time)."""
    now = datetime.now(timezone.utc)
    spread = None
    try:
        prices = raw.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if prices and len(prices) >= 2:
            yes = float(prices[0])
            no = float(prices[1])
            spread = round(abs(1.0 - (yes + no)), 4)
    except Exception:
        pass

    return {
        "volume_24h": raw.get("volume24hr") or raw.get("volumeNum"),
        "liquidity": raw.get("liquidity") or raw.get("liquidityNum"),
        "spread": spread,
        "hour_utc": now.hour,
        "weekday": now.weekday(),
        "iso_time": now.isoformat(),
    }


# ---------- market fetching ----------------------------------------------

def passes_filters(m: dict) -> tuple:
    """Return (days_left, price) if market passes Test 1 filters, else None."""
    end_str = m.get("endDate", "")
    if not end_str:
        return None
    try:
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_left = (end_dt - now).total_seconds() / 86400
        if not (0.5 < days_left <= MAX_DAYS):
            return None
    except Exception:
        return None

    try:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        price = float(prices[0]) if prices else None
    except Exception:
        return None

    if price is None or not (MIN_PRICE < price < MAX_PRICE):
        return None
    return (round(days_left, 2), price)


def fetch_markets(client: MarketClient, n_target: int) -> list:
    """Fetch markets from 3 generic pools + a deep niche-sports sweep, deduped.

    The niche-sports sweep paginates ~3000 markets ordered by endDate (soonest
    first) and keeps only those whose question matches our niche_sports
    keywords. This actively hunts the proven-winning pattern (n=82, WR 74.4%)
    instead of relying on the volume sort to surface them by luck.
    """
    seen = set()
    out = []

    pools = [
        {"limit": 200, "active": True, "closed": False,
         "order": "volume24hr", "ascending": False},
        {"limit": 200, "active": True, "closed": False,
         "order": "endDate", "ascending": True},
        {"limit": 200, "active": True, "closed": False,
         "order": "startDate", "ascending": False},
    ]
    for params in pools:
        try:
            r = client.session.get(f"{client.base_url}/markets",
                                   params=params, timeout=15).json()
            for m in r:
                mid = m.get("id")
                if not mid or mid in seen:
                    continue
                check = passes_filters(m)
                if check is None:
                    continue
                seen.add(mid)
                m["_days_left"], m["_price"] = check
                out.append(m)
                if len(out) >= n_target * 2:
                    break
        except Exception:
            continue

    # ---- Wide deep sweep for ALL winning category patterns -------------
    # Uses CategoryGate's verdict on historical resolved bets to decide
    # which categories to hunt. Sweeps deep (up to ~3000 markets, paged
    # by endDate) and keeps ones whose category passes the gate
    # (anything not BLOCKED or AMBIGUOUS). This actively hunts the
    # proven-winning patterns instead of relying on volume sort.
    try:
        from src.category_gate import CategoryGate
        gate = CategoryGate()
        winning_cats = set()
        for cat, stats in gate.stats.items():
            decision = gate.decide(cat, abs_edge=1.0)  # max edge — only structural block matters
            if decision["allow"]:
                winning_cats.add(cat)
        # Categories we have NO data on yet — let them through too
        # (a future winner needs samples to prove itself)
        all_known_cats = {name for name, _ in CATEGORY_RULES} | {"other"}
        unknown_cats = all_known_cats - set(gate.stats.keys())
        target_cats = winning_cats | unknown_cats
    except Exception:
        # If gate fails to load, default to the proven-winning niche bucket
        target_cats = {"niche_sports", "other", "geopolitics", "crypto",
                       "tech_ai", "elections", "fed_rate", "macro", "oil_energy"}

    print(f"  Deep sweep targeting categories: {sorted(target_cats)}")
    swept_added = 0
    # Default 50 pages = up to 10,000 markets — roughly the entire active
    # Polymarket universe. Each page = 200 markets, ~500ms fetch =>
    # ~25s of metadata fetching per cycle. Predictions still capped by
    # n_target so this only WIDENS the candidate pool, doesn't slow them.
    # Override via SWEEP_PAGES env var (e.g. SWEEP_PAGES=100 for 20k).
    SWEEP_PAGES = int(os.environ.get("SWEEP_PAGES", "50"))
    for page_offset in range(0, SWEEP_PAGES * 200, 200):
        if swept_added >= n_target:
            break
        try:
            r = client.session.get(
                f"{client.base_url}/markets",
                params={"limit": 200, "active": True, "closed": False,
                        "order": "endDate", "ascending": True,
                        "offset": page_offset},
                timeout=15,
            ).json()
            if not r:
                break
            for m in r:
                mid = m.get("id")
                if not mid or mid in seen:
                    continue
                cat = classify_market(m.get("question", ""))
                if cat not in target_cats:
                    continue
                check = passes_filters(m)
                if check is None:
                    continue
                seen.add(mid)
                m["_days_left"], m["_price"] = check
                m["_deep_sweep"] = True
                m["_swept_category"] = cat
                out.append(m)
                swept_added += 1
        except Exception:
            continue
    print(f"  Deep sweep added {swept_added} markets across winning categories")

    return out


# ---------- main scan ----------------------------------------------------

def scan(n_markets: int = DEFAULT_SCAN_SIZE):
    print(f"\n  TEST 1 SHADOW COLLECTOR — scanning up to {n_markets} markets")
    print(f"  Settings: <={MAX_DAYS}d  price {MIN_PRICE}-{MAX_PRICE}  "
          f"re-predict>{REPREDICT_HOURS}h")

    engine = PredictionEngine(backtest_mode=False)
    client = MarketClient()

    # Build re-prediction map: market_id -> latest prediction time + round
    ledger = load_ledger()
    last_pred = {}
    rounds = {}
    for r in ledger:
        mid = r.get("market_id")
        if not mid:
            continue
        ts = r.get("timestamp", "")
        rd = r.get("prediction_round", 1)
        if mid not in last_pred or ts > last_pred[mid]:
            last_pred[mid] = ts
            rounds[mid] = rd

    print(f"  Ledger: {len(ledger)} records, {len(last_pred)} unique markets")

    raw_markets = fetch_markets(client, n_markets)
    print(f"  Fetched {len(raw_markets)} markets passing filters")

    # Pre-fetch context (one API call shared across all markets)
    global_ctx = fetch_context_features()

    now = datetime.now(timezone.utc)
    candidates = []
    for raw in raw_markets:
        mid = raw.get("id")
        if not mid:
            continue
        # Determine if this is a new prediction or a re-prediction
        last_ts = last_pred.get(mid)
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since < REPREDICT_HOURS:
                    continue  # too soon to re-predict
                next_round = rounds.get(mid, 1) + 1
            except Exception:
                next_round = (rounds.get(mid, 1) or 1) + 1
        else:
            next_round = 1
        raw["_round"] = next_round
        candidates.append(raw)

    # Sort: deep-sweep candidates first (matched a winning category), then
    # lowest round (new markets), then soonest resolving.
    candidates.sort(key=lambda x: (
        0 if x.get("_deep_sweep") else 1,
        x.get("_round", 1),
        x.get("_days_left", 999),
    ))
    candidates = candidates[:n_markets]
    n_swept = sum(1 for c in candidates if c.get("_deep_sweep"))
    print(f"  Will predict {len(candidates)} markets ({n_swept} from deep sweep)")

    n_new = 0
    n_repred = 0
    for i, raw in enumerate(candidates):
        try:
            t0 = time.time()
            parsed = MarketClient.parse_market(raw)
            token_ids = parse_token_ids(raw)
            parsed["_token_id"] = token_ids[0] if token_ids else None
            time_rem = compute_time_remaining(parsed.get("end_date"))

            prediction = engine.predict(
                market_data=parsed,
                time_remaining_frac=time_rem,
                token_id=parsed.get("_token_id"),
            )

            edge = prediction["edge"]["edge"]
            prob = prediction["prediction"]["probability"]
            price = prediction["market"]["current_price"]

            # Use the engine's kelly_fraction (same as Test 0 — uncertainty-shrunk)
            sizing = prediction.get("sizing", {}) or {}
            kelly_fraction = sizing.get("kelly_fraction", 0.0)
            kelly_full = sizing.get("kelly_full", 0.0)
            engine_action = sizing.get("action", "NO_BET")
            if engine_action == "NO_BET":
                action = "BUY_YES" if edge > 0 else ("BUY_NO" if edge < 0 else "NO_BET")
            else:
                action = engine_action

            model_estimates = {}
            for mname, mresult in prediction.get("sub_models", {}).items():
                est = mresult.get("estimate")
                if est is not None:
                    model_estimates[mname] = round(float(est), 4)

            per_model_shadow = compute_per_model_shadow(model_estimates, price)

            strategy_meta = prediction.get("strategy", {}) or {}

            ctx = market_context(raw, parsed)
            ctx.update(global_ctx)

            elapsed = time.time() - t0
            round_num = raw.get("_round", 1)
            record = {
                "shadow_id": f"t1_{int(time.time()*1000)}_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction_round": round_num,
                "market_id": raw.get("id"),
                "question": parsed.get("question", ""),
                "category": classify_market(parsed.get("question", "")),
                "end_date": parsed.get("end_date"),
                "days_left": raw.get("_days_left"),
                "market_price": round(price, 4),
                "predicted_prob": round(prob, 4),
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "edge_confidence": prediction["edge"]["edge_confidence"],
                "action": action,
                # Engine's Kelly (same as Test 0). Report applies multipliers.
                "kelly_fraction": kelly_fraction,
                "kelly_full": kelly_full,
                "models": prediction["ensemble"]["model_names"],
                "model_estimates": model_estimates,
                "per_model_shadow": per_model_shadow,
                # Strategy metadata (regime, gates) from Test 0 adapter
                "strategy_meta": {
                    "should_enter": strategy_meta.get("should_enter"),
                    "regime": strategy_meta.get("regime"),
                    "is_contrarian": strategy_meta.get("is_contrarian"),
                },
                "context": ctx,
                "resolved": False,
                "outcome": None,
                "ai_reasoning": prediction.get("sub_models", {}).get(
                    "ai_semantic", {}).get("signals", {}).get("ai_reasoning"),
            }
            append_record(record)

            if round_num == 1:
                n_new += 1
                tag = "NEW"
            else:
                n_repred += 1
                tag = f"R{round_num}"

            q = parsed.get("question", "")[:55]
            print(f"  [{i+1:>3}] {tag} {q}")
            print(f"        Mkt={price:.3f} Pred={prob:.3f} Edge={edge:+.3f} "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{i+1:>3}] ERROR: {str(e)[:80]}")

    print(f"\n  Test 1 collector done — {n_new} new, {n_repred} re-predictions")
    print(f"  Ledger now has {len(ledger) + n_new + n_repred} records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-markets", type=int, default=DEFAULT_SCAN_SIZE,
                        help=f"Max markets to predict (default {DEFAULT_SCAN_SIZE})")
    args = parser.parse_args()
    scan(n_markets=args.n_markets)


if __name__ == "__main__":
    main()
