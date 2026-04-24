"""Time Series Backtest — Honest, one-shot experiment.

Answers one question: does the TimeSeriesModel (HMM + GARCH + Hurst) have
any measurable edge over the market price on resolved Polymarket markets?

This model has never been backtested before because it needs 20+ historical
price bars per market, and the live snapshot system only captures every 2h
(so markets need 40+ hours of observation before it activates). Polymarket
exposes tick-level trades via data-api.polymarket.com/trades, which we bucket
into 2-hour bars (matching the live snapshot cadence) to produce honest per-
market history. Free, no auth, no 12h cap like /prices-history has on closed
markets.

HONEST RULES (no leakage):
  - At prediction time T, we only pass the model price bars with timestamp <= T
  - current_price is the last bar <= T (not the resolution outcome)
  - time_remaining_frac is computed from T and the market end date
  - The resolution outcome is ONLY used for scoring, never fed to the model
  - HMM/GARCH refit from scratch per market (no cross-market state)

Usage:
    python scripts/backtest_time_series.py             # 500 markets
    python scripts/backtest_time_series.py -n 1000     # custom count
    python scripts/backtest_time_series.py --report    # summary only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
import numpy as np

from src.prediction_engine import TimeSeriesModel

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "backtest_time_series.jsonl")
SUMMARY_FILE = os.path.join(DATA_DIR, "backtest_time_series_summary.json")

MIN_BARS = 20            # TimeSeriesModel requires 20+ bars
PREDICT_OFFSET_H = 24    # Make prediction 24h before market resolution
MIN_MARKET_DAYS = 5      # Need enough market life for 20 bars at 2h cadence (>= ~42h)
BAR_SECONDS = 2 * 3600   # 2h bars — matches the live snapshot cadence
MIN_VOLUME = 5000.0      # Skip illiquid markets with too few trades to bar


CATEGORY_RULES = [
    ("fed_rate",    ["fed ", "interest rate", "fomc", "federal reserve", "bps"]),
    ("crypto",      ["bitcoin", "btc ", "ethereum", "eth ", "crypto", "solana"]),
    ("oil_energy",  ["oil", "crude", "wti ", "brent", "opec", "energy"]),
    ("macro",       ["gdp", "inflation", "cpi ", "recession", "unemployment", "tariff"]),
    ("geopolitics", ["invade", "invasion", "war ", "military", "regime", "sanctions",
                     "iran", "taiwan", "ukraine", "russia"]),
    ("elections",   ["election", "president", "governor", "senate", "nominee", "primary"]),
    ("sports",      ["fifa", "world cup", "nba ", "nfl ", "mlb ", "f1 ", "champion",
                     "tournament", "match", "winner", "beat", "vs.", "ufc", "nhl "]),
    ("tech_ai",     ["openai", "chatgpt", " ai ", "google", "apple", "tesla"]),
    ("weather",     ["temperature", "weather", "rain", "snow", "hurricane"]),
]


def classify(q: str) -> str:
    ql = q.lower()
    for cat, kws in CATEGORY_RULES:
        if any(k in ql for k in kws):
            return cat
    return "other"


def fetch_resolved_markets(target: int, session: requests.Session) -> list:
    """Pull resolved binary markets sorted by volume. Return the ones long-lived
    enough to have 20+ bars at 12h fidelity and with a non-extreme pre-price."""
    print(f"  Fetching resolved markets (target {target})...")
    out = []
    offset = 0
    skipped = defaultdict(int)

    while len(out) < target:
        try:
            r = session.get(f"{GAMMA_API}/markets", params={
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }, timeout=20)
            if r.status_code != 200:
                time.sleep(2); offset += 100; continue
            batch = r.json()
            if not batch:
                break

            for m in batch:
                # Volume filter first (cheap early out)
                vol = float(m.get("volume", 0) or 0)
                if vol < MIN_VOLUME:
                    skipped["low_volume"] += 1; continue

                # Parse outcome prices
                prices_raw = m.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    try: prices = json.loads(prices_raw)
                    except Exception: skipped["bad_prices"] += 1; continue
                else: prices = prices_raw
                try: fprices = [float(p) for p in prices]
                except Exception: skipped["unparseable"] += 1; continue
                if len(fprices) != 2 or max(fprices) < 0.95:
                    skipped["not_clearly_resolved"] += 1; continue

                winning_idx = fprices.index(max(fprices))
                actual = 1.0 if winning_idx == 0 else 0.0

                # conditionId — required for data-api /trades filter
                condition_id = m.get("conditionId")
                if not condition_id:
                    skipped["no_condition_id"] += 1; continue

                # Dates
                start_date = m.get("startDate") or m.get("createdAt")
                end_date = m.get("endDate")
                if not start_date or not end_date:
                    skipped["no_dates"] += 1; continue
                try:
                    t_start = datetime.fromisoformat(start_date.replace("Z", "+00:00")).timestamp()
                    t_end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
                except Exception:
                    skipped["bad_dates"] += 1; continue

                duration_days = (t_end - t_start) / 86400
                if duration_days < MIN_MARKET_DAYS:
                    skipped["too_short"] += 1; continue

                out.append({
                    "id": m.get("id"),
                    "question": m.get("question", ""),
                    "category": classify(m.get("question", "")),
                    "condition_id": condition_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "duration_days": round(duration_days, 2),
                    "actual": actual,
                    "volume": vol,
                })
                if len(out) >= target:
                    break

            offset += 100
            if len(out) and len(out) % 200 == 0:
                print(f"    ...{len(out)} collected (scanned {offset})")
            time.sleep(0.05)
        except Exception as e:
            print(f"    err at offset {offset}: {e}")
            time.sleep(2); offset += 100

    print(f"  Collected {len(out)} markets long enough for backtest")
    for k, v in sorted(skipped.items(), key=lambda x: -x[1])[:5]:
        print(f"    skipped {k}: {v}")
    return out


def fetch_trades(condition_id: str, session: requests.Session) -> list:
    """Pull every Yes-side trade for a resolved market via data-api /trades.

    Paginates with limit=500. Returns a list of {timestamp, price} dicts
    sorted oldest-first. Free, no auth, no 12h granularity cap.
    """
    trades = []
    offset = 0
    while True:
        try:
            r = session.get(f"{DATA_API}/trades", params={
                "market": condition_id,
                "limit": 500,
                "offset": offset,
            }, timeout=20)
        except Exception:
            break
        if r.status_code != 200:
            break
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        trades.extend(batch)
        if len(batch) < 500:
            break
        offset += 500
        if offset >= 10000:  # hard cap — 10k trades is plenty for bar fitting
            break

    # Keep only Yes-side trades, convert to (ts, price) tuples
    out = []
    for t in trades:
        if t.get("outcomeIndex") != 0:  # 0 = Yes, 1 = No
            continue
        try:
            ts = int(t["timestamp"])
            price = float(t["price"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0.0 < price < 1.0:
            out.append((ts, price))
    out.sort(key=lambda x: x[0])
    return out


def bucket_to_bars(trades: list, t_start: float, t_end: float,
                   bar_seconds: int = BAR_SECONDS) -> list:
    """Bucket tick-level (ts, price) trades into fixed time bars.

    Each bar's price is the last-trade price within [bar_start, bar_end).
    Bars with no trades inherit the previous bar's price (carry-forward).
    Returns a list of (bar_end_ts, price) tuples.
    """
    if not trades:
        return []
    n_bars = int((t_end - t_start) / bar_seconds)
    if n_bars < 1:
        return []

    bars = []
    last_price = None
    trade_idx = 0
    for i in range(n_bars):
        bar_start = t_start + i * bar_seconds
        bar_end = bar_start + bar_seconds
        # Advance through trades within this bar
        bar_price = None
        while trade_idx < len(trades) and trades[trade_idx][0] < bar_end:
            if trades[trade_idx][0] >= bar_start:
                bar_price = trades[trade_idx][1]
            trade_idx += 1
        if bar_price is not None:
            last_price = bar_price
        if last_price is not None:
            bars.append((bar_end, last_price))
    return bars


def run_market(market: dict, ts_model: TimeSeriesModel,
               session: requests.Session) -> dict | None:
    """Run TimeSeriesModel on one market at T = t_end - 24h.

    Pipeline: fetch all Yes-side trades for the market -> bucket into 2h bars ->
    slice to bars with end <= T -> feed to TimeSeriesModel -> score vs outcome.
    """
    condition_id = market["condition_id"]
    t_start = market["t_start"]
    t_end = market["t_end"]
    t_predict = t_end - PREDICT_OFFSET_H * 3600

    trades = fetch_trades(condition_id, session)
    if not trades:
        return {"skip_reason": "no_trades"}

    bars = bucket_to_bars(trades, t_start, t_end, BAR_SECONDS)
    if not bars:
        return {"skip_reason": "no_bars_built"}

    # Leakage guard — drop any bar whose window ends after prediction time
    bars_pre = [(ts, p) for (ts, p) in bars if ts <= t_predict]
    if len(bars_pre) < MIN_BARS:
        return {"skip_reason": f"too_few_bars_{len(bars_pre)}"}

    prices = [p for (_, p) in bars_pre]
    current_price = prices[-1]
    if current_price < 0.05 or current_price > 0.95:
        return {"skip_reason": "extreme_pre_price"}

    total_duration = t_end - t_start
    remaining = t_end - t_predict
    time_remaining_frac = max(0.001, min(1.0, remaining / total_duration))

    try:
        result = ts_model.analyze(prices, current_price, time_remaining_frac)
    except Exception as e:
        return {"skip_reason": f"model_error: {str(e)[:60]}"}

    estimate = result.get("estimate")
    if estimate is None:
        return {"skip_reason": "no_estimate"}

    actual = market["actual"]
    return {
        "id": market["id"],
        "question": market["question"][:120],
        "category": market["category"],
        "n_bars": len(bars_pre),
        "n_trades": len(trades),
        "duration_days": market["duration_days"],
        "pre_price": round(current_price, 4),
        "ts_estimate": round(float(estimate), 4),
        "actual": actual,
        "ts_brier": round((float(estimate) - actual) ** 2, 6),
        "market_brier": round((current_price - actual) ** 2, 6),
        "edge": round(float(estimate) - current_price, 4),
        "signals": {
            k: v for k, v in (result.get("signals") or {}).items()
            if k in ("hurst_exponent", "regime_type", "current_regime",
                     "garch_vol_1step", "uncertainty_std")
        },
    }


def load_done_ids() -> set:
    if not os.path.exists(OUT_FILE):
        return set()
    done = set()
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line).get("id"))
            except Exception:
                pass
    return done


def summarize():
    if not os.path.exists(OUT_FILE):
        print("  No results yet.")
        return
    records = []
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            try: records.append(json.loads(line))
            except Exception: pass
    if not records:
        print("  No valid records.")
        return

    n = len(records)
    ts_brier = sum(r["ts_brier"] for r in records) / n
    mkt_brier = sum(r["market_brier"] for r in records) / n
    edge = mkt_brier - ts_brier

    per_cat = defaultdict(lambda: {"ts": [], "mkt": []})
    for r in records:
        per_cat[r["category"]]["ts"].append(r["ts_brier"])
        per_cat[r["category"]]["mkt"].append(r["market_brier"])

    print("=" * 74)
    print("  TIME SERIES BACKTEST — SUMMARY")
    print("=" * 74)
    print(f"  Markets tested:      {n}")
    print(f"  TimeSeriesModel Brier: {ts_brier:.4f}")
    print(f"  Market baseline Brier: {mkt_brier:.4f}")
    print(f"  Edge vs market:        {edge:+.4f}  " +
          ("(time_series BEATS market)" if edge > 0.001
           else "(matches market)" if abs(edge) <= 0.001
           else "(time_series LOSES to market)"))
    print()
    print("  PER-CATEGORY (time_series Brier vs market Brier):")
    print(f"    {'category':<15} {'ts':>8} {'mkt':>8} {'edge':>9}  {'n':>5}")
    for cat in sorted(per_cat.keys(), key=lambda c: -len(per_cat[c]["ts"])):
        ts = per_cat[cat]["ts"]; mk = per_cat[cat]["mkt"]
        ts_b = sum(ts) / len(ts)
        mk_b = sum(mk) / len(mk)
        print(f"    {cat:<15} {ts_b:>8.4f} {mk_b:>8.4f} {mk_b - ts_b:>+9.4f}  {len(ts):>5}")
    print()

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_markets": n,
        "ts_brier": round(ts_brier, 6),
        "market_brier": round(mkt_brier, 6),
        "edge_vs_market": round(edge, 6),
        "per_category": {
            cat: {
                "n": len(v["ts"]),
                "ts_brier": round(sum(v["ts"]) / len(v["ts"]), 6),
                "market_brier": round(sum(v["mkt"]) / len(v["mkt"]), 6),
            }
            for cat, v in per_cat.items()
        },
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {SUMMARY_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=500,
                    help="Target number of markets to test")
    ap.add_argument("--report", action="store_true",
                    help="Print summary from existing results, no new fetching")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.report:
        summarize()
        return

    print("=" * 74)
    print("  TIME SERIES BACKTEST — honest, no leakage")
    print(f"  Prediction time T = market_end - {PREDICT_OFFSET_H}h")
    print(f"  Requires >= {MIN_BARS} price bars before T")
    print("=" * 74)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    # Resume support
    done_ids = load_done_ids()
    if done_ids:
        print(f"  Resume: {len(done_ids)} markets already in {OUT_FILE}")

    markets = fetch_resolved_markets(args.count + len(done_ids) + 200, session)
    markets = [m for m in markets if m["id"] not in done_ids][: args.count]
    if not markets:
        print("  Nothing new to test.")
        summarize()
        return

    print(f"  Running TimeSeriesModel on {len(markets)} markets...")
    ts_model = TimeSeriesModel()

    n_done = 0
    n_skipped = 0
    skip_reasons = defaultdict(int)

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for i, m in enumerate(markets, 1):
            res = run_market(m, ts_model, session)
            if res is None or "skip_reason" in (res or {}):
                n_skipped += 1
                if res and "skip_reason" in res:
                    skip_reasons[res["skip_reason"].split(":")[0]] += 1
                continue
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            n_done += 1
            if i % 25 == 0:
                print(f"    [{i}/{len(markets)}] done={n_done} skipped={n_skipped}")
            time.sleep(0.1)  # gentle on the CLOB API

    print()
    print(f"  Tested: {n_done}   Skipped: {n_skipped}")
    if skip_reasons:
        print("  Skip reasons:")
        for k, v in sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"    {k}: {v}")
    print()
    summarize()


if __name__ == "__main__":
    main()
