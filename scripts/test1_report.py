"""TEST 1 — Compounded Equity Report

Replays the resolved Test 1 ledger chronologically at multiple Kelly fractions
(1/8x, 1/5x, 1/4x, 1/3x, 1/2x, 1x). Each fraction runs as its own compounding
$10,000 book — bet size at each step depends on the CURRENT equity, not a fixed
notional. This is the only way to honestly compare allocation strategies.

Kelly determination is identical to Test 0. Only the multiplier changes.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shadow_ledger import (
    KELLY_FRACTIONS,
    DEFAULT_NOTIONAL,
    replay_kelly_fraction,
    replay_per_model_kelly_fraction,
)

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "test1_ledger.jsonl")


def load_all() -> list:
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


def fmt_money(v):
    return f"${v:>10,.2f}"


def fmt_pct(v):
    return f"{v:>+7.2f}%"


def report():
    records = load_all()
    if not records:
        print("\n  TEST 1 ledger empty.\n")
        return

    resolved = [r for r in records if r.get("resolved")]
    unresolved = [r for r in records if not r.get("resolved")]

    print("\n" + "=" * 78)
    print("  TEST 1 — COMPOUNDED EQUITY REPORT")
    print("=" * 78)
    print(f"  Total records:       {len(records)}")
    print(f"  Resolved:            {len(resolved)}")
    print(f"  Unresolved:          {len(unresolved)}")

    markets = set(r.get("market_id") for r in records)
    rounds = defaultdict(int)
    for r in records:
        rounds[r.get("prediction_round", 1)] += 1
    print(f"  Unique markets:      {len(markets)}")
    print(f"  Prediction rounds:   " + ", ".join(
        f"R{k}={v}" for k, v in sorted(rounds.items())
    ))

    if not resolved:
        print("\n  No resolved records yet — wait for markets to close.\n")
        return

    print(f"\n  Starting equity:     ${DEFAULT_NOTIONAL:,.2f}")

    # ----------------------------------------------------------------
    # Replay each Kelly fraction with compounding
    # ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  COMPOUNDED EQUITY BY KELLY FRACTION  (all resolved trades)")
    print("-" * 78)
    print(f"  {'Fraction':<14} {'Trades':>7} {'Win%':>6} "
          f"{'Final $':>13} {'ROI':>9} {'MaxDD':>8} {'Ruin':>6}")
    print("  " + "-" * 70)

    results = {}
    for label, fraction in KELLY_FRACTIONS.items():
        res = replay_kelly_fraction(resolved, fraction)
        results[label] = res
        ruin_tag = "YES" if res["ruin"] else "no"
        print(f"  {label:<14} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
              f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])} "
              f"{res['max_drawdown_pct']:>6.1f}% {ruin_tag:>6}")

    # Best by ROI
    best = max(results, key=lambda k: results[k]["roi_pct"])
    print(f"\n  BEST ROI:  {best}  ({fmt_pct(results[best]['roi_pct'])}, "
          f"final {fmt_money(results[best]['final_equity'])})")

    # ----------------------------------------------------------------
    # Per-category breakdown at the default 1/3x
    # ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  CATEGORY BREAKDOWN  (replay at kelly_1/3x)")
    print("-" * 78)
    by_cat = defaultdict(list)
    for r in resolved:
        by_cat[r.get("category", "other")].append(r)
    if by_cat:
        print(f"  {'Category':<14} {'Trades':>7} {'Win%':>6} "
              f"{'Final $':>13} {'ROI':>9}")
        print("  " + "-" * 53)
        cat_results = []
        for cat, recs in by_cat.items():
            res = replay_kelly_fraction(recs, 1.0 / 3.0)
            cat_results.append((cat, res))
        cat_results.sort(key=lambda x: -x[1]["roi_pct"])
        for cat, res in cat_results:
            print(f"  {cat:<14} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
                  f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])}")

    # ----------------------------------------------------------------
    # Per-model standalone (each sub-model trading on its own at 1/3 Kelly)
    # ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  PER-MODEL SOLO REPLAY  (each model alone at kelly_1/3x)")
    print("-" * 78)
    model_names = set()
    for r in resolved:
        for name in (r.get("per_model_shadow") or {}).keys():
            model_names.add(name)
    if model_names:
        print(f"  {'Model':<18} {'Trades':>7} {'Win%':>6} "
              f"{'Final $':>13} {'ROI':>9}")
        print("  " + "-" * 57)
        model_results = []
        for name in model_names:
            res = replay_per_model_kelly_fraction(resolved, name, 1.0 / 3.0)
            model_results.append((name, res))
        model_results.sort(key=lambda x: -x[1]["roi_pct"])
        for name, res in model_results:
            print(f"  {name:<18} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
                  f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])}")
    else:
        print("  (no per-model data yet)")

    # ----------------------------------------------------------------
    # Edge confidence breakdown at 1/3x
    # ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  EDGE CONFIDENCE BREAKDOWN  (replay at kelly_1/3x)")
    print("-" * 78)
    by_conf = defaultdict(list)
    for r in resolved:
        by_conf[r.get("edge_confidence", "?")].append(r)
    if by_conf:
        print(f"  {'Confidence':<14} {'Trades':>7} {'Win%':>6} "
              f"{'Final $':>13} {'ROI':>9}")
        print("  " + "-" * 53)
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            recs = by_conf.get(conf, [])
            if not recs:
                continue
            res = replay_kelly_fraction(recs, 1.0 / 3.0)
            print(f"  {conf:<14} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
                  f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])}")

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    report()
