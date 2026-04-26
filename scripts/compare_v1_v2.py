"""Side-by-side comparison of v1 (test1_ledger) vs v2 (v2_ledger).

Both ledgers use the same prediction-engine outputs and the same Polymarket
market universe; only the ensemble blending and entry gate differ. Compare
on the intersection of market_ids + on the union, at multiple Kelly
fractions, to see whether v2 is actually better — not just lucky.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shadow_ledger import (
    KELLY_FRACTIONS, DEFAULT_NOTIONAL, replay_kelly_fraction,
)
from src.data_validator import load_validated_jsonl

DATA_DIR = "data"
V1_LEDGER = os.path.join(DATA_DIR, "test1_ledger.jsonl")
V2_LEDGER = os.path.join(DATA_DIR, "v2_ledger.jsonl")


def _summarize(records: list, label: str) -> dict:
    summary = {"label": label, "n_resolved_bets": len(records)}
    if not records:
        return summary
    for kelly_label, fraction in KELLY_FRACTIONS.items():
        res = replay_kelly_fraction(records, fraction)
        summary[kelly_label] = {
            "trades": res["n_trades"],
            "wins": res["wins"],
            "losses": res["losses"],
            "win_rate": round(res["win_rate_pct"], 2),
            "final_equity": round(res["final_equity"], 2),
            "roi_pct": round(res["roi_pct"], 2),
            "max_dd_pct": round(res["max_drawdown_pct"], 3),
        }
    return summary


def main():
    print("\n" + "=" * 78)
    print("  v1 vs v2 — SIDE BY SIDE")
    print("=" * 78)

    v1_recs, v1_stats = load_validated_jsonl(V1_LEDGER, drop_duplicates=True)
    v2_recs, v2_stats = load_validated_jsonl(V2_LEDGER, drop_duplicates=True)

    print("\n  Validation:")
    print(f"  v1 ({V1_LEDGER}):")
    print(v1_stats.report())
    print(f"  v2 ({V2_LEDGER}):")
    print(v2_stats.report())

    if not v1_recs and not v2_recs:
        print("\n  No data on either side. Nothing to compare.\n")
        return

    print(f"\n  v1 validated bets: {len(v1_recs)}")
    print(f"  v2 validated bets: {len(v2_recs)}")

    # Overlap on market_id (only relevant if v2 has run for a while)
    v1_by_mid = {r["market_id"]: r for r in v1_recs}
    v2_by_mid = {r["market_id"]: r for r in v2_recs}
    overlap = set(v1_by_mid) & set(v2_by_mid)
    print(f"  Overlap on market_id: {len(overlap)}")

    sum_v1 = _summarize(v1_recs, "v1 (test1)")
    sum_v2 = _summarize(v2_recs, "v2")

    print("\n" + "-" * 78)
    print(f"  ALL VALIDATED BETS  (starting equity ${DEFAULT_NOTIONAL:,.0f})")
    print("-" * 78)
    print(f"  {'Fraction':<14} {'v1 trades':>10} {'v1 ROI':>10} "
          f"{'v2 trades':>11} {'v2 ROI':>10} {'d_ROI':>10}")
    print("  " + "-" * 70)
    for label in KELLY_FRACTIONS:
        v1 = sum_v1.get(label) or {}
        v2 = sum_v2.get(label) or {}
        v1_roi = v1.get("roi_pct", 0.0)
        v2_roi = v2.get("roi_pct", 0.0)
        delta = v2_roi - v1_roi if (v1_recs and v2_recs) else 0.0
        print(f"  {label:<14} {v1.get('trades', 0):>10} {v1_roi:>+9.2f}% "
              f"{v2.get('trades', 0):>11} {v2_roi:>+9.2f}% "
              f"{delta:>+9.2f}%")

    # If overlap exists, replay both restricted to it
    if overlap:
        v1_overlap = [v1_by_mid[m] for m in overlap if v1_by_mid[m].get("resolved")]
        v2_overlap = [v2_by_mid[m] for m in overlap if v2_by_mid[m].get("resolved")]
        sum_v1_o = _summarize(v1_overlap, "v1 overlap")
        sum_v2_o = _summarize(v2_overlap, "v2 overlap")
        print("\n" + "-" * 78)
        print(f"  OVERLAPPING MARKETS ONLY  ({len(overlap)} markets)")
        print(f"  Apples-to-apples — same markets, different policies")
        print("-" * 78)
        print(f"  {'Fraction':<14} {'v1 ROI':>10} {'v2 ROI':>10} {'d_ROI':>10}")
        print("  " + "-" * 50)
        for label in KELLY_FRACTIONS:
            v1 = sum_v1_o.get(label) or {}
            v2 = sum_v2_o.get(label) or {}
            v1_roi = v1.get("roi_pct", 0.0)
            v2_roi = v2.get("roi_pct", 0.0)
            print(f"  {label:<14} {v1_roi:>+9.2f}% {v2_roi:>+9.2f}% "
                  f"{v2_roi - v1_roi:>+9.2f}%")

    # Per-category snapshot at 1/3x
    print("\n" + "-" * 78)
    print("  PER-CATEGORY (kelly_1/3x)")
    print("-" * 78)

    def cat_breakdown(records):
        by_cat = defaultdict(list)
        for r in records:
            by_cat[r.get("category", "other")].append(r)
        out = {}
        for cat, recs in by_cat.items():
            res = replay_kelly_fraction(recs, 1.0 / 3.0)
            out[cat] = res
        return out

    v1_cats = cat_breakdown(v1_recs)
    v2_cats = cat_breakdown(v2_recs)
    cats = sorted(set(v1_cats) | set(v2_cats))

    print(f"  {'Category':<14} {'v1 n':>6} {'v1 WR':>7} {'v1 ROI':>10} "
          f"{'v2 n':>6} {'v2 WR':>7} {'v2 ROI':>10}")
    print("  " + "-" * 76)
    for cat in cats:
        v1 = v1_cats.get(cat) or {}
        v2 = v2_cats.get(cat) or {}
        print(f"  {cat:<14} "
              f"{v1.get('n_trades', 0):>6} {v1.get('win_rate_pct', 0):>6.1f}% "
              f"{v1.get('roi_pct', 0):>+9.2f}% "
              f"{v2.get('n_trades', 0):>6} {v2.get('win_rate_pct', 0):>6.1f}% "
              f"{v2.get('roi_pct', 0):>+9.2f}%")

    # Verdict guidance
    print("\n" + "-" * 78)
    if not v2_recs:
        verdict = "v2 has no resolved bets yet — wait a few cycles."
    elif sum_v2.get("kelly_1/3x", {}).get("trades", 0) < 50:
        verdict = "v2 sample too small (<50). Wait for more resolutions."
    else:
        v1_roi_third = sum_v1.get("kelly_1/3x", {}).get("roi_pct", 0.0)
        v2_roi_third = sum_v2.get("kelly_1/3x", {}).get("roi_pct", 0.0)
        diff = v2_roi_third - v1_roi_third
        if diff > 1.0:
            verdict = f"v2 leads by +{diff:.2f}% ROI at 1/3x. Promising - keep running."
        elif diff < -1.0:
            verdict = f"v2 LAGS by {diff:.2f}% ROI. Investigate before merging."
        else:
            verdict = f"v2 ~= v1 (dROI {diff:+.2f}%). Need more data to distinguish."
    print(f"  Verdict: {verdict}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
