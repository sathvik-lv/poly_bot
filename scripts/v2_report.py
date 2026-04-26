"""V2 report — Kelly-fraction compounding on data/v2_ledger.jsonl.

Same compute logic as test1_report.py; loads the v2 ledger and runs the
shadow-ledger replay. Also surfaces v2-specific signals (gate decisions,
adaptive blend used).
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
from src.data_validator import iter_validated, ValidationStats

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "v2_ledger.jsonl")


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


def fmt_money(v): return f"${v:>10,.2f}"
def fmt_pct(v): return f"{v:>+7.2f}%"


def report():
    records = load_all()
    print("\n" + "=" * 78)
    print("  V2 — COMPOUNDED EQUITY REPORT (adaptive ensemble + category gate)")
    print("=" * 78)

    if not records:
        print("  V2 ledger empty — wait for first cycle to populate.\n")
        return

    # Validate first
    stats = ValidationStats()
    validated = list(iter_validated(records, stats=stats))
    print(f"  Total records:        {len(records)}")
    print(f"  Validated bets:       {len(validated)}")
    print(f"  {stats.report().strip()}")

    n_gated = sum(1 for r in records if not (r.get("gate_decision") or {}).get("allow", True))
    print(f"  Category-gated out:   {n_gated}")

    if not validated:
        print("\n  Nothing to replay yet.\n")
        return

    print(f"\n  Starting equity:      ${DEFAULT_NOTIONAL:,.2f}")
    print("\n" + "-" * 78)
    print("  COMPOUNDED EQUITY BY KELLY FRACTION  (validated bets only)")
    print("-" * 78)
    print(f"  {'Fraction':<14} {'Trades':>7} {'Win%':>6} "
          f"{'Final $':>13} {'ROI':>9} {'MaxDD':>8} {'Ruin':>6}")
    print("  " + "-" * 70)

    for label, fraction in KELLY_FRACTIONS.items():
        res = replay_kelly_fraction(validated, fraction)
        ruin_tag = "YES" if res["ruin"] else "no"
        print(f"  {label:<14} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
              f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])} "
              f"{res['max_drawdown_pct']:>6.1f}%  {ruin_tag:>4}")

    # Category breakdown at 1/3x
    print("\n" + "-" * 78)
    print("  CATEGORY BREAKDOWN  (replay at kelly_1/3x)")
    print("-" * 78)
    by_cat = defaultdict(list)
    for r in validated:
        by_cat[r.get("category", "other")].append(r)

    print(f"  {'Category':<14} {'Trades':>7} {'Win%':>6} {'Final $':>13} {'ROI':>9}")
    print("  " + "-" * 60)
    for cat in sorted(by_cat.keys()):
        recs = by_cat[cat]
        res = replay_kelly_fraction(recs, 1.0 / 3.0)
        print(f"  {cat:<14} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
              f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])}")

    # Per-model solo
    print("\n" + "-" * 78)
    print("  PER-MODEL SOLO REPLAY  (each model alone at kelly_1/3x)")
    print("-" * 78)
    print(f"  {'Model':<18} {'Trades':>7} {'Win%':>6} {'Final $':>13} {'ROI':>9}")
    print("  " + "-" * 60)
    all_models = set()
    for r in validated:
        for m in r.get("models", []):
            all_models.add(m)
    for name in sorted(all_models):
        try:
            res = replay_per_model_kelly_fraction(validated, name, 1.0 / 3.0)
            print(f"  {name:<18} {res['n_trades']:>7} {res['win_rate_pct']:>5.1f}% "
                  f"{fmt_money(res['final_equity'])} {fmt_pct(res['roi_pct'])}")
        except Exception:
            continue

    print("=" * 78 + "\n")


if __name__ == "__main__":
    report()
