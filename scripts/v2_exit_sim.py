"""V2 — early-exit-on-edge-decay simulator.

Walks the validated v2_ledger + test1_ledger, simulates "what if we'd
exited each open position when its edge decayed below a threshold,"
and writes a report. Pure observation — does NOT touch any live
positions and does NOT modify any ledger.

Output: data/v2_exit_sim_report.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exit_simulator import simulate_ledger

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "v2_exit_sim_report.json")
LEDGER_SOURCES = [
    os.path.join(DATA_DIR, "v2_ledger.jsonl"),
    os.path.join(DATA_DIR, "test1_ledger.jsonl"),
]
THRESHOLDS = [0.5, 0.7, 0.9]
SPREAD_COSTS = [0.0, 0.02, 0.05]


def load_records() -> list:
    out = []
    for path in LEDGER_SOURCES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def main():
    print("\n" + "=" * 70)
    print("  V2 EXIT-DECAY SIMULATOR")
    print("=" * 70)

    records = load_records()
    print(f"  Sources: {LEDGER_SOURCES}")
    print(f"  Loaded {len(records)} ledger records")

    if not records:
        print("  No data — nothing to simulate.")
        return

    grid = []
    print()
    print(f"  {'decay':>5} {'spread':>7} {'markets':>8} {'exits':>6} "
          f"{'hold $':>9} {'exit $':>9} {'diff':>9} {'hold WR':>9} {'exit WR':>9}")
    print("  " + "-" * 80)
    for decay in THRESHOLDS:
        for spread in SPREAD_COSTS:
            r = simulate_ledger(records, decay, spread)
            n = r["n_markets"] or 1
            hold_wr = r["held_wins"] / n * 100
            exit_wr = r["exit_wins"] / n * 100
            grid.append({
                "decay_threshold": decay,
                "spread_cost": spread,
                "n_markets": r["n_markets"],
                "n_exited": r["n_exited"],
                "held_total_pnl": round(r["held_total_pnl"], 4),
                "exit_total_pnl": round(r["exit_total_pnl"], 4),
                "diff": round(r["diff"], 4),
                "held_win_rate_pct": round(hold_wr, 2),
                "exit_win_rate_pct": round(exit_wr, 2),
            })
            print(f"  {decay:>5.2f} {spread:>7.2f} {r['n_markets']:>8} {r['n_exited']:>6} "
                  f"{r['held_total_pnl']:>+8.2f} {r['exit_total_pnl']:>+8.2f} "
                  f"{r['diff']:>+8.2f} {hold_wr:>8.1f}% {exit_wr:>8.1f}%")

    # Pick the best (decay, spread=0.02 = realistic) row to report as headline
    realistic = [g for g in grid if abs(g["spread_cost"] - 0.02) < 1e-6]
    best = max(realistic, key=lambda g: g["diff"]) if realistic else None
    headline = None
    if best and best["n_markets"] > 0:
        rel = best["diff"] / best["held_total_pnl"] * 100 if best["held_total_pnl"] else 0
        headline = {
            "best_decay": best["decay_threshold"],
            "best_spread": best["spread_cost"],
            "n_markets": best["n_markets"],
            "n_exited": best["n_exited"],
            "held_pnl": best["held_total_pnl"],
            "exit_pnl": best["exit_total_pnl"],
            "diff": best["diff"],
            "diff_pct_of_held": round(rel, 2),
        }
        print()
        print(f"  HEADLINE (realistic 2% spread, best threshold):")
        print(f"    decay={best['decay_threshold']}  exits={best['n_exited']}/{best['n_markets']}  "
              f"held=${best['held_total_pnl']:+.2f}  "
              f"exit=${best['exit_total_pnl']:+.2f}  "
              f"diff=${best['diff']:+.2f} ({rel:+.1f}%)")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_records_seen": len(records),
        "headline": headline,
        "grid": grid,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {REPORT_FILE}\n")


if __name__ == "__main__":
    main()
