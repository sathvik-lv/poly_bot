"""Same-window comparison across ALL 6 arms.

Reads `data/comparison_anchor.json` for the start timestamp, then for each
arm replays equity from $10,000 using ONLY records timestamped at or after
the anchor. Pre-anchor data is preserved on disk but excluded from the
comparison window.

This gives a clean apples-to-apples report for V3a/V3b (which start fresh)
vs Test 0 (which has 35 days of pre-anchor history) vs the shadow arms.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exit_simulator import held_pnl
from src.ledger_reader import iter_ledger

DATA_DIR = "data"
ANCHOR_FILE = os.path.join(DATA_DIR, "comparison_anchor.json")


def load_anchor() -> str:
    if not os.path.exists(ANCHOR_FILE):
        return datetime.now(timezone.utc).isoformat()
    with open(ANCHOR_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("comparison_anchor_at",
                                 datetime.now(timezone.utc).isoformat())


def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def verdict(wins, n, roi):
    if n < 30:
        return f"TOO SMALL (n<30)"
    lo, hi = wilson(wins, n)
    if lo > 0.50 and roi > 1:
        return f"PROVEN EDGE (CI[{lo:.2f},{hi:.2f}])"
    if lo > 0.50:
        return f"WR proven, ROI weak"
    if hi < 0.50:
        return f"PROVEN LOSER"
    if roi < 0:
        return "losing despite WR"
    return f"noise (CI[{lo:.2f},{hi:.2f}])"


def replay_paper_arm(state_path, anchor_dt) -> dict:
    """Replay a paper_trader state file, including only post-anchor closed positions."""
    if not os.path.exists(state_path):
        return {"n": 0, "wins": 0, "final": 10000.0, "roi": 0.0,
                "max_dd": 0.0, "ret_dd": 0.0, "started": None}
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    closed = state.get("closed_positions", [])
    closed = [p for p in closed
              if (parse_iso(p.get("resolved_at")) or parse_iso(p.get("timestamp")))
              and (parse_iso(p.get("resolved_at")) or parse_iso(p.get("timestamp"))) >= anchor_dt]
    closed.sort(key=lambda p: p.get("resolved_at") or p.get("timestamp") or "")

    eq = 10000.0
    peak = eq
    max_dd = 0.0
    wins = 0
    for p in closed:
        eq += p.get("pnl", 0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak else 0
        if dd > max_dd:
            max_dd = dd
        if p.get("pnl", 0) > 0:
            wins += 1
    roi = (eq - 10000) / 100
    return {
        "n": len(closed), "wins": wins, "final": eq, "roi": roi,
        "max_dd": max_dd * 100, "ret_dd": roi / max(max_dd * 100, 0.01),
        "started": closed[0].get("timestamp") if closed else None,
    }


def replay_shadow_arm(ledger_name, anchor_dt, kelly_const=1/3) -> dict:
    """Replay a shadow ledger (test1 / v2) using stored kelly_fraction.

    ``ledger_name`` is the stem ("test1" or "v2") — iter_ledger unions the
    current file with any rotated .gz archives under data/archive/.
    """
    bets = []
    for r in iter_ledger(ledger_name):
        if not r.get("resolved"):
            continue
        if r.get("action") not in ("BUY_YES", "BUY_NO"):
            continue
        if r.get("outcome") not in (0.0, 1.0):
            continue
        ts = parse_iso(r.get("resolved_at") or r.get("timestamp"))
        if ts is None or ts < anchor_dt:
            continue
        mp = r.get("market_price")
        if mp is None:
            continue
        bets.append(r)
    if not bets:
        return {"n": 0, "wins": 0, "final": 10000.0, "roi": 0.0,
                "max_dd": 0.0, "ret_dd": 0.0, "started": None}
    bets.sort(key=lambda r: r.get("resolved_at") or r.get("timestamp") or "")

    eq = 10000.0
    peak = eq
    max_dd = 0.0
    wins = 0
    for r in bets:
        kelly = abs(float(r.get("kelly_fraction", 0.05))) * kelly_const
        bet_pct = min(kelly, 0.25)
        bet = eq * bet_pct
        delta = bet * held_pnl(r["action"], float(r["market_price"]),
                                float(r["outcome"]))
        eq += delta
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak else 0
        if dd > max_dd:
            max_dd = dd
        if delta > 0:
            wins += 1
    roi = (eq - 10000) / 100
    return {
        "n": len(bets), "wins": wins, "final": eq, "roi": roi,
        "max_dd": max_dd * 100, "ret_dd": roi / max(max_dd * 100, 0.01),
        "started": bets[0].get("timestamp") if bets else None,
    }


def main():
    anchor_str = load_anchor()
    anchor_dt = parse_iso(anchor_str)
    print("\n" + "=" * 110)
    print("  SAME-WINDOW COMPARISON ACROSS ALL ARMS")
    print(f"  Anchor: {anchor_str}")
    print(f"  All arms reset to $10,000 at anchor; only post-anchor activity counts.")
    print("=" * 110)

    arms = [
        ("Test 0 (real, all categories)",
         lambda: replay_paper_arm("data/paper_trades.json", anchor_dt)),
        ("Test 0-TIER (real, all cats, tier caps 3k/1k/250)",
         lambda: replay_paper_arm("data/test0_tier_paper_trades.json", anchor_dt)),
        ("V3a (real, niche+geo+other)",
         lambda: replay_paper_arm("data/v3a_paper_trades.json", anchor_dt)),
        ("V3b (real, niche+geo+sports)",
         lambda: replay_paper_arm("data/v3b_paper_trades.json", anchor_dt)),
        ("V3c (real, niche+geo+favorite-other)",
         lambda: replay_paper_arm("data/v3c_paper_trades.json", anchor_dt)),
        ("V5 (real, live-candidate: niche+geo+other, 1/4 K, 2.5% cap)",
         lambda: replay_paper_arm("data/v5_live_candidate_paper_trades.json", anchor_dt)),
        ("V5-TIER (real, niche+geo+other, tier caps 3k/1k/250)",
         lambda: replay_paper_arm("data/v5_tier_paper_trades.json", anchor_dt)),
        ("V6 (real, V2-pipeline as paper: adaptive+meta+gate, 2.5% cap)",
         lambda: replay_paper_arm("data/v6_v2_paper_trades.json", anchor_dt)),
        ("V6-TIER (real, V2 pipeline + tier caps 3k/1k/250)",
         lambda: replay_paper_arm("data/v6_tier_paper_trades.json", anchor_dt)),
        ("V2 (shadow, adaptive+meta, 1/3x Kelly)",
         lambda: replay_shadow_arm("v2", anchor_dt)),
    ]

    print(f"\n  {'Strategy':<42} {'n':>5} {'WR':>6} {'Final':>10} "
          f"{'ROI':>8} {'MaxDD':>7} {'Ret/DD':>7}  Verdict")
    print("  " + "-" * 130)
    rows = []
    for label, fn in arms:
        try:
            r = fn()
        except Exception as e:
            print(f"  {label:<42} ERROR: {str(e)[:50]}")
            continue
        rows.append((label, r))
        wr = r["wins"] / r["n"] * 100 if r["n"] else 0
        v = verdict(r["wins"], r["n"], r["roi"])
        print(f"  {label:<42} {r['n']:>5} {wr:>5.1f}% ${r['final']:>9,.0f} "
              f"{r['roi']:>+7.2f}% {r['max_dd']:>6.2f}% {r['ret_dd']:>6.2f}  {v}")

    print()
    real_arms = [r for label, r in rows if "real" in label.lower()]
    if any(r["n"] > 0 for r in real_arms):
        leader = max((r for label, r in rows if "real" in label.lower() and r["n"] > 0),
                     key=lambda r: r["roi"], default=None)
        if leader:
            for label, r in rows:
                if r is leader:
                    print(f"  REAL-MONEY LEADER: {label.strip()}  "
                          f"ROI {r['roi']:+.2f}%  Ret/DD {r['ret_dd']:.2f}")
                    break
    else:
        print("  No real-arm trades since anchor yet — wait for V3a/V3b to accumulate.")
    print()


if __name__ == "__main__":
    main()
