"""V2 grid sweep — finds optimal (Kelly, exit-decay, entry-threshold)
on the full validated ledger via walk-forward simulation, then computes
per-category Kelly allocation tiers (more capital to low-DD/high-return
categories, drop categories where DD > return).

Methodology adapted from `forex/STRATEGY_LOCKED_CARRY_TREND_V1.md`:
  - Grid sweep across hyperparameters
  - Composite score: Sharpe + CAGR/MaxDD + WR + trade-count penalty
  - Per-pair (here: per-category) Kelly with cap and total-exposure clamp

Output: data/v2_grid_sweep_report.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.exit_simulator import per_dollar_pnl, should_exit, held_pnl

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "v2_grid_sweep_report.json")
LEDGER_SOURCES = [
    os.path.join(DATA_DIR, "v2_ledger.jsonl"),
    os.path.join(DATA_DIR, "test1_ledger.jsonl"),
]

ENTRY_THRESHOLDS = [0.005, 0.01, 0.02, 0.03]
DECAY_THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.1]   # 1.1 = effectively no exit
KELLY_FRACTIONS = [1/8, 1/5, 1/4, 1/3, 1/2, 1.0]
SPREAD_COST = 0.02

# Per-category allocation tier thresholds
HIGH_TIER_RET_OVER_DD = 5.0     # return >= 5x maxDD -> high allocation
MED_TIER_RET_OVER_DD = 1.5      # return >= 1.5x maxDD -> medium allocation
                                # below 1.5 -> drop


def load_records():
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


def group_by_market(records):
    by_mid = defaultdict(list)
    for r in records:
        if r.get("market_id") and r.get("timestamp"):
            by_mid[r["market_id"]].append(r)
    for rs in by_mid.values():
        rs.sort(key=lambda x: x["timestamp"])
    return dict(by_mid)


def simulate_market(rs, entry_threshold, decay_threshold, spread_cost):
    """Return (per-dollar pnl, category, action_taken) or (None, None, None) if no trade."""
    resolved = next((r for r in rs if r.get("resolved")), None)
    if resolved is None:
        return None, None, None
    outcome = resolved.get("outcome")
    if outcome not in (0.0, 1.0):
        return None, None, None
    outcome = float(outcome)

    entry = None
    for r in rs:
        if r.get("market_price") is None or r.get("edge") is None:
            continue
        if abs(float(r["edge"])) >= entry_threshold:
            entry = r
            break
    if entry is None:
        return None, None, None

    entry_price = float(entry["market_price"])
    entry_edge = float(entry["edge"])
    if not (0.01 <= entry_price <= 0.99):
        return None, None, None
    action = "BUY_YES" if entry_edge > 0 else "BUY_NO"
    category = entry.get("category", "other")
    ts_entry = entry.get("timestamp", "")

    for later in rs:
        if (later.get("timestamp") or "") <= ts_entry:
            continue
        if later.get("edge") is None:
            continue
        if should_exit(entry_edge, float(later["edge"]), decay_threshold):
            exit_price = float(later.get("market_price", entry_price))
            if not (0.01 <= exit_price <= 0.99):
                continue
            return per_dollar_pnl(action, entry_price, exit_price) - spread_cost, category, action

    return held_pnl(action, entry_price, outcome), category, action


def compounded_replay(per_dollar_pnls, kelly_per_trade, starting_equity=10000.0,
                      max_bet_pct=0.25):
    """Replay a sequence of (per_dollar_pnl, kelly_pct) chronologically.

    Returns final_equity, max_dd_pct, sharpe, cagr_proxy, n_trades, wins.
    """
    equity = starting_equity
    peak = equity
    max_dd = 0.0
    daily_returns = []
    wins = 0
    for pnl, k in zip(per_dollar_pnls, kelly_per_trade):
        bet_pct = min(abs(k), max_bet_pct)
        bet = equity * bet_pct
        delta = bet * pnl
        if equity > 0:
            daily_returns.append(delta / equity)
        equity = max(0.0, equity + delta)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if delta > 0:
            wins += 1

    n = len(per_dollar_pnls)
    final_pnl_pct = (equity - starting_equity) / starting_equity
    sharpe = 0.0
    if len(daily_returns) > 1:
        m = mean(daily_returns)
        s = stdev(daily_returns)
        if s > 0:
            sharpe = m / s * math.sqrt(252)  # annualized assuming 1 trade/day approx
    cagr_over_dd = final_pnl_pct / max(max_dd, 0.001)
    return {
        "final_equity": equity,
        "roi_pct": final_pnl_pct * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "cagr_over_dd": cagr_over_dd,
        "n": n,
        "wins": wins,
        "wr_pct": wins / n * 100 if n else 0,
    }


def run_sweep(by_mid):
    grid = []
    for entry_thr in ENTRY_THRESHOLDS:
        for decay_thr in DECAY_THRESHOLDS:
            # Build the trade list once per (entry, decay)
            trades = []  # list of (per_dollar_pnl, category)
            for mid, rs in by_mid.items():
                pnl, cat, _ = simulate_market(rs, entry_thr, decay_thr, SPREAD_COST)
                if pnl is None:
                    continue
                trades.append((pnl, cat))
            if len(trades) < 30:
                continue
            for kelly in KELLY_FRACTIONS:
                # Use a rough proxy: every trade gets `kelly` * (engine kelly_full
                # ~= 0.05) — i.e., flat sizing across trades for the sweep.
                k_per_trade = [kelly * 0.05] * len(trades)
                pnl_seq = [p for p, _ in trades]
                m = compounded_replay(pnl_seq, k_per_trade)
                m.update({
                    "entry_threshold": entry_thr,
                    "decay_threshold": decay_thr,
                    "kelly_fraction": kelly,
                })
                grid.append(m)
    return grid


def composite_score(rows):
    """Add a composite score column.

    SCORE = 0.40 * z(roi) + 0.25 * z(cagr_over_dd) + 0.20 * z(sharpe)
          + 0.10 * z(wr) + 0.05 * z(n_trades_clip)
    Filter: n >= 30, max_dd <= 25%.
    """
    if not rows:
        return rows

    def z(values):
        if not values:
            return [0] * len(values)
        m = mean(values)
        s = stdev(values) if len(values) > 1 else 1.0
        if s == 0: s = 1.0
        return [(v - m) / s for v in values]

    eligible = [r for r in rows if r["n"] >= 30 and r["max_dd_pct"] <= 25]
    if not eligible:
        for r in rows:
            r["score"] = -999
            r["eligible"] = False
        return rows

    z_roi = z([r["roi_pct"] for r in eligible])
    z_cagr_dd = z([r["cagr_over_dd"] for r in eligible])
    z_sharpe = z([r["sharpe"] for r in eligible])
    z_wr = z([r["wr_pct"] for r in eligible])
    z_n = z([min(r["n"], 200) for r in eligible])
    for i, r in enumerate(eligible):
        r["score"] = (0.40 * z_roi[i] + 0.25 * z_cagr_dd[i] + 0.20 * z_sharpe[i]
                      + 0.10 * z_wr[i] + 0.05 * z_n[i])
        r["eligible"] = True
    for r in rows:
        if "score" not in r:
            r["score"] = -999
            r["eligible"] = False
    return rows


def per_category_analysis(by_mid, entry_thr, decay_thr):
    """For each category, compute WR, avg payoff, Kelly, max DD, return."""
    by_cat = defaultdict(list)
    for mid, rs in by_mid.items():
        pnl, cat, _ = simulate_market(rs, entry_thr, decay_thr, SPREAD_COST)
        if pnl is None:
            continue
        by_cat[cat].append(pnl)
    out = {}
    for cat, pnls in by_cat.items():
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n if n else 0
        avg_win = mean(wins) if wins else 0
        avg_loss = abs(mean(losses)) if losses else 1
        payoff = avg_win / avg_loss if avg_loss else 0
        kelly = wr - (1 - wr) / payoff if payoff > 0 else 0
        # Compounded equity replay at flat 1/3 Kelly for DD measurement
        m = compounded_replay(pnls, [1/3 * 0.05] * n)
        out[cat] = {
            "n": n, "wins": len(wins), "wr_pct": wr * 100,
            "avg_win_per_dollar": round(avg_win, 4),
            "avg_loss_per_dollar": round(-avg_loss, 4),
            "payoff_ratio": round(payoff, 3),
            "kelly_raw": round(kelly, 4),
            "kelly_third": round(max(0, kelly / 3), 4),
            "total_pnl_per_dollar": round(sum(pnls), 4),
            "roi_pct_at_third_kelly": round(m["roi_pct"], 2),
            "max_dd_pct_at_third_kelly": round(m["max_dd_pct"], 2),
            "ret_over_dd": round(m["roi_pct"] / max(m["max_dd_pct"], 0.01), 2),
        }
    return out


def allocation_tiers(cat_stats):
    """Bucket categories into HIGH/MEDIUM/DROP based on return/DD ratio."""
    tiers = {"HIGH": [], "MEDIUM": [], "DROP": []}
    for cat, s in cat_stats.items():
        r = s["ret_over_dd"]
        if s["n"] < 20:
            tiers["DROP"].append((cat, s, "insufficient_data_n<20"))
        elif r >= HIGH_TIER_RET_OVER_DD and s["wr_pct"] >= 55:
            tiers["HIGH"].append((cat, s, f"ret/dd={r}>={HIGH_TIER_RET_OVER_DD}"))
        elif r >= MED_TIER_RET_OVER_DD and s["wr_pct"] >= 50:
            tiers["MEDIUM"].append((cat, s, f"ret/dd={r}>={MED_TIER_RET_OVER_DD}"))
        else:
            tiers["DROP"].append((cat, s, f"ret/dd={r}<{MED_TIER_RET_OVER_DD} OR WR<50"))
    return tiers


def split_by_time(by_mid, train_frac=0.70):
    """Split markets into train/holdout chronologically by entry timestamp."""
    items = []
    for mid, rs in by_mid.items():
        first_ts = rs[0].get("timestamp", "") if rs else ""
        items.append((first_ts, mid, rs))
    items.sort(key=lambda x: x[0])
    split = int(len(items) * train_frac)
    train = {mid: rs for _, mid, rs in items[:split]}
    holdout = {mid: rs for _, mid, rs in items[split:]}
    return train, holdout


def evaluate_config(by_mid, entry_thr, decay_thr, kelly):
    """Evaluate a single config on the given market set. Mirror of sweep entry."""
    trades = []
    for mid, rs in by_mid.items():
        pnl, cat, _ = simulate_market(rs, entry_thr, decay_thr, SPREAD_COST)
        if pnl is None:
            continue
        trades.append((pnl, cat))
    if not trades:
        return None
    pnl_seq = [p for p, _ in trades]
    k_per_trade = [kelly * 0.05] * len(trades)
    m = compounded_replay(pnl_seq, k_per_trade)
    m["entry_threshold"] = entry_thr
    m["decay_threshold"] = decay_thr
    m["kelly_fraction"] = kelly
    return m


def main():
    print("\n" + "=" * 78)
    print("  V2 GRID SWEEP — entry threshold x decay threshold x Kelly fraction")
    print("=" * 78)

    records = load_records()
    by_mid = group_by_market(records)
    print(f"  Loaded {len(records)} records, {len(by_mid)} unique markets")

    # ---- Walk-forward split: pick config on TRAIN, verify on HOLDOUT ----
    train_mids, holdout_mids = split_by_time(by_mid, train_frac=0.70)
    print(f"  Walk-forward split: train={len(train_mids)} markets, "
          f"holdout={len(holdout_mids)} markets (chronological)")

    print("\n  Step 1 — search best config on TRAIN ONLY:")
    grid_train = run_sweep(train_mids)
    grid_train = composite_score(grid_train)
    grid_train.sort(key=lambda r: r.get("score", -999), reverse=True)
    if not grid_train or grid_train[0].get("score", -999) == -999:
        print("  No eligible configs on train slice — sample too small.")
        return
    best_train = grid_train[0]
    print(f"    Best on TRAIN: entry={best_train['entry_threshold']}, "
          f"decay={best_train['decay_threshold']}, "
          f"kelly={best_train['kelly_fraction']:.3f} -> "
          f"ROI {best_train['roi_pct']:+.2f}%, "
          f"MaxDD {best_train['max_dd_pct']:.2f}%, "
          f"Sharpe {best_train['sharpe']:.2f}")

    print("\n  Step 2 — APPLY same config to HOLDOUT (no fitting):")
    holdout_result = evaluate_config(
        holdout_mids,
        best_train["entry_threshold"],
        best_train["decay_threshold"],
        best_train["kelly_fraction"],
    )
    if holdout_result is None:
        print("    Holdout produced no trades. Cannot verify.")
        walkforward_pass = False
    else:
        print(f"    HOLDOUT: n={holdout_result['n']}, WR={holdout_result['wr_pct']:.1f}%, "
              f"ROI {holdout_result['roi_pct']:+.2f}%, "
              f"MaxDD {holdout_result['max_dd_pct']:.2f}%, "
              f"Sharpe {holdout_result['sharpe']:.2f}")
        # Pass criteria: holdout ROI must be positive AND > 30% of train ROI
        # (some shrinkage expected, but should not flip negative)
        train_roi = best_train["roi_pct"]
        wf_pass_threshold = max(0.0, train_roi * 0.30)
        walkforward_pass = (holdout_result["roi_pct"] > 0
                            and holdout_result["roi_pct"] >= wf_pass_threshold)
        verdict = "PASSED" if walkforward_pass else "FAILED"
        print(f"    WALK-FORWARD VERDICT: {verdict}  "
              f"(holdout ROI {holdout_result['roi_pct']:+.2f}% vs threshold "
              f">+{wf_pass_threshold:.2f}%)")

    print()
    print("  Step 3 — full-sample sweep (for reference / per-category report):")
    grid = run_sweep(by_mid)
    grid = composite_score(grid)
    grid.sort(key=lambda r: r.get("score", -999), reverse=True)

    print()
    print("  TOP 10 CONFIGS (composite score):")
    print(f"  {'rank':>4} {'entry':>6} {'decay':>6} {'kelly':>7} "
          f"{'n':>4} {'WR':>5} {'ROI%':>7} {'MaxDD%':>7} {'Sharpe':>7} {'score':>7}")
    print("  " + "-" * 78)
    for i, r in enumerate(grid[:10], 1):
        print(f"  {i:>4} {r['entry_threshold']:>6.3f} {r['decay_threshold']:>6.2f} "
              f"{r['kelly_fraction']:>6.3f} {r['n']:>4} {r['wr_pct']:>4.1f}% "
              f"{r['roi_pct']:>+6.2f}% {r['max_dd_pct']:>6.2f}% "
              f"{r['sharpe']:>7.2f} {r['score']:>+6.3f}")

    if not grid:
        print("\n  No eligible configs (insufficient data).")
        return

    best = grid[0]
    print(f"\n  WINNER: entry={best['entry_threshold']}, decay={best['decay_threshold']}, "
          f"kelly={best['kelly_fraction']:.3f}")

    # Per-category analysis at the winning (entry, decay) settings
    cat_stats = per_category_analysis(by_mid, best["entry_threshold"], best["decay_threshold"])
    tiers = allocation_tiers(cat_stats)

    print()
    print(f"  PER-CATEGORY ANALYSIS  (at entry={best['entry_threshold']}, decay={best['decay_threshold']})")
    print(f"  {'category':<14} {'n':>4} {'WR':>6} {'payoff':>7} {'kelly':>7} "
          f"{'ROI%':>7} {'DD%':>6} {'ret/DD':>7} {'tier':>6}")
    print("  " + "-" * 78)
    cat_to_tier = {}
    for tier, items in tiers.items():
        for cat, s, _ in items:
            cat_to_tier[cat] = tier
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        tier = cat_to_tier.get(cat, "?")
        print(f"  {cat:<14} {s['n']:>4} {s['wr_pct']:>5.1f}% {s['payoff_ratio']:>6.2f} "
              f"{s['kelly_raw']:>6.3f} {s['roi_pct_at_third_kelly']:>+6.2f}% "
              f"{s['max_dd_pct_at_third_kelly']:>5.2f}% {s['ret_over_dd']:>6.2f} {tier:>6}")

    print()
    print("  ALLOCATION TIERS:")
    for tier, items in tiers.items():
        if not items:
            continue
        print(f"    {tier}: {[c for c, _, _ in items]}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "winner_full_sample": best,
        "winner_train": best_train,
        "holdout_result": holdout_result,
        "walkforward_pass": walkforward_pass,
        "top_10_full_sample": grid[:10],
        "per_category": cat_stats,
        "allocation_tiers": {
            tier: [{"category": c, "stats": s, "reason": reason}
                   for c, s, reason in items]
            for tier, items in tiers.items()
        },
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {REPORT_FILE}\n")


if __name__ == "__main__":
    main()
