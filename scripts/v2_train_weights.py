"""V2 — train per-model weights from validated resolved outcomes.

Reads:
  - data/v2_ledger.jsonl  (preferred — measures the v2 pipeline's own outcomes)
  - data/test1_ledger.jsonl  (fallback / bootstrap — same prediction engine)

For each model m:
  brier_m = mean( (model_estimate_m - outcome)^2 ) over validated bets
  brier_market = mean( (market_price - outcome)^2 ) over the same bets
  improvement_m = brier_market - brier_m   (positive = beats the market)

Weight rule:
  raw_m  = max(0, improvement_m)
  weight_m = raw_m / sum(raw_*)        (renormalized; sums to 1)

If no model beats the market for a given category, all weights collapse to
zero and the AdaptiveEnsemble falls back to inverse-variance.

Hard guardrails:
  - Refuse to write if total validated bets < MIN_TRAIN_N
  - Refuse to write if validation rejection rate > MAX_REJECT_RATE (data
    is too dirty to trust)
  - Atomic write via .tmp + os.replace
  - Backup previous weights to model_weights.prev.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_validator import (
    KNOWN_MODELS, ValidationStats, iter_validated, load_validated_jsonl,
)

DATA_DIR = "data"
WEIGHTS_FILE = os.path.join(DATA_DIR, "model_weights.json")
WEIGHTS_BACKUP = os.path.join(DATA_DIR, "model_weights.prev.json")
TRAIN_REPORT = os.path.join(DATA_DIR, "v2_train_report.json")

MIN_TRAIN_N = 30
# Reasons that indicate the record was simply not eligible (not a quality
# problem). These don't count toward the data-quality abort threshold.
EXPECTED_FILTER_REASONS = {
    "not_resolved",
    "duplicate_market_round_action",
}
EXPECTED_FILTER_PREFIXES = ("action_not_a_bet:",)
# True data-quality budget: of the records that COULD have been training
# input (resolved, real bets), how many fail integrity checks?
MAX_QUALITY_REJECT_RATE = 0.10
LEDGER_SOURCES = [
    os.path.join(DATA_DIR, "v2_ledger.jsonl"),
    os.path.join(DATA_DIR, "test1_ledger.jsonl"),
]


def collect_validated_bets() -> tuple[list, ValidationStats]:
    combined: list = []
    overall_stats = ValidationStats()
    for path in LEDGER_SOURCES:
        if not os.path.exists(path):
            continue
        recs, stats = load_validated_jsonl(path, drop_duplicates=True)
        combined.extend(recs)
        overall_stats.seen += stats.seen
        overall_stats.accepted += stats.accepted
        overall_stats.rejected += stats.rejected
        for k, v in stats.reasons.items():
            overall_stats.reasons[k] += v
    return combined, overall_stats


def compute_brier_per_model(bets: list) -> dict:
    """Return per-model {brier, n, wins, model_brier_minus_market_brier}."""
    per_model: dict[str, dict] = {
        m: {"sq_err_sum": 0.0, "n": 0, "wins": 0, "market_sq_err_sum": 0.0}
        for m in KNOWN_MODELS
    }
    for r in bets:
        outcome = float(r["outcome"])
        market_price = float(r["market_price"])
        market_sq = (market_price - outcome) ** 2
        for m, est in (r.get("model_estimates") or {}).items():
            if m not in per_model:
                continue
            try:
                e = float(est)
            except (TypeError, ValueError):
                continue
            per_model[m]["sq_err_sum"] += (e - outcome) ** 2
            per_model[m]["market_sq_err_sum"] += market_sq
            per_model[m]["n"] += 1
            # Model's directional call vs outcome
            model_call = 1.0 if e > market_price else 0.0
            won = (model_call == 1.0 and outcome == 1.0) or \
                  (model_call == 0.0 and outcome == 0.0)
            if won:
                per_model[m]["wins"] += 1

    out = {}
    for m, d in per_model.items():
        if d["n"] == 0:
            out[m] = {"n": 0, "brier": None, "market_brier": None,
                      "improvement": 0.0, "wins": 0}
            continue
        brier = d["sq_err_sum"] / d["n"]
        mkt = d["market_sq_err_sum"] / d["n"]
        out[m] = {
            "n": d["n"],
            "brier": round(brier, 6),
            "market_brier": round(mkt, 6),
            "improvement": round(mkt - brier, 6),
            "wins": d["wins"],
            "wr": round(d["wins"] / d["n"], 4),
        }
    return out


def normalize_weights(model_stats: dict) -> dict[str, float]:
    raw = {m: max(0.0, s["improvement"] or 0.0) for m, s in model_stats.items()}
    total = sum(raw.values())
    if total <= 0:
        # Fallback: equal weights for any model with data
        active = [m for m, s in model_stats.items() if s["n"] > 0]
        if not active:
            return {}
        return {m: round(1.0 / len(active), 6) for m in active}
    return {m: round(v / total, 6) for m, v in raw.items()}


def main():
    print("\n" + "=" * 70)
    print("  V2 WEIGHT TRAINER")
    print("=" * 70)

    bets, stats = collect_validated_bets()
    print(f"  Sources: {LEDGER_SOURCES}")
    print(f"  {stats.report().strip()}")

    if stats.seen == 0:
        print("\n  No data available. Skipping write.")
        return

    # Split reasons into "expected filtering" vs "data quality issue"
    quality_rejected = 0
    expected_filtered = 0
    for reason, n in stats.reasons.items():
        if reason in EXPECTED_FILTER_REASONS or reason.startswith(EXPECTED_FILTER_PREFIXES):
            expected_filtered += n
        else:
            quality_rejected += n

    eligible = stats.accepted + quality_rejected
    quality_reject_rate = (quality_rejected / eligible) if eligible else 0.0

    print(f"\n  Eligible records (resolved bets): {eligible}")
    print(f"  Quality-rejected:                 {quality_rejected}")
    print(f"  Expected-filtered (not bets):     {expected_filtered}")
    print(f"  Quality reject rate:              {quality_reject_rate:.2%}")

    if quality_reject_rate > MAX_QUALITY_REJECT_RATE:
        print(f"\n  ABORT: quality reject rate {quality_reject_rate:.1%} > "
              f"max {MAX_QUALITY_REJECT_RATE:.0%} — data integrity too low to train")
        return

    if len(bets) < MIN_TRAIN_N:
        print(f"\n  WAIT: only {len(bets)} validated bets, need >= {MIN_TRAIN_N}.")
        print("  No write — keeping existing weights.")
        return

    model_stats = compute_brier_per_model(bets)

    print("\n  Per-model performance vs market baseline:")
    print(f"  {'Model':<18} {'n':>5} {'Brier':>8} {'MktBrier':>10} {'Edge':>9} {'WR':>6}")
    print("  " + "-" * 62)
    for m in sorted(model_stats.keys()):
        s = model_stats[m]
        if s["n"] == 0:
            print(f"  {m:<18} {0:>5} {'-':>8} {'-':>10} {'-':>9} {'-':>6}")
            continue
        print(f"  {m:<18} {s['n']:>5} {s['brier']:>8.4f} "
              f"{s['market_brier']:>10.4f} {s['improvement']:>+9.4f} "
              f"{s['wr']:>5.1%}")

    weights = normalize_weights(model_stats)
    print("\n  Normalized weights:")
    for m in sorted(weights.keys()):
        print(f"    {m:<18} {weights[m]:.4f}")

    # Backup + atomic write
    if os.path.exists(WEIGHTS_FILE):
        try:
            shutil.copy(WEIGHTS_FILE, WEIGHTS_BACKUP)
        except OSError:
            pass

    tmp = WEIGHTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    os.replace(tmp, WEIGHTS_FILE)
    print(f"\n  Wrote {WEIGHTS_FILE}")

    # Side report
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_validated_bets": len(bets),
        "validation_rejected": stats.rejected,
        "quality_rejected": quality_rejected,
        "expected_filtered": expected_filtered,
        "quality_reject_rate": round(quality_reject_rate, 4),
        "rejection_reasons": dict(stats.reasons),
        "model_stats": model_stats,
        "weights": weights,
    }
    with open(TRAIN_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to {TRAIN_REPORT}\n")


if __name__ == "__main__":
    main()
