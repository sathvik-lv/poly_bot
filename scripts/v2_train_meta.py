"""V2 — train the XGBoost meta-model from validated ledger.

Reads BOTH v2_ledger and test1_ledger (since they share the same prediction
engine outputs and feature schema), validates strictly via data_validator,
splits chronologically into train/val (no leakage from future), and writes:

  data/meta_model.xgb           # binary booster
  data/meta_model.xgb.info.json # MetaModelInfo (n_train, val_brier, etc.)
  data/v2_meta_train_report.json # human-readable training summary

Hard guardrails (same philosophy as v2_train_weights.py):
  - Refuse to train if validated bets < MIN_TRAIN_N
  - Refuse if val_brier >= market_brier on the held-out window (no edge)
  - Refuse if quality-reject rate > MAX_QUALITY_REJECT_RATE
  - Atomic write via .tmp + os.replace
  - Backup previous model to .prev
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data_validator import iter_validated, ValidationStats
from src.meta_model import (
    FEATURE_NAMES, MetaModelInfo, XGBoostMetaModel, build_feature_vector,
)

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "meta_model.xgb")
MODEL_BACKUP = MODEL_FILE + ".prev"
TRAIN_REPORT = os.path.join(DATA_DIR, "v2_meta_train_report.json")

LEDGER_SOURCES = [
    os.path.join(DATA_DIR, "v2_ledger.jsonl"),
    os.path.join(DATA_DIR, "test1_ledger.jsonl"),
]

MIN_TRAIN_N = 80           # don't fit XGBoost on tiny samples
VAL_FRACTION = 0.20        # last 20% by time = held-out validation
MAX_QUALITY_REJECT_RATE = 0.10
EXPECTED_FILTER_REASONS = {"not_resolved", "duplicate_market_round_action"}
EXPECTED_FILTER_PREFIXES = ("action_not_a_bet:",)


def _read_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_validated_bets():
    stats = ValidationStats()
    bets = []
    for src in LEDGER_SOURCES:
        bets.extend(iter_validated(_read_jsonl(src), stats=stats, drop_duplicates=True))
    return bets, stats


def chronological_split(bets, val_fraction: float):
    bets_sorted = sorted(bets, key=lambda r: r.get("resolved_at") or r.get("timestamp") or "")
    n = len(bets_sorted)
    n_val = max(int(n * val_fraction), 1)
    return bets_sorted[: n - n_val], bets_sorted[n - n_val:]


def build_xy(bets):
    X = np.zeros((len(bets), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(len(bets), dtype=np.float32)
    market_p = np.zeros(len(bets), dtype=np.float32)
    for i, r in enumerate(bets):
        X[i] = build_feature_vector(r)
        y[i] = float(r["outcome"])
        market_p[i] = float(r["market_price"])
    return X, y, market_p


def main():
    print("\n" + "=" * 70)
    print("  V2 META-MODEL TRAINER (XGBoost)")
    print("=" * 70)

    bets, stats = collect_validated_bets()
    print(f"  Sources: {LEDGER_SOURCES}")
    print(stats.report())

    quality_rejected = sum(
        n for r, n in stats.reasons.items()
        if r not in EXPECTED_FILTER_REASONS and not r.startswith(EXPECTED_FILTER_PREFIXES)
    )
    eligible = stats.accepted + quality_rejected
    quality_rate = quality_rejected / eligible if eligible else 0.0
    print(f"  Eligible: {eligible}  Quality rejects: {quality_rejected} ({quality_rate:.2%})")

    if quality_rate > MAX_QUALITY_REJECT_RATE:
        print(f"  ABORT: quality reject rate {quality_rate:.1%} > "
              f"{MAX_QUALITY_REJECT_RATE:.0%} -- data integrity too low")
        return

    if len(bets) < MIN_TRAIN_N:
        print(f"  WAIT: only {len(bets)} validated bets, need >= {MIN_TRAIN_N}")
        return

    train_bets, val_bets = chronological_split(bets, VAL_FRACTION)
    print(f"  Chronological split: train={len(train_bets)}  val={len(val_bets)}")

    if not train_bets or not val_bets:
        print("  ABORT: split produced empty side")
        return

    X_train, y_train, _ = build_xy(train_bets)
    X_val, y_val, market_p_val = build_xy(val_bets)

    # Lazy import — keeps the validator + load() path import-cheap
    try:
        import xgboost as xgb
    except ImportError:
        print("  ABORT: xgboost not installed (pip install xgboost)")
        return

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "verbosity": 0,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Brier on validation
    val_pred = booster.predict(dval)
    val_brier = float(np.mean((val_pred - y_val) ** 2))
    market_brier = float(np.mean((market_p_val - y_val) ** 2))
    train_pred = booster.predict(dtrain)
    train_brier = float(np.mean((train_pred - y_train) ** 2))
    improvement = market_brier - val_brier

    print()
    print(f"  Train Brier:    {train_brier:.4f}  (n={len(train_bets)})")
    print(f"  Val Brier:      {val_brier:.4f}  (n={len(val_bets)})")
    print(f"  Market Brier:   {market_brier:.4f}  (held-out window)")
    print(f"  Improvement:    {improvement:+.4f}  (positive = beats market)")

    if improvement <= 0:
        print()
        print(f"  ABORT: meta-model val_brier {val_brier:.4f} >= market_brier "
              f"{market_brier:.4f} -- no edge on held-out window. "
              f"Keeping previous model (if any).")
        # Still write the report so we can see why
        report = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": len(train_bets),
            "n_val": len(val_bets),
            "train_brier": train_brier,
            "val_brier": val_brier,
            "market_brier": market_brier,
            "improvement": improvement,
            "written": False,
            "abort_reason": "no_edge_on_validation",
        }
        with open(TRAIN_REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return

    info = MetaModelInfo(
        n_train=len(train_bets),
        n_features=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,
        train_brier=round(train_brier, 6),
        val_brier=round(val_brier, 6),
        market_brier=round(market_brier, 6),
        improvement=round(improvement, 6),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )

    # Backup + atomic write
    if os.path.exists(MODEL_FILE):
        try:
            shutil.copy(MODEL_FILE, MODEL_BACKUP)
        except OSError:
            pass

    model = XGBoostMetaModel(booster=booster, info=info)
    tmp = MODEL_FILE + ".tmp"
    model.save(tmp)
    os.replace(tmp, MODEL_FILE)
    # save() wrote .info.json next to the .tmp file; move it to final path too
    tmp_info = tmp + ".info.json"
    final_info = MODEL_FILE + ".info.json"
    if os.path.exists(tmp_info):
        os.replace(tmp_info, final_info)

    print(f"\n  Wrote {MODEL_FILE}")

    # Feature importance (top 10)
    try:
        imp = booster.get_score(importance_type="gain")
        top = sorted(imp.items(), key=lambda x: -x[1])[:10]
        print("\n  Top features by gain:")
        for fname, gain in top:
            print(f"    {fname:<25} {gain:>9.2f}")
    except Exception:
        pass

    report = {
        "trained_at": info.trained_at,
        "n_train": info.n_train,
        "n_val": len(val_bets),
        "n_features": info.n_features,
        "feature_names": info.feature_names,
        "train_brier": info.train_brier,
        "val_brier": info.val_brier,
        "market_brier": info.market_brier,
        "improvement": info.improvement,
        "written": True,
        "feature_importance": dict(sorted(
            (booster.get_score(importance_type="gain") or {}).items(),
            key=lambda x: -x[1],
        )),
    }
    with open(TRAIN_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {TRAIN_REPORT}\n")


if __name__ == "__main__":
    main()
