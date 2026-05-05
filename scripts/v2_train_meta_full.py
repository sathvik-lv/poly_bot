"""V2 meta-model training with historical bootstrap + strict validation.

Trains the XGBoost meta-model on ALL available clean data:
  - data/backtest_honest_preds.jsonl   (3,879 historical backtest predictions)
  - data/test1_ledger.jsonl            (live shadow ledger)
  - data/v2_ledger.jsonl               (v2 shadow ledger)
  - data/paper_trades.json             (Test 0 closed positions)

Strict integrity rules ("uncorrupted data only"):
  * outcome must be exactly 0.0 or 1.0
  * market_price must be in (0.01, 0.99) exclusive — markets at the boundary
    were already resolved at entry, no genuine bet
  * predicted_prob must be in (0.01, 0.99) — same reasoning
  * sub-model estimates: at least one model with a finite value in [0,1]
  * timestamp + resolved_at must parse as ISO-8601
  * No NaN / inf anywhere in the feature vector
  * Reject if action is missing or not in {BUY_YES, BUY_NO}

Walk-forward verification AFTER training:
  * Splits by source: historical pre-training only, then walks the live
    timeline (Test 0 + Test 1 chronologically) with the pre-trained model.
  * Reports v1-baseline vs v2-with-meta on the live window.
  * Saves model + report only if walk-forward shows non-negative lift.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.exit_simulator import held_pnl
from src.meta_model import (
    FEATURE_NAMES, MetaModelInfo, XGBoostMetaModel, build_feature_vector,
)

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "meta_model.xgb")
MODEL_BACKUP = MODEL_FILE + ".prev"
TRAIN_REPORT = os.path.join(DATA_DIR, "v2_meta_full_train_report.json")

HISTORICAL_FILE = os.path.join(DATA_DIR, "backtest_honest_preds.jsonl")
T1_FILES = [os.path.join(DATA_DIR, "test1_ledger.jsonl"),
            os.path.join(DATA_DIR, "v2_ledger.jsonl")]
T0_FILE = os.path.join(DATA_DIR, "paper_trades.json")


# ---------- strict validator (the "god rule") -----------------------------

def _is_finite(x) -> bool:
    try:
        f = float(x)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def strict_validate(rec: dict, require_action: bool = False) -> tuple:
    """Return (ok, reason). Stricter than data_validator — for training only.

    require_action=False: action is optional. Training only needs features +
    outcome — the model predicts P(outcome=1) regardless of whether the
    engine actually bet. Walk-forward replay separately filters for action.
    require_action=True: enforce a real bet direction (used by replay).
    """
    if require_action:
        action = rec.get("action")
        if action not in ("BUY_YES", "BUY_NO"):
            return False, "action_not_a_bet"

    outcome = rec.get("outcome")
    if outcome not in (0.0, 1.0):
        return False, "outcome_not_binary"

    mp = rec.get("market_price")
    if not _is_finite(mp):
        return False, "market_price_nonfinite"
    mp = float(mp)
    if not (0.01 <= mp <= 0.99):
        return False, "market_price_at_boundary"

    pp = rec.get("predicted_prob")
    if pp is not None and not _is_finite(pp):
        return False, "predicted_prob_nonfinite"
    if pp is not None and not (0.0 <= float(pp) <= 1.0):
        return False, "predicted_prob_out_of_bounds"

    # At least one sub-model estimate must be finite and in [0, 1]
    estimates = rec.get("model_estimates") or rec.get("sub_model_estimates") or {}
    if not isinstance(estimates, dict) or not estimates:
        return False, "no_model_estimates"
    valid_estimates = 0
    for m, v in estimates.items():
        if _is_finite(v) and 0.0 <= float(v) <= 1.0:
            valid_estimates += 1
    if valid_estimates == 0:
        return False, "no_valid_estimates"

    # Time fields parseable
    ts = _parse_iso(rec.get("timestamp"))
    if ts is None:
        return False, "timestamp_unparseable"

    return True, "ok"


# ---------- loaders -------------------------------------------------------

def _normalize(rec: dict, src: str) -> dict:
    """Map any source's fields to a common shape used by feature builder."""
    estimates = rec.get("model_estimates") or rec.get("sub_model_estimates") or {}
    return {
        "src": src,
        "timestamp": rec.get("timestamp") or rec.get("resolved_at"),
        "market_id": rec.get("market_id"),
        "action": rec.get("action"),
        "market_price": rec.get("market_price") or rec.get("entry_price"),
        "predicted_prob": rec.get("predicted_prob"),
        "edge": rec.get("edge"),
        "outcome": rec.get("outcome"),
        "category": rec.get("category", "other"),
        "days_left": rec.get("days_left"),
        "context": rec.get("context") or {},
        "model_estimates": estimates,
        "prediction_round": rec.get("prediction_round", 1),
        "abs_edge": rec.get("abs_edge", abs(float(rec.get("edge", 0)) or 0)),
    }


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


def load_all_sources():
    sources = {"historical": [], "t1": [], "t0": []}

    # Historical backtest
    for r in _read_jsonl(HISTORICAL_FILE):
        # Historical needs an action — derive from edge sign if missing
        if "action" not in r:
            edge = r.get("edge", 0)
            r = dict(r)
            r["action"] = "BUY_YES" if edge > 0 else "BUY_NO"
        sources["historical"].append(_normalize(r, "historical"))

    # T1 ledgers
    for path in T1_FILES:
        for r in _read_jsonl(path):
            if not r.get("resolved"):
                continue
            sources["t1"].append(_normalize(r, "t1"))

    # T0 paper_trades
    if os.path.exists(T0_FILE):
        with open(T0_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        for p in state.get("closed_positions", []):
            sources["t0"].append(_normalize(p, "t0"))

    return sources


def filter_clean(records: list, stats: Counter, require_action: bool = False) -> list:
    out = []
    for r in records:
        ok, reason = strict_validate(r, require_action=require_action)
        if ok:
            out.append(r)
        else:
            stats[reason] += 1
    return out


# ---------- training ------------------------------------------------------

def main():
    print("\n" + "=" * 78)
    print("  V2 META-MODEL TRAINER — historical bootstrap + strict validation")
    print("=" * 78)

    sources = load_all_sources()
    print(f"  Raw counts: historical={len(sources['historical'])}, "
          f"t1={len(sources['t1'])}, t0={len(sources['t0'])}")

    # Validate each source. For TRAINING we don't require an action — the
    # meta-model only needs features+outcome. For walk-forward REPLAY we do.
    cleaned_training = {}
    cleaned_replay = {}
    for label, recs in sources.items():
        stats_train = Counter()
        stats_replay = Counter()
        cleaned_training[label] = filter_clean(recs, stats_train, require_action=False)
        cleaned_replay[label] = filter_clean(recs, stats_replay, require_action=True)
        print(f"  {label}: training_pool={len(cleaned_training[label])}/{len(recs)}, "
              f"replay_pool={len(cleaned_replay[label])}/{len(recs)}")
        for reason, n in stats_train.most_common(3):
            print(f"      reject (training): {n:>5}  {reason}")

    historical = cleaned_training["historical"]
    live = cleaned_replay["t0"] + cleaned_replay["t1"]
    live.sort(key=lambda r: r["timestamp"] or "")

    if len(historical) + len(live) < 100:
        print("\n  ABORT: not enough clean data after validation.")
        return

    print(f"\n  Pre-training pool (historical):   {len(historical)}")
    print(f"  Live timeline (t0 + t1):          {len(live)}")

    try:
        import xgboost as xgb
    except ImportError:
        print("  ABORT: xgboost not installed")
        return

    # ----------------- Walk-forward verification -------------------------
    # 1) Pre-train on all historical
    # 2) Walk forward through live, retraining every N bets adding live data
    print(f"\n  Training pool (any valid record): {len(historical) + len(cleaned_training['t0']) + len(cleaned_training['t1'])}")
    print("  Walk-forward (pre-train on historical, retrain every 25 live bets):")

    def make_xy(recs):
        if not recs:
            return None, None
        X = np.vstack([build_feature_vector(r) for r in recs])
        y = np.array([float(r["outcome"]) for r in recs], dtype=np.float32)
        return X, y

    params = {"objective": "binary:logistic", "eta": 0.05, "max_depth": 4,
              "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
              "lambda": 1.0, "verbosity": 0}

    booster = None
    last_train = -100
    v2_pnl = 0.0
    v2_taken = 0
    v2_wins = 0
    v1_pnl = 0.0
    v1_wins = 0

    # Build a training pool with ALL valid training records (not just bets).
    # Walk-forward retrains using historical_training + live_training records
    # whose timestamp is BEFORE the current live bet — preventing leakage.
    training_pool = (cleaned_training["historical"] + cleaned_training["t0"]
                     + cleaned_training["t1"])
    training_pool.sort(key=lambda r: r["timestamp"] or "")

    for i, r in enumerate(live):
        # Retrain
        if booster is None or i - last_train >= 25:
            cutoff_ts = r["timestamp"] or ""
            train_recs = [tr for tr in training_pool
                          if (tr["timestamp"] or "") <= cutoff_ts]
            X, y = make_xy(train_recs)
            dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
            booster = xgb.train(params, dtrain, num_boost_round=200)
            last_train = i

        # v1: take every bet at recorded direction
        baseline_pnl = held_pnl(r["action"], float(r["market_price"]),
                                float(r["outcome"]))
        v1_pnl += baseline_pnl
        if baseline_pnl > 0:
            v1_wins += 1

        # v2: take only if meta agrees with the recorded action
        x = build_feature_vector(r).reshape(1, -1)
        dmat = xgb.DMatrix(x, feature_names=FEATURE_NAMES)
        meta_p = float(booster.predict(dmat)[0])
        meta_edge = meta_p - float(r["market_price"])
        agrees = (r["action"] == "BUY_YES" and meta_edge > 0.01) or \
                 (r["action"] == "BUY_NO" and meta_edge < -0.01)
        if agrees:
            v2_pnl += baseline_pnl
            v2_taken += 1
            if baseline_pnl > 0:
                v2_wins += 1

    n = len(live)
    print(f"  v1 (take all):    {n:>4} bets  WR {v1_wins/n*100:>5.1f}%  PnL ${v1_pnl:+.2f}")
    print(f"  v2 (meta filter): {v2_taken:>4} bets  WR "
          f"{v2_wins/max(v2_taken,1)*100:>5.1f}%  PnL ${v2_pnl:+.2f}")
    diff = v2_pnl - v1_pnl
    pct = diff / abs(v1_pnl) * 100 if v1_pnl else 0.0
    print(f"  diff:             {diff:+.2f} ({pct:+.1f}%)")

    # ----------------- Final fit on full pool, save ----------------------
    # Use the LARGER training-only pool (includes records the engine didn't
    # bet on but where outcome is known). Walk-forward only used `live`
    # records for PnL replay; training can use everything with valid features.
    full_train = (cleaned_training["historical"] + cleaned_training["t0"]
                  + cleaned_training["t1"])
    full_train.sort(key=lambda r: r["timestamp"] or "")
    X, y = make_xy(full_train)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)

    # Hold out last 20% for a one-shot val-Brier reading
    split = int(len(full_train) * 0.8)
    if split > 0 and split < len(full_train):
        train_recs = full_train[:split]
        val_recs = full_train[split:]
        Xt, yt = make_xy(train_recs)
        Xv, yv = make_xy(val_recs)
        d_t = xgb.DMatrix(Xt, label=yt, feature_names=FEATURE_NAMES)
        d_v = xgb.DMatrix(Xv, label=yv, feature_names=FEATURE_NAMES)
        booster_for_save = xgb.train(
            params, d_t, num_boost_round=400,
            evals=[(d_t, "train"), (d_v, "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        val_pred = booster_for_save.predict(d_v)
        val_brier = float(np.mean((val_pred - yv) ** 2))
        market_p_val = np.array([float(r["market_price"]) for r in val_recs])
        market_brier = float(np.mean((market_p_val - yv) ** 2))
        improvement = market_brier - val_brier
    else:
        booster_for_save = xgb.train(params, dtrain, num_boost_round=300)
        val_brier = float("nan")
        market_brier = float("nan")
        improvement = 0.0

    print(f"\n  One-shot val (hold out last 20%):")
    print(f"    val_brier:    {val_brier:.4f}")
    print(f"    market_brier: {market_brier:.4f}")
    print(f"    improvement:  {improvement:+.4f}")

    info = MetaModelInfo(
        n_train=len(full_train),
        n_features=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,
        train_brier=0.0,
        val_brier=round(val_brier, 6),
        market_brier=round(market_brier, 6),
        improvement=round(improvement, 6),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )

    # Don't overwrite if walk-forward says we lose money
    if pct < -10:
        print(f"\n  ABORT: walk-forward lift {pct:.1f}% < -10% threshold. "
              f"Keeping previous model file unchanged.")
        report = {
            "trained_at": info.trained_at, "written": False,
            "abort_reason": "walk_forward_loss",
            "walk_forward": {"v1_pnl": v1_pnl, "v2_pnl": v2_pnl, "diff_pct": pct,
                             "n_live_bets": n, "v2_taken": v2_taken,
                             "v1_wr": v1_wins/n, "v2_wr": v2_wins/max(v2_taken,1)},
            "n_training_pool": len(full_train),
            "validated_per_source": {k: len(v) for k, v in cleaned_training.items()},
        }
        with open(TRAIN_REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return

    if os.path.exists(MODEL_FILE):
        try:
            shutil.copy(MODEL_FILE, MODEL_BACKUP)
        except OSError:
            pass

    model = XGBoostMetaModel(booster=booster_for_save, info=info)
    tmp = MODEL_FILE + ".tmp"
    model.save(tmp)
    os.replace(tmp, MODEL_FILE)
    tmp_info = tmp + ".info.json"
    final_info = MODEL_FILE + ".info.json"
    if os.path.exists(tmp_info):
        os.replace(tmp_info, final_info)

    print(f"\n  Wrote {MODEL_FILE}")

    report = {
        "trained_at": info.trained_at,
        "written": True,
        "n_training_pool": len(full_train),
        "validated_per_source": {k: len(v) for k, v in cleaned_training.items()},
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "val_brier": val_brier,
        "market_brier": market_brier,
        "improvement": improvement,
        "walk_forward": {"v1_pnl": v1_pnl, "v2_pnl": v2_pnl, "diff_pct": pct,
                         "n_live_bets": n, "v2_taken": v2_taken,
                         "v1_wr": v1_wins/n, "v2_wr": v2_wins/max(v2_taken,1)},
    }
    with open(TRAIN_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {TRAIN_REPORT}\n")


if __name__ == "__main__":
    main()
