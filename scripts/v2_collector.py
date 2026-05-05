"""V2 — parallel collector with adaptive ensemble + category gate.

Mirrors test1_collector but:
  - swaps EnsemblePredictor -> AdaptiveEnsemble (loads model_weights.json)
  - applies CategoryGate to the action decision
  - writes to data/v2_ledger.jsonl (does NOT touch test1_ledger or paper_trades)

Behavior on missing weights / missing category data: falls back to v1
behavior. So worst case v2 == v1; best case v2 > v1.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.adaptive_ensemble import AdaptiveEnsemble
from src.category_gate import CategoryGate
from src.market_client import MarketClient
from src.meta_model import XGBoostMetaModel
from src.prediction_engine import PredictionEngine
from src.shadow_ledger import compute_per_model_shadow

# Reuse helpers from test1_collector (same fetch/filter logic)
from scripts.test1_collector import (
    classify_market, parse_token_ids, compute_time_remaining,
    fetch_context_features, market_context, fetch_markets,
    DEFAULT_SCAN_SIZE, REPREDICT_HOURS, MAX_DAYS, MIN_PRICE, MAX_PRICE,
)

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "v2_ledger.jsonl")
META_MODEL_FILE = os.path.join(DATA_DIR, "meta_model.xgb")
ADAPTIVE_BLEND = float(os.environ.get("V2_ADAPTIVE_BLEND", "0.5"))
# Blend weight for meta-model when it's loaded (the rest goes to ensemble).
# 0.0 -> meta ignored, 1.0 -> meta replaces ensemble entirely.
META_BLEND = float(os.environ.get("V2_META_BLEND", "0.5"))


def load_ledger() -> list:
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


def append_record(record: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def install_adaptive_ensemble(engine: PredictionEngine, blend: float) -> None:
    """Replace the engine's static ensemble with the adaptive one in-place.

    The engine's combine() call passes (estimates, variances). The adaptive
    one accepts an optional model_names kwarg; we monkey-patch to forward it.
    """
    adaptive = AdaptiveEnsemble(blend=blend)
    original_combine = engine.ensemble.combine

    def adaptive_combine(estimates, variances, _model_names_holder=[]):
        names = _model_names_holder[0] if _model_names_holder else None
        try:
            return adaptive.combine(estimates, variances, model_names=names)
        except Exception:
            # Safety net: never let v2 break the cycle. Fall back to v1.
            return original_combine(estimates, variances)

    engine.ensemble.combine = adaptive_combine
    # Stash adaptive on engine for later inspection
    engine.ensemble._adaptive = adaptive


def scan(n_markets: int = DEFAULT_SCAN_SIZE):
    print(f"\n  V2 COLLECTOR — scanning up to {n_markets} markets")
    print(f"  Settings: <={MAX_DAYS}d  price {MIN_PRICE}-{MAX_PRICE}  "
          f"re-predict>{REPREDICT_HOURS}h  blend={ADAPTIVE_BLEND}")

    engine = PredictionEngine(backtest_mode=False)
    install_adaptive_ensemble(engine, blend=ADAPTIVE_BLEND)

    gate = CategoryGate()
    print("  Category gate state:")
    print(gate.report())

    meta = XGBoostMetaModel.load(META_MODEL_FILE)
    if meta is not None and meta.loaded:
        info = meta.info
        print(f"  Meta-model: LOADED  n_train={info.n_train}  "
              f"val_brier={info.val_brier:.4f}  market_brier={info.market_brier:.4f}  "
              f"improvement={info.improvement:+.4f}  blend={META_BLEND}")
    else:
        print("  Meta-model: not loaded (will fall back to adaptive ensemble only)")

    client = MarketClient()

    ledger = load_ledger()
    last_pred = {}
    rounds = {}
    for r in ledger:
        mid = r.get("market_id")
        if not mid:
            continue
        ts = r.get("timestamp", "")
        rd = r.get("prediction_round", 1)
        if mid not in last_pred or ts > last_pred[mid]:
            last_pred[mid] = ts
            rounds[mid] = rd

    print(f"  V2 ledger: {len(ledger)} records, {len(last_pred)} unique markets")

    raw_markets = fetch_markets(client, n_markets)
    print(f"  Fetched {len(raw_markets)} markets passing filters")

    global_ctx = fetch_context_features()

    now = datetime.now(timezone.utc)
    candidates = []
    for raw in raw_markets:
        mid = raw.get("id")
        if not mid:
            continue
        last_ts = last_pred.get(mid)
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since < REPREDICT_HOURS:
                    continue
                next_round = rounds.get(mid, 1) + 1
            except Exception:
                next_round = (rounds.get(mid, 1) or 1) + 1
        else:
            next_round = 1
        raw["_round"] = next_round
        candidates.append(raw)

    candidates.sort(key=lambda x: (x.get("_round", 1), x.get("_days_left", 999)))
    candidates = candidates[:n_markets]
    print(f"  Will predict {len(candidates)} markets")

    n_new = 0
    n_repred = 0
    n_gated_out = 0
    for i, raw in enumerate(candidates):
        try:
            t0 = time.time()
            parsed = MarketClient.parse_market(raw)
            token_ids = parse_token_ids(raw)
            parsed["_token_id"] = token_ids[0] if token_ids else None
            time_rem = compute_time_remaining(parsed.get("end_date"))

            prediction = engine.predict(
                market_data=parsed,
                time_remaining_frac=time_rem,
                token_id=parsed.get("_token_id"),
            )

            ensemble_edge = prediction["edge"]["edge"]
            ensemble_prob = prediction["prediction"]["probability"]
            price = prediction["market"]["current_price"]

            # Build market context once (also used later for the ledger record)
            ctx = market_context(raw, parsed)
            ctx.update(global_ctx)

            model_estimates_for_meta = {}
            for mname, mresult in prediction.get("sub_models", {}).items():
                est = mresult.get("estimate")
                if est is not None:
                    model_estimates_for_meta[mname] = float(est)
            meta_input = {
                "market_price": price,
                "days_left": raw.get("_days_left"),
                "context": ctx,
                "model_estimates": model_estimates_for_meta,
                "prediction_round": raw.get("_round", 1),
                "abs_edge": abs(ensemble_edge),
            }

            meta_prob = meta.predict_proba(meta_input) if meta is not None and meta.loaded else None

            if meta_prob is not None and 0.0 < META_BLEND <= 1.0:
                prob = (1 - META_BLEND) * ensemble_prob + META_BLEND * meta_prob
            else:
                prob = ensemble_prob
            edge = prob - price

            sizing = prediction.get("sizing", {}) or {}
            kelly_fraction = sizing.get("kelly_fraction", 0.0)
            kelly_full = sizing.get("kelly_full", 0.0)
            engine_action = sizing.get("action", "NO_BET")
            base_action = engine_action if engine_action != "NO_BET" else (
                "BUY_YES" if edge > 0 else ("BUY_NO" if edge < 0 else "NO_BET")
            )

            category = classify_market(parsed.get("question", ""))
            gate_decision = gate.decide(category, abs_edge=abs(edge))

            if base_action != "NO_BET" and not gate_decision["allow"]:
                action = "NO_BET"
                n_gated_out += 1
            else:
                action = base_action

            model_estimates = {}
            for mname, mresult in prediction.get("sub_models", {}).items():
                est = mresult.get("estimate")
                if est is not None:
                    model_estimates[mname] = round(float(est), 4)

            per_model_shadow = compute_per_model_shadow(model_estimates, price)
            ensemble_meta = prediction.get("ensemble", {}) or {}
            strategy_meta = prediction.get("strategy", {}) or {}

            elapsed = time.time() - t0
            round_num = raw.get("_round", 1)
            record = {
                "shadow_id": f"v2_{int(time.time()*1000)}_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction_round": round_num,
                "market_id": raw.get("id"),
                "question": parsed.get("question", ""),
                "category": category,
                "end_date": parsed.get("end_date"),
                "days_left": raw.get("_days_left"),
                "market_price": round(price, 4),
                "predicted_prob": round(prob, 4),
                "ensemble_prob": round(ensemble_prob, 4),
                "meta_prob": round(meta_prob, 4) if meta_prob is not None else None,
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "edge_confidence": prediction["edge"]["edge_confidence"],
                "action": action,
                "base_action_before_gate": base_action,
                "gate_decision": gate_decision,
                "kelly_fraction": kelly_fraction,
                "kelly_full": kelly_full,
                "models": ensemble_meta.get("model_names", []),
                "model_estimates": model_estimates,
                "ensemble_weights": ensemble_meta.get("weights"),
                "per_model_shadow": per_model_shadow,
                "strategy_meta": {
                    "should_enter": strategy_meta.get("should_enter"),
                    "regime": strategy_meta.get("regime"),
                    "is_contrarian": strategy_meta.get("is_contrarian"),
                },
                "context": ctx,
                "resolved": False,
                "outcome": None,
                "ai_reasoning": prediction.get("sub_models", {}).get(
                    "ai_semantic", {}).get("signals", {}).get("ai_reasoning"),
                "v2": True,
            }
            append_record(record)

            if round_num == 1:
                n_new += 1
                tag = "NEW"
            else:
                n_repred += 1
                tag = f"R{round_num}"

            q = parsed.get("question", "")[:55]
            gate_tag = "" if gate_decision["allow"] else " [GATED]"
            print(f"  [{i+1:>3}] {tag} {q}{gate_tag}")
            print(f"        Mkt={price:.3f} Pred={prob:.3f} Edge={edge:+.3f} "
                  f"act={action} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1:>3}] ERROR: {str(e)[:80]}")

    print(f"\n  V2 collector done — {n_new} new, {n_repred} re-pred, "
          f"{n_gated_out} category-gated out")
    print(f"  V2 ledger now has {len(ledger) + n_new + n_repred} records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-markets", type=int, default=DEFAULT_SCAN_SIZE)
    args = parser.parse_args()
    scan(n_markets=args.n_markets)


if __name__ == "__main__":
    main()
