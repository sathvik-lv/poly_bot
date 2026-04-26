"""Strict data integrity checks for the self-training pipeline.

Self-training only updates model weights from records that pass EVERY check
here. Anything ambiguous, mismatched, duplicated, or out-of-bounds is
dropped with a logged reason — never silently included.

Design principle: false rejection is cheap (we have lots of records).
False acceptance corrupts the learned weights and is expensive.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional


VALID_ACTIONS = {"BUY_YES", "BUY_NO"}
VALID_OUTCOMES = {0.0, 1.0}
KNOWN_MODELS = {
    "microstructure", "time_series", "external_data",
    "orderbook", "ai_semantic",
}


@dataclass
class ValidationStats:
    seen: int = 0
    accepted: int = 0
    rejected: int = 0
    reasons: Counter = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] += 1

    def accept(self) -> None:
        self.accepted += 1

    def report(self) -> str:
        if self.seen == 0:
            return "  (no records seen)"
        lines = [
            f"  Validated {self.seen} records: "
            f"{self.accepted} accepted ({self.accepted/self.seen*100:.1f}%), "
            f"{self.rejected} rejected"
        ]
        if self.reasons:
            lines.append("  Rejection reasons:")
            for reason, n in self.reasons.most_common():
                lines.append(f"    {n:>6}  {reason}")
        return "\n".join(lines)


def _is_finite_number(x) -> bool:
    try:
        f = float(x)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def validate_record(rec: dict) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means safe to train on."""
    if not isinstance(rec, dict):
        return False, "not_a_dict"

    # --- Structural fields ---
    if not rec.get("resolved"):
        return False, "not_resolved"

    action = rec.get("action")
    if action not in VALID_ACTIONS:
        return False, f"action_not_a_bet:{action!r}"

    outcome = rec.get("outcome")
    # Outcome must be EXACTLY 0.0 or 1.0 — partial / ambiguous resolutions out
    if outcome not in VALID_OUTCOMES:
        # Coerce safely (some writers may store as int)
        if isinstance(outcome, (int, float)) and float(outcome) in VALID_OUTCOMES:
            pass
        else:
            return False, f"outcome_not_binary:{outcome!r}"

    # --- Price sanity ---
    price = rec.get("market_price")
    if not _is_finite_number(price):
        return False, "missing_or_nonfinite_market_price"
    price = float(price)
    # Strict bounds: a price at 0 or 1 means market was already resolved at
    # entry time → no genuine bet. Reject extremes that would produce
    # division-by-zero or absurd payoffs.
    if not (0.01 <= price <= 0.99):
        return False, f"price_out_of_bounds:{price:.4f}"

    pred = rec.get("predicted_prob")
    if not _is_finite_number(pred):
        return False, "missing_or_nonfinite_predicted_prob"
    pred = float(pred)
    if not (0.0 <= pred <= 1.0):
        return False, f"predicted_prob_out_of_bounds:{pred:.4f}"

    # `edge` is a derived/advisory field that the engine may zero out when
    # it decides "no entry" even if pred != price. It is NOT authoritative
    # for training — we use pred and price directly. Only check finiteness.
    edge = rec.get("edge")
    if edge is not None and not _is_finite_number(edge):
        return False, "edge_nonfinite"

    # --- Models / per-model breakdown ---
    models = rec.get("models")
    if not isinstance(models, list) or not models:
        return False, "missing_models_list"
    unknown = [m for m in models if m not in KNOWN_MODELS]
    if unknown:
        return False, f"unknown_model_name:{unknown[:1]}"

    estimates = rec.get("model_estimates", {})
    if not isinstance(estimates, dict):
        return False, "model_estimates_not_dict"
    for m in models:
        if m not in estimates:
            return False, f"model_estimate_missing:{m}"
        v = estimates[m]
        if not _is_finite_number(v):
            return False, f"model_estimate_nonfinite:{m}"
        if not (0.0 <= float(v) <= 1.0):
            return False, f"model_estimate_out_of_bounds:{m}"

    # --- Kelly fields if present ---
    for fname in ("kelly_fraction", "kelly_full"):
        v = rec.get(fname)
        if v is not None and not _is_finite_number(v):
            return False, f"{fname}_nonfinite"

    # --- Time ordering ---
    pred_ts = _parse_iso(rec.get("timestamp"))
    res_ts = _parse_iso(rec.get("resolved_at"))
    if pred_ts is None:
        return False, "missing_or_unparseable_prediction_timestamp"
    if res_ts is None:
        return False, "missing_or_unparseable_resolution_timestamp"
    if res_ts < pred_ts:
        return False, "resolution_before_prediction"
    # Sanity: predicted in the future relative to now → clock skew or fake
    now = datetime.now(timezone.utc)
    if pred_ts > now:
        return False, "prediction_timestamp_in_future"

    # --- Market identity ---
    mid = rec.get("market_id")
    if not mid or not isinstance(mid, (str, int)):
        return False, "missing_market_id"

    return True, "ok"


def iter_validated(
    records: Iterable[dict],
    stats: Optional[ValidationStats] = None,
    drop_duplicates: bool = True,
) -> Iterator[dict]:
    """Yield only records that pass validation. Optionally dedup per (market, round)."""
    if stats is None:
        stats = ValidationStats()
    seen_keys: set = set()
    for rec in records:
        stats.seen += 1
        ok, reason = validate_record(rec)
        if not ok:
            stats.reject(reason)
            continue

        if drop_duplicates:
            key = (rec.get("market_id"), rec.get("prediction_round"), rec.get("action"))
            if key in seen_keys:
                stats.reject("duplicate_market_round_action")
                continue
            seen_keys.add(key)

        stats.accept()
        yield rec


def load_validated_jsonl(
    path: str,
    drop_duplicates: bool = True,
) -> tuple[list[dict], ValidationStats]:
    """Convenience loader. Returns (validated_records, stats)."""
    stats = ValidationStats()
    if not os.path.exists(path):
        return [], stats

    def _gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    stats.seen += 1
                    stats.reject("malformed_json_line")

    return list(iter_validated(_gen(), stats=stats, drop_duplicates=drop_duplicates)), stats
