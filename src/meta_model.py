"""XGBoost meta-model — learns to combine sub-model estimates + market context.

Inputs (per market): a fixed feature vector built from
  - market state: market_price, days_left, hour_utc, weekday
  - market microstructure context: volume_24h, liquidity, spread
  - macro context: btc_price, eth_price, fear_greed
  - sub-model estimates: microstructure, orderbook, external_data, time_series
  - meta: prediction_round (re-prediction depth)

Output: P(outcome=1) — the meta-model's calibrated probability.

Failure modes:
  - Model file missing -> caller falls back to AdaptiveEnsemble (load() returns None)
  - Feature missing -> filled with safe defaults (-1 for prices, 0.5 for probs)
  - Prediction throws -> caller catches and falls back

This is a pure-Python, locally-trained, locally-served model. No external API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Feature schema — the order MUST stay stable across train + predict.
FEATURE_NAMES = [
    "market_price",
    "days_left",
    "hour_utc",
    "weekday",
    "volume_24h",
    "liquidity",
    "spread",
    "btc_price",
    "eth_price",
    "fear_greed",
    "est_microstructure",
    "est_orderbook",
    "est_external_data",
    "est_time_series",
    "prediction_round",
    "abs_edge",
]
N_FEATURES = len(FEATURE_NAMES)


def _coerce_float(v, default: float = -1.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if not (f == f) or f in (float("inf"), float("-inf")):  # NaN/inf check
            return default
        return f
    except (TypeError, ValueError):
        return default


def build_feature_vector(record: dict) -> np.ndarray:
    """Build a feature vector from a ledger record (or live prediction context).

    Accepts both validated-ledger records (with 'market_price', 'context'
    nested dict, etc.) and live prediction inputs. Missing fields are filled
    with -1 (prices/volumes) or 0.5 (probabilities).
    """
    ctx = record.get("context", {}) if isinstance(record.get("context"), dict) else {}
    estimates = record.get("model_estimates", {}) if isinstance(record.get("model_estimates"), dict) else {}

    return np.array([
        _coerce_float(record.get("market_price"), 0.5),
        _coerce_float(record.get("days_left"), 7.0),
        _coerce_float(ctx.get("hour_utc"), 12.0),
        _coerce_float(ctx.get("weekday"), 3.0),
        _coerce_float(ctx.get("volume_24h"), 0.0),
        _coerce_float(ctx.get("liquidity"), 0.0),
        _coerce_float(ctx.get("spread"), 0.0),
        _coerce_float(ctx.get("btc_price"), -1.0),
        _coerce_float(ctx.get("eth_price"), -1.0),
        _coerce_float(ctx.get("fear_greed"), 50.0),
        _coerce_float(estimates.get("microstructure"), 0.5),
        _coerce_float(estimates.get("orderbook"), 0.5),
        _coerce_float(estimates.get("external_data"), 0.5),
        _coerce_float(estimates.get("time_series"), 0.5),
        _coerce_float(record.get("prediction_round"), 1.0),
        _coerce_float(record.get("abs_edge"), 0.0),
    ], dtype=np.float32)


@dataclass
class MetaModelInfo:
    n_train: int
    n_features: int
    feature_names: list
    train_brier: float
    val_brier: float
    market_brier: float
    improvement: float
    trained_at: str


class XGBoostMetaModel:
    """Thin wrapper. Lazy-imports xgboost to avoid hard dependency at startup."""

    def __init__(self, booster=None, info: Optional[MetaModelInfo] = None):
        self._booster = booster
        self.info = info

    @property
    def loaded(self) -> bool:
        return self._booster is not None

    @classmethod
    def load(cls, path: str) -> Optional["XGBoostMetaModel"]:
        """Load model from disk. Returns None if file missing or load fails."""
        if not os.path.exists(path):
            return None
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(path)
            info_path = path + ".info.json"
            info = None
            if os.path.exists(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                info = MetaModelInfo(**raw)
            return cls(booster=booster, info=info)
        except Exception:
            return None

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("Cannot save — model not trained")
        self._booster.save_model(path)
        if self.info is not None:
            info_path = path + ".info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(self.info.__dict__, f, indent=2)

    def predict_proba(self, record: dict) -> Optional[float]:
        """Return P(outcome=1) for a single record. None on any failure."""
        if self._booster is None:
            return None
        try:
            import xgboost as xgb
            x = build_feature_vector(record).reshape(1, -1)
            dmat = xgb.DMatrix(x, feature_names=FEATURE_NAMES)
            p = float(self._booster.predict(dmat)[0])
            return float(np.clip(p, 0.001, 0.999))
        except Exception:
            return None
