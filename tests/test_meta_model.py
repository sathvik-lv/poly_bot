"""Tests for src/meta_model.py — XGBoost meta-model wrapper.

xgboost is an optional dependency at test time; tests that need it skip
cleanly when it's not installed. The non-xgb-dependent paths (feature
vector builder, safe load fallback) always run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.meta_model import (
    FEATURE_NAMES, N_FEATURES, XGBoostMetaModel,
    build_feature_vector,
)


class TestFeatureBuilder:
    def test_full_record_produces_correct_length(self):
        rec = {
            "market_price": 0.42,
            "days_left": 3.5,
            "prediction_round": 2,
            "abs_edge": 0.07,
            "context": {
                "hour_utc": 9, "weekday": 2,
                "volume_24h": 12000.0, "liquidity": 5000.0, "spread": 0.01,
                "btc_price": 70000.0, "eth_price": 3500.0, "fear_greed": 45,
            },
            "model_estimates": {
                "microstructure": 0.55, "orderbook": 0.5,
                "external_data": 0.5, "time_series": 0.49,
            },
        }
        v = build_feature_vector(rec)
        assert v.shape == (N_FEATURES,)
        assert v[FEATURE_NAMES.index("market_price")] == pytest.approx(0.42)
        assert v[FEATURE_NAMES.index("est_microstructure")] == pytest.approx(0.55)
        assert v[FEATURE_NAMES.index("fear_greed")] == pytest.approx(45)

    def test_missing_fields_default_safely(self):
        v = build_feature_vector({})
        assert v.shape == (N_FEATURES,)
        # market_price defaults to 0.5
        assert v[FEATURE_NAMES.index("market_price")] == pytest.approx(0.5)
        # btc_price defaults to -1
        assert v[FEATURE_NAMES.index("btc_price")] == pytest.approx(-1.0)
        # fear_greed defaults to 50 (neutral)
        assert v[FEATURE_NAMES.index("fear_greed")] == pytest.approx(50.0)

    def test_nan_inf_stripped(self):
        v = build_feature_vector({
            "market_price": float("nan"),
            "days_left": float("inf"),
            "context": {"volume_24h": float("-inf")},
        })
        # NaN/inf coerced back to defaults
        assert v[FEATURE_NAMES.index("market_price")] == pytest.approx(0.5)
        assert v[FEATURE_NAMES.index("days_left")] == pytest.approx(7.0)
        assert v[FEATURE_NAMES.index("volume_24h")] == pytest.approx(0.0)
        assert all(math.isfinite(float(x)) for x in v)

    def test_string_inputs_coerced(self):
        v = build_feature_vector({
            "market_price": "0.6", "days_left": "10",
            "context": {"hour_utc": "23"},
        })
        assert v[FEATURE_NAMES.index("market_price")] == pytest.approx(0.6)
        assert v[FEATURE_NAMES.index("hour_utc")] == pytest.approx(23.0)

    def test_garbage_inputs_dont_crash(self):
        v = build_feature_vector({
            "market_price": "not a number",
            "context": "this should be a dict",
            "model_estimates": [1, 2, 3],  # wrong type
        })
        assert v.shape == (N_FEATURES,)
        assert all(math.isfinite(float(x)) for x in v)


class TestModelLoadFallback:
    def test_load_missing_file_returns_none(self):
        assert XGBoostMetaModel.load("/nonexistent/path/model.xgb") is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        bad = tmp_path / "model.xgb"
        bad.write_bytes(b"not a real xgboost model")
        assert XGBoostMetaModel.load(str(bad)) is None

    def test_unloaded_model_predict_returns_none(self):
        m = XGBoostMetaModel(booster=None)
        assert not m.loaded
        assert m.predict_proba({"market_price": 0.5}) is None


# These tests exercise the full train + load + predict cycle.
# xgboost is optional — skip if not available.
xgb = pytest.importorskip("xgboost")


class TestTrainPredictRoundTrip:
    def _toy_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        bets = []
        for i in range(n):
            price = float(rng.uniform(0.2, 0.8))
            est_micro = float(np.clip(price + rng.normal(0, 0.05), 0.05, 0.95))
            # Outcome strongly correlated with est_micro (so model can learn)
            outcome = 1.0 if est_micro > 0.5 + rng.normal(0, 0.05) else 0.0
            bets.append({
                "market_price": price,
                "days_left": 3.0,
                "prediction_round": 1,
                "abs_edge": abs(est_micro - price),
                "context": {
                    "hour_utc": 12, "weekday": 3,
                    "volume_24h": 1000.0, "liquidity": 500.0, "spread": 0.01,
                    "btc_price": 70000.0, "eth_price": 3500.0, "fear_greed": 50,
                },
                "model_estimates": {
                    "microstructure": est_micro,
                    "orderbook": float(rng.uniform(0.4, 0.6)),
                    "external_data": float(rng.uniform(0.4, 0.6)),
                    "time_series": 0.5,
                },
                "outcome": outcome,
            })
        return bets

    def test_train_and_predict(self, tmp_path):
        from src.meta_model import MetaModelInfo
        bets = self._toy_data(n=200)
        X = np.vstack([build_feature_vector(b) for b in bets])
        y = np.array([b["outcome"] for b in bets], dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
        booster = xgb.train(
            {"objective": "binary:logistic", "eta": 0.2, "max_depth": 3, "verbosity": 0},
            dtrain,
            num_boost_round=20,
        )
        info = MetaModelInfo(
            n_train=200, n_features=N_FEATURES, feature_names=FEATURE_NAMES,
            train_brier=0.1, val_brier=0.12, market_brier=0.18, improvement=0.06,
            trained_at="2026-04-26T00:00:00+00:00",
        )
        m = XGBoostMetaModel(booster=booster, info=info)
        path = str(tmp_path / "model.xgb")
        m.save(path)

        # Reload and predict
        m2 = XGBoostMetaModel.load(path)
        assert m2 is not None and m2.loaded
        assert m2.info is not None
        assert m2.info.n_train == 200

        p = m2.predict_proba(bets[0])
        assert p is not None
        assert 0.0 < p < 1.0
