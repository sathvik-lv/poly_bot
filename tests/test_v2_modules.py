"""Tests for the v2 stack: data_validator, adaptive_ensemble, category_gate."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.adaptive_ensemble import AdaptiveEnsemble, load_learned_weights
from src.category_gate import CategoryGate, _wilson_ci
from src.data_validator import (
    KNOWN_MODELS, ValidationStats, iter_validated, validate_record,
)


# ---------- helpers --------------------------------------------------------

def _good_record(**overrides):
    now = datetime.now(timezone.utc)
    rec = {
        "shadow_id": "t_test_1",
        "timestamp": (now - timedelta(hours=2)).isoformat(),
        "resolved_at": now.isoformat(),
        "prediction_round": 1,
        "market_id": "12345",
        "question": "Will X happen?",
        "category": "other",
        "end_date": (now - timedelta(hours=1)).isoformat(),
        "market_price": 0.40,
        "predicted_prob": 0.55,
        "edge": 0.15,
        "action": "BUY_YES",
        "kelly_fraction": 0.05,
        "kelly_full": 0.40,
        "models": ["microstructure", "external_data"],
        "model_estimates": {"microstructure": 0.60, "external_data": 0.50},
        "resolved": True,
        "outcome": 1.0,
    }
    rec.update(overrides)
    return rec


# ---------- data_validator -------------------------------------------------

class TestValidator:
    def test_good_record_accepted(self):
        ok, reason = validate_record(_good_record())
        assert ok, reason
        assert reason == "ok"

    def test_unresolved_rejected(self):
        ok, _ = validate_record(_good_record(resolved=False))
        assert not ok

    def test_outcome_must_be_binary(self):
        ok, reason = validate_record(_good_record(outcome=0.5))
        assert not ok
        assert "outcome_not_binary" in reason

    def test_price_zero_rejected(self):
        ok, _ = validate_record(_good_record(market_price=0.0))
        assert not ok

    def test_price_one_rejected(self):
        ok, _ = validate_record(_good_record(market_price=1.0))
        assert not ok

    def test_missing_market_id(self):
        rec = _good_record()
        rec.pop("market_id")
        ok, _ = validate_record(rec)
        assert not ok

    def test_unknown_model_rejected(self):
        ok, reason = validate_record(_good_record(models=["microstructure", "fake_model"]))
        assert not ok
        assert "unknown_model_name" in reason

    def test_nonfinite_edge_rejected(self):
        ok, reason = validate_record(_good_record(edge=float("nan")))
        assert not ok
        assert "edge_nonfinite" in reason

    def test_resolution_before_prediction_rejected(self):
        now = datetime.now(timezone.utc)
        ok, _ = validate_record(_good_record(
            timestamp=now.isoformat(),
            resolved_at=(now - timedelta(hours=1)).isoformat(),
        ))
        assert not ok

    def test_future_prediction_rejected(self):
        future = datetime.now(timezone.utc) + timedelta(days=1)
        ok, _ = validate_record(_good_record(timestamp=future.isoformat()))
        assert not ok

    def test_missing_model_estimate_rejected(self):
        ok, _ = validate_record(_good_record(
            models=["microstructure", "external_data"],
            model_estimates={"microstructure": 0.60},  # external_data missing
        ))
        assert not ok

    def test_iter_dedups(self):
        rec = _good_record()
        records = [rec, rec, rec]
        stats = ValidationStats()
        out = list(iter_validated(records, stats=stats))
        assert len(out) == 1
        assert stats.rejected == 2

    def test_iter_keeps_distinct_actions(self):
        rec_yes = _good_record(action="BUY_YES")
        rec_no = _good_record(action="BUY_NO", market_price=0.6,
                              predicted_prob=0.4, edge=-0.2,
                              model_estimates={"microstructure": 0.40, "external_data": 0.42})
        out = list(iter_validated([rec_yes, rec_no]))
        assert len(out) == 2

    def test_known_models_set_is_complete(self):
        assert KNOWN_MODELS == {
            "microstructure", "time_series", "external_data",
            "orderbook", "ai_semantic",
        }


# ---------- adaptive_ensemble ----------------------------------------------

class TestAdaptiveEnsemble:
    def test_no_weights_falls_back_to_iv(self, tmp_path):
        empty = tmp_path / "weights.json"
        empty.write_text("{}")
        ens = AdaptiveEnsemble(weights_path=str(empty), blend=0.5)
        out = ens.combine(
            estimates=[0.6, 0.4],
            variances=[0.01, 0.04],
            model_names=["microstructure", "external_data"],
        )
        assert out["blend_used"] == 0.0
        assert out["learned_weights"] is None
        # Inverse-variance: weights ∝ 1/var → weights = [0.8, 0.2]
        assert out["weights"][0] > out["weights"][1]

    def test_blend_uses_learned_weights(self, tmp_path):
        wf = tmp_path / "weights.json"
        wf.write_text(json.dumps({
            "microstructure": 0.9,  # heavily favored
            "external_data": 0.1,
            "orderbook": 0.0,
        }))
        ens = AdaptiveEnsemble(weights_path=str(wf), blend=1.0)
        out = ens.combine(
            estimates=[0.7, 0.3],
            variances=[0.04, 0.01],  # IV would favor external_data
            model_names=["microstructure", "external_data"],
        )
        # With blend=1.0, learned weights dominate → microstructure favored
        assert out["weights"][0] > out["weights"][1]

    def test_malformed_weights_file_safe(self, tmp_path):
        bad = tmp_path / "weights.json"
        bad.write_text("{not valid json")
        ens = AdaptiveEnsemble(weights_path=str(bad), blend=0.8)
        out = ens.combine(
            [0.6, 0.4], [0.01, 0.04],
            model_names=["microstructure", "external_data"],
        )
        # Falls back, no crash
        assert out["blend_used"] == 0.0

    def test_missing_weights_file_safe(self):
        ens = AdaptiveEnsemble(weights_path="/nonexistent/path.json", blend=0.5)
        out = ens.combine([0.5], [0.01], model_names=["microstructure"])
        assert out["blend_used"] == 0.0

    def test_negative_weights_dropped(self, tmp_path):
        wf = tmp_path / "weights.json"
        wf.write_text(json.dumps({"microstructure": -0.5, "external_data": 0.5}))
        ens = AdaptiveEnsemble(weights_path=str(wf), blend=1.0)
        # microstructure should be ignored; only external_data has weight
        weights = load_learned_weights(str(wf))
        assert "microstructure" not in weights
        assert weights["external_data"] == 0.5


# ---------- category_gate --------------------------------------------------

class TestCategoryGate:
    def test_wilson_ci_zero_n(self):
        lo, hi = _wilson_ci(0, 0)
        assert (lo, hi) == (0.0, 1.0)

    def test_wilson_ci_clear_winner(self):
        lo, hi = _wilson_ci(80, 100)
        assert lo > 0.65 and hi < 0.95

    def test_no_data_allows_all(self, tmp_path):
        gate = CategoryGate(ledger_paths=[str(tmp_path / "missing.jsonl")])
        decision = gate.decide("sports", abs_edge=0.03)
        assert decision["allow"] is True
        assert decision["category_n"] == 0

    def test_blocks_clearly_losing_category(self, tmp_path):
        # Build a synthetic ledger where "sports" loses 80/100 (only 20% WR)
        ledger_path = tmp_path / "ledger.jsonl"
        records = []
        now = datetime.now(timezone.utc)
        for i in range(100):
            outcome = 0.0 if i < 80 else 1.0
            records.append({
                "resolved": True,
                "action": "BUY_YES",
                "outcome": outcome,
                "market_price": 0.50,
                "category": "sports",
                "market_id": f"s_{i}",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "resolved_at": now.isoformat(),
            })
        ledger_path.write_text("\n".join(json.dumps(r) for r in records))

        gate = CategoryGate(
            ledger_paths=[str(ledger_path)],
            min_trades_for_decision=30,
            block_threshold=0.48,
        )
        decision = gate.decide("sports", abs_edge=0.10)
        assert decision["allow"] is False, decision

    def test_strong_category_passes(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        records = []
        now = datetime.now(timezone.utc)
        for i in range(60):
            outcome = 1.0 if i < 50 else 0.0  # 83% WR
            records.append({
                "resolved": True,
                "action": "BUY_YES",
                "outcome": outcome,
                "market_price": 0.50,
                "category": "other",
                "market_id": f"o_{i}",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "resolved_at": now.isoformat(),
            })
        ledger_path.write_text("\n".join(json.dumps(r) for r in records))

        gate = CategoryGate(ledger_paths=[str(ledger_path)],
                            min_trades_for_decision=30,
                            ambiguous_threshold=0.55)
        decision = gate.decide("other", abs_edge=0.03)
        assert decision["allow"] is True

    def test_ambiguous_requires_larger_edge(self, tmp_path):
        # 26 wins / 50 = 52% — CI straddles 50%
        ledger_path = tmp_path / "ledger.jsonl"
        records = []
        now = datetime.now(timezone.utc)
        for i in range(50):
            outcome = 1.0 if i < 26 else 0.0
            records.append({
                "resolved": True,
                "action": "BUY_YES",
                "outcome": outcome,
                "market_price": 0.50,
                "category": "geopolitics",
                "market_id": f"g_{i}",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "resolved_at": now.isoformat(),
            })
        ledger_path.write_text("\n".join(json.dumps(r) for r in records))

        gate = CategoryGate(
            ledger_paths=[str(ledger_path)],
            min_trades_for_decision=30,
            ambiguous_threshold=0.70,
            ambiguous_edge_required=0.05,
        )
        # Small edge → blocked even though category isn't dead
        small_edge = gate.decide("geopolitics", abs_edge=0.02)
        assert small_edge["allow"] is False
        # Large edge → allowed
        large_edge = gate.decide("geopolitics", abs_edge=0.10)
        assert large_edge["allow"] is True
