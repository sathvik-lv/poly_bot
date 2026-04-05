"""Tests for the self-improvement system."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.self_improver import (
    PredictionTracker,
    CalibrationAuditor,
    ModelWeightOptimizer,
    SignalStrengthAnalyzer,
    SelfCorrector,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_predictions():
    """Generate realistic resolved predictions for testing."""
    np.random.seed(42)
    preds = []
    for i in range(50):
        true_prob = np.random.uniform(0.2, 0.8)
        market_price = true_prob + np.random.normal(0, 0.05)
        market_price = np.clip(market_price, 0.05, 0.95)
        predicted_prob = true_prob + np.random.normal(0, 0.08)
        predicted_prob = np.clip(predicted_prob, 0.05, 0.95)
        outcome = 1.0 if np.random.random() < true_prob else 0.0

        preds.append({
            "pred_id": f"test_{i}",
            "market_id": f"market_{i}",
            "predicted_prob": float(predicted_prob),
            "raw_prob": float(predicted_prob),
            "market_price": float(market_price),
            "edge": float(predicted_prob - market_price),
            "outcome": float(outcome),
            "resolved": True,
            "sub_model_estimates": {
                "microstructure": float(market_price + np.random.normal(0, 0.03)),
                "time_series": float(true_prob + np.random.normal(0, 0.1)),
                "external_data": float(true_prob + np.random.normal(0, 0.12)),
            },
        })
    return preds


# ===========================================================================
# PredictionTracker Tests
# ===========================================================================

class TestPredictionTracker:

    def test_record_and_load(self, tmp_dir):
        tracker = PredictionTracker(os.path.join(tmp_dir, "preds.jsonl"))
        pred = {
            "market": {"id": "m1", "question": "Test?"},
            "prediction": {"probability": 0.7},
            "edge": {"edge": 0.1},
            "sizing": {"action": "BUY_YES", "kelly_fraction": 0.05},
            "ensemble": {"weights": [0.5, 0.5], "model_names": ["a", "b"], "n_models": 2},
            "sub_models": {},
        }
        pred_id = tracker.record_prediction(pred)
        assert pred_id.startswith("pred_")

        all_preds = tracker.get_all()
        assert len(all_preds) == 1
        assert all_preds[0]["market_id"] == "m1"

    def test_record_outcome(self, tmp_dir):
        tracker = PredictionTracker(os.path.join(tmp_dir, "preds.jsonl"))
        pred = {
            "market": {"id": "m2"},
            "prediction": {"probability": 0.6},
            "edge": {"edge": 0.05},
            "sizing": {"action": "BUY_YES", "kelly_fraction": 0.03},
            "ensemble": {"weights": [], "model_names": [], "n_models": 0},
            "sub_models": {},
        }
        tracker.record_prediction(pred)
        updated = tracker.record_outcome("m2", 1.0)
        assert updated == 1

        resolved = tracker.get_resolved_predictions()
        assert len(resolved) == 1
        assert resolved[0]["outcome"] == 1.0

    def test_unresolved_tracking(self, tmp_dir):
        tracker = PredictionTracker(os.path.join(tmp_dir, "preds.jsonl"))
        for i in range(5):
            pred = {
                "market": {"id": f"m{i}"},
                "prediction": {"probability": 0.5},
                "edge": {"edge": 0},
                "sizing": {"action": "NO_BET", "kelly_fraction": 0},
                "ensemble": {"weights": [], "model_names": [], "n_models": 0},
                "sub_models": {},
            }
            tracker.record_prediction(pred)

        assert len(tracker.get_unresolved_predictions()) == 5
        tracker.record_outcome("m0", 1.0)
        tracker.record_outcome("m1", 0.0)
        assert len(tracker.get_unresolved_predictions()) == 3
        assert len(tracker.get_resolved_predictions()) == 2


# ===========================================================================
# CalibrationAuditor Tests
# ===========================================================================

class TestCalibrationAuditor:

    def test_perfect_predictor_brier_zero(self):
        auditor = CalibrationAuditor()
        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        assert auditor.brier_score(probs, outcomes) == 0.0

    def test_worst_predictor_brier_one(self):
        auditor = CalibrationAuditor()
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert auditor.brier_score(probs, outcomes) == 1.0

    def test_random_predictor_brier_around_quarter(self):
        auditor = CalibrationAuditor()
        probs = np.full(1000, 0.5)
        outcomes = np.random.binomial(1, 0.5, 1000).astype(float)
        brier = auditor.brier_score(probs, outcomes)
        assert 0.2 < brier < 0.3  # should be ~0.25

    def test_full_audit_runs(self, sample_predictions):
        auditor = CalibrationAuditor()
        result = auditor.full_audit(sample_predictions)
        assert "brier_score" in result
        assert "log_loss" in result
        assert "calibration_curve" in result
        assert "expected_calibration_error" in result
        assert "discrimination" in result
        assert result["n_predictions"] == 50

    def test_discrimination_perfect(self):
        auditor = CalibrationAuditor()
        # Perfect separation
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        outcomes = np.array([0, 0, 0, 1, 1, 1])
        disc = auditor.discrimination_score(probs, outcomes)
        assert disc > 0.9

    def test_discrimination_random(self):
        auditor = CalibrationAuditor()
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 100)
        outcomes = np.random.binomial(1, 0.5, 100).astype(float)
        disc = auditor.discrimination_score(probs, outcomes)
        assert 0.3 < disc < 0.7  # near 0.5 for random

    def test_ece_perfect_calibration(self):
        auditor = CalibrationAuditor()
        # Perfectly calibrated: 30% predictions resolve 30% of the time, etc.
        np.random.seed(42)
        n = 1000
        probs = np.random.uniform(0, 1, n)
        outcomes = np.array([1.0 if np.random.random() < p else 0.0 for p in probs])
        ece = auditor.expected_calibration_error(probs, outcomes)
        assert ece < 0.1  # should be well-calibrated

    def test_edge_hit_rate(self):
        preds = [
            {"edge": 0.1, "outcome": 1.0},  # correct
            {"edge": 0.1, "outcome": 0.0},  # wrong
            {"edge": -0.1, "outcome": 0.0},  # correct
            {"edge": -0.1, "outcome": 1.0},  # wrong
        ]
        rate = CalibrationAuditor.edge_hit_rate(preds)
        assert rate == 0.5

    def test_empty_predictions(self):
        auditor = CalibrationAuditor()
        result = auditor.full_audit([])
        assert "error" in result


# ===========================================================================
# ModelWeightOptimizer Tests
# ===========================================================================

class TestModelWeightOptimizer:

    def test_optimize_weights(self, sample_predictions):
        optimizer = ModelWeightOptimizer()
        result = optimizer.optimize_weights(sample_predictions)
        assert "weights" in result
        assert len(result["weights"]) > 0
        # Weights should sum to ~1
        assert abs(sum(result["weights"].values()) - 1.0) < 0.01

    def test_improvement_non_negative(self, sample_predictions):
        optimizer = ModelWeightOptimizer()
        result = optimizer.optimize_weights(sample_predictions)
        # Optimized should be at least as good as equal weights
        assert result.get("improvement_vs_equal", 0) >= -0.001

    def test_insufficient_data(self):
        optimizer = ModelWeightOptimizer()
        result = optimizer.optimize_weights([{"outcome": 1.0}])
        assert "error" in result


# ===========================================================================
# SignalStrengthAnalyzer Tests
# ===========================================================================

class TestSignalStrengthAnalyzer:

    def test_analyze_signals(self, sample_predictions):
        analyzer = SignalStrengthAnalyzer()
        result = analyzer.analyze_signals(sample_predictions)
        assert "signal_rankings" in result
        assert len(result["signal_rankings"]) > 0
        # Each signal should have correlation and p_value
        for sig in result["signal_rankings"]:
            assert "correlation" in sig
            assert "p_value" in sig
            assert "brier_score" in sig

    def test_insufficient_data(self):
        analyzer = SignalStrengthAnalyzer()
        result = analyzer.analyze_signals([{"outcome": 1.0}])
        assert "error" in result


# ===========================================================================
# SelfCorrector Integration Tests
# ===========================================================================

class TestSelfCorrector:

    def test_full_improvement_cycle(self, tmp_dir, sample_predictions):
        corrector = SelfCorrector(tmp_dir)
        # Seed predictions
        for p in sample_predictions:
            with open(os.path.join(tmp_dir, "predictions.jsonl"), "a") as f:
                f.write(json.dumps(p) + "\n")

        report = corrector.run_improvement_cycle()
        assert "audit" in report
        assert "optimized_weights" in report
        assert "signal_analysis" in report
        assert "recommendations" in report
        assert "health_score" in report
        assert report["health_score"] >= 0

    def test_saves_weights_and_calibration(self, tmp_dir, sample_predictions):
        corrector = SelfCorrector(tmp_dir)
        for p in sample_predictions:
            with open(os.path.join(tmp_dir, "predictions.jsonl"), "a") as f:
                f.write(json.dumps(p) + "\n")

        corrector.run_improvement_cycle()

        # Check files were saved
        assert os.path.exists(os.path.join(tmp_dir, "model_weights.json"))
        assert os.path.exists(os.path.join(tmp_dir, "latest_improvement_report.json"))

    def test_insufficient_data_handled(self, tmp_dir):
        corrector = SelfCorrector(tmp_dir)
        report = corrector.run_improvement_cycle()
        assert report["status"] == "INSUFFICIENT_DATA"
