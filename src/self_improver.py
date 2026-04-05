"""Self-Testing, Self-Improvement, Self-Audit, Self-Correction System.

This module implements a closed-loop learning system that:
1. SELF-TESTS: Validates predictions against known outcomes
2. SELF-AUDITS: Detects calibration drift, model degradation, data staleness
3. SELF-CORRECTS: Adjusts model weights, recalibrates probabilities
4. SELF-IMPROVES: Learns from errors, strengthens winning signals, prunes losers

Architecture:
    PredictionTracker — Records all predictions with timestamps
    CalibrationAuditor — Measures how calibrated predictions are (Brier score, etc.)
    ModelWeightOptimizer — Adjusts ensemble weights based on historical accuracy
    SignalStrengthAnalyzer — Identifies which signals actually predict outcomes
    SelfCorrector — The master loop that ties everything together
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from src.statistics import (
    BetaBinomialModel,
    IsotonicCalibrator,
    jensen_shannon_divergence,
)


# ===========================================================================
# Prediction Tracker — Record Everything
# ===========================================================================

class PredictionTracker:
    """Persistent storage for predictions and outcomes.

    Stores every prediction made, and later records the actual outcome.
    This data feeds the self-improvement loop.
    """

    def __init__(self, storage_path: str = "data/predictions.jsonl"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path) if os.path.dirname(storage_path) else ".", exist_ok=True)

    def record_prediction(self, prediction: dict) -> str:
        """Store a prediction. Returns a unique prediction ID."""
        record = {
            "pred_id": f"pred_{int(time.time() * 1000)}_{id(prediction) % 10000}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": prediction.get("market", {}).get("id"),
            "question": prediction.get("market", {}).get("question", ""),
            "predicted_prob": prediction.get("prediction", {}).get("probability"),
            "raw_prob": prediction.get("prediction", {}).get("raw_probability"),
            "market_price": prediction.get("market", {}).get("current_price"),
            "edge": prediction.get("edge", {}).get("edge"),
            "action": prediction.get("sizing", {}).get("action"),
            "kelly_fraction": prediction.get("sizing", {}).get("kelly_fraction"),
            "model_weights": prediction.get("ensemble", {}).get("weights"),
            "model_names": prediction.get("ensemble", {}).get("model_names"),
            "n_models": prediction.get("ensemble", {}).get("n_models"),
            "sub_model_estimates": {
                name: result.get("estimate")
                for name, result in prediction.get("sub_models", {}).items()
                if result.get("estimate") is not None
            },
            "outcome": None,  # filled later
            "resolved": False,
        }
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record["pred_id"]

    def record_outcome(self, market_id: str, outcome: float) -> int:
        """Record the actual outcome for a market. Returns number of records updated."""
        records = self._load_all()
        updated = 0
        for r in records:
            if r.get("market_id") == market_id and not r.get("resolved"):
                r["outcome"] = outcome
                r["resolved"] = True
                r["resolved_at"] = datetime.now(timezone.utc).isoformat()
                updated += 1
        self._save_all(records)
        return updated

    def get_resolved_predictions(self) -> list[dict]:
        """Get all predictions that have known outcomes."""
        return [r for r in self._load_all() if r.get("resolved")]

    def get_unresolved_predictions(self) -> list[dict]:
        return [r for r in self._load_all() if not r.get("resolved")]

    def get_all(self) -> list[dict]:
        return self._load_all()

    def _load_all(self) -> list[dict]:
        if not os.path.exists(self.storage_path):
            return []
        records = []
        with open(self.storage_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _save_all(self, records: list[dict]):
        with open(self.storage_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


# ===========================================================================
# Calibration Auditor — How Good Are We?
# ===========================================================================

class CalibrationAuditor:
    """Measure prediction quality using multiple scoring rules.

    Implements:
    - Brier Score: Mean squared error of probability estimates (lower = better)
    - Log Loss: Logarithmic scoring (more punishing for confident wrong predictions)
    - Calibration Curve: How well do predicted probabilities match actual frequencies?
    - Discrimination: Can the model distinguish between outcomes?
    - Expected Calibration Error (ECE): Industry-standard calibration metric
    """

    def full_audit(self, predictions: list[dict]) -> dict:
        """Run complete audit on resolved predictions."""
        if not predictions:
            return {"error": "No resolved predictions to audit", "n_predictions": 0}

        probs = np.array([p["predicted_prob"] for p in predictions if p.get("predicted_prob") is not None])
        outcomes = np.array([p["outcome"] for p in predictions if p.get("outcome") is not None])

        if len(probs) == 0 or len(outcomes) == 0:
            return {"error": "No valid prediction/outcome pairs", "n_predictions": 0}

        # Align arrays
        n = min(len(probs), len(outcomes))
        probs = probs[:n]
        outcomes = outcomes[:n]

        return {
            "n_predictions": n,
            "brier_score": self.brier_score(probs, outcomes),
            "log_loss": self.log_loss(probs, outcomes),
            "calibration_curve": self.calibration_curve(probs, outcomes),
            "expected_calibration_error": self.expected_calibration_error(probs, outcomes),
            "discrimination": self.discrimination_score(probs, outcomes),
            "profit_if_bet": self.simulated_profit(predictions[:n]),
            "roi": self.return_on_investment(predictions[:n]),
            "mean_edge": float(np.mean([p.get("edge", 0) or 0 for p in predictions[:n]])),
            "edge_hit_rate": self.edge_hit_rate(predictions[:n]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Brier Score: mean((prob - outcome)^2). Perfect = 0, worst = 1."""
        return float(np.mean((probs - outcomes) ** 2))

    @staticmethod
    def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Log Loss: -mean(outcome*log(prob) + (1-outcome)*log(1-prob))."""
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))

    @staticmethod
    def calibration_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[dict]:
        """Calibration curve: predicted probability vs actual frequency per bin."""
        bins = np.linspace(0, 1, n_bins + 1)
        curve = []
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                curve.append({
                    "bin_center": round((bins[i] + bins[i + 1]) / 2, 2),
                    "mean_predicted": round(float(np.mean(probs[mask])), 4),
                    "mean_actual": round(float(np.mean(outcomes[mask])), 4),
                    "count": int(mask.sum()),
                    "gap": round(abs(float(np.mean(probs[mask])) - float(np.mean(outcomes[mask]))), 4),
                })
        return curve

    @staticmethod
    def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """ECE: weighted average of |predicted - actual| across bins."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                bin_acc = np.mean(outcomes[mask])
                bin_conf = np.mean(probs[mask])
                ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def discrimination_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
        """AUC-ROC equivalent: can the model separate Yes from No outcomes?"""
        pos_probs = probs[outcomes == 1]
        neg_probs = probs[outcomes == 0]
        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return 0.5
        # Mann-Whitney U test
        u_stat, _ = sp_stats.mannwhitneyu(pos_probs, neg_probs, alternative="greater")
        auc = u_stat / (len(pos_probs) * len(neg_probs))
        return float(auc)

    @staticmethod
    def simulated_profit(predictions: list[dict]) -> float:
        """Simulate profit if we bet $1 on every prediction with positive edge."""
        total_profit = 0.0
        for p in predictions:
            edge = p.get("edge", 0) or 0
            outcome = p.get("outcome")
            market_price = p.get("market_price", 0.5)
            if edge > 0 and outcome is not None:
                # Bought YES at market_price
                if outcome == 1:
                    total_profit += (1 - market_price)
                else:
                    total_profit -= market_price
            elif edge < 0 and outcome is not None:
                # Bought NO at (1 - market_price)
                if outcome == 0:
                    total_profit += market_price
                else:
                    total_profit -= (1 - market_price)
        return round(total_profit, 4)

    @staticmethod
    def return_on_investment(predictions: list[dict]) -> Optional[float]:
        """ROI: profit / total amount wagered."""
        total_wagered = 0.0
        total_profit = 0.0
        for p in predictions:
            edge = p.get("edge", 0) or 0
            outcome = p.get("outcome")
            market_price = p.get("market_price", 0.5)
            if outcome is None:
                continue
            if edge > 0:
                total_wagered += market_price
                if outcome == 1:
                    total_profit += (1 - market_price)
                else:
                    total_profit -= market_price
            elif edge < 0:
                total_wagered += (1 - market_price)
                if outcome == 0:
                    total_profit += market_price
                else:
                    total_profit -= (1 - market_price)
        if total_wagered == 0:
            return None
        return round(total_profit / total_wagered, 4)

    @staticmethod
    def edge_hit_rate(predictions: list[dict]) -> Optional[float]:
        """How often does positive edge lead to correct prediction?"""
        correct = 0
        total = 0
        for p in predictions:
            edge = p.get("edge", 0) or 0
            outcome = p.get("outcome")
            if outcome is None or abs(edge) < 0.01:
                continue
            total += 1
            if (edge > 0 and outcome == 1) or (edge < 0 and outcome == 0):
                correct += 1
        if total == 0:
            return None
        return round(correct / total, 4)


# ===========================================================================
# Model Weight Optimizer — Learn Which Models Work Best
# ===========================================================================

class ModelWeightOptimizer:
    """Optimize ensemble weights based on historical performance.

    Uses the tracked predictions to determine which sub-models
    have been most accurate, and adjusts their weights accordingly.
    """

    def optimize_weights(self, predictions: list[dict]) -> dict:
        """Find optimal model weights by minimizing Brier score.

        Args:
            predictions: Resolved predictions with sub_model_estimates and outcomes

        Returns:
            Optimized weight for each model name
        """
        # Collect sub-model estimates and outcomes
        model_names = set()
        for p in predictions:
            model_names.update(p.get("sub_model_estimates", {}).keys())
        model_names = sorted(model_names)

        if not model_names or len(predictions) < 5:
            return {"weights": {}, "n_predictions": len(predictions), "error": "insufficient data"}

        # Build matrices
        n = len(predictions)
        k = len(model_names)
        X = np.zeros((n, k))  # model estimates
        y = np.zeros(n)  # outcomes

        valid_mask = np.ones(n, dtype=bool)
        for i, p in enumerate(predictions):
            if p.get("outcome") is None:
                valid_mask[i] = False
                continue
            y[i] = p["outcome"]
            estimates = p.get("sub_model_estimates", {})
            for j, name in enumerate(model_names):
                if name in estimates and estimates[name] is not None:
                    X[i, j] = estimates[name]
                else:
                    X[i, j] = 0.5  # default for missing

        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < 5:
            return {"weights": {}, "n_predictions": len(y), "error": "insufficient valid data"}

        # Optimize: minimize Brier score subject to weights summing to 1
        def brier_with_weights(w):
            w = np.abs(w) / np.abs(w).sum()  # softmax normalization
            combined = X @ w
            return float(np.mean((combined - y) ** 2))

        x0 = np.ones(k) / k  # equal weights start
        result = minimize(brier_with_weights, x0, method="Nelder-Mead",
                         options={"maxiter": 1000, "xatol": 1e-6})

        optimal_w = np.abs(result.x) / np.abs(result.x).sum()

        # Per-model Brier scores
        model_scores = {}
        for j, name in enumerate(model_names):
            model_brier = float(np.mean((X[:, j] - y) ** 2))
            model_scores[name] = round(model_brier, 4)

        return {
            "weights": {name: round(float(w), 4) for name, w in zip(model_names, optimal_w)},
            "model_brier_scores": model_scores,
            "combined_brier": round(float(result.fun), 4),
            "n_predictions": len(y),
            "improvement_vs_equal": round(float(brier_with_weights(np.ones(k) / k) - result.fun), 6),
        }


# ===========================================================================
# Signal Strength Analyzer — What Actually Predicts Outcomes?
# ===========================================================================

class SignalStrengthAnalyzer:
    """Analyze which signals from sub-models actually predict outcomes.

    Uses point-biserial correlation and information gain to rank signals.
    """

    def analyze_signals(self, predictions: list[dict]) -> dict:
        """Rank all signals by their predictive power."""
        # Collect all signal names
        all_signals = {}
        outcomes = []

        for p in predictions:
            if p.get("outcome") is None:
                continue
            outcomes.append(p["outcome"])
            idx = len(outcomes) - 1
            # We store sub-model estimates as our signals for now
            for name, est in p.get("sub_model_estimates", {}).items():
                if est is not None:
                    if name not in all_signals:
                        all_signals[name] = {}
                    all_signals[name][idx] = est

        if len(outcomes) < 10:
            return {"error": "Need at least 10 resolved predictions", "n": len(outcomes)}

        outcomes = np.array(outcomes)
        signal_rankings = []

        for signal_name, values in all_signals.items():
            indices = sorted(values.keys())
            if len(indices) < 5:
                continue
            signal_vals = np.array([values[i] for i in indices])
            signal_outcomes = outcomes[indices]

            # Point-biserial correlation
            try:
                corr, p_value = sp_stats.pointbiserialr(signal_outcomes, signal_vals)
            except Exception:
                corr, p_value = 0.0, 1.0

            # Brier score for this signal alone
            brier = float(np.mean((signal_vals - signal_outcomes) ** 2))

            signal_rankings.append({
                "signal": signal_name,
                "correlation": round(float(corr), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05,
                "brier_score": round(brier, 4),
                "n_observations": len(indices),
                "mean_estimate": round(float(np.mean(signal_vals)), 4),
            })

        # Sort by correlation strength
        signal_rankings.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "signal_rankings": signal_rankings,
            "n_predictions": len(outcomes),
            "top_signal": signal_rankings[0]["signal"] if signal_rankings else None,
            "n_significant": sum(1 for s in signal_rankings if s["significant"]),
        }


# ===========================================================================
# Self-Corrector — The Master Improvement Loop
# ===========================================================================

class SelfCorrector:
    """Master self-improvement system.

    Runs the full audit → optimize → correct → improve cycle.

    Outputs:
    - Updated model weights (saved to data/model_weights.json)
    - Updated calibration data (saved to data/calibration.json)
    - Performance report
    - Recommendations for which models to trust/distrust
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.tracker = PredictionTracker(os.path.join(data_dir, "predictions.jsonl"))
        self.auditor = CalibrationAuditor()
        self.weight_optimizer = ModelWeightOptimizer()
        self.signal_analyzer = SignalStrengthAnalyzer()
        os.makedirs(data_dir, exist_ok=True)

    def run_improvement_cycle(self) -> dict:
        """Execute one full self-improvement cycle.

        1. Audit current performance
        2. Identify weak spots
        3. Optimize model weights
        4. Update calibration
        5. Generate recommendations
        """
        resolved = self.tracker.get_resolved_predictions()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_resolved": len(resolved),
            "n_unresolved": len(self.tracker.get_unresolved_predictions()),
        }

        if len(resolved) < 5:
            report["status"] = "INSUFFICIENT_DATA"
            report["message"] = f"Need at least 5 resolved predictions, have {len(resolved)}"
            return report

        # Step 1: Full audit
        audit = self.auditor.full_audit(resolved)
        report["audit"] = audit

        # Step 2: Optimize weights
        weight_result = self.weight_optimizer.optimize_weights(resolved)
        report["optimized_weights"] = weight_result

        # Save optimized weights
        if weight_result.get("weights"):
            self._save_json("model_weights.json", weight_result["weights"])

        # Step 3: Signal analysis
        signal_result = self.signal_analyzer.analyze_signals(resolved)
        report["signal_analysis"] = signal_result

        # Step 4: Update calibration data
        probs = np.array([p["predicted_prob"] for p in resolved if p.get("predicted_prob") is not None])
        outcomes = np.array([p["outcome"] for p in resolved if p.get("outcome") is not None])
        n = min(len(probs), len(outcomes))
        if n >= 10:
            calibrator = IsotonicCalibrator()
            calibrator.fit(probs[:n], outcomes[:n])
            cal_data = {"x": probs[:n].tolist(), "y": outcomes[:n].tolist()}
            self._save_json("calibration.json", cal_data)
            report["calibration_updated"] = True

        # Step 5: Generate recommendations
        report["recommendations"] = self._generate_recommendations(audit, weight_result, signal_result)

        # Step 6: Overall health score
        report["health_score"] = self._compute_health_score(audit)

        # Save report
        self._save_json("latest_improvement_report.json", report)

        return report

    def get_optimized_weights(self) -> Optional[dict]:
        """Load previously optimized model weights."""
        return self._load_json("model_weights.json")

    def get_calibration_data(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Load calibration data for the prediction engine."""
        data = self._load_json("calibration.json")
        if data and "x" in data and "y" in data:
            return np.array(data["x"]), np.array(data["y"])
        return None

    def _generate_recommendations(self, audit: dict, weights: dict, signals: dict) -> list[str]:
        """Generate actionable recommendations from the audit."""
        recs = []

        # Brier score check
        brier = audit.get("brier_score", 1.0)
        if brier < 0.15:
            recs.append("EXCELLENT: Brier score < 0.15 — predictions are well-calibrated")
        elif brier < 0.25:
            recs.append("GOOD: Brier score < 0.25 — room for improvement but performing decently")
        else:
            recs.append(f"WARNING: Brier score = {brier:.3f} — predictions need significant improvement")

        # Calibration check
        ece = audit.get("expected_calibration_error", 1.0)
        if ece > 0.1:
            recs.append(f"RECALIBRATE: ECE = {ece:.3f} — isotonic calibration should be retrained")

        # Discrimination check
        disc = audit.get("discrimination", 0.5)
        if disc < 0.6:
            recs.append("LOW DISCRIMINATION: Model can't distinguish Yes from No outcomes well")
        elif disc > 0.8:
            recs.append("STRONG DISCRIMINATION: Model effectively separates outcomes")

        # Edge hit rate
        ehr = audit.get("edge_hit_rate")
        if ehr is not None:
            if ehr < 0.5:
                recs.append(f"EDGE FAILING: Edge hit rate = {ehr:.1%} — edge detection is worse than random")
            elif ehr > 0.6:
                recs.append(f"EDGE WORKING: Edge hit rate = {ehr:.1%} — edge detection adds value")

        # Model weight recommendations
        model_weights = weights.get("weights", {})
        if model_weights:
            best_model = max(model_weights, key=model_weights.get)
            worst_model = min(model_weights, key=model_weights.get)
            recs.append(f"BEST MODEL: '{best_model}' (weight: {model_weights[best_model]:.3f})")
            if model_weights[worst_model] < 0.05:
                recs.append(f"CONSIDER DROPPING: '{worst_model}' has negligible weight ({model_weights[worst_model]:.3f})")

        # Signal recommendations
        if signals.get("signal_rankings"):
            sig_recs = signals["signal_rankings"]
            strong = [s for s in sig_recs if s.get("significant")]
            if strong:
                recs.append(f"STRONG SIGNALS: {', '.join(s['signal'] for s in strong[:3])}")
            weak = [s for s in sig_recs if not s.get("significant")]
            if weak:
                recs.append(f"WEAK SIGNALS (consider removing): {', '.join(s['signal'] for s in weak[:2])}")

        # Profitability
        roi = audit.get("roi")
        if roi is not None:
            if roi > 0:
                recs.append(f"PROFITABLE: ROI = {roi:.1%}")
            else:
                recs.append(f"UNPROFITABLE: ROI = {roi:.1%} — strategy needs adjustment")

        return recs

    def _compute_health_score(self, audit: dict) -> float:
        """Compute overall health score (0-100)."""
        score = 50.0  # start neutral

        # Brier score component (0-30 points)
        brier = audit.get("brier_score", 0.5)
        score += max(0, 30 * (1 - brier / 0.25))

        # Discrimination component (0-20 points)
        disc = audit.get("discrimination", 0.5)
        score += max(0, 20 * (disc - 0.5) / 0.5)

        # Calibration component (0-20 points)
        ece = audit.get("expected_calibration_error", 0.5)
        score += max(0, 20 * (1 - ece / 0.1))

        # ROI component (0-30 points, can go negative)
        roi = audit.get("roi")
        if roi is not None:
            score += min(30, max(-20, roi * 100))

        return round(max(0, min(100, score)), 1)

    def _save_json(self, filename: str, data):
        path = os.path.join(self.data_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, filename: str) -> Optional[dict]:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
