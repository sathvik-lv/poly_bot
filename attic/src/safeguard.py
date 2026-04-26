"""Safeguard System — Corruption Detection, Rollback & Alternative Approaches.

Monitors training health and automatically:
1. Snapshots the best-performing state
2. Detects performance regression or data corruption
3. Rolls back to best state when things go wrong
4. Switches to a different approach/strategy after rollback
5. Logs every state transition to git for full traceability

Architecture:
    StateSnapshot — Serializable copy of model weights, calibration, metrics
    RegressionDetector — Detects when metrics are degrading
    ApproachManager — Tracks which approaches have been tried, picks next one
    Safeguard — Master controller tying everything together
"""

import copy
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
PROGRESSION_LOG = os.path.join(DATA_DIR, "progression.jsonl")


class StateSnapshot:
    """Immutable snapshot of the system state at a point in time."""

    def __init__(
        self,
        snapshot_id: str,
        cycle: int,
        brier_score: float,
        ece: float,
        discrimination: float,
        roi: Optional[float],
        edge_hit_rate: Optional[float],
        health_score: float,
        model_weights: dict,
        approach: str,
        n_predictions: int,
        timestamp: str,
    ):
        self.snapshot_id = snapshot_id
        self.cycle = cycle
        self.brier_score = brier_score
        self.ece = ece
        self.discrimination = discrimination
        self.roi = roi
        self.edge_hit_rate = edge_hit_rate
        self.health_score = health_score
        self.model_weights = model_weights
        self.approach = approach
        self.n_predictions = n_predictions
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "StateSnapshot":
        return cls(**d)

    @property
    def composite_score(self) -> float:
        """Single score combining all metrics. Higher = better."""
        score = 0.0
        # Brier: lower is better, weight heavily (0-0.5 typical range)
        score += max(0, (0.25 - self.brier_score) / 0.25) * 40
        # Discrimination: higher is better (0.5-1.0 range)
        score += max(0, (self.discrimination - 0.5) / 0.5) * 25
        # ECE: lower is better
        score += max(0, (0.1 - self.ece) / 0.1) * 15
        # ROI: higher is better
        if self.roi is not None:
            score += min(20, max(-10, self.roi * 50))
        return round(score, 2)


class RegressionDetector:
    """Detects performance regression and data corruption.

    Triggers on:
    - Brier score increasing by > threshold over N cycles
    - Health score dropping below minimum
    - Sudden metric jumps (possible corruption)
    - NaN/inf in any metric
    - Edge hit rate collapsing
    """

    def __init__(
        self,
        brier_threshold: float = 0.05,
        health_floor: float = 30.0,
        lookback: int = 5,
        spike_multiplier: float = 10.0,
    ):
        self.brier_threshold = brier_threshold
        self.health_floor = health_floor
        self.lookback = lookback
        self.spike_multiplier = spike_multiplier

    def check(self, history: list[dict], current: dict) -> dict:
        """Check for regression or corruption.

        Returns:
            {
                "status": "OK" | "REGRESSION" | "CORRUPTION" | "DEGRADING",
                "reason": str,
                "severity": 0-10,
                "should_rollback": bool,
            }
        """
        # Check for NaN/inf (corruption)
        for key in ["brier_score", "ece", "discrimination", "health_score"]:
            val = current.get(key)
            if val is not None and (np.isnan(val) or np.isinf(val)):
                return {
                    "status": "CORRUPTION",
                    "reason": f"{key} is NaN/inf",
                    "severity": 10,
                    "should_rollback": True,
                }

        if not history:
            return {"status": "OK", "reason": "No history", "severity": 0, "should_rollback": False}

        # Check health floor
        health = current.get("health_score", 0)
        if health < self.health_floor:
            return {
                "status": "REGRESSION",
                "reason": f"Health {health:.1f} below floor {self.health_floor}",
                "severity": 7,
                "should_rollback": True,
            }

        # Check Brier regression vs recent average
        recent_briers = [h.get("brier_score", 1.0) for h in history[-self.lookback:] if h.get("brier_score") is not None]
        if recent_briers:
            avg_brier = np.mean(recent_briers)
            current_brier = current.get("brier_score", 1.0)
            if current_brier > avg_brier + self.brier_threshold:
                return {
                    "status": "REGRESSION",
                    "reason": f"Brier {current_brier:.4f} > avg {avg_brier:.4f} + {self.brier_threshold}",
                    "severity": 6,
                    "should_rollback": True,
                }

        # Check for sudden spike (possible corruption)
        if len(history) >= 3:
            recent_health = [h.get("health_score", 50) for h in history[-3:]]
            avg_health = np.mean(recent_health)
            health_std = np.std(recent_health) + 1e-6
            if avg_health > 0 and abs(health - avg_health) > self.spike_multiplier * max(health_std, 15):
                return {
                    "status": "CORRUPTION",
                    "reason": f"Health spike: {health:.1f} vs avg {avg_health:.1f} (>{self.spike_multiplier}x std)",
                    "severity": 8,
                    "should_rollback": True,
                }

        # Check gradual degradation (3+ cycles of declining health)
        if len(history) >= 3:
            recent = [h.get("health_score", 50) for h in history[-3:]] + [health]
            if all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                decline = recent[0] - recent[-1]
                if decline > 10:
                    return {
                        "status": "DEGRADING",
                        "reason": f"3+ cycles declining: {recent[0]:.1f} -> {recent[-1]:.1f}",
                        "severity": 4,
                        "should_rollback": decline > 20,
                    }

        return {"status": "OK", "reason": "All checks passed", "severity": 0, "should_rollback": False}


class ApproachManager:
    """Manages different prediction approaches and selects alternatives.

    When the current approach fails, this picks a different one to try.
    Tracks which approaches have been tried and their performance.
    """

    # Available approaches with different configurations
    APPROACHES = {
        "balanced": {
            "description": "Equal weight all models, standard parameters",
            "mc_simulations": 20000,
            "time_series_weight_boost": 1.0,
            "microstructure_weight_boost": 1.0,
            "external_data_weight_boost": 1.0,
            "calibration_enabled": True,
        },
        "time_series_heavy": {
            "description": "Prioritize HMM/GARCH time series signals",
            "mc_simulations": 30000,
            "time_series_weight_boost": 2.5,
            "microstructure_weight_boost": 0.5,
            "external_data_weight_boost": 0.7,
            "calibration_enabled": True,
        },
        "microstructure_focus": {
            "description": "Prioritize bid/ask spread and volume signals",
            "mc_simulations": 15000,
            "time_series_weight_boost": 0.7,
            "microstructure_weight_boost": 2.5,
            "external_data_weight_boost": 0.5,
            "calibration_enabled": True,
        },
        "conservative": {
            "description": "High Monte Carlo, only bet on large edges",
            "mc_simulations": 50000,
            "time_series_weight_boost": 1.0,
            "microstructure_weight_boost": 1.0,
            "external_data_weight_boost": 1.0,
            "calibration_enabled": True,
            "min_edge": 0.08,
        },
        "aggressive_momentum": {
            "description": "Follow recent price momentum heavily",
            "mc_simulations": 20000,
            "time_series_weight_boost": 3.0,
            "microstructure_weight_boost": 1.5,
            "external_data_weight_boost": 0.3,
            "calibration_enabled": False,
        },
        "external_data_driven": {
            "description": "Prioritize crypto/economic/sentiment data",
            "mc_simulations": 20000,
            "time_series_weight_boost": 0.5,
            "microstructure_weight_boost": 0.5,
            "external_data_weight_boost": 3.0,
            "calibration_enabled": True,
        },
    }

    def __init__(self, state_file: str = os.path.join(DATA_DIR, "approach_state.json")):
        self.state_file = state_file
        self.state = self._load()

    @property
    def current_approach(self) -> str:
        return self.state.get("current", "balanced")

    @property
    def current_config(self) -> dict:
        return self.APPROACHES.get(self.current_approach, self.APPROACHES["balanced"])

    def get_next_approach(self, failed_approach: str) -> str:
        """Pick the next approach to try after a failure.

        Strategy: Try each approach in order of how well it's done historically.
        If none tried yet, use a round-robin of the list.
        """
        # Record failure
        failures = self.state.setdefault("failures", {})
        failures[failed_approach] = failures.get(failed_approach, 0) + 1

        # Sort approaches by: least failures first, then by best historical score
        scores = self.state.get("best_scores", {})
        candidates = [
            a for a in self.APPROACHES
            if a != failed_approach and failures.get(a, 0) < 3  # max 3 tries per approach
        ]

        if not candidates:
            # Reset failure counts and try again
            self.state["failures"] = {}
            candidates = [a for a in self.APPROACHES if a != failed_approach]

        # Sort by best score (descending), then least failures
        candidates.sort(key=lambda a: (-scores.get(a, 0), failures.get(a, 0)))

        next_approach = candidates[0] if candidates else "balanced"
        self.state["current"] = next_approach
        self.state.setdefault("history", []).append({
            "from": failed_approach,
            "to": next_approach,
            "reason": "regression_detected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return next_approach

    def record_score(self, approach: str, score: float):
        """Record the composite score for an approach."""
        best = self.state.setdefault("best_scores", {})
        if score > best.get(approach, 0):
            best[approach] = score
        self.state["current"] = approach
        self._save()

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {"current": "balanced", "failures": {}, "best_scores": {}, "history": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)


class Safeguard:
    """Master safeguard controller.

    Wraps around the training loop to provide:
    - Automatic state snapshots after each cycle
    - Regression/corruption detection
    - Rollback to best-known state
    - Approach switching
    - Full git logging of every transition
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.detector = RegressionDetector()
        self.approach_mgr = ApproachManager()
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    def take_snapshot(self, cycle: int, audit: dict, weights: dict, approach: str) -> StateSnapshot:
        """Take a snapshot of current state."""
        snap = StateSnapshot(
            snapshot_id=f"snap_{cycle}_{int(time.time())}",
            cycle=cycle,
            brier_score=audit.get("brier_score", 1.0),
            ece=audit.get("expected_calibration_error", 1.0),
            discrimination=audit.get("discrimination", 0.5),
            roi=audit.get("roi"),
            edge_hit_rate=audit.get("edge_hit_rate"),
            health_score=audit.get("health_score", 0) if "health_score" in audit else self._compute_health(audit),
            model_weights=weights,
            approach=approach,
            n_predictions=audit.get("n_predictions", 0),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Save snapshot
        snap_path = os.path.join(SNAPSHOTS_DIR, f"{snap.snapshot_id}.json")
        with open(snap_path, "w") as f:
            json.dump(snap.to_dict(), f, indent=2)

        # Log to progression
        self._log_progression(snap)

        return snap

    def check_and_protect(self, cycle: int, audit: dict) -> dict:
        """Check for regression and take protective action if needed.

        Returns:
            {
                "action": "CONTINUE" | "ROLLBACK" | "SWITCH_APPROACH",
                "details": str,
                "new_approach": str | None,
            }
        """
        history = self._load_progression_history()
        current_metrics = {
            "brier_score": audit.get("brier_score"),
            "ece": audit.get("expected_calibration_error"),
            "discrimination": audit.get("discrimination"),
            "health_score": audit.get("health_score", self._compute_health(audit)),
            "roi": audit.get("roi"),
        }

        check = self.detector.check(history, current_metrics)

        if check["should_rollback"]:
            # Find best snapshot
            best = self._find_best_snapshot()
            if best:
                # Rollback
                self._restore_snapshot(best)

                # Switch approach
                old_approach = self.approach_mgr.current_approach
                new_approach = self.approach_mgr.get_next_approach(old_approach)

                # Git log the rollback
                self._git_commit(
                    f"ROLLBACK: cycle {cycle} | {check['status']}: {check['reason']} | "
                    f"rolling back to {best.snapshot_id} | switching {old_approach} -> {new_approach}"
                )

                return {
                    "action": "ROLLBACK",
                    "details": f"{check['status']}: {check['reason']}. "
                               f"Rolled back to snapshot {best.snapshot_id} (cycle {best.cycle}). "
                               f"Switched approach: {old_approach} -> {new_approach}",
                    "new_approach": new_approach,
                    "rolled_back_to": best.snapshot_id,
                }

        if check["status"] == "DEGRADING":
            return {
                "action": "SWITCH_APPROACH",
                "details": f"Performance degrading: {check['reason']}",
                "new_approach": None,  # let caller decide
            }

        return {
            "action": "CONTINUE",
            "details": check["reason"],
            "new_approach": None,
        }

    def get_best_snapshot(self) -> Optional[StateSnapshot]:
        return self._find_best_snapshot()

    def get_progression(self) -> list[dict]:
        """Get full progression history for display."""
        return self._load_progression_history()

    def _find_best_snapshot(self) -> Optional[StateSnapshot]:
        """Find the snapshot with the best composite score."""
        if not os.path.exists(SNAPSHOTS_DIR):
            return None
        best = None
        best_score = -float("inf")
        for fname in os.listdir(SNAPSHOTS_DIR):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(SNAPSHOTS_DIR, fname)) as f:
                data = json.load(f)
            snap = StateSnapshot.from_dict(data)
            if snap.composite_score > best_score:
                best_score = snap.composite_score
                best = snap
        return best

    def _restore_snapshot(self, snap: StateSnapshot):
        """Restore model weights from a snapshot."""
        weights_path = os.path.join(self.data_dir, "model_weights.json")
        with open(weights_path, "w") as f:
            json.dump(snap.model_weights, f, indent=2)

    def _log_progression(self, snap: StateSnapshot):
        """Append to progression log."""
        entry = {
            "snapshot_id": snap.snapshot_id,
            "cycle": snap.cycle,
            "approach": snap.approach,
            "brier": snap.brier_score,
            "ece": snap.ece,
            "discrimination": snap.discrimination,
            "roi": snap.roi,
            "edge_hit_rate": snap.edge_hit_rate,
            "health": snap.health_score,
            "composite": snap.composite_score,
            "n_predictions": snap.n_predictions,
            "timestamp": snap.timestamp,
        }
        with open(PROGRESSION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_progression_history(self) -> list[dict]:
        if not os.path.exists(PROGRESSION_LOG):
            return []
        entries = []
        with open(PROGRESSION_LOG) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def _compute_health(audit: dict) -> float:
        score = 50.0
        brier = audit.get("brier_score", 0.5)
        score += max(0, 30 * (1 - brier / 0.25))
        disc = audit.get("discrimination", 0.5)
        score += max(0, 20 * (disc - 0.5) / 0.5)
        ece = audit.get("expected_calibration_error", 0.5)
        score += max(0, 20 * (1 - ece / 0.1))
        roi = audit.get("roi")
        if roi is not None:
            score += min(30, max(-20, roi * 100))
        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _git_commit(message: str):
        try:
            subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(PROJECT_ROOT), capture_output=True, timeout=30,
            )
        except Exception:
            pass
