"""Adaptive ensemble — blends inverse-variance weights with learned per-model
performance weights derived from resolved outcomes.

Live cycle uses the static `EnsemblePredictor` in src/statistics.py. This v2
combiner reads `data/model_weights.json` (refreshed by self_improver), and
combines the two weight sources via a blend factor so unproven weights don't
swing the prediction wildly.

Failure mode: if model_weights.json is missing/malformed/empty, behavior
falls back to pure inverse-variance — i.e. exactly the v1 ensemble.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np


DATA_DIR = "data"
WEIGHTS_FILE = os.path.join(DATA_DIR, "model_weights.json")


def load_learned_weights(path: str = WEIGHTS_FILE) -> dict[str, float]:
    """Load per-model weights. Returns empty dict on any failure."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        # Coerce to floats, drop non-numeric/NaN
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                fv = float(v)
                if math.isfinite(fv) and fv >= 0:
                    out[k] = fv
            except (TypeError, ValueError):
                continue
        return out
    except (json.JSONDecodeError, OSError):
        return {}


class AdaptiveEnsemble:
    """Inverse-variance ensemble with optional learned-weight blending.

    blend = 0.0 → pure inverse-variance (matches v1 EnsemblePredictor)
    blend = 1.0 → pure learned weights from model_weights.json
    blend = 0.5 → equal mix; default leans inverse-variance until n is large
    """

    def __init__(
        self,
        weights_path: str = WEIGHTS_FILE,
        blend: float = 0.5,
        min_trades_for_weight: int = 30,
    ):
        self.weights_path = weights_path
        self.blend = max(0.0, min(1.0, blend))
        self.min_trades_for_weight = min_trades_for_weight
        self._cached_weights: Optional[dict] = None

    def _learned_weights(self, model_names: list[str]) -> Optional[np.ndarray]:
        """Return learned-weight vector aligned to model_names, or None if unusable."""
        raw = load_learned_weights(self.weights_path)
        if not raw:
            return None

        vec = np.array([raw.get(name, 0.0) for name in model_names], dtype=np.float64)
        total = vec.sum()
        if total <= 0:
            return None
        return vec / total

    def combine(
        self,
        estimates: list[float],
        variances: list[float],
        model_names: Optional[list[str]] = None,
    ) -> dict:
        """Combine estimates. Same return shape as EnsemblePredictor.combine."""
        estimates_arr = np.asarray(estimates, dtype=np.float64)
        variances_arr = np.clip(np.asarray(variances, dtype=np.float64), 1e-10, None)

        # Inverse-variance weights (the v1 baseline)
        iv_weights = 1.0 / variances_arr
        iv_weights = iv_weights / iv_weights.sum()

        # Learned weights, if available and applicable
        learned = self._learned_weights(model_names) if model_names else None

        if learned is None or self.blend <= 0.0:
            final_weights = iv_weights
            blend_used = 0.0
        else:
            final_weights = (1 - self.blend) * iv_weights + self.blend * learned
            final_weights = final_weights / final_weights.sum()
            blend_used = self.blend

        combined_mean = float(np.dot(final_weights, estimates_arr))
        # Variance under arbitrary weights w: sum(w_i^2 * var_i)
        combined_var = float(np.sum(final_weights ** 2 * variances_arr))

        return {
            "probability": float(np.clip(combined_mean, 0.0, 1.0)),
            "variance": combined_var,
            "std": math.sqrt(combined_var),
            "ci_95": (
                max(0.0, combined_mean - 1.96 * math.sqrt(combined_var)),
                min(1.0, combined_mean + 1.96 * math.sqrt(combined_var)),
            ),
            "weights": final_weights.tolist(),
            "iv_weights": iv_weights.tolist(),
            "learned_weights": learned.tolist() if learned is not None else None,
            "blend_used": blend_used,
            "n_models": len(estimates_arr),
        }
