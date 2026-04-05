"""Strategy Adapter — Proven calendar spread concepts adapted for Polymarket.

Adapts the ATM_ITM_MIXED_CALENDAR_V1 strategy's proven risk management,
capital allocation, and signal hierarchy patterns for prediction markets.

Key concepts ported from the options strategy (2,390% backtest return):
    1. Fractional Kelly (1/3 Kelly) — conservative sizing that still compounds
    2. Regime-based capital allocation — VIX splits -> Fear&Greed/volatility proxy
    3. OI reversal -> orderbook imbalance flip as regime change signal
    4. Priority-based signal hierarchy — not all signals weighted equally
    5. Max deployment cap (60%) — never overexpose
    6. Entry filter gates — ALL must pass before any position
    7. Trend decay / stall detection — exit when edge evaporates
    8. Win-rate x payoff ratio allocation — more capital to proven edges
    9. No stop-loss philosophy — binary markets already cap max loss

The calendar spread strategy proved that:
    - 1/3 Kelly >> full Kelly for real-world compounding
    - Regime-aware sizing dramatically improves drawdown-adjusted returns
    - Priority-based exits outperform fixed thresholds
    - Multi-filter gates reduce false positives significantly
"""

import math
import time
from typing import Optional
from dataclasses import dataclass, field

import numpy as np


# ===========================================================================
# Capital Allocation Engine (from VIX-based CE/PE splits)
# ===========================================================================

@dataclass
class MarketRegime:
    """Current market regime classification (analog of VIX regime)."""
    fear_greed_value: float = 50.0       # 0=extreme fear, 100=extreme greed
    volatility_percentile: float = 0.5   # 0-1, where market vol sits vs history
    regime_label: str = "neutral"        # fear, neutral, greed, extreme_fear, extreme_greed
    timestamp: float = 0.0

    @classmethod
    def from_fear_greed(cls, fg_value: float) -> "MarketRegime":
        """Classify regime from Fear & Greed Index (VIX analog)."""
        if fg_value < 15:
            label = "extreme_fear"
        elif fg_value < 30:
            label = "fear"
        elif fg_value < 70:
            label = "neutral"
        elif fg_value < 85:
            label = "greed"
        else:
            label = "extreme_greed"

        # Map F&G to volatility percentile (inverse — high fear = high vol)
        vol_pct = 1.0 - fg_value / 100.0

        return cls(
            fear_greed_value=fg_value,
            volatility_percentile=vol_pct,
            regime_label=label,
            timestamp=time.time(),
        )


# Regime-based allocation splits (adapted from VIX CE/PE splits)
# In options: VIX < 14 -> 80% CE / 20% PE
# In Polymarket: low fear -> aggressive on high-confidence, high fear -> conservative
REGIME_ALLOCATION = {
    "extreme_fear": {
        "high_confidence_budget": 0.30,  # 30% on high-confidence trades
        "speculative_budget": 0.10,      # 10% speculative (was 5% — too tight)
        "contrarian_bonus": 0.10,        # Bonus for contrarian bets (fear = opp)
        "max_per_market": 0.08,          # Max 8% per single market
    },
    "fear": {
        "high_confidence_budget": 0.35,
        "speculative_budget": 0.12,
        "contrarian_bonus": 0.05,
        "max_per_market": 0.10,
    },
    "neutral": {
        "high_confidence_budget": 0.45,
        "speculative_budget": 0.10,
        "contrarian_bonus": 0.0,
        "max_per_market": 0.12,
    },
    "greed": {
        "high_confidence_budget": 0.50,
        "speculative_budget": 0.12,
        "contrarian_bonus": 0.0,
        "max_per_market": 0.12,
    },
    "extreme_greed": {
        "high_confidence_budget": 0.40,  # Pull back — greed often precedes drops
        "speculative_budget": 0.05,
        "contrarian_bonus": 0.08,        # Contrarian against extreme greed
        "max_per_market": 0.10,
    },
}


class CapitalAllocator:
    """Regime-aware capital allocation engine.

    Adapted from the calendar spread strategy's VIX-based budget splits.
    The key insight: allocate MORE to high-confidence trades in calm regimes,
    and reserve capital for contrarian opportunities in extreme regimes.
    """

    MAX_TOTAL_DEPLOYMENT = 0.60  # NEVER deploy more than 60% of bankroll
    KELLY_FRACTION = 1 / 3       # 1/3 Kelly — proven optimal in backtests

    def __init__(self, total_equity: float = 10000.0):
        self.total_equity = total_equity
        self.deployed = 0.0
        self.positions: dict[str, float] = {}  # market_id -> deployed amount
        self.regime = MarketRegime()

    def update_regime(self, fear_greed_value: float):
        """Update market regime from Fear & Greed Index."""
        self.regime = MarketRegime.from_fear_greed(fear_greed_value)

    def available_capital(self) -> float:
        """Capital available for new positions."""
        max_deploy = self.total_equity * self.MAX_TOTAL_DEPLOYMENT
        return max(0, max_deploy - self.deployed)

    def compute_position_size(
        self,
        kelly_fraction: float,
        edge: float,
        confidence: str,
        is_contrarian: bool = False,
    ) -> dict:
        """Compute position size using 1/3 Kelly with regime adjustment.

        This is the core insight from the calendar strategy:
        - Use 1/3 Kelly (not full Kelly) for real-world compounding
        - Cap per-market allocation based on regime
        - Apply deployment cap (60% total)
        - Bonus allocation for contrarian trades in extreme regimes

        Args:
            kelly_fraction: Raw Kelly fraction from model
            edge: Estimated edge (probability - market price)
            confidence: "HIGH", "MEDIUM", or "LOW"
            is_contrarian: Whether this bet goes against market consensus

        Returns:
            Position sizing result
        """
        regime_alloc = REGIME_ALLOCATION.get(
            self.regime.regime_label,
            REGIME_ALLOCATION["neutral"],
        )

        # Step 1: Apply 1/3 Kelly (proven in backtest to outperform full Kelly)
        kelly_adjusted = kelly_fraction * self.KELLY_FRACTION

        # Step 2: Determine budget category
        if confidence in ("HIGH", "MEDIUM") and abs(edge) > 0.02:
            budget_pct = regime_alloc["high_confidence_budget"]
        else:
            budget_pct = regime_alloc["speculative_budget"]

        # Step 3: Contrarian bonus in extreme regimes
        if is_contrarian and regime_alloc["contrarian_bonus"] > 0:
            budget_pct += regime_alloc["contrarian_bonus"]

        # Step 4: Cap per-market size
        max_per_market = regime_alloc["max_per_market"]
        position_pct = min(abs(kelly_adjusted), budget_pct, max_per_market)

        # Step 5: Cap by available capital (deployment limit)
        available = self.available_capital()
        position_dollars = min(position_pct * self.total_equity, available)

        # Step 6: Minimum viable position
        if position_dollars < self.total_equity * 0.002:  # Less than 0.2% ($20 on $10k)
            return {
                "size_dollars": 0,
                "size_pct": 0,
                "action": "NO_BET",
                "reason": "Below minimum position size",
                "kelly_raw": kelly_fraction,
                "kelly_adjusted": kelly_adjusted,
                "regime": self.regime.regime_label,
                "budget_category": confidence,
            }

        return {
            "size_dollars": round(position_dollars, 2),
            "size_pct": round(position_dollars / self.total_equity, 4),
            "action": "BUY_YES" if edge > 0 else "BUY_NO",
            "kelly_raw": round(kelly_fraction, 4),
            "kelly_adjusted": round(kelly_adjusted, 4),
            "regime": self.regime.regime_label,
            "budget_category": confidence,
            "max_per_market": max_per_market,
            "deployment_used": round(self.deployed / self.total_equity, 4),
            "deployment_remaining": round(available / self.total_equity, 4),
        }

    def record_position(self, market_id: str, amount: float):
        """Record a new position deployment."""
        self.positions[market_id] = amount
        self.deployed = sum(self.positions.values())

    def close_position(self, market_id: str):
        """Remove a closed position."""
        if market_id in self.positions:
            del self.positions[market_id]
            self.deployed = sum(self.positions.values())


# ===========================================================================
# Entry Filter Gates (from options strategy Section 4)
# ===========================================================================

class EntryFilterGate:
    """All-must-pass entry filter system.

    Adapted from the calendar spread strategy where:
    - Ratio filter: near/far premium ratio must be 0.65-1.10
    - OI sanity check: no extreme imbalance
    - VIX gate: regime-appropriate

    For Polymarket:
    - Spread sanity: bid-ask spread must be reasonable
    - Liquidity gate: minimum depth to avoid slippage
    - Volume gate: market must have real activity
    - Edge confidence gate: model agreement must be sufficient
    - Imbalance gate: orderbook not one-sided beyond threshold
    """

    # Thresholds (calibrated for prediction markets)
    MAX_SPREAD_PCT = 0.10         # Max 10% bid-ask spread
    MIN_LIQUIDITY = 500           # Minimum $500 liquidity
    MIN_VOLUME_24H = 100          # Minimum $100 in 24h volume
    MIN_MODEL_AGREEMENT = 0.40    # At least 40% model agreement
    MAX_IMBALANCE = 0.85          # Orderbook imbalance cap (from OI_DOMINANCE_X)
    MIN_EDGE = 0.02               # Minimum 2% edge to enter
    PRICE_SANITY_RANGE = (0.05, 0.95)  # Don't bet on near-certain outcomes

    def check_all(self, market_data: dict, prediction: dict) -> dict:
        """Run all entry filters. ALL must pass for a valid entry.

        Returns:
            {
                "pass": bool,
                "filters": {name: {"pass": bool, "value": ..., "threshold": ...}},
                "failed_filters": [names of failed filters],
            }
        """
        filters = {}

        # 1. Spread sanity (analog of ratio filter 0.65-1.10)
        spread = market_data.get("spread")
        ask = market_data.get("best_ask", 1)
        if spread is not None and ask and ask > 0:
            spread_pct = spread / ask
            filters["spread_sanity"] = {
                "pass": spread_pct <= self.MAX_SPREAD_PCT,
                "value": round(spread_pct, 4),
                "threshold": self.MAX_SPREAD_PCT,
            }
        else:
            # No spread data — soft pass (Gamma API doesn't always have it)
            filters["spread_sanity"] = {"pass": True, "value": None, "threshold": self.MAX_SPREAD_PCT}

        # 2. Liquidity gate
        liquidity = market_data.get("liquidity", 0) or 0
        filters["liquidity"] = {
            "pass": liquidity >= self.MIN_LIQUIDITY,
            "value": liquidity,
            "threshold": self.MIN_LIQUIDITY,
        }

        # 3. Volume gate
        volume = market_data.get("volume_24h", 0) or 0
        filters["volume"] = {
            "pass": volume >= self.MIN_VOLUME_24H,
            "value": volume,
            "threshold": self.MIN_VOLUME_24H,
        }

        # 4. Model agreement gate
        agreement = prediction.get("ensemble", {}).get("model_agreement", 0)
        filters["model_agreement"] = {
            "pass": agreement >= self.MIN_MODEL_AGREEMENT,
            "value": round(agreement, 4),
            "threshold": self.MIN_MODEL_AGREEMENT,
        }

        # 5. Edge significance gate
        edge = abs(prediction.get("edge", {}).get("edge", 0))
        filters["edge_significance"] = {
            "pass": edge >= self.MIN_EDGE,
            "value": round(edge, 4),
            "threshold": self.MIN_EDGE,
        }

        # 6. Price sanity (don't bet on extremes — like ratio filter)
        price = prediction.get("market", {}).get("current_price", 0.5)
        in_range = self.PRICE_SANITY_RANGE[0] <= price <= self.PRICE_SANITY_RANGE[1]
        filters["price_sanity"] = {
            "pass": in_range,
            "value": round(price, 4),
            "threshold": self.PRICE_SANITY_RANGE,
        }

        # 7. Orderbook imbalance gate (analog of OI_DOMINANCE_X)
        ob_signals = prediction.get("sub_models", {}).get("orderbook", {}).get("signals", {})
        imbalance = abs(ob_signals.get("imbalance", 0) or 0)
        filters["imbalance"] = {
            "pass": imbalance <= self.MAX_IMBALANCE,
            "value": round(imbalance, 4),
            "threshold": self.MAX_IMBALANCE,
        }

        failed = [name for name, f in filters.items() if not f["pass"]]
        return {
            "pass": len(failed) == 0,
            "filters": filters,
            "failed_filters": failed,
            "n_passed": sum(1 for f in filters.values() if f["pass"]),
            "n_total": len(filters),
        }


# ===========================================================================
# Orderbook Imbalance Flip Detector (from OI Reversal exit)
# ===========================================================================

class ImbalanceFlipDetector:
    """Detect orderbook imbalance reversals (analog of OI reversal exit).

    In the calendar spread strategy, if the dominant OI side flips on
    2 consecutive days, that's an exit signal. Here, we track orderbook
    imbalance over time and detect flips.

    An imbalance flip means the market's directional pressure has changed —
    your edge may have evaporated or reversed.
    """

    def __init__(self, confirmation_periods: int = 2):
        """
        Args:
            confirmation_periods: Number of consecutive opposite readings
                                 required to confirm a flip (default: 2, like OI reversal)
        """
        self.confirmation_periods = confirmation_periods
        self.history: dict[str, list[dict]] = {}  # market_id -> imbalance readings

    def record(self, market_id: str, imbalance: float, timestamp: float = 0):
        """Record an imbalance reading for a market."""
        if market_id not in self.history:
            self.history[market_id] = []
        self.history[market_id].append({
            "imbalance": imbalance,
            "side": "BUY" if imbalance > 0 else "SELL" if imbalance < 0 else "NEUTRAL",
            "timestamp": timestamp or time.time(),
        })
        # Keep last 20 readings
        self.history[market_id] = self.history[market_id][-20:]

    def check_flip(self, market_id: str) -> dict:
        """Check if imbalance has flipped sides with confirmation.

        Returns:
            {
                "flip_detected": bool,
                "from_side": str,
                "to_side": str,
                "consecutive_opposite": int,
                "signal": "EXIT" | "HOLD" | "INSUFFICIENT_DATA",
            }
        """
        readings = self.history.get(market_id, [])
        if len(readings) < self.confirmation_periods + 1:
            return {
                "flip_detected": False,
                "signal": "INSUFFICIENT_DATA",
                "n_readings": len(readings),
            }

        # Determine entry-side (side at oldest reading)
        entry_side = readings[0]["side"]

        # Check recent readings for consecutive opposite
        recent = readings[-(self.confirmation_periods):]
        consecutive_opposite = 0
        for r in recent:
            if r["side"] != entry_side and r["side"] != "NEUTRAL":
                consecutive_opposite += 1
            else:
                consecutive_opposite = 0

        flip_detected = consecutive_opposite >= self.confirmation_periods

        return {
            "flip_detected": flip_detected,
            "from_side": entry_side,
            "to_side": recent[-1]["side"] if recent else "UNKNOWN",
            "consecutive_opposite": consecutive_opposite,
            "required": self.confirmation_periods,
            "signal": "EXIT" if flip_detected else "HOLD",
        }

    def clear(self, market_id: str):
        """Clear history for a market (after position closed)."""
        self.history.pop(market_id, None)


# ===========================================================================
# Trend Decay / Stall Detector (from TREND_DECAY_EXIT)
# ===========================================================================

class StallDetector:
    """Detect stalled markets where edge is evaporating.

    Adapted from TREND_DECAY_EXIT:
        "premium_remaining > 62% of entry AND price_move <= 0.4%"
        Both must hold for 2 consecutive days.

    For Polymarket: if the price hasn't moved significantly and our
    predicted edge hasn't materialized, the market is stalled.
    """

    PRICE_MOVE_THRESHOLD = 0.005  # 0.5% (analog of 0.4% from strike)
    EDGE_DECAY_THRESHOLD = 0.50   # Edge fallen to 50% of entry edge
    CONFIRMATION_PERIODS = 2       # Must hold for 2 periods (like 2 consecutive days)

    def __init__(self):
        self.entry_state: dict[str, dict] = {}  # market_id -> entry snapshot
        self.stall_counts: dict[str, int] = {}   # consecutive stall periods

    def record_entry(self, market_id: str, entry_price: float, entry_edge: float):
        """Record state at position entry."""
        self.entry_state[market_id] = {
            "entry_price": entry_price,
            "entry_edge": entry_edge,
            "timestamp": time.time(),
        }
        self.stall_counts[market_id] = 0

    def check_stall(self, market_id: str, current_price: float, current_edge: float) -> dict:
        """Check if market has stalled (analog of TREND_DECAY_EXIT).

        Both conditions must hold for CONFIRMATION_PERIODS consecutive checks:
        1. Price hasn't moved beyond threshold from entry
        2. Edge has decayed significantly from entry

        Returns:
            {
                "stalled": bool,
                "signal": "EXIT_STALL" | "HOLD",
                "price_move": float,
                "edge_remaining_pct": float,
                "consecutive_stalls": int,
            }
        """
        state = self.entry_state.get(market_id)
        if not state:
            return {"stalled": False, "signal": "NO_ENTRY_RECORDED"}

        entry_price = state["entry_price"]
        entry_edge = state["entry_edge"]

        # Condition 1: Price hasn't moved (analog of price_move <= 0.4%)
        price_move = abs(current_price - entry_price)
        price_stalled = price_move <= self.PRICE_MOVE_THRESHOLD

        # Condition 2: Edge hasn't materialized (analog of premium_remaining > 62%)
        if abs(entry_edge) > 1e-6:
            edge_remaining_pct = current_edge / entry_edge
        else:
            edge_remaining_pct = 1.0  # No edge to decay

        edge_decayed = edge_remaining_pct < self.EDGE_DECAY_THRESHOLD

        # Both must hold
        is_stalling = price_stalled and edge_decayed

        if is_stalling:
            self.stall_counts[market_id] = self.stall_counts.get(market_id, 0) + 1
        else:
            self.stall_counts[market_id] = 0

        consecutive = self.stall_counts.get(market_id, 0)
        confirmed_stall = consecutive >= self.CONFIRMATION_PERIODS

        return {
            "stalled": confirmed_stall,
            "signal": "EXIT_STALL" if confirmed_stall else "HOLD",
            "price_move": round(price_move, 4),
            "price_stalled": price_stalled,
            "edge_remaining_pct": round(edge_remaining_pct, 4),
            "edge_decayed": edge_decayed,
            "consecutive_stalls": consecutive,
            "required": self.CONFIRMATION_PERIODS,
        }

    def clear(self, market_id: str):
        """Clear state for a market."""
        self.entry_state.pop(market_id, None)
        self.stall_counts.pop(market_id, None)


# ===========================================================================
# Priority-Based Signal Hierarchy (from exit priority system)
# ===========================================================================

class SignalHierarchy:
    """Priority-based signal system adapted from the calendar spread exits.

    Original priority order:
        OI_REVERSAL > TREND_DECAY > TIME_EXIT > EXPIRY_FORCE_CLOSE > YEAR_END

    Polymarket adaptation:
        IMBALANCE_FLIP > STALL_DECAY > TIME_EXIT > EXPIRY_CLOSE > EDGE_EVAPORATED

    Higher priority signals override lower ones.
    """

    PRIORITY_ORDER = [
        "IMBALANCE_FLIP",    # Priority 1: Orderbook pressure reversed
        "STALL_DECAY",       # Priority 2: Market stalled, edge not materializing
        "TIME_EXIT",         # Priority 3: Max hold time reached
        "EXPIRY_CLOSE",      # Priority 4: Market about to resolve
        "EDGE_EVAPORATED",   # Priority 5: Edge shrunk below threshold
    ]

    # Max hold periods per confidence level (analog of MAX_HOLD_DAYS per index)
    MAX_HOLD_PERIODS = {
        "HIGH": 10,      # 10 cycles for high confidence
        "MEDIUM": 6,     # 6 cycles for medium confidence
        "LOW": 3,        # 3 cycles for low confidence
    }

    # Edge evaporation threshold
    MIN_REMAINING_EDGE = 0.01  # 1% minimum edge to hold

    def evaluate(
        self,
        imbalance_flip: dict,
        stall_check: dict,
        hold_periods: int,
        confidence: str,
        current_edge: float,
        time_to_expiry_frac: float,
    ) -> dict:
        """Evaluate all exit signals in priority order.

        Returns the HIGHEST PRIORITY signal that fires.
        """
        signals = []

        # Priority 1: Imbalance flip (OI reversal analog)
        if imbalance_flip.get("flip_detected"):
            signals.append({
                "signal": "IMBALANCE_FLIP",
                "priority": 1,
                "action": "EXIT",
                "reason": f"Orderbook pressure flipped from {imbalance_flip.get('from_side')} to {imbalance_flip.get('to_side')}",
            })

        # Priority 2: Stall decay (trend decay analog)
        if stall_check.get("stalled"):
            signals.append({
                "signal": "STALL_DECAY",
                "priority": 2,
                "action": "EXIT",
                "reason": f"Market stalled for {stall_check.get('consecutive_stalls')} periods, edge not materializing",
            })

        # Priority 3: Time exit (max hold days analog)
        max_hold = self.MAX_HOLD_PERIODS.get(confidence, 6)
        if hold_periods >= max_hold:
            signals.append({
                "signal": "TIME_EXIT",
                "priority": 3,
                "action": "EXIT",
                "reason": f"Max hold periods reached ({hold_periods}/{max_hold})",
            })

        # Priority 4: Expiry close (force close near expiry)
        if time_to_expiry_frac < 0.05:  # Less than 5% time remaining
            signals.append({
                "signal": "EXPIRY_CLOSE",
                "priority": 4,
                "action": "EXIT",
                "reason": f"Market near expiry ({time_to_expiry_frac:.1%} remaining)",
            })

        # Priority 5: Edge evaporated
        if abs(current_edge) < self.MIN_REMAINING_EDGE:
            signals.append({
                "signal": "EDGE_EVAPORATED",
                "priority": 5,
                "action": "EXIT",
                "reason": f"Edge shrunk to {current_edge:.4f}, below minimum {self.MIN_REMAINING_EDGE}",
            })

        if not signals:
            return {
                "signal": "HOLD",
                "priority": 0,
                "action": "HOLD",
                "reason": "No exit signals triggered",
                "all_clear": True,
            }

        # Return highest priority (lowest number) signal
        signals.sort(key=lambda s: s["priority"])
        top = signals[0]
        top["all_signals"] = signals
        top["n_exit_signals"] = len(signals)
        return top


# ===========================================================================
# Win-Rate x Payoff Model Weight Allocator
# ===========================================================================

class WinRatePayoffAllocator:
    """Allocate model weights based on win-rate x payoff ratio.

    Adapted from the calendar strategy's allocation rationale where:
        FINNIFTY PE: WR=82.8%, Payoff=8.26x -> Kelly=81% -> Alloc=25%
        FINNIFTY CE: WR=80.0%, Payoff=1.38x -> Kelly=65% -> Alloc=22%
        ...

    For Polymarket sub-models: track each model's win rate and average
    payoff, then allocate ensemble weight proportionally.
    """

    def __init__(self):
        self.model_stats: dict[str, dict] = {}

    def record_outcome(self, model_name: str, predicted_prob: float,
                       market_price: float, actual_outcome: int):
        """Record a prediction outcome for a model.

        Args:
            model_name: Name of the sub-model
            predicted_prob: Model's probability estimate
            market_price: Market price at time of prediction
            actual_outcome: 1 if resolved Yes, 0 if resolved No
        """
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "wins": 0, "losses": 0,
                "total_payoff": 0.0, "total_risk": 0.0,
                "predictions": 0,
            }

        stats = self.model_stats[model_name]
        stats["predictions"] += 1

        # Did model's edge call win?
        edge = predicted_prob - market_price
        if edge > 0:
            # Model said BUY_YES
            if actual_outcome == 1:
                stats["wins"] += 1
                stats["total_payoff"] += (1 - market_price)  # profit per dollar
            else:
                stats["losses"] += 1
                stats["total_risk"] += market_price  # loss per dollar
        else:
            # Model said BUY_NO
            if actual_outcome == 0:
                stats["wins"] += 1
                stats["total_payoff"] += market_price
            else:
                stats["losses"] += 1
                stats["total_risk"] += (1 - market_price)

    def get_allocation_weights(self, model_names: list[str]) -> dict[str, float]:
        """Compute allocation weights based on win-rate x payoff.

        Uses the same logic as the calendar spread's Kelly-derived allocations:
        higher WR + higher payoff = more weight.
        """
        raw_scores = {}

        for name in model_names:
            stats = self.model_stats.get(name)
            if not stats or stats["predictions"] < 10:
                # Not enough data — equal weight
                raw_scores[name] = 1.0
                continue

            total = stats["wins"] + stats["losses"]
            if total == 0:
                raw_scores[name] = 1.0
                continue

            win_rate = stats["wins"] / total
            avg_payoff = stats["total_payoff"] / max(stats["wins"], 1)
            avg_risk = stats["total_risk"] / max(stats["losses"], 1)
            payoff_ratio = avg_payoff / avg_risk if avg_risk > 0 else 2.0

            # Kelly-like score: WR * payoff_ratio - (1 - WR)
            kelly_score = win_rate * payoff_ratio - (1 - win_rate)
            # Clamp and shift to positive
            raw_scores[name] = max(kelly_score + 0.5, 0.1)

        # Normalize to sum to 1
        total_score = sum(raw_scores.values())
        if total_score <= 0:
            # Equal weight fallback
            n = len(model_names)
            return {name: 1.0 / n for name in model_names}

        return {name: score / total_score for name, score in raw_scores.items()}

    def get_model_report(self) -> list[dict]:
        """Generate a report of all model performance (like the per-index table)."""
        report = []
        for name, stats in self.model_stats.items():
            total = stats["wins"] + stats["losses"]
            if total == 0:
                continue

            win_rate = stats["wins"] / total
            avg_payoff = stats["total_payoff"] / max(stats["wins"], 1)
            avg_risk = stats["total_risk"] / max(stats["losses"], 1)
            payoff_ratio = avg_payoff / avg_risk if avg_risk > 0 else 0

            report.append({
                "model": name,
                "predictions": stats["predictions"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 4),
                "avg_payoff": round(avg_payoff, 4),
                "avg_risk": round(avg_risk, 4),
                "payoff_ratio": round(payoff_ratio, 2),
                "kelly_pct": round(max(0, win_rate * payoff_ratio - (1 - win_rate)) * 100, 1),
            })

        report.sort(key=lambda x: x["kelly_pct"], reverse=True)
        return report


# ===========================================================================
# Master Strategy Adapter (combines all components)
# ===========================================================================

class StrategyAdapter:
    """Master adapter integrating all proven calendar spread concepts.

    Provides a single interface for the PredictionEngine to:
    1. Check entry filters
    2. Get regime-aware position sizing
    3. Evaluate exit signals
    4. Track model performance for weight optimization
    """

    def __init__(self, total_equity: float = 10000.0):
        self.allocator = CapitalAllocator(total_equity=total_equity)
        self.entry_gate = EntryFilterGate()
        self.imbalance_detector = ImbalanceFlipDetector(confirmation_periods=2)
        self.stall_detector = StallDetector()
        self.signal_hierarchy = SignalHierarchy()
        self.wr_allocator = WinRatePayoffAllocator()
        self.hold_periods: dict[str, int] = {}

    def evaluate_entry(self, market_data: dict, prediction: dict) -> dict:
        """Full entry evaluation: filters + sizing.

        Returns:
            {
                "should_enter": bool,
                "filters": {...},
                "sizing": {...},
                "regime": str,
            }
        """
        # Run all entry filters
        filter_result = self.entry_gate.check_all(market_data, prediction)

        if not filter_result["pass"]:
            return {
                "should_enter": False,
                "filters": filter_result,
                "sizing": {"action": "NO_BET", "reason": f"Failed filters: {filter_result['failed_filters']}"},
                "regime": self.allocator.regime.regime_label,
            }

        # Compute position size
        kelly = prediction.get("sizing", {}).get("kelly_fraction", 0)
        edge = prediction.get("edge", {}).get("edge", 0)
        confidence = prediction.get("edge", {}).get("edge_confidence", "LOW")

        # Check if this is a contrarian bet
        is_contrarian = self._is_contrarian(prediction)

        sizing = self.allocator.compute_position_size(
            kelly_fraction=kelly,
            edge=edge,
            confidence=confidence,
            is_contrarian=is_contrarian,
        )

        return {
            "should_enter": sizing["action"] != "NO_BET",
            "filters": filter_result,
            "sizing": sizing,
            "regime": self.allocator.regime.regime_label,
            "is_contrarian": is_contrarian,
        }

    def evaluate_exit(
        self,
        market_id: str,
        current_edge: float,
        current_price: float,
        confidence: str,
        time_to_expiry_frac: float,
        imbalance: Optional[float] = None,
    ) -> dict:
        """Full exit evaluation using priority-based signal hierarchy.

        Returns the highest priority exit signal, or HOLD.
        """
        # Record imbalance if available
        if imbalance is not None:
            self.imbalance_detector.record(market_id, imbalance)

        # Increment hold counter
        self.hold_periods[market_id] = self.hold_periods.get(market_id, 0) + 1

        # Check each signal source
        imbalance_flip = self.imbalance_detector.check_flip(market_id)
        stall_check = self.stall_detector.check_stall(market_id, current_price, current_edge)

        # Evaluate hierarchy
        result = self.signal_hierarchy.evaluate(
            imbalance_flip=imbalance_flip,
            stall_check=stall_check,
            hold_periods=self.hold_periods.get(market_id, 0),
            confidence=confidence,
            current_edge=current_edge,
            time_to_expiry_frac=time_to_expiry_frac,
        )

        return result

    def record_entry(self, market_id: str, entry_price: float, entry_edge: float, amount: float):
        """Record a new position entry."""
        self.stall_detector.record_entry(market_id, entry_price, entry_edge)
        self.allocator.record_position(market_id, amount)
        self.hold_periods[market_id] = 0

    def record_exit(self, market_id: str):
        """Record a position exit and clean up."""
        self.imbalance_detector.clear(market_id)
        self.stall_detector.clear(market_id)
        self.allocator.close_position(market_id)
        self.hold_periods.pop(market_id, None)

    def record_model_outcome(self, model_name: str, predicted_prob: float,
                             market_price: float, actual_outcome: int):
        """Record outcome for model performance tracking."""
        self.wr_allocator.record_outcome(model_name, predicted_prob, market_price, actual_outcome)

    def get_model_weights(self, model_names: list[str]) -> dict[str, float]:
        """Get optimized model weights based on WR x payoff performance."""
        return self.wr_allocator.get_allocation_weights(model_names)

    def get_status(self) -> dict:
        """Get full strategy status."""
        return {
            "regime": self.allocator.regime.regime_label,
            "fear_greed": self.allocator.regime.fear_greed_value,
            "total_equity": self.allocator.total_equity,
            "deployed_pct": round(self.allocator.deployed / self.allocator.total_equity, 4),
            "available_pct": round(self.allocator.available_capital() / self.allocator.total_equity, 4),
            "n_positions": len(self.allocator.positions),
            "max_deployment": self.allocator.MAX_TOTAL_DEPLOYMENT,
            "kelly_fraction": self.allocator.KELLY_FRACTION,
            "model_report": self.wr_allocator.get_model_report(),
        }

    @staticmethod
    def _is_contrarian(prediction: dict) -> bool:
        """Detect if this bet is contrarian (against market consensus).

        A contrarian bet is when our model strongly disagrees with
        the market price direction in an extreme regime.
        """
        edge = prediction.get("edge", {}).get("edge", 0)
        price = prediction.get("market", {}).get("current_price", 0.5)

        # Contrarian = betting against extreme market sentiment
        if price > 0.85 and edge < -0.05:
            return True  # Market says very likely, we disagree
        if price < 0.15 and edge > 0.05:
            return True  # Market says very unlikely, we disagree
        return False
