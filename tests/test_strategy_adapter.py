"""Tests for the Strategy Adapter — calendar spread concepts for Polymarket."""

import time
import pytest
from src.strategy_adapter import (
    MarketRegime,
    CapitalAllocator,
    EntryFilterGate,
    ImbalanceFlipDetector,
    StallDetector,
    SignalHierarchy,
    WinRatePayoffAllocator,
    StrategyAdapter,
    REGIME_ALLOCATION,
)


# ===========================================================================
# MarketRegime Tests
# ===========================================================================

class TestMarketRegime:
    def test_extreme_fear(self):
        r = MarketRegime.from_fear_greed(10)
        assert r.regime_label == "extreme_fear"
        assert r.volatility_percentile == 0.9

    def test_fear(self):
        r = MarketRegime.from_fear_greed(25)
        assert r.regime_label == "fear"

    def test_neutral(self):
        r = MarketRegime.from_fear_greed(50)
        assert r.regime_label == "neutral"

    def test_greed(self):
        r = MarketRegime.from_fear_greed(75)
        assert r.regime_label == "greed"

    def test_extreme_greed(self):
        r = MarketRegime.from_fear_greed(90)
        assert r.regime_label == "extreme_greed"


# ===========================================================================
# CapitalAllocator Tests
# ===========================================================================

class TestCapitalAllocator:
    def test_max_deployment_60pct(self):
        alloc = CapitalAllocator(total_equity=10000)
        assert alloc.available_capital() == 6000  # 60% of 10000

    def test_kelly_fraction_one_third(self):
        assert CapitalAllocator.KELLY_FRACTION == pytest.approx(1/3)

    def test_position_size_high_confidence(self):
        alloc = CapitalAllocator(total_equity=10000)
        alloc.update_regime(50)  # neutral
        result = alloc.compute_position_size(
            kelly_fraction=0.30,
            edge=0.08,
            confidence="HIGH",
        )
        assert result["action"] in ("BUY_YES", "BUY_NO")
        assert result["kelly_adjusted"] == pytest.approx(0.10, abs=0.01)  # 0.30 * 1/3
        assert result["size_pct"] <= 0.12  # max per market in neutral

    def test_position_size_below_minimum(self):
        alloc = CapitalAllocator(total_equity=10000)
        result = alloc.compute_position_size(
            kelly_fraction=0.001,
            edge=0.001,
            confidence="LOW",
        )
        assert result["action"] == "NO_BET"

    def test_deployment_cap(self):
        alloc = CapitalAllocator(total_equity=10000)
        # Deploy 5500 already
        alloc.record_position("m1", 3000)
        alloc.record_position("m2", 2500)
        assert alloc.available_capital() == 500  # 6000 - 5500
        # New position can only use remaining 500
        result = alloc.compute_position_size(
            kelly_fraction=0.50,
            edge=0.10,
            confidence="HIGH",
        )
        assert result["size_dollars"] <= 500

    def test_close_position_frees_capital(self):
        alloc = CapitalAllocator(total_equity=10000)
        alloc.record_position("m1", 3000)
        assert alloc.available_capital() == 3000
        alloc.close_position("m1")
        assert alloc.available_capital() == 6000

    def test_contrarian_bonus_extreme_fear(self):
        alloc = CapitalAllocator(total_equity=10000)
        alloc.update_regime(10)  # extreme fear
        result_normal = alloc.compute_position_size(0.20, 0.06, "HIGH", is_contrarian=False)
        result_contrarian = alloc.compute_position_size(0.20, 0.06, "HIGH", is_contrarian=True)
        # Contrarian should get bonus allocation in extreme regime
        assert result_contrarian["size_pct"] >= result_normal["size_pct"]


# ===========================================================================
# EntryFilterGate Tests
# ===========================================================================

class TestEntryFilterGate:
    def _make_market(self, **overrides):
        base = {
            "spread": 0.02, "best_ask": 0.55, "best_bid": 0.53,
            "liquidity": 5000, "volume_24h": 1000,
        }
        base.update(overrides)
        return base

    def _make_prediction(self, **overrides):
        base = {
            "ensemble": {"model_agreement": 0.70},
            "edge": {"edge": 0.05},
            "market": {"current_price": 0.50},
            "sub_models": {"orderbook": {"signals": {"imbalance": 0.3}}},
        }
        base.update(overrides)
        return base

    def test_all_pass(self):
        gate = EntryFilterGate()
        result = gate.check_all(self._make_market(), self._make_prediction())
        assert result["pass"] is True
        assert result["n_passed"] == result["n_total"]

    def test_low_liquidity_fails(self):
        gate = EntryFilterGate()
        result = gate.check_all(
            self._make_market(liquidity=100),
            self._make_prediction(),
        )
        assert result["pass"] is False
        assert "liquidity" in result["failed_filters"]

    def test_wide_spread_fails(self):
        gate = EntryFilterGate()
        result = gate.check_all(
            self._make_market(spread=0.10, best_ask=0.55),
            self._make_prediction(),
        )
        assert result["pass"] is False
        assert "spread_sanity" in result["failed_filters"]

    def test_low_agreement_fails(self):
        gate = EntryFilterGate()
        result = gate.check_all(
            self._make_market(),
            self._make_prediction(ensemble={"model_agreement": 0.2}),
        )
        assert result["pass"] is False
        assert "model_agreement" in result["failed_filters"]

    def test_extreme_price_fails(self):
        gate = EntryFilterGate()
        result = gate.check_all(
            self._make_market(),
            self._make_prediction(market={"current_price": 0.98}),
        )
        assert result["pass"] is False
        assert "price_sanity" in result["failed_filters"]

    def test_high_imbalance_fails(self):
        gate = EntryFilterGate()
        result = gate.check_all(
            self._make_market(),
            self._make_prediction(
                sub_models={"orderbook": {"signals": {"imbalance": 0.95}}},
            ),
        )
        assert result["pass"] is False
        assert "imbalance" in result["failed_filters"]


# ===========================================================================
# ImbalanceFlipDetector Tests
# ===========================================================================

class TestImbalanceFlipDetector:
    def test_no_data_no_flip(self):
        det = ImbalanceFlipDetector(confirmation_periods=2)
        result = det.check_flip("m1")
        assert result["flip_detected"] is False
        assert result["signal"] == "INSUFFICIENT_DATA"

    def test_consistent_no_flip(self):
        det = ImbalanceFlipDetector(confirmation_periods=2)
        for _ in range(5):
            det.record("m1", 0.3)  # All positive = BUY side
        result = det.check_flip("m1")
        assert result["flip_detected"] is False

    def test_flip_detected_after_confirmation(self):
        det = ImbalanceFlipDetector(confirmation_periods=2)
        det.record("m1", 0.5)   # BUY
        det.record("m1", 0.3)   # BUY
        det.record("m1", -0.4)  # SELL (flip 1)
        det.record("m1", -0.6)  # SELL (flip 2 = confirmed)
        result = det.check_flip("m1")
        assert result["flip_detected"] is True
        assert result["from_side"] == "BUY"
        assert result["to_side"] == "SELL"

    def test_single_opposite_not_enough(self):
        det = ImbalanceFlipDetector(confirmation_periods=2)
        det.record("m1", 0.5)
        det.record("m1", -0.3)  # One opposite
        det.record("m1", 0.2)   # Back to original
        result = det.check_flip("m1")
        assert result["flip_detected"] is False

    def test_clear_resets(self):
        det = ImbalanceFlipDetector(confirmation_periods=2)
        det.record("m1", 0.5)
        det.clear("m1")
        result = det.check_flip("m1")
        assert result["signal"] == "INSUFFICIENT_DATA"


# ===========================================================================
# StallDetector Tests
# ===========================================================================

class TestStallDetector:
    def test_no_entry_no_stall(self):
        det = StallDetector()
        result = det.check_stall("m1", 0.50, 0.03)
        assert result["stalled"] is False

    def test_active_market_no_stall(self):
        det = StallDetector()
        det.record_entry("m1", 0.50, 0.05)
        # Price moved, edge intact
        result = det.check_stall("m1", 0.55, 0.04)
        assert result["stalled"] is False

    def test_stall_after_confirmation(self):
        det = StallDetector()
        det.record_entry("m1", 0.50, 0.05)
        # Price stuck, edge decayed — need 2 consecutive
        det.check_stall("m1", 0.501, 0.01)  # Stall 1
        result = det.check_stall("m1", 0.502, 0.01)  # Stall 2 = confirmed
        assert result["stalled"] is True
        assert result["signal"] == "EXIT_STALL"

    def test_recovery_resets_count(self):
        det = StallDetector()
        det.record_entry("m1", 0.50, 0.05)
        det.check_stall("m1", 0.501, 0.01)  # Stall 1
        det.check_stall("m1", 0.56, 0.04)   # Recovery (price moved)
        result = det.check_stall("m1", 0.501, 0.01)  # Stall again but count reset
        assert result["stalled"] is False
        assert result["consecutive_stalls"] == 1


# ===========================================================================
# SignalHierarchy Tests
# ===========================================================================

class TestSignalHierarchy:
    def test_all_clear(self):
        sh = SignalHierarchy()
        result = sh.evaluate(
            imbalance_flip={"flip_detected": False},
            stall_check={"stalled": False},
            hold_periods=2,
            confidence="HIGH",
            current_edge=0.05,
            time_to_expiry_frac=0.5,
        )
        assert result["signal"] == "HOLD"
        assert result["all_clear"] is True

    def test_imbalance_flip_highest_priority(self):
        sh = SignalHierarchy()
        result = sh.evaluate(
            imbalance_flip={"flip_detected": True, "from_side": "BUY", "to_side": "SELL"},
            stall_check={"stalled": True, "consecutive_stalls": 3},  # Also triggered
            hold_periods=15,  # Also triggered
            confidence="HIGH",
            current_edge=0.005,  # Also triggered
            time_to_expiry_frac=0.5,
        )
        # Should return highest priority (imbalance flip) even though others fired
        assert result["signal"] == "IMBALANCE_FLIP"
        assert result["priority"] == 1
        assert result["n_exit_signals"] >= 3

    def test_time_exit(self):
        sh = SignalHierarchy()
        result = sh.evaluate(
            imbalance_flip={"flip_detected": False},
            stall_check={"stalled": False},
            hold_periods=10,
            confidence="HIGH",  # max 10 periods for HIGH
            current_edge=0.05,
            time_to_expiry_frac=0.5,
        )
        assert result["signal"] == "TIME_EXIT"

    def test_expiry_close(self):
        sh = SignalHierarchy()
        result = sh.evaluate(
            imbalance_flip={"flip_detected": False},
            stall_check={"stalled": False},
            hold_periods=2,
            confidence="HIGH",
            current_edge=0.05,
            time_to_expiry_frac=0.03,  # 3% remaining
        )
        assert result["signal"] == "EXPIRY_CLOSE"

    def test_edge_evaporated(self):
        sh = SignalHierarchy()
        result = sh.evaluate(
            imbalance_flip={"flip_detected": False},
            stall_check={"stalled": False},
            hold_periods=2,
            confidence="HIGH",
            current_edge=0.005,  # Below 1% minimum
            time_to_expiry_frac=0.5,
        )
        assert result["signal"] == "EDGE_EVAPORATED"


# ===========================================================================
# WinRatePayoffAllocator Tests
# ===========================================================================

class TestWinRatePayoffAllocator:
    def test_equal_weights_no_data(self):
        alloc = WinRatePayoffAllocator()
        weights = alloc.get_allocation_weights(["model_a", "model_b"])
        assert weights["model_a"] == pytest.approx(0.5)
        assert weights["model_b"] == pytest.approx(0.5)

    def test_better_model_gets_more_weight(self):
        alloc = WinRatePayoffAllocator()
        # Model A: 80% WR, good payoff
        for i in range(80):
            alloc.record_outcome("model_a", 0.60, 0.50, 1)
        for i in range(20):
            alloc.record_outcome("model_a", 0.60, 0.50, 0)
        # Model B: 50% WR, mediocre
        for i in range(50):
            alloc.record_outcome("model_b", 0.55, 0.50, 1)
        for i in range(50):
            alloc.record_outcome("model_b", 0.55, 0.50, 0)

        weights = alloc.get_allocation_weights(["model_a", "model_b"])
        assert weights["model_a"] > weights["model_b"]

    def test_model_report(self):
        alloc = WinRatePayoffAllocator()
        for i in range(15):
            alloc.record_outcome("test_model", 0.60, 0.50, 1 if i < 12 else 0)
        report = alloc.get_model_report()
        assert len(report) == 1
        assert report[0]["model"] == "test_model"
        assert report[0]["win_rate"] == pytest.approx(0.80, abs=0.01)


# ===========================================================================
# StrategyAdapter Integration Tests
# ===========================================================================

class TestStrategyAdapter:
    def _make_market(self):
        return {
            "spread": 0.02, "best_ask": 0.55, "best_bid": 0.53,
            "liquidity": 5000, "volume_24h": 1000,
        }

    def _make_prediction(self):
        return {
            "ensemble": {"model_agreement": 0.70},
            "edge": {"edge": 0.05, "edge_confidence": "HIGH"},
            "market": {"current_price": 0.50},
            "sizing": {"kelly_fraction": 0.15},
            "sub_models": {"orderbook": {"signals": {"imbalance": 0.3}}},
        }

    def test_full_entry_evaluation(self):
        sa = StrategyAdapter(total_equity=10000)
        result = sa.evaluate_entry(self._make_market(), self._make_prediction())
        assert "should_enter" in result
        assert "filters" in result
        assert "sizing" in result
        assert "regime" in result

    def test_entry_with_regime(self):
        sa = StrategyAdapter(total_equity=10000)
        sa.allocator.update_regime(10)  # extreme fear
        result = sa.evaluate_entry(self._make_market(), self._make_prediction())
        assert result["regime"] == "extreme_fear"

    def test_full_exit_evaluation(self):
        sa = StrategyAdapter(total_equity=10000)
        sa.record_entry("m1", 0.50, 0.05, 1000)
        result = sa.evaluate_exit(
            market_id="m1",
            current_edge=0.04,
            current_price=0.52,
            confidence="HIGH",
            time_to_expiry_frac=0.5,
            imbalance=0.3,
        )
        assert "signal" in result
        assert "action" in result

    def test_status(self):
        sa = StrategyAdapter(total_equity=10000)
        status = sa.get_status()
        assert status["total_equity"] == 10000
        assert status["max_deployment"] == 0.60
        assert status["kelly_fraction"] == pytest.approx(1/3)
