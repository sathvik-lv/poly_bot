"""Tests for src/exit_simulator.py — pure functions for early-exit logic."""

from __future__ import annotations

import pytest

from src.exit_simulator import (
    group_records_by_market, held_pnl, per_dollar_pnl, should_exit,
    simulate_ledger, simulate_market,
)


class TestPerDollarPnL:
    def test_buy_yes_winning_resolution(self):
        # Bought at 0.50, market resolves YES (1.0)
        # Per-dollar: 1.0/0.50 - 1 = 1.0 (100% gain)
        assert per_dollar_pnl("BUY_YES", 0.50, 1.0) == pytest.approx(1.0)

    def test_buy_yes_losing_resolution(self):
        # Bought at 0.50, market resolves NO (0.0)
        assert per_dollar_pnl("BUY_YES", 0.50, 0.0) == pytest.approx(-1.0)

    def test_buy_no_winning_resolution(self):
        # Bought NO at 0.50 (no_price = 0.50), resolves NO (outcome=0)
        # Per-dollar: (1-0)/(1-0.5) - 1 = 1.0
        assert per_dollar_pnl("BUY_NO", 0.50, 0.0) == pytest.approx(1.0)

    def test_buy_yes_partial_exit(self):
        # Bought YES at 0.40, exit at 0.60 (price moved up)
        # 0.60/0.40 - 1 = 0.5 (50% gain)
        assert per_dollar_pnl("BUY_YES", 0.40, 0.60) == pytest.approx(0.5)

    def test_held_pnl_matches_resolution_payoff(self):
        assert held_pnl("BUY_YES", 0.30, 1.0) == pytest.approx(1/0.30 - 1)
        assert held_pnl("BUY_NO", 0.70, 0.0) == pytest.approx(1/(1-0.70) - 1)

    def test_unknown_action_returns_zero(self):
        assert per_dollar_pnl("HOLD", 0.5, 0.7) == 0.0

    def test_zero_price_safe(self):
        assert per_dollar_pnl("BUY_YES", 0.0, 0.5) == 0.0
        assert per_dollar_pnl("BUY_NO", 1.0, 0.5) == 0.0


class TestShouldExit:
    def test_no_decay_means_no_exit(self):
        assert not should_exit(entry_edge=0.10, current_edge=0.09, decay_threshold=0.5)

    def test_more_than_50pct_decay_triggers(self):
        # entry 0.10, current 0.04 -> 60% decay > 50% threshold
        assert should_exit(entry_edge=0.10, current_edge=0.04, decay_threshold=0.5)

    def test_sign_flip_always_triggers(self):
        # was bullish, now bearish — exit regardless of magnitude
        assert should_exit(entry_edge=0.10, current_edge=-0.01, decay_threshold=0.99)

    def test_zero_entry_edge_never_triggers(self):
        # if we entered with no edge, can't decay from nothing
        assert not should_exit(entry_edge=0.0, current_edge=-0.5, decay_threshold=0.5)

    def test_threshold_boundary(self):
        # entry 1.0, current 0.5, threshold 0.5 -> need < 0.5*1.0 = 0.5; 0.5 is NOT < 0.5
        assert not should_exit(entry_edge=1.0, current_edge=0.5, decay_threshold=0.5)


class TestSimulateMarket:
    def _make_records(self, action, entry_price, entry_edge, later_edge,
                      later_price, outcome, ts_offset_min=60):
        return [
            {"market_id": "m1", "timestamp": "2026-04-26T00:00:00+00:00",
             "action": action, "market_price": entry_price, "edge": entry_edge,
             "resolved": False},
            {"market_id": "m1",
             "timestamp": f"2026-04-26T0{ts_offset_min // 60}:00:00+00:00",
             "action": action, "market_price": later_price, "edge": later_edge,
             "resolved": False},
            {"market_id": "m1", "timestamp": "2026-04-27T00:00:00+00:00",
             "action": action, "market_price": entry_price, "edge": entry_edge,
             "resolved": True, "outcome": outcome},
        ]

    def test_no_decay_holds_to_resolution(self):
        # entry edge 0.10, later still 0.09 (no significant decay)
        recs = self._make_records("BUY_YES", 0.50, 0.10, 0.09, 0.51, 1.0)
        out = simulate_market(recs, decay_threshold=0.5, spread_cost=0.0)
        assert out is not None
        assert not out["exited"]
        assert out["held_pnl"] == pytest.approx(out["exit_pnl"])

    def test_decay_exits_at_later_price(self):
        # entry edge 0.10, later 0.02 (80% decay > 50%)
        # later price 0.55 (moved up from 0.50 entry) -> profit
        recs = self._make_records("BUY_YES", 0.50, 0.10, 0.02, 0.55, 1.0)
        out = simulate_market(recs, decay_threshold=0.5, spread_cost=0.0)
        assert out["exited"]
        assert out["exit_price"] == pytest.approx(0.55)
        # exit captures 0.55/0.50 - 1 = 0.10 vs holding 1.0
        assert out["exit_pnl"] == pytest.approx(0.10)
        assert out["held_pnl"] == pytest.approx(1.0)

    def test_decay_with_spread_cost(self):
        recs = self._make_records("BUY_YES", 0.50, 0.10, 0.02, 0.55, 1.0)
        out = simulate_market(recs, decay_threshold=0.5, spread_cost=0.05)
        # 0.10 gross - 0.05 spread = 0.05
        assert out["exit_pnl"] == pytest.approx(0.05)

    def test_no_resolved_record_returns_none(self):
        recs = [{"market_id": "m1", "timestamp": "2026-04-26T00:00:00+00:00",
                 "action": "BUY_YES", "market_price": 0.5, "edge": 0.1,
                 "resolved": False}]
        assert simulate_market(recs, 0.5, 0.0) is None

    def test_no_betting_action_returns_none(self):
        recs = [{"market_id": "m1", "timestamp": "2026-04-26T00:00:00+00:00",
                 "action": "NO_BET", "market_price": 0.5, "edge": 0.1,
                 "resolved": False},
                {"market_id": "m1", "timestamp": "2026-04-27T00:00:00+00:00",
                 "action": "NO_BET", "market_price": 0.5, "edge": 0.1,
                 "resolved": True, "outcome": 1.0}]
        assert simulate_market(recs, 0.5, 0.0) is None

    def test_ambiguous_outcome_returns_none(self):
        recs = self._make_records("BUY_YES", 0.5, 0.1, 0.05, 0.5, 0.5)
        assert simulate_market(recs, 0.5, 0.0) is None


class TestSimulateLedger:
    def test_aggregates_across_markets(self):
        # Two markets: one held wins, one exits
        recs = [
            # Market A: held wins big
            {"market_id": "A", "timestamp": "2026-04-26T00:00:00+00:00",
             "action": "BUY_YES", "market_price": 0.5, "edge": 0.1, "resolved": False},
            {"market_id": "A", "timestamp": "2026-04-27T00:00:00+00:00",
             "action": "BUY_YES", "market_price": 0.5, "edge": 0.1,
             "resolved": True, "outcome": 1.0},
            # Market B: edge decays, exit recommended
            {"market_id": "B", "timestamp": "2026-04-26T00:00:00+00:00",
             "action": "BUY_YES", "market_price": 0.5, "edge": 0.1, "resolved": False},
            {"market_id": "B", "timestamp": "2026-04-26T06:00:00+00:00",
             "action": "BUY_YES", "market_price": 0.55, "edge": 0.01, "resolved": False},
            {"market_id": "B", "timestamp": "2026-04-27T00:00:00+00:00",
             "action": "BUY_YES", "market_price": 0.5, "edge": 0.1,
             "resolved": True, "outcome": 0.0},
        ]
        out = simulate_ledger(recs, decay_threshold=0.5, spread_cost=0.0)
        assert out["n_markets"] == 2
        assert out["n_exited"] == 1
        # Hold: A=1.0 + B=-1.0 = 0
        # Exit: A=1.0 (no decay) + B=0.10 (exit at 0.55 vs 0.50 entry) = 1.10
        assert out["held_total_pnl"] == pytest.approx(0.0)
        assert out["exit_total_pnl"] == pytest.approx(1.10)
        assert out["diff"] == pytest.approx(1.10)


class TestGroupByMarket:
    def test_groups_and_sorts_by_timestamp(self):
        recs = [
            {"market_id": "x", "timestamp": "2026-04-27T00:00:00+00:00", "v": 2},
            {"market_id": "x", "timestamp": "2026-04-26T00:00:00+00:00", "v": 1},
            {"market_id": "y", "timestamp": "2026-04-26T12:00:00+00:00", "v": 3},
        ]
        grouped = group_records_by_market(recs)
        assert set(grouped.keys()) == {"x", "y"}
        assert [r["v"] for r in grouped["x"]] == [1, 2]
        assert [r["v"] for r in grouped["y"]] == [3]

    def test_skips_records_without_market_id(self):
        grouped = group_records_by_market([{"timestamp": "x"}, {"market_id": None}])
        assert grouped == {}
