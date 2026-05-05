"""Tests for category-tier multiplier behavior in paper_trader + v2_collector.

Both modules import the same CATEGORY_TIER_MULT table from paper_trader and
duplicate it in v2_collector for independence. Verify they agree and that the
ENABLE_CATEGORY_TIERS env var gates them off by default.
"""

from __future__ import annotations

import os

import pytest


class TestPaperTraderTiers:
    def test_table_present(self):
        from scripts import paper_trader
        assert "sports" in paper_trader.CATEGORY_TIER_MULT
        assert "niche_sports" in paper_trader.CATEGORY_TIER_MULT
        assert "other" in paper_trader.CATEGORY_TIER_MULT

    def test_high_tier_full_kelly(self):
        from scripts import paper_trader
        assert paper_trader.CATEGORY_TIER_MULT["sports"] == 1.0
        assert paper_trader.CATEGORY_TIER_MULT["niche_sports"] == 1.0

    def test_medium_tier_half_kelly(self):
        from scripts import paper_trader
        assert paper_trader.CATEGORY_TIER_MULT["other"] == 0.5

    def test_skip_tier_zero(self):
        from scripts import paper_trader
        for cat in ("crypto", "geopolitics", "elections", "tech_ai"):
            assert paper_trader.CATEGORY_TIER_MULT[cat] == 0.0

    def test_default_is_skip(self):
        from scripts import paper_trader
        assert paper_trader.DEFAULT_TIER_MULT == 0.0


class TestV2CollectorTierFn:
    def test_default_off_returns_full_mult(self, monkeypatch):
        monkeypatch.delenv("ENABLE_CATEGORY_TIERS", raising=False)
        from scripts import v2_collector
        mult, label = v2_collector.tier_for("sports")
        assert mult == 1.0
        assert label == "ungated"
        # Even SKIP-tier categories pass through ungated when env off
        mult2, label2 = v2_collector.tier_for("crypto")
        assert mult2 == 1.0
        assert label2 == "ungated"

    def test_enabled_high_tier(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CATEGORY_TIERS", "1")
        from scripts import v2_collector
        mult, label = v2_collector.tier_for("sports")
        assert mult == 1.0
        assert label == "HIGH"

    def test_enabled_medium_tier(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CATEGORY_TIERS", "1")
        from scripts import v2_collector
        mult, label = v2_collector.tier_for("other")
        assert mult == 0.5
        assert label == "MEDIUM"

    def test_enabled_skip_tier(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CATEGORY_TIERS", "1")
        from scripts import v2_collector
        for cat in ("crypto", "geopolitics", "elections", "tech_ai", "fed_rate"):
            mult, label = v2_collector.tier_for(cat)
            assert mult == 0.0
            assert label == "SKIP"

    def test_enabled_unknown_category_skipped(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CATEGORY_TIERS", "1")
        from scripts import v2_collector
        mult, label = v2_collector.tier_for("brand_new_category_xyz")
        assert mult == 0.0
        assert label == "SKIP"


class TestTierTablesAgree:
    def test_paper_trader_and_v2_collector_have_same_mults(self):
        from scripts import paper_trader, v2_collector
        for cat in paper_trader.CATEGORY_TIER_MULT:
            assert (paper_trader.CATEGORY_TIER_MULT[cat]
                    == v2_collector.CATEGORY_TIER_MULT.get(cat)), \
                f"Disagreement on category {cat}"
