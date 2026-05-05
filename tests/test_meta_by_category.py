"""Tests for per-category meta blending in v2_collector."""

from __future__ import annotations

import os

import pytest


class TestDefaultBehavior:
    def test_default_off_uses_global_blend_and_allows_all(self, monkeypatch):
        monkeypatch.delenv("ENABLE_META_BY_CATEGORY", raising=False)
        # Need to reload the module so META_BLEND env is re-read
        import importlib
        from scripts import v2_collector
        importlib.reload(v2_collector)
        for cat in ("sports", "crypto", "geopolitics", "totally_unknown"):
            blend, allows = v2_collector.meta_blend_for(cat)
            assert allows is True
            assert blend == v2_collector.META_BLEND


class TestEnabledBehavior:
    def test_sports_zero_blend_allowed(self, monkeypatch):
        monkeypatch.setenv("ENABLE_META_BY_CATEGORY", "1")
        from scripts import v2_collector
        blend, allows = v2_collector.meta_blend_for("sports")
        assert allows is True
        assert blend == 0.0

    def test_niche_sports_zero_blend(self, monkeypatch):
        monkeypatch.setenv("ENABLE_META_BY_CATEGORY", "1")
        from scripts import v2_collector
        blend, allows = v2_collector.meta_blend_for("niche_sports")
        assert allows is True
        assert blend == 0.0

    def test_other_half_blend(self, monkeypatch):
        monkeypatch.setenv("ENABLE_META_BY_CATEGORY", "1")
        from scripts import v2_collector
        blend, allows = v2_collector.meta_blend_for("other")
        assert allows is True
        assert blend == 0.5

    def test_geopolitics_full_meta(self, monkeypatch):
        monkeypatch.setenv("ENABLE_META_BY_CATEGORY", "1")
        from scripts import v2_collector
        blend, allows = v2_collector.meta_blend_for("geopolitics")
        assert allows is True
        assert blend == 1.0

    def test_unlisted_category_not_allowed(self, monkeypatch):
        monkeypatch.setenv("ENABLE_META_BY_CATEGORY", "1")
        from scripts import v2_collector
        for cat in ("crypto", "elections", "tech_ai", "fed_rate", "macro",
                    "oil_energy", "totally_unknown"):
            blend, allows = v2_collector.meta_blend_for(cat)
            assert not allows, f"category {cat} should be blocked"
            assert blend == 0.0


class TestTableConsistency:
    def test_blend_values_in_valid_range(self):
        from scripts import v2_collector
        for cat, blend in v2_collector.META_BLEND_BY_CATEGORY.items():
            assert 0.0 <= blend <= 1.0, f"{cat}: blend {blend} out of [0,1]"

    def test_proven_categories_have_entries(self):
        """Sports + niche_sports + other + geopolitics — the four categories
        with measured per-category meta-model walk-forward results."""
        from scripts import v2_collector
        for cat in ("sports", "niche_sports", "other", "geopolitics"):
            assert cat in v2_collector.META_BLEND_BY_CATEGORY
