"""Tests for paper_trader env-var overrides used by V3a/V3b parallel arms."""

from __future__ import annotations

import importlib

import pytest


def reload_paper_trader(monkeypatch, **env):
    """Force-reload scripts.paper_trader after setting env vars."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import scripts.paper_trader as pt
    importlib.reload(pt)
    return pt


class TestStateFileOverride:
    def test_default_state_file(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_STATE_FILE=None,
        )
        assert pt.PAPER_FILE.endswith("paper_trades.json")
        assert "v3" not in pt.PAPER_FILE.lower()

    def test_v3a_state_file(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_STATE_FILE="data/v3a_paper_trades.json",
        )
        assert pt.PAPER_FILE == "data/v3a_paper_trades.json"

    def test_v3b_state_file(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_STATE_FILE="data/v3b_paper_trades.json",
        )
        assert pt.PAPER_FILE == "data/v3b_paper_trades.json"


class TestAnalyticsFileOverride:
    def test_default_analytics(self, monkeypatch):
        pt = reload_paper_trader(monkeypatch, PAPER_TRADER_ANALYTICS_FILE=None)
        assert pt.ANALYTICS_FILE.endswith("weight_analytics.json")

    def test_overridden_analytics(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_ANALYTICS_FILE="data/v3a_weight_analytics.json",
        )
        assert pt.ANALYTICS_FILE == "data/v3a_weight_analytics.json"


class TestCategoryFilter:
    def test_default_no_filter(self, monkeypatch):
        pt = reload_paper_trader(monkeypatch, PAPER_TRADER_TRADE_CATEGORIES=None)
        assert pt.TRADE_CATEGORIES_FILTER is None

    def test_empty_string_no_filter(self, monkeypatch):
        pt = reload_paper_trader(monkeypatch, PAPER_TRADER_TRADE_CATEGORIES="")
        assert pt.TRADE_CATEGORIES_FILTER is None

    def test_v3a_filter(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_TRADE_CATEGORIES="niche_sports,geopolitics,other",
        )
        assert pt.TRADE_CATEGORIES_FILTER == {"niche_sports", "geopolitics", "other"}

    def test_v3b_filter(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_TRADE_CATEGORIES="niche_sports,geopolitics,sports",
        )
        assert pt.TRADE_CATEGORIES_FILTER == {"niche_sports", "geopolitics", "sports"}

    def test_filter_strips_whitespace(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_TRADE_CATEGORIES=" sports , niche_sports ",
        )
        assert pt.TRADE_CATEGORIES_FILTER == {"sports", "niche_sports"}

    def test_filter_drops_empty_entries(self, monkeypatch):
        pt = reload_paper_trader(
            monkeypatch,
            PAPER_TRADER_TRADE_CATEGORIES="sports,,,other",
        )
        assert pt.TRADE_CATEGORIES_FILTER == {"sports", "other"}


def teardown_module(module):
    """Reset paper_trader to defaults after this test module."""
    import os
    for var in ("PAPER_TRADER_STATE_FILE", "PAPER_TRADER_ANALYTICS_FILE",
                "PAPER_TRADER_TRADE_CATEGORIES"):
        os.environ.pop(var, None)
    import scripts.paper_trader as pt
    importlib.reload(pt)
