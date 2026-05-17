"""Tests for the enhanced classify_market: new keywords + metadata-aware path.

These tests cover the recent fixes that flipped Italian Open tennis and MLB
baseball from 'other' back to their proper categories. Also exercises the
new optional raw_market argument that lets the classifier inspect event
titles/tickers.
"""

from __future__ import annotations

import pytest


# Import both copies — they should agree on the new keywords
from scripts.paper_trader import classify_market as cls_pt
from scripts.test1_collector import classify_market as cls_t1


class TestNicheSportsKeywords:
    def test_internazionali_italian_open(self):
        # Real market we saw silently bucketed as 'other' before fix
        q = "Internazionali BNL d'Italia: Casper Ruud vs Karen Khachanov"
        assert cls_pt(q) == "niche_sports"
        assert cls_t1(q) == "niche_sports"

    def test_italian_open_english(self):
        assert cls_pt("Italian Open: Sinner vs Alcaraz") == "niche_sports"

    def test_atp_tour_event(self):
        assert cls_pt("ATP Tour Madrid: Djokovic vs Medvedev") == "niche_sports"

    def test_challenger_circuit(self):
        assert cls_pt("ATP Challenger Phoenix: Player A vs Player B") == "niche_sports"

    def test_next_gen_finals(self):
        assert cls_pt("Next Gen ATP Finals: Round Robin") == "niche_sports"


class TestSportsKeywordsExpanded:
    @pytest.mark.parametrize("q,expected_cat", [
        ("Yankees vs Red Sox", "sports"),
        ("Colorado Rockies vs Philadelphia Phillies", "sports"),
        ("Atlanta Braves vs Los Angeles Dodgers", "sports"),
        ("Detroit Tigers vs New York Mets", "sports"),
        ("San Diego Padres vs Cleveland Guardians", "sports"),
        ("Toronto Blue Jays at Baltimore Orioles", "sports"),
        ("Houston Astros vs Chicago Cubs", "sports"),
        ("Boston Bruins win 2026 Stanley Cup?", "sports"),
    ])
    def test_mlb_nhl_team_names_classified_as_sports(self, q, expected_cat):
        assert cls_pt(q) == expected_cat, f"{q!r} -> {cls_pt(q)}"
        assert cls_t1(q) == expected_cat, f"{q!r} -> {cls_t1(q)}"


class TestNicheSportsTakesPriorityOverSports:
    """niche_sports must come BEFORE sports in CATEGORY_RULES order so that
    tennis tournament names don't get caught by sports' 'tournament' keyword.
    """

    def test_tennis_tournament_wins_over_sports(self):
        # 'tournament' is a sports keyword but 'ATP' should win first
        q = "ATP Tour championship tournament: A vs B"
        assert cls_pt(q) == "niche_sports"
        assert cls_t1(q) == "niche_sports"

    def test_qualifier_wins_over_match_keyword(self):
        q = "Qualifying round: Player A vs Player B match"
        assert cls_pt(q) == "niche_sports"


class TestMetadataAwareClassifier:
    def test_question_only_still_works(self):
        # Backwards-compat: every existing caller passes just the question
        assert cls_pt("Will Bitcoin reach $100k?") == "crypto"

    def test_passing_none_raw_market_is_safe(self):
        assert cls_pt("Lakers vs Celtics", None) == "sports"

    def test_event_title_pulls_through_to_classification(self):
        # Question alone has no league keyword
        q = "Will Team A beat Team B?"
        market = {"events": [{"title": "MLB 2026 Season", "ticker": "mlb"}]}
        assert cls_pt(q, market) == "sports"

    def test_event_slug_pulls_through(self):
        market = {"events": [{"slug": "atp-finals-2026"}]}
        assert cls_pt("Will A beat B?", market) == "niche_sports"

    def test_event_description_pulls_through(self):
        market = {"events": [{"description": "FIFA World Cup quarter-final"}]}
        assert cls_pt("Will A beat B?", market) == "sports"

    def test_malformed_market_does_not_crash(self):
        # Defensive: weird shapes should fall back to question-only
        for bad in (
            {"events": None},
            {"events": "not a list"},
            {"events": [{"title": None}]},
            {"events": [None]},
            {"events": [{"title": 12345}]},  # int, not str
        ):
            # Should not raise
            cat = cls_pt("Will Bitcoin reach $100k?", bad)
            assert cat in ("crypto", "other")  # falls through to keyword match

    def test_empty_question_with_metadata_still_classifies(self):
        market = {"events": [{"title": "NBA Playoffs 2026"}]}
        assert cls_pt("", market) == "sports"

    def test_both_copies_agree_on_metadata_path(self):
        market = {"events": [{"title": "MLB", "ticker": "mlb"}]}
        q = "Team A wins by 5"
        assert cls_pt(q, market) == cls_t1(q, market) == "sports"


class TestBackwardsCompatibility:
    """The signature change (optional second arg) must not break any
    existing call site that passes just the question."""

    def test_no_kwargs_just_question(self):
        # All these existing call patterns must still work
        assert cls_pt("Will Trump win 2028?") in ("elections", "other")
        assert cls_pt("Bitcoin price prediction") == "crypto"
        assert cls_pt("") == "other"
        assert cls_pt(None) == "other"
