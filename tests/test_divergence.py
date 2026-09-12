"""Tests for src/divergence.py — pure helpers for the sportsbook-vs-Polymarket study."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.divergence import (
    DRAW, analyze, book_consensus, closing_snapshots, cluster_summarize,
    credits_for_poll, decimal_to_prob, devig, edges, key_priority,
    market_shape, match_markets, mde, name_score, pair_score, parse_ts,
    per_dollar, resolution_from_market, side_prices, split_title, summarize,
    target_for_market,
)


def _book(key, outcomes):
    return {"key": key, "markets": [{"key": "h2h", "outcomes": [
        {"name": n, "price": p} for n, p in outcomes]}]}


def _mkt(outcomes, q="A vs. B", smt="moneyline", bid=0.40, ask=0.42):
    return {"outcomes": json.dumps(outcomes), "question": q,
            "sportsMarketType": smt, "bestBid": bid, "bestAsk": ask}


class TestParseTs:
    @pytest.mark.parametrize("raw", ["2026-09-12T11:35:00Z", "2026-09-12 11:35:00+00",
                                     "2026-09-12T11:35:00+00:00"])
    def test_formats_from_both_apis(self, raw):
        assert parse_ts(raw) == datetime(2026, 9, 12, 11, 35, tzinfo=timezone.utc)

    def test_garbage(self):
        assert parse_ts(None) is None
        assert parse_ts("soon") is None

    def test_naive_is_treated_as_utc(self):
        assert parse_ts("2026-09-12T11:35:00").tzinfo is not None


class TestOddsMath:
    def test_decimal_to_prob(self):
        assert decimal_to_prob(2.0) == pytest.approx(0.5)
        assert decimal_to_prob("1.25") == pytest.approx(0.8)

    @pytest.mark.parametrize("bad", [None, "", "x", 1.0, 0.5, float("nan"), float("inf")])
    def test_decimal_to_prob_rejects_garbage(self, bad):
        assert decimal_to_prob(bad) is None

    def test_devig_removes_margin(self):
        fair = devig([1 / 1.9, 1 / 1.9])
        assert fair == pytest.approx([0.5, 0.5])

    def test_devig_rejects_missing(self):
        assert devig([0.5, None]) is None
        assert devig([]) is None


class TestBookConsensus:
    def test_median_across_books(self):
        books = [("a", [("A", 1.5), ("B", 3.0)]),
                 ("b", [("A", 1.6), ("B", 2.6)]),
                 ("pinnacle", [("A", 1.55), ("B", 2.8)])]
        c = book_consensus({"bookmakers": [_book(k, o) for k, o in books]})
        fair_a = sorted(devig([1 / o[0][1], 1 / o[1][1]])[0] for _, o in books)
        assert c["n_books"] == 3
        assert c["probs"]["A"] == pytest.approx(fair_a[1])
        assert sum(c["probs"].values()) == pytest.approx(1.0)
        assert c["pinnacle"] is not None

    def test_two_and_three_way_lines_are_not_blended(self):
        c = book_consensus({"bookmakers": [
            _book("a", [("A", 2.0), ("B", 2.0)]),
            _book("b", [("A", 2.1), ("B", 3.5), ("Draw", 3.2)]),
            _book("c", [("A", 2.2), ("B", 3.4), ("Draw", 3.1)]),
        ]})
        assert set(c["probs"]) == {"A", "B", "Draw"}
        assert c["n_books"] == 2

    def test_other_market_keys_ignored(self):
        ev = {"bookmakers": [{"key": "a", "markets": [{"key": "spreads", "outcomes": [
            {"name": "A", "price": 1.9}, {"name": "B", "price": 1.9}]}]}]}
        assert book_consensus(ev)["n_books"] == 0

    def test_empty(self):
        c = book_consensus({"bookmakers": []})
        assert c["n_books"] == 0 and c["probs"] == {}


class TestNames:
    @pytest.mark.parametrize("a,b", [
        ("Texas A&M Aggies", "Texas A&M"),
        ("Army Black Knights", "Army"),
        ("Wuhan San Zhen FC", "wuhan san zhen"),
        ("Alex de Minaur", "Alex De Minaur"),
        ("José Aldo", "Jose Aldo"),
    ])
    def test_same_entity_scores_high(self, a, b):
        assert name_score(a, b) >= 0.75

    def test_different_entities_score_low(self):
        assert name_score("Michigan Wolverines", "Oklahoma Sooners") < 0.75

    def test_empty(self):
        assert name_score("", "Army") == 0.0

    def test_pair_score_detects_orientation(self):
        score, swapped = pair_score("Texas A&M Aggies", "Arizona State Sun Devils",
                                    "Arizona State", "Texas A&M")
        assert score >= 0.75 and swapped is True


class TestSplitTitle:
    def test_basic(self):
        assert split_title("Arizona State vs. Texas A&M") == ("Arizona State", "Texas A&M")

    def test_event_prefix_and_weight_class_stripped(self):
        assert split_title("Noche UFC: Jean Silva vs. Jose Miguel Delgado (Featherweight, Main Card)") \
            == ("Jean Silva", "Jose Miguel Delgado")

    def test_sub_events_rejected(self):
        assert split_title("Wuhan San Zhen FC vs. Henan FC - Exact Score") is None

    def test_not_a_matchup(self):
        assert split_title("Will Bitcoin reach $100k?") is None


class TestMarketShape:
    def test_competitor_moneyline(self):
        assert market_shape(_mkt(["Army", "South Florida"])) == "h2h"

    def test_competitor_without_type_allowed(self):
        assert market_shape(_mkt(["Zverev", "Halys"], q="Zverev vs Halys", smt=None)) == "h2h"

    def test_yesno_requires_moneyline_type(self):
        assert market_shape(_mkt(["Yes", "No"], q="Will Henan FC win on 2026-09-12?")) == "yesno"
        # outright futures are Yes/No too, but not tagged moneyline
        assert market_shape(_mkt(["Yes", "No"], q="Will Zverev win the US Open?", smt=None)) is None

    @pytest.mark.parametrize("smt", ["spreads", "totals", "team_totals"])
    def test_props_rejected_by_type(self, smt):
        assert market_shape(_mkt(["A", "B"], smt=smt)) is None

    def test_props_rejected_by_wording(self):
        assert market_shape(_mkt(["Zverev", "Halys"], q="Zverev vs Halys: Set 1 Winner", smt=None)) is None

    def test_over_under_and_three_way_rejected(self):
        assert market_shape(_mkt(["Over", "Under"], smt=None)) is None
        assert market_shape(_mkt(["A", "B", "C"])) is None


class TestTargetForMarket:
    def test_h2h_first_outcome_can_be_the_away_team(self):
        m = _mkt(["Arizona State", "Texas A&M"])
        assert target_for_market(m, "h2h", "Texas A&M Aggies", "Arizona State Sun Devils") \
            == "Arizona State Sun Devils"

    def test_h2h_unrelated_game(self):
        assert target_for_market(_mkt(["Lakers", "Celtics"]), "h2h",
                                 "Texas A&M Aggies", "Arizona State Sun Devils") is None

    def test_yesno_team_win(self):
        m = _mkt(["Yes", "No"], q="Will Henan FC win on 2026-09-12?")
        assert target_for_market(m, "yesno", "Wuhan San Zhen FC", "Henan FC",
                                 "Wuhan San Zhen FC vs. Henan FC") == "Henan FC"

    def test_yesno_draw(self):
        m = _mkt(["Yes", "No"], q="Will Wuhan San Zhen FC vs. Henan FC end in a draw?")
        assert target_for_market(m, "yesno", "Wuhan San Zhen FC", "Henan FC",
                                 "Wuhan San Zhen FC vs. Henan FC") == DRAW

    def test_yesno_parent_must_be_the_same_game(self):
        m = _mkt(["Yes", "No"], q="Will Henan FC win on 2026-09-12?")
        assert target_for_market(m, "yesno", "Arsenal", "Chelsea",
                                 "Wuhan San Zhen FC vs. Henan FC") is None


class TestMatchMarkets:
    BOOK = ("americanfootball_ncaaf", {"id": "e1", "home_team": "Texas A&M Aggies",
                                       "away_team": "Arizona State Sun Devils",
                                       "commence_time": "2026-09-13T19:00:00Z"})

    def _m(self, start):
        return dict(_mkt(["Arizona State", "Texas A&M"]), id="m1", gameStartTime=start)

    def test_pairs_market_with_book_event(self):
        out = match_markets([self.BOOK], [(self._m("2026-09-13 19:00:00+00"), {}, "h2h")])
        assert len(out) == 1 and out[0]["target"] == "Arizona State Sun Devils"

    def test_time_guard_rejects_a_different_date(self):
        assert match_markets([self.BOOK], [(self._m("2026-09-20 19:00:00+00"), {}, "h2h")]) == []

    def test_market_paired_once_even_if_listed_under_two_keys(self):
        dup = ("americanfootball_ncaaf_fcs", dict(self.BOOK[1], id="e2"))
        out = match_markets([self.BOOK, dup], [(self._m("2026-09-13 19:00:00+00"), {}, "h2h")])
        assert len(out) == 1 and out[0]["sport"] == "americanfootball_ncaaf"


class TestQuotesAndEdges:
    def test_side_prices(self):
        px = side_prices({"bestBid": 0.48, "bestAsk": 0.50})
        assert px["mid"] == pytest.approx(0.49) and px["spread"] == pytest.approx(0.02)

    @pytest.mark.parametrize("bid,ask", [(None, 0.5), (0.5, None), (0.5, 0.5), (0.6, 0.5),
                                         (0, 0.5), (0.5, 1.0), ("x", 0.5)])
    def test_missing_or_crossed_quote_is_none_not_fifty_cents(self, bid, ask):
        assert side_prices({"bestBid": bid, "bestAsk": ask}) is None

    PX = {"bid": 0.50, "ask": 0.52, "mid": 0.51, "spread": 0.02}

    def test_book_above_ask_buys_target(self):
        e = edges(0.60, self.PX)
        assert e["best_side"] == "target"
        assert e["best_edge"] == pytest.approx(0.08)
        assert e["entry_price"] == pytest.approx(0.52)
        assert e["div_mid"] == pytest.approx(0.09)

    def test_book_below_bid_buys_the_other_side(self):
        e = edges(0.40, self.PX)
        assert e["best_side"] == "against"
        assert e["best_edge"] == pytest.approx(0.10)
        assert e["entry_price"] == pytest.approx(0.50)

    def test_inside_the_spread_is_negative_edge(self):
        assert edges(0.51, self.PX)["best_edge"] == pytest.approx(-0.01)


class TestPerDollar:
    def test_target(self):
        assert per_dollar("target", 0.25, True) == pytest.approx(3.0)
        assert per_dollar("target", 0.25, False) == -1.0

    def test_against(self):
        assert per_dollar("against", 0.5, False) == pytest.approx(1.0)
        assert per_dollar("against", 0.5, True) == -1.0

    def test_unknown_side(self):
        with pytest.raises(ValueError):
            per_dollar("yes", 0.5, True)


class TestResolution:
    def test_settled(self):
        assert resolution_from_market({"closed": True, "outcomePrices": '["1", "0"]'}) is True
        assert resolution_from_market({"closed": True, "outcomePrices": '["0", "1"]'}) is False

    def test_open_market_is_pending(self):
        assert resolution_from_market({"closed": False, "outcomePrices": '["1", "0"]'}) is None

    def test_closed_but_undecided_is_pending_not_void(self):
        assert resolution_from_market({"closed": True, "outcomePrices": '["0.97", "0.03"]'}) is None

    def test_void(self):
        assert resolution_from_market({"closed": True, "outcomePrices": '["0.5", "0.5"]',
                                       "umaResolutionStatus": "resolved"}) == "void"


class TestCreditPacing:
    NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)

    def test_spreads_quota_until_month_end(self):
        # 467 spendable over 456h / 3h = 152 polls -> 3 per poll
        assert credits_for_poll(492, self.NOW, reserve=25, poll_hours=3.0) == 3

    def test_reserve_is_never_spent(self):
        assert credits_for_poll(25, self.NOW, reserve=25) == 0
        assert credits_for_poll(10, self.NOW, reserve=25) == 0
        assert credits_for_poll(None, self.NOW) == 0

    def test_spends_faster_near_reset_but_capped(self):
        late = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
        assert credits_for_poll(200, late, reserve=25, poll_hours=3.0) == 20

    def test_december_rolls_to_january(self):
        now = datetime(2026, 12, 31, 21, tzinfo=timezone.utc)
        assert credits_for_poll(40, now, reserve=25, poll_hours=3.0, cap=100) == 15

    def test_key_priority_favours_imminent_games(self):
        assert key_priority([1, 5, 40]) == pytest.approx(2.2)
        assert key_priority([-1, None]) == 0.0
        assert key_priority([2]) > key_priority([30])


class TestStats:
    def test_summarize(self):
        s = summarize([1.0, -1.0, 1.0, -1.0])
        assert s["n"] == 4 and s["mean"] == pytest.approx(0.0)
        assert s["lo"] < 0 < s["hi"]

    def test_summarize_small(self):
        assert summarize([])["mean"] is None
        s = summarize([0.5])
        assert s["mean"] == 0.5 and s["sd"] is None

    def test_correlated_legs_of_one_game_count_once(self):
        s = cluster_summarize([("g1", 1.0), ("g1", 1.0), ("g1", -1.0), ("g2", -1.0)])
        assert s["n"] == 2 and s["n_obs"] == 4
        assert s["mean"] == pytest.approx(((1 / 3) - 1.0) / 2)

    def test_mde(self):
        assert mde(0.5, 100) == pytest.approx(2.80 * 0.05)
        assert mde(None, 100) is None and mde(0.5, 1) is None


def _rec(i, book_p, won, bid=0.50, ask=0.52, h=2.0, ts="2026-09-12T10:00:00+00:00"):
    px = side_prices({"bestBid": bid, "bestAsk": ask})
    return {"ts": ts, "poly_market_id": f"m{i}", "book_event_id": f"e{i}",
            "h_to_start": h, "book_p": book_p, **px, **edges(book_p, px),
            "resolved": True, "void": False, "target_won": won}


class TestAnalyze:
    def test_insufficient_data(self):
        assert analyze([_rec(1, 0.6, True)])["verdict"] == "INSUFFICIENT_DATA"

    def test_closing_snapshot_is_last_before_start(self):
        early = _rec(1, 0.60, True, ts="2026-09-12T08:00:00+00:00")
        late = _rec(1, 0.55, True, bid=0.54, ask=0.56, ts="2026-09-12T10:00:00+00:00")
        inplay = _rec(1, 0.90, True, bid=0.88, ask=0.90, h=-0.5, ts="2026-09-12T12:00:00+00:00")
        snaps = closing_snapshots([early, late, inplay])
        assert len(snaps) == 1 and snaps[0]["book_p"] == 0.55

    def test_edge_when_results_follow_the_book(self):
        # books say 70%, Polymarket asks 52c, and 70% of games go the book's way
        a = analyze([_rec(i, 0.70, won=(i % 10 < 7)) for i in range(200)])
        assert a["verdict"] == "EDGE"
        assert a["brier"]["mean"] < 0

    def test_negative_when_results_contradict_the_book(self):
        a = analyze([_rec(i, 0.70, won=(i % 10 < 3)) for i in range(200)])
        assert a["verdict"] == "NEGATIVE"
