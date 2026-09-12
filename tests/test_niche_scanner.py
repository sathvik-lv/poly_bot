"""Tests for scripts/niche_scanner.py — crypto threshold question parsing."""

from __future__ import annotations

import pytest

from scripts.niche_scanner import parse_crypto_target


@pytest.mark.parametrize("question,coin,direction,target", [
    # The old greedy regex parsed these three as targets of $2, $3 and $6.
    ("Will the price of Bitcoin be above $84,000 on September 12?", "bitcoin", "above", 84_000),
    ("Will Bitcoin reach $84,000 September 7-13?", "bitcoin", "above", 84_000),
    ("Will Bitcoin reach $95,000 by December 31, 2026?", "bitcoin", "above", 95_000),
    ("Will Bitcoin hit $100k by December 31?", "bitcoin", "above", 100_000),
    ("Will Bitcoin reach $1.5m by 2030?", "bitcoin", "above", 1_500_000),
    ("Will Bitcoin dip to $70,000 in September?", "bitcoin", "below", 70_000),
    ("Will Ethereum be above $4,500 on Friday?", "ethereum", "above", 4_500),
    # 'price' used to be a direction word and mapped to 'below'.
    ("Will Bitcoin's price be above $90,000?", "bitcoin", "above", 90_000),
])
def test_takes_first_dollar_amount_after_direction(question, coin, direction, target):
    parsed = parse_crypto_target(question)
    assert parsed == {"coin": coin, "direction": direction,
                      "target_price": pytest.approx(target)}


@pytest.mark.parametrize("question", [
    "Bitcoin Up or Down - September 12",
    "Will BTC ETF inflows exceed $1B?",
    "Will the Lakers win?",
    "",
    None,
])
def test_non_threshold_questions_are_skipped(question):
    assert parse_crypto_target(question) is None
