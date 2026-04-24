"""Measure prospective Brier headroom for three forex-repo transfers.

Does NOT modify the engine. Buckets resolved markets into the subsets each
forex signal would plausibly apply to, then compares current-ensemble Brier
against market-baseline Brier on that subset. If ensemble already matches
market, the signal has zero room to help — integrating it adds noise.

Subsets tested:
  A. econ_calendar        -> Fed / FOMC / CPI / NFP / rate / jobs markets
  B. finbert_sentiment    -> news-reactive markets (elections, geopolitics, policy)
  C. dxy_regime / rvol    -> price-level markets on crypto / SPX / VIX / FX

Usage: python scripts/test_forex_transfer.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rolling_accuracy.json"


def brier(rows, pred_key: str) -> float:
    if not rows:
        return float("nan")
    return sum((r[pred_key] - r["actual"]) ** 2 for r in rows) / len(rows)


def load_resolved() -> list[dict]:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    out = []
    for r in d["results"]:
        if r.get("actual") not in (0.0, 1.0):
            continue
        if r.get("ensemble_pred") is None or r.get("pre_price") is None:
            continue
        out.append(r)
    return out


# Subset A — econ calendar
ECON_RE = re.compile(
    r"\b(fed|fomc|jerome powell|rate cut|rate hike|cpi|inflation|unemployment|nfp|"
    r"nonfarm|payroll|jobs report|interest rate|fed funds|pce|gdp|recession)\b",
    re.IGNORECASE,
)

# Subset B — news-reactive (FinBERT territory)
NEWS_CATS = {"geopolitics", "elections", "tweets", "tech_ai"}
NEWS_RE = re.compile(
    r"\b(election|war|invade|sanction|treaty|summit|tariff|ceasefire|strike|"
    r"resign|impeach|nominee|confirmation|bill|legislation|ruling|verdict)\b",
    re.IGNORECASE,
)

# Subset C — price-level / regime markets (DXY / RVol territory)
PRICE_LEVEL_RE = re.compile(
    r"\b(above|below|reach|hit|close|price of|all[- ]time high|ath|"
    r"s&p|sp500|spx|nasdaq|dow|vix|dxy|dollar index|eur/?usd|usd/?jpy|gbp/?usd)\b"
    r"|\$\d",
    re.IGNORECASE,
)
MICROMOVE_RE = re.compile(r"\bup or down\b", re.IGNORECASE)  # exclude these — 5-min tick markets


def subset_A(rows):
    return [r for r in rows if r.get("category") == "macro" or ECON_RE.search(r["question"])]


def subset_B(rows):
    return [
        r for r in rows
        if r.get("category") in NEWS_CATS or NEWS_RE.search(r["question"])
    ]


def subset_C(rows):
    return [
        r for r in rows
        if PRICE_LEVEL_RE.search(r["question"]) and not MICROMOVE_RE.search(r["question"])
    ]


def report(name: str, subset: list[dict], total: int) -> None:
    n = len(subset)
    pct = 100 * n / total if total else 0
    if n == 0:
        print(f"\n{name}: 0 markets (0.0% of universe) — signal has no addressable surface")
        return
    b_ens = brier(subset, "ensemble_pred")
    b_mkt = brier(subset, "pre_price")
    gap = b_ens - b_mkt
    base_rate = sum(r["actual"] for r in subset) / n
    print(f"\n{name}")
    print(f"  coverage        : {n} / {total} markets ({pct:.1f}%)")
    print(f"  base rate (YES) : {base_rate:.3f}")
    print(f"  Brier ensemble  : {b_ens:.4f}")
    print(f"  Brier market    : {b_mkt:.4f}")
    print(f"  ensemble - mkt  : {gap:+.4f}   (>0 means ensemble worse, signal has headroom)")
    if gap > 0.001:
        print(f"  verdict         : HEADROOM exists - signal *could* help if it closes {gap:.4f}")
    elif abs(gap) <= 0.001:
        print(f"  verdict         : NO HEADROOM - ensemble already ~= market on this subset")
    else:
        print(f"  verdict         : ensemble ALREADY BEATS market by {-gap:.4f}, no transfer needed")


def main() -> None:
    rows = load_resolved()
    total = len(rows)
    b_ens_all = brier(rows, "ensemble_pred")
    b_mkt_all = brier(rows, "pre_price")
    print(f"=== Forex-transfer headroom test ===")
    print(f"universe: {total} resolved markets")
    print(f"Brier ensemble (all): {b_ens_all:.4f}")
    print(f"Brier market   (all): {b_mkt_all:.4f}")
    print(f"overall gap        : {b_ens_all - b_mkt_all:+.4f}")

    report("A. econ_calendar  (Fed/FOMC/CPI/NFP tags)", subset_A(rows), total)
    report("B. finbert_sentiment  (news-reactive markets)", subset_B(rows), total)
    report("C. dxy_regime / rvol  (price-level crypto/index/FX)", subset_C(rows), total)


if __name__ == "__main__":
    main()
