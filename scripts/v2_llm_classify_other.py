"""V2 — LLM-classify the 'other' bucket into fine-grained sub-categories.

Uses Groq (preferred for speed/throughput) or Gemini as fallback. Sends each
unique 'other' market question to the LLM with a constrained list of
candidate sub-categories and parses the response.

Caches results in data/llm_subcategories.json keyed by market_id so we
never re-classify the same market. Idempotent — safe to re-run.

After classification, prints WR + ROI per LLM sub-category from the
v2 + test1 ledgers — directly answering "which slice of 'other' is
winning vs losing?".
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests

from src.exit_simulator import held_pnl

DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "llm_subcategories.json")
LEDGER_SOURCES = [
    os.path.join(DATA_DIR, "v2_ledger.jsonl"),
    os.path.join(DATA_DIR, "test1_ledger.jsonl"),
]

# Sub-categories the LLM is constrained to choose from
SUB_CATEGORIES = [
    "tennis_match",         # singular tennis match (mostly already in niche_sports, but safety)
    "cricket_match",
    "esports",
    "basketball_game",
    "soccer_game",
    "baseball_game",
    "hockey_game",
    "ufc_fight",
    "boxing",
    "horse_race",
    "esports_prop",         # over/under kills, etc
    "stock_price",          # will TSLA close above $X?
    "crypto_price",         # BTC/ETH price thresholds (some leak past keyword classifier)
    "fed_macro",
    "election_prop",
    "geopolitics_event",
    "celebrity_event",      # marriage, birth, scandal
    "social_media_count",   # tweet counts, follower counts
    "award_show",
    "tv_show_event",
    "book_release",
    "weather_event",
    "scientific_event",
    "legal_indictment",
    "billionaire_action",   # Musk/Bezos/Zuck specific
    "default_other",        # genuinely uncategorisable
]


def load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
    os.replace(tmp, CACHE_FILE)


def call_llm_classify(question: str) -> tuple[str, str]:
    """Return (sub_category, provider). Falls back through providers on failure."""
    prompt = (
        "Classify this prediction market question into EXACTLY ONE of these "
        "sub-categories. Respond with only the sub-category name, nothing else.\n\n"
        f"QUESTION: {question}\n\n"
        f"OPTIONS: {', '.join(SUB_CATEGORIES)}\n\n"
        "Sub-category:"
    )

    # Try Groq first (fast)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 30,
                },
                timeout=15,
            )
            if resp.ok:
                content = resp.json()["choices"][0]["message"]["content"].strip().lower()
                # Extract first valid option
                for sub in SUB_CATEGORIES:
                    if sub in content:
                        return sub, "groq"
        except Exception:
            pass

    # Fall back to Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
                headers={"X-goog-api-key": gemini_key,
                         "Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 30},
                },
                timeout=15,
            )
            if resp.ok:
                cands = resp.json().get("candidates") or []
                if cands and cands[0].get("content"):
                    parts = cands[0]["content"].get("parts") or []
                    if parts:
                        content = (parts[0].get("text") or "").strip().lower()
                        for sub in SUB_CATEGORIES:
                            if sub in content:
                                return sub, "gemini"
        except Exception:
            pass

    return "default_other", "fallback"


def main():
    print("\n" + "=" * 70)
    print("  V2 LLM-CLASSIFIER for 'other' bucket")
    print("=" * 70)

    cache = load_cache()
    print(f"  Cache: {len(cache)} pre-classified markets")

    # Collect all 'other'-bucket records from both ledgers
    other_records = []
    for path in LEDGER_SOURCES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("category") != "other":
                    continue
                if not r.get("question") or not r.get("market_id"):
                    continue
                other_records.append(r)

    # Unique market_ids (cap to keep run quick)
    seen = {}
    for r in other_records:
        mid = r["market_id"]
        if mid not in seen:
            seen[mid] = r
    unique = list(seen.values())
    print(f"  'other'-bucket markets: {len(other_records)} records / "
          f"{len(unique)} unique market_ids")

    # Classify any not in cache
    to_classify = [r for r in unique if str(r["market_id"]) not in cache]
    print(f"  To classify: {len(to_classify)} new")
    if to_classify and not (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("  No GROQ_API_KEY or GEMINI_API_KEY in env — cannot classify.")
        print("  Set one of those and re-run. Aborting.")
        return

    classified = 0
    t0 = time.time()
    for i, r in enumerate(to_classify):
        mid = str(r["market_id"])
        q = (r["question"] or "")[:200]
        sub, provider = call_llm_classify(q)
        cache[mid] = {"sub_category": sub, "provider": provider,
                      "question_snippet": q[:80]}
        classified += 1
        if (i + 1) % 25 == 0 or i == len(to_classify) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(f"    Classified {i + 1}/{len(to_classify)}  "
                  f"({rate:.1f}/s)  last: {sub}")
            save_cache(cache)
        # small jitter to be polite
        time.sleep(0.05)

    save_cache(cache)
    print(f"  Total classified this run: {classified}")
    print(f"  Cache size now: {len(cache)}")

    # Now: aggregate WR + ROI per sub-category
    by_sub = defaultdict(list)
    for r in other_records:
        if not r.get("resolved"):
            continue
        if r.get("action") not in ("BUY_YES", "BUY_NO"):
            continue
        if r.get("outcome") not in (0.0, 1.0):
            continue
        mid = str(r["market_id"])
        if mid not in cache:
            continue
        sub = cache[mid]["sub_category"]
        pnl = held_pnl(r["action"], float(r["market_price"]), float(r["outcome"]))
        by_sub[sub].append({"pnl": pnl, "action": r["action"], "outcome": r["outcome"]})

    print()
    print("  SUB-CATEGORY WR + ROI (from 'other' bucket):")
    print(f"  {'sub_category':<22} {'n':>4} {'wins':>5} {'WR':>6} "
          f"{'total PnL/$':>12} {'avg PnL/$':>11}")
    print("  " + "-" * 70)
    for sub in sorted(by_sub.keys(), key=lambda s: -sum(t["pnl"] for t in by_sub[s])):
        rs = by_sub[sub]
        n = len(rs)
        wins = sum(1 for r in rs if r["pnl"] > 0)
        pnl = sum(r["pnl"] for r in rs)
        avg = pnl / n if n else 0
        verdict = ""
        if n >= 20:
            if pnl > 0 and wins / n > 0.55:
                verdict = "  WINNER"
            elif pnl < 0:
                verdict = "  LOSER"
        print(f"  {sub:<22} {n:>4} {wins:>5} {wins/n*100 if n else 0:>5.1f}% "
              f"{pnl:>+11.2f} {avg:>+10.4f}{verdict}")


if __name__ == "__main__":
    main()
