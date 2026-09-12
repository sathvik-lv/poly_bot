"""Divergence study: sportsbook consensus vs Polymarket on the same games.

Thesis under test (stated in niche_scanner's docstring, never actually run):
sportsbooks move far more money per game than Polymarket's books, so they
should be sharper. If so, where Polymarket's price diverges from the book
consensus, the result should go the book's way often enough to beat
Polymarket's executable price.

OBSERVATION ONLY. Nothing here places or simulates live orders.

    python scripts/divergence_study.py collect   # log book vs Polymarket pairs
    python scripts/divergence_study.py resolve   # fill in results
    python scripts/divergence_study.py report    # the verdict

Odds API budget: /sports and /events are free; /odds costs 1 credit per sport
key per region and returns every game in that key. The collector polls at most
every DIVERGENCE_POLL_HOURS and paces credits so the free 500/month lasts
until the quota resets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

from src.divergence import (
    analyze, book_consensus, credits_for_poll, edges, key_priority,
    market_shape, match_markets, parse_ts, resolution_from_market,
    side_prices, to_float,
)

load_dotenv()

ODDS = "https://api.the-odds-api.com/v4"
GAMMA = "https://gamma-api.polymarket.com"
DATA_DIR = os.environ.get("DIVERGENCE_DATA_DIR", "data")
LOG_FILE = os.path.join(DATA_DIR, "divergence_log.jsonl")
STATE_FILE = os.path.join(DATA_DIR, "divergence_state.json")
REPORT_FILE = os.path.join(DATA_DIR, "divergence_report.json")

POLL_HOURS = float(os.environ.get("DIVERGENCE_POLL_HOURS", "3"))
WINDOW_HOURS = 48          # only games starting within this window are logged
RESERVE_CREDITS = 25       # never spend below this
GAMES_TAG = 100639         # Gamma's shared tag for every sports game market
GAMMA_PAGE = 100           # Gamma caps page size at 100
GAMMA_MAX_OFFSET = 2000    # Gamma rejects offsets beyond ~2000 (HTTP 422)
RESOLVE_AFTER_HOURS = 3
MAX_RESOLVE_PER_RUN = 400
# Sport keys better covered by US books; everything else uses the EU region,
# which includes Pinnacle and Betfair.
US_REGION_PREFIXES = ("americanfootball_", "baseball_mlb", "basketball_nba",
                      "basketball_wnba", "basketball_ncaa", "icehockey_nhl")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def warn(msg: str) -> None:
    # '::warning::' surfaces as an annotation in the GitHub Actions UI.
    print(f"::warning::divergence_study: {msg}")


# ---------------------------------------------------------------- storage

def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    _atomic_write(STATE_FILE, json.dumps(state, indent=2, default=str))


def load_records() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    out = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_records(records: list) -> None:
    if not records:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def rewrite_records(records: list) -> None:
    _atomic_write(LOG_FILE, "".join(json.dumps(r, default=str) + "\n" for r in records))


# ---------------------------------------------------------------- fetching

def odds_get(path: str, key: str, **params):
    """Returns (json, remaining_credits, last_call_cost). json is None on any
    failure, and failures are reported loudly rather than swallowed."""
    try:
        r = requests.get(f"{ODDS}{path}", params={"apiKey": key, **params}, timeout=25)
    except requests.RequestException as e:
        warn(f"Odds API {path}: {type(e).__name__}: {e}")
        return None, None, None
    remaining = to_float(r.headers.get("x-requests-remaining"))
    cost = to_float(r.headers.get("x-requests-last"))
    if r.status_code != 200:
        warn(f"Odds API {path} HTTP {r.status_code}: {r.text[:200]}")
        return None, remaining, cost
    return r.json(), remaining, cost


def fetch_book_events(key: str, sport_keys: list, now: datetime) -> list:
    """(sport_key, event) for games starting within the window. Free calls."""
    horizon = now + timedelta(hours=WINDOW_HOURS)

    def one(sport):
        data, _, _ = odds_get(f"/sports/{sport}/events", key)
        return sport, data or []

    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for sport, events in pool.map(one, sport_keys):
            for ev in events:
                c = parse_ts(ev.get("commence_time"))
                if c is not None and now < c <= horizon:
                    out.append((sport, ev))
    return out


def fetch_gamma_markets(now: datetime) -> list:
    """(market, parent_event, shape) for open Polymarket match-winner markets."""
    base = {"tag_id": GAMES_TAG, "closed": "false", "limit": GAMMA_PAGE,
            "end_date_min": now.isoformat(),
            "end_date_max": (now + timedelta(hours=WINDOW_HOURS + 72)).isoformat(),
            "order": "endDate", "ascending": "true"}
    events, offset = [], 0
    while offset <= GAMMA_MAX_OFFSET:
        try:
            r = requests.get(f"{GAMMA}/events", params={**base, "offset": offset}, timeout=30)
        except requests.RequestException as e:
            warn(f"Gamma /events: {type(e).__name__}: {e}")
            break
        if r.status_code != 200:
            if offset == 0 and "order" in base:
                base.pop("order")
                base.pop("ascending")
                continue
            break
        page = r.json()
        events.extend(page)
        if len(page) < GAMMA_PAGE:
            break
        offset += GAMMA_PAGE

    out = []
    for ev in events:
        for m in ev.get("markets") or []:
            if m.get("closed") or m.get("active") is False:
                continue
            shape = market_shape(m)
            if shape:
                out.append((m, ev, shape))
    return out


# ---------------------------------------------------------------- collect

def collect(args) -> int:
    state = load_state()
    now = now_utc()
    last = parse_ts(state.get("last_poll"))
    if last and not args.force and now - last < timedelta(hours=POLL_HOURS):
        hrs = (now - last).total_seconds() / 3600
        print(f"  divergence collect: last poll {hrs:.1f}h ago (< {POLL_HOURS}h) — skipping, no API calls")
        return 0

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        warn("ODDS_API_KEY is not set — study cannot collect")
        return 1

    sports, remaining, _ = odds_get("/sports", key)
    if sports is None:
        return 1
    sport_keys = [s["key"] for s in sports if s.get("active") and not s.get("has_outrights")]

    book = fetch_book_events(key, sport_keys, now)
    poly = fetch_gamma_markets(now)
    matches = match_markets(book, poly)
    for mt in matches:
        c = parse_ts(mt["event"].get("commence_time"))
        mt["h_to_start"] = (c - now).total_seconds() / 3600 if c else None

    by_key = defaultdict(list)
    for mt in matches:
        by_key[mt["sport"]].append(mt)
    ranked = sorted(by_key, key=lambda k: (key_priority(m["h_to_start"] for m in by_key[k]),
                                           len(by_key[k])), reverse=True)
    allowance = (args.max_credits if args.max_credits is not None
                 else credits_for_poll(remaining, now, RESERVE_CREDITS, POLL_HOURS))
    chosen = ranked[:allowance]

    print(f"  book events in next {WINDOW_HOURS}h: {len(book)} across {len({s for s, _ in book})} sport keys")
    print(f"  Polymarket match-winner markets: {len(poly)}")
    print(f"  matched markets: {len(matches)} across {len(by_key)} sport keys")
    print(f"  credits remaining: {remaining}   allowance this poll: {allowance}")
    for k in ranked[:12]:
        mark = "BUY " if k in chosen else "    "
        print(f"    {mark}{k:<40} matched={len(by_key[k]):<4} "
              f"priority={key_priority(m['h_to_start'] for m in by_key[k]):.1f}")
    if args.dry_run:
        print("  dry run — no /odds calls, nothing logged")
        return 0

    records, skipped, spent = [], Counter(), 0
    ts = now.isoformat()
    for sport in chosen:
        region = "us" if sport.startswith(US_REGION_PREFIXES) else "eu"
        odds, rem, cost = odds_get(f"/sports/{sport}/odds", key, regions=region,
                                   markets="h2h", oddsFormat="decimal")
        if rem is not None:
            remaining = rem
        spent += int(cost or 0)
        if odds is None:
            skipped["odds_error"] += len(by_key[sport])
            continue
        by_id = {e.get("id"): e for e in odds}
        for mt in by_key[sport]:
            ev = by_id.get(mt["event"].get("id"))
            if ev is None:
                skipped["no_book_line"] += 1
                continue
            cons = book_consensus(ev)
            if cons["n_books"] < 2:
                skipped["fewer_than_2_books"] += 1
                continue
            book_p = cons["probs"].get(mt["target"])
            if book_p is None:
                skipped["target_not_priced_by_books"] += 1
                continue
            px = side_prices(mt["market"])
            if px is None:
                skipped["no_polymarket_quote"] += 1
                continue
            m = mt["market"]
            pin = cons["pinnacle"] or {}
            records.append({
                "ts": ts, "sport": sport, "region": region,
                "book_event_id": ev.get("id"), "commence": ev.get("commence_time"),
                "h_to_start": round(mt["h_to_start"], 3),
                "home": ev.get("home_team"), "away": ev.get("away_team"),
                "poly_market_id": str(m.get("id")), "question": m.get("question"),
                "shape": mt["shape"], "target": mt["target"],
                "book_p": book_p, "book_n": cons["n_books"],
                "book_min": cons["min"].get(mt["target"]),
                "book_max": cons["max"].get(mt["target"]),
                "pinnacle_p": pin.get(mt["target"]),
                **px,
                "liquidity": to_float(m.get("liquidityNum")),
                "volume24h": to_float(m.get("volume24hr")),
                **edges(book_p, px),
                "resolved": False, "target_won": None, "void": False,
            })

    append_records(records)
    history = state.get("history", [])[-199:]
    history.append({"ts": ts, "credits_spent": spent, "logged": len(records),
                    "remaining": remaining, "keys": chosen})
    state.update(last_poll=ts, polls=state.get("polls", 0) + 1,
                 credits_spent_total=state.get("credits_spent_total", 0) + spent,
                 last_remaining=remaining, history=history)
    save_state(state)

    print(f"\n  logged {len(records)} pairs, spent {spent} credits, {remaining} left")
    if skipped:
        print("  skipped:", dict(skipped))
    if records:
        print(f"\n  {'sport':<28} {'target':<24} {'book':>6} {'bid':>6} {'ask':>6} {'div':>7} {'exec':>7}")
        for r in sorted(records, key=lambda r: -abs(r["div_mid"]))[:15]:
            print(f"  {r['sport'][:28]:<28} {str(r['target'])[:24]:<24} {r['book_p']:>6.3f} "
                  f"{r['bid']:>6.3f} {r['ask']:>6.3f} {r['div_mid']:>+7.3f} {r['best_edge']:>+7.3f}")
    return 0


# ---------------------------------------------------------------- resolve

def resolve(args) -> int:
    records = load_records()
    now = now_utc()
    pending = []
    for r in records:
        if r.get("resolved") or r.get("void"):
            continue
        c = parse_ts(r.get("commence"))
        if c and now - c > timedelta(hours=RESOLVE_AFTER_HOURS):
            pending.append(str(r["poly_market_id"]))
    ids = list(dict.fromkeys(pending))[:MAX_RESOLVE_PER_RUN]
    if not ids:
        print("  divergence resolve: nothing due")
        return 0

    def one(mid):
        try:
            r = requests.get(f"{GAMMA}/markets/{mid}", timeout=20)
        except requests.RequestException:
            return mid, None
        return mid, (resolution_from_market(r.json()) if r.status_code == 200 else None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(one, ids))

    changed = Counter()
    for r in records:
        res = results.get(str(r.get("poly_market_id")))
        if r.get("resolved") or res is None:
            continue
        if res == "void":
            r["void"] = True
            changed["void"] += 1
        else:
            r["resolved"], r["target_won"] = True, bool(res)
            changed["resolved"] += 1
    if changed:
        rewrite_records(records)
    print(f"  divergence resolve: checked {len(ids)} markets, updated rows {dict(changed)}")
    return 0


# ---------------------------------------------------------------- report

def _pct(x):
    return "   n/a" if x is None else f"{x * 100:+6.2f}%"


def report(args) -> int:
    records = load_records()
    a = analyze(records)
    _atomic_write(REPORT_FILE, json.dumps({"generated": now_utc().isoformat(), **a},
                                          indent=2, default=str))
    print("=" * 96)
    print("  DIVERGENCE STUDY — sportsbook consensus vs Polymarket (observation only)")
    print("=" * 96)
    print(f"  pairs logged {a['n_records']} | markets {a['n_markets']} | games {a['n_events']} "
          f"| polls {a['n_polls']} | resolved markets {a['n_resolved']}")
    d = a["divergence"]
    if d["median_abs_mid"] is not None:
        print(f"  |divergence at mid|: median {d['median_abs_mid']:.3f}  p90 {d['p90_abs_mid']:.3f}  "
              f">2pp {d['share_abs_mid_gt_2pp']:.1%}  >5pp {d['share_abs_mid_gt_5pp']:.1%}  "
              f"| executable edge >2pp: {d['share_exec_edge_gt_2pp']:.1%}")
    b = a["brier"]
    if b["mean"] is not None:
        ci = "" if b["lo"] is None else f"  95% CI [{b['lo']:+.4f}, {b['hi']:+.4f}]"
        print(f"  Brier book - Polymarket (per game): {b['mean']:+.4f}{ci}  n={b['n']}  "
              "(negative = books sharper)")
    print(f"\n  {'min edge':>8} {'games':>6} {'bets':>5} {'mean/$':>8} {'95% CI':>20} "
          f"{'win':>6} {'implied':>7} {'MDE':>7}")
    for thr, s in a["strategy"].items():
        c = s["clustered"]
        ci = "" if c["lo"] is None else f"[{c['lo'] * 100:+6.1f}%,{c['hi'] * 100:+6.1f}%]"
        win = "   n/a" if s["win_rate"] is None else f"{s['win_rate']:6.1%}"
        imp = "    n/a" if s["implied_win_rate"] is None else f"{s['implied_win_rate']:7.1%}"
        mde_ = "    n/a" if s["mde"] is None else f"{s['mde'] * 100:6.1f}%"
        print(f"  {thr:>8} {c['n']:>6} {c['n_obs']:>5} {_pct(c['mean']):>8} {ci:>20} {win} {imp} {mde_}")
    print(f"\n  VERDICT (edge >= {a['headline_threshold']:.2f}, clustered per game): {a['verdict']}")
    print(f"  saved {REPORT_FILE}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--force", action="store_true", help="ignore the poll interval")
    c.add_argument("--dry-run", action="store_true", help="discover and match only; spend no credits")
    c.add_argument("--max-credits", type=int, default=None, help="override the paced allowance")
    sub.add_parser("resolve")
    sub.add_parser("report")
    args = p.parse_args()
    return {"collect": collect, "resolve": resolve, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
