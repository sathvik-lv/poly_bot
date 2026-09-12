"""Pure helpers for the sportsbook-vs-Polymarket divergence study.

Everything here is side-effect free so it can be unit tested; network I/O
lives in scripts/divergence_study.py.

Rule carried over from the 2026-09 audit: a price we cannot observe is None,
never a default like 0.5. Anything that cannot be computed returns None and
the caller drops the record.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable, Optional

DRAW = "Draw"
MIN_NAME_SCORE = 0.75

_DROP_TOKENS = {"fc", "cf", "sc", "afc", "the", "club"}
# Words that mark a player/team prop rather than the plain match winner.
_PROP_WORDS = (" spread", " o/u", "over/under", " total", "handicap", " set ",
               " 1st ", " first ", " half", "quarter", "inning", " map ",
               " game 1", " game 2", "corners", "exact", " method", " round ")
_TITLE_SEP = re.compile(r"\s+(?:vs\.?|v\.?|@)\s+", re.IGNORECASE)
_WILL_WIN = re.compile(r"will\s+(.+?)\s+win\b", re.IGNORECASE)


# ---------------------------------------------------------------- parsing

def to_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_list(value) -> list:
    """Gamma returns list fields as JSON strings ('["Yes", "No"]')."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def parse_ts(value) -> Optional[datetime]:
    """ISO timestamps from both APIs, incl. Gamma's '2026-09-12 11:35:00+00'."""
    if not value:
        return None
    s = str(value).strip().replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "T" in s and re.search(r"[+-]\d{2}$", s):
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------- odds math

def decimal_to_prob(price) -> Optional[float]:
    d = to_float(price)
    if d is None or d <= 1.0:
        return None
    return 1.0 / d


def devig(probs: list) -> Optional[list]:
    """Normalise one bookmaker's implied probabilities to sum to 1."""
    if not probs or any(p is None or p <= 0 for p in probs):
        return None
    total = sum(probs)
    return [p / total for p in probs]


def book_consensus(event: dict, market_key: str = "h2h") -> dict:
    """Median de-vigged probability per outcome across bookmakers.

    Books whose outcome set differs from the majority are dropped so 2-way and
    3-way (draw) lines never get blended.
    """
    per_book = []
    for bm in event.get("bookmakers") or []:
        for mk in bm.get("markets") or []:
            if mk.get("key") != market_key:
                continue
            outs = mk.get("outcomes") or []
            names = [o.get("name") for o in outs]
            if len(outs) < 2 or any(not n for n in names):
                continue
            fair = devig([decimal_to_prob(o.get("price")) for o in outs])
            if fair is None:
                continue
            per_book.append((bm.get("key"), dict(zip(names, fair))))

    empty = {"probs": {}, "n_books": 0, "min": {}, "max": {}, "pinnacle": None}
    if not per_book:
        return empty
    majority = Counter(frozenset(d) for _, d in per_book).most_common(1)[0][0]
    kept = [(k, d) for k, d in per_book if frozenset(d) == majority]

    medians, lo, hi = {}, {}, {}
    for name in majority:
        vals = sorted(d[name] for _, d in kept)
        medians[name] = statistics.median(vals)
        lo[name], hi[name] = vals[0], vals[-1]
    total = sum(medians.values())
    probs = {n: p / total for n, p in medians.items()}
    pinnacle = next((d for k, d in kept if k == "pinnacle"), None)
    return {"probs": probs, "n_books": len(kept), "min": lo, "max": hi,
            "pinnacle": pinnacle}


# ---------------------------------------------------------------- names

def norm_name(value) -> str:
    s = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _tokens(value) -> list:
    return [t for t in norm_name(value).split() if t not in _DROP_TOKENS]


def name_score(a, b) -> float:
    """1.0 identical, 0.9 when one name's tokens contain the other's
    ('Army' / 'Army Black Knights'), else fuzzy ratio."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    sa, sb = set(ta), set(tb)
    if sa <= sb or sb <= sa:
        return 0.9
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def pair_score(home, away, a, b) -> tuple:
    """Match (home, away) against (a, b) in whichever orientation fits.

    Score is the WEAKER of the two name matches — both sides must agree.
    Returns (score, swapped) where swapped means home pairs with b.
    """
    straight = min(name_score(home, a), name_score(away, b))
    swapped = min(name_score(home, b), name_score(away, a))
    return (straight, False) if straight >= swapped else (swapped, True)


def split_title(title) -> Optional[tuple]:
    """'Arizona State vs. Texas A&M' -> ('Arizona State', 'Texas A&M').

    Gamma sub-events ('... - Exact Score') are rejected; an event-name prefix
    ('Noche UFC: A vs. B') and trailing parentheses are stripped.
    """
    t = str(title or "").strip()
    if not t or " - " in t:
        return None
    if ":" in t:
        t = t.split(":", 1)[1].strip()
    t = re.sub(r"\s*\(.*?\)\s*$", "", t)
    parts = _TITLE_SEP.split(t, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    return (a, b) if a and b else None


# ---------------------------------------------------------------- markets

def market_shape(market: dict) -> Optional[str]:
    """'h2h' for a two-competitor match-winner market, 'yesno' for a
    'Will X win?' / 'end in a draw?' moneyline leg, None for anything else."""
    outs = [str(o).strip() for o in parse_list(market.get("outcomes"))]
    if len(outs) != 2:
        return None
    low = {o.lower() for o in outs}
    smt = market.get("sportsMarketType")
    if low == {"yes", "no"}:
        return "yesno" if smt == "moneyline" else None
    if low & {"yes", "no", "over", "under", "draw"}:
        return None
    if smt not in (None, "moneyline"):
        return None
    q = " " + str(market.get("question") or "").lower() + " "
    if any(w in q for w in _PROP_WORDS):
        return None
    return "h2h"


def target_for_market(market: dict, shape: str, home, away,
                      parent_title=None) -> Optional[str]:
    """The sportsbook outcome name that the market's FIRST outcome refers to.

    Gamma's bestBid/bestAsk quote outcome[0], so orienting on it lets every
    record use the quote directly.
    """
    outs = [str(o) for o in parse_list(market.get("outcomes"))]
    if shape == "h2h":
        if len(outs) != 2:
            return None
        score, swapped = pair_score(home, away, outs[0], outs[1])
        if score < MIN_NAME_SCORE:
            return None
        return away if swapped else home
    if shape == "yesno":
        teams = split_title(parent_title)
        if not teams:
            return None
        score, _ = pair_score(home, away, teams[0], teams[1])
        if score < MIN_NAME_SCORE:
            return None
        q = str(market.get("question") or "")
        if re.search(r"\bdraw\b", q, re.IGNORECASE):
            return DRAW
        m = _WILL_WIN.search(q)
        if not m:
            return None
        sh, sa = name_score(m.group(1), home), name_score(m.group(1), away)
        if max(sh, sa) < MIN_NAME_SCORE or sh == sa:
            return None
        return home if sh > sa else away
    return None


def market_start(market: dict, parent: Optional[dict]) -> Optional[datetime]:
    return (parse_ts(market.get("gameStartTime"))
            or parse_ts((parent or {}).get("startTime"))
            or parse_ts(market.get("endDate")))


def match_markets(book_events: Iterable, poly_markets: Iterable,
                  tolerance_hours: float = 36.0) -> list:
    """Pair Polymarket game markets with sportsbook events.

    book_events:  (sport_key, odds_api_event)
    poly_markets: (gamma_market, gamma_parent_event, shape)
    Each Polymarket market is paired at most once — the first qualifying book
    event wins, so a game listed under two sport keys is not double counted.
    """
    indexed = []
    for market, parent, shape in poly_markets:
        if shape == "h2h":
            names = [str(o) for o in parse_list(market.get("outcomes"))]
        else:
            teams = split_title((parent or {}).get("title"))
            if not teams:
                continue
            names = list(teams)
        toks = set()
        for n in names:
            toks |= set(_tokens(n))
        indexed.append((market, parent, shape, toks, market_start(market, parent)))

    out, used = [], set()
    for sport, ev in book_events:
        home, away = ev.get("home_team"), ev.get("away_team")
        commence = parse_ts(ev.get("commence_time"))
        if not home or not away or commence is None:
            continue
        btoks = set(_tokens(home)) | set(_tokens(away))
        for market, parent, shape, toks, start in indexed:
            mid = str(market.get("id"))
            if mid in used or not (btoks & toks):
                continue
            if start is None or abs((start - commence).total_seconds()) > tolerance_hours * 3600:
                continue
            target = target_for_market(market, shape, home, away, (parent or {}).get("title"))
            if target is None:
                continue
            used.add(mid)
            out.append({"sport": sport, "event": ev, "market": market,
                        "parent": parent, "shape": shape, "target": target})
    return out


def side_prices(market: dict) -> Optional[dict]:
    """Executable quote for outcome[0]. None — never 0.5 — if the book is empty
    or crossed."""
    bid, ask = to_float(market.get("bestBid")), to_float(market.get("bestAsk"))
    if bid is None or ask is None or not (0.0 < bid < ask < 1.0):
        return None
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "spread": ask - bid}


def edges(book_p: float, px: dict) -> dict:
    """Divergence at the mid, and the edge you could actually take after
    crossing the spread. 'against' means buying the other side at 1 - bid."""
    edge_buy = book_p - px["ask"]
    edge_sell = px["bid"] - book_p
    if edge_buy >= edge_sell:
        side, best, entry = "target", edge_buy, px["ask"]
    else:
        side, best, entry = "against", edge_sell, 1.0 - px["bid"]
    return {"div_mid": book_p - px["mid"], "edge_buy": edge_buy,
            "edge_sell": edge_sell, "best_edge": best, "best_side": side,
            "entry_price": entry}


def per_dollar(side: str, entry_price: float, target_won) -> float:
    if side not in ("target", "against"):
        raise ValueError(f"unknown side {side!r}")
    won = bool(target_won) if side == "target" else not bool(target_won)
    return (1.0 / entry_price - 1.0) if won else -1.0


def resolution_from_market(market: dict):
    """True / False if outcome[0] settled YES / NO, 'void' if it settled
    indecisively, None while still undecided."""
    prices = [to_float(p) for p in parse_list(market.get("outcomePrices"))]
    if not prices or prices[0] is None or not market.get("closed"):
        return None
    p0 = prices[0]
    if p0 >= 0.99:
        return True
    if p0 <= 0.01:
        return False
    if str(market.get("umaResolutionStatus") or "").lower() == "resolved":
        return "void"
    return None


# ---------------------------------------------------------------- budget

def credits_for_poll(remaining, now: datetime, reserve: int = 25,
                     poll_hours: float = 3.0, cap: int = 20) -> int:
    """Odds API credits this poll may spend.

    Spreads what is left above `reserve` evenly over the polls remaining until
    the quota resets, assumed to be the 1st of next UTC month. If it really
    resets earlier we merely under-spend.
    """
    if remaining is None:
        return 0
    spendable = int(remaining) - reserve
    if spendable <= 0:
        return 0
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    reset = datetime(year, month, 1, tzinfo=timezone.utc)
    polls_left = max((reset - now).total_seconds() / 3600.0 / poll_hours, 1.0)
    return max(1, min(cap, int(spendable / polls_left)))


def key_priority(hours_to_start: Iterable) -> float:
    """Weight a sport key by how soon its matched games start: the line just
    before kick-off is the sharpest reference, so imminent games count most."""
    w = 0.0
    for h in hours_to_start:
        if h is None or h < 0:
            continue
        w += 1.0 if h <= 12 else 0.5 if h <= 36 else 0.2
    return w


# ---------------------------------------------------------------- statistics

def summarize(values: Iterable) -> dict:
    xs = [float(x) for x in values]
    out = {"n": len(xs), "mean": None, "sd": None, "se": None,
           "t": None, "lo": None, "hi": None}
    if not xs:
        return out
    mean = statistics.mean(xs)
    out["mean"] = mean
    if len(xs) < 2:
        return out
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(len(xs))
    out.update(sd=sd, se=se, lo=mean - 1.96 * se, hi=mean + 1.96 * se,
               t=(mean / se) if se > 0 else None)
    return out


def cluster_summarize(pairs: Iterable) -> dict:
    """Summarise per-cluster means, so correlated rows (the three legs of one
    soccer match) count as one observation."""
    groups = defaultdict(list)
    n_obs = 0
    for key, value in pairs:
        groups[key].append(float(value))
        n_obs += 1
    out = summarize(statistics.mean(v) for v in groups.values())
    out["n_obs"] = n_obs
    return out


def mde(sd, n, z_alpha: float = 1.96, z_power: float = 0.84) -> Optional[float]:
    """Smallest mean effect detectable at 5% two-sided with 80% power."""
    if sd is None or n is None or n < 2:
        return None
    return (z_alpha + z_power) * sd / math.sqrt(n)


def closing_snapshots(records: Iterable) -> list:
    """Last pre-start snapshot per Polymarket market (the closing line)."""
    best = {}
    for r in records:
        h = r.get("h_to_start")
        if h is None or h < 0:
            continue
        key = str(r.get("poly_market_id"))
        if key not in best or str(r.get("ts")) > str(best[key].get("ts")):
            best[key] = r
    return list(best.values())


def _quantile(sorted_xs: list, q: float) -> Optional[float]:
    if not sorted_xs:
        return None
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def analyze(records: list, thresholds=(0.0, 0.01, 0.02, 0.03, 0.05),
            headline: float = 0.02, min_resolved: int = 30,
            entry_band=(0.05, 0.95)) -> dict:
    """The study's verdict.

    Two questions, both on the closing snapshot of each resolved market and
    clustered per game:
      1. Is the book consensus better calibrated than Polymarket's mid?
         (paired Brier; negative = book sharper)
      2. Does betting the book's side at Polymarket's EXECUTABLE price, when
         the edge after the spread clears a threshold, make money?
    Entry prices outside `entry_band` are excluded so a handful of long-shot
    payoffs cannot dominate the mean.
    """
    absdiv = sorted(abs(r["div_mid"]) for r in records if r.get("div_mid") is not None)
    exec_edges = [r["best_edge"] for r in records if r.get("best_edge") is not None]
    closing = closing_snapshots(records)
    resolved = [r for r in closing if r.get("resolved") and not r.get("void")
                and r.get("target_won") is not None]

    brier_pairs = []
    for r in resolved:
        y = 1.0 if r["target_won"] else 0.0
        brier_pairs.append((str(r.get("book_event_id")),
                            (r["book_p"] - y) ** 2 - (r["mid"] - y) ** 2))

    strategy = {}
    for thr in thresholds:
        rows = [r for r in resolved
                if r.get("best_edge") is not None and r["best_edge"] >= thr
                and entry_band[0] <= r["entry_price"] <= entry_band[1]]
        pnl = [(str(r.get("book_event_id")),
                per_dollar(r["best_side"], r["entry_price"], r["target_won"])) for r in rows]
        clustered = cluster_summarize(pnl)
        strategy[f"{thr:.2f}"] = {
            "naive": summarize(v for _, v in pnl),
            "clustered": clustered,
            "win_rate": statistics.mean(1.0 if v > 0 else 0.0 for _, v in pnl) if pnl else None,
            "implied_win_rate": statistics.mean(r["entry_price"] for r in rows) if rows else None,
            "mde": mde(clustered["sd"], clustered["n"]),
        }

    head = strategy[f"{headline:.2f}"]["clustered"]
    if len(resolved) < min_resolved or head["lo"] is None:
        verdict = "INSUFFICIENT_DATA"
    elif head["lo"] > 0:
        verdict = "EDGE"
    elif head["hi"] < 0:
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n_records": len(records),
        "n_markets": len({str(r.get("poly_market_id")) for r in records}),
        "n_events": len({str(r.get("book_event_id")) for r in records}),
        "n_polls": len({r.get("ts") for r in records}),
        "n_resolved": len(resolved),
        "divergence": {
            "median_abs_mid": _quantile(absdiv, 0.5),
            "p90_abs_mid": _quantile(absdiv, 0.9),
            "share_abs_mid_gt_2pp": (sum(d > 0.02 for d in absdiv) / len(absdiv)) if absdiv else None,
            "share_abs_mid_gt_5pp": (sum(d > 0.05 for d in absdiv) / len(absdiv)) if absdiv else None,
            "share_exec_edge_gt_2pp": (sum(e > 0.02 for e in exec_edges) / len(exec_edges)) if exec_edges else None,
        },
        "brier": cluster_summarize(brier_pairs),
        "strategy": strategy,
        "headline_threshold": headline,
        "verdict": verdict,
    }
