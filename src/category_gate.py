"""Category-aware entry gate. v2 only.

Reads category-level resolved performance from data/v2_ledger.jsonl (or
test1_ledger as a bootstrap source) and decides per-trade whether to allow
entry. Categories with too few trades fall back to the global default.

Decision rule per category:
- n < min_trades_for_decision → use global edge threshold (no special rule)
- WR clearly < 0.50 → BLOCK entries entirely
- WR ambiguous (CI overlaps 0.50) → require larger absolute edge
- WR clearly > 0.50 → use global threshold (or slightly relaxed)

Conservatism: when no data file exists, behave identically to v1 — every
category is allowed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass


DATA_DIR = "data"


@dataclass
class CategoryStats:
    name: str
    n: int
    wins: int
    wr: float
    wr_ci_low: float
    wr_ci_high: float
    pnl_per_trade: float


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Better than normal approx for small n."""
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_by_category(ledger_path: str) -> dict[str, CategoryStats]:
    """Walk a ledger jsonl and bucket resolved bets by category."""
    if not os.path.exists(ledger_path):
        return {}

    by_cat: dict[str, dict] = {}
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("resolved"):
                continue
            action = r.get("action")
            if action not in ("BUY_YES", "BUY_NO"):
                continue
            cat = r.get("category", "other") or "other"
            outcome = r.get("outcome")
            if outcome is None:
                continue
            price = r.get("market_price")
            if price is None:
                continue

            # Per-dollar P&L for this bet
            if action == "BUY_YES":
                pnl = (1 / price - 1) if outcome == 1.0 else -1.0
            else:
                pnl = (1 / (1 - price) - 1) if outcome == 0.0 else -1.0

            won = pnl > 0
            d = by_cat.setdefault(cat, {"n": 0, "wins": 0, "pnl_sum": 0.0})
            d["n"] += 1
            if won:
                d["wins"] += 1
            d["pnl_sum"] += pnl

    out: dict[str, CategoryStats] = {}
    for cat, d in by_cat.items():
        n, w = d["n"], d["wins"]
        wr = w / n if n else 0.0
        lo, hi = _wilson_ci(w, n)
        out[cat] = CategoryStats(
            name=cat, n=n, wins=w, wr=wr,
            wr_ci_low=lo, wr_ci_high=hi,
            pnl_per_trade=d["pnl_sum"] / n if n else 0.0,
        )
    return out


class CategoryGate:
    """Per-category gate decisions.

    Args:
        ledger_paths: list of historical ledger files to learn from
        min_trades_for_decision: below this n, treat category as unknown
        block_threshold: WR upper-CI <= this → block category
        ambiguous_threshold: WR upper-CI > this → allow normal threshold;
            otherwise require larger edge to enter
        ambiguous_edge_required: required abs edge for ambiguous categories
    """

    def __init__(
        self,
        ledger_paths: list[str] | None = None,
        min_trades_for_decision: int = 30,
        block_threshold: float = 0.48,
        ambiguous_threshold: float = 0.55,
        ambiguous_edge_required: float = 0.05,
    ):
        if ledger_paths is None:
            ledger_paths = [
                os.path.join(DATA_DIR, "v2_ledger.jsonl"),
                os.path.join(DATA_DIR, "test1_ledger.jsonl"),
            ]
        self.min_trades_for_decision = min_trades_for_decision
        self.block_threshold = block_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.ambiguous_edge_required = ambiguous_edge_required

        merged: dict[str, CategoryStats] = {}
        for p in ledger_paths:
            for cat, s in aggregate_by_category(p).items():
                if cat in merged:
                    # Combine
                    n = merged[cat].n + s.n
                    wins = merged[cat].wins + s.wins
                    pnl_avg = (merged[cat].pnl_per_trade * merged[cat].n + s.pnl_per_trade * s.n) / max(n, 1)
                    lo, hi = _wilson_ci(wins, n)
                    merged[cat] = CategoryStats(cat, n, wins, wins / n if n else 0.0, lo, hi, pnl_avg)
                else:
                    merged[cat] = s
        self.stats = merged

    def decide(self, category: str, abs_edge: float) -> dict:
        """Return decision for a candidate trade.

        Returns:
            {
              "allow": bool,
              "reason": str,
              "category_n": int,
              "category_wr": float | None,
              "required_edge": float,
            }
        """
        cat = category or "other"
        s = self.stats.get(cat)

        # No data or below threshold → trust caller (default policy applies)
        if s is None or s.n < self.min_trades_for_decision:
            return {
                "allow": True,
                "reason": f"insufficient_data (n={s.n if s else 0})",
                "category_n": s.n if s else 0,
                "category_wr": s.wr if s else None,
                "required_edge": 0.0,
            }

        # Clearly losing category → block
        if s.wr_ci_high <= self.block_threshold:
            return {
                "allow": False,
                "reason": f"category_blocked (WR CI top {s.wr_ci_high:.2f} <= {self.block_threshold})",
                "category_n": s.n,
                "category_wr": s.wr,
                "required_edge": float("inf"),
            }

        # Ambiguous (CI straddles 50%) → demand larger edge
        if s.wr_ci_high <= self.ambiguous_threshold:
            req = self.ambiguous_edge_required
            allow = abs_edge >= req
            return {
                "allow": allow,
                "reason": (
                    f"category_ambiguous (WR={s.wr:.2f}, CI=[{s.wr_ci_low:.2f},{s.wr_ci_high:.2f}]) "
                    f"need edge >= {req}"
                ),
                "category_n": s.n,
                "category_wr": s.wr,
                "required_edge": req,
            }

        # Clearly winning category → trust caller's default threshold
        return {
            "allow": True,
            "reason": f"category_strong (WR={s.wr:.2f})",
            "category_n": s.n,
            "category_wr": s.wr,
            "required_edge": 0.0,
        }

    def report(self) -> str:
        if not self.stats:
            return "  (no category data yet)"
        lines = []
        lines.append(f"  {'Category':<14} {'n':>5} {'WR':>6} {'CI_low':>7} {'CI_high':>8}  Decision")
        lines.append("  " + "-" * 70)
        for cat in sorted(self.stats.keys()):
            s = self.stats[cat]
            d = self.decide(cat, abs_edge=0.03)  # using default threshold to characterize
            verdict = "BLOCK" if not d["allow"] and d["required_edge"] == float("inf") \
                else "AMBIG" if d["required_edge"] > 0 else "ALLOW"
            lines.append(
                f"  {s.name:<14} {s.n:>5} {s.wr:>6.2f} {s.wr_ci_low:>7.2f} {s.wr_ci_high:>8.2f}  {verdict}"
            )
        return "\n".join(lines)
