"""Pure functions for early-exit-on-edge-decay simulation.

Used by both:
  - scripts/v2_exit_sim.py  (whole-ledger backtest, no live impact)
  - scripts/paper_trader.py (live decision per open position when
    ENABLE_EARLY_EXIT=1 is set)

Keeping the math here lets both code paths use exactly the same rule.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional


def per_dollar_pnl(action: str, entry_price: float, exit_price: float) -> float:
    """Per-dollar P&L if a position is closed at exit_price.

    For BUY_YES: bought at entry_price, get back exit_price/entry_price - 1
        (or 1/entry_price - 1 if exit_price=1, the resolution YES payoff)
    For BUY_NO: bought at (1 - entry_price), get back (1 - exit_price)/(1 - entry_price) - 1
    """
    if action == "BUY_YES":
        if entry_price <= 0:
            return 0.0
        return exit_price / entry_price - 1.0
    if action == "BUY_NO":
        if entry_price >= 1:
            return 0.0
        return (1.0 - exit_price) / (1.0 - entry_price) - 1.0
    return 0.0


def held_pnl(action: str, entry_price: float, outcome: float) -> float:
    """P&L if held to resolution. outcome is 0.0 or 1.0."""
    return per_dollar_pnl(action, entry_price, exit_price=outcome)


def should_exit(entry_edge: float, current_edge: float, decay_threshold: float) -> bool:
    """True if absolute edge has decayed by >= decay_threshold fraction.

    decay_threshold=0.5 means: exit if abs(current_edge) < 0.5 * abs(entry_edge).
    Edge sign flip (current_edge has opposite sign of entry_edge) always counts
    as full decay → exit.
    """
    if entry_edge == 0:
        return False  # nothing to decay from
    if (entry_edge > 0) != (current_edge > 0) and current_edge != 0:
        return True  # edge flipped sign → exit
    return abs(current_edge) < (1.0 - decay_threshold) * abs(entry_edge)


def group_records_by_market(records: Iterable[dict]) -> dict[str, list[dict]]:
    """Group records (each must have market_id + timestamp) chronologically per market."""
    by_mid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        mid = r.get("market_id")
        if not mid:
            continue
        by_mid[mid].append(r)
    for rs in by_mid.values():
        rs.sort(key=lambda r: r.get("timestamp", "") or "")
    return dict(by_mid)


def simulate_market(
    rounds: list[dict],
    decay_threshold: float,
    spread_cost: float,
) -> Optional[dict]:
    """Simulate hold-vs-early-exit for one market.

    rounds: chronological list of all ledger records for this market_id
    Returns None if no resolved record or no actionable entry. Otherwise:
      {
        "outcome": float,
        "action": "BUY_YES"|"BUY_NO",
        "entry_price": float, "entry_edge": float,
        "held_pnl": float,
        "exit_pnl": float,
        "exited": bool, "exit_round": int|None, "exit_price": float|None,
      }
    """
    if not rounds:
        return None
    # Need an entry that was actually a bet
    entry = None
    for r in rounds:
        if r.get("action") in ("BUY_YES", "BUY_NO"):
            entry = r
            break
    if entry is None:
        return None
    # Need a resolved record for the same market
    resolved = next((r for r in rounds if r.get("resolved")), None)
    if resolved is None:
        return None
    outcome = resolved.get("outcome")
    if outcome not in (0.0, 1.0):
        return None

    action = entry["action"]
    entry_price = float(entry.get("market_price", 0.5))
    entry_edge = float(entry.get("edge", 0.0))

    # Hold-to-resolve baseline
    h = held_pnl(action, entry_price, float(outcome))

    # Walk forward looking for an exit trigger
    exited = False
    exit_round = None
    exit_price = None
    e_pnl = h  # default = hold
    for i, later in enumerate(rounds):
        if later is entry:
            continue
        if later.get("timestamp", "") <= entry.get("timestamp", ""):
            continue
        new_edge = later.get("edge")
        if new_edge is None:
            continue
        if should_exit(entry_edge, float(new_edge), decay_threshold):
            exit_price = float(later.get("market_price", entry_price))
            e_pnl = per_dollar_pnl(action, entry_price, exit_price) - spread_cost
            exited = True
            exit_round = i
            break

    return {
        "outcome": float(outcome),
        "action": action,
        "entry_price": entry_price,
        "entry_edge": entry_edge,
        "held_pnl": h,
        "exit_pnl": e_pnl,
        "exited": exited,
        "exit_round": exit_round,
        "exit_price": exit_price,
    }


def simulate_ledger(
    records: list[dict],
    decay_threshold: float,
    spread_cost: float,
) -> dict:
    """Aggregate simulation across the whole ledger."""
    by_mid = group_records_by_market(records)
    n_markets = 0
    n_exited = 0
    held_total = 0.0
    exit_total = 0.0
    held_wins = 0
    exit_wins = 0
    for mid, rs in by_mid.items():
        result = simulate_market(rs, decay_threshold, spread_cost)
        if result is None:
            continue
        n_markets += 1
        held_total += result["held_pnl"]
        exit_total += result["exit_pnl"]
        if result["held_pnl"] > 0:
            held_wins += 1
        if result["exit_pnl"] > 0:
            exit_wins += 1
        if result["exited"]:
            n_exited += 1
    return {
        "n_markets": n_markets,
        "n_exited": n_exited,
        "held_total_pnl": held_total,
        "exit_total_pnl": exit_total,
        "held_wins": held_wins,
        "exit_wins": exit_wins,
        "diff": exit_total - held_total,
    }
