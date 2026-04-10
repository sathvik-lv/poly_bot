"""Shadow Ledger — Kelly fraction stress testing.

The shadow ledger answers ONE question: at what fraction of full Kelly should
we allocate capital, given the bot's actual edge realization?

We don't precompute fixed dollar sizes. Instead, the collector records the raw
prediction (edge, price, action) and the resolver records the outcome. The
REPORT then replays the entire history for each Kelly fraction with COMPOUNDING
equity — bet size at each step depends on the current equity, not a fixed
notional. That's the only way to honestly stress-test sizing.

Kelly fractions tested:
    1/8x, 1/5x, 1/4x, 1/3x, 1/2x, 1x

The 1/3x default comes from the strategy_adapter; 1/2x is more aggressive,
1/5x more conservative. Full 1x is the theoretical growth-optimal but
historically too volatile in real markets.

Per-model shadow:
    Each sub-model also gets a solo "what if this model alone made the call"
    trade at 1/3 Kelly for fair comparison.
"""

from __future__ import annotations


# Kelly fractions to test in the report (compounded equity curves)
KELLY_FRACTIONS = {
    "kelly_1/8x":  1.0 / 8.0,
    "kelly_1/5x":  1.0 / 5.0,
    "kelly_1/4x":  1.0 / 4.0,
    "kelly_1/3x":  1.0 / 3.0,   # current strategy default
    "kelly_1/2x":  1.0 / 2.0,
    "kelly_1x":    1.0,
}

DEFAULT_PER_MODEL_FRACTION = 1.0 / 3.0  # 1/3 Kelly for per-model solo trades
DEFAULT_NOTIONAL = 10_000.0             # starting equity for replay


def compute_full_kelly(edge: float, price: float) -> float:
    """Full Kelly fraction for a binary prediction-market bet.

    For BUY_YES at price p with predicted prob q (edge = q - p):
        f = (q - p) / (1 - p) = edge / (1 - p)
    For BUY_NO (edge < 0):
        f = (p - q) / p = -edge / p
    Returns the unsigned fraction of bankroll Kelly says to bet.
    """
    if edge > 0:
        denom = 1.0 - price
        if denom <= 1e-6:
            return 0.0
        return edge / denom
    elif edge < 0:
        if price <= 1e-6:
            return 0.0
        return -edge / price
    return 0.0


def trade_pnl(amount: float, action: str, entry_price: float, outcome: float) -> float:
    """Realized P&L of a single bet given the resolved outcome (1.0 or 0.0)."""
    if amount <= 0:
        return 0.0
    if action == "BUY_YES":
        if entry_price <= 1e-6:
            return 0.0
        shares = amount / entry_price
        return shares * outcome - amount
    elif action == "BUY_NO":
        denom = 1.0 - entry_price
        if denom <= 1e-6:
            return 0.0
        shares = amount / denom
        return shares * (1.0 - outcome) - amount
    return 0.0


def replay_kelly_fraction(
    resolved_records: list,
    fraction: float,
    starting_equity: float = DEFAULT_NOTIONAL,
    max_bet_pct: float = 0.25,
) -> dict:
    """Replay all resolved trades chronologically at a given Kelly fraction.

    Uses the SAME `kelly_fraction` value the prediction engine produced for
    Test 0 (uncertainty-shrunk Kelly), and applies `fraction` as the
    allocation multiplier. So 1/3 reproduces Test 0's strategy adapter,
    1/2 is more aggressive, 1/8 is more conservative. Equity compounds.

    Returns:
        {
            "fraction": fraction,
            "starting_equity": starting_equity,
            "final_equity": ...,
            "total_pnl": ...,
            "roi_pct": ...,
            "n_trades": ...,
            "wins": ..., "losses": ...,
            "max_equity": ..., "min_equity": ...,
            "max_drawdown_pct": ...,
            "ruin": bool (True if equity ever hit 0 or near-0),
            "equity_curve": [...]   # one point per resolved trade
        }
    """
    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    wins = 0
    losses = 0
    n_trades = 0
    ruin = False
    curve = [equity]

    # Sort by resolution time so compounding reflects actual sequence
    sorted_records = sorted(
        resolved_records,
        key=lambda r: r.get("resolved_at") or r.get("timestamp") or "",
    )

    for r in sorted_records:
        price = r.get("market_price")
        action = r.get("action")
        outcome = r.get("outcome")
        # Use the engine's stored Kelly (uncertainty-shrunk) — same as Test 0
        kelly = r.get("kelly_fraction")
        if price is None or outcome is None or kelly is None:
            continue
        if action not in ("BUY_YES", "BUY_NO"):
            continue
        if equity <= 0.01:
            ruin = True
            curve.append(0.0)
            continue

        # |kelly| because BUY_NO gives a negative kelly_fraction
        full_k = abs(float(kelly))
        if full_k <= 0:
            curve.append(equity)
            continue

        bet_pct = min(full_k * fraction, max_bet_pct)
        bet = bet_pct * equity
        pnl = trade_pnl(bet, action, price, outcome)
        equity += pnl
        if equity < 0:
            equity = 0.0
            ruin = True

        n_trades += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        curve.append(round(equity, 4))

    return {
        "fraction": fraction,
        "starting_equity": starting_equity,
        "final_equity": round(equity, 4),
        "total_pnl": round(equity - starting_equity, 4),
        "roi_pct": round((equity / starting_equity - 1.0) * 100, 4) if starting_equity > 0 else 0,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / n_trades * 100, 2) if n_trades else 0,
        "max_equity": round(peak, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "ruin": ruin,
        "equity_curve": curve,
    }


def compute_per_model_shadow(
    model_estimates: dict,
    price: float,
) -> dict:
    """For each sub-model, compute its solo full-Kelly (raw) and intended action.

    Stored at prediction time so the resolver/report can later compute that
    model's standalone P&L. Sizing is left to the report (compounding).
    """
    out = {}
    for name, est in (model_estimates or {}).items():
        if est is None:
            continue
        try:
            est = float(est)
        except (TypeError, ValueError):
            continue
        edge = est - price
        full_k = compute_full_kelly(edge, price)
        action = "BUY_YES" if edge > 0 else ("BUY_NO" if edge < 0 else "NO_BET")
        out[name] = {
            "edge": round(edge, 4),
            "action": action,
            "full_kelly": round(full_k, 6),
        }
    return out


def replay_per_model_kelly_fraction(
    resolved_records: list,
    model_name: str,
    fraction: float = DEFAULT_PER_MODEL_FRACTION,
    starting_equity: float = DEFAULT_NOTIONAL,
) -> dict:
    """Replay a single sub-model's solo trades chronologically with compounding.

    Pulls per_model_shadow[model_name] from each record, uses that model's
    standalone edge and action, sizes via Kelly fraction off compounding equity.
    """
    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    wins = 0
    losses = 0
    n_trades = 0
    ruin = False

    sorted_records = sorted(
        resolved_records,
        key=lambda r: r.get("resolved_at") or r.get("timestamp") or "",
    )

    for r in sorted_records:
        outcome = r.get("outcome")
        price = r.get("market_price")
        if outcome is None or price is None:
            continue
        pms = (r.get("per_model_shadow") or {}).get(model_name)
        if not pms:
            continue
        action = pms.get("action")
        full_k = pms.get("full_kelly", 0)
        if action not in ("BUY_YES", "BUY_NO") or full_k <= 0:
            continue
        if equity <= 0.01:
            ruin = True
            continue

        bet = min(full_k * fraction, 1.0) * equity
        pnl = trade_pnl(bet, action, price, outcome)
        equity += pnl
        if equity < 0:
            equity = 0.0
            ruin = True

        n_trades += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return {
        "model": model_name,
        "fraction": fraction,
        "final_equity": round(equity, 4),
        "total_pnl": round(equity - starting_equity, 4),
        "roi_pct": round((equity / starting_equity - 1.0) * 100, 4) if starting_equity > 0 else 0,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / n_trades * 100, 2) if n_trades else 0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "ruin": ruin,
    }
