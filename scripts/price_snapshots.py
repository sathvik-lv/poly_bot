"""Price Snapshot Collector — Builds price history for time series model.

Runs periodically to capture market prices over time. After enough
snapshots (10+), the time series model can detect trends, momentum,
and mean-reversion patterns.

Usage:
    python scripts/price_snapshots.py              # take one snapshot
    python scripts/price_snapshots.py --loop       # snapshot every 2 hours
    python scripts/price_snapshots.py --report     # show collected history
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

DATA_DIR = "data"
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
HISTORY_FILE = os.path.join(DATA_DIR, "price_history.json")
GAMMA_API = "https://gamma-api.polymarket.com"


def load_history() -> dict:
    """Load price history. Format: {market_id: [{timestamp, price, volume}, ...]}"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {}


def save_history(history: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


def take_snapshot(n_markets: int = 60):
    """Snapshot current prices for top active markets."""
    print(f"  Taking price snapshot at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}...")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_markets = []
    for offset in [0, 100]:
        resp = session.get(f"{GAMMA_API}/markets", params={
            "limit": 100, "active": True, "closed": False,
            "order": "volume24hr", "ascending": False, "offset": offset,
        }, timeout=15)
        batch = resp.json()
        if not batch:
            break
        all_markets.extend(batch)

    history = load_history()
    now = datetime.now(timezone.utc).isoformat()
    new_points = 0

    for m in all_markets[:n_markets]:
        mid = m.get("id") or m.get("condition_id")
        if not mid:
            continue

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        if not prices:
            continue

        yes_price = float(prices[0])
        if yes_price <= 0 or yes_price >= 1:
            continue

        volume = float(m.get("volume", 0) or 0)
        spread = m.get("spread")

        point = {
            "timestamp": now,
            "price": round(yes_price, 4),
            "volume": volume,
        }
        if spread is not None:
            point["spread"] = float(spread)

        if mid not in history:
            history[mid] = {
                "question": m.get("question", "")[:100],
                "end_date": m.get("endDate") or m.get("end_date"),
                "snapshots": [],
            }

        # Avoid duplicate snapshots within 30 minutes
        existing = history[mid]["snapshots"]
        if existing:
            last_ts = existing[-1].get("timestamp", "")
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                if (now_dt - last_dt).total_seconds() < 1800:
                    continue
            except Exception:
                pass

        history[mid]["snapshots"].append(point)
        new_points += 1

        # Keep max 500 snapshots per market (covers ~40 days at 2h intervals)
        if len(history[mid]["snapshots"]) > 500:
            history[mid]["snapshots"] = history[mid]["snapshots"][-500:]

    save_history(history)

    # Also save raw snapshot file for git history
    snap_ts = int(time.time())
    snap_file = os.path.join(SNAPSHOTS_DIR, f"snap_{datetime.now().strftime('%j')}_{snap_ts}.json")
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snap_data = {}
    for m in all_markets[:n_markets]:
        mid = m.get("id") or m.get("condition_id")
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        if prices:
            snap_data[mid] = {
                "price": float(prices[0]),
                "question": (m.get("question") or "")[:80],
            }
    with open(snap_file, "w") as f:
        json.dump({"timestamp": now, "markets": snap_data}, f)

    total_markets = len(history)
    total_points = sum(len(h["snapshots"]) for h in history.values())
    ts_ready = sum(1 for h in history.values() if len(h["snapshots"]) >= 10)

    print(f"  New data points: {new_points}")
    print(f"  Total markets tracked: {total_markets}")
    print(f"  Total data points: {total_points}")
    print(f"  Markets with 10+ snapshots (time series ready): {ts_ready}")

    return new_points


def get_price_history(market_id: str) -> list[float]:
    """Get price history for a specific market (used by prediction engine)."""
    history = load_history()
    entry = history.get(market_id)
    if not entry:
        return []
    return [s["price"] for s in entry.get("snapshots", [])]


def report():
    """Show price history summary."""
    history = load_history()
    if not history:
        print("\n  No price history collected yet. Run --snapshot first.")
        return

    total_points = sum(len(h["snapshots"]) for h in history.values())
    ts_ready = [(mid, h) for mid, h in history.items() if len(h["snapshots"]) >= 10]

    print(f"\n{'='*70}")
    print(f"  PRICE HISTORY REPORT")
    print(f"{'='*70}")
    print(f"  Markets tracked:  {len(history)}")
    print(f"  Total snapshots:  {total_points}")
    print(f"  TS-ready (10+):   {len(ts_ready)}")

    # Top markets by data points
    sorted_markets = sorted(history.items(), key=lambda x: -len(x[1]["snapshots"]))
    print(f"\n  TOP MARKETS BY DATA POINTS:")
    for mid, h in sorted_markets[:15]:
        snaps = h["snapshots"]
        n = len(snaps)
        q = h.get("question", "")[:50]
        first_p = snaps[0]["price"] if snaps else 0
        last_p = snaps[-1]["price"] if snaps else 0
        change = last_p - first_p
        print(f"    [{n:>3} pts] {q}")
        print(f"             First={first_p:.3f} Last={last_p:.3f} Change={change:+.3f}")

    # Time series ready markets
    if ts_ready:
        print(f"\n  TIME SERIES READY ({len(ts_ready)} markets):")
        for mid, h in ts_ready[:10]:
            snaps = h["snapshots"]
            prices = [s["price"] for s in snaps]
            q = h.get("question", "")[:50]
            trend = "UP" if prices[-1] > prices[0] else "DOWN" if prices[-1] < prices[0] else "FLAT"
            vol = float(__import__("numpy").std(prices)) if len(prices) > 2 else 0
            print(f"    {q}")
            print(f"      Points={len(snaps)} Trend={trend} Vol={vol:.4f} "
                  f"Range=[{min(prices):.3f}-{max(prices):.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Price Snapshot Collector")
    parser.add_argument("--loop", action="store_true", help="Take snapshots every 2 hours")
    parser.add_argument("--report", action="store_true", help="Show history report")
    parser.add_argument("--interval", type=int, default=7200, help="Loop interval seconds (default 2h)")
    parser.add_argument("-n", type=int, default=60, help="Number of markets to snapshot")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.report:
        report()
    elif args.loop:
        print(f"  Snapshot loop: every {args.interval//3600}h {(args.interval%3600)//60}m")
        while True:
            try:
                take_snapshot(args.n)
                print(f"  Next snapshot in {args.interval//3600}h...\n")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n  Stopped.")
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(300)
    else:
        take_snapshot(args.n)


if __name__ == "__main__":
    main()
