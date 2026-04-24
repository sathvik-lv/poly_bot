"""TEST 1 — Shadow Resolver

Reads data/test1_ledger.jsonl, finds unresolved records, checks Polymarket
for resolution, and stores the outcome (1.0 / 0.0) on each record.

P&L is NOT precomputed — the report does compounded equity replay at multiple
Kelly fractions, so storing outcomes is enough.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "test1_ledger.jsonl")
GAMMA_API = "https://gamma-api.polymarket.com"


def load_all() -> list:
    if not os.path.exists(LEDGER_FILE):
        return []
    out = []
    with open(LEDGER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def save_all(records: list):
    """Atomic rewrite: write to .tmp then rename."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = LEDGER_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, LEDGER_FILE)


def get_outcome(session: requests.Session, market_id: str):
    """Return 1.0 / 0.0 / None if not resolved."""
    try:
        r = session.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
        if r.status_code != 200:
            return None
        m = r.json()
        if not m.get("closed"):
            return None
        outcome_str = m.get("outcome")
        if outcome_str == "Yes":
            return 1.0
        if outcome_str == "No":
            return 0.0
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if prices:
            p = float(prices[0])
            if p > 0.95:
                return 1.0
            if p < 0.05:
                return 0.0
        return None
    except Exception:
        return None


MAX_API_CALLS = 500
GRACE_SECONDS = 3600  # only check markets whose end_date is at least this far in the past


def _end_date_passed(end_date_str, now_ts: float) -> bool:
    """True if end_date is missing, unparseable, or already past (minus grace)."""
    if not end_date_str:
        return True  # unknown end date — fall through to API check
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return dt.timestamp() <= now_ts - GRACE_SECONDS
    except (ValueError, TypeError):
        return True


def resolve():
    records = load_all()
    if not records:
        print("\n  Test 1 ledger empty — nothing to resolve.")
        return

    unresolved = [r for r in records if not r.get("resolved")]
    print(f"\n  TEST 1 RESOLVER — {len(unresolved)} unresolved / {len(records)} total")

    if not unresolved:
        print("  Nothing to resolve.")
        return

    # Group by market_id, keeping the earliest plausible end_date per market
    now_ts = datetime.now(timezone.utc).timestamp()
    by_market: dict[str, list[int]] = {}
    market_due: dict[str, bool] = {}
    for idx, r in enumerate(records):
        if r.get("resolved"):
            continue
        mid = r.get("market_id")
        if not mid:
            continue
        by_market.setdefault(mid, []).append(idx)
        # If any record for this market indicates end_date has passed, mark due
        if _end_date_passed(r.get("end_date"), now_ts):
            market_due[mid] = True
        else:
            market_due.setdefault(mid, False)

    due_markets = [mid for mid, due in market_due.items() if due]
    skipped = len(by_market) - len(due_markets)
    print(f"  {len(due_markets)} markets due / {skipped} skipped (end_date in future)")

    # Cap API calls for bounded CI runtime
    if len(due_markets) > MAX_API_CALLS:
        print(f"  Capping to {MAX_API_CALLS} oldest markets this run")
        due_markets = sorted(
            due_markets,
            key=lambda mid: records[by_market[mid][0]].get("end_date") or "",
        )[:MAX_API_CALLS]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    n_resolved = 0
    for mid in due_markets:
        outcome = get_outcome(session, mid)
        if outcome is None:
            continue
        for idx in by_market[mid]:
            r = records[idx]
            r["resolved"] = True
            r["outcome"] = outcome
            r["resolved_at"] = datetime.now(timezone.utc).isoformat()
            n_resolved += 1

        time.sleep(0.05)  # be polite to API

    if n_resolved > 0:
        save_all(records)

    print(f"  Resolved {n_resolved} predictions across {len(due_markets)} markets checked")


if __name__ == "__main__":
    resolve()
