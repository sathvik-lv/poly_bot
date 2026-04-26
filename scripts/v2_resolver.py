"""V2 resolver — same logic as test1_resolver, pointed at data/v2_ledger.jsonl.

Filters to past-end_date markets only, caps API calls, parallelizes via thread
pool. Mirrors the resolver fix shipped in commit c9eb4c2.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from requests.adapters import HTTPAdapter

DATA_DIR = "data"
LEDGER_FILE = os.path.join(DATA_DIR, "v2_ledger.jsonl")
GAMMA_API = "https://gamma-api.polymarket.com"
POOL_SIZE = 16
REQUEST_TIMEOUT = 6
MAX_API_CALLS = 500
GRACE_SECONDS = 3600


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
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = LEDGER_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, LEDGER_FILE)


def get_outcome(session: requests.Session, market_id: str):
    try:
        r = session.get(f"{GAMMA_API}/markets/{market_id}", timeout=REQUEST_TIMEOUT)
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


def _end_date_passed(end_date_str, now_ts: float) -> bool:
    if not end_date_str:
        return True
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return dt.timestamp() <= now_ts - GRACE_SECONDS
    except (ValueError, TypeError):
        return True


def resolve():
    records = load_all()
    if not records:
        print("\n  V2 ledger empty — nothing to resolve.")
        return

    unresolved = [r for r in records if not r.get("resolved")]
    print(f"\n  V2 RESOLVER — {len(unresolved)} unresolved / {len(records)} total")
    if not unresolved:
        print("  Nothing to resolve.")
        return

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
        if _end_date_passed(r.get("end_date"), now_ts):
            market_due[mid] = True
        else:
            market_due.setdefault(mid, False)

    due_markets = [mid for mid, due in market_due.items() if due]
    skipped = len(by_market) - len(due_markets)
    print(f"  {len(due_markets)} markets due / {skipped} skipped")

    if len(due_markets) > MAX_API_CALLS:
        print(f"  Capping to {MAX_API_CALLS} oldest markets")
        due_markets = sorted(
            due_markets,
            key=lambda mid: records[by_market[mid][0]].get("end_date") or "",
        )[:MAX_API_CALLS]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    adapter = HTTPAdapter(pool_connections=POOL_SIZE, pool_maxsize=POOL_SIZE)
    session.mount("https://", adapter)

    n_resolved = 0
    resolved_at = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=POOL_SIZE) as pool:
        futures = {pool.submit(get_outcome, session, mid): mid for mid in due_markets}
        for fut in as_completed(futures):
            outcome = fut.result()
            if outcome is None:
                continue
            mid = futures[fut]
            for idx in by_market[mid]:
                r = records[idx]
                r["resolved"] = True
                r["outcome"] = outcome
                r["resolved_at"] = resolved_at
                n_resolved += 1

    if n_resolved > 0:
        save_all(records)

    print(f"  Resolved {n_resolved} predictions across {len(due_markets)} markets")


if __name__ == "__main__":
    resolve()
