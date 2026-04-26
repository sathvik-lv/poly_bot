"""Snapshot rotation — keep last N days of price snapshots.

Snapshots are tiny (~1KB each) but accumulate forever. This script enforces
a retention window so the data dir doesn't grow without bound. Safe to run
in CI before commit; only deletes files whose mtime is older than the cutoff.
"""

from __future__ import annotations

import os
import sys
import time

DATA_DIR = "data"
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
RETENTION_DAYS = int(os.environ.get("SNAPSHOT_RETENTION_DAYS", "60"))


def main():
    if not os.path.isdir(SNAPSHOT_DIR):
        print(f"  No snapshot dir at {SNAPSHOT_DIR} — nothing to rotate.")
        return

    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    kept = 0
    failed = 0
    for name in os.listdir(SNAPSHOT_DIR):
        path = os.path.join(SNAPSHOT_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            failed += 1
            continue
        if mtime < cutoff:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                failed += 1
        else:
            kept += 1

    print(f"  Snapshot rotation (retention={RETENTION_DAYS}d):")
    print(f"    Removed: {removed}")
    print(f"    Kept:    {kept}")
    if failed:
        print(f"    Failed:  {failed}")


if __name__ == "__main__":
    main()
