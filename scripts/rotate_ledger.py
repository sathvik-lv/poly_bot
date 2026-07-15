"""Rotate ledger files to stay under GitHub's 100 MB per-file limit.

If a ledger exceeds THRESHOLD_MB, move records with resolved_at older than
KEEP_DAYS into per-month gzipped archives under data/archive/. Unresolved
records and recent resolved records stay in the live file so the collectors
and reporters see them without needing to open the archives.

Downstream readers that need historical data use src.ledger_reader.iter_ledger
which globs archives + current transparently.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")

# Rotate any ledger over this size (leaves headroom below the 100 MB GitHub cap).
THRESHOLD_MB = 80
# Records older than this stay in archives; newer records stay in the live file.
KEEP_DAYS = 45

LEDGERS = ["test1_ledger.jsonl", "v2_ledger.jsonl"]


def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def rotate_one(path: str) -> None:
    if not os.path.exists(path):
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < THRESHOLD_MB:
        print(f"[rotate] {path}: {size_mb:.1f} MB (below {THRESHOLD_MB} MB, skip)")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    keep_lines: list[str] = []
    archived_by_month: dict[str, list[str]] = {}
    total = 0
    kept = 0
    archived = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # Malformed line — safer to keep than to drop.
                keep_lines.append(line)
                kept += 1
                continue
            resolved = bool(r.get("resolved"))
            ts = parse_iso(r.get("resolved_at") or r.get("timestamp"))
            # Keep everything unresolved, everything with no parseable ts, and
            # anything recent. Only archive OLD RESOLVED records.
            if not resolved or ts is None or ts >= cutoff:
                keep_lines.append(line)
                kept += 1
            else:
                month = ts.strftime("%Y-%m")
                archived_by_month.setdefault(month, []).append(line)
                archived += 1

    if archived == 0:
        print(f"[rotate] {path}: {size_mb:.1f} MB but no records older than "
              f"{cutoff.date()} to archive")
        return

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    for month, lines in sorted(archived_by_month.items()):
        arch = os.path.join(ARCHIVE_DIR, f"{base}_{month}.jsonl.gz")
        # Append: a later rotation may add more records to a month that was
        # already partially archived on a previous run.
        with gzip.open(arch, "at", encoding="utf-8") as gf:
            for ln in lines:
                gf.write(ln + "\n")
        print(f"[rotate] archived {len(lines):>6} records -> {arch}")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in keep_lines:
            f.write(ln + "\n")
    os.replace(tmp, path)
    new_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[rotate] {path}: total={total} kept={kept} archived={archived} "
          f"({size_mb:.1f} MB -> {new_mb:.1f} MB)")


def main():
    for name in LEDGERS:
        try:
            rotate_one(os.path.join(DATA_DIR, name))
        except Exception as e:
            print(f"[rotate] ERROR on {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
