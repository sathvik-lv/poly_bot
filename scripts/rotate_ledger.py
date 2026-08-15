"""Rotate ledger files to stay under GitHub's 100 MB per-file limit.

If a ledger exceeds THRESHOLD_MB, move old records into per-month gzipped
archives under data/archive/:
  - resolved records older than KEEP_DAYS
  - unresolved records older than MAX_UNRESOLVED_DAYS (long past any plausible
    resolution; without this they pin the live file open forever)
Recent records — and anything with no parseable timestamp — stay in the live
file so collectors and reporters see them without opening the archives.

Ledgers in RETIRED_LEDGERS are no longer written by CI and are archived in
full, ignoring both the size threshold and resolution status.

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
# Resolved records older than this move to archives; newer ones stay live.
KEEP_DAYS = 45
# An UNRESOLVED record this old is dead. The scanner targets markets resolving
# within ~14 days, so by 90 days the resolver has had ~75+ days past market end
# and still not closed it.
#
# Without this rule unresolved records could NEVER be archived, which gave the
# live file a permanently rising floor that rotation could not claw back. That
# is how test1_ledger reached 94.9 MB while still reporting "no records older
# than <cutoff> to archive" every cycle: 98.9% of its records were unresolved.
MAX_UNRESOLVED_DAYS = 90

LEDGERS = ["test1_ledger.jsonl", "v2_ledger.jsonl"]

# Ledgers no longer written by CI (test1 was pulled from the cycle 2026-08-06).
# These are archived in FULL, regardless of size or resolution status. Every
# consumer reads them through src.ledger_reader / src.data_validator, both of
# which glob archives + current and dedup across the two, so archiving loses
# nothing for trainers, reports, or the category gate. The only readers that
# open the live file directly are test1_collector/test1_resolver, neither of
# which runs in CI any more.
RETIRED_LEDGERS = {"test1_ledger.jsonl"}


def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def rotate_one(path: str, retired: bool = False) -> None:
    if not os.path.exists(path):
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if retired:
        if os.path.getsize(path) == 0:
            print(f"[rotate] {path}: retired and already fully archived, skip")
            return
        print(f"[rotate] {path}: RETIRED ledger, archiving in full ({size_mb:.1f} MB)")
    elif size_mb < THRESHOLD_MB:
        print(f"[rotate] {path}: {size_mb:.1f} MB (below {THRESHOLD_MB} MB, skip)")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    unresolved_cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_UNRESOLVED_DAYS)
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
            # No parseable timestamp — no safe month bucket, so always keep.
            if ts is None:
                keep_lines.append(line)
                kept += 1
                continue
            if retired:
                # Frozen ledger: everything is history, archive it all.
                archive_it = True
            elif resolved:
                archive_it = ts < cutoff
            else:
                # Unresolved records are archivable only once they are far
                # enough past any plausible resolution date. See
                # MAX_UNRESOLVED_DAYS — this is what stops the live file
                # accumulating a floor rotation can never reclaim.
                archive_it = ts < unresolved_cutoff
            if archive_it:
                month = ts.strftime("%Y-%m")
                archived_by_month.setdefault(month, []).append(line)
                archived += 1
            else:
                keep_lines.append(line)
                kept += 1

    if archived == 0:
        print(f"[rotate] {path}: {size_mb:.1f} MB but nothing archivable "
              f"(resolved before {cutoff.date()}, or unresolved before "
              f"{unresolved_cutoff.date()})")
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
            rotate_one(os.path.join(DATA_DIR, name), retired=name in RETIRED_LEDGERS)
        except Exception as e:
            print(f"[rotate] ERROR on {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
