"""Unified reader for ledger files that may have been rotated into archives.

Ledger files (test1_ledger, v2_ledger, ...) grow past GitHub's 100 MB per-file
limit if left alone. scripts/rotate_ledger.py moves old resolved records into
gzipped per-month archives under data/archive/. This helper hides the split
so downstream readers keep a single call site.

Usage:
    from src.ledger_reader import iter_ledger, ledger_paths
    for rec in iter_ledger("test1"):
        ...
    # or if a caller wants the raw paths (for tooling that grep/wc/etc):
    for path in ledger_paths("v2"):
        ...
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from typing import Iterator

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")


def expand_with_archives(paths: list[str]) -> list[str]:
    """Expand a list of ledger paths to include their rotated archives.

    For any path ``data/<name>_ledger.jsonl``, prepend
    ``data/archive/<name>_ledger_*.jsonl(.gz)`` matches (sorted, oldest-first)
    so archives are read before the current file. Non-ledger paths
    (e.g. live_predictions.jsonl) pass through unchanged.
    """
    out: list[str] = []
    for p in paths:
        base = os.path.basename(p)
        if base.endswith("_ledger.jsonl"):
            name = base[: -len("_ledger.jsonl")]
            pattern = os.path.join(ARCHIVE_DIR, f"{name}_ledger_*.jsonl*")
            out.extend(sorted(glob.glob(pattern)))
        out.append(p)
    return out


def ledger_paths(name: str) -> list[str]:
    """All files that contribute records for a ledger, oldest-first.

    Order: archives sorted lexicographically (naming is YYYY-MM so this is
    chronological), then the current file. Callers that want strict time
    order should also sort records by resolved_at/timestamp.
    """
    pattern = os.path.join(ARCHIVE_DIR, f"{name}_ledger_*.jsonl*")
    paths = sorted(glob.glob(pattern))
    current = os.path.join(DATA_DIR, f"{name}_ledger.jsonl")
    if os.path.exists(current):
        paths.append(current)
    return paths


def iter_ledger(name: str) -> Iterator[dict]:
    """Yield parsed records from every archive + the current file."""
    for path in ledger_paths(name):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def read_ledger_lines(name: str) -> Iterator[str]:
    """Yield raw JSON lines from every archive + the current file.

    For readers that pre-existed and pass raw lines through json.loads at their
    own site. Keeps their control flow unchanged.
    """
    for path in ledger_paths(name):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except OSError:
            continue
