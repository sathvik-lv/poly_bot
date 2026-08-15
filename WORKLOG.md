# Worklog

Newest entries at the top. Each device: pull + read this before working, append + push after working.

## 2026-08-15 — Windows PC
- Security audit across all repos. This one is public and was committing `data/`
  hourly — publishing live paper positions, per-arm ledgers and fitted weights.
  Traffic showed ~280 unique automated cloners/fortnight vs 1 human viewer.
- Moved `data/` to the private repo `sathvik-lv/poly_bot-data`, seeded with full
  current state. Repo stays public to keep unlimited Actions minutes.
- `cycle.yml` clones the data repo into `data/` via a write **deploy key**
  (no PAT expiry) and commits results there. Two guards added because the
  scripts would otherwise overwrite ledgers with empty state: the checkout has
  no `continue-on-error`, and a follow-up step aborts unless
  `data/paper_trades.json` exists.
- `private-dashboards/refresh.yml` updated to clone the data repo into
  `_src/poly_bot/data` (read-only deploy key), so `POLY_ROOT` resolves as before.
- Cancelled the 01:26 UTC scheduled run mid-flight: it was on the pre-migration
  SHA and its `-X theirs` rebase would have resurrected `data/`. One cycle lost.
- **To run locally:** `git clone git@github.com:sathvik-lv/poly_bot-data.git data`
  — your existing local `data/` is now untracked and will go stale.
- **Ledger rotation fixed** (this was misdiagnosed at first — test1 is frozen at
  94.9 MB since 2026-08-06, NOT growing, so it was never going to hit 100 MB).
  Real defect: rotation only archived records that were *resolved* AND older
  than `KEEP_DAYS`, so unresolved records could never be archived. test1 is
  98.9% unresolved — hence "no records to archive" every cycle at 94.9 MB.
  v2_ledger has the same defect slower: ~40% of aging records unresolved, floor
  climbing ~0.5 MB/day toward the 100 MB push-rejection cap (~2-3 months out).
  Fix: `MAX_UNRESOLVED_DAYS = 90` makes dead unresolved records archivable, and
  `RETIRED_LEDGERS` archives frozen test1 in full.
  Verified on a clone of real data: test1 95.0 MB -> 0 bytes with 100,333
  records still readable and 0 duplicates; v2 (threshold forced) 75.8 -> 54.0 MB,
  48,667 readable, 0 duplicates, 0 old-unresolved left pinned; 219 tests pass.
  Safe because every CI consumer reads via `ledger_reader`/`data_validator`,
  which glob archives + current and dedup.
- Not done (deliberate): parameters stay in `cycle.yml` rather than Secrets —
  GitHub masks a secret's exact string everywhere, so `"0.5"` would blank every
  probability in the logs, and `"4"` is under the 3-char masking floor anyway.
  Keeping them versioned preserves the A/B experiment audit trail.

## 2026-07-16 — MacBook
- Set up cross-device worklog. Verified this repo is fully synced with GitHub (no unpushed/uncommitted changes).
