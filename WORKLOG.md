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
- **Unresolved:** `data/test1_ledger.jsonl` is 94.9 MB, against GitHub's hard
  100 MB per-file limit. `rotate_ledger.py` is not keeping up. When it crosses,
  pushes to the data repo will be rejected and results will stop saving.
- Not done (deliberate): parameters stay in `cycle.yml` rather than Secrets —
  GitHub masks a secret's exact string everywhere, so `"0.5"` would blank every
  probability in the logs, and `"4"` is under the 3-char masking floor anyway.
  Keeping them versioned preserves the A/B experiment audit trail.

## 2026-07-16 — MacBook
- Set up cross-device worklog. Verified this repo is fully synced with GitHub (no unpushed/uncommitted changes).
