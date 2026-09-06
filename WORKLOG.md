# Worklog

Newest entries at the top. Each device: pull + read this before working, append + push after working.

## 2026-09-06 — MacBook
- Caught this device up: it was 156 commits behind and still had the old tracked
  `data/`. Fast-forwarded, then re-cloned the data repo into `data/` over HTTPS
  (no SSH key on this machine; the keychain credential works).
- Reviewed paper-trading results. Headline numbers are **not real**: the main arm
  shows +48.3% (201 closed, 74.1% win) but 90% of its closed trades — 98-100% on
  v3c/v6 — booked `entry_price` at exactly $0.500.
- Cause: `src/prediction_engine.py:999` does
  `outcome_prices.get("Yes", 0.5)`. Markets whose outcomes are player/team names
  ("US Open ATP: Zverev vs Halys" -> `['Alexander Zverev','Quentin Halys']`) have
  no `Yes` key, so the engine silently prices them at 0.5. The scanner's own
  filter (`scripts/paper_trader.py:424-430`) already falls back to the first
  outcome price, so these markets pass the 0.05-0.95 gate on their real price and
  are then modelled, sized and settled at a fabricated 50c. Winners pay 2x.
- Re-priced the closed trades against real snapshot prices in `price_history.json`
  (nearest snapshot within 1h, n=135): ROI falls from +61% to +20.3%, per-trade
  mean +4.4% with sd 64% (t=0.79 — not distinguishable from zero), and 5 trades
  supply $1,022 of the $1,158. Model Brier on those trades **0.226 vs the real
  market price's 0.142** — the ensemble is much worse than just reading the price.
- Same fallback contaminates training: `market_price == 0.5` on 87.6% of
  `v2_ledger.jsonl` and 62.7% of `live_predictions.jsonl`. So the "market_brier
  0.2337 / improvement +0.009" in `v2_train_report.json` is measured against a
  constant coin flip (Brier 0.25), not against the market.
- v4_ai is the control: it traded real Yes/No markets at real prices and lost
  19.8%. Consistent with the honest backtest (no edge vs market).
- Next: fix is a first-outcome fallback in the engine (and line 823/499), but it
  invalidates every arm's ledger history — those need resetting or re-labelling
  before any cross-arm comparison means anything. Not touched yet.
- Power check on v6_tier (n=91, the arm closest to a "go live" call): 90/91 are
  0.5-priced, 68 re-priceable. Re-priced it is **−10.4%/trade** (95% CI −24% to
  +3.3%, t=−1.49), winning 72.1% where the real prices imply 79.4%. Needs ~240
  trades to call that effect, ~496 for its Brier gap, and the 91 trades are only
  ~29 days / 49 day-family blocks, so effective n is roughly half of nominal.
- V6-TIER vs its control Test 0-TIER (same tier caps, V1 vs V2 pipeline), both
  re-priced: v6 −10.4%/trade vs test0 −0.2%/trade. They agree on direction on
  38/39 shared markets, so P&L can't separate them; v6's *extra* trades (the ones
  the V2 gate admits and the control skips) are the worse half, −15.4% vs −4.4%.
  V2 does sharpen probabilities — Brier 0.194 vs control's 0.234 on identical
  markets — but the real market price is 0.176, so there is nothing to harvest.
  V6-TIER is labelled LIVE-LAUNCH CANDIDATE in `cycle.yml`; it should not be.
- The "50+ resolved trades" bar in CLAUDE.md is far too low: per-trade sd is ~57%,
  so n=50 can only detect an edge of ≥23%/trade. P&L is the wrong yardstick.
  Paired Brier on the shadow ledger is: 3,424 resolved records already carry a
  real price, model 0.1515 vs market 0.1530, diff −0.0015, t=−4.20 — significant
  but economically tiny. That is the only channel with enough n today.

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
- The downstream dashboards repo's refresh workflow was updated to clone the data
  repo into `_src/poly_bot/data` (read-only deploy key) so `POLY_ROOT` resolves as
  before. Details in that repo's own worklog.
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
