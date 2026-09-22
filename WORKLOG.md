# Worklog

Newest entries at the top. Three devices — **Windows PC**, **MacBook**, **Mobile** —
share this file and nothing else: no shared chat, memory or disk. Each device:
`git pull` + read this *before* working; append an entry + commit + push *after*
working. Work recorded here is done — don't redo it. Use one of those three device
labels exactly (`brain` parses the field). Full protocol in `CLAUDE.md`.

## 2026-09-22 — Windows PC (line-level audit of all 23.7k LOC + 1,791 commits)

Read-only forensic pass over every module and the whole commit history, executing the
math against reference values rather than reading it. New findings:

- **`src/training_runner.py:258 _simulate_resolutions()` fabricates outcomes.** It calls
  `tracker.record_outcome()` with `1.0` if price>0.9, `0.0` if price<0.1, else *the
  direction of the price move* when |Δ|>0.15. Those fake outcomes feed the self-audit
  that prints Brier/ECE/discrimination/ROI and sets `best_brier`. The third branch is
  circular — outcome is defined as "did price rise", scored against models that read
  price momentum. **Mitigation: dead code.** Nothing imports or runs it (only a
  docstring in `historical_trainer.py:732` mentions it). Leave it dead; if anyone
  revives it, that function must go first.
- **The `brier=0.063` in 641 commit messages is a label-imbalance artifact — proven.**
  `backtest_honest_preds.jsonl` has a YES base rate of **1.68%** (65 of 3,879). The
  source `historical_markets.jsonl` is *fine* (16,319 YES / 28,351 NO of 44,670), so
  the 0.05-0.95 price filter in `historical_trainer.py:318` selected a 98.3%-NO subset.
  A **constant 0.23 predictor scores Brier 0.0619** on those labels — i.e. the
  "achievement" was predicting one class against labels that were almost all that
  class. CLAUDE.md already called the 0.063 wrong; this is the mechanism.
  **Blast radius is contained:** only `v2_train_meta_full.py` trains on that file and
  it is **not in CI**. CI's `v2_train_meta.py` uses the ledgers only.
- **The 0.5 fabrication is systemic — ~20 sites, not the 3 from 09-06.** Beyond
  `prediction_engine.py` 499/823/999: `meta_model.py:75,85-88` defaults market_price
  *and all four sub-model estimates* to 0.5 in the feature vector (its docstring admits
  it), `strategy_adapter.py:320,912` defaults price to 0.5 inside the price-sanity
  filter and contrarian detector, `exit_simulator.py:104`, `self_improver.py:228,251`,
  `paper_trader.py:907`, `prediction_engine.py:154,1053`.
- **Hurst call-site bug:** `prediction_engine.py:197` computes `returns`, then line 202
  passes **`prices`** (levels). Verified H saturates at 1.000 on random walks at n=500
  and n=5000, so `regime_type` is always "trending". **Inert** — it only sets a
  diagnostic label (the estimate comes from HMM+GARCH) and `time_series` weight is 0.0.
- **150 `except: pass/continue` sites** (data_sources 10, niche_scanner 10,
  prediction_engine 8). This is the mechanism that hid every bug above for five months.

**What is genuinely correct — verified by execution, not by reading:**

- **`src/statistics.py` (839 lines) is the real asset.** GARCH(1,1) recovered
  ω=1.002e-05, α=0.1000, β=0.8500, persistence=0.9500 against true
  1e-05/0.10/0.85/0.95 — near-exact MLE. GaussianHMM recovered regime means
  −0.0204/+0.0193 against true ∓0.02. `kelly_with_uncertainty` gives full Kelly 0.20
  for p=0.6 at even odds (exactly right) with shrinkage 0.909→0.667→0.400 as σ rises.
  BetaBinomial(1,1)+update(8,2) → Beta(9,3), mean 0.750, CI (0.4822, 0.9398) — correct,
  and immutable-style (returns a new model; `b.update(...)` alone is a no-op, which
  looks like a bug and isn't). IsotonicCalibrator returns the correct PAV solution.
  KL/JS are 0 for identical inputs. The "deterministic, no random sampling" claim holds.
- **Wilson CI** in `category_gate.py` and `compare_arms.py` returns (0.4902, 0.9433) for
  8/10, which matches the formula worked by hand. Correct — do not "fix" it against a
  Clopper-Pearson reference.
- **`rolling_backtest.py` / `rolling_accuracy.json` is the one trustworthy backtest:**
  base rate 24.5%, n=5,211, market Brier 0.1670 vs ensemble 0.1672 (+0.0002). It
  honestly reported no alpha in April and has survived every re-test since.
- **Test suite: 310 tests / 434 assertions, zero skips, zero tautologies.**
- `paper_trader.py:424-430` is the one place the price fallback was written correctly.
- Several comments are honestly self-critical (`prediction_engine.py:152` records
  "cannot beat market price (Brier 0.1781 vs market 0.1665)"; self_improver is
  deliberately disabled with the reason stated). `historical_trainer.py`'s alarming
  `sim_price` is just a rename of the real stored price — honest despite the name.

- **Next:** nothing new shipped. If the engine 0.5 fix is ever done, it is ~20 sites and
  `meta_model.build_feature_vector` matters as much as the engine lines. 19 of 31
  scripts are in CI; the rest (incl. both old backtests and `v2_train_meta_full`) are
  manual/dead.

## 2026-09-22 — Windows PC (divergence study: 10-day checkpoint, read-only)

- No code changed. Cloned `poly_bot-data` fresh and read the 10-day accumulation to
  answer "how are the results coming through" — recording it here so the headline
  number in `divergence_report.json` doesn't get quoted without the caveat below.
- 46 polls / 9.6 days, 2,448 pairs, 1,007 markets, 497 games, **957 resolved**. 149
  credits spent, 338 left — on pace for the free tier.
- Cadence is running at ~4.8 polls/day, not the ~8/day the 3h gate allows (median gap
  5.0h, 31 of 45 gaps >4h, max 8.6h). Cause not dug into; likely the hourly cycle
  occasionally overruns and the concurrency group (`cancel-in-progress: false`) blocks
  the next scheduled tick rather than queuing it. Not urgent — the credit budget has
  headroom either way — but explains why n is building slower than the "3-4 weeks to
  a first look" estimate from 09-12 assumed.
- **The two well-powered checks both say no detectable edge:**
  - Brier, book vs Polymarket mid, clustered per game (n=467 games / 957 obs):
    diff **+0.0000173**, t=0.044, 95% CI [-0.00075, +0.00078]. Dead center on zero —
    sportsbooks are not measurably sharper than Polymarket here.
  - Strategy at the **0.00** edge threshold (bet the better side whenever there is
    any daylight at all) — the best-powered strategy bucket, n=345 clustered games:
    mean **-0.8%/bet**, t=-0.09, 95% CI [-19.0%, +17.3%]. Also centered on zero, and
    the CI already rules out anything larger than ~19%/bet in either direction.
  - Separately, raw calibration on all 957 resolved, no edge filter: book-implied
    win probability for the modelled side averaged 0.340 against a realised win rate
    of 0.339 — matches to a tenth of a point.
- **The report's headline bucket (edge ≥2pp, n=23 games) shows +51.9%/trade — this is
  one outlier, not a signal.** One bet (Oregon Ducks, entry 6c, "against" the 92%
  favourite, won) returned +1567% and alone accounts for the entire positive mean.
  Drop that one row and the other 22 average **-17.0%**. MDE at n=23 is ±210%/bet, so
  the bucket cannot currently distinguish a real edge from noise regardless of sign;
  do not read anything into it in either direction yet. Same shape one bucket down
  (0-2pp, n=495 individual bets): another single-bet outlier at +1011%, mean still
  -1.1% overall.
- Read together: no edge has shown up wherever there is enough data to tell, and the
  one place that looks exciting is exactly the place with the least data. Correct
  verdict per the tool's own gate is `INCONCLUSIVE`; my read leans toward "heading to
  NEGATIVE/no-edge" rather than "just needs more data," but the >=2pp bucket itself
  is nowhere near powered enough yet to call either way.
- **Next:** keep collecting, untouched. Re-check after another ~4-6 weeks — at the
  current ~2.3 qualifying (>=2pp) games/10 days, MDE only drops to a merely-usable
  ~50-70%/bet range around n=150-200, i.e. several more months, unless a busier sports
  window changes the arrival rate. If someone wants a faster read, the threshold=0.00
  bucket is already reasonably powered and is the one to trust sooner. Worth a look:
  why cadence is 4.8/day not ~8/day (cycle overlap vs the concurrency group).

## 2026-09-12 — Windows PC (divergence study: sportsbooks vs Polymarket)

- Built an **observation-only** study of the one defensible thesis left in the repo:
  do de-vigged sportsbook lines (Odds API) disagree with Polymarket on the *same*
  game, and when they do, does the result go the book's way often enough to beat
  Polymarket's **executable** price? `scripts/divergence_study.py collect|resolve|report`,
  pure logic in `src/divergence.py`, CI steps right after the niche scanner. Writes
  `data/divergence_log.jsonl`, `divergence_state.json`, `divergence_report.json`.
- **Why niche_scanner's sports half never produced a comparison in 5 months:** it asked
  for `markets=h2h,outrights`, which the Odds API rejects for game sports —
  HTTP 422 `INVALID_MARKET_COMBO`, costs 0 credits, swallowed by
  `if status != 200: continue`. The CI key is fine. **Removed rather than repaired**:
  a naive fix is 9 sports × 2 regions × ~20 cycles/day ≈ 360 credits/day and would
  drain the 500/month free tier in about a day, starving the study.
- Quota facts, verified via `x-requests-last`: `/sports` and `/events` are **free**;
  `/odds` costs 1 credit per sport key per region and returns every game in that key
  (one credit on NCAAF prices ~40-70 games). The collector polls at most every 3h and
  paces spend evenly to the 1st of next month with a 25-credit reserve (~3/poll now).
  Spent 10 credits this session (8 probing, 2 on a live test written to a scratch dir,
  not the data repo); 490 left.
- Coverage is good: 420 Polymarket match-winner markets paired to book games across
  32 sport keys in a 48h window (NCAAF, EFL, EPL, Bundesliga, J-League, NFL, MMA…).
  Tennis is thin: the API lists only Grand Slam draws, and Gamma's ATP tag currently
  holds only outrights.
- Method guards carried over from the audit: a missing price is `None`, never 0.5;
  divergence is measured against bid/ask, not mid; one closing snapshot per market;
  stats clustered per game (a soccer match's three legs are one observation); entry
  prices limited to 0.05–0.95; no verdict below 30 resolved markets.
- **First live snapshot** (1 poll, EFL League One/Two, 69 pairs, 23 games): median
  |divergence| 0.5pp, p90 1.4pp, **zero above 2pp**. Polymarket prices favourites
  ~1-2pp *above* the books, leaving ~1pp executable on the other side. One poll proves
  nothing; this is only a first look.
- niche_scanner crypto had two bugs, both fixed with tests: (1) returned 0.5 when
  Binance volatility was unavailable — Binance blocks US IPs, so that was every CI run —
  now returns None and skips; **CI should now emit zero crypto signals instead of 13
  fake ones**. (2) the greedy target regex captured the *last digit* in the question
  ("above $84,000 on September 12" → a $2 target → "99% edge"); now
  `parse_crypto_target`. **Still wrong and deliberately not fixed** (out of scope):
  time to expiry is floored at one whole day, and "reach/hit/dip" are touch questions
  priced with a terminal-price model. Run locally it still prints 11-31pp "edges" —
  do not trust `niche_signals.json`; nothing in this repo reads it.
- Tests 219 → 310.
- **Next:** let it collect 3-4 weeks, then `python scripts/divergence_study.py report`
  (also in the CI log and `data/divergence_report.json`). ≥30 resolved markets for any
  verdict; realistically a few hundred games before the MDE drops under ~5%.
  `divergence_log.jsonl` grows ~0.5-1 MB/day — add it to `rotate_ledger.py` within
  ~3 months. The engine 0.5 fix (lines 499/823/999) is still not done.

## 2026-09-10 — Windows PC (full repo audit: the model is worse than the market)

- Re-ran the 09-06 MacBook audit end-to-end on **fresh data**. It holds, but the key
  sign is the other way round: **against an independent baseline the model is
  significantly WORSE than the market**, not −0.0015 better.
- **Local `data/` was stale and I nearly published numbers off it.** It has no `.git`
  — it is the abandoned pre-migration tracked directory, ~4 weeks behind (V6-TIER
  n=27 locally vs n=102 real). Cloned `poly_bot-data` to a scratch dir and redid
  everything. Anyone auditing must clone the data repo first; `git pull` in this
  repo does *not* refresh `data/`.
- **Why the MacBook's "−0.0015 model better" was wrong in direction:** it scored the
  model against the ledger's own `market_price` column. Scored against
  `price_history.json` snapshots instead, on records where both exist:

  | snapshot window | n | uniq mkts | model Brier | market Brier | diff | t |
  |---|---|---|---|---|---|---|
  | ±2h  | 1253 | 711  | 0.1941 | 0.1763 | **+0.0178** | +5.59 |
  | ±6h  | 1618 | 899  | 0.1920 | 0.1718 | **+0.0202** | +6.90 |
  | ±24h | 2015 | 1055 | 0.1912 | 0.1679 | **+0.0233** | +8.35 |

  Positive = model worse. Dedup to one observation per market makes it worse
  (+0.0265 to +0.0352, t=+5.3 to +7.9), so it is not a clustering artifact. Every
  window agrees. This matches `rolling_accuracy.json` (April, n=5,211, real
  `pre_price`): ensemble 0.1672 vs market 0.1670 — that honest backtest was right
  all along, and live is now measurably *negative*.
- Ledger `market_price` sanity-checked against snapshots: bug-priced rows mean
  |diff| 0.186 / 5.8% agree; real-priced rows mean |diff| 0.006 / **95.3% agree**.
  So the real-priced *prices* are fine — it is using that column as the *baseline*
  that flatters the model.
- **Contamination, fresh data, 907 closed trades — 858 bug-priced (94.6%):**
  V3a/V3b/V3c/V5/V5-TIER/Test0-TIER **100%**, V6-TIER 99%, V6 98%, Test 0 91%.
  v2_ledger 83% of rows at 0.5, live_predictions 63%. V4-AI is 3.7% — it traded
  real Yes/No markets and is the accidental clean control.
- **Re-priced at real snapshots, no arm has an edge.** Booked +50%..+81%/trade
  collapses to −7%..+9%, and **every 95% CI contains zero**: Test 0 +3.7% (t=0.70),
  Test0-TIER +0.6% (0.09), V3a −5.0% (−0.85), V3b +9.2% (1.31), V3c −4.4% (−0.79),
  V5 +3.5% (0.46), V5-TIER +8.2% (1.32), V6 −7.3% (−1.37), **V6-TIER −7.1%
  (−1.14)**, V4-AI −12.4% (−0.75).
- Tried and **rejected**: "the Brier edge converts to money". Betting the model's
  side at |edge|>0.01 on real prices appeared to give +29.8%/trade, t=+16.6, with
  realised WR 73.8% vs market-implied 55.9%. A +17.9pp gap over 3,348 markets is
  not credible and is arithmetically inconsistent with a 0.0037 Brier gap — it is
  an artifact of scoring against the contaminated `market_price` column. Do not
  resurrect this test in that form.
- **Trained artifacts are fitted to a coin flip.** `v2_train_report.json`
  market_brier 0.2337 and `meta_model.xgb.info.json` market_brier 0.2388 — a
  constant 0.5 scores 0.25, and the honest market is ~0.17. So the meta model's
  val_brier 0.2301 is ~0.06 *worse* than the market while reporting
  "improvement +0.0087". `model_weights.json` (ai_semantic 0.429 / microstructure
  0.446) is fitted on those same comparisons.
- Bug provenance: lines 499 / 823 / 999 have had `outcome_prices.get("Yes", 0.5)`
  since `2467dcf8`, the **first commit** of `prediction_engine.py`. Not a
  regression. **Zero tests reference `outcome_prices`** — nothing ever guarded it.
- Sound and worth keeping: the CI/rotation/`ledger_reader`/`data_validator`/tier-cap
  plumbing, `paper_trader.py:424-430`'s correct first-outcome fallback, the
  `pre_price` honest-backtest methodology, and V4-AI's honest −19.8%.
- **Next:** do not deploy capital; the VPS/live plan is shelved, not scheduled. Fix
  order if resumed: (1) first-outcome fallback at engine 499/823/999, (2) a test
  that fails on a no-`Yes` market, (3) reset or hard-label every arm ledger,
  (4) re-accumulate and re-measure against snapshots — not against
  `market_price`. Expect the honest answer to stay "no alpha".

## 2026-09-07 — Mobile (3-device worklog sync)

- Set the cross-device protocol up for **three** devices — Windows PC, MacBook and
  **Mobile** (this one, Claude on the web). It had only ever named two.
- Added **`CLAUDE.md`** carrying the protocol. This was the actual gap: the convention
  only lived inside `WORKLOG.md`, so a session that had not already been told to read
  the worklog never read it — which is why context kept being re-explained by hand and
  work got repeated. `CLAUDE.md` is loaded automatically on every device, so
  "pull → read WORKLOG → don't redo → append + push" now happens unprompted.
- Normalised this file's header to the wording shared by all 12 repos, naming the three
  device labels `brain` parses.
- Verified this repo was already exactly at `origin` before the change: clean tree,
  nothing ahead or behind.
- Protocol added to the **existing** `CLAUDE.md` (above `## Architecture`) rather than
  replacing it — the engine/architecture notes there are unchanged.
- **Documented that `data/` is a separate private repo**, which `CLAUDE.md` did not say.
  It is git-ignored here and CI clones `sathvik-lv/poly_bot-data` into it, so a fresh
  clone of poly_bot alone has no `data/` and every script reading it fails. Both the SSH
  and HTTPS clone lines are recorded — the MacBook has no SSH key and needs HTTPS.
- **Next:** nothing open here. The protocol applies from the next session on any device.

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
- Dashboards updated to match (see that repo's worklog for detail): V6-Tier and
  Test 0-Tier were missing from its arm list entirely, and its poly page now
  carries a re-settled-at-real-prices section beside every booked ROI. Its
  refresh workflow could also silently overwrite good bundles with zeros when the
  private data checkout failed — the code repo is public so the root still looked
  valid; guarded now.
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
