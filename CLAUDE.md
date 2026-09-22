# Poly Bot — Polymarket Prediction Engine

## What This Is
Automated prediction engine for Polymarket. Scans markets, runs ensemble predictions, paper trades with simulated $10k, tracks accuracy, and collects data for weight allocation.

## Cross-device sync — read this first

This repo is worked on from **three devices**: **Windows PC**, **MacBook**, and
**Mobile** (Claude on the web/phone). They share no chat history, no memory and no
disk. `WORKLOG.md` at the repo root is the *only* thing that carries context between
them, so it is treated as part of the code, not as a diary.

**Start of every session, before touching anything:**

1. `git pull` — never begin from a stale checkout.
2. Read `WORKLOG.md` from the top until you have the current state. Newest entries
   are first.
3. Treat it as authoritative. **Work recorded there is done** — re-deriving it,
   re-running it or "just double-checking" it is waste, not diligence. Decisions
   recorded as rejected stay rejected unless there is a stated reason to revisit.
4. Read the newest entry's **Next** line before planning: it names work already
   claimed or in flight on another device.

**End of every session that changed anything** — append a new entry at the *top* of
`WORKLOG.md`, under the header, in this shape:

```markdown
## YYYY-MM-DD — <Windows PC | MacBook | Mobile> (short topic)

- What changed, and the reasoning behind anything non-obvious.
- What was tried and rejected, and why — so the next device does not retry it.
- **Next:** the concrete next step, or `nothing open`.
```

then **commit and push**. An entry that is not pushed does not exist to the other two
devices — that is the single most common way work gets repeated.

Use exactly one of `Windows PC`, `MacBook` or `Mobile` as the device label. The
`brain` repo parses that field to build the cross-repo timeline and open-threads
view, and an unrecognised label drops the entry into `unknown`.

**Ask before re-doing.** If something looks unfinished but the worklog says it was
completed, ask rather than redo — the other device may simply not have pushed yet.

## Architecture

### Core Engine (`src/`)
- `prediction_engine.py` — Main engine. Runs 5 sub-models, combines via inverse-variance weighted ensemble:
  - **microstructure** — volume/liquidity signals
  - **time_series** — price history trends (needs 10+ snapshots to activate)
  - **external_data** — Fear & Greed, metadata signals (no mean-reversion)
  - **orderbook** — CLOB order flow, anchored on market price (max ±5% adjustment)
  - **ai_semantic** — Rotates through 25 free OpenRouter models with auto-fallback on 429s
- `strategy_adapter.py` — Entry gates + 1/3 Kelly sizing + regime-based allocation
- `market_client.py` — Polymarket Gamma API wrapper
- `clob_client.py` — CLOB orderbook data
- `data_sources.py` — External data (Fear & Greed, etc.)
- `news_rag.py` — News context for AI model

### Scripts (`scripts/`)
- `paper_trader.py` — Paper trading ($10k simulated). Scan, resolve, report with weight analytics
- `live_tracker.py` — Records predictions, checks accuracy when markets resolve
- `finance_scanner.py` — Fed/oil/BTC/macro market analysis with quant models
- `arbitrage_scanner.py` — Internal Polymarket arbitrage (multi-outcome sum check)
- `cross_platform.py` — Compares odds across Manifold, Metaculus, PredictIt
- `price_snapshots.py` — Captures prices every 2h for time series
- `daily_cycle.py` — Runs everything in sequence
- `autostart.py` — Windows startup automation (disabled — using GitHub Actions now)

### Automation
GitHub Actions runs 24/7:
- `.github/workflows/cycle.yml` — Full cycle every 6 hours
- `.github/workflows/snapshot.yml` — Price snapshots every 2 hours
- Secrets: `OPENROUTER_API_KEY`, `METACULUS_API_TOKEN`, `RELAYER_API_KEY`, `RELAYER_API_KEY_ADDRESS`

## Key Design Decisions
- **Calibrator is DISABLED** — was overfitting (trained on same data it predicted on). Raw ensemble is more honest.
- **Orderbook anchors on market price** — extreme limit orders (bids at 0.001) produce garbage VWAP. Only extracts directional signal ±5%.
- **No mean-reversion signals** — pulling toward 0.5 was destroying edge. External data only adds uncertainty, no directional pull.
- **No real trades. The edge question is closed, not pending** — see Current Status. The
  old "50+ resolved trades" bar was statistically meaningless: per-trade sd is ~57%, so
  n=50 can only detect an edge of ≥23%/trade. P&L is the wrong yardstick at these sizes;
  use paired Brier against real snapshot prices.
- **Short-dated market targeting** — scanner fetches markets resolving within 14 days for faster feedback.
- **AI model rotation** — 25 free OpenRouter models, tries up to 8 per prediction, auto-skips on rate limit.

## Data Files (`data/`)

**`data/` is a separate private repo, not part of this checkout.** This repo is public,
so the ledgers moved to `sathvik-lv/poly_bot-data`; `data/` is git-ignored here and CI
clones it in (see `.github/workflows/cycle.yml`). A fresh clone of poly_bot alone has
no `data/` and every script that reads it will fail until you add it:

```bash
git clone git@github.com:sathvik-lv/poly_bot-data.git data      # SSH
git clone https://github.com/sathvik-lv/poly_bot-data.git data  # if no SSH key on this device
```

- `paper_trades.json` — Open/closed positions with entry price, model estimates, category
- `weight_analytics.json` — Model P&L, category P&L, edge calibration (generated after trades resolve)
- `price_history.json` — Time series snapshots per market
- `live_predictions.jsonl` — Prediction log with per-model breakdown
- `arbitrage_scan.json` — Latest arbitrage scan results
- `cross_platform.json` — Cross-platform odds comparison

## Running Locally
```bash
python scripts/paper_trader.py --scan -n 30   # scan + paper trade
python scripts/paper_trader.py --resolve       # check resolutions
python scripts/paper_trader.py --report        # P&L + weight analytics
python scripts/daily_cycle.py                  # full cycle once
python scripts/daily_cycle.py --loop           # repeat every 6h
```

## Environment Variables (`.env`)
```
OPENROUTER_API_KEY=...     # AI model access (25 free models)
METACULUS_API_TOKEN=...    # Metaculus predictions (needs Bot Benchmarking tier for full access)
RELAYER_API_KEY=...        # Polymarket relayer (for future real trades)
RELAYER_API_KEY_ADDRESS=...
```

## Current Status

**There is no measured edge. This is settled, not open.** Three independent
well-powered measurements agree, so do not re-open it without a genuinely new
information source — a better ensemble over the same five sub-models will not change it.

1. `data/rolling_accuracy.json` (April, n=5,211, real `pre_price`, base rate 24.5%):
   ensemble Brier **0.1672** vs market **0.1670** → edge **−0.0002**. No alpha.
2. Live predictions re-scored against independent `price_history.json` snapshots
   (2026-09, n=1,055 unique markets): model **+0.0233 WORSE** than market, t=**+8.35**,
   consistent at ±2h/±6h/±24h and worse when deduped per market.
3. Sportsbook-vs-Polymarket divergence study (2026-09, n=957 resolved): paired Brier
   diff **+0.0000** (t=0.04) and the best-powered strategy bucket **−0.8%/bet**
   (t=−0.09, n=345 games). Still collecting; `INCONCLUSIVE` only in the thin ≥2pp bucket.

**Paper-trading arms were removed from CI on 2026-09-22** (Test 0, Test 0-TIER,
V3a/b/c, V5, V5-TIER, V6, V6-TIER, the V2 block, `compare_arms`). No data was deleted
and every script remains unit-tested. Reason: 94.6% of the 907 closed trades were
booked at a **fabricated $0.500** entry price.

### Known fabrications — read before trusting ANY stored metric
- **`outcome_prices.get("Yes", 0.5)`** at `prediction_engine.py` **499 / 823 / 999**.
  Markets whose outcomes are competitor names (most sports) have no `"Yes"` key, so the
  engine silently prices them at 0.5 and winners appear to pay ~2x. **Not yet fixed.**
  The same pattern is at ~20 sites — `meta_model.py:75,85-88` (the whole feature
  vector), `strategy_adapter.py:320,912`, `exit_simulator.py:104`,
  `self_improver.py:228,251`, `paper_trader.py:907`. `paper_trader.py:424-430` is the
  one place it was written correctly.
- **`data/backtest_honest_preds.jsonl` is unusable**: YES base rate **1.68%** (65 of
  3,879) from a selection-biased filter. A constant 0.23 predictor scores Brier 0.0619
  on it — which is the origin of the **`brier=0.063` in 641 commit messages**. Only
  `v2_train_meta_full.py` reads it, and that is not in CI.
- **`training_runner._simulate_resolutions()`** invents outcomes from price moves and
  feeds them to the self-audit. **Dead code — keep it dead.**
- `v2_train_report.json` / `meta_model.xgb.info.json` "market_brier" ≈ 0.234/0.239 are
  measured against those 0.5 prices, i.e. against a coin flip (0.25), not the market.
- `hurst_exponent` is fed price levels instead of returns
  (`prediction_engine.py:197` vs `202`), so H≈1.0 always. Inert: diagnostic label only,
  and `time_series` weight is 0.0.

### What is verified-correct and worth keeping
- **`src/statistics.py`** — GARCH(1,1) recovers its true parameters to 4 dp, GaussianHMM
  recovers regime means, `kelly_with_uncertainty` gives exactly full-Kelly 0.20 for
  p=0.6 at even odds with sane shrinkage, Beta-Binomial and IsotonicCalibrator are
  correct, KL/JS are 0 for identical inputs. Deterministic as claimed. **Note:
  `BetaBinomialModel.update()` is immutable — it returns a new model; `b.update(...)`
  alone is a no-op and is NOT a bug.** Wilson CI in `category_gate.py`/`compare_arms.py`
  is also correct as written — do not "fix" it against a Clopper-Pearson reference.
- `rolling_backtest.py` — the only honest backtest apparatus.
- `price_snapshots.py` — real observed data, no model in the loop. This is what made
  every honest re-pricing possible.
- Test suite: 310 tests / 434 assertions, no skips, no tautologies.

### Environment notes
- Metaculus community predictions restricted (needs Bot Benchmarking Access Tier)
- Manifold cross-platform matching works (13 real matches found)
- Kalshi not accessible in user's region
- Binance blocks GitHub's US runner IPs, so BTC volatility is unavailable in CI
- `RELAYER_API_KEY` is read by **no code** — there is no execution layer
