# Poly Bot — Polymarket Prediction Engine

## What This Is
Automated prediction engine for Polymarket. Scans markets, runs ensemble predictions, paper trades with simulated $10k, tracks accuracy, and collects data for weight allocation.

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
- **Paper mode only** — no real trades until weight analytics prove edge with 50+ resolved trades.
- **Short-dated market targeting** — scanner fetches markets resolving within 14 days for faster feedback.
- **AI model rotation** — 25 free OpenRouter models, tries up to 8 per prediction, auto-skips on rate limit.

## Data Files (`data/`)
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
- Backtest Brier (honest, n=5211 resolved markets, `data/rolling_accuracy.json`):
  - Ensemble: **0.1672**
  - Market baseline: **0.1670**
  - Edge vs market: **−0.0002** (ensemble matches market, no measurable alpha)
  - Previous "0.063" claim was wrong — did not reflect the real rolling backtest
- Backtest coverage per model:
  - `microstructure` — backtested on 5,211 markets (Brier 0.1677)
  - `external_data` — backtested on 5,211 markets (Brier 0.1670)
  - `orderbook` — effectively untested (only 3 markets; no free historical bid/ask depth source)
  - `time_series` — **not backtested** (needs historical price bars; `/prices-history` backfill not yet wired)
  - `ai_semantic` — **not backtested** (LLM training-cutoff leakage makes honest backtest impossible; forward-test only)
- 2 open paper trades (geopolitics: Iran, Taiwan)
- Waiting for market resolutions to build weight analytics
- Metaculus returns questions but community predictions are restricted (need Bot Benchmarking Access Tier)
- Manifold cross-platform matching works (13 real matches found)
- Kalshi not accessible in user's region
