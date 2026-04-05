"""Historical Data Downloader, Backtester & Training Pipeline.

Downloads resolved Polymarket markets, trains the prediction engine,
runs self-improvement, detects regression, rolls back if needed.

Every batch result is logged and committed to git.

Usage:
    python -m src.historical_trainer                # full pipeline
    python -m src.historical_trainer --download     # download only
    python -m src.historical_trainer --train        # train on existing data
    python -m src.historical_trainer --report       # show progression report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_client import MarketClient
from src.prediction_engine import PredictionEngine
from src.self_improver import (
    PredictionTracker,
    CalibrationAuditor,
    ModelWeightOptimizer,
    SignalStrengthAnalyzer,
    SelfCorrector,
)
from src.safeguard import Safeguard, ApproachManager
from src.strategy_adapter import StrategyAdapter

DATA_DIR = str(PROJECT_ROOT / "data")
HISTORICAL_FILE = os.path.join(DATA_DIR, "historical_markets.jsonl")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_predictions.jsonl")
TRAIN_CHECKPOINT = os.path.join(DATA_DIR, "train_checkpoint.json")


# ===========================================================================
# Historical Downloader
# ===========================================================================

class HistoricalDownloader:
    """Download resolved markets from Polymarket Gamma API."""

    def __init__(self):
        self.client = MarketClient()
        os.makedirs(DATA_DIR, exist_ok=True)

    def download_all(self, max_markets: int = 2000) -> int:
        """Download resolved markets + events. Git commit after each batch."""
        total = 0
        total += self._download_markets(max_markets=max_markets)
        total += self._download_events(max_events=100)
        return total

    def _download_markets(self, max_markets: int = 2000, batch_size: int = 100) -> int:
        existing_ids = self._load_existing_ids()
        print(f"  [{len(existing_ids)} existing markets in cache]")

        total = 0
        offset = 0
        empty_streak = 0

        while total < max_markets and empty_streak < 5:
            try:
                raw = self.client.session.get(
                    f"{self.client.base_url}/markets",
                    params={
                        "limit": batch_size, "closed": True,
                        "order": "volume", "ascending": False, "offset": offset,
                    },
                    timeout=30,
                ).json()
            except Exception as e:
                print(f"    Fetch error at offset {offset}: {e}")
                time.sleep(3)
                offset += batch_size
                empty_streak += 1
                continue

            if not raw:
                empty_streak += 1
                offset += batch_size
                continue

            batch = 0
            for m in raw:
                parsed = MarketClient.parse_market(m)
                if parsed.get("id") in existing_ids:
                    continue
                outcome = self._determine_outcome(parsed)
                if outcome is None:
                    continue
                self._save_market(parsed, outcome)
                existing_ids.add(parsed["id"])
                batch += 1
                total += 1

            empty_streak = 0 if batch > 0 else empty_streak + 1
            offset += batch_size
            print(f"    Markets batch: +{batch} (total: {total}, offset: {offset})")

            # Git commit every 500 markets
            if total > 0 and total % 500 == 0:
                self._git_commit(f"data: downloaded {total} historical markets")

            time.sleep(0.3)

        if total > 0:
            self._git_commit(f"data: downloaded {total} historical markets (complete)")
        print(f"  Markets download: {total} new")
        return total

    def _download_events(self, max_events: int = 100) -> int:
        existing_ids = self._load_existing_ids()
        total = 0
        offset = 0

        while offset < max_events * 20:  # rough bound
            try:
                events = self.client.session.get(
                    f"{self.client.base_url}/events",
                    params={
                        "limit": 20, "closed": True,
                        "order": "volume", "ascending": False, "offset": offset,
                    },
                    timeout=30,
                ).json()
            except Exception:
                break

            if not events:
                break

            for event in events:
                for m in event.get("markets", []):
                    parsed = MarketClient.parse_market(m)
                    if parsed.get("id") in existing_ids:
                        continue
                    outcome = self._determine_outcome(parsed)
                    if outcome is None:
                        continue
                    record = parsed.copy()
                    record["event_title"] = event.get("title", "")
                    self._save_market(record, outcome)
                    existing_ids.add(parsed["id"])
                    total += 1

            offset += 20
            print(f"    Events batch (total markets from events: {total})")
            time.sleep(0.3)

        if total > 0:
            self._git_commit(f"data: downloaded {total} markets from events")
        print(f"  Events download: {total} new")
        return total

    def count_historical(self) -> int:
        return len(self._load_existing_ids())

    def _load_existing_ids(self) -> set:
        ids = set()
        if os.path.exists(HISTORICAL_FILE):
            with open(HISTORICAL_FILE) as f:
                for line in f:
                    if line.strip():
                        ids.add(json.loads(line).get("id"))
        return ids

    def _save_market(self, parsed: dict, outcome: float):
        record = {
            "id": parsed.get("id"),
            "question": parsed.get("question", ""),
            "outcomes": parsed.get("outcomes", []),
            "outcome_prices": parsed.get("outcome_prices", {}),
            "volume": parsed.get("volume"),
            "liquidity": parsed.get("liquidity"),
            "last_trade_price": parsed.get("last_trade_price"),
            "best_bid": parsed.get("best_bid"),
            "best_ask": parsed.get("best_ask"),
            "spread": parsed.get("spread"),
            "end_date": parsed.get("end_date"),
            "event_title": parsed.get("event_title", ""),
            "resolved_outcome": outcome,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(HISTORICAL_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _determine_outcome(market: dict):
        prices = market.get("outcome_prices", {})
        yes_price = prices.get("Yes")
        no_price = prices.get("No")
        if yes_price is not None and no_price is not None:
            if yes_price >= 0.95:
                return 1.0
            if no_price >= 0.95:
                return 0.0
        for name, price in prices.items():
            if price is not None and price >= 0.95:
                outcomes = market.get("outcomes", [])
                return 1.0 if outcomes and name == outcomes[0] else 0.0
        return None

    @staticmethod
    def _git_commit(msg: str):
        try:
            subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
            subprocess.run(["git", "commit", "-m", msg], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
        except Exception:
            pass


# ===========================================================================
# Backtester with Safeguard Integration
# ===========================================================================

class Backtester:
    """Backtest prediction engine on historical data with safeguard checks."""

    def __init__(self, equity: float = 10000.0):
        # Load calibration data if available — critical for accurate predictions
        cal_data = None
        cal_path = os.path.join(DATA_DIR, "calibration.json")
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    cal_json = json.load(f)
                if "x" in cal_json and "y" in cal_json:
                    cal_data = (np.array(cal_json["x"]), np.array(cal_json["y"]))
                    print(f"  Loaded calibration data: {len(cal_json['x'])} points")
            except Exception:
                pass

        self.engine = PredictionEngine(total_equity=equity, calibration_data=cal_data, backtest_mode=True)
        self.strategy = self.engine.strategy  # Calendar spread adapter
        self.auditor = CalibrationAuditor()
        self.safeguard = Safeguard()
        self.approach_mgr = ApproachManager()
        self.equity = equity
        os.makedirs(DATA_DIR, exist_ok=True)

    def load_historical(self) -> list[dict]:
        if not os.path.exists(HISTORICAL_FILE):
            return []
        markets = []
        with open(HISTORICAL_FILE) as f:
            for line in f:
                if line.strip():
                    markets.append(json.loads(line))
        return markets

    def run_backtest(self, batch_size: int = 50) -> dict:
        """Backtest in batches with safeguard checks between batches.

        After each batch:
        1. Audit current performance
        2. Snapshot state
        3. Check for regression -> rollback if needed
        4. Git commit the batch results
        5. Continue or switch approach
        """
        # Resume from checkpoint
        checkpoint = self._load_checkpoint()
        start_idx = checkpoint.get("last_idx", 0)
        current_approach = checkpoint.get("approach", self.approach_mgr.current_approach)

        # Clear backtest file if starting fresh
        if start_idx == 0 and os.path.exists(BACKTEST_FILE):
            os.remove(BACKTEST_FILE)

        tracker = PredictionTracker(BACKTEST_FILE)
        markets = self.load_historical()
        if not markets:
            return {"error": "No historical data"}

        total = len(markets)
        print(f"\n  Backtesting {total} markets in batches of {batch_size}")
        print(f"  Starting from idx {start_idx}, approach: {current_approach}")

        batch_num = start_idx // batch_size
        overall_preds = 0
        overall_errors = 0
        last_brier = "N/A"

        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_num += 1
            batch_preds = 0
            batch_errors = 0

            print(f"\n  --- Batch {batch_num} [{batch_start}-{batch_end}] approach={current_approach} ---")

            for i in range(batch_start, batch_end):
                market = markets[i]
                try:
                    outcome = market["resolved_outcome"]

                    # Use ACTUAL stored market price — no fabricated data
                    stored_price = market.get("last_trade_price")
                    yes_price = market.get("outcome_prices", {}).get("Yes")

                    # HONEST FILTER: Skip markets with post-resolution prices
                    # Prices near 0 or 1 mean the market already resolved — predicting
                    # these is cheating because the answer is in the price.
                    price_to_use = stored_price if stored_price else yes_price
                    if price_to_use is None:
                        continue  # No price data at all — skip
                    price_to_use = float(price_to_use)
                    if price_to_use < 0.05 or price_to_use > 0.95:
                        continue  # Post-resolution price — skip (would inflate metrics)

                    sim_price = price_to_use
                    stored_spread = market.get("spread") or 0.02
                    stored_bid = market.get("best_bid") or (sim_price - stored_spread / 2)
                    stored_ask = market.get("best_ask") or (sim_price + stored_spread / 2)

                    market_data = {
                        "id": market.get("id"),
                        "question": market.get("question", ""),
                        "outcomes": market.get("outcomes", ["Yes", "No"]),
                        "outcome_prices": {"Yes": sim_price, "No": 1 - sim_price},
                        "volume": market.get("volume"),
                        "volume_24h": market.get("volume"),
                        "liquidity": market.get("liquidity"),
                        "last_trade_price": sim_price,
                        "best_bid": stored_bid,
                        "best_ask": stored_ask,
                        "spread": stored_spread,
                        "end_date": market.get("end_date"),
                        "active": True, "closed": False,
                    }

                    prediction = self.engine.predict(market_data=market_data, time_remaining_frac=0.5)
                    tracker.record_prediction(prediction)
                    tracker.record_outcome(market.get("id"), outcome)
                    batch_preds += 1
                except Exception:
                    batch_errors += 1

            overall_preds += batch_preds
            overall_errors += batch_errors

            # Audit after batch
            resolved = tracker.get_resolved_predictions()
            if len(resolved) >= 10:
                audit = self.auditor.full_audit(resolved)
                brier = audit.get("brier_score", 1.0)
                ece = audit.get("expected_calibration_error", 1.0)
                disc = audit.get("discrimination", 0.5)
                roi = audit.get("roi")
                ehr = audit.get("edge_hit_rate")

                last_brier = f"{brier:.4f}"
                roi_str = f"{roi:.1%}" if roi is not None else "N/A"
                ehr_str = f"{ehr:.1%}" if ehr is not None else "N/A"
                print(f"    Preds: {batch_preds} | Brier: {brier:.4f} | ECE: {ece:.4f} | "
                      f"Disc: {disc:.4f} | ROI: {roi_str} | EdgeHit: {ehr_str}")

                # Snapshot
                weights = {}
                weights_path = os.path.join(DATA_DIR, "model_weights.json")
                if os.path.exists(weights_path):
                    with open(weights_path) as f:
                        weights = json.load(f)

                snap = self.safeguard.take_snapshot(
                    cycle=batch_num, audit=audit, weights=weights, approach=current_approach
                )
                self.approach_mgr.record_score(current_approach, snap.composite_score)

                # Safeguard check
                protection = self.safeguard.check_and_protect(batch_num, audit)
                if protection["action"] == "ROLLBACK":
                    print(f"    ** ROLLBACK: {protection['details']}")
                    current_approach = protection["new_approach"]
                    print(f"    ** Switching to approach: {current_approach}")
                elif protection["action"] == "SWITCH_APPROACH":
                    print(f"    ** DEGRADING: {protection['details']}")

                # Self-improver DISABLED — it overwrites calibration/weights
                # with data from broken models. We tune manually until models are solid.
                # if batch_num % 5 == 0 and len(resolved) >= 20:
                #     corrector = SelfCorrector(DATA_DIR)
                #     corrector.tracker = tracker
                #     improvement = corrector.run_improvement_cycle()

            # Checkpoint
            self._save_checkpoint({"last_idx": batch_end, "approach": current_approach, "batch": batch_num})

            # Git commit batch results
            self._git_commit(
                f"backtest: batch {batch_num} | preds={overall_preds} errors={overall_errors} | "
                f"brier={last_brier} approach={current_approach}"
            )

        # Final audit
        resolved = tracker.get_resolved_predictions()
        final_audit = self.auditor.full_audit(resolved) if resolved else {}
        self._save_checkpoint({"last_idx": total, "completed": True, "approach": current_approach})

        return {
            "n_predictions": overall_preds,
            "n_errors": overall_errors,
            "n_batches": batch_num,
            "final_approach": current_approach,
            "audit": final_audit,
        }

    def run_strategy_backtest(self, batch_size: int = 50) -> dict:
        """Backtest with strategy adapter — compares baseline vs strategy-filtered.

        Runs every market through the prediction engine, then applies
        the calendar spread strategy adapter (entry gates, 1/3 Kelly,
        regime allocation) to see how filtering and sizing affect ROI.

        Returns A/B comparison: baseline (bet everything) vs strategy (filtered).
        """
        markets = self.load_historical()
        if not markets:
            return {"error": "No historical data"}

        total = len(markets)
        print(f"\n{'=' * 70}")
        print(f"  STRATEGY BACKTEST: Baseline vs Calendar-Spread-Adapted")
        print(f"  Markets: {total} | Equity: ${self.equity:,.0f}")
        print(f"  Strategy: 1/3 Kelly | 60% max deploy | 7 entry filters")
        print(f"{'=' * 70}")

        # Track both approaches
        baseline = {"trades": 0, "wins": 0, "pnl": 0.0, "bets": []}
        strategy = {"trades": 0, "wins": 0, "pnl": 0.0, "bets": [],
                     "filtered_out": 0, "filter_reasons": {}}

        equity_baseline = self.equity
        equity_strategy = self.equity
        equity_curve_baseline = [self.equity]
        equity_curve_strategy = [self.equity]

        batch_num = 0
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_num += 1

            for i in range(batch_start, batch_end):
                market = markets[i]
                try:
                    outcome = market["resolved_outcome"]

                    # HONEST FILTER: Only use pre-resolution prices
                    stored_price = market.get("last_trade_price")
                    yes_price = market.get("outcome_prices", {}).get("Yes")
                    price_to_use = stored_price if stored_price else yes_price
                    if price_to_use is None:
                        continue
                    price_to_use = float(price_to_use)
                    if price_to_use < 0.05 or price_to_use > 0.95:
                        continue  # Post-resolution — skip

                    sim_price = price_to_use
                    stored_spread = market.get("spread") or 0.02
                    stored_bid = market.get("best_bid") or (sim_price - stored_spread / 2)
                    stored_ask = market.get("best_ask") or (sim_price + stored_spread / 2)

                    market_data = {
                        "id": market.get("id"),
                        "question": market.get("question", ""),
                        "outcomes": market.get("outcomes", ["Yes", "No"]),
                        "outcome_prices": {"Yes": sim_price, "No": 1 - sim_price},
                        "volume": market.get("volume"),
                        "volume_24h": market.get("volume") or 500,
                        "liquidity": market.get("liquidity") or 1000,
                        "last_trade_price": sim_price,
                        "best_bid": stored_bid,
                        "best_ask": stored_ask,
                        "spread": stored_spread,
                        "end_date": market.get("end_date"),
                        "active": True, "closed": False,
                    }

                    prediction = self.engine.predict(market_data=market_data, time_remaining_frac=0.5)
                    edge = prediction["edge"]["edge"]
                    prob = prediction["prediction"]["probability"]
                    kelly = prediction["sizing"]["kelly_fraction"]

                    # === BASELINE: bet on everything with positive edge ===
                    if abs(edge) > 0.005:
                        bet_size_base = min(abs(kelly) * equity_baseline, equity_baseline * 0.25)
                        if bet_size_base > 0:
                            baseline["trades"] += 1
                            if edge > 0:
                                if outcome == 1:
                                    profit = bet_size_base * (1 - sim_price) / sim_price
                                    baseline["wins"] += 1
                                else:
                                    profit = -bet_size_base
                            else:
                                if outcome == 0:
                                    profit = bet_size_base * sim_price / (1 - sim_price)
                                    baseline["wins"] += 1
                                else:
                                    profit = -bet_size_base

                            baseline["pnl"] += profit
                            equity_baseline += profit
                            equity_baseline = max(equity_baseline, 100)  # floor

                    # === STRATEGY: apply calendar spread adapter ===
                    strat_eval = prediction.get("strategy", {})
                    should_enter = strat_eval.get("should_enter", False)

                    if should_enter and abs(edge) > 0.005:
                        sizing = strat_eval.get("sizing_adjusted", {})
                        bet_size_strat = sizing.get("size_dollars", 0)
                        # Scale to current equity
                        size_pct = sizing.get("size_pct", 0)
                        bet_size_strat = size_pct * equity_strategy

                        if bet_size_strat > 0:
                            strategy["trades"] += 1
                            if edge > 0:
                                if outcome == 1:
                                    profit = bet_size_strat * (1 - sim_price) / sim_price
                                    strategy["wins"] += 1
                                else:
                                    profit = -bet_size_strat
                            else:
                                if outcome == 0:
                                    profit = bet_size_strat * sim_price / (1 - sim_price)
                                    strategy["wins"] += 1
                                else:
                                    profit = -bet_size_strat

                            strategy["pnl"] += profit
                            equity_strategy += profit
                            equity_strategy = max(equity_strategy, 100)

                            # Track model outcomes for WR x payoff optimization
                            for model_name in prediction.get("ensemble", {}).get("model_names", []):
                                sub = prediction.get("sub_models", {}).get(model_name, {})
                                if sub.get("estimate") is not None:
                                    self.strategy.record_model_outcome(
                                        model_name, sub["estimate"], sim_price, int(outcome)
                                    )
                    else:
                        strategy["filtered_out"] += 1
                        if not should_enter:
                            failed = strat_eval.get("filters", {}).get("failed_filters", ["unknown"])
                            for f in failed:
                                strategy["filter_reasons"][f] = strategy["filter_reasons"].get(f, 0) + 1

                except Exception:
                    continue

            equity_curve_baseline.append(equity_baseline)
            equity_curve_strategy.append(equity_strategy)

            # Progress
            if batch_num % 20 == 0 or batch_end == total:
                b_wr = baseline["wins"] / max(baseline["trades"], 1) * 100
                s_wr = strategy["wins"] / max(strategy["trades"], 1) * 100
                print(f"  Batch {batch_num:>4} | "
                      f"Base: ${equity_baseline:>10,.0f} ({b_wr:.0f}% WR, {baseline['trades']} trades) | "
                      f"Strat: ${equity_strategy:>10,.0f} ({s_wr:.0f}% WR, {strategy['trades']} trades)")

        # === FINAL REPORT ===
        print(f"\n{'=' * 70}")
        print(f"  STRATEGY BACKTEST RESULTS")
        print(f"{'=' * 70}")

        b_wr = baseline["wins"] / max(baseline["trades"], 1)
        s_wr = strategy["wins"] / max(strategy["trades"], 1)
        b_roi = (equity_baseline - self.equity) / self.equity
        s_roi = (equity_strategy - self.equity) / self.equity

        print(f"\n  {'Metric':<30} {'Baseline':>15} {'Strategy':>15} {'Delta':>12}")
        print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")
        print(f"  {'Starting Equity':<30} {'${:,.0f}'.format(self.equity):>15} {'${:,.0f}'.format(self.equity):>15}")
        print(f"  {'Final Equity':<30} {'${:,.0f}'.format(equity_baseline):>15} {'${:,.0f}'.format(equity_strategy):>15} "
              f"{'${:+,.0f}'.format(equity_strategy - equity_baseline):>12}")
        print(f"  {'Total Return':<30} {b_roi:>14.1%} {s_roi:>14.1%} {s_roi - b_roi:>+11.1%}")
        print(f"  {'Total Trades':<30} {baseline['trades']:>15,} {strategy['trades']:>15,} "
              f"{strategy['trades'] - baseline['trades']:>+12,}")
        print(f"  {'Win Rate':<30} {b_wr:>14.1%} {s_wr:>14.1%} {s_wr - b_wr:>+11.1%}")
        print(f"  {'Filtered Out':<30} {'N/A':>15} {strategy['filtered_out']:>15,}")
        print(f"  {'Avg Bet (Base)':<30} {'${:,.0f}'.format(abs(baseline['pnl']) / max(baseline['trades'], 1)):>15}")
        print(f"  {'Kelly Mode':<30} {'Full Kelly':>15} {'1/3 Kelly':>15}")
        print(f"  {'Max Deploy':<30} {'25% cap':>15} {'60% total cap':>15}")

        # Filter breakdown
        if strategy["filter_reasons"]:
            print(f"\n  Entry Filter Rejection Breakdown:")
            for reason, count in sorted(strategy["filter_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason:<25} {count:>6} ({count/max(strategy['filtered_out'],1)*100:.0f}%)")

        # Max drawdown
        def max_drawdown(curve):
            peak = curve[0]
            max_dd = 0
            for v in curve:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            return max_dd

        b_dd = max_drawdown(equity_curve_baseline)
        s_dd = max_drawdown(equity_curve_strategy)
        print(f"\n  {'Max Drawdown':<30} {b_dd:>14.1%} {s_dd:>14.1%} {s_dd - b_dd:>+11.1%}")

        # Sharpe-like ratio (return / volatility)
        b_returns = [(equity_curve_baseline[i+1] - equity_curve_baseline[i]) / max(equity_curve_baseline[i], 1)
                     for i in range(len(equity_curve_baseline) - 1)]
        s_returns = [(equity_curve_strategy[i+1] - equity_curve_strategy[i]) / max(equity_curve_strategy[i], 1)
                     for i in range(len(equity_curve_strategy) - 1)]

        b_sharpe = np.mean(b_returns) / max(np.std(b_returns), 1e-6)
        s_sharpe = np.mean(s_returns) / max(np.std(s_returns), 1e-6)
        print(f"  {'Sharpe Ratio (batch)':<30} {b_sharpe:>14.3f} {s_sharpe:>14.3f} {s_sharpe - b_sharpe:>+11.3f}")

        # Model performance report from WR x payoff allocator
        model_report = self.strategy.wr_allocator.get_model_report()
        if model_report:
            print(f"\n  Sub-Model Performance (Calendar-Style WR x Payoff Table):")
            print(f"  {'Model':<20} {'Preds':>6} {'WR':>7} {'AvgWin':>8} {'AvgLoss':>8} {'Payoff':>7} {'Kelly%':>7}")
            print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
            for m in model_report:
                print(f"  {m['model']:<20} {m['predictions']:>6} {m['win_rate']:>6.1%} "
                      f"{m['avg_payoff']:>8.4f} {m['avg_risk']:>8.4f} {m['payoff_ratio']:>6.2f}x {m['kelly_pct']:>6.1f}%")

        # Git commit
        self._git_commit(
            f"strategy-backtest: Base ROI={b_roi:.1%} vs Strategy ROI={s_roi:.1%} | "
            f"Base WR={b_wr:.1%} vs Strat WR={s_wr:.1%} | "
            f"Filtered={strategy['filtered_out']}"
        )

        return {
            "baseline": {
                "equity": equity_baseline, "trades": baseline["trades"],
                "win_rate": b_wr, "roi": b_roi, "max_drawdown": b_dd,
                "sharpe": b_sharpe,
            },
            "strategy": {
                "equity": equity_strategy, "trades": strategy["trades"],
                "win_rate": s_wr, "roi": s_roi, "max_drawdown": s_dd,
                "sharpe": s_sharpe, "filtered_out": strategy["filtered_out"],
            },
            "improvement": {
                "roi_delta": s_roi - b_roi,
                "wr_delta": s_wr - b_wr,
                "dd_delta": s_dd - b_dd,
                "sharpe_delta": s_sharpe - b_sharpe,
                "trade_reduction": baseline["trades"] - strategy["trades"],
            },
            "model_report": model_report,
        }

    def _load_checkpoint(self) -> dict:
        if os.path.exists(TRAIN_CHECKPOINT):
            with open(TRAIN_CHECKPOINT) as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, data: dict):
        with open(TRAIN_CHECKPOINT, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _git_commit(msg: str):
        try:
            subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
            subprocess.run(["git", "commit", "-m", msg], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
        except Exception:
            pass


# ===========================================================================
# Full Pipeline
# ===========================================================================

class HistoricalTrainer:
    """Full pipeline: Download -> Backtest -> Self-Improve -> Report."""

    def __init__(self):
        self.downloader = HistoricalDownloader()
        self.backtester = Backtester()
        self.safeguard = Safeguard()

    def run_full_pipeline(self, max_markets: int = 1000):
        print("=" * 60)
        print("  HISTORICAL TRAINING PIPELINE")
        print("=" * 60)

        # Step 1: Download
        print("\n[1/3] Downloading resolved markets...")
        n = self.downloader.download_all(max_markets=max_markets)
        total_hist = self.downloader.count_historical()
        print(f"  Total historical markets: {total_hist}")

        # Step 2: Backtest with safeguard
        print("\n[2/3] Running backtest with safeguard...")
        result = self.backtester.run_backtest(batch_size=50)

        if "audit" in result:
            audit = result["audit"]
            self._print_audit(audit, result)

        # Step 3: Self-improvement DISABLED — manual tuning until models are solid
        print("\n[3/3] Self-improvement DISABLED (manual tuning mode)")
        print("  Skipping auto weight/calibration overwrite to protect manual fixes.")

        # Git commit final state
        self._git_commit(
            f"training: complete | {result.get('n_predictions', 0)} predictions | "
            f"brier={result.get('audit', {}).get('brier_score', '?')}"
        )

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("  Run: python -m src.training_runner  (for live mode)")
        print("  Run: python -m src.historical_trainer --report  (for report)")
        print("=" * 60)

    def show_report(self):
        """Show full progression report."""
        progression = self.safeguard.get_progression()
        if not progression:
            print("No progression data. Run training first.")
            return

        print(f"\n{'=' * 70}")
        print(f"  TRAINING PROGRESSION REPORT  ({len(progression)} snapshots)")
        print(f"{'=' * 70}")
        print(f"  {'Batch':>5} {'Approach':>22} {'Brier':>8} {'ECE':>8} {'Disc':>8} "
              f"{'ROI':>8} {'Health':>7} {'Score':>7}")
        print(f"  {'-'*5} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

        best_score = 0
        best_entry = None
        for entry in progression:
            roi_str = f"{entry['roi']:.1%}" if entry.get("roi") is not None else "N/A"
            ehr_str = f"{entry['edge_hit_rate']:.1%}" if entry.get("edge_hit_rate") is not None else "N/A"
            score = entry.get("composite", 0)
            marker = " *" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_entry = entry
            print(f"  {entry['cycle']:>5} {entry.get('approach','?'):>22} "
                  f"{entry['brier']:>8.4f} {entry['ece']:>8.4f} {entry['discrimination']:>8.4f} "
                  f"{roi_str:>8} {entry['health']:>7.1f} {score:>6.1f}{marker}")

        if best_entry:
            print(f"\n  BEST: Batch {best_entry['cycle']} | "
                  f"Approach: {best_entry.get('approach')} | "
                  f"Score: {best_entry.get('composite', 0):.1f} | "
                  f"Brier: {best_entry['brier']:.4f}")

        # Show approach history
        approach_mgr = ApproachManager()
        state = approach_mgr.state
        if state.get("history"):
            print(f"\n  Approach switches:")
            for h in state["history"]:
                print(f"    {h['from']} -> {h['to']} ({h.get('reason', '')})")

        # Show best scores per approach
        if state.get("best_scores"):
            print(f"\n  Best score per approach:")
            for a, s in sorted(state["best_scores"].items(), key=lambda x: -x[1]):
                print(f"    {a:25s} {s:.1f}")

    @staticmethod
    def _print_audit(audit: dict, result: dict):
        print(f"\n  BACKTEST RESULTS:")
        print(f"    Predictions:    {result['n_predictions']}")
        print(f"    Errors:         {result['n_errors']}")
        print(f"    Batches:        {result['n_batches']}")
        print(f"    Final approach: {result['final_approach']}")
        print(f"    Brier Score:    {audit.get('brier_score', '?'):.4f}")
        print(f"    Log Loss:       {audit.get('log_loss', '?'):.4f}")
        print(f"    ECE:            {audit.get('expected_calibration_error', '?'):.4f}")
        print(f"    Discrimination: {audit.get('discrimination', '?'):.4f}")
        if audit.get("roi") is not None:
            print(f"    ROI:            {audit['roi']:.1%}")
        if audit.get("edge_hit_rate") is not None:
            print(f"    Edge Hit Rate:  {audit['edge_hit_rate']:.1%}")

    @staticmethod
    def _git_commit(msg: str):
        try:
            subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
            subprocess.run(["git", "commit", "-m", msg], cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Historical Training Pipeline")
    parser.add_argument("--download", action="store_true", help="Download only")
    parser.add_argument("--train", action="store_true", help="Train on existing data only")
    parser.add_argument("--report", action="store_true", help="Show progression report")
    parser.add_argument("--strategy", action="store_true", help="Run strategy A/B backtest (baseline vs calendar-adapted)")
    parser.add_argument("--max-markets", type=int, default=1000, help="Max markets to download")
    parser.add_argument("--equity", type=float, default=10000.0, help="Starting equity for strategy backtest")
    args = parser.parse_args()

    trainer = HistoricalTrainer()

    if args.download:
        trainer.downloader.download_all(max_markets=args.max_markets)
    elif args.train:
        trainer.backtester.run_backtest()
    elif args.report:
        trainer.show_report()
    elif args.strategy:
        backtester = Backtester(equity=args.equity)
        backtester.run_strategy_backtest()
    else:
        trainer.run_full_pipeline(max_markets=args.max_markets)


if __name__ == "__main__":
    main()
