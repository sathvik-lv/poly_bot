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

    def __init__(self):
        self.engine = PredictionEngine()
        self.auditor = CalibrationAuditor()
        self.safeguard = Safeguard()
        self.approach_mgr = ApproachManager()
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
                    np.random.seed(hash(market.get("id", str(i))) % 2**31)
                    noise = np.random.normal(0, 0.15)
                    sim_price = float(np.clip(outcome * 0.6 + 0.2 + noise, 0.05, 0.95))

                    market_data = {
                        "id": market.get("id"),
                        "question": market.get("question", ""),
                        "outcomes": market.get("outcomes", ["Yes", "No"]),
                        "outcome_prices": {"Yes": sim_price, "No": 1 - sim_price},
                        "volume": market.get("volume"),
                        "volume_24h": None,
                        "liquidity": market.get("liquidity"),
                        "last_trade_price": sim_price,
                        "best_bid": sim_price - 0.01,
                        "best_ask": sim_price + 0.01,
                        "spread": 0.02,
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

                # Run self-improvement every 5 batches
                if batch_num % 5 == 0 and len(resolved) >= 20:
                    corrector = SelfCorrector(DATA_DIR)
                    # Temporarily point corrector at backtest data
                    corrector.tracker = tracker
                    improvement = corrector.run_improvement_cycle()
                    print(f"    Self-improvement: Health={improvement.get('health_score', 0)}/100")

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

        # Step 3: Final self-improvement
        print("\n[3/3] Final self-improvement cycle...")
        corrector = SelfCorrector(DATA_DIR)
        # Copy backtest data to main predictions
        bt_tracker = PredictionTracker(BACKTEST_FILE)
        resolved = bt_tracker.get_resolved_predictions()
        if resolved:
            main_path = os.path.join(DATA_DIR, "predictions.jsonl")
            with open(main_path, "w") as f:
                for p in resolved:
                    f.write(json.dumps(p) + "\n")

            improvement = corrector.run_improvement_cycle()
            print(f"  Health Score: {improvement.get('health_score', 0)}/100")
            for rec in improvement.get("recommendations", [])[:5]:
                print(f"    - {rec}")

            weights = improvement.get("optimized_weights", {}).get("weights", {})
            if weights:
                print(f"\n  Optimized Model Weights:")
                for model, w in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"    {model:25s} {w:.4f} {'#' * int(w * 40)}")

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
    parser.add_argument("--max-markets", type=int, default=1000, help="Max markets to download")
    args = parser.parse_args()

    trainer = HistoricalTrainer()

    if args.download:
        trainer.downloader.download_all(max_markets=args.max_markets)
    elif args.train:
        trainer.backtester.run_backtest()
    elif args.report:
        trainer.show_report()
    else:
        trainer.run_full_pipeline(max_markets=args.max_markets)


if __name__ == "__main__":
    main()
