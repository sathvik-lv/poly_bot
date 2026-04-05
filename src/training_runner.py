"""Training, Auditing & Self-Improvement Runner with Git Checkpointing.

This runner:
1. Fetches live Polymarket data and generates predictions
2. Tracks predictions persistently (JSONL)
3. Simulates resolution for backtesting (uses price convergence as proxy)
4. Runs self-audit cycle (Brier score, calibration, signal analysis)
5. Optimizes model weights based on accumulated data
6. Checkpoints state to git after each cycle
7. Resumes from last checkpoint on restart

Memory: Designed to stay under 4GB RAM. Uses streaming/chunked processing.
GPU: Not needed — all computation is CPU-based (numpy/scipy are already optimized).

Usage:
    python -m src.training_runner              # run continuous loop
    python -m src.training_runner --once       # run single cycle
    python -m src.training_runner --audit      # run audit only
    python -m src.training_runner --status     # show current status
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure project root is on path
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


DATA_DIR = str(PROJECT_ROOT / "data")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.json")
LOG_FILE = os.path.join(DATA_DIR, "training_log.jsonl")
MAX_RAM_MB = 3500  # stay under 4GB with headroom


class TrainingRunner:
    """Continuous training loop with git checkpointing and resume."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.market_client = MarketClient()
        self.engine = PredictionEngine()
        self.corrector = SelfCorrector(data_dir)
        self.tracker = self.corrector.tracker
        self.auditor = CalibrationAuditor()

        # Load checkpoint
        self.checkpoint = self._load_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint / Resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r") as f:
                cp = json.load(f)
            print(f"[RESUME] Loaded checkpoint from cycle {cp.get('cycle', 0)}")
            return cp
        return {
            "cycle": 0,
            "total_predictions": 0,
            "total_resolved": 0,
            "last_cycle_time": None,
            "best_brier": 1.0,
            "best_health_score": 0,
            "markets_seen": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_checkpoint(self):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self.checkpoint, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Git Operations
    # ------------------------------------------------------------------

    @staticmethod
    def _git_commit(message: str) -> bool:
        """Stage data/ and commit with message. Returns True if committed."""
        try:
            subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT),
                         capture_output=True, timeout=30)
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"  [GIT] Committed: {message}")
                return True
            else:
                if "nothing to commit" in result.stdout:
                    print("  [GIT] Nothing to commit")
                return False
        except Exception as e:
            print(f"  [GIT] Commit failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Training Log
    # ------------------------------------------------------------------

    def _log(self, entry: dict):
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------
    # Core Training Cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Execute one full training cycle.

        Steps:
            1. Fetch active markets
            2. Generate predictions for top markets
            3. Simulate resolution for old predictions (price convergence proxy)
            4. Run self-audit
            5. Run self-improvement (weight optimization)
            6. Checkpoint to git
        """
        cycle_num = self.checkpoint["cycle"] + 1
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle_num}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*60}")

        cycle_result = {"cycle": cycle_num, "steps": {}}

        # Step 1: Fetch markets
        print("\n[1/6] Fetching active markets...")
        try:
            raw_markets = self.market_client.get_markets(limit=30, order="volume24hr")
            markets = [MarketClient.parse_market(m) for m in raw_markets]
            active_markets = [m for m in markets if m.get("active") and not m.get("closed")]
            cycle_result["steps"]["fetch"] = {"n_markets": len(active_markets), "status": "OK"}
            print(f"       Found {len(active_markets)} active markets")
        except Exception as e:
            print(f"       FAILED: {e}")
            cycle_result["steps"]["fetch"] = {"status": "FAILED", "error": str(e)}
            active_markets = []

        # Step 2: Generate predictions
        print("\n[2/6] Generating predictions...")
        predictions_made = 0
        for market in active_markets[:15]:  # limit to 15 per cycle for memory
            try:
                prediction = self.engine.predict(
                    market_data=market,
                    time_remaining_frac=0.5,
                )
                pred_id = self.tracker.record_prediction(prediction)
                predictions_made += 1

                edge = prediction["edge"]["edge"]
                prob = prediction["prediction"]["probability"]
                price = prediction["market"]["current_price"]
                action = prediction["sizing"]["action"]

                if abs(edge) > 0.05:
                    print(f"       [{action}] {market['question'][:50]}...")
                    print(f"              Price: {price:.2f} | Pred: {prob:.2f} | Edge: {edge:+.2f}")
            except Exception as e:
                # Skip individual market failures
                continue

        self.checkpoint["total_predictions"] += predictions_made
        cycle_result["steps"]["predict"] = {"n_predictions": predictions_made, "status": "OK"}
        print(f"       Made {predictions_made} predictions")

        # Step 3: Simulate resolution for old predictions
        print("\n[3/6] Simulating resolutions (price convergence proxy)...")
        resolved_count = self._simulate_resolutions(active_markets)
        self.checkpoint["total_resolved"] += resolved_count
        cycle_result["steps"]["resolve"] = {"n_resolved": resolved_count}
        print(f"       Resolved {resolved_count} predictions")

        # Step 4: Self-audit
        print("\n[4/6] Running self-audit...")
        resolved_preds = self.tracker.get_resolved_predictions()
        if len(resolved_preds) >= 5:
            audit = self.auditor.full_audit(resolved_preds)
            cycle_result["steps"]["audit"] = {
                "brier_score": audit.get("brier_score"),
                "ece": audit.get("expected_calibration_error"),
                "discrimination": audit.get("discrimination"),
                "roi": audit.get("roi"),
                "n_resolved": audit.get("n_predictions"),
            }
            brier = audit.get("brier_score", 1.0)
            print(f"       Brier Score: {brier:.4f}")
            print(f"       ECE: {audit.get('expected_calibration_error', 0):.4f}")
            print(f"       Discrimination: {audit.get('discrimination', 0):.4f}")
            if audit.get("roi") is not None:
                print(f"       ROI: {audit['roi']:.1%}")

            if brier < self.checkpoint["best_brier"]:
                self.checkpoint["best_brier"] = brier
                print(f"       *** NEW BEST Brier: {brier:.4f} ***")
        else:
            cycle_result["steps"]["audit"] = {"status": "INSUFFICIENT_DATA", "n_resolved": len(resolved_preds)}
            print(f"       Need 5+ resolved predictions, have {len(resolved_preds)}")

        # Step 5: Self-improvement
        print("\n[5/6] Running self-improvement...")
        if len(resolved_preds) >= 5:
            improvement = self.corrector.run_improvement_cycle()
            health = improvement.get("health_score", 0)
            self.checkpoint["best_health_score"] = max(
                self.checkpoint.get("best_health_score", 0), health
            )
            cycle_result["steps"]["improve"] = {
                "health_score": health,
                "n_recommendations": len(improvement.get("recommendations", [])),
            }
            print(f"       Health Score: {health}/100")
            for rec in improvement.get("recommendations", [])[:3]:
                print(f"         - {rec}")
        else:
            cycle_result["steps"]["improve"] = {"status": "SKIPPED"}
            print("       Skipped (insufficient data)")

        # Step 6: Checkpoint & git commit
        print("\n[6/6] Checkpointing...")
        self.checkpoint["cycle"] = cycle_num
        self.checkpoint["last_cycle_time"] = datetime.now(timezone.utc).isoformat()
        self._save_checkpoint()
        self._log(cycle_result)

        commit_msg = (
            f"training: cycle {cycle_num} | "
            f"preds={predictions_made} resolved={resolved_count} "
            f"total={self.checkpoint['total_predictions']}"
        )
        if cycle_result["steps"].get("audit", {}).get("brier_score") is not None:
            commit_msg += f" | brier={cycle_result['steps']['audit']['brier_score']:.4f}"

        self._git_commit(commit_msg)

        # Print summary
        print(f"\n{'-'*60}")
        print(f"  Cycle {cycle_num} complete")
        print(f"  Total predictions: {self.checkpoint['total_predictions']}")
        print(f"  Total resolved: {self.checkpoint['total_resolved']}")
        print(f"  Best Brier: {self.checkpoint['best_brier']:.4f}")
        print(f"  Health Score: {self.checkpoint.get('best_health_score', 0)}/100")
        print(f"{'-'*60}")

        return cycle_result

    def _simulate_resolutions(self, current_markets: list[dict]) -> int:
        """Resolve old predictions using current market prices as proxy.

        If a market's price moved significantly since prediction,
        use the new price direction as a proxy for the outcome.
        This isn't perfect but enables the self-improvement loop
        to start learning immediately without waiting for actual resolution.
        """
        unresolved = self.tracker.get_unresolved_predictions()
        current_prices = {m["id"]: m.get("outcome_prices", {}).get("Yes") for m in current_markets}

        resolved = 0
        for pred in unresolved:
            market_id = pred.get("market_id")
            if market_id not in current_prices:
                continue

            old_price = pred.get("market_price")
            new_price = current_prices.get(market_id)
            if old_price is None or new_price is None:
                continue

            # Resolution proxy: if price moved strongly toward 0 or 1
            if new_price > 0.9:
                self.tracker.record_outcome(market_id, 1.0)
                resolved += 1
            elif new_price < 0.1:
                self.tracker.record_outcome(market_id, 0.0)
                resolved += 1
            elif abs(new_price - old_price) > 0.15:
                # Significant move — use direction as soft outcome
                outcome = 1.0 if new_price > old_price else 0.0
                self.tracker.record_outcome(market_id, outcome)
                resolved += 1

        return resolved

    # ------------------------------------------------------------------
    # Continuous Run
    # ------------------------------------------------------------------

    def run_continuous(self, interval_seconds: int = 300, max_cycles: int = 0):
        """Run training loop continuously.

        Args:
            interval_seconds: Seconds between cycles (default 5 min)
            max_cycles: Stop after this many cycles (0 = infinite)
        """
        print(f"+{'='*58}+")
        print(f"|  POLY_BOT TRAINING RUNNER                                |")
        print(f"|  Interval: {interval_seconds}s | Max cycles: {max_cycles or 'infinite':>10}         |")
        print(f"|  Resuming from cycle: {self.checkpoint['cycle']:<10}                      |")
        print(f"+{'='*58}+")

        cycles_run = 0
        try:
            while True:
                try:
                    self.run_cycle()
                    cycles_run += 1

                    if max_cycles > 0 and cycles_run >= max_cycles:
                        print(f"\nReached max cycles ({max_cycles}). Stopping.")
                        break

                    # Memory check
                    self._check_memory()

                    print(f"\nSleeping {interval_seconds}s until next cycle...")
                    time.sleep(interval_seconds)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\n[ERROR] Cycle failed: {e}")
                    traceback.print_exc()
                    self._log({"event": "cycle_error", "error": str(e)})
                    print("Retrying in 60s...")
                    time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Graceful shutdown — checkpointing...")
            self._save_checkpoint()
            self._git_commit(f"training: checkpoint at cycle {self.checkpoint['cycle']} (stopped)")
            print("State saved. Run again to resume.")

    def _check_memory(self):
        """Warn if memory usage is getting high."""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > MAX_RAM_MB:
                print(f"  [WARN] Memory usage: {mem_mb:.0f}MB > {MAX_RAM_MB}MB limit")
                # Force garbage collection
                import gc
                gc.collect()
        except ImportError:
            pass  # psutil not installed, skip check

    # ------------------------------------------------------------------
    # Status / Audit Commands
    # ------------------------------------------------------------------

    def show_status(self):
        """Print current training status."""
        cp = self.checkpoint
        print(f"\n{'='*60}")
        print(f"  POLY_BOT TRAINING STATUS")
        print(f"{'='*60}")
        print(f"  Cycles completed:    {cp['cycle']}")
        print(f"  Total predictions:   {cp['total_predictions']}")
        print(f"  Total resolved:      {cp['total_resolved']}")
        print(f"  Best Brier Score:    {cp['best_brier']:.4f}")
        print(f"  Best Health Score:   {cp.get('best_health_score', 0)}/100")
        print(f"  Last cycle:          {cp.get('last_cycle_time', 'never')}")
        print(f"  Started:             {cp.get('started_at', 'unknown')}")
        print(f"{'='*60}")

        # Show recent log entries
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
            recent = lines[-5:] if len(lines) >= 5 else lines
            print(f"\n  Recent log entries ({len(lines)} total):")
            for line in recent:
                entry = json.loads(line)
                cycle = entry.get("cycle", "?")
                ts = entry.get("timestamp", "?")[:19]
                print(f"    Cycle {cycle} @ {ts}")

    def run_audit_only(self):
        """Run audit without making new predictions."""
        print("\n[AUDIT] Running full audit on historical predictions...")
        resolved = self.tracker.get_resolved_predictions()
        if len(resolved) < 5:
            print(f"  Need 5+ resolved predictions, have {len(resolved)}")
            return

        report = self.corrector.run_improvement_cycle()

        print(f"\n  Health Score: {report.get('health_score', 0)}/100")
        print(f"  N resolved: {report.get('n_resolved', 0)}")

        audit = report.get("audit", {})
        if audit:
            print(f"  Brier Score: {audit.get('brier_score', '?')}")
            print(f"  ECE: {audit.get('expected_calibration_error', '?')}")
            print(f"  Discrimination: {audit.get('discrimination', '?')}")
            print(f"  ROI: {audit.get('roi', '?')}")

        recs = report.get("recommendations", [])
        if recs:
            print(f"\n  Recommendations:")
            for r in recs:
                print(f"    - {r}")


# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Poly Bot Training Runner")
    parser.add_argument("--once", action="store_true", help="Run single cycle")
    parser.add_argument("--audit", action="store_true", help="Run audit only")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max cycles (0=infinite)")
    args = parser.parse_args()

    runner = TrainingRunner()

    if args.status:
        runner.show_status()
    elif args.audit:
        runner.run_audit_only()
    elif args.once:
        runner.run_cycle()
    else:
        runner.run_continuous(interval_seconds=args.interval, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
