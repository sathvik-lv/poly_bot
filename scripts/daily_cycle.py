"""Daily Automation Cycle — Run everything in one shot.

Runs:
1. Paper trader scan (place new trades)
2. Paper trader resolve (check outcomes)
3. Live tracker resolve (check prediction accuracy)
4. Finance scanner (check financial markets)
5. Report everything
6. Git commit results

Usage:
    python scripts/daily_cycle.py           # full cycle
    python scripts/daily_cycle.py --loop    # repeat every 6 hours
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run_script(name: str, args: list[str] = None, timeout: int = 600) -> bool:
    """Run a script and print output."""
    cmd = [PYTHON, "-u", os.path.join(PROJECT_ROOT, "scripts", name)]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=timeout)
        print(result.stdout)
        if result.stderr:
            print(result.stderr[-500:])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {name} exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {name} failed: {e}")
        return False


def git_commit(message: str):
    """Commit data/ to git."""
    try:
        subprocess.run(["git", "add", "data/"], cwd=PROJECT_ROOT,
                       capture_output=True, timeout=30)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  [GIT] {message}")
        elif "nothing to commit" in (result.stdout or ""):
            print(f"  [GIT] Nothing to commit")
    except Exception as e:
        print(f"  [GIT] Failed: {e}")


def run_cycle():
    """Run one full daily cycle."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*70}")
    print(f"  DAILY CYCLE — {ts}")
    print(f"{'='*70}")

    # 1. Paper trader: scan new markets + place trades
    print(f"\n--- STEP 1: Paper Trades (scan + trade) ---")
    run_script("paper_trader.py", ["--scan", "-n", "30"], timeout=900)

    # 2. Paper trader: resolve closed markets
    print(f"\n--- STEP 2: Paper Trades (resolve) ---")
    run_script("paper_trader.py", ["--resolve"], timeout=120)

    # 3. Paper trader: P&L report
    print(f"\n--- STEP 3: Paper P&L Report ---")
    run_script("paper_trader.py", ["--report"], timeout=30)

    # 4. Live tracker: resolve predictions
    print(f"\n--- STEP 4: Prediction Accuracy (resolve) ---")
    run_script("live_tracker.py", ["--resolve"], timeout=120)

    # 5. Live tracker: accuracy report
    print(f"\n--- STEP 5: Prediction Accuracy Report ---")
    run_script("live_tracker.py", ["--report"], timeout=30)

    # 6. Finance scanner
    print(f"\n--- STEP 6: Financial Markets ---")
    run_script("finance_scanner.py", timeout=300)

    # 7. Price snapshots (build time series data)
    print(f"\n--- STEP 7: Price Snapshots ---")
    run_script("price_snapshots.py", timeout=60)

    # 8. Arbitrage scanner
    print(f"\n--- STEP 8: Arbitrage Scanner ---")
    run_script("arbitrage_scanner.py", timeout=120)

    # 9. Cross-platform odds comparison
    print(f"\n--- STEP 9: Cross-Platform Comparison ---")
    run_script("cross_platform.py", timeout=180)

    # 10. Logical arbitrage (constraint graph)
    print(f"\n--- STEP 10: Logical Arbitrage ---")
    run_script("logical_arb.py", timeout=120)

    # 11. Niche data alpha (crypto quant + sports odds)
    print(f"\n--- STEP 11: Niche Data Scanner ---")
    run_script("niche_scanner.py", timeout=120)

    # 12. Git commit
    print(f"\n--- STEP 12: Git Checkpoint ---")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    git_commit(f"daily: cycle {now} | paper trades + accuracy tracking")

    print(f"\n{'='*70}")
    print(f"  CYCLE COMPLETE — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Daily Automation Cycle")
    parser.add_argument("--loop", action="store_true", help="Repeat every 6 hours")
    parser.add_argument("--interval", type=int, default=21600, help="Loop interval in seconds (default 6h)")
    args = parser.parse_args()

    if args.loop:
        print(f"  Running in loop mode (every {args.interval}s / {args.interval//3600}h)")
        while True:
            try:
                run_cycle()
                print(f"\n  Sleeping {args.interval//3600}h until next cycle...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n  Stopped.")
                break
            except Exception as e:
                print(f"\n  Cycle error: {e}")
                time.sleep(300)
    else:
        run_cycle()


if __name__ == "__main__":
    main()
