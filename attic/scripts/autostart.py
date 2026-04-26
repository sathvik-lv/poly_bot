"""Auto-Start Service — Runs everything continuously on PC startup.

This is the master process that runs ALL bot operations:
- Price snapshots every 2 hours
- Paper trades + resolve every 6 hours
- Arbitrage scan every 6 hours
- Financial market scan every 6 hours
- Live prediction tracking every 6 hours
- Git commits after each cycle

Setup: Run once to create Windows startup shortcut:
    python scripts/autostart.py --install

Then it auto-starts when you log in. Or run manually:
    python scripts/autostart.py
"""

import os
import sys
import time
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable
LOG_FILE = PROJECT_ROOT / "data" / "autostart.log"


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_script(name: str, args: list = None, timeout: int = 600) -> bool:
    cmd = [PYTHON, "-u", str(PROJECT_ROOT / "scripts" / name)]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                                text=True, timeout=timeout)
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                log(f"  {name}: {line.strip()}")
        if result.returncode != 0 and result.stderr:
            log(f"  {name} ERROR: {result.stderr[-200:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  {name} TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        log(f"  {name} FAILED: {e}")
        return False


def git_commit(message: str):
    try:
        subprocess.run(["git", "add", "data/"], cwd=str(PROJECT_ROOT),
                       capture_output=True, timeout=30)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            log(f"  GIT: {message}")
    except Exception:
        pass


def run_snapshot():
    """Quick task: price snapshot (runs every 2 hours)."""
    log("--- SNAPSHOT ---")
    run_script("price_snapshots.py", timeout=60)


def run_full_cycle():
    """Full cycle: everything (runs every 6 hours)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    log(f"=== FULL CYCLE {ts} ===")

    log("--- Paper Trader: Scan ---")
    run_script("paper_trader.py", ["--scan", "-n", "30"], timeout=900)

    log("--- Paper Trader: Resolve ---")
    run_script("paper_trader.py", ["--resolve"], timeout=120)

    log("--- Paper Trader: Report ---")
    run_script("paper_trader.py", ["--report"], timeout=30)

    log("--- Live Tracker: Resolve ---")
    run_script("live_tracker.py", ["--resolve"], timeout=120)

    log("--- Live Tracker: Report ---")
    run_script("live_tracker.py", ["--report"], timeout=30)

    log("--- Finance Scanner ---")
    run_script("finance_scanner.py", timeout=300)

    log("--- Arbitrage Scanner ---")
    run_script("arbitrage_scanner.py", timeout=120)

    log("--- Cross-Platform Comparison ---")
    run_script("cross_platform.py", timeout=180)

    log("--- Price Snapshot ---")
    run_script("price_snapshots.py", timeout=60)

    log("--- Git Commit ---")
    git_commit(f"auto: cycle {ts} | paper+track+finance+arb+snapshot")

    log(f"=== CYCLE DONE ===\n")


def install_startup():
    """Create Windows startup shortcut so bot runs on PC boot."""
    import winreg

    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

    # Create a .bat file in startup folder
    bat_path = startup_dir / "poly_bot_autostart.bat"
    bat_content = f"""@echo off
title Poly Bot - Auto Trading
cd /d "{PROJECT_ROOT}"
"{PYTHON}" -u scripts/autostart.py
pause
"""
    bat_path.write_text(bat_content)
    log(f"Installed startup script: {bat_path}")
    print(f"\n  Startup script installed at:")
    print(f"  {bat_path}")
    print(f"\n  The bot will auto-start when you log into Windows.")
    print(f"  To remove: delete that file from your Startup folder.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Poly Bot Auto-Start Service")
    parser.add_argument("--install", action="store_true", help="Install Windows startup shortcut")
    args = parser.parse_args()

    if args.install:
        install_startup()
        return

    log("=" * 60)
    log("  POLY BOT AUTO-START SERVICE")
    log("  Snapshot: every 2h | Full cycle: every 6h")
    log("=" * 60)

    # Run full cycle immediately on start
    try:
        run_full_cycle()
    except Exception as e:
        log(f"Initial cycle error: {e}")

    # Main loop: snapshot every 2h, full cycle every 6h
    SNAPSHOT_INTERVAL = 7200   # 2 hours
    FULL_CYCLE_INTERVAL = 21600  # 6 hours

    last_snapshot = time.time()
    last_full = time.time()

    try:
        while True:
            now = time.time()

            # Check if full cycle is due (every 6h)
            if now - last_full >= FULL_CYCLE_INTERVAL:
                try:
                    run_full_cycle()
                    last_full = now
                    last_snapshot = now  # full cycle includes snapshot
                except Exception as e:
                    log(f"Full cycle error: {e}")
                    traceback.print_exc()

            # Check if snapshot is due (every 2h)
            elif now - last_snapshot >= SNAPSHOT_INTERVAL:
                try:
                    run_snapshot()
                    last_snapshot = now
                except Exception as e:
                    log(f"Snapshot error: {e}")

            # Sleep 5 minutes between checks
            time.sleep(300)

    except KeyboardInterrupt:
        log("Stopped by user.")
        print("\n  Bot stopped. Run again or use --install for auto-start.")


if __name__ == "__main__":
    main()
