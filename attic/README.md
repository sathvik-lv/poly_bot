# attic

Quarantined modules — not imported by any active script or workflow as of 2026-04-26.

Kept for reference / re-activation rather than deleted outright.

- `src/safeguard.py` — training-loop watchdog. Designed to monitor weight
  drift, calibration health, and auto-rollback. Was wired to a now-defunct
  training_runner loop. Re-activate alongside `src/training_runner.py` if
  the continuous training pipeline is ever turned on.
- `scripts/autostart.py` — Windows boot automation. Replaced by the
  GitHub Actions cycle workflow (`cycle.yml`).

If anything in the live path imports from `attic/*`, the import will fail
loudly — that's intentional. Move the file back to its original location
before re-using.
