"""Unified cheat-free end-to-end benchmark runner.

Runs the complete 3-stage evaluation loop:
Stage 1: Machine track ingestion & dual-stream extraction (zero-cheat, raw video/tracks only)
Stage 2: Zero-seed cold start & incremental SQLite library building
Stage 3: Independent process T4 firewall benchmark scoring (on fixed 70 denominator)

Usage:
    cd player-library
    uv run python run_e2e_benchmark.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.run_full_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
