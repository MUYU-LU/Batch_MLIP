#!/usr/bin/env python3
"""Execute a signed controlled workload using a configured BatchCalculator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
