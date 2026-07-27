#!/usr/bin/env python3
"""Create a sealed deterministic repeated-pool workload manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import (  # noqa: E402
    read_workload_manifest,
    repeat_workload_manifest,
    write_workload_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--repeat-count", type=int, required=True)
    args = parser.parse_args()

    repeated = repeat_workload_manifest(
        read_workload_manifest(args.input),
        repeat_count=args.repeat_count,
        workload_id=args.workload_id,
    )
    write_workload_manifest(args.output, repeated)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "workload_id": repeated.workload_id,
                "jobs": len(repeated.jobs),
                "manifest_sha256": repeated.manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
