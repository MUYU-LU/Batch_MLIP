"""Validate a constructed OMC-CSP workload directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_mlip import validate_omc_csp_workload_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_omc_csp_workload_directory(args.workloads),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
