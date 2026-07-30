"""Build signed unique-candidate workload manifests for OMC CSP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_mlip import (
    OMCCSPWorkloadInputs,
    build_omc_csp_workloads,
    write_omc_csp_workloads,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=[64, 512, 2048],
    )
    parser.add_argument("--selection-seed", type=int, default=20260729)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    inputs = OMCCSPWorkloadInputs(
        dataset_dir=args.dataset,
        pool_sizes=tuple(sorted(set(args.pool_sizes))),
        selection_seed=args.selection_seed,
        cutoff_A=args.cutoff,
        skin_A=args.skin,
    )
    workloads, catalog = build_omc_csp_workloads(
        inputs,
        workers=args.workers,
    )
    write_omc_csp_workloads(
        args.output,
        workloads,
        catalog,
        write_csv=not args.no_csv,
    )
    print(
        json.dumps(
            {
                "construction_sha256": catalog["construction_sha256"],
                "family_count": catalog["nonempty_family_count"],
                "workload_count": catalog["workload_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
