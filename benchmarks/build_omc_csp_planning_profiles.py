"""Build layered AtomBit/BFGS planning sidecars for frozen OMC-CSP workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from batch_mlip import (
    BatchedBFGS,
    FrechetCellFilter,
    planning_profile_from_manifest,
    write_planning_profile,
)
from batch_mlip.workloads import read_workload_manifest, topology_key


def _canonical_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="AtomBit-smooth-rms-fp32")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--force-mode", default="autograd")
    parser.add_argument("--model-dtype", default="torch.float32")
    parser.add_argument("--neighbor-backend", default="auto")
    args = parser.parse_args()

    manifest_dir = args.workloads / "manifests"
    paths = sorted(manifest_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no workload manifests in {manifest_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    active_key = topology_key(args.cutoff, 0.0)
    candidate_key = topology_key(args.cutoff, args.skin)
    records = {}
    for path in paths:
        manifest = read_workload_manifest(path)
        profile = planning_profile_from_manifest(
            manifest,
            model_id=args.model_id,
            cutoff_A=args.cutoff,
            active_edge_key=active_key,
            candidate_edge_key=candidate_key,
            force_mode=args.force_mode,
            model_dtype=args.model_dtype,
            optimizer=BatchedBFGS(optimizer_dtype="float64"),
            variable_cell=True,
            cell_method=FrechetCellFilter(),
            skin_A=args.skin,
            neighbor_backend=args.neighbor_backend,
        )
        output = args.output / f"{manifest.workload_id}.json"
        write_planning_profile(output, profile)
        records[manifest.workload_id] = {
            "workload_manifest_sha256": manifest.manifest_sha256,
            "structure_workload_sha256": profile.structure_workload_sha256,
            "planning_profile_sha256": profile.profile_sha256,
            "system_count": len(profile.systems),
            "path": output.name,
        }
    index = {
        "schema_version": 1,
        "profile_kind": "model_task_execution_sidecar",
        "universal_scalar_cost": False,
        "layers": {
            "general_structure": ["atom_count"],
            "mlip_graph": [
                "model_id",
                "cutoff_A",
                "active_edge_count",
                "force_mode",
                "model_dtype",
            ],
            "task_auxiliary": [
                "algorithm",
                "variable_cell",
                "stress_required",
                "cell_method",
                "cell_degrees_of_freedom",
                "state_dtype",
                "generalized_dimension",
                "dense_state_elements",
                "dense_linear_algebra_work",
                "horizon_kind",
            ],
            "graph_execution_policy": [
                "skin_A",
                "candidate_edge_count",
                "cache_enabled",
                "neighbor_backend",
            ],
            "hardware": "bound later by the execution planner",
        },
        "contract": {
            "model_id": args.model_id,
            "cutoff_A": args.cutoff,
            "skin_A": args.skin,
            "force_mode": args.force_mode,
            "model_dtype": args.model_dtype,
            "optimizer": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "neighbor_backend": args.neighbor_backend,
        },
        "profiles": records,
    }
    index["index_sha256"] = _canonical_sha256(index)
    (args.output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "index_sha256": index["index_sha256"],
                "output": str(args.output),
                "profile_count": len(records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
