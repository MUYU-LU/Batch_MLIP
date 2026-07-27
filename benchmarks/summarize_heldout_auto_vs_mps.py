#!/usr/bin/env python3
"""Summarize frozen automatic BFGS scheduling against four-GPU CUDA MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, float | int]:
    """Return stable integer-sample convergence-horizon statistics."""

    if not values:
        raise ValueError("distribution requires at least one value")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _mps_records(
    experiment: Path,
    *,
    pool_size: int,
) -> tuple[list[tuple[Any, dict[str, Any]]], list[dict[str, Any]]]:
    raw = experiment / "results" / "raw"
    paired = []
    results = []
    for gpu_index in range(4):
        manifest_path = (
            experiment
            / "workloads"
            / f"mps_r{pool_size}"
            / (
                f"OPT-HOLDOUT-MIX-R{pool_size}-v1-"
                f"MPS-G{gpu_index}.json"
            )
        )
        manifest = read_workload_manifest(manifest_path)
        result = _load(raw / f"mps_r{pool_size}_gpu{gpu_index}.json")
        records = [
            record
            for worker in result["worker_results"]
            for record in worker["records"]
        ]
        if len(records) != len(manifest.jobs):
            raise ValueError("MPS result and signed shard sizes differ")
        paired.extend(zip(manifest.jobs, records, strict=True))
        results.append(result)
    return paired, results


def summarize_case(experiment: Path, *, pool_size: int) -> dict[str, Any]:
    """Build one source-audited automatic-versus-MPS comparison."""

    raw = experiment / "results" / "raw"
    base_manifest_path = (
        experiment
        / "workloads"
        / "manifests"
        / f"OPT-HOLDOUT-MIX-R{pool_size}-v1.json"
    )
    manifest = read_workload_manifest(base_manifest_path)
    automatic_path = raw / f"auto_r{pool_size}.json"
    automatic = _load(automatic_path)
    call = automatic["call_records"][0]
    schedule = call["schedule"]
    auto_external = _load(raw / f"auto_r{pool_size}_external.json")[
        "external_process_wall_seconds"
    ]
    mps_external = _load(raw / f"mps_r{pool_size}_external.json")[
        "external_process_wall_seconds"
    ]
    mps_pairs, mps_results = _mps_records(
        experiment,
        pool_size=pool_size,
    )

    base_sources = [job.source_path for job in manifest.jobs]
    mps_sources = [job.source_path for job, _ in mps_pairs]
    auto_steps = [int(value) for value in call["converged_steps"]]
    mps_steps = [int(record["steps"]) for _, record in mps_pairs]
    mps_production = max(
        float(result["timing"]["wall_seconds"])
        for result in mps_results
    )
    auto_production = float(schedule["production_run_seconds"])
    auto_api = float(call["wall_time_s"])
    total_atoms = sum(job.atom_count for job in manifest.jobs)
    gpu_total = int(automatic["environment"]["gpu_total_memory_bytes"])
    auto_worker_reserved = int(call["peak_reserved_bytes"])
    auto_probe_reserved = int(schedule["probe"]["peak_reserved_bytes"])
    auto_conservative = auto_worker_reserved + auto_probe_reserved
    mps_peak = max(
        int(result["peak_gpu_memory_bytes_nvidia_smi"])
        for result in mps_results
    )
    mps_converged = sum(
        int(result["converged"]) for result in mps_results
    )

    family_auto_steps: dict[str, list[int]] = defaultdict(list)
    family_mps_steps: dict[str, list[int]] = defaultdict(list)
    family_auto_converged: Counter[str] = Counter()
    family_mps_converged: Counter[str] = Counter()
    for job, step in zip(manifest.jobs, auto_steps, strict=True):
        family = job.source_path.split("/", 1)[0]
        family_auto_steps[family].append(step)
        family_auto_converged[family] += int(step >= 0)
    for (job, record), step in zip(mps_pairs, mps_steps, strict=True):
        family = job.source_path.split("/", 1)[0]
        family_mps_steps[family].append(step)
        family_mps_converged[family] += int(record["converged"])

    families = {}
    for family in sorted(family_auto_steps):
        families[family] = {
            "jobs": len(family_auto_steps[family]),
            "automatic_converged": family_auto_converged[family],
            "mps_converged": family_mps_converged[family],
            "automatic_steps": distribution(family_auto_steps[family]),
            "mps_steps": distribution(family_mps_steps[family]),
        }

    return {
        "pool_size": pool_size,
        "jobs": len(manifest.jobs),
        "total_atoms": total_atoms,
        "manifest_sha256": manifest.manifest_sha256,
        "coverage": {
            "base_unique_sources": len(set(base_sources)),
            "mps_unique_sources": len(set(mps_sources)),
            "same_source_multiset": Counter(base_sources)
            == Counter(mps_sources),
            "automatic_job_count": len(auto_steps),
            "mps_job_count": len(mps_pairs),
        },
        "timing_seconds": {
            "automatic_production": auto_production,
            "automatic_api": auto_api,
            "automatic_external_process": auto_external,
            "mps_production": mps_production,
            "mps_external_process": mps_external,
        },
        "speedup_over_mps": {
            "automatic_production": mps_production / auto_production,
            "automatic_api": mps_production / auto_api,
            "automatic_external_process": mps_external / auto_external,
        },
        "throughput": {
            "automatic_production_systems_per_second": (
                pool_size / auto_production
            ),
            "automatic_api_systems_per_second": pool_size / auto_api,
            "mps_production_systems_per_second": (
                pool_size / mps_production
            ),
            "automatic_production_atoms_per_second": (
                total_atoms / auto_production
            ),
            "automatic_api_atoms_per_second": total_atoms / auto_api,
            "mps_production_atoms_per_second": total_atoms / mps_production,
        },
        "memory": {
            "gpu_total_bytes": gpu_total,
            "automatic_worker_peak_reserved_bytes": auto_worker_reserved,
            "automatic_parent_probe_peak_reserved_bytes": (
                auto_probe_reserved
            ),
            "automatic_conservative_peak_bytes": auto_conservative,
            "automatic_conservative_peak_fraction": (
                auto_conservative / gpu_total
            ),
            "mps_peak_sampled_device_bytes": mps_peak,
        },
        "convergence": {
            "automatic": int(call["converged"]),
            "mps": mps_converged,
            "automatic_steps": distribution(auto_steps),
            "mps_steps": distribution(mps_steps),
            "families": families,
        },
        "execution": {
            "automatic_resident_chunks": schedule[
                "resident_plan_chunk_count"
            ],
            "automatic_execution_chunks": schedule[
                "execution_chunk_count"
            ],
            "automatic_batch_sizes": [
                int(chunk["system_count"])
                for chunk in schedule["planned_chunks"]
            ],
            "automatic_model_evaluations": int(
                call["model_evaluations"]
            ),
            "mps_model_evaluations": sum(
                int(result["model_evaluations_total"])
                for result in mps_results
            ),
        },
        "gates": {
            "same_job_coverage": Counter(base_sources)
            == Counter(mps_sources),
            "automatic_memory_at_most_85_percent": (
                auto_conservative / gpu_total <= 0.85
            ),
            "automatic_convergence_not_below_mps": (
                int(call["converged"]) >= mps_converged
            ),
            "automatic_production_faster_than_mps": (
                auto_production < mps_production
            ),
            "automatic_api_faster_than_mps_production": (
                auto_api < mps_production
            ),
            "automatic_external_process_faster_than_mps": (
                auto_external < mps_external
            ),
        },
        "raw_sha256": {
            "automatic": _sha256(automatic_path),
            **{
                f"mps_gpu_{gpu_index}": _sha256(
                    raw / f"mps_r{pool_size}_gpu{gpu_index}.json"
                )
                for gpu_index in range(4)
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        summarize_case(args.experiment_dir, pool_size=pool_size)
        for pool_size in (256, 1024)
    ]
    result = {
        "schema_version": 1,
        "status": "complete",
        "comparison": (
            "frozen automatic BatchExecutor versus four ASE CUDA MPS "
            "workers per GPU"
        ),
        "cases": cases,
        "all_primary_gates_pass": all(
            case["gates"][gate]
            for case in cases
            for gate in (
                "same_job_coverage",
                "automatic_memory_at_most_85_percent",
                "automatic_convergence_not_below_mps",
                "automatic_production_faster_than_mps",
                "automatic_api_faster_than_mps_production",
            )
        ),
        "all_external_process_gates_pass": all(
            case["gates"][
                "automatic_external_process_faster_than_mps"
            ]
            for case in cases
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
