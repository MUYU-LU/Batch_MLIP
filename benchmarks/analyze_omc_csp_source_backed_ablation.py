#!/usr/bin/env python3
"""Analyze source-backed OMC-CSP runs against frozen eager and MPS records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from benchmarks.analyze_omc_csp_scheduler_development import (
        MEMORY_LIMIT_FRACTION,
        _auto_memory,
        _contract_audit,
        _phase_summary,
        _timing_summary,
        distribution,
        endpoint_comparison,
    )
except ModuleNotFoundError:
    from analyze_omc_csp_scheduler_development import (
        MEMORY_LIMIT_FRACTION,
        _auto_memory,
        _contract_audit,
        _phase_summary,
        _timing_summary,
        distribution,
        endpoint_comparison,
    )

PLAN_KEYS = (
    "probe",
    "parallel_chunk_policy",
    "resident_plan_chunk_count",
    "execution_chunk_count",
    "resident_plan_chunks",
    "planned_chunks",
)
CONTRACT_KEYS = (
    "optimizer",
    "optimizer_dtype",
    "cell_filter",
    "cutoff_A",
    "skin_A",
    "force_mode",
    "fmax_eV_per_A",
    "max_steps",
    "scheduling",
    "linear_algebra_backend",
    "tail_recovery",
    "tail_recovery_optimizer",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_path(output_path: Path) -> Path:
    return output_path.with_suffix(".external.json")


def _external_seconds(output_path: Path) -> float:
    return float(
        _load(_external_path(output_path))[
            "external_process_wall_seconds"
        ]
    )


def _same_contract(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_contract = left["contract"]
    right_contract = right["contract"]

    def value(contract: dict[str, Any], key: str) -> Any:
        if key == "linear_algebra_backend":
            return contract.get(key, "auto")
        if key == "tail_recovery":
            return contract.get(key, "none")
        if key == "tail_recovery_optimizer" and contract.get(
            "tail_recovery",
            "none",
        ) == "none":
            return None
        return contract.get(key)

    return all(
        value(left_contract, key) == value(right_contract, key)
        for key in CONTRACT_KEYS
    )


def _same_plan(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    return all(
        left["scheduling"][key] == right["scheduling"][key]
        for key in PLAN_KEYS
    )


def _source_order(payload: dict[str, Any]) -> list[str]:
    return [str(record["source"]) for record in payload["records"]]


def analyze_row(item: dict[str, Any]) -> dict[str, Any]:
    lazy_path = Path(item["source_backed"]["output"])
    eager_path = Path(item["frozen_references"]["eager"]["path"])
    mps_path = Path(item["frozen_references"]["ase_cuda_mps"]["path"])
    lazy = _load(lazy_path)
    eager = _load(eager_path)
    mps = _load(mps_path)
    lazy_external = float(
        item["source_backed"]["external_process_wall_seconds"]
    )
    eager_external = _external_seconds(eager_path)
    mps_external = _external_seconds(mps_path)
    lazy_eager_endpoints = endpoint_comparison(
        lazy["records"],
        eager["records"],
    )
    lazy_mps_endpoints = endpoint_comparison(
        lazy["records"],
        mps["records"],
    )
    memory = _auto_memory(lazy)
    timing = _timing_summary(
        lazy,
        mps,
        automatic_external=lazy_external,
        mps_external=mps_external,
    )
    eager_timing = _timing_summary(
        eager,
        mps,
        automatic_external=eager_external,
        mps_external=mps_external,
    )
    lazy_sources = _source_order(lazy)
    eager_sources = _source_order(eager)
    mps_sources = _source_order(mps)
    same_source_identity = (
        lazy_sources == eager_sources
        and set(lazy_sources) == set(mps_sources)
    )
    lazy_converged = int(lazy["converged_count"])
    eager_converged = int(eager["converged_count"])
    mps_converged = int(mps["converged_count"])
    recovery_enabled = lazy["contract"].get("tail_recovery") != "none"
    gates = {
        "status_complete": lazy["status"] == "complete",
        "exact_identity_and_order": (
            same_source_identity
            and len(lazy["records"]) == int(lazy["pool_size"])
        ),
        "same_plan_signature_as_eager": _same_plan(lazy, eager),
        "same_requested_contract_as_eager": _same_contract(lazy, eager),
        "same_requested_contract_as_mps": bool(
            _contract_audit(lazy, mps)["same_requested_contract"]
        ),
        "no_oom": lazy["status"] == "complete",
        "memory_at_most_85_percent": (
            memory["max_conservative_peak_fraction"]
            <= MEMORY_LIMIT_FRACTION
        ),
        "convergence_matches_eager": lazy_converged == eager_converged,
        "convergence_not_below_mps_after_required_recovery": (
            not recovery_enabled or lazy_converged >= mps_converged
        ),
        "external_faster_than_eager": lazy_external < eager_external,
        "external_faster_than_mps": lazy_external < mps_external,
    }
    scientific_gate_names = (
        "status_complete",
        "exact_identity_and_order",
        "same_plan_signature_as_eager",
        "same_requested_contract_as_eager",
        "same_requested_contract_as_mps",
        "no_oom",
        "memory_at_most_85_percent",
        "convergence_matches_eager",
        "convergence_not_below_mps_after_required_recovery",
    )
    materialization = lazy["scheduling"]["structure_materialization"]
    return {
        "workload_id": lazy["workload_id"],
        "pool_size": int(lazy["pool_size"]),
        "timing": {
            "source_backed_external_seconds": lazy_external,
            "eager_external_seconds": eager_external,
            "mps_external_seconds": mps_external,
            "speedup_over_eager": eager_external / lazy_external,
            "speedup_over_mps": mps_external / lazy_external,
            "source_backed_components_seconds": timing[
                "automatic_components_seconds"
            ],
            "eager_components_seconds": eager_timing[
                "automatic_components_seconds"
            ],
        },
        "materialization": materialization,
        "memory": memory,
        "convergence": {
            "source_backed": lazy_converged,
            "eager": eager_converged,
            "mps": mps_converged,
            "mps_non_regression_gate_applicable": recovery_enabled,
        },
        "plan_signature_equal": _same_plan(lazy, eager),
        "endpoints": {
            "source_backed_vs_eager": lazy_eager_endpoints,
            "source_backed_vs_mps": lazy_mps_endpoints,
        },
        "phases": _phase_summary(lazy),
        "gates": gates,
        "all_scientific_and_resource_gates_pass": all(
            gates[name] for name in scientific_gate_names
        ),
        "raw": {
            "source_backed_path": str(lazy_path),
            "source_backed_sha256": _sha256(lazy_path),
            "eager_path": str(eager_path),
            "eager_sha256": _sha256(eager_path),
            "mps_path": str(mps_path),
            "mps_sha256": _sha256(mps_path),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lazy_components: dict[str, float] = defaultdict(float)
    eager_components: dict[str, float] = defaultdict(float)
    for row in rows:
        for name, value in row["timing"][
            "source_backed_components_seconds"
        ].items():
            lazy_components[name] += float(value)
        for name, value in row["timing"]["eager_components_seconds"].items():
            eager_components[name] += float(value)
    lazy_total = sum(lazy_components.values())
    eager_total = sum(eager_components.values())
    return {
        "workloads": len(rows),
        "jobs": sum(row["pool_size"] for row in rows),
        "all_scientific_and_resource_gates_pass": all(
            row["all_scientific_and_resource_gates_pass"] for row in rows
        ),
        "all_plan_signatures_equal": all(
            row["plan_signature_equal"] for row in rows
        ),
        "source_backed_wins_over_eager": sum(
            row["gates"]["external_faster_than_eager"] for row in rows
        ),
        "source_backed_wins_over_mps": sum(
            row["gates"]["external_faster_than_mps"] for row in rows
        ),
        "speedup_over_eager": distribution(
            row["timing"]["speedup_over_eager"] for row in rows
        ),
        "speedup_over_mps": distribution(
            row["timing"]["speedup_over_mps"] for row in rows
        ),
        "source_backed_external_seconds": sum(
            row["timing"]["source_backed_external_seconds"] for row in rows
        ),
        "eager_external_seconds": sum(
            row["timing"]["eager_external_seconds"] for row in rows
        ),
        "mps_external_seconds": sum(
            row["timing"]["mps_external_seconds"] for row in rows
        ),
        "aggregate_speedup_over_eager": (
            sum(row["timing"]["eager_external_seconds"] for row in rows)
            / sum(
                row["timing"]["source_backed_external_seconds"]
                for row in rows
            )
        ),
        "aggregate_speedup_over_mps": (
            sum(row["timing"]["mps_external_seconds"] for row in rows)
            / sum(
                row["timing"]["source_backed_external_seconds"]
                for row in rows
            )
        ),
        "source_backed_components": {
            "seconds": dict(lazy_components),
            "fraction": {
                name: value / lazy_total
                for name, value in lazy_components.items()
            },
        },
        "eager_components": {
            "seconds": dict(eager_components),
            "fraction": {
                name: value / eager_total
                for name, value in eager_components.items()
            },
        },
        "worker_materialization_seconds": sum(
            float(row["materialization"]["worker_seconds"]) for row in rows
        ),
        "maximum_memory_fraction": max(
            row["memory"]["max_conservative_peak_fraction"] for row in rows
        ),
        "source_backed_converged": sum(
            row["convergence"]["source_backed"] for row in rows
        ),
        "eager_converged": sum(
            row["convergence"]["eager"] for row in rows
        ),
        "mps_converged": sum(
            row["convergence"]["mps"] for row in rows
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "workload_id",
        "pool_size",
        "source_backed_external_seconds",
        "eager_external_seconds",
        "mps_external_seconds",
        "speedup_over_eager",
        "speedup_over_mps",
        "worker_materialization_seconds",
        "peak_memory_fraction",
        "source_backed_converged",
        "eager_converged",
        "mps_converged",
        "all_scientific_and_resource_gates_pass",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "workload_id": row["workload_id"],
                    "pool_size": row["pool_size"],
                    "source_backed_external_seconds": row["timing"][
                        "source_backed_external_seconds"
                    ],
                    "eager_external_seconds": row["timing"][
                        "eager_external_seconds"
                    ],
                    "mps_external_seconds": row["timing"][
                        "mps_external_seconds"
                    ],
                    "speedup_over_eager": row["timing"][
                        "speedup_over_eager"
                    ],
                    "speedup_over_mps": row["timing"]["speedup_over_mps"],
                    "worker_materialization_seconds": row[
                        "materialization"
                    ]["worker_seconds"],
                    "peak_memory_fraction": row["memory"][
                        "max_conservative_peak_fraction"
                    ],
                    "source_backed_converged": row["convergence"][
                        "source_backed"
                    ],
                    "eager_converged": row["convergence"]["eager"],
                    "mps_converged": row["convergence"]["mps"],
                    "all_scientific_and_resource_gates_pass": row[
                        "all_scientific_and_resource_gates_pass"
                    ],
                }
            )


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# OMC-CSP Source-Backed P2048 Ablation",
        "",
        "| Workload | Lazy (s) | Eager (s) | MPS (s) | vs eager | vs MPS | Peak memory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        timing = row["timing"]
        lines.append(
            "| {workload} | {lazy:.3f} | {eager:.3f} | {mps:.3f} | "
            "{vs_eager:.3f}x | {vs_mps:.3f}x | {memory:.1f}% |".format(
                workload=row["workload_id"],
                lazy=timing["source_backed_external_seconds"],
                eager=timing["eager_external_seconds"],
                mps=timing["mps_external_seconds"],
                vs_eager=timing["speedup_over_eager"],
                vs_mps=timing["speedup_over_mps"],
                memory=100
                * row["memory"]["max_conservative_peak_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Scientific/resource gates: `{summary['all_scientific_and_resource_gates_pass']}`",
            f"- Plan signatures equal: `{summary['all_plan_signatures_equal']}`",
            f"- Wins over eager: `{summary['source_backed_wins_over_eager']}/{summary['workloads']}`",
            f"- Wins over MPS: `{summary['source_backed_wins_over_mps']}/{summary['workloads']}`",
            f"- Aggregate speedup over eager: `{summary['aggregate_speedup_over_eager']:.3f}x`",
            f"- Aggregate speedup over MPS: `{summary['aggregate_speedup_over_mps']:.3f}x`",
            f"- Convergence: `{summary['source_backed_converged']}/{summary['jobs']}` source-backed, "
            f"`{summary['eager_converged']}/{summary['jobs']}` eager, "
            f"`{summary['mps_converged']}/{summary['jobs']}` MPS",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    completion = _load(args.completion)
    if completion.get("status") != "complete":
        raise ValueError("source-backed ablation is not complete")
    rows = [analyze_row(item) for item in completion["workloads"]]
    summary = aggregate(rows)
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment": completion["experiment"],
        "source": {
            "completion_path": str(args.completion.resolve()),
            "completion_sha256": _sha256(args.completion),
        },
        "aggregate": summary,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "source_backed_analysis.json"
    csv_path = args.output_dir / "source_backed_analysis.csv"
    markdown_path = args.output_dir / "source_backed_analysis.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows, summary)
    print(
        json.dumps(
            {
                "all_scientific_and_resource_gates_pass": summary[
                    "all_scientific_and_resource_gates_pass"
                ],
                "output_dir": str(args.output_dir),
                "workloads": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
