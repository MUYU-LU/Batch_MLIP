"""Calculator-independent execution of signed controlled workloads."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import write

from ..core.calculator import BatchCalculator
from ..dynamics.integrators import (
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from ..interfaces.api import evaluate
from ..profiling import (
    RunTelemetry,
    RuntimeProfiler,
    append_run_telemetry_csv,
    runtime_profile_registry_fields,
    write_run_telemetry_json,
)
from .materialize import materialize_workload
from .schema import WorkloadManifest


@dataclass(frozen=True)
class WorkloadRunSpec:
    """Identity and execution choices that are not properties of the workload."""

    run_id: str
    study_id: str
    model_name: str
    model_checkpoint_sha256: str
    code_commit: str
    resident_batch_size: int
    equivalence_tier: str
    validation_pass: bool
    repeat_index: int = 0
    dirty_tree_hash: str | None = None
    memory_safety_fraction: float = 0.85
    compile_mode: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.resident_batch_size <= 0:
            raise ValueError("resident_batch_size must be positive")
        if re.fullmatch(r"[0-9a-f]{64}", self.model_checkpoint_sha256) is None:
            raise ValueError("model_checkpoint_sha256 must be a SHA-256 digest")
        if not 0.0 < self.memory_safety_fraction <= 1.0:
            raise ValueError("memory_safety_fraction must be in (0, 1]")


@dataclass
class WorkloadExecutionResult:
    """Ordered final structures and machine-readable records from one run."""

    structures: list[Atoms]
    telemetry: RunTelemetry
    runtime_profile: dict[str, Any]
    summary: dict[str, Any]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    _synchronize(device)
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _merge_runtime_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, float | int]] = {}
    events = []
    samples = []
    peak_memory = None
    for profile in profiles:
        events.extend(profile.get("events", []))
        samples.extend(profile.get("samples", []))
        current_peak = profile.get("peak_memory_bytes")
        if current_peak is not None:
            peak_memory = max(peak_memory or 0, int(current_peak))
        for name, values in profile.get("phases", {}).items():
            aggregate = phases.setdefault(
                name,
                {
                    "count": 0,
                    "total_seconds": 0.0,
                    "max_seconds": 0.0,
                    "total_host_seconds": 0.0,
                },
            )
            aggregate["count"] += int(values["count"])
            aggregate["total_seconds"] += float(values["total_seconds"])
            aggregate["max_seconds"] = max(
                float(aggregate["max_seconds"]), float(values["max_seconds"])
            )
            aggregate["total_host_seconds"] += float(values["total_host_seconds"])
    for values in phases.values():
        values["mean_seconds"] = float(values["total_seconds"]) / int(values["count"])
    merged = {
        "schema_version": 1,
        "total_seconds": sum(float(profile["total_seconds"]) for profile in profiles),
        "phases": phases,
        "events": events,
        "samples": samples,
    }
    if peak_memory is not None:
        merged["peak_memory_bytes"] = peak_memory
    return merged


def _gpu_fields(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"gpu_model": None, "gpu_memory_GB": None}
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_model": properties.name,
        "gpu_memory_GB": float(properties.total_memory) / 1e9,
    }


def _chunks(size: int, resident_batch_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + resident_batch_size, size))
        for start in range(0, size, resident_batch_size)
    ]


def _run_evaluation(
    systems: list[Atoms],
    calculator: BatchCalculator,
    *,
    resident_batch_size: int,
    compute_stress: bool,
) -> tuple[
    list[Atoms],
    list[dict[str, Any]],
    float,
    float,
    int | None,
    int | None,
    dict[str, Any] | None,
]:
    startup_started = time.perf_counter()
    evaluate([systems[0]], calculator, compute_stress=compute_stress)
    _synchronize(calculator.device)
    startup_seconds = time.perf_counter() - startup_started
    _reset_peak_memory(calculator.device)

    output: list[Atoms] = []
    first_result_seconds = 0.0
    with RuntimeProfiler(device=calculator.device) as profiler:
        measured_started = time.perf_counter()
        for start, stop in _chunks(len(systems), resident_batch_size):
            result = evaluate(systems[start:stop], calculator, compute_stress=compute_stress)
            output.extend(result.structures)
            if stop == min(resident_batch_size, len(systems)):
                first_result_seconds = time.perf_counter() - measured_started
    allocated, reserved = _memory_snapshot(calculator.device)
    return (
        output,
        [profiler.summary()],
        startup_seconds,
        first_result_seconds,
        allocated,
        reserved,
        None,
    )


def _run_nve(
    manifest: WorkloadManifest,
    systems: list[Atoms],
    calculator: BatchCalculator,
    *,
    resident_batch_size: int,
) -> tuple[
    list[Atoms],
    list[dict[str, Any]],
    float,
    float,
    int | None,
    int | None,
    dict[str, Any],
]:
    metadata = manifest.metadata
    timestep_fs = float(metadata["timestep_fs"])
    warmup_steps = int(metadata["warmup_steps"])
    measured_steps = int(metadata["measured_steps"])
    temperature = float(metadata["initial_temperature_K"])
    profiles = []
    output: list[Atoms] = []
    startup_seconds = 0.0
    first_result_seconds = 0.0
    peak_allocated = None
    peak_reserved = None
    measured_elapsed = 0.0
    initial_total_energy_eV: list[float] = []
    final_total_energy_eV: list[float] = []

    for start, stop in _chunks(len(systems), resident_batch_size):
        startup_started = time.perf_counter()
        state = calculator.create_state(systems[start:stop])
        seeds = [job.random_seed for job in manifest.jobs[start:stop]]
        if any(seed is None for seed in seeds):
            raise ValueError("NVE workload jobs must define per-system random seeds")
        initialize_maxwell_boltzmann(
            state,
            temperature,
            seed=[int(seed) for seed in seeds if seed is not None],
            remove_com=bool(metadata["remove_initial_com"]),
            force_exact_temperature=bool(metadata["force_exact_initial_temperature"]),
        )
        if warmup_steps:
            batched_velocity_verlet(
                state,
                calculator,
                timestep_fs=timestep_fs,
                n_steps=warmup_steps,
            )
        _synchronize(calculator.device)
        startup_seconds += time.perf_counter() - startup_started
        _reset_peak_memory(calculator.device)

        with RuntimeProfiler(device=calculator.device) as profiler:
            result = batched_velocity_verlet(
                state,
                calculator,
                timestep_fs=timestep_fs,
                n_steps=measured_steps,
            )
        profile = profiler.summary()
        profiles.append(profile)
        measured_elapsed += float(profile["total_seconds"])
        if not output:
            first_result_seconds = measured_elapsed
        output.extend(result.structures)
        if result.initial_total_energy is None:
            raise RuntimeError("NVE integrator did not report its initial total energy")
        initial_total_energy_eV.extend(result.initial_total_energy.detach().cpu().tolist())
        final_total_energy_eV.extend(
            (result.evaluation.energy + result.kinetic_energy).detach().cpu().tolist()
        )
        allocated, reserved = _memory_snapshot(calculator.device)
        if allocated is not None:
            peak_allocated = max(peak_allocated or 0, allocated)
            peak_reserved = max(peak_reserved or 0, int(reserved or 0))
    drift_eV = [
        final - initial
        for initial, final in zip(initial_total_energy_eV, final_total_energy_eV, strict=True)
    ]
    abs_drift_per_atom = [
        abs(drift) / job.atom_count for drift, job in zip(drift_eV, manifest.jobs, strict=True)
    ]
    diagnostics = {
        "initial_total_energy_eV": initial_total_energy_eV,
        "final_total_energy_eV": final_total_energy_eV,
        "total_energy_drift_eV": drift_eV,
        "mean_abs_energy_drift_eV_per_atom": sum(abs_drift_per_atom) / len(abs_drift_per_atom),
        "rms_energy_drift_eV_per_atom": (
            sum(value * value for value in abs_drift_per_atom) / len(abs_drift_per_atom)
        )
        ** 0.5,
        "max_abs_energy_drift_eV_per_atom": max(abs_drift_per_atom),
    }
    return (
        output,
        profiles,
        startup_seconds,
        first_result_seconds,
        peak_allocated,
        peak_reserved,
        diagnostics,
    )


def execute_workload(
    manifest: WorkloadManifest,
    dataset_dir: str | Path,
    calculator: BatchCalculator,
    spec: WorkloadRunSpec,
    *,
    output_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> WorkloadExecutionResult:
    """Execute a static or NVE workload with ordered resident microbatches."""

    if manifest.operation not in {"force_evaluation", "md_nve"}:
        raise ValueError("unified workload runner currently supports force_evaluation and md_nve")
    io_started = time.perf_counter()
    systems = materialize_workload(manifest, dataset_dir)
    io_seconds = time.perf_counter() - io_started
    resident_batch_size = min(spec.resident_batch_size, len(systems))
    started_at = datetime.now(timezone.utc)

    if manifest.operation == "force_evaluation":
        run_output = _run_evaluation(
            systems,
            calculator,
            resident_batch_size=resident_batch_size,
            compute_stress=bool(manifest.metadata.get("compute_stress", False)),
        )
        algorithm = "single_point"
        useful_units = len(systems)
        useful_unit_name = "structures"
    else:
        run_output = _run_nve(
            manifest,
            systems,
            calculator,
            resident_batch_size=resident_batch_size,
        )
        algorithm = "velocity_verlet"
        useful_units = len(systems) * int(manifest.metadata["measured_steps"])
        useful_unit_name = "replica_steps"
    structures, profiles, startup_seconds, first_result, allocated, reserved, diagnostics = (
        run_output
    )
    ended_at = datetime.now(timezone.utc)
    profile = _merge_runtime_profiles(profiles)
    measured = runtime_profile_registry_fields(profile)
    measured["peak_allocated_GB"] = None if allocated is None else allocated / 1e9

    gpu = _gpu_fields(calculator.device)
    telemetry = RunTelemetry.create(
        run_id=spec.run_id,
        study_id=spec.study_id,
        workload_id=manifest.workload_id,
        workload_manifest_sha256=manifest.manifest_sha256,
        model_name=spec.model_name,
        model_checkpoint_sha256=spec.model_checkpoint_sha256,
        code_commit=spec.code_commit,
        dirty_tree_hash=spec.dirty_tree_hash,
        algorithm=algorithm,
        cell_mode=manifest.cell_mode,
        force_mode=getattr(calculator, "force_mode", None),
        model_dtype=str(calculator.dtype).removeprefix("torch."),
        optimizer_dtype=None,
        skin_A=calculator.skin,
        cache_policy=getattr(calculator, "graph_mode", "calculator_auto"),
        refill_policy="disabled",
        batch_policy="ordered_contiguous_microbatches",
        resident_graph_limit=resident_batch_size,
        memory_safety_fraction=spec.memory_safety_fraction,
        micro_pool_size=resident_batch_size,
        gpu_count=1,
        worker_mode="single_process_persistent_model",
        cold_or_warm="warm",
        compile_mode=spec.compile_mode,
        seed="per-job-manifest" if manifest.operation == "md_nve" else None,
        repeat_index=spec.repeat_index,
        start_timestamp=started_at.isoformat(),
        end_timestamp=ended_at.isoformat(),
        io_time_s=io_seconds,
        startup_time_s=startup_seconds,
        peak_reserved_GB=None if reserved is None else reserved / 1e9,
        accepted_jobs=len(structures),
        failed_jobs=0,
        time_to_first_result_s=first_result,
        equivalence_tier=spec.equivalence_tier,
        validation_pass=spec.validation_pass,
        notes=spec.notes or "wall_time_s excludes CIF I/O and startup/warm-up",
        **measured,
        **gpu,
    )
    wall_time = float(telemetry.values["wall_time_s"])
    end_to_end_time = io_seconds + startup_seconds + wall_time
    summary = {
        "schema_version": 1,
        "run_spec": asdict(spec),
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "operation": manifest.operation,
        "jobs": len(systems),
        "unique_structures": len({job.normalized_structure_sha256 for job in manifest.jobs}),
        "resident_batch_size": resident_batch_size,
        "microbatch_count": len(_chunks(len(systems), resident_batch_size)),
        "useful_unit": useful_unit_name,
        "useful_units": useful_units,
        "throughput_per_s": None if wall_time == 0.0 else useful_units / wall_time,
        "wall_time_s": wall_time,
        "end_to_end_time_s": end_to_end_time,
        "timing_scope": {
            "wall_time_s": "synchronized measured region only",
            "end_to_end_time_s": (
                "verified input materialization + startup/warm-up + measured region; "
                "excludes calculator construction and output serialization"
            ),
        },
        "io_time_s": io_seconds,
        "startup_time_s": startup_seconds,
        "peak_allocated_GB": telemetry.values["peak_allocated_GB"],
        "peak_reserved_GB": telemetry.values["peak_reserved_GB"],
        "output_system_ids": [atoms.info["workload_system_id"] for atoms in structures],
    }
    if diagnostics is not None:
        summary["md_energy"] = diagnostics
    result = WorkloadExecutionResult(structures, telemetry, profile, summary)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_run_telemetry_json(output / "telemetry.json", telemetry)
        (output / "runtime_profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write(output / "final.extxyz", structures, format="extxyz")
    if registry_path is not None:
        append_run_telemetry_csv(registry_path, telemetry)
    return result
