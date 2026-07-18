"""Public structure-level API shared by relaxation and MD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ase import Atoms

from ..core.calculator import BatchCalculator
from ..core.types import EvaluationResult, MDResult, RelaxationResult
from ..dynamics.integrators import batched_langevin_baoab, batched_velocity_verlet
from ..optimization.registry import BatchOptimizer, create_optimizer


def _normalize_systems(systems: Atoms | Sequence[Atoms]) -> list[Atoms]:
    normalized = [systems] if isinstance(systems, Atoms) else list(systems)
    if not normalized:
        raise ValueError("systems must contain at least one ASE Atoms object")
    if not all(isinstance(atoms, Atoms) for atoms in normalized):
        raise TypeError("every system must be an ASE Atoms object")
    return normalized


def evaluate(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    compute_stress: bool = False,
) -> EvaluationResult:
    """Evaluate structures in one calculator call and preserve input order."""

    state = calculator.create_state(_normalize_systems(systems))
    evaluation = calculator(state, compute_stress=compute_stress)
    return EvaluationResult(state=state, evaluation=evaluation)


def relax(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    optimizer: str | BatchOptimizer = "fire",
    **optimizer_kwargs: Any,
) -> RelaxationResult:
    """Relax structures with a registered name or optimizer object."""

    resolved = create_optimizer(optimizer) if isinstance(optimizer, str) else optimizer
    if not isinstance(resolved, BatchOptimizer):
        raise TypeError(
            "optimizer must be a registered name or implement BatchOptimizer"
        )
    normalized = _normalize_systems(systems)
    capabilities = resolved.capabilities()
    lazy_refill = (
        optimizer_kwargs.get("refill_batch_size") is not None
        and getattr(capabilities, "active_refill", False)
    )
    state = (
        calculator.create_state(normalized, build_neighbors=False)
        if lazy_refill
        else calculator.create_state(normalized)
    )
    return resolved.run(state, calculator, **optimizer_kwargs)


def molecular_dynamics(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    ensemble: Literal[
        "nve", "nvt", "nvt_langevin", "npt", "npt_mtk"
    ] = "nve",
    **md_kwargs: Any,
) -> MDResult:
    """Run fixed-cell batch MD with the same calculator used for relaxation."""

    state = calculator.create_state(_normalize_systems(systems))
    if ensemble == "nve":
        return batched_velocity_verlet(state, calculator, **md_kwargs)
    if ensemble in ("nvt", "nvt_langevin"):
        return batched_langevin_baoab(state, calculator, **md_kwargs)
    if ensemble in ("npt", "npt_mtk"):
        raise NotImplementedError(
            "the NPT API slot is reserved; no validated batch barostat is implemented"
        )
    raise ValueError(f"unsupported ensemble {ensemble!r}")
