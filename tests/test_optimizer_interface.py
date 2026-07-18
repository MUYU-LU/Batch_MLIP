from __future__ import annotations

from typing import Any

import pytest
import torch
from ase import Atoms

from batch_mlip import (
    BatchCalculator,
    BatchedFIRE,
    BatchedGradientDescent,
    BatchEvaluation,
    BatchOptimizer,
    OptimizerCapabilities,
    available_optimizers,
    batched_gradient_descent,
    create_optimizer,
    register_optimizer,
    relax,
)


class QuadraticBatchCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.5, device="cpu", dtype=torch.float64)

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy
        if compute_stress:
            raise NotImplementedError
        atom_energy = 0.5 * (state.positions * state.positions).sum(dim=-1)
        energy = torch.zeros(
            state.n_systems, device=state.device, dtype=state.dtype
        )
        energy.index_add_(0, state.system_idx, atom_energy)
        return BatchEvaluation(energy=energy, forces=-state.positions.clone())


class RecordingOptimizer:
    def __init__(self) -> None:
        self.received_options: dict[str, Any] | None = None

    def capabilities(self) -> OptimizerCapabilities:
        return OptimizerCapabilities()

    def run(self, state, calculator, **options):
        self.received_options = options
        return batched_gradient_descent(state, calculator, **options)


def test_relax_accepts_a_direct_custom_optimizer_object():
    optimizer = RecordingOptimizer()
    assert isinstance(optimizer, BatchOptimizer)

    result = relax(
        Atoms("H", positions=[[0.5, 0.0, 0.0]]),
        QuadraticBatchCalculator(),
        optimizer=optimizer,
        step_size=0.1,
        max_steps=1,
        fmax=1e-12,
    )

    assert result.steps == 1
    assert optimizer.received_options == {
        "step_size": 0.1,
        "max_steps": 1,
        "fmax": 1e-12,
    }


def test_factory_builds_builtin_optimizer_objects_and_preserves_aliases():
    assert {"fire", "gd", "gradient_descent"} <= set(available_optimizers())
    assert isinstance(create_optimizer("FIRE"), BatchedFIRE)
    assert isinstance(create_optimizer("gradient-descent"), BatchedGradientDescent)

    result = relax(
        Atoms("H", positions=[[0.5, 0.0, 0.0]]),
        QuadraticBatchCalculator(),
        optimizer=create_optimizer("gd", max_steps=0),
        fmax=1e-12,
    )
    assert result.steps == 0


def test_custom_factory_can_be_registered_and_created():
    name = "recording_optimizer_for_test"
    register_optimizer(name, RecordingOptimizer)

    optimizer = create_optimizer(name)
    assert isinstance(optimizer, RecordingOptimizer)
    with pytest.raises(ValueError, match="already registered"):
        register_optimizer(name, RecordingOptimizer)


def test_invalid_optimizer_and_factory_fail_early():
    with pytest.raises(TypeError, match="registered name or implement"):
        relax(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            QuadraticBatchCalculator(),
            optimizer=object(),
        )

    register_optimizer("invalid_factory_for_test", lambda **options: object())
    with pytest.raises(TypeError, match="did not return"):
        create_optimizer("invalid_factory_for_test")


@pytest.mark.parametrize(
    "option,value,error",
    [
        ("active_compaction", True, "active-batch compaction"),
        ("cell_filter", object(), "variable-cell relaxation"),
    ],
)
def test_capability_checks_reject_unsupported_modes(option, value, error):
    optimizer = BatchedGradientDescent(**{option: value})
    with pytest.raises(ValueError, match=error):
        relax(
            Atoms("H", positions=[[0.5, 0.0, 0.0]]),
            QuadraticBatchCalculator(),
            optimizer=optimizer,
            max_steps=0,
        )


def test_unknown_name_lists_registered_choices():
    with pytest.raises(ValueError, match="available optimizers:.*fire"):
        create_optimizer("not_an_optimizer")
