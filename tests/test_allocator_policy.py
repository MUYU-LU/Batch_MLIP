from __future__ import annotations

import pytest

from batch_mlip import (
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    BatchedBFGS,
    BatchedFIRE,
    MACEBatchCalculator,
    select_cuda_allocator,
)


def test_auto_allocator_targets_measured_atombit_variable_cell_optimizers():
    calculator = object.__new__(AtomBitBatchCalculator)

    selected = select_cuda_allocator(
        calculator,
        BatchedBFGS(),
        variable_cell=True,
    )
    fixed_cell = select_cuda_allocator(
        calculator,
        BatchedBFGS(),
        variable_cell=False,
    )
    fire = select_cuda_allocator(
        calculator,
        BatchedFIRE(),
        variable_cell=True,
    )

    assert selected.selected_policy == "expandable_segments"
    assert selected.environment() == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    assert fixed_cell.selected_policy == "native"
    assert fire.selected_policy == "expandable_segments"


def test_auto_allocator_keeps_mace_native():
    plan = select_cuda_allocator(
        object.__new__(MACEBatchCalculator),
        BatchedBFGS(),
        variable_cell=True,
    )

    assert plan.selected_policy == "native"
    assert set(plan.environment().values()) == {None}


def test_auto_allocator_uses_declared_calculator_policy_family():
    calculator = type(
        "ExternalAtomBitAdapter",
        (),
        {"execution_policy_family": "atombit"},
    )()

    plan = select_cuda_allocator(
        calculator,
        BatchedBFGS(),
        variable_cell=True,
    )

    assert plan.selected_policy == "expandable_segments"


def test_explicit_allocator_policy_overrides_auto_selection():
    calculator = object.__new__(AtomBitBatchCalculator)

    native = select_cuda_allocator(
        calculator,
        BatchedBFGS(),
        variable_cell=True,
        policy="native",
    )
    expandable = select_cuda_allocator(
        object(),
        BatchedFIRE(),
        variable_cell=False,
        policy="expandable_segments",
    )

    assert native.selected_policy == "native"
    assert expandable.selected_policy == "expandable_segments"


def test_auto_scheduler_rejects_unknown_allocator_policy():
    with pytest.raises(ValueError, match="cuda_allocator_policy"):
        AutoSchedulerConfig(cuda_allocator_policy="unknown")  # type: ignore[arg-type]
