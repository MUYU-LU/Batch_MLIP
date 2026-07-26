"""Evidence-based CUDA allocator selection for fresh worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models.potential import AtomBitBatchCalculator
from ..optimization.registry import BatchedBFGS, BatchOptimizer

CudaAllocatorPolicy = Literal["auto", "native", "expandable_segments"]

_ALLOCATOR_ENVIRONMENT_KEYS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)
_EXPANDABLE_SEGMENTS = "expandable_segments:True"


@dataclass(frozen=True)
class CudaAllocatorPlan:
    """Allocator configuration to install before a worker initializes CUDA."""

    requested_policy: CudaAllocatorPolicy
    selected_policy: Literal["native", "expandable_segments"]
    reason: str

    def environment(self) -> dict[str, str | None]:
        value = (
            _EXPANDABLE_SEGMENTS
            if self.selected_policy == "expandable_segments"
            else None
        )
        return {key: value for key in _ALLOCATOR_ENVIRONMENT_KEYS}

    def metadata(self) -> dict[str, object]:
        return {
            "requested_policy": self.requested_policy,
            "selected_policy": self.selected_policy,
            "reason": self.reason,
            "environment": self.environment(),
        }


def select_cuda_allocator(
    calculator: object,
    optimizer: BatchOptimizer,
    *,
    variable_cell: bool,
    policy: CudaAllocatorPolicy = "auto",
) -> CudaAllocatorPlan:
    """Choose only allocator modes supported by measured evidence.

    Expandable segments are currently automatic only for AtomBit variable-cell
    BFGS. Other combinations remain native until a matched benchmark supports a
    broader rule. Explicit policies are useful for controlled experiments.
    """

    if policy not in ("auto", "native", "expandable_segments"):
        raise ValueError(
            "cuda_allocator_policy must be 'auto', 'native', or "
            "'expandable_segments'"
        )
    if policy == "native":
        return CudaAllocatorPlan(
            requested_policy=policy,
            selected_policy="native",
            reason="native allocator explicitly requested",
        )
    if policy == "expandable_segments":
        return CudaAllocatorPlan(
            requested_policy=policy,
            selected_policy="expandable_segments",
            reason="expandable segments explicitly requested",
        )
    if (
        isinstance(calculator, AtomBitBatchCalculator)
        and isinstance(optimizer, BatchedBFGS)
        and variable_cell
    ):
        return CudaAllocatorPlan(
            requested_policy=policy,
            selected_policy="expandable_segments",
            reason=(
                "measured AtomBit variable-cell BFGS allocator-fragmentation "
                "policy"
            ),
        )
    return CudaAllocatorPlan(
        requested_policy=policy,
        selected_policy="native",
        reason="no matched evidence for a non-native allocator",
    )
