"""Evidence-based CUDA allocator selection for fresh worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..optimization.registry import BatchedBFGS, BatchedFIRE, BatchOptimizer

CudaAllocatorPolicy = Literal["auto", "native", "expandable_segments"]

_ALLOCATOR_ENVIRONMENT_KEYS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)
# PyTorch 2.9.1 reports the new spelling but only applies expandable segments
# reliably through the deprecated CUDA-prefixed spelling. Keep both identical
# until the installed-version matrix proves the compatibility alias unnecessary.
_EXPANDABLE_SEGMENTS = "expandable_segments:True"

_AUTO_ALLOCATOR_RULES = {
    ("atombit", "bfgs", True): (
        "expandable_segments",
        "measured AtomBit variable-cell FIRE/BFGS fragmentation policy",
    ),
    ("atombit", "fire", True): (
        "expandable_segments",
        "measured AtomBit variable-cell FIRE/BFGS fragmentation policy",
    ),
    ("mace", "bfgs", True): (
        "expandable_segments",
        "measured MACE variable-cell BFGS fragmentation policy",
    ),
}


def _optimizer_policy_family(optimizer: BatchOptimizer) -> str:
    if isinstance(optimizer, BatchedBFGS):
        return "bfgs"
    if isinstance(optimizer, BatchedFIRE):
        return "fire"
    return type(optimizer).__name__.lower()


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

    Expandable segments are automatic for measured AtomBit variable-cell FIRE
    and BFGS workloads and MACE variable-cell BFGS workloads. Other
    combinations remain native until a matched benchmark supports a broader
    rule. Explicit policies remain available for controlled experiments.
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
    calculator_family = str(
        getattr(calculator, "execution_policy_family", "generic")
    )
    rule = _AUTO_ALLOCATOR_RULES.get(
        (calculator_family, _optimizer_policy_family(optimizer), variable_cell)
    )
    if rule is not None:
        selected_policy, reason = rule
        return CudaAllocatorPlan(
            requested_policy=policy,
            selected_policy=selected_policy,
            reason=reason,
        )
    return CudaAllocatorPlan(
        requested_policy=policy,
        selected_policy="native",
        reason="no matched evidence for a non-native allocator",
    )
