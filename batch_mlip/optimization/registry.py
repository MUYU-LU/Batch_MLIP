"""Optimizer protocols, objects, registry, and name-based construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..core.calculator import BatchCalculator
from ..core.state import AseGraphBatch
from ..core.types import RelaxationResult
from .bfgs import batched_bfgs_relax
from .bfgs_line_search import batched_bfgs_line_search_relax
from .fire import batched_fire_relax, batched_gradient_descent


@dataclass(frozen=True)
class OptimizerCapabilities:
    """Optional relaxation modes implemented by an optimizer."""

    variable_cell: bool = False
    active_compaction: bool = False
    active_refill: bool = False


@runtime_checkable
class BatchOptimizer(Protocol):
    """Contract implemented by optimizer objects accepted by ``relax``.

    The optimizer receives the model-independent batch state and calculator.
    Implementations may consume additional keyword options and must return a
    complete :class:`RelaxationResult` in the original system order.
    """

    def capabilities(self) -> OptimizerCapabilities:
        """Declare support for optional variable-cell and compaction modes."""

    def run(
        self,
        state: AseGraphBatch,
        calculator: BatchCalculator,
        **options: Any,
    ) -> RelaxationResult:
        """Optimize ``state`` using forces supplied by ``calculator``."""


@runtime_checkable
class OptimizerFactory(Protocol):
    """Callable that constructs a :class:`BatchOptimizer`."""

    def __call__(self, **options: Any) -> BatchOptimizer:
        """Create an optimizer configured with default run options."""


def _merged_options(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    return {**defaults, **overrides}


def _validate_capabilities(
    optimizer: BatchOptimizer,
    options: Mapping[str, Any],
) -> None:
    capabilities = optimizer.capabilities()
    if not isinstance(capabilities, OptimizerCapabilities):
        raise TypeError("optimizer.capabilities() must return OptimizerCapabilities")
    if options.get("cell_filter") is not None and not capabilities.variable_cell:
        raise ValueError(
            f"{type(optimizer).__name__} does not support variable-cell relaxation"
        )
    if options.get("active_compaction", False) and not capabilities.active_compaction:
        raise ValueError(
            f"{type(optimizer).__name__} does not support active-batch compaction"
        )
    if options.get("refill_batch_size") is not None and not capabilities.active_refill:
        raise ValueError(
            f"{type(optimizer).__name__} does not support active-batch refill"
        )


class BatchedFIRE:
    """Object interface for the validated batched FIRE implementation."""

    def __init__(self, **options: Any) -> None:
        self.options = MappingProxyType(dict(options))

    def capabilities(self) -> OptimizerCapabilities:
        return OptimizerCapabilities(variable_cell=True, active_compaction=True)

    def run(
        self,
        state: AseGraphBatch,
        calculator: BatchCalculator,
        **options: Any,
    ) -> RelaxationResult:
        resolved = _merged_options(self.options, options)
        _validate_capabilities(self, resolved)
        return batched_fire_relax(state, calculator, **resolved)


class BatchedBFGS:
    """Object interface for ASE-compatible full batched BFGS."""

    def __init__(self, **options: Any) -> None:
        self.options = MappingProxyType(dict(options))

    def capabilities(self) -> OptimizerCapabilities:
        return OptimizerCapabilities(
            variable_cell=True,
            active_compaction=True,
            active_refill=True,
        )

    def run(
        self,
        state: AseGraphBatch,
        calculator: BatchCalculator,
        **options: Any,
    ) -> RelaxationResult:
        resolved = _merged_options(self.options, options)
        _validate_capabilities(self, resolved)
        return batched_bfgs_relax(state, calculator, **resolved)


class BatchedBFGSLineSearch:
    """Object interface for ASE-compatible batched BFGSLineSearch."""

    def __init__(self, **options: Any) -> None:
        self.options = MappingProxyType(dict(options))

    def capabilities(self) -> OptimizerCapabilities:
        return OptimizerCapabilities(variable_cell=True, active_compaction=True)

    def run(
        self,
        state: AseGraphBatch,
        calculator: BatchCalculator,
        **options: Any,
    ) -> RelaxationResult:
        resolved = _merged_options(self.options, options)
        _validate_capabilities(self, resolved)
        return batched_bfgs_line_search_relax(state, calculator, **resolved)


# Match ASE: QuasiNewton is another name for BFGSLineSearch.
BatchedQuasiNewton = BatchedBFGSLineSearch


class BatchedGradientDescent:
    """Object interface for the fixed-cell steepest-descent baseline."""

    def __init__(self, **options: Any) -> None:
        self.options = MappingProxyType(dict(options))

    def capabilities(self) -> OptimizerCapabilities:
        return OptimizerCapabilities()

    def run(
        self,
        state: AseGraphBatch,
        calculator: BatchCalculator,
        **options: Any,
    ) -> RelaxationResult:
        resolved = _merged_options(self.options, options)
        _validate_capabilities(self, resolved)
        return batched_gradient_descent(state, calculator, **resolved)


_OPTIMIZER_FACTORIES: dict[str, Callable[..., BatchOptimizer]] = {}


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("optimizer name must not be empty")
    return normalized


def register_optimizer(
    name: str,
    factory: Callable[..., BatchOptimizer],
    *,
    replace: bool = False,
) -> None:
    """Register a named optimizer factory for configuration-driven use."""

    normalized = _normalize_name(name)
    if not callable(factory):
        raise TypeError("optimizer factory must be callable")
    if normalized in _OPTIMIZER_FACTORIES and not replace:
        raise ValueError(f"optimizer {normalized!r} is already registered")
    _OPTIMIZER_FACTORIES[normalized] = factory


def available_optimizers() -> tuple[str, ...]:
    """Return registered optimizer names in deterministic order."""

    return tuple(sorted(_OPTIMIZER_FACTORIES))


def create_optimizer(name: str, **options: Any) -> BatchOptimizer:
    """Construct a registered optimizer with default run options."""

    normalized = _normalize_name(name)
    try:
        factory = _OPTIMIZER_FACTORIES[normalized]
    except KeyError as exc:
        choices = ", ".join(available_optimizers())
        raise ValueError(
            f"unsupported optimizer {name!r}; available optimizers: {choices}"
        ) from exc
    optimizer = factory(**options)
    if not isinstance(optimizer, BatchOptimizer):
        raise TypeError(
            f"optimizer factory {normalized!r} did not return a BatchOptimizer"
        )
    capabilities = optimizer.capabilities()
    if not isinstance(capabilities, OptimizerCapabilities):
        raise TypeError(
            f"optimizer factory {normalized!r} returned invalid capabilities"
        )
    return optimizer


register_optimizer("fire", BatchedFIRE)
register_optimizer("bfgs", BatchedBFGS)
register_optimizer("bfgslinesearch", BatchedBFGSLineSearch)
register_optimizer("bfgs_line_search", BatchedBFGSLineSearch)
register_optimizer("quasinewton", BatchedBFGSLineSearch)
register_optimizer("gradient_descent", BatchedGradientDescent)
register_optimizer("gd", BatchedGradientDescent)
