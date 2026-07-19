from __future__ import annotations

import importlib
import pickle

import atombit_batch
import batch_mlip
from batch_mlip.core.calculator import BatchCalculator
from batch_mlip.core.state import AseGraphBatch
from batch_mlip.interfaces.api import relax
from batch_mlip.models.loaders import build_model
from batch_mlip.models.mace import MACEBatchCalculator
from batch_mlip.models.potential import AtomBitBatchCalculator
from batch_mlip.models.toy_models import QuadraticWellModel
from batch_mlip.optimization.registry import BatchedBFGS
from batch_mlip.profiling import RuntimeProfiler

LEGACY_MODULES = {
    "api": "interfaces.api",
    "bfgs": "optimization.bfgs",
    "calculator": "core.calculator",
    "cli": "interfaces.cli",
    "config": "interfaces.config",
    "filters": "optimization.cell_filters",
    "loaders": "models.loaders",
    "math_utils": "core.math_utils",
    "md": "dynamics.integrators",
    "neighbors": "core.neighbors",
    "optimize": "optimization.fire",
    "optimizers": "optimization.registry",
    "potential": "models.potential",
    "reporting": "interfaces.reporting",
    "state": "core.state",
    "toy_models": "models.toy_models",
    "types": "core.types",
}


def test_root_public_symbols_keep_their_new_canonical_identities():
    assert batch_mlip.BatchCalculator is BatchCalculator
    assert batch_mlip.AseGraphBatch is AseGraphBatch
    assert batch_mlip.AtomBitBatchCalculator is AtomBitBatchCalculator
    assert batch_mlip.BatchedBFGS is BatchedBFGS
    assert batch_mlip.MACEBatchCalculator is MACEBatchCalculator
    assert batch_mlip.RuntimeProfiler is RuntimeProfiler
    assert callable(batch_mlip.MACEBatchCalculator.from_off)
    assert batch_mlip.relax is relax


def test_pre_rename_root_symbols_are_compatibility_aliases():
    assert atombit_batch.AtomBitBatchCalculator is AtomBitBatchCalculator
    assert atombit_batch.BatchedPotential is AtomBitBatchCalculator
    assert batch_mlip.BatchedPotential is AtomBitBatchCalculator
    assert atombit_batch.FrechetCellFilter is batch_mlip.FrechetCellFilter
    assert atombit_batch.RuntimeProfiler is RuntimeProfiler
    assert atombit_batch.BatchedFrechetCellFilter is batch_mlip.FrechetCellFilter
    assert atombit_batch.relax is relax


def test_all_legacy_module_paths_alias_canonical_modules():
    for legacy_name, canonical_name in LEGACY_MODULES.items():
        legacy = importlib.import_module(f"batch_mlip.{legacy_name}")
        canonical = importlib.import_module(f"batch_mlip.{canonical_name}")
        assert legacy is canonical
        assert getattr(batch_mlip, legacy_name) is canonical

        old_namespace = importlib.import_module(f"atombit_batch.{legacy_name}")
        old_canonical = importlib.import_module(f"atombit_batch.{canonical_name}")
        assert old_namespace is canonical
        assert old_canonical is canonical


def test_new_and_legacy_yaml_factory_paths_build_models():
    model = build_model(
        "batch_mlip.toy_models:build_quadratic_model",
        {"k": 2.0},
    )
    assert isinstance(model, QuadraticWellModel)
    assert model.k == 2.0

    legacy_model = build_model(
        "atombit_batch.toy_models:build_quadratic_model",
        {"k": 4.0},
    )
    assert isinstance(legacy_model, QuadraticWellModel)
    assert legacy_model.k == 4.0


def test_legacy_serialized_model_path_remains_loadable():
    original_module = QuadraticWellModel.__module__
    try:
        QuadraticWellModel.__module__ = "atombit_batch.toy_models"
        payload = pickle.dumps(QuadraticWellModel(k=3.0))
    finally:
        QuadraticWellModel.__module__ = original_module

    model = pickle.loads(payload)
    assert isinstance(model, QuadraticWellModel)
    assert model.k == 3.0
