from __future__ import annotations

import importlib
import pickle

import atombit_batch
from atombit_batch.core.calculator import BatchCalculator
from atombit_batch.core.state import AseGraphBatch
from atombit_batch.interfaces.api import relax
from atombit_batch.models.loaders import build_model
from atombit_batch.models.mace import MACEBatchCalculator
from atombit_batch.models.potential import BatchedPotential
from atombit_batch.models.toy_models import QuadraticWellModel
from atombit_batch.optimization.registry import BatchedBFGS

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
    assert atombit_batch.BatchCalculator is BatchCalculator
    assert atombit_batch.AseGraphBatch is AseGraphBatch
    assert atombit_batch.BatchedPotential is BatchedPotential
    assert atombit_batch.BatchedBFGS is BatchedBFGS
    assert atombit_batch.MACEBatchCalculator is MACEBatchCalculator
    assert atombit_batch.relax is relax


def test_all_legacy_module_paths_alias_canonical_modules():
    for legacy_name, canonical_name in LEGACY_MODULES.items():
        legacy = importlib.import_module(f"atombit_batch.{legacy_name}")
        canonical = importlib.import_module(f"atombit_batch.{canonical_name}")
        assert legacy is canonical
        assert getattr(atombit_batch, legacy_name) is canonical


def test_legacy_yaml_factory_path_still_builds_a_model():
    model = build_model(
        "atombit_batch.toy_models:build_quadratic_model",
        {"k": 2.0},
    )
    assert isinstance(model, QuadraticWellModel)
    assert model.k == 2.0


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
