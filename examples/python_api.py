"""Minimal Python-API example using a deterministic reference potential."""

import torch
from ase.io import read, write

from atombit_batch import (
    AseGraphBatch,
    BatchedPotential,
    JSONLReporter,
    batched_fire_relax,
)
from atombit_batch.toy_models import QuadraticWellModel

systems = read("data/demo.extxyz", index=":")
state = AseGraphBatch.from_ase(
    systems,
    cutoff=4.0,
    skin=0.5,
    device="cpu",
    dtype=torch.float64,
)
potential = BatchedPotential(
    QuadraticWellModel(k=1.0),
    device="cpu",
    dtype=torch.float64,
    force_mode="autograd",
)
result = batched_fire_relax(
    state,
    potential,
    fmax=1e-4,
    max_steps=500,
    callback=JSONLReporter("runs/python_api/diagnostics.jsonl"),
    callback_interval=5,
)
write(
    "runs/python_api/final.extxyz",
    result.state.to_ase(result.evaluation, wrap=True),
)
print("converged:", result.converged.tolist())
print("steps:", result.steps)
