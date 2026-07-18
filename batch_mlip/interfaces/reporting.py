"""Public trajectory, diagnostics, and checkpoint callbacks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from ase.io import write

from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation, StepCallback


def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected scalar tensor")
        value = value.detach().cpu().item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


class CompositeReporter:
    """Fan out a callback to multiple reporters."""

    def __init__(self, reporters: Iterable[StepCallback]) -> None:
        self.reporters = list(reporters)

    def __call__(self, step, state, evaluation, diagnostics) -> None:
        for reporter in self.reporters:
            reporter(step, state, evaluation, diagnostics)


class ExtXYZReporter:
    """Append every system and selected scalar diagnostics to an extxyz file."""

    def __init__(
        self,
        filename: str | Path,
        *,
        overwrite: bool = True,
        wrap: bool = False,
    ) -> None:
        self.filename = Path(filename)
        self.wrap = wrap
        self._has_written = False
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.filename.unlink(missing_ok=True)

    def __call__(
        self,
        step: int,
        state: AseGraphBatch,
        evaluation: BatchEvaluation,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        frames = state.to_ase(evaluation, wrap=self.wrap)
        for system_id, atoms in enumerate(frames):
            atoms.info["step"] = int(step)
            atoms.info["system_id"] = int(system_id)
            reserved = set(getattr(atoms.calc, "results", {}))
            for key, value in diagnostics.items():
                if isinstance(value, torch.Tensor) and value.ndim == 1:
                    output_key = key if key not in reserved and key not in atoms.info else f"diag_{key}"
                    atoms.info[output_key] = _json_scalar(value[system_id])

        write(
            self.filename,
            frames,
            format="extxyz",
            append=self._has_written,
        )
        self._has_written = True


class JSONLReporter:
    """Write one machine-readable diagnostics row per system and callback."""

    def __init__(self, filename: str | Path, *, overwrite: bool = True) -> None:
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self.filename.unlink(missing_ok=True)

    def __call__(
        self,
        step: int,
        state: AseGraphBatch,
        evaluation: BatchEvaluation,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        with self.filename.open("a", encoding="utf-8") as handle:
            for system_id in range(state.n_systems):
                row = {
                    "step": int(step),
                    "system_id": int(system_id),
                    "n_atoms": int(state.counts[system_id].item()),
                    "energy": float(evaluation.energy[system_id].detach().cpu()),
                }
                for key, value in diagnostics.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        row[key] = _json_scalar(value[system_id])
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class TorchStateCheckpointReporter:
    """Atomically overwrite a compact tensor-state checkpoint."""

    def __init__(self, filename: str | Path) -> None:
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        step: int,
        state: AseGraphBatch,
        evaluation: BatchEvaluation,
        diagnostics: dict[str, torch.Tensor],
    ) -> None:
        payload = {
            "step": int(step),
            "positions": state.positions.detach().cpu(),
            "velocities": state.velocities.detach().cpu(),
            "cells": state.cells.detach().cpu(),
            "z": state.z.detach().cpu(),
            "system_idx": state.system_idx.detach().cpu(),
            "ptr": state.ptr.detach().cpu(),
            "energy": evaluation.energy.detach().cpu(),
            "forces": evaluation.forces.detach().cpu(),
            "diagnostics": {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in diagnostics.items()
            },
        }
        temporary = self.filename.with_suffix(self.filename.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.filename)


def build_reporter(
    *,
    trajectory: str | Path | None = None,
    diagnostics: str | Path | None = None,
    checkpoint: str | Path | None = None,
    wrap: bool = False,
) -> StepCallback | None:
    reporters: list[StepCallback] = []
    if trajectory is not None:
        reporters.append(ExtXYZReporter(trajectory, wrap=wrap))
    if diagnostics is not None:
        reporters.append(JSONLReporter(diagnostics))
    if checkpoint is not None:
        reporters.append(TorchStateCheckpointReporter(checkpoint))
    if not reporters:
        return None
    if len(reporters) == 1:
        return reporters[0]
    return CompositeReporter(reporters)
