"""Optional native-batch adapter for MACE models."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

import torch
from ase import Atoms

from ..core.calculator import BatchCalculator, NeighborPolicy
from ..core.math_utils import model_dtype
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation

_DEFAULT_DTYPE_LOCK = RLock()


def _mace_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from mace import data
        from mace.calculators import mace_off
        from mace.tools import torch_geometric, utils
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "MACE support requires the optional 'mace-torch' package"
        ) from exc
    return data, torch_geometric, utils, mace_off


class MACEBatchCalculator(BatchCalculator):
    """Evaluate heterogeneous structures with one native MACE graph batch.

    MACE remains responsible for constructing its ``AtomicData`` graphs and
    computing direct forces and stress. The common state only supplies ASE
    structures and keeps optimizer/MD state independent of MACE internals.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        cutoff: float | None = None,
        skin: float = 0.0,
        head: str | None = None,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
    ) -> None:
        data, torch_geometric, utils, _ = _mace_imports()
        resolved_dtype = model_dtype(model) if dtype is None else dtype
        if resolved_dtype not in (torch.float32, torch.float64):
            raise ValueError("MACE adapter dtype must be float32 or float64")

        model_cutoff = float(torch.as_tensor(model.r_max).detach().cpu())
        if cutoff is not None and abs(float(cutoff) - model_cutoff) > 1e-12:
            raise ValueError(
                f"cutoff {cutoff} does not match the MACE model cutoff {model_cutoff}"
            )
        if energy_units_to_eV <= 0.0 or length_units_to_A <= 0.0:
            raise ValueError("MACE unit conversion factors must be positive")

        super().__init__(
            cutoff=model_cutoff,
            skin=skin,
            device=device,
            dtype=resolved_dtype,
        )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._data = data
        self._torch_geometric = torch_geometric
        self.z_table = utils.AtomicNumberTable(
            [int(value) for value in self.model.atomic_numbers]
        )
        self.available_heads = list(getattr(self.model, "heads", ["Default"]))
        if not self.available_heads:
            self.available_heads = ["Default"]
        if head is None:
            defaults = [
                value for value in self.available_heads if value.lower() == "default"
            ]
            self.head = defaults[0] if defaults else self.available_heads[0]
        elif head not in self.available_heads:
            raise ValueError(
                f"unknown MACE head {head!r}; available heads: {self.available_heads}"
            )
        else:
            self.head = head
        self.energy_units_to_eV = float(energy_units_to_eV)
        self.length_units_to_A = float(length_units_to_A)

    def create_state(self, systems: Sequence[Atoms]) -> AseGraphBatch:
        unsupported = sorted(
            {
                int(number)
                for atoms in systems
                for number in atoms.numbers
                if int(number) not in self.z_table.zs
            }
        )
        if unsupported:
            raise ValueError(f"atomic numbers not supported by MACE model: {unsupported}")
        return super().create_state(systems)

    def _build_batch(self, state: AseGraphBatch) -> Any:
        systems = state.to_ase(evaluation=None, wrap=False)

        # AtomicData.from_config follows torch's default dtype. Limit the
        # temporary change to graph construction and restore process state.
        with _DEFAULT_DTYPE_LOCK:
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                dataset = [
                    self._data.AtomicData.from_config(
                        self._data.config_from_atoms(atoms, head_name=self.head),
                        z_table=self.z_table,
                        cutoff=self.cutoff,
                        heads=self.available_heads,
                    )
                    for atoms in systems
                ]
                loader = self._torch_geometric.dataloader.DataLoader(
                    dataset=dataset,
                    batch_size=len(dataset),
                    shuffle=False,
                    drop_last=False,
                )
                batch = next(iter(loader))
            finally:
                torch.set_default_dtype(previous_dtype)
        return batch.to(self.device)

    def calculate(
        self,
        state: AseGraphBatch,
        *,
        neighbor_policy: NeighborPolicy = "auto",
        compute_stress: bool = False,
    ) -> BatchEvaluation:
        if neighbor_policy not in ("auto", "always", "never"):
            raise ValueError(f"unsupported neighbor policy {neighbor_policy!r}")
        if state.device != self.device or state.dtype != self.dtype:
            raise ValueError("state device and dtype must match the MACE adapter")

        batch = self._build_batch(state)
        output = self.model(
            batch.to_dict(),
            compute_force=True,
            compute_stress=compute_stress,
            training=False,
        )
        energy = output["energy"].reshape(state.n_systems)
        forces = output["forces"].reshape(state.n_atoms, 3)
        stress = output["stress"] if compute_stress else None
        if stress is not None:
            stress = stress.reshape(state.n_systems, 3, 3)

        energy_scale = self.energy_units_to_eV
        length_scale = self.length_units_to_A
        return BatchEvaluation(
            energy=(energy * energy_scale).detach(),
            forces=(forces * energy_scale / length_scale).detach(),
            stress=(
                None
                if stress is None
                else (stress * energy_scale / length_scale**3).detach()
            ),
        )


def load_mace_off_batch(
    model: str = "small",
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    **adapter_kwargs: Any,
) -> MACEBatchCalculator:
    """Load a MACE-OFF foundation model as a native batch calculator."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_initialized():
        # Legacy pickled MACE-OFF models in some environments must be loaded
        # after CUDA runtime initialization and before importing MACE/e3nn.
        torch.empty(0, device=resolved_device)
    _, _, _, mace_off = _mace_imports()
    raw_model = mace_off(
        model=model,
        device=str(resolved_device),
        return_raw_model=True,
    )
    return MACEBatchCalculator(
        raw_model,
        device=resolved_device,
        dtype=dtype,
        **adapter_kwargs,
    )
