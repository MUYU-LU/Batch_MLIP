"""Optional native-batch adapter for MACE models."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any, Literal

import torch
from ase import Atoms

from ..core.calculator import BatchCalculator, NeighborPolicy
from ..core.math_utils import model_dtype
from ..core.neighbors import NeighborBackend
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation
from ..profiling.runtime import profile_event, profile_phase

_DEFAULT_DTYPE_LOCK = RLock()
MACEGraphMode = Literal["cached", "rebuild"]


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

    Cached mode projects the common tensor state directly into MACE inputs and
    reuses its candidate graph. Rebuild mode retains MACE ``AtomicData`` graph
    construction as a compatibility and validation path.
    """

    @classmethod
    def from_off(
        cls,
        model: str = "small",
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        **adapter_kwargs: Any,
    ) -> MACEBatchCalculator:
        """Load a MACE-OFF foundation model as a native batch calculator."""

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_initialized():
            # Some legacy MACE-OFF checkpoints must be loaded after CUDA
            # initialization and before importing MACE/e3nn.
            torch.empty(0, device=resolved_device)
        _, _, _, mace_off = _mace_imports()
        raw_model = mace_off(
            model=model,
            device=str(resolved_device),
            return_raw_model=True,
        )
        return cls(
            raw_model,
            device=resolved_device,
            dtype=dtype,
            **adapter_kwargs,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        cutoff: float | None = None,
        skin: float = 0.0,
        neighbor_backend: NeighborBackend = "auto",
        graph_mode: MACEGraphMode = "rebuild",
        head: str | None = None,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
    ) -> None:
        data, torch_geometric, utils, _ = _mace_imports()
        resolved_dtype = model_dtype(model) if dtype is None else dtype
        if resolved_dtype not in (torch.float32, torch.float64):
            raise ValueError("MACE adapter dtype must be float32 or float64")
        if graph_mode not in ("cached", "rebuild"):
            raise ValueError("MACE graph_mode must be 'cached' or 'rebuild'")

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
            neighbor_backend=neighbor_backend,
        )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.graph_mode = graph_mode
        cached_base_supported = any(
            base.__name__ in {"MACE", "ScaleShiftMACE"}
            for base in type(self.model).mro()
        )
        embedding_specs = getattr(self.model, "embedding_specs", {})
        if graph_mode == "cached" and (
            not cached_base_supported or embedding_specs
        ):
            raise ValueError(
                "cached MACE graphs currently support standard energy MACE models "
                "without additional embedding features; use graph_mode='rebuild'"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._data = data
        self._torch_geometric = torch_geometric
        self.z_table = utils.AtomicNumberTable(
            [int(value) for value in self.model.atomic_numbers]
        )
        max_atomic_number = max(self.z_table.zs)
        self._z_to_index = torch.full(
            (max_atomic_number + 1,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        for index, atomic_number in enumerate(self.z_table.zs):
            self._z_to_index[int(atomic_number)] = index
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
        self._head_index = self.available_heads.index(self.head)
        self.energy_units_to_eV = float(energy_units_to_eV)
        self.length_units_to_A = float(length_units_to_A)

    def create_state(
        self,
        systems: Sequence[Atoms],
        *,
        build_neighbors: bool = True,
    ) -> AseGraphBatch:
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
        return super().create_state(
            systems,
            build_neighbors=build_neighbors and self.graph_mode == "cached",
        )

    def _build_batch(self, state: AseGraphBatch) -> Any:
        with profile_phase(
            "graph.state_to_ase",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
        ):
            systems = state.to_ase(evaluation=None, wrap=False)

        # AtomicData.from_config follows torch's default dtype. Limit the
        # temporary change to graph construction and restore process state.
        with _DEFAULT_DTYPE_LOCK:
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                with profile_phase(
                    "graph.mace_atomic_data",
                    device=state.device,
                    systems=state.n_systems,
                    atoms=state.n_atoms,
                ):
                    dataset = [
                        self._data.AtomicData.from_config(
                            self._data.config_from_atoms(
                                atoms, head_name=self.head
                            ),
                            z_table=self.z_table,
                            cutoff=self.cutoff,
                            heads=self.available_heads,
                        )
                        for atoms in systems
                    ]
                with profile_phase(
                    "graph.mace_collate",
                    device=state.device,
                    systems=state.n_systems,
                    atoms=state.n_atoms,
                ):
                    loader = self._torch_geometric.dataloader.DataLoader(
                        dataset=dataset,
                        batch_size=len(dataset),
                        shuffle=False,
                        drop_last=False,
                    )
                    batch = next(iter(loader))
            finally:
                torch.set_default_dtype(previous_dtype)
        with profile_phase(
            "graph.to_device",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=batch.edge_index.shape[1],
        ):
            batch = batch.to(self.device)
        profile_event(
            "native_graph",
            adapter="mace",
            graph_mode="rebuild",
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=batch.edge_index.shape[1],
        )
        return batch

    def _build_cached_input(self, state: AseGraphBatch) -> dict[str, torch.Tensor]:
        """Project persistent common tensors into the native MACE input schema."""

        graph = state.as_model_data()
        with profile_phase(
            "graph.mace_tensor_state",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=graph.edge_index.shape[1],
        ):
            node_indices = self._z_to_index[state.z]
            if bool((node_indices < 0).any()):
                raise ValueError("state contains atomic numbers unsupported by MACE")
            node_attrs = torch.nn.functional.one_hot(
                node_indices, num_classes=len(self.z_table)
            ).to(dtype=self.dtype)
            unit_shifts = graph.shifts_int.to(dtype=self.dtype)
            edge_systems = state.system_idx[graph.edge_index[0]]
            shifts = torch.einsum(
                "ei,eij->ej", unit_shifts, state.cells[edge_systems]
            )
            model_input = {
                "positions": state.positions.detach().clone(),
                "cell": state.cells.detach(),
                "ptr": state.ptr,
                "batch": state.system_idx,
                "edge_index": graph.edge_index,
                "unit_shifts": unit_shifts,
                "shifts": shifts,
                "node_attrs": node_attrs,
                "head": torch.full(
                    (state.n_systems,),
                    self._head_index,
                    device=state.device,
                    dtype=torch.long,
                ),
            }
        profile_event(
            "native_graph",
            adapter="mace",
            graph_mode="cached",
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=graph.edge_index.shape[1],
        )
        return model_input

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

        rebuilds_before = state.neighbor_rebuild_count
        if self.graph_mode == "cached":
            with profile_phase(
                "calculator.neighbor_update",
                device=state.device,
                systems=state.n_systems,
                atoms=state.n_atoms,
            ):
                if neighbor_policy == "always":
                    state.rebuild_neighbor_list()
                elif neighbor_policy == "auto":
                    state.ensure_neighbor_list()
            model_input = self._build_cached_input(state)
            edge_count = model_input["edge_index"].shape[1]
        else:
            batch = self._build_batch(state)
            model_input = batch.to_dict()
            edge_count = batch.edge_index.shape[1]
        with profile_phase(
            "model.forward",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=edge_count,
        ):
            output = self.model(
                model_input,
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
        profile_event(
            "model_evaluation",
            adapter="mace",
            graph_mode=self.graph_mode,
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=edge_count,
            neighbor_rebuilds=(
                state.neighbor_rebuild_count - rebuilds_before
                if self.graph_mode == "cached"
                else 1
            ),
            compute_stress=compute_stress,
        )
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
    dtype: str | torch.dtype = torch.float64,
    **adapter_kwargs: Any,
) -> MACEBatchCalculator:
    """Compatibility wrapper for :meth:`MACEBatchCalculator.from_off`."""

    from .loaders import parse_dtype

    return MACEBatchCalculator.from_off(
        model=model,
        device=device,
        dtype=parse_dtype(dtype),
        **adapter_kwargs,
    )
