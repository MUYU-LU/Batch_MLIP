"""Template for adapting another graph MLIP to the engine.

The engine passes a small attribute container with:
  z [N], pos [N,3], cell [B,3,3], edge_index [2,E], shifts_int [E,3],
  batch [N], and num_graphs.
"""

import torch


class MyBatchModel(torch.nn.Module):
    def __init__(self, wrapped_model):
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, data):
        # Translate the generic fields into your model's batch format.
        prediction = self.wrapped_model(
            atomic_numbers=data.z,
            positions=data.pos,
            cells=data.cell,
            edge_index=data.edge_index,
            cell_shifts=data.shifts_int,
            graph_index=data.batch,
            n_graphs=data.num_graphs,
        )
        # Required: one total energy per graph. Optional: direct forces.
        return {
            "energy": prediction["energy"],
            "forces": prediction.get("forces"),
        }
