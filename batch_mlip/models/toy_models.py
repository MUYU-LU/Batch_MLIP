"""Small batch-capable models for tests and integration experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core.math_utils import scatter_sum


class QuadraticWellModel(nn.Module):
    """Independent isotropic wells: E = 1/2 k sum_i |r_i-center|^2.

    This is not translationally invariant and is not a physical MLIP. It is a
    deterministic reference potential for optimizer and integrator tests.
    """

    def __init__(self, k: float = 1.0, center: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(float(k), dtype=torch.float64))
        self.register_buffer("center", torch.tensor(float(center), dtype=torch.float64))

    def forward(self, data):
        atomic = 0.5 * self.k * ((data.pos - self.center) ** 2).sum(dim=-1)
        return scatter_sum(atomic, data.batch, data.num_graphs)


class PairHarmonicModel(nn.Module):
    """Directed-edge harmonic pair model with smooth cosine cutoff."""

    def __init__(self, k: float = 5.0, r0: float = 1.2, cutoff: float = 4.0) -> None:
        super().__init__()
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.register_buffer("k", torch.tensor(float(k), dtype=torch.float64))
        self.register_buffer("r0", torch.tensor(float(r0), dtype=torch.float64))
        self.register_buffer("cutoff", torch.tensor(float(cutoff), dtype=torch.float64))

    def forward(self, data):
        center, neighbor = data.edge_index
        if center.numel() == 0:
            return torch.zeros(
                data.num_graphs, device=data.pos.device, dtype=data.pos.dtype
            )
        shifts = torch.matmul(
            data.shifts_int.unsqueeze(1).to(data.pos.dtype),
            data.cell[data.batch[center]],
        ).squeeze(1)
        vector = data.pos[center] - data.pos[neighbor] - shifts
        distance = torch.linalg.vector_norm(vector, dim=-1).clamp_min(1e-12)
        x = distance / self.cutoff
        envelope = torch.where(
            x < 1.0,
            0.5 * (torch.cos(math.pi * x) + 1.0),
            torch.zeros_like(x),
        )
        # ASE/matscipy returns both i->j and j->i, hence factor 1/4 rather
        # than 1/2 for each directed edge.
        edge_energy = 0.25 * self.k * (distance - self.r0) ** 2 * envelope
        return scatter_sum(edge_energy, data.batch[center], data.num_graphs)


class PairLennardJonesModel(nn.Module):
    """Simple graph-based Lennard-Jones reference model."""

    def __init__(
        self,
        epsilon: float = 0.0103,
        sigma: float = 3.4,
        cutoff: float = 8.5,
    ) -> None:
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(float(epsilon), dtype=torch.float64))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float64))
        self.register_buffer("cutoff", torch.tensor(float(cutoff), dtype=torch.float64))

    def forward(self, data):
        center, neighbor = data.edge_index
        if center.numel() == 0:
            return torch.zeros(
                data.num_graphs, device=data.pos.device, dtype=data.pos.dtype
            )
        shifts = torch.matmul(
            data.shifts_int.unsqueeze(1).to(data.pos.dtype),
            data.cell[data.batch[center]],
        ).squeeze(1)
        vector = data.pos[center] - data.pos[neighbor] - shifts
        distance = torch.linalg.vector_norm(vector, dim=-1).clamp_min(1e-8)
        sr6 = (self.sigma / distance) ** 6
        raw = 4.0 * self.epsilon * (sr6 * sr6 - sr6)
        src6 = (self.sigma / self.cutoff) ** 6
        shifted = raw - 4.0 * self.epsilon * (src6 * src6 - src6)
        edge_energy = torch.where(distance < self.cutoff, 0.5 * shifted, 0.0)
        return scatter_sum(edge_energy, data.batch[center], data.num_graphs)


def build_quadratic_model(**kwargs) -> QuadraticWellModel:
    return QuadraticWellModel(**kwargs)


def build_pair_harmonic_model(**kwargs) -> PairHarmonicModel:
    return PairHarmonicModel(**kwargs)


def build_lennard_jones_model(**kwargs) -> PairLennardJonesModel:
    return PairLennardJonesModel(**kwargs)
