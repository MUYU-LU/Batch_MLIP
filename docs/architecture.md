# Architecture

## Data path

```text
list[ase.Atoms]
       |
       v
AseGraphBatch.from_ase
  - concatenate z, positions, masses, velocities
  - create system_idx and ptr
  - build each graph's ijS neighbour list
  - offset atom indices and concatenate edges
       |
       v
BatchedPotential
  - optional neighbour-list rebuild
  - create GraphData attributes
  - one model forward for B systems
  - direct forces or -grad(sum(E_g), positions)
  - optional dE/dstrain stress
       |
       +------------------+
       |                  |
       v                  v
batched_fire_relax     batched MD
per-system dt/alpha    velocity-Verlet / BAOAB
       |                  |
       +------------------+
               |
               v
reporters -> extxyz, JSONL, tensor checkpoint, summary JSON
```

## Batch tensor contract

For systems with atom counts `n_0 ... n_(B-1)`, `N = sum n_g`. Atomic tensors are concatenated. `ptr[g]:ptr[g+1]` selects graph `g`; `system_idx[i]` gives the graph owning atom `i`.

Neighbour lists are constructed per ASE object before concatenation. The atom offset is added to both rows of each local `edge_index`, which guarantees graph isolation. `assert_graph_integrity()` checks this invariant after every rebuild.

## Why no runtime PyTorch Geometric dependency

The uploaded model accesses `data` through attributes only. `GraphData` supplies those attributes and therefore avoids installing the full PyG stack for inference. Models that genuinely require PyG methods can be adapted in a factory or the container can be replaced with a PyG `Data`/`Batch` object.

## Force calculation

For conservative forces:

```python
energy = model(data).reshape(B)
forces = -torch.autograd.grad(energy.sum(), positions)[0]
```

The graph energies can be summed because cross-system dependencies are prohibited. A separately predicted direct-force head can be selected, but it must be validated for energy consistency before NVE use.

## Periodic shifts

The AtomBit model computes:

```text
r_ji = r_center - r_neighbor - S_ij @ cell_graph
```

`shifts_int` therefore stores the integer ASE image shift without converting it to Cartesian coordinates in the batch builder.

## Neighbour skin

A neighbour list built at `cutoff + skin` remains valid until an atom has moved more than `skin/2` from the rebuild reference. Internally unwrapped coordinates are retained during MD, avoiding false rebuild triggers at periodic boundaries. Optional output wrapping does not alter the model state unless `wrap_interval` is requested.

## Per-system optimizer state

FIRE state is shaped by graph:

```text
dt[B]
alpha[B]
n_positive[B]
converged_step[B]
```

The fictitious velocity is atomic: `velocity[N,3]`. The atom-to-graph map broadcasts per-system parameters to atoms. Converged graphs are frozen but are not yet physically removed from the inference batch.

## Units

- position: Å
- velocity: Å/fs
- time: fs
- energy: eV
- force: eV/Å
- mass: atomic mass unit
- stress: eV/Å³
- temperature: K
- Langevin friction: fs⁻¹
MD acceleration uses the explicit conversion constant in `state.py`.
