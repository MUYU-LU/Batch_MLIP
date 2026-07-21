# Architecture

## Data path

```text
list[ase.Atoms]
       |
       v
AseGraphBatch.from_ase
  - concatenate z, positions, masses, velocities
  - create system_idx and ptr
  - select CPU or dense CUDA neighbour construction
  - offset atom indices and concatenate edges
       |
       v
AtomBitBatchCalculator
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

Neighbour lists are constructed independently per system. The CPU path uses
matscipy for fully periodic full-rank cells and ASE otherwise. The dense CUDA
path groups compatible atom counts and image ranges, evaluates candidates in
float64 under a temporary-memory budget, and emits canonical center, neighbor,
and integer-shift ordering. The atom offset is added to both rows of each local
`edge_index`, which guarantees graph isolation. `assert_graph_integrity()`
checks this invariant after every rebuild.

`neighbor_backend="auto"` uses conservative cutover rules measured for short
(MACE-like) and long (AtomBit-like) cutoffs. Explicit `matscipy` and
`cuda_dense` modes are available for validation. Auto falls back to CPU for
degenerate periodic geometry; explicit CUDA raises instead of silently changing
the requested method.

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

A neighbour list built at `cutoff + skin` stores candidate topology. Current
distances are filtered at the exact physical cutoff on GPU before model
evaluation, so skin-only edges do not enter message passing. Fixed cells use
the standard `skin/2` displacement criterion. Fully periodic changing cells
use a conservative bound combining non-affine atomic motion and inverse cell
deformation. Invalidity and rebuilding are per structure. Internally unwrapped
coordinates are retained during MD, avoiding false rebuild triggers at
periodic boundaries.

## Per-system optimizer state

FIRE state is shaped by graph:

```text
dt[B]
alpha[B]
n_positive[B]
converged_step[B]
```

The fictitious velocity is atomic: `velocity[N,3]`. The atom-to-graph map
broadcasts per-system parameters to atoms. Masked optimization freezes
converged graphs, active compaction removes them from the inference batch, and
active refill can insert pending systems up to a bounded resident capacity.
The optional `planning` layer profiles atoms, directed edges, and optimizer
state, then emits memory-safe queues and resident capacities. It does not enter
the calculator or optimizer contracts and does not execute work automatically.

Full BFGS retains one independent dense Hessian per system. Equal-dimensional
small CUDA Hessians can be stacked for vectorized updates and a grouped
eigendecomposition; large Hessians and singleton groups retain independent
serial eigensolves. The automatic boundary is a measured execution policy, not
a change to the BFGS equations or optimizer protocol.

## Runtime profiling

`RuntimeProfiler` activates instrumentation through a context-local collector,
so public calculator and optimizer protocols remain unchanged. CPU work uses
`perf_counter`; CUDA phases use deferred events and synchronize once when the
profile closes. Calculator adapters report their own graph translation phases,
while optimizers report only generic update, compaction, refill, and occupancy
events. Model-native types therefore do not enter the scheduler contract.

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
