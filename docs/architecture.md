# Architecture

## Data path

```text
list[ase.Atoms] --------------------------+
                                          |
signed manifest + planning sidecar        |
       |                                  |
       v                                  |
parent verifies and plans from profiles   |
       |                                  |
       v                                  |
workers load assigned structures ---------+
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

The manifest path is an execution optimization, not a second numerical
implementation. `relax_manifest` verifies that the signed planning profile is
bound to the workload and transfers immutable source descriptors to
process-owned workers. Large file-backed chunks can be parsed by bounded
worker-local `spawn` pools selected from static workload and CPU-capacity
gates; light workloads remain serial to avoid startup and filesystem overhead.
An exact signed hardware-capacity match plans directly
from the sidecar and performs zero model probes; an unmatched contract probes
only a bounded representative set. The workers load their assigned production
chunks and then enter the same batch construction, calculator, optimizer,
compaction, and result-assembly path as eager `relax`. Manifest order is
restored before returning.

For repeated source-backed pools, `BatchExecutor` moves materialization into a
bounded parent-owned producer. It prepares a global cost-ordered ready buffer
while the persistent GPU pool executes the current wave. Ready chunks are
dispatched only after a worker returns a result, retaining dynamic work
stealing. Compatible calls reuse GPU worker PIDs, calculator/model instances,
CUDA contexts, and the CPU loader generation; allocator or loader-capacity
changes restart only the affected generation.

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
Admission initializes new FIRE state while survivor velocity, `dt`, `alpha`,
positive-power count, local step count, and Frechet state remain unchanged.
The `planning` layer does not define one universal scalar system size. It
retains five explicit layers:

1. general structure: atom count;
2. MLIP graph: model identity, cutoff, active edges, force mode, and dtype;
3. task auxiliary: algorithm, stress/cell mode, generalized dimension,
   numerical state, dense state `D^2`, dense linear-algebra work `D^3`, and
   horizon;
4. graph execution policy: skin, candidate edges, cache, and neighbour backend;
5. hardware binding: devices, memory, safety fraction, and calibrated
   coefficients.

Only a hardware- and contract-specific cost model may reduce these features to
time or bytes. `SystemProfile(atom_count, edge_count, dof_squared)` remains a
scheduler-v1 compatibility projection; its attached `bound_cost` layers are
the semantic source for new planning. Ordinary cutoff-based relaxation calls
populate these layers automatically. Explicit `scheduling="single_batch"`
remains the unmanaged compatibility path. Planning does not enter the
calculator or optimizer contracts.

`HardwareCalibratedBatchPlanner` consumes a signed reserved-memory model and
charges the fixed model/allocator term once per resident batch. With an
explicit calibrated planner, the public automatic relaxation path constructs
the layered profiles once and plans queues directly; it performs no timing
pilot. A calibration is valid only for its recorded model, optimizer, cell
filter, graph policy, allocator, hardware, and software contract.

The packaged manifest policy additionally hashes and matches the exact model
state and verifies the planning-profile task layers before enabling offline
capacity. It preserves the outer scheduler's cost buckets and uses the
hardware model only to pack resident chunks inside each bucket. Both the
calibration safety factor and the scheduler's optimization-growth margin are
applied before the 85% memory gate. Any mismatch falls back to the
representative forward rather than extrapolating the model.

Full BFGS retains one independent dense Hessian per system. Equal-dimensional
small CUDA Hessians can be stacked for vectorized updates and a grouped
eigendecomposition; large Hessians and singleton groups retain independent
serial eigensolves. The automatic boundary is a measured execution policy, not
a change to the BFGS equations or optimizer protocol.

`BFGSLineSearch` (also registered as `QuasiNewton`) keeps an independent inverse
Hessian and strong-Wolfe state for each structure. Trial points requested in the
same line-search round are evaluated together. Structures that finish a line
search early remain at their accepted coordinates while the remaining searches
continue. This correctness-first scheduler supports active compaction but not
pending-queue refill.

The authoritative separation between current mechanisms, scheduler-v1
decisions, and future chemical-transfer planning is maintained in
[`research/project/README.md`](../research/project/README.md).

## Policy composition

Automatic relaxation records
`metadata["scheduling"]["policy_manifest"]`. This is the normalized connection
between the abstract planner and the executed task. It contains:

- detected fixed- or variable-cell numerical task;
- separate atom, active-edge, candidate-edge, task-state, and dense
  linear-algebra distributions;
- structural mixing, cost buckets, and resident-wave pressure;
- available and active devices, chunk sizes, and assignment policy;
- cutoff, skin/cache, and neighbour-backend contract;
- compaction, drain/refill, evidence source, and fallback reasons;
- observed convergence-step dispersion for future offline policy fitting.

Immutable structure identity is hashed separately from model/task/policy
sidecars. Changing the cutoff, optimizer, skin, or hardware therefore requires
a new planning profile or execution plan, not a new CIF selection.

The runtime cannot determine that a periodic structure belongs to an OMC CSP
application from atoms alone. It reports the computational task as periodic
variable-cell relaxation; the application study retains the OMC label in its
workload manifest. The same decision is available directly as
`result.execution_policy`.

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
