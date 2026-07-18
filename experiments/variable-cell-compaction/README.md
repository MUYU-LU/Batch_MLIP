# Variable-cell active compaction

## Hypothesis

Variable-cell FIRE can remove converged systems without changing its trajectory
if graph state and all Frechet/FIRE tensors use one shared selection mapping.

## Scope

- Compact positions, cells, generalized coordinates, log strain, reference
  cells, cell factors, pressures, velocities, and FIRE counters.
- Preserve original-order callback and final energy/force/stress tensors.
- Retain the existing full-periodicity and no-`FixAtoms` limitations.
- Do not claim wall-time acceleration from graph-count reduction alone.

## Validation

A heterogeneous Lennard-Jones batch compares masked and active trajectories at
every tenth step, including per-system pressures and cell factors. The
production validator forces one real T2 structure to converge at step zero and
checks that AtomBit evaluates only the remaining structure afterward.

## Results

- Masked and active callback trajectories agree for positions, cells, energy,
  force, stress, and convergence state.
- Convergence steps are `[0, 54, 81]` with heterogeneous pressures and cell
  factors.
- Graph evaluations decrease from 246 to 138 (`43.9%`) while model evaluation
  count remains identical.
- Production AtomBit active sizes are `[2, 1, 1]`, with batch-versus-single
  errors below `2.4e-6 eV/A` for force and `7.5e-9 eV/A^3` for stress.
- Final suite: 19 passed; Ruff passed; the package wheel built successfully.

No wall-time speedup is claimed yet. The next benchmark must use naturally
converging production structures and report synchronized repeated timings.
