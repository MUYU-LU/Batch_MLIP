# MACE tensor-state graph cache

## Hypothesis

MACE-OFF-Small previously rebuilt `ASE Atoms -> Config -> AtomicData ->
DataLoader -> GPU batch` for every optimizer evaluation. The experiment tests
whether projecting the persistent generic batch tensors directly into MACE and
reusing a skin candidate graph improves B64 variable-cell BFGS throughput.

No MACE-specific state enters BFGS. The adapter translates generic positions,
cells, graph indices, integer shifts, species, and heads into the MACE input
dictionary. A `5.5 A` candidate graph is filtered to MACE's physical `5.0 A`
cutoff before every model forward.

## Matrix

Each point is one deterministic timing over the same 256 fixed structures,
using immediate refill, B64, float64 MACE and BFGS state,
`FrechetCellFilter`, `fmax=0.05`, and at most 500 steps.

| atoms | rebuild (s) | cached (s) | time reduction | throughput increase | rebuild peak | cached peak |
|--:|--:|--:|--:|--:|--:|--:|
| 46 | 131.439 | **127.109** | **3.3%** | **3.4%** | 4.621 GB | 4.628 GB |
| 276 | 320.555 | **289.454** | **9.7%** | **10.7%** | 27.443 GB | 27.487 GB |

For 276 atoms, native graph preparation costs `54.934 s`. Cached neighbor
maintenance plus tensor projection costs `21.306 s`. Model-forward time is
effectively unchanged (`70.772 s` versus `71.394 s`). The cache therefore
removes graph construction rather than accelerating MACE itself.

## Correctness

- B1 and B2 energy, force, and stress agree with MACE `AtomicData` within the
  existing declared tolerances.
- B1 variable-cell BFGS agrees with ordinary ASE MACE for the validated steps.
- Cached and rebuild NVE energy drift agree at five time points.
- Every structure converges in all four production points, and convergence
  flags match.
- The 46-atom step counts match exactly; the maximum final-position difference
  is `1.05e-5 A`.
- At 276 atoms, 83 step counts differ. Median final-position and energy
  differences are `3.16e-4 A` and `2.16e-5 eV`, but local-minimum outliers reach
  `1.95 A` and `2.71 eV`.

The large-workload divergence comes from floating-point edge-reduction order:
the generic cache uses canonical order while MACE `AtomicData` retains
matscipy's cutoff-dependent enumeration. Per-evaluation B1/B2 physics agrees,
but long BFGS schedules can amplify roundoff into different local minima.

## Decision

Retain `graph_mode="cached"` with `skin=0.5` as an explicit acceleration mode,
especially for large repeated workloads. Keep `graph_mode="rebuild"` as the
compatibility default rather than silently changing established trajectories.

The first prototype passed all skin candidates directly to MACE. Although
MACE's radial cutoff zeroed their physical contribution, it increased memory
and changed reduction order unnecessarily. That prototype was rejected;
physical-cutoff filtering is part of the retained implementation.

Only one screening timing was requested per point, so no timing uncertainty is
claimed. Raw JSON and logs remain under `runs/mace_tensor_state_cache/` on the
benchmark host.
