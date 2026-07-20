# Production dense CUDA neighbors

## Scope

The benchmark-only 27-image prototype was replaced by an integrated backend
that supports full, partial, and nonperiodic cells, unwrapped coordinates,
arbitrary safe image ranges, heterogeneous atom counts, selective cache
rebuilds, canonical CPU-compatible edge ordering, and a 512 MiB default
temporary-work budget. Topology decisions use float64 independently of model
precision.

The public setting is `neighbor_backend="auto" | "matscipy" | "cuda_dense"`.
Auto retains a CPU fallback and uses conservative cutoff-aware H100 screening
boundaries. An explicit CUDA request raises on degenerate periodic vectors.

## Correctness

All 24 frozen H46/H276 integrated points match the CPU path exactly, including
edge order and integer shifts. Tests additionally cover randomized triclinic
cells, partial/nonperiodic PBC, rank-deficient nonperiodic dimensions, atoms
outside the primary cell, small cells requiring multiple images, empty graphs,
and selective position/cell rebuilds.

The randomized gate exposed a matscipy 1.2 issue for unwrapped coordinates in
partial/nonperiodic rank-3 cells: it can emit shifts along nonperiodic axes.
The CPU wrapper now limits matscipy to fully periodic rank-3 cells and uses ASE
otherwise. This does not affect the fully periodic frozen performance matrix.

## Integrated rebuild performance

Each entry is CUDA speedup over the integrated CPU rebuild from resident tensor
geometry. Timings include canonical ordering, replacement of cached graph
blocks, synchronization, and returned GPU tensors. Each point has one warmup
and one measured run.

| Cutoff (A) | Distribution | B1 | B2 | B4 | B8 | B16 | B32 |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 4.5 | H46 | 0.41x | 0.39x | 0.60x | 0.93x | 1.40x | 1.98x |
| 4.5 | H276 | 0.56x | 0.89x | 1.49x | 2.43x | 3.69x | 4.36x |
| 6.0 | H46 | 0.46x | 0.68x | 1.17x | 1.94x | 3.03x | 4.23x |
| 6.0 | H276 | 1.06x | 1.79x | 3.14x | 5.08x | 8.01x | 9.69x |

The largest measured CUDA neighbor peak is 0.308 GB at H276 B32, compared with
0.165 GB for the CPU path's returned GPU graphs. These values exclude model
memory.

## Complete evaluation

Order-controlled full model calls use two unmeasured same-shape warmups and one
measurement. Peak memory changes by less than 0.1% because model state and
autograd dominate the temporary dense graph work.

| Model | Distribution | Pool | Resident B | EVAL speedup |
|:--|:--|--:|--:|--:|
| AtomBit | H46 | 32 | 32 | 2.12x |
| AtomBit | H46 | 256 | 128 | 2.59x |
| AtomBit | H276 | 32 | 16 | 2.04x |
| AtomBit | H276 | 256 | 16 | 2.03x |
| MACE-OFF-Small | H46 | 32 | 32 | 1.38x |
| MACE-OFF-Small | H46 | 256 | 128 | 1.64x |
| MACE-OFF-Small | H276 | 32 | 32 | 1.49x |
| MACE-OFF-Small | H276 | 256 | 64 | 1.52x |

MACE float64 backend outputs agree within `3.50e-10 eV` and
`9.77e-15 eV/A`. AtomBit float32 differences are at ordinary nondeterministic
CUDA levels: at most `6.49e-5 eV` total energy and `5.54e-6 eV/A`, below the
existing per-atom and force validation gates.

## Short BFGS screen

Three-step variable-cell AtomBit BFGS improves by 1.39x for H46 B32 and 1.50x
for H276 B16. Maximum position/cell differences are `4.27e-6 A` and
`2.38e-7 A`, respectively. The optimizer algorithm and mixed-precision Hessian
state are unchanged.

MACE three-step BFGS showed a method-order-independent first-allocation penalty:
whichever backend was measured first paid roughly 1.5 s of lazy allocation.
Steady observations suggest only about 1.1-1.2x benefit because MACE and BFGS
dominate, but this is not recorded as a formal speedup. A longer isolated-process
frontier run is required for a paper claim.

## Decision

The production backend is suitable for explicit use and conservative auto
selection. It does not replace matscipy/ASE universally. Short-cutoff H46-like
workloads remain on CPU below B16; long-cutoff workloads use the lower measured
boundary. Cached MD benefits only on rebuild steps, so its end-to-end gain will
be much smaller than EVAL or skin-zero variable-cell optimization.
