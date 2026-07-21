# BFGS grouped mixed-workload confirmation

This experiment confirms the measured `D <= 285` automatic BFGS policy on one
interleaved MIX-R256 workload containing 64 structures from each H46, H92,
H184, and H276 category. A single B128 active-refill queue preserves
heterogeneous resident batches: `auto` groups H46/H92 linear algebra and keeps
H184/H276 serial.

## Production result

Each point is one deterministic timing. All four runs converged 256/256 jobs;
there is no uncertainty estimate.

| Model | Backend | Wall (s) | systems/s | Optimizer (s) | Model (s) | Neighbors (s) | Peak allocated (GB) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| AtomBit | serial | 193.162 | 1.325 | 115.462 | 53.547 | 10.898 | 29.478 |
| AtomBit | auto | 186.371 | 1.374 | 108.195 | 53.330 | 10.926 | 29.474 |
| MACE | serial | 212.469 | 1.205 | 133.832 | 51.645 | 12.970 | 29.556 |
| MACE | auto | 199.036 | 1.286 | 122.272 | 50.998 | 12.474 | 29.549 |

`auto` improves AtomBit throughput by 3.64% and MACE by 6.75%. MACE passes the
declared 5% mixed-workload gate. AtomBit is positive but below the gate and is
therefore inconclusive rather than a claimed speedup. Optimizer time falls by
6.29% and 8.64%, respectively. Memory is effectively unchanged.

## Numerical result

Convergence flags match for every job. Maximum production differences by size
are:

| Model | Atoms | Step mismatches | Energy (eV/atom) | Position RMSD (A) | Cell RMSD (A) |
|:--|--:|--:|--:|--:|--:|
| AtomBit | 46 | 3 | 1.296e-4 | 0.0303 | 0.0336 |
| AtomBit | 92 | 6 | 9.686e-4 | 0.1617 | 0.0899 |
| AtomBit | 184 | 2 | 4.588e-5 | 0.0085 | 0.0008 |
| AtomBit | 276 | 0 | 7.049e-7 | 0.0007 | 5.775e-5 |
| MACE | 46 | 0 | 7.165e-10 | 4.757e-7 | 2.727e-7 |
| MACE | 92 | 22 | 7.343e-4 | 0.0793 | 0.0351 |
| MACE | 184 | 7 | 1.060e-3 | 0.1430 | 0.0541 |
| MACE | 276 | 8 | 6.161e-3 | 0.5777 | 0.1569 |

A separate three-step control agrees exactly for AtomBit except a
`5.55e-17 A` H92 cell component. MACE maximum differences are
`1.13e-13 eV/A` in fmax, `1.41e-16 eV/A^3` in stress, and `3.55e-15 A` in
positions.
Every fixed-step flag and step count matches. Long-run differences are therefore
full-BFGS trajectory amplification, not a dispatch or equation error.

## Decision

Retain the simple dimension-only automatic policy. It is safe, unchanged in
memory, and faster for both measured models. Do not add composition-specific
logic from a single workload: it would overfit a 3.64% AtomBit result below the
claim threshold. The defensible statement is that mixed-workload benefit is
model- and composition-dependent.

The reusable memory-planner benchmark now supports four-way mixed workloads,
fixed B128 execution, explicit linear-algebra backends, short-step validation,
and cached MACE graphs. Raw artifacts remain on the server under
`runs/bfgs_grouped_mixed_confirmation/`.
