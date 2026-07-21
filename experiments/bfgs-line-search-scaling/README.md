# BFGS line-search scaling

## Question

For full variable-cell relaxation, is batched ASE `QuasiNewton`/
`BFGSLineSearch` faster than sequential ASE, and does it outperform the current
batched standard BFGS?

## Matrix

The frozen R32 H46 and H276 pools are run with AtomBit and MACE-OFF-Small.
Every model/workload pair compares sequential ASE BFGSLineSearch, active B32
BFGSLineSearch, and active B32 standard BFGS. The convergence target is
`fmax=0.05 eV/A` on ASE Frechet generalized forces with at most 500 accepted
steps. Standard BFGS uses `alpha=70`; BFGSLineSearch uses `alpha=10`.

One synchronized timing observation is recorded per point, as requested for
the project screening stages. Results below 2% are inconclusive. Every timing
must also report model evaluations, summed accepted steps, convergence count,
and peak allocated GPU memory.

## Results

All times cover full R32 variable-cell optimization. Memory is peak CUDA
allocation. `Batch QN` is active B32 BFGSLineSearch; `Batch BFGS` is active B32
standard BFGS.

| Model | Atoms | Method | Time (s) | Speedup vs ASE QN | Model evals | Steps | Peak (GB) |
|---|---:|---|---:|---:|---:|---:|---:|
| MACE float64 | 46 | ASE QN | 373.11 | 1.00x | 11,415 | 4,334 | 0.15 |
| MACE float64 | 46 | Batch QN | 34.29 | 10.88x | 440 | 4,335 | 2.34 |
| MACE float64 | 46 | Batch BFGS | 22.51 | 16.58x | 263 | 4,644 | 2.33 |
| MACE float64 | 276 | ASE QN | 965.23 | 1.00x | 13,739 | 4,597 | 0.52 |
| MACE float64 | 276 | Batch QN | 64.98 | 14.85x | 741 | 4,787 | 13.66 |
| MACE float64 | 276 | Batch BFGS | 43.66 | 22.11x | 255 | 2,855 | 13.64 |
| AtomBit float32 | 46 | ASE QN | >1,800 | timeout | - | - | - |
| AtomBit float32 | 46 | Batch QN | >1,400 | failed | - | - | - |
| AtomBit float32 | 46 | Batch BFGS | 22.59 | - | 267 | 3,893 | 2.37 |
| AtomBit float32 | 276 | ASE QN | >1,800 | timeout | - | - | - |
| AtomBit float32 | 276 | Batch QN | >1,800 | timeout | - | - | - |
| AtomBit float32 | 276 | Batch BFGS | 39.58 | - | 181 | 2,405 | 12.38 |

All completed points converged 32/32 structures. MACE Batch QN is 10.88x to
14.85x faster than sequential ASE QN, demonstrating that independent line
search trials can be batched. It is nevertheless 1.52x and 1.49x slower than
Batch BFGS because its lower H46 accepted-step count does not offset 440 versus
263 model calls, and at H276 it increases both model calls and accepted steps.

For MACE H46, batch and ASE QN endpoints agree within 0.087 meV/atom for all
32 structures. At H276, all jobs converge and the median energy difference is
0.0037 meV/atom, but three trajectories enter measurably different basins;
29/32 are within 0.1 meV/atom and the worst difference is 2.56 meV/atom. This
is convergence equivalence, not trajectory or endpoint identity.

AtomBit float32 is a negative result. Batch QN exceeded 100 trial evaluations
for one H46 system, while the other QN runs exceeded the 30-minute screening
limit. Strong-Wolfe energy comparisons are not robust enough at this model's
precision for this workload. BFGSLineSearch therefore remains an explicit
option for suitable calculators, while standard BFGS remains the default.

These are single observations run concurrently on separate H100 GPUs, so they
are screening results without uncertainty and with possible shared-host CPU
contention. Raw artifact hashes and complete counters are in `results.json`.
