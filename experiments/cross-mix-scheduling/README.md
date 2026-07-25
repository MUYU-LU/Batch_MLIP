# CROSS-MIX scheduling and public auto-relaxation

## Hypothesis

On the signed heterogeneous `OPT-RB-CROSS-MIX-R192-v1` variable-cell workload,
the largest calibrated memory-safe resident pool should outperform fixed FIFO
chunks, cost bucketing, planned queues, active refill, and a 32-worker ASE/CUDA
MPS reference without changing BFGS or the convergence criterion.

The pool contains 192 unique jobs from six families and eight atom counts
between 44 and 296. AtomBit smooth-RMS fp32 and MACE-OFF-Small use deterministic
CUDA, one CPU thread, `BFGS + FrechetCellFilter`, `fmax=0.05 eV/A`, and at most
500 steps. Every point is one timing, as requested.

## Results

All B64, B128, and B192 one-step capacity probes pass. The B192 reserved peaks
are 42.93 GiB for AtomBit and 44.98 GiB for MACE. Full optimization gives:

| MLIP | policy | end-to-end (s) | systems/s | allocated (GiB) | reserved (GiB) | speedup vs MPS32 |
|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | FIFO B64 | 93.045 | 2.064 | 16.379 | 60.689 | 1.317x |
| AtomBit | FIFO B128 | 84.903 | 2.261 | 22.105 | 40.244 | 1.443x |
| AtomBit | FIFO B192 | 76.504 | 2.510 | 38.366 | 77.689 | 1.601x |
| AtomBit | refill B128 | 81.559 | 2.354 | 30.145 | 77.934 | 1.502x |
| AtomBit | bucketed B64 | 102.780 | 1.868 | 23.047 | 77.523 | 1.192x |
| AtomBit | planned | 94.690 | 2.028 | 23.988 | 78.195 | 1.294x |
| AtomBit | **auto B192** | **75.357** | **2.548** | **38.366** | **77.689** | **1.626x** |
| MACE | FIFO B64 | 75.211 | 2.553 | 15.273 | 24.057 | 1.062x |
| MACE | FIFO B128 | 69.968 | 2.744 | 24.782 | 30.598 | 1.142x |
| MACE | FIFO B192 | 64.841 | 2.961 | 39.977 | 49.041 | 1.232x |
| MACE | refill B128 | **63.955** | **3.002** | **30.777** | 75.959 | **1.249x** |
| MACE | bucketed B64 | 79.379 | 2.419 | 23.158 | 27.387 | 1.006x |
| MACE | planned | 71.833 | 2.673 | 29.413 | 34.859 | 1.112x |
| MACE | **auto B192** | 64.606 | 2.972 | 39.977 | **49.041** | **1.236x** |

ASE/MPS32 takes 122.503 s for AtomBit and 79.881 s for MACE; all 192 jobs
converge in every run. Automatic B192 is 1.63x/1.24x faster than those
references and 1.23x/1.16x faster than FIFO B64.

MACE refill is 1.02% faster than automatic B192 in this single timing but holds
75.96 GiB reserved instead of 49.04 GiB. It does not meet the established 5%
gate for trading away memory headroom. Refill therefore remains explicit rather
than automatic. AtomBit refill is slower than B192. The strict hypothesis that
the largest resident pool would be the measured fastest point for both MLIPs is
therefore partially falsified, even though the decision-gated recommendation is
unchanged.

## Correctness

The full-relaxation gate is convergence, not endpoint identity on a nonconvex
surface. Convergence flags match ASE/MPS for all 192 jobs. Median selected-policy
versus ASE/MPS energy differences are `6.01e-6 eV/atom` for AtomBit and
`1.32e-8 eV/atom` for MACE; some jobs enter different basins, with maximum
position RMSDs of 0.503 and 1.527 A.

The public API short-horizon gate uses four mixed-family jobs and two BFGS
steps. Whole-pool auto matches direct AtomBit exactly. MACE maximum differences
are `6.14e-14 eV/A` in force and `1.78e-15 A` in position. Forced B2 refill
returns all four systems in original order.

## Decision

`relax(..., scheduling="auto", planner=planner)` is implemented as an opt-in
generic policy. It selects one whole resident pool when calibrated allocation
and count limits pass; otherwise it executes planned queues and uses refill only
for an optimizer that declares refill support. Results include the scheduling
decision, predicted bytes, profiling time, queue sizes, capacities, and refill
flags.

`relax_ase()` is a separate native-ASE reference path requiring an ASE
`Calculator`. It avoids claiming that a tensor calculator adapter reproduces
ordinary ASE calculator execution.

Machine-readable data are in `results.json`, `results.csv`, and
`api-validation/`. Raw full records remain under the ignored
`runs/robustness/cross_mix/` directory.

## Limitations

These are single timings on one 80 GiB GPU class. The calibrated coefficients
are not portable across MLIPs, optimizer precision, or GPU architecture.
Allocator reserved memory can materially exceed live allocation, particularly
for AtomBit and refill, so production budgets must retain physical headroom.
The next experiment should validate automatic scheduling on application-shaped
pools, then calibrate a separate fixed-step MD planner rather than reusing BFGS
memory coefficients.
