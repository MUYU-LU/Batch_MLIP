# Cross-family refill validation

This experiment tests immediate BFGS refill against active drain on six
non-T2 workload families. Every primary case uses 256 signed jobs, resident
batch 64, AtomBit smooth-RMS fp32, float64 optimizer state, variable cells,
the automatic linear-algebra backend, deterministic PyTorch settings, and
the production dual-variable expandable-segment allocator configuration.
Each manifest contains 32 selected structures repeated eight times; the
repetition creates a controlled throughput pool and is not claimed as 256
independent structures.

## Primary results

| Family | Atoms | Active (s) | Refill (s) | Refill speedup | Active occupancy | Refill occupancy | Refill reserved (GiB) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GUFJOG44 | 44 | 90.67 | 72.77 | 1.246x | 0.355 | 0.639 | 5.93 | Exclude: endpoint-sensitive |
| XATMOV88 | 88 | 26.25 | 25.20 | 1.042x | 0.499 | 0.766 | 12.83 | Active: below 5% gate |
| XAFPAY172 | 172 | 93.32 | 82.25 | 1.135x | 0.344 | 0.666 | 18.74 | Refill |
| OBEQIX220 | 220 | 109.31 | 102.69 | 1.065x | 0.433 | 0.717 | 23.51 | Refill |
| ROFB296 | 296 | 288.71 | 267.47 | 1.079x | 0.364 | 0.664 | 30.81 | Refill |
| ROFA-MIX | 74-296 | 66.23 | 55.41 | 1.195x | 0.298 | 0.639 | 16.52 | Refill |

All 3,072 primary optimization outcomes converged. Peak reserved memory was
at most 38.7% of the 79.6 GiB H100, so memory did not reject refill in this
matrix. XAFPAY172, OBEQIX220, ROFB296, and ROFA-MIX pass the declared 5%
speed, 85% memory, convergence, and 5 meV/atom endpoint gates. XATMOV88 is a
real but subthreshold 4.2% gain and remains on active drain.

Refill does not reduce the required graph evaluations: active and refill
perform nearly the same optimization work. It increases average resident
occupancy and combines that work into fewer model calls. For example,
ROFA-MIX reduces model evaluations from 756 to 353 while graph evaluations
remain 14,440 versus 14,443. The resulting gain is therefore controlled by
tail dispersion and the cost of underfilled model calls, not atom count
alone.

## GUFJOG diagnostic

GUFJOG44 has four active/refill endpoint differences above 5 meV/atom, with
a maximum of 17.61 meV/atom. The inputs at repeated positions are bitwise
identical, and a direct initial evaluation produced exactly equal energies,
forces, and stresses. Nevertheless, identical copies also bifurcate within
the active-drain control after compaction. A refill-repack diagnostic took
72.70 s, essentially the same as slot refill at 72.77 s, and also showed
basin bifurcation. This rules out a simple slot-state initialization error,
but it does not establish endpoint invariance. GUFJOG is consequently
excluded from automatic refill evidence rather than counted as a positive
result.

## Policy consequence

The safe plug-and-play default remains deterministic active drain. Refill is
selected only when the pool exceeds resident capacity, predicted memory is
below 85%, and either a validated matching workload record or online
occupancy evidence predicts at least a 5% gain. Endpoint-sensitive evidence
is rejected. This experiment supports refill for four new workload regimes
but does not justify a universal atom-count threshold.

The experiment has one timing observation per method. Active and refill for
each family ran concurrently on separate idle H100 GPUs, and the two waves
used identical launch settings. The lack of timing repeats limits claims
close to the 5% boundary.

Raw output, including the GUFJOG repack diagnostic, is archived in
`results/raw-results.tar.gz`. Machine-readable primary summaries are in
`results/summary.json` and `results/summary.csv`.
