# T2 P6000: automatic OMC-CSP versus ASE/CUDA-MPS

This experiment uses every CIF in `T2_test.tgz` exactly once. The signed
workload contains 6,000 unique structures and is executed on eight H100 GPUs
with the frozen automatic OMC-CSP workflow and with both eight and sixteen
persistent ASE BFGS CUDA-MPS workers per GPU.

| Method | Execution/makespan (s) | Full script (s) | Systems/s | Converged | Peak GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|
| Automatic batching, neighbour-control v2 | 364.90 | 368.82 | 16.44 | 5,999/6,000 | 61.66 reserved |
| Automatic batching | 442.16 | 470.28 | 13.57 | 5,999/6,000 | 61.66 reserved |
| ASE/CUDA-MPS, 64 workers | 925.81 | 976.85 | 6.48 | 5,999/6,000 | 23.21 sampled |
| ASE/CUDA-MPS, 128 workers | 741.25 | 811.32 | 8.09 | 5,999/6,000 | 39.13 sampled |

The automatic workflow is 2.094x faster by production makespan and 2.077x
faster by full-script time than MPS8. Against MPS16, the corresponding
speedups are 1.676x and 1.725x. Doubling MPS concurrency improves its
production makespan by only 1.249x while increasing sampled peak GPU memory
by 1.686x, demonstrating diminishing returns beyond eight workers per GPU.

The automatic workflow selected the signed offline H100 capacity model, 31
memory-safe execution chunks, work stealing, and four CIF loader processes
per GPU without workload-specific overrides. Tensor execution first converged
5,998 jobs; the frozen ASE tail recovery recovered one additional job in
29.69 seconds. All three methods leave the same source unconverged.

The two MPS configurations have bitwise-identical endpoint records. Batched
and ASE BFGS have identical source coverage and convergence flags but their
trajectories are not bitwise identical. Full endpoint distributions and raw
artifact hashes are retained in `results/summary.json`.

## Automatic hotspot profile

A separate, identically configured automatic run recorded compact phase totals
inside all 31 persistent-worker chunks. It took 446.51 s versus 442.16 s for
the unprofiled execution, a 0.98% overhead, and produced bitwise-identical
endpoint records and optimization steps.

| Non-overlapping worker phase | Summed work (s) | Share |
|---|---:|---:|
| Model autograd | 1,144.34 | 42.44% |
| Model forward | 528.74 | 19.61% |
| Neighbour update | 591.45 | 21.93% |
| BFGS update | 207.36 | 7.69% |
| Active compaction | 93.27 | 3.46% |
| Graph-view construction | 9.18 | 0.34% |
| Unprofiled remainder | 122.24 | 4.53% |

The model accounts for 62.04% of summed worker work. The largest
framework-controlled phase is neighbour update at 21.93%, but actual neighbour
search is only 56.48 s, or 9.55% of that phase. The remaining neighbour-update
work is therefore the next inner-scheduler target, especially cache-validity,
graph-repacking, integrity checks, and their host/device synchronization.

Worker time has a 1.118 max/mean ratio. Perfectly balancing the same measured
chunk work would remove at most 39.77 s, or 10.35% of the measured production
makespan. This is the next outer-scheduler opportunity, but it is an upper
bound rather than an expected speedup.

## Neighbour-control v2

The optimized implementation vectorizes fully periodic variable-cell cache
validity and removes the redundant integrity scan on evaluations where graph
topology did not change. Full integrity validation remains at graph creation,
selection, replacement, concatenation, and neighbour-rebuild boundaries.

With the same signed workload, 31 planned chunks, capacity model, and 61.66 GiB
peak reserved memory, execution fell from 446.51 s for the matched profiled
baseline to 364.90 s. Production time fell from 384.21 s to 300.18 s. This is a
1.224x end-to-end execution speedup and a 1.280x production speedup.

Summed neighbour-update work fell from 591.45 s to 103.09 s, an 82.57%
reduction. The new neighbour-update decomposition is 56.07 s search, 24.31 s
cache validity, 11.78 s rebuild preparation, 7.20 s graph assembly, 2.50 s
mutation-boundary integrity validation, and 0.95 s invalid-system selection.

All source IDs, convergence states, step counts, energies, forces, stresses,
positions, and cells are bitwise identical to the matched baseline. Against
MPS16, the optimized automatic execution is 2.031x faster; full-script speedup
is 2.200x. Worker max/mean time also falls from 1.118 to 1.060, leaving only a
16.45 s ideal-balance upper bound for the next outer-scheduler refinement.
