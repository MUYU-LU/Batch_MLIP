# Refill memory control

This experiment resolves the apparent 78-GiB memory cost of AtomBit
STEPVAR-H276 BFGS refill. All cases optimize the same signed 256-job pool once
with stepwise convergence checking and immediate slot refill.

## Root cause

The project supports both PyTorch allocator environment spellings:

```text
PYTORCH_ALLOC_CONF=expandable_segments:True
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

On the installed PyTorch 2.9.1 build, setting only the newer
`PYTORCH_ALLOC_CONF` string is reported in allocator metadata but does not
actually prevent cache growth. The deprecated CUDA-prefixed spelling activates
expandable segments. The production allocator planner already sets both; the
blockwise-refill experiment launcher set only the ineffective spelling.

## Results

| Case | Wall (s) | Speedup vs active | Allocated (GiB) | Reserved (GiB) | Retries |
|:--|--:|--:|--:|--:|--:|
| H46 dual-variable active B64 | 78.67 | 1.000x | 4.33 | 4.88 | 0 |
| **H46 dual-variable refill B64** | **67.92** | **1.158x** | 4.53 | **5.11** | **0** |
| Dual-variable active B64 | 105.70 | 1.000x | 24.23 | 27.19 | 0 |
| **Dual-variable refill B64** | **101.55** | **1.041x** | 26.25 | **29.48** | **0** |
| CUDA-prefixed-only refill B64 | 101.05 | 1.046x | 26.25 | 29.48 | 0 |
| New-variable-only + GC80 B64 | 103.34 | 1.023x | 26.25 | 78.22 | 10 |
| New-variable-only native GC80 B64 | 102.96 | 1.027x | 26.25 | 78.34 | 10 |
| New-variable-only auto B48 | 106.68 | 0.991x | 19.67 | 78.10 | 4 |
| New-variable-only auto B32 | 109.25 | 0.967x | 13.14 | 78.16 | 2 |
| New-variable-only serial B64 | 252.54 | 0.419x | 26.26 | 78.22 | 9 |

All 2,048 case-jobs converged. Dual-variable refill and the CUDA-prefixed-only
control have identical convergence flags, step counts, energies, positions,
and cells. Therefore the allocator compatibility change does not alter the
optimizer trajectory.

Garbage-collection thresholds, smaller resident batches, and serial BFGS do
not repair an allocator that never enabled expandable segments. Serial BFGS
also costs 2.49x the selected refill wall time.

## Decision

Use both allocator variable names in fresh worker processes. This reduces
STEPVAR-H276 immediate-refill peak reservation from 78.22 to 29.48 GiB, or
37.2% of the H100, while retaining a 4.1% throughput advantage over matched
active drain. H46 immediate refill is 15.8% faster than active drain and
reserves only 5.11 GiB.

The plug-and-play allocator planner already supplies both variables, so no
semantic optimizer or capacity change is required. Immediate refill remains a
valid choice for matched, sufficiently variable workloads; blockwise
convergence checking remains rejected.
