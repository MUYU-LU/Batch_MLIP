# Large-batch relaxation on a 1,024-system pool

## Method

The same 16 fixed T2 structures in each atom-count group are repeated 64 times
in a fixed cyclic order. ASE, masked FIRE, and active-compaction FIRE all process
the complete 1,024-system pool. Batch timings use three CUDA-synchronized repeats.

## Results

ASE has no batch-size parameter, so each atom group has one sequential baseline.
Times below are median seconds for the full 1,024-system pool. Parentheses show
speedup versus the common ASE baseline.

| atoms | B | masked | active | active vs masked |
|---:|---:|---:|---:|---:|
| 46 | 64 | 457.7 (3.36x) | 207.4 (7.43x) | 2.21x |
| 46 | 128 | 443.3 (3.47x) | 182.5 (8.44x) | 2.43x |
| 46 | 256 | 441.1 (3.49x) | 171.2 (9.00x) | 2.58x |
| 46 | 512 | 435.0 (3.54x) | 167.9 (9.17x) | 2.59x |
| 46 | 1024 | 436.5 (3.53x) | 167.3 (9.21x) | 2.61x |
| 92 | 64 | 1576.5 (1.47x) | 479.9 (4.81x) | 3.28x |
| 92 | 128 | 1505.5 (1.53x) | 447.6 (5.16x) | 3.36x |
| 92 | 256 | 1484.3 (1.56x) | 433.1 (5.34x) | 3.43x |
| 184 | 64 | 878.4 (1.64x) | 424.8 (3.39x) | 2.07x |
| 184 | 128 | 853.3 (1.69x) | 409.3 (3.52x) | 2.08x |
| 276 | 64 | 942.8 (1.38x) | 534.6 (2.44x) | 1.76x |
| 276 | 128 | 838.5 (1.56x) | 523.4 (2.49x) | 1.60x |

Common ASE baselines were 1540.1, 2310.6, 1441.3, and 1305.0 seconds
for 46, 92, 184, and 276 atoms, respectively.

## Capacity

| atoms | largest feasible B | peak allocation at largest B | first OOM |
|---:|---:|---:|---:|
| 46 | 1024 | 67.8 GiB | not reached |
| 92 | 256 | 37.3 GiB | 512 |
| 184 | 128 | 35.3 GiB | 256 |
| 276 | 128 | 44.6 GiB | 256 |

Masked and active modes have the same capacity because the first model call
contains every graph. Compaction reduces later computation, not initial memory.

## Validation

Every feasible point converged. Both 184-atom modes and all active 46-atom
points passed every final-state gate. The other feasible points selected the
known alternate CUDA/FIRE near-threshold branch relative to their independent
ASE reference and are retained as `validation_failed`; this also occurs in
masked mode and does not track compaction.

The 92/B128 and B256 masked medians use three independent one-repeat workers on
identical H100s. This is statistically equivalent to three sequential repeats
and reduced experiment wall time. The longer interrupted serial attempts are
preserved as partial artifacts.

## Artifacts

- `runs/large_batch_relaxation_1024_summary.json`: merged result.
- `runs/large_pool1024_atoms{N}_{masked,active}.json`: primary raw results.
- `runs/large_pool1024_atoms92_masked_b{128,256}_rep*.json`: parallel replicas.
- `runs/large_pool1024_atoms92_masked_b{512,1024}_aux.json`: OOM probes.
