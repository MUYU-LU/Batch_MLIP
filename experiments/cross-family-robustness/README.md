# Cross-family robustness

This experiment replaces single-chemistry scaling claims with fixed workloads
selected from the supplied multi-family CSP test set.

Each positive family first contributes 256 deterministic candidates. Thirty-two
unique structures are then selected uniformly across their directed 6 A
edge-per-atom ranking and repeated eight times to form a matched 256-job pool.
`rof-a` is additionally balanced across 74, 148, 222, and 296 atoms before
density stratification.

The positive controls cover small, heavy-element, dense, anisotropic, large,
porous, and heterogeneous workloads. XIFZOF is an applicability control because
it contains Si; it cannot support an AtomBit performance claim unless a
checkpoint explicitly declaring Si support is supplied.

The experiment advances to optimization timing only after finite prediction,
B1 equivalence, and batch-versus-single equivalence checks for both AtomBit and
MACE-OFF-Small. Performance comparisons use identical ordered structures,
variable-cell BFGS, warm-up, synchronized CUDA timing, and recorded peak memory.

## Compatibility result

Both models pass finite prediction and mixed-size batch-versus-B1 equivalence
on the low-, median-, and high-edge representatives from all six positive
families. AtomBit's maximum energy, force, and stress differences are
`2.29e-5 eV`, `3.81e-6 eV/A`, and `1.77e-8 eV/A^3`. MACE-OFF-Small differences
are `1.75e-10 eV`, `1.18e-14 eV/A`, and `5.55e-17 eV/A^3`.

Both checkpoint element tables exclude Si. XIFZOF is therefore excluded from
performance and optimizer conclusions rather than being treated as a batching
failure.

## Reduced timing matrix

The full optimization comparison advances four workloads for both models:

- `GUFJOG44`: small-structure throughput and edge-density variation.
- `XATMOV88`: medium structures with the widest positive-control edge range.
- `OBEQIX220`: large dense full-BFGS Hessians and memory capacity.
- `ROFA-MIX`: heterogeneous 74/148/222/296-atom porous scheduling.

`SOXLEX48` remains a heavy-halogen correctness control because its batching
shape is redundant with `GUFJOG44`. `ROFB296` remains a large-capacity control
and advances only if `OBEQIX220` and `ROFA-MIX` give conflicting memory-policy
conclusions. The cross-family R192 pool is reserved for the final bucketing
confirmation after homogeneous capacity is measured.

## Optimization result

All timings below optimize the same 256 signed jobs with variable-cell BFGS,
`fmax=0.05 eV/A`, at most 500 steps, deterministic algorithms, and one CPU
thread per process or MPS worker. Every reported point converged 256/256 jobs.
These are single-run screens as requested; differences below 2% are
inconclusive.

| MLIP | Workload | Tensor policy | ASE1 s | MPS32 s | Tensor s | vs ASE1 | vs MPS32 | Tensor alloc/reserved GiB | MPS GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AtomBit | GUFJOG44 | active B256 | 1031.07 | 89.75 | 58.81 | 17.53x | 1.53x | 17.79/78.15 | 32.77 |
| AtomBit | XATMOV88 | active B256 | 288.32 | 29.59 | 21.33 | 13.52x | 1.39x | 40.61/74.58 | 34.76 |
| AtomBit | OBEQIX220 | active B128 | 1876.26 | 127.10 | 98.58 | 19.03x | 1.29x | 39.12/77.02 | 52.87 |
| AtomBit | ROFA-MIX | active B256 | 1134.14 | 86.88 | 46.83 | 24.22x | 1.86x | 52.47/77.74 | 36.11 |
| MACE | GUFJOG44 | active B256 | 936.88 | 71.25 | 52.11 | 17.98x | 1.37x | 15.78/19.57 | 31.70 |
| MACE | XATMOV88 | active B256 | 429.65 | 48.69 | 33.58 | 12.79x | 1.45x | 32.33/39.24 | 34.09 |
| MACE | OBEQIX220 | active B128 | 1931.65 | 117.51 | 118.97 | 16.24x | 0.99x | 39.97/49.05 | 42.22 |
| MACE | ROFA-MIX | active B256 | 1271.25 | 118.98 | 64.55 | 19.69x | 1.84x | 66.80/77.78 | 39.08 |

Tensor batching beats common ASE by 12.79-24.22x. It beats MPS32 by
1.29-1.86x on seven of eight model/workload pairs. MACE OBEQIX220 is parity:
the 0.99x ratio is inside the 2% inconclusive band.

## Capacity and refill policy

The batch-size screen rejects a universal "fill all memory" rule:

- B256 is faster than B128 for GUFJOG44 and XATMOV88, where model throughput
  benefits from the larger resident batch.
- OBEQIX220 B128, active B192, and refill B192 differ by less than 2% for both
  MLIPs. Select B128 because it uses less allocated memory. MACE B224 OOMs;
  B192 is the measured capacity limit.
- ROFA-MIX B256 is 12.7% faster than B128 for AtomBit and 5.7% faster for MACE.
  Refill B192 improves over active-drain B192 by 6.9% and 2.3%, but remains
  slower than B256. Refill is therefore a tail-elimination fallback when the
  full pool does not fit, not the default when it does.
- Peak reserved memory can be much larger than live allocated memory because
  the CUDA allocator retains reusable blocks. Planning must record both and
  use a capacity probe for near-frontier workloads.

The resulting rule is workload-aware: use the largest measured efficient batch
for small or heterogeneous pools, prefer the smallest point inside the 2%
performance band for dense full-BFGS systems, and enable refill only when the
pool exceeds safe resident capacity by a modest tail.

## Numerical interpretation

The deterministic three-step gate passes for ASE BFGS, batched eigen-BFGS, and
batched Cholesky-BFGS. Cholesky and eigen trajectories are identical for
AtomBit and agree near float64 roundoff for MACE.

Full relaxation is basin-sensitive. XATMOV88 and ROFA-MIX pass strict endpoint
tolerances, but GUFJOG44 and OBEQIX220 do not, despite every run satisfying
`fmax`. A full B1 tensor diagnostic also changes basins for 9-14 of the first
32 jobs relative to ASE; batching changes another 4-9. MACE MPS32 changes a
smaller subset as well. This shows accumulated sensitivity to tiny numerical
perturbations rather than a wrong batched BFGS update.

The defensible claim is accelerated converged-throughput, not identical
trajectory or minimum identity. Application studies must additionally compare
energy distributions, low-energy hit rates, rankings, duplicate-minimum
counts, and target observables. Exact per-job endpoint matching remains a
diagnostic, not a valid universal requirement for a basin-sensitive search.

Machine-readable timing, memory, endpoint distributions, failed job IDs,
raw-artifact hashes, and B1 diagnostics are stored in
`results/optimization_summary.json`; the compact table is in
`results/optimization_summary.csv`.
