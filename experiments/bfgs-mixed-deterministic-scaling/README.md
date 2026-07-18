# Deterministic mixed-precision BFGS scaling

## Setup

- AtomBit model inference, forces, and stress: float32.
- BFGS coordinates, Frechet log deformation, Hessians, and eigensolves: float64.
- Determinism: `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Data: the same 32 unique manifest structures at 46, 92, 184, and 276 atoms.
- Full anisotropic Frechet cell optimization, `fmax=0.05 eV/A`, 500-step cap,
  `alpha=70 eV/A^2`, `maxstep=0.2 A`, and 6.0 A cutoff.
- Three synchronized repeats for every ASE and active point; one CPU thread
  and one H100 worker per process.

## Scaling results

End-to-end seconds for each 32-structure pool. All values are medians of three
deterministic repetitions.

| atoms | common ASE | B1 | B2 | B4 | B8 | B16 | B32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 128.37 | 127.15 | 89.91 | 61.20 | 41.29 | 27.17 | 20.42 |
| 92 | 150.51 | 140.37 | 94.88 | 68.59 | 45.86 | 34.94 | 32.19 |
| 184 | 221.80 | 138.52 | 108.34 | 79.93 | 62.96 | 51.90 | 45.34 |
| 276 | 303.63 | 111.39 | 84.37 | 67.12 | 51.89 | 44.23 | 44.09 |

| atoms | B1 vs ASE | B32 vs ASE | B32 vs B1 | B32 peak memory |
|---:|---:|---:|---:|---:|
| 46 | 1.01x | 6.29x | 6.23x | 2.20 GiB |
| 92 | 1.07x | 4.68x | 4.36x | 4.70 GiB |
| 184 | 1.60x | 4.89x | 3.06x | 9.08 GiB |
| 276 | 2.73x | 6.89x | 2.53x | 11.52 GiB |

B32 timing ranges were 20.40-20.47, 32.17-32.37, 45.31-45.41, and
44.00-44.17 seconds in increasing atom-count order. ASE ranges were
128.23-130.02, 148.84-150.79, 221.35-223.35, and 302.77-309.18 seconds.
Active compaction avoided 4,632, 6,592, 6,778, and 3,298 graph evaluations at
B32. Active B32 took slightly more total optimizer steps than ASE in every
group, so the speedups do not come from performing fewer steps.

The large-system B1 advantage comes from executing the dense eigensolve on the
GPU rather than ASE/SciPy's CPU path. B32 scaling eventually flattens because
each system still requires an independent dense `O(D^3)` eigensolve.

## Correctness

A deterministic three-forced-step control over all 128 structures passes every
declared gate:

| maximum error across all groups | value |
|:---|---:|
| energy | 1.80e-7 eV/atom |
| maximum force | 2.93e-4 eV/A |
| stress tensor | 2.68e-7 eV/A^3 |
| position RMSD | 9.19e-7 A |
| cell RMSD | 1.22e-6 A |
| step difference | 0 |

Every converged-run flag also matches common ASE. Long final structures can
still occupy different minima because native and ASE calculator/filter paths
are not bitwise identical and full BFGS amplifies micro-level differences.
These basin differences are retained in `results.json` but are not interpreted
as direct force or BFGS-equation errors.

Verification:

```text
python -m pytest -q
40 passed in 30.01s

python -m ruff check atombit_batch tests benchmarks
All checks passed!
```

## Artifacts

- `results.json`: merged B1-B32 performance and long-run comparisons.
- `fixed3_results.json`: deterministic three-step correctness validation.
- `raw/scaling/`: eight complete production scaling artifacts.
- `raw/fixed3/`: eight fixed-step control artifacts.

This experiment replaces the earlier float32/nondeterministic BFGS scaling
comparison for any ASE performance claim.
