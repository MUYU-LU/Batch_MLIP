# Refill Pool-Size and Multi-GPU Transfer

This experiment extends the R256 single-GPU refill evidence without reusing
historical timings. It separates two questions:

1. Does refill transfer to R128 and R512 on one GPU?
2. Does refill transfer when G2/G4 static sharding gives every GPU one
   independent R256 queue?

The complete contract, fit/held-out split, fairness controls, and gates are in
`experiment.yaml`.

## Result

All 32 contract-identical runs completed. The single-GPU fit points selected
refill in all four measured regions:

| Pool | Resident | XATMOV | XAFPAY | Held-out SOXLEX |
|---:|---:|---:|---:|---:|
| 128 | 32 | 1.262x | 1.211x | 1.433x |
| 128 | 64 | 1.126x | 1.078x | 1.121x |
| 512 | 64 | 1.160x | 1.185x | 1.257x |
| 512 | 128 | 1.066x | 1.092x | 1.207x |

The rule was locked before SOXLEX ran. It predicted 4/4 held-out speed modes,
with no convergence, memory, or endpoint failures. These exact R128/R512
records are accepted into offline policy v2 alongside the original R256
evidence.

Multi-GPU transfer was rejected:

| Per-worker queue | GPUs | Fit XATMOV | Held-out BOQWIN | Outcome |
|---:|---:|---:|---:|---|
| R256/B64 | 2 | 1.049x | 1.144x | GPU-count rule predicted only 1/2 |
| R256/B64 | 4 | 1.167x | 1.201x | BOQWIN endpoint difference 6.53 meV/atom |

Static sharding changes convergence-horizon distributions and exposes the
slowest worker, so one single-GPU R256 result cannot predict multi-GPU refill.
Automatic multi-GPU execution therefore remains active drain.

The public API policy-v2 validation on held-out SOXLEX R512/B128 selected
refill, converged 512/512 jobs, and matched the explicit refill endpoint at
0.0 meV/atom. Its 47.29 s wall time includes workload profiling and the memory
probe and is not a paired speed measurement.

Two pre-timing harness failures were excluded: bare GPU numbers were initially
not normalized to `cuda:N`, and active drain initially received a refill-only
option. Both failed before a valid active/refill pair existed; the corrected
matrix was regenerated from the signed manifests.

## Artifacts

- `locked-policy.json`: rule frozen before held-out execution.
- `results/summary.{json,csv}`: validated metrics and scientific gates.
- `results/raw-{fit,heldout}-results.tar.gz`: raw JSON outputs.
- `workloads/`: signed nested unique-CIF manifests.
- `batch_mlip/planning/data/refill_policy_v2.json`: accepted runtime evidence.
