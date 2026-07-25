# Task-aware policy validation

## Question

Can a scheduler fitted from H46/H276 measurements select a safe, fast mode
for unseen H92, H184, and four-size mixed pools at R32/R64/R256?

The fitted inputs are:

- one deterministic R32 step-distribution pilot for FIRE and BFGS;
- one warmed stress evaluation at B1/B8/B16/B32/B64/B128;
- calibrated BFGS allocated-memory coefficients;
- prior MPS32 throughput for H46/H276.

The held-out signed workloads contain H92, H184, or round-robin
H46/H92/H184/H276 structures. No held-out timing is used to choose a mode.
All reported measurements use one timing run.

## Implementation

`OptimizationPilot` stores optimizer-specific step samples, graph regimes,
batch timing curves, and optional MPS throughput. `TaskAwarePolicy` simulates
active drain and immediate refill at each measured, memory-safe capacity,
applies a 5% refill gate, and compares the selected tensor schedule with MPS.
The public API is:

```python
result = relax(
    systems,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    planner=planner,
    pilot=pilot,
    system_profiles=cached_profiles,
    cell_filter=FrechetCellFilter(),
)
```

The validation exposed and fixed four production issues:

1. Refill cannot be extrapolated safely from atom count alone. It now requires
   a matched atom/edge regime unless explicitly overridden.
2. Signed per-job edge counts can be passed as cached `SystemProfile` objects,
   avoiding repeated CPU neighbor profiling.
3. Completed planner buckets are offloaded before the next bucket so result
   retention does not invalidate the resident-memory bound.
4. MPS warmup failures now terminate workers, and invocation-unique worker
   files prevent stale processes from corrupting a later parent timing.

The initial unsafe AtomBit H92 B128-refill result is retained outside the final
matrix. It reserved 78.3 GiB against a 68 GiB policy gate. The corrected
matched-evidence rule selected two B128 drain chunks: 80.79 s, 18.6 GiB peak
allocated, and 46.8 GiB peak reserved.

## BFGS result

The oracle is deliberately restricted to the policy tensor schedule, MPS32,
and one smaller refill candidate (plus a direct B128 drain check for AtomBit
H92 R256). All 12 BFGS rows converged completely.

| Model | Workload | R | Policy | Time (s) | Fastest measured | Regret | Selected peak (GiB) |
|:--|:--|--:|:--|--:|:--|--:|--:|
| AtomBit | H92 | 32 | MPS32 | 18.20 | tensor, 17.62 s | 1.033x | 34.2 |
| AtomBit | H92 | 64 | tensor | 26.22 | tensor | 1.000x | 17.1 |
| AtomBit | H92 | 256 | tensor | 80.79 | tensor | 1.000x | 46.8 |
| MACE | H92 | 32 | MPS32 | 18.99 | MPS32 | 1.000x | 33.9 |
| MACE | H92 | 64 | tensor | 30.08 | MPS32, 24.78 s | 1.214x | 10.0 |
| MACE | H92 | 256 | tensor | 97.60 | MPS32, 85.70 s | 1.139x | 20.0 |
| AtomBit | MIX4 | 32 | MPS32 | 33.85 | MPS32 | 1.000x | 36.7 |
| AtomBit | MIX4 | 64 | MPS32 | 38.46 | refill B32, 35.29 s | 1.090x | 37.0 |
| AtomBit | MIX4 | 256 | tensor | 94.11 | tensor | 1.000x | 76.9 |
| MACE | MIX4 | 32 | MPS32 | 37.34 | tensor, 34.94 s | 1.068x | 37.0 |
| MACE | MIX4 | 64 | MPS32 | 46.57 | refill B32, 41.39 s | 1.125x | 39.5 |
| MACE | MIX4 | 256 | tensor | 120.48 | MPS32, 111.24 s | 1.083x | 53.2 |

For MPS, peak is device memory sampled by `nvidia-smi`; for tensor execution,
it is PyTorch peak reserved memory. The policy is within 5% of the restricted
oracle in 6/12 rows. Geometric-mean regret is 1.061x and maximum regret is
1.214x.

Two failure modes remain:

- MACE's tensor/MPS crossover at H92 and MIX4 is not predicted reliably by
  interpolating H46/H276 throughput.
- MIX4 R64 benefits from B32 refill, but enabling it without a matched target
  pilot would repeat the unsafe refill extrapolation observed for H92.

The AtomBit MIX4 R256 tensor run allocates 41.1 GiB but reserves 76.9 GiB
inside its largest B128 bucket. Result offloading prevents cross-bucket
retention, but allocator reservation still needs a separate target-specific
gate; allocated-memory calibration alone is insufficient near capacity.

## FIRE result

FIRE is not a quality-valid H184 optimizer under the frozen 2,000-step limit.
The policy/MPS paths converge 31/32, 62/64, and 248/256 jobs for both MLIPs.
Refill changes the asynchronous trajectory and reaches 63/64 and 252/256, but
still does not complete the pool. Therefore no FIRE speed or regret claim is
reported for H184.

This is an optimizer-selection result, not a scheduler win: a target pilot
must reject FIRE at this step ceiling or increase the ceiling and revalidate
quality before comparing throughput.

## Conclusion

The individual mechanisms work, but an H46/H276-only policy is not yet a
general best-mode solver. It provides safe conservative drain decisions after
the refill-evidence fix, but its MPS crossover and refill benefit estimates
need a small target-regime pilot.

The next engineering stage should cache a target pilot keyed by MLIP,
optimizer, cell mode, atom/edge regime, and hardware. That pilot must measure
convergence completeness, B1-to-capacity stress scaling, allocated and
reserved memory, and a short MPS sample. Only then should the scheduler enable
refill or claim a low-regret automatic choice.

Machine-readable results are in `results/results.json` and
`results/results.csv`. Raw and superseded diagnostic outputs remain under
`runs/task_aware_policy/`.
