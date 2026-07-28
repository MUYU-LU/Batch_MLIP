# Offline Refill Predictor

## Question

Can `scheduling="auto"` choose between active drain and active refill without
relaxing a pilot workload, while excluding timings collected under different
models, optimizers, hardware, or queue construction?

## Fair Matrix

The primary matrix contains 54 newly generated runs: nine chemical families,
256 unique CIFs per family, B32/B64/B128, and paired active/refill BFGS.
Six families were used to lock the speed rule before the three held-out
families were summarized. Every pair used the same signed manifest and the
contract in `experiment.yaml`. There was one warm-up and one timed observation
per point, as requested; consequently the result has no run-to-run uncertainty
estimate.

The earlier R256 experiment repeated 32 CIFs eight times in periodic order.
That ordering exaggerated refill's lane-rebalancing benefit, especially for
ROFA-MIX. It is retained under `results-repeated-control/` as a negative
control and is never used by the runtime predictor.

## Result

The speed boundary locked from the fit families was:

```text
refill speed candidate when resident_capacity * mean_atoms <= 12000
active otherwise
```

On the nine held-out points, this rule predicted the faster mode in 9/9 cases,
selected refill in eight, and selected no speed loss. Refill speedups among
those eight were 1.091x to 1.494x. Two AXOSOW points failed the separately
declared endpoint-equivalence gate despite both modes converging all jobs.
They therefore become active records in the production evidence table.

The shipped policy is intentionally stricter than the fitted speed boundary.
It selects refill only when all of the following match:

- AtomBit smooth-rms fp32 epoch-5 model state and calculator settings.
- BFGS/Frechet, float64 optimizer state, `smax=None`, and active compaction.
- H100 80GB, exact PyTorch/CUDA/allocator/deterministic execution settings.
- A 256-structure homogeneous pool and measured B32/B64/B128 descriptor record.
- At least 1.05x measured speedup, at most 85% reserved memory, full
  convergence, and the endpoint gate.

Everything else falls back to active drain. This is evidence matching, not an
unvalidated extrapolating regressor. It makes the current supported cases
plug-and-play while keeping unsupported MLIPs, pool sizes, and heterogeneous
workloads scientifically conservative. The initial policy is applied to
current-process single-GPU automatic scheduling. Multi-GPU sharding remains
active drain until a per-worker pool/refill matrix is available.

The public-API validation on XATMOV88-R256/B64 selected refill, converged
256/256 jobs, and matched the explicit matrix refill endpoint with a maximum
energy difference of 0.0 meV/atom. Its 32.52 s wall time includes workload
profiling and the representative memory probe and is not used as a new paired
speed claim.

## Artifacts

- `workloads/`: signed 256-unique-CIF manifests and index.
- `results/raw-results.tar.gz`: all primary JSON outputs and logs.
- `results/summary.{json,csv}`: contract validation, metrics, and endpoints.
- `locked-policy.json`: speed rule frozen before held-out analysis.
- `repeated-control-policy.json`: rejected repeated-order rule.
- `batch_mlip/planning/data/refill_policy_v1.json`: runtime evidence table.
- `validate_auto_policy.py`: public-API H100 selection and endpoint check.

Run `run.sh` to regenerate the GPU matrix, `summarize.py` to validate the
contract and endpoints, and `build_policy.py` to rebuild the canonical runtime
artifact.
