# OMC-CSP Scheduler v1 Freeze Validation

This validation executes six chemically held-out OMC-CSP workloads through one
persistent seven-H100 `BatchExecutor`. The sequence contains 7,680 unique
variable-cell BFGS relaxations:

```text
ROF-A P2048 -> ROF-C P2048 -> XULDUD P2048
             -> test MIX P512 -> OBEQIX P512 -> JAYDUI P512
```

It tests heavy-to-heavy model reuse, heavy-to-light loader adaptation,
intra-family and inter-family pools, narrow and wide cost distributions, and
P512/P2048 pool sizes without constructing an unregistered mixed P2048 pool.

## Results

| Workload | Jobs | Persistent call (s) | MPS script (s) | Resident call / MPS |
|---|---:|---:|---:|---:|
| ROF-A P2048 | 2,048 | 85.212 | 433.095 | 5.083x |
| ROF-C P2048 | 2,048 | 41.773 | 206.150 | 4.935x |
| XULDUD P2048 | 2,048 | 42.440 | 155.813 | 3.671x |
| Test MIX P512 | 512 | 22.770 | 145.326 | 6.382x |
| OBEQIX P512 | 512 | 41.101 | 278.830 | 6.784x |
| JAYDUI P512 | 512 | 4.993 | 40.891 | 8.189x |

The primary same-accounting comparison is total internal script makespan:
`244.359 s` for the persistent tensor sequence versus `1260.104 s` for the
summed frozen MPS scripts, or `5.157x`. Production-only time is `209.038 s`
versus `1054.223 s`, or `5.043x`. The MPS references use the same manifests,
checkpoint, BFGS/`FrechetCellFilter` contract, force threshold, and step limit.

## Gates

- All 7,680 jobs converged and retained manifest order.
- Every final maximum force is below `0.05 eV/A`; the observed maximum is
  `0.0499987 eV/A`.
- Every sequence endpoint is bitwise identical to an independent current-code
  standalone result with the same chunk plan.
- All six MPS references also converge every job. MPS endpoint differences are
  retained as schedule-sensitive BFGS trajectory diagnostics, not hidden.
- All calls matched the signed offline capacity contract and performed zero
  representative model probes.
- Peak worker reservation is `54.08 GB`, or `63.61%` of H100 memory, below the
  `85%` limit.
- The same seven GPU worker PIDs and worker generation are retained throughout.
- The CPU loader policy changes exactly once, from 28 processes for ROF-A/C to
  seven for XULDUD and all P512 pools, without restarting GPU workers.
- All 42 eligible prefetched chunks are ready on dispatch. Their total dispatch
  wait is `0.0025 s`, versus `37.157 s` for the six initial waves.
- All workers acknowledge shutdown. CUDA interpreter teardown exceeds the
  deliberate 0.1-second post-acknowledgment grace and is boundedly terminated;
  no worker or loader process remains.

## Decision

Freeze OMC-CSP Scheduler v1 for the exact signed AtomBit smooth-rms fp32,
float64 BFGS, `FrechetCellFilter`, 6.0 A cutoff, 0.5 A skin, expandable
allocator, PyTorch/CUDA, H100 contract represented by this epoch.

The frozen policy is signed offline capacity planning, cost bucketing, 14
memory-safe work-stealing chunks for these seven-GPU pools, active compaction,
multi-GPU active drain, automatic one/four-process-per-GPU CIF loading,
bounded global prefetch, and persistent GPU/model workers across consecutive
pools.

This freeze does not claim automatic GPU-count selection, multi-GPU refill,
MACE or other-hardware transfer, molecular conformer scheduling, or MD
scheduling. Those require separate task-instance policies.
