# OMC-CSP Scheduler Validation Analysis

## Scope

This report audits 10 frozen validation workloads and compares the current automatic tensor scheduler with four ASE BFGS CUDA-MPS workers per GPU. External full-process makespan is the primary performance metric.

## Primary Results

| Pool | Workloads | Median speedup | Range | Auto wins | Convergence gate |
|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 1.755x | 1.755-1.835x | 2/2 | 2/2 |
| 512 | 6 | 1.468x | 1.397-2.043x | 6/6 | 6/6 |
| 2048 | 2 | 1.327x | 1.327-1.365x | 2/2 | 2/2 |

## Gates

- Exact identity and coverage: **True**
- Requested numerical contract: **True**
- No OOM: **True**
- Conservative 85% memory limit: **True**
- Convergence-count non-regression: **True**
- Aggregate convergence: automatic **7295/7296**, MPS **7294/7296**
- Automatic nonconverged tail: **1 jobs**
- ASE tail recovery: **2/3 recovered**, cost **46.68 s**
- Scientific/resource gate failures: none

Endpoint differences are reported diagnostically. No endpoint-difference pass threshold is applied because the frozen epoch contract did not declare one.

## Automatic Full-Process Time

| Component | Fraction of aggregate external time |
|---|---:|
| worker_execution | 42.5% |
| structure_model_setup_and_output | 36.2% |
| worker_startup | 10.7% |
| external_wrapper | 4.1% |
| tail_recovery | 3.7% |
| scheduler_other | 1.6% |
| profiling | 1.4% |

## Automatic Worker Work

| Phase | Fraction of summed chunk work |
|---|---:|
| model_autograd | 38.0% |
| neighbor_update | 20.8% |
| model_forward | 19.5% |
| unprofiled | 10.8% |
| bfgs_update | 8.7% |
| active_compaction | 2.2% |

## Weakest External Speedups

| Workload | Speedup over MPS |
|---|---:|
| OPT-OMC-SCHED-E1-VALIDATION-WIDBAO-P2048-INTRA-NARROW-v2 | 1.327x |
| OPT-OMC-SCHED-E1-VALIDATION-BOQQUT-P2048-INTRA-WIDE-v2 | 1.365x |
| OPT-OMC-SCHED-E1-VALIDATION-WICZUF-P512-INTRA-NARROW-v2 | 1.397x |
| OPT-OMC-SCHED-E1-VALIDATION-XAFQIH-P512-INTRA-NARROW-v2 | 1.433x |
| OPT-OMC-SCHED-E1-VALIDATION-MIX-ALL-P512-INTER-WIDE-v2 | 1.468x |

## Refinement Decision

Pending expert selection after reviewing the machine-readable analysis.

## Workload Table

| Workload | Auto (s) | MPS (s) | Speedup | Conv. auto/MPS | Peak memory |
|---|---:|---:|---:|---:|---:|
| OPT-OMC-SCHED-E1-VALIDATION-BOQQUT-P64-INTRA-WIDE-v2 | 49.64 | 91.09 | 1.835x | 64/64 | 16.4% |
| OPT-OMC-SCHED-E1-VALIDATION-WIDBAO-P64-INTRA-NARROW-v2 | 31.44 | 55.18 | 1.755x | 64/64 | 8.2% |
| OPT-OMC-SCHED-E1-VALIDATION-AXOSOW-P512-INTRA-NARROW-v2 | 81.23 | 165.98 | 2.043x | 512/512 | 16.2% |
| OPT-OMC-SCHED-E1-VALIDATION-BOQQUT-P512-INTRA-WIDE-v2 | 119.35 | 206.39 | 1.729x | 512/512 | 28.8% |
| OPT-OMC-SCHED-E1-VALIDATION-MIX-ALL-P512-INTER-WIDE-v2 | 115.11 | 168.94 | 1.468x | 512/512 | 22.7% |
| OPT-OMC-SCHED-E1-VALIDATION-WICZUF-P512-INTRA-NARROW-v2 | 173.60 | 242.46 | 1.397x | 511/510 | 22.0% |
| OPT-OMC-SCHED-E1-VALIDATION-WIDBAO-P512-INTRA-NARROW-v2 | 77.38 | 118.53 | 1.532x | 512/512 | 16.9% |
| OPT-OMC-SCHED-E1-VALIDATION-XAFQIH-P512-INTRA-NARROW-v2 | 100.27 | 143.69 | 1.433x | 512/512 | 23.1% |
| OPT-OMC-SCHED-E1-VALIDATION-BOQQUT-P2048-INTRA-WIDE-v2 | 324.39 | 442.80 | 1.365x | 2048/2048 | 74.0% |
| OPT-OMC-SCHED-E1-VALIDATION-WIDBAO-P2048-INTRA-NARROW-v2 | 196.44 | 260.73 | 1.327x | 2048/2048 | 46.2% |
