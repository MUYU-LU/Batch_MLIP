# OMC-CSP Scheduler Test Analysis

## Scope

This report audits 16 frozen test workloads and compares the current automatic tensor scheduler with four ASE BFGS CUDA-MPS workers per GPU. External full-process makespan is the primary performance metric.

## Primary Results

| Pool | Workloads | Median speedup | Range | Auto wins | Convergence gate |
|---:|---:|---:|---:|---:|---:|
| 64 | 5 | 2.159x | 1.564-2.628x | 5/5 | 5/5 |
| 512 | 6 | 1.403x | 1.241-1.626x | 6/6 | 6/6 |
| 2048 | 5 | 1.053x | 0.643-1.223x | 3/5 | 5/5 |

## Gates

- Exact identity and coverage: **True**
- Requested numerical contract: **True**
- No OOM: **True**
- Conservative 85% memory limit: **True**
- Convergence-count non-regression: **True**
- Aggregate convergence: automatic **13632/13632**, MPS **13632/13632**
- Automatic nonconverged tail: **0 jobs**
- ASE tail recovery: **0/0 recovered**, cost **0.00 s**
- Scientific/resource gate failures: none

Endpoint differences are reported diagnostically. No endpoint-difference pass threshold is applied because the frozen epoch contract did not declare one.

## Automatic Full-Process Time

| Component | Fraction of aggregate external time |
|---|---:|
| structure_model_setup_and_output | 59.0% |
| worker_execution | 24.6% |
| worker_startup | 10.1% |
| external_wrapper | 3.7% |
| scheduler_other | 1.4% |
| profiling | 1.1% |
| tail_recovery | 0.0% |

## Automatic Worker Work

| Phase | Fraction of summed chunk work |
|---|---:|
| model_autograd | 36.0% |
| unprofiled | 19.4% |
| model_forward | 18.3% |
| neighbor_update | 17.6% |
| bfgs_update | 7.4% |
| active_compaction | 1.2% |

## Weakest External Speedups

| Workload | Speedup over MPS |
|---|---:|
| OPT-OMC-SCHED-E1-TEST-JAYDUI-P2048-INTRA-NARROW-v2 | 0.643x |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2 | 0.982x |
| OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2 | 1.053x |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2 | 1.159x |
| OPT-OMC-SCHED-E1-TEST-OBEQIX-P2048-INTRA-NARROW-v2 | 1.223x |

## Refinement Decision

Pending expert selection after reviewing the machine-readable analysis.

## Workload Table

| Workload | Auto (s) | MPS (s) | Speedup | Conv. auto/MPS | Peak memory |
|---|---:|---:|---:|---:|---:|
| OPT-OMC-SCHED-E1-TEST-JAYDUI-P64-INTRA-NARROW-v2 | 15.76 | 24.66 | 1.564x | 64/64 | 6.0% |
| OPT-OMC-SCHED-E1-TEST-OBEQIX-P64-INTRA-NARROW-v2 | 55.00 | 144.53 | 2.628x | 64/64 | 16.0% |
| OPT-OMC-SCHED-E1-TEST-ROF-A-P64-INTRA-WIDE-v2 | 33.74 | 77.02 | 2.283x | 64/64 | 12.4% |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P64-INTRA-WIDE-v2 | 28.18 | 51.28 | 1.820x | 64/64 | 12.0% |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P64-INTRA-WIDE-v2 | 22.04 | 47.58 | 2.159x | 64/64 | 11.8% |
| OPT-OMC-SCHED-E1-TEST-JAYDUI-P512-INTRA-NARROW-v2 | 35.70 | 44.31 | 1.241x | 512/512 | 11.7% |
| OPT-OMC-SCHED-E1-TEST-MIX-ALL-P512-INTER-WIDE-v2 | 100.54 | 149.23 | 1.484x | 512/512 | 26.8% |
| OPT-OMC-SCHED-E1-TEST-OBEQIX-P512-INTRA-NARROW-v2 | 174.16 | 283.22 | 1.626x | 512/512 | 36.1% |
| OPT-OMC-SCHED-E1-TEST-ROF-A-P512-INTRA-WIDE-v2 | 126.98 | 178.21 | 1.403x | 512/512 | 32.4% |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P512-INTRA-WIDE-v2 | 83.11 | 103.84 | 1.249x | 512/512 | 15.4% |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P512-INTRA-WIDE-v2 | 57.40 | 88.52 | 1.542x | 512/512 | 19.9% |
| OPT-OMC-SCHED-E1-TEST-JAYDUI-P2048-INTRA-NARROW-v2 | 115.34 | 74.12 | 0.643x | 2048/2048 | 31.7% |
| OPT-OMC-SCHED-E1-TEST-OBEQIX-P2048-INTRA-NARROW-v2 | 536.60 | 656.30 | 1.223x | 2048/2048 | 72.0% |
| OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2 | 418.93 | 441.26 | 1.053x | 2048/2048 | 64.4% |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2 | 215.92 | 212.01 | 0.982x | 2048/2048 | 52.5% |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2 | 138.72 | 160.82 | 1.159x | 2048/2048 | 55.0% |
