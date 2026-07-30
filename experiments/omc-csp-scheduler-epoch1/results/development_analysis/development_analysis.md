# OMC-CSP Scheduler Development Analysis

## Scope

This report audits 23 frozen development workloads and compares the current automatic tensor scheduler with four ASE BFGS CUDA-MPS workers per GPU. External full-process makespan is the primary performance metric.

## Primary Results

| Pool | Workloads | Median speedup | Range | Auto wins | Convergence gate |
|---:|---:|---:|---:|---:|---:|
| 64 | 5 | 2.097x | 1.833-4.308x | 5/5 | 3/5 |
| 512 | 13 | 2.008x | 1.233-2.885x | 13/13 | 7/13 |
| 2048 | 5 | 1.967x | 1.325-2.463x | 5/5 | 3/5 |

## Gates

- Exact identity and coverage: **True**
- Requested numerical contract: **True**
- No OOM: **True**
- Conservative 85% memory limit: **True**
- Convergence-count non-regression: **False**
- Aggregate convergence: automatic **17072/17216**, MPS **17132/17216**
- Automatic nonconverged tail: **144 jobs**
- Scientific/resource gate failures: OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P64-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P64-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P512-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-HAMTIZ-P512-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P512-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-MIX-ALL-P512-INTER-WIDE-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-NACJAF-P512-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-OBEQUJ-P512-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P2048-INTRA-NARROW-v2, OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P2048-INTRA-NARROW-v2

Endpoint differences are reported diagnostically. No endpoint-difference pass threshold is applied because the frozen epoch contract did not declare one.

## Automatic Full-Process Time

| Component | Fraction of aggregate external time |
|---|---:|
| worker_execution | 49.7% |
| structure_model_setup_and_output | 37.7% |
| worker_startup | 7.5% |
| external_wrapper | 3.2% |
| scheduler_other | 1.0% |
| profiling | 0.9% |

## Automatic Worker Work

| Phase | Fraction of summed chunk work |
|---|---:|
| model_autograd | 36.5% |
| neighbor_update | 21.9% |
| model_forward | 18.0% |
| bfgs_update | 12.8% |
| unprofiled | 8.9% |
| active_compaction | 1.9% |

## Weakest External Speedups

| Workload | Speedup over MPS |
|---|---:|
| OPT-OMC-SCHED-E1-DEVELOPMENT-XATMOV-P512-INTRA-NARROW-v2 | 1.233x |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P2048-INTRA-WIDE-v2 | 1.325x |
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P512-INTRA-NARROW-v2 | 1.353x |
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P2048-INTRA-NARROW-v2 | 1.384x |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P512-INTRA-WIDE-v2 | 1.589x |

## Refinement Decision

**Persistent ASE-BFGS recovery for the nonconverged tensor tail**

Rerunning only tensor jobs that reach the 500-step limit with deterministic ASE BFGS from their frozen initial structures, using one persistent model process per assigned GPU, will satisfy convergence-count non-regression while retaining a lower full-process makespan than the complete four-workers-per-GPU MPS baseline.

Mechanism: Complete the normal tensor schedule first. Collect only nonconverged source IDs, dispatch their original structures across persistent per-GPU ASE-calculator workers, run the unchanged BFGS and FrechetCellFilter contract once, and replace those endpoint records unconditionally with the recovery records. Do not change optimizer, cutoff, precision, force threshold, or max steps within either attempt.

Validation: Implement and unit-test source-stable tail extraction, persistent worker reuse, endpoint replacement, timing, retry-count, and peak-memory telemetry. Run the frozen 10-workload validation split once against the unchanged MPS baseline. Accept only if coverage, contract, no-OOM, 85% memory, convergence-count non-regression, and full-process makespan non-regression all pass.

## Workload Table

| Workload | Auto (s) | MPS (s) | Speedup | Conv. auto/MPS | Peak memory |
|---|---:|---:|---:|---:|---:|
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P64-INTRA-NARROW-v2 | 50.22 | 92.04 | 1.833x | 64/64 | 12.5% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P64-INTRA-NARROW-v2 | 56.37 | 118.15 | 2.096x | 64/64 | 4.0% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P64-INTRA-NARROW-v2 | 72.83 | 159.76 | 2.194x | 62/63 | 6.9% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P64-INTRA-NARROW-v2 | 125.82 | 542.03 | 4.308x | 62/64 | 21.7% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P64-INTRA-WIDE-v2 | 45.71 | 95.86 | 2.097x | 64/64 | 17.4% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P512-INTRA-NARROW-v2 | 135.52 | 183.31 | 1.353x | 512/512 | 27.0% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P512-INTRA-NARROW-v2 | 99.14 | 219.32 | 2.212x | 508/511 | 7.1% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-HAMTIZ-P512-INTRA-NARROW-v2 | 127.56 | 256.09 | 2.008x | 510/512 | 20.6% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P512-INTRA-NARROW-v2 | 139.99 | 333.84 | 2.385x | 497/504 | 13.6% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-MIX-ALL-P512-INTER-WIDE-v2 | 146.25 | 345.82 | 2.365x | 507/511 | 27.7% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-NACJAF-P512-INTRA-NARROW-v2 | 108.93 | 230.55 | 2.117x | 505/511 | 7.5% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-OBEQUJ-P512-INTRA-NARROW-v2 | 124.22 | 228.57 | 1.840x | 511/512 | 23.4% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-PAHYON-P512-INTRA-NARROW-v2 | 126.81 | 294.37 | 2.321x | 504/504 | 14.9% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P512-INTRA-NARROW-v2 | 352.76 | 1017.75 | 2.885x | 508/505 | 67.8% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-SOXLEX-P512-INTRA-NARROW-v2 | 46.92 | 87.91 | 1.874x | 512/512 | 6.7% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-UJIRIO-P512-INTRA-NARROW-v2 | 136.89 | 256.33 | 1.873x | 512/512 | 20.9% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P512-INTRA-WIDE-v2 | 130.29 | 207.02 | 1.589x | 512/512 | 25.0% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XATMOV-P512-INTRA-NARROW-v2 | 60.05 | 74.05 | 1.233x | 512/512 | 15.6% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P2048-INTRA-NARROW-v2 | 265.88 | 368.02 | 1.384x | 2048/2048 | 76.2% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P2048-INTRA-NARROW-v2 | 189.05 | 465.59 | 2.463x | 2041/2047 | 19.8% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P2048-INTRA-NARROW-v2 | 310.12 | 716.37 | 2.310x | 1974/2008 | 37.6% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P2048-INTRA-NARROW-v2 | 1052.82 | 2071.28 | 1.967x | 2035/2032 | 69.5% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P2048-INTRA-WIDE-v2 | 332.02 | 439.89 | 1.325x | 2048/2048 | 70.5% |
