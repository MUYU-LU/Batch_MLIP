# Held-Out Automatic Scheduler vs CUDA MPS

The frozen automatic policy was evaluated without recalibration on distinct
structures from AXOSOW, BOQQUT, XAFPAY, and WICZUF. The comparison used four
H100 GPUs and four independent ASE BFGS workers per GPU for the CUDA MPS
baseline.

| Pool | Auto production | Auto API | MPS production | Production speedup | Auto/MPS converged |
|---|---:|---:|---:|---:|---:|
| R256 | 33.21 s | 42.93 s | 97.48 s | 2.94x | 256/256 |
| R1024 | 91.05 s | 102.77 s | 308.33 s | 3.39x | 1024/1024 |

All signed sources were covered exactly once by both methods. The conservative
automatic memory bound was 10.54 GiB for R256 and 45.70 GiB for R1024, or
13.3% and 57.7% of an H100.

The external process result exposes a separate engineering defect. R256 takes
193.08 s as a complete automatic CLI process versus 131.70 s for MPS because
worker-pool shutdown stalls for approximately 120 seconds after the 42.93 s
API call. R1024 amortizes that delay and remains 1.14x faster as a complete
process. The next engineering action is therefore deterministic prompt worker
shutdown, not another scheduler recalibration.
