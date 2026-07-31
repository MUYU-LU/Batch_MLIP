# T2 P6000: automatic OMC-CSP versus ASE/CUDA-MPS

This experiment uses every CIF in `T2_test.tgz` exactly once. The signed
workload contains 6,000 unique structures and is executed on eight H100 GPUs
with the frozen automatic OMC-CSP workflow and with eight persistent ASE BFGS
CUDA-MPS workers per GPU.

| Method | Production time (s) | Full script (s) | Systems/s | Converged | Peak GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|
| Automatic batching | 442.16 | 470.28 | 13.57 | 5,999/6,000 | 61.66 reserved |
| ASE/CUDA-MPS, 64 workers | 925.81 | 976.85 | 6.48 | 5,999/6,000 | 23.21 sampled |

The automatic workflow is 2.094x faster by production makespan and 2.077x
faster by full-script time. It selected the signed offline H100 capacity
model, 31 memory-safe execution chunks, work stealing, and four CIF loader
processes per GPU without workload-specific overrides. Tensor execution first
converged 5,998 jobs; the frozen ASE tail recovery recovered one additional
job in 29.69 seconds. Both methods leave the same source unconverged.

The methods have identical source coverage and convergence flags, but their
BFGS trajectories are not bitwise identical. The full endpoint distributions
and hashes of the raw remote artifacts are retained in `results/summary.json`.
