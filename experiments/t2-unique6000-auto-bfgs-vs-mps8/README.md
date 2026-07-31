# T2 P6000: automatic OMC-CSP versus ASE/CUDA-MPS

This experiment uses every CIF in `T2_test.tgz` exactly once. The signed
workload contains 6,000 unique structures and is executed on eight H100 GPUs
with the frozen automatic OMC-CSP workflow and with both eight and sixteen
persistent ASE BFGS CUDA-MPS workers per GPU.

| Method | Production time (s) | Full script (s) | Systems/s | Converged | Peak GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|
| Automatic batching | 442.16 | 470.28 | 13.57 | 5,999/6,000 | 61.66 reserved |
| ASE/CUDA-MPS, 64 workers | 925.81 | 976.85 | 6.48 | 5,999/6,000 | 23.21 sampled |
| ASE/CUDA-MPS, 128 workers | 741.25 | 811.32 | 8.09 | 5,999/6,000 | 39.13 sampled |

The automatic workflow is 2.094x faster by production makespan and 2.077x
faster by full-script time than MPS8. Against MPS16, the corresponding
speedups are 1.676x and 1.725x. Doubling MPS concurrency improves its
production makespan by only 1.249x while increasing sampled peak GPU memory
by 1.686x, demonstrating diminishing returns beyond eight workers per GPU.

The automatic workflow selected the signed offline H100 capacity model, 31
memory-safe execution chunks, work stealing, and four CIF loader processes
per GPU without workload-specific overrides. Tensor execution first converged
5,998 jobs; the frozen ASE tail recovery recovered one additional job in
29.69 seconds. All three methods leave the same source unconverged.

The two MPS configurations have bitwise-identical endpoint records. Batched
and ASE BFGS have identical source coverage and convergence flags but their
trajectories are not bitwise identical. Full endpoint distributions and raw
artifact hashes are retained in `results/summary.json`.
