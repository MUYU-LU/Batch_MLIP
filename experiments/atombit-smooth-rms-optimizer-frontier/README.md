# Smooth-RMS AtomBit optimizer frontier

Status: complete.

This experiment measures the maximum safe single-GPU throughput of native-fp32
smooth-RMS AtomBit variable-cell relaxation for FIRE, full BFGS, and
BFGSLineSearch across H46, H92, H184, and H276 structures.

The R256 workload repeats the first 64 frozen manifest structures exactly four
times. Common ASE is measured on the matching R64 source pool for every
optimizer and size. Timings are one CUDA-synchronized observation on one NVIDIA
H100 80 GB GPU. A point is valid only when all jobs converge and peak PyTorch
reserved memory is at most 85% of physical GPU memory.

## Recommended policy

| Size | Fastest optimizer | Mode | B | R256 s | systems/s | vs matching ASE | Reserved GB |
|---:|:--|:--|---:|---:|---:|---:|---:|
| H46 | BFGS | refill | 256 | 82.501 | 3.103 | 12.120x | 46.766 |
| H92 | BFGS | active | 128 | 174.947 | 1.463 | 7.602x | 50.281 |
| H184 | BFGS | active | 64 | 316.634 | 0.809 | 5.793x | 61.736 |
| H276 | FIRE | active | 64 | 100.911 | 2.537 | 6.310x | 62.923 |

H46 is a close screening result: FIRE B256 takes 86.969 s, only 5.4% slower
than BFGS. Confirmation repeats are required before presenting that difference
as a paper result. The other optimizer choices have substantially larger
margins.

The complete optimizer-specific frontier is in `results.md`; `results.csv` and
`results.json` contain the selected points, rejected candidates, performance
counters, endpoint diagnostics, and raw-file hashes.

The complete external raw archive is
`/tmp/smooth_rms_optimizer_frontier_raw_final.tgz` on the benchmark server
(`SHA-256 42b7806acc21dca0592a3c82979cb0ba988d42b945d434ae20670f1d447693a8`).

## Scheduling and memory

Peak allocation alone is not a sufficient production memory metric. Several
successful points allocate 20-52 GB but reserve 82-84 GB and are rejected by
the safety rule:

- BFGS refill H92 B128/B256 and H276 B64/B128 are memory-unsafe. Active-drain
  falls back to H92 B128 and H276 B64 with 50.3 and 64.3 GB reserved.
- FIRE H276 B128 is faster than B64 but reserves 82.6 GB, so B64 is selected.
- BFGSLineSearch H92 B128/B256, H184 B128, and H276 B128 are unsafe. The valid
  fallbacks are B64, B32, and B64.
- H184 BFGSLineSearch has an inverse frontier: B32 takes 1037.8 s, faster than
  B64 at 1074.4 s and B16 at 1105.2 s.

Refill is therefore not a universal choice. It remains the best safe mode for
H46 BFGS, but active-drain is required for larger structures in this workload.

## Convergence and endpoints

All selected points converge 256/256 jobs. FIRE needs a 2000-step cap for H92.
FIRE H184 still converges only 63/64 ASE structures at 2000 steps, so no FIRE
frontier is reported for that class.

FIRE remains close to its ASE endpoints: selected maximum differences are at
most `0.271 meV/atom`, `0.00821 A` raw position RMSD, and `0.00479 A` cell RMSD.
Full BFGS and BFGSLineSearch are trajectory-sensitive and can enter different
local minima. Their selected maxima reach `27.1/38.0 meV/atom` and
`1.09/2.06 A` raw position RMSD, respectively. These are throughput
measurements with matching convergence, not claims of identical minima. The
separate zero-step smooth-RMS validation establishes initial calculator parity.

## Limitations

- Each timing point has one observation, as requested for this screening stage.
- The H100-derived memory and batch policy must be recalibrated on other GPUs.
- The pool is homogeneous by atom count; heterogeneous edge-aware scheduling
  remains a separate workload.
- Raw position RMSD is descriptive and is not symmetry- or molecule-aligned.
