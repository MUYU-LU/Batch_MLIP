# Variable-cell neighbor-cache scaling

## Scope

AtomBit optimized a fixed 256-structure workload with full variable-cell BFGS,
active refill, and deterministic mixed precision. Each paired case used the same
structures and resident schedule; only the neighbor skin changed. Timings are
single screening runs, as requested, rather than repeated statistical results.

The model and optimizer used float32 and float64 respectively. Candidate lists
used `cutoff + 0.5 A`, while the physical graph was filtered at the 6.0 A model
cutoff before message passing.

## Correctness correction

The first B64 large-system run exposed two missing directed edges at evaluation
31. ASE performs its cutoff comparison in float64, while the cached physical
filter used float32. A stored separation of `5.9999998969 A` can round to
exactly `6.0 A` in float32 and be excluded incorrectly.

The physical cutoff decision now uses float64 without changing model precision.
A regression test covers this boundary. All measurements below use the corrected
filter. Within every skin-zero/cached pair, active-batch histories, physical edge
counts, optimizer steps, convergence flags, and all 256 final records are exact.

## Results

| atoms | resident | skin | time (s) | systems/s | neighbor (s) | rebuilt systems | candidate/physical | peak allocated (GB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 46 | 64 | 0.0 | 129.994 | 1.969 | 45.389 | 28,227 | 1.000x | 4.872 |
| 46 | 64 | 0.5 | 123.282 | 2.077 | 37.609 | 11,335 | 1.220x | 4.878 |
| 276 | 64 | 0.0 | 383.696 | 0.667 | 126.064 | 20,134 | 1.000x | 27.289 |
| 276 | 64 | 0.5 | 336.147 | 0.762 | 80.647 | 9,350 | 1.200x | 27.320 |
| 46 | 128 | 0.0 | 126.773 | 2.019 | 46.462 | 28,212 | 1.000x | 9.699 |
| 46 | 128 | 0.5 | 123.005 | 2.081 | 42.736 | 12,382 | 1.220x | 9.711 |
| 276 | 128 | 0.0 | 381.764 | 0.671 | 123.315 | 20,561 | 1.000x | 54.064 |
| 276 | 128 | 0.5 | 359.010 | 0.713 | 100.612 | 11,316 | 1.200x | 54.126 |

| atoms | resident | cache speedup | neighbor-time reduction | rebuilt-system reduction |
|--:|--:|--:|--:|--:|
| 46 | 64 | 5.4% | 17.1% | 59.8% |
| 276 | 64 | 14.1% | 36.0% | 53.6% |
| 46 | 128 | 3.1% | 8.0% | 56.1% |
| 276 | 128 | 6.3% | 18.4% | 45.0% |

## Decision

The cache passes the Stage 1 correctness and performance gates for AtomBit.
`skin=0.5 A` is retained as an available policy, with adaptive enablement still
required for workloads that have short reuse intervals.

B128 is not the best large-system operating point. Cached 276-atom B128 is 6.8%
slower than cached B64 and doubles peak allocated memory from 27.3 to 54.1 GB.
For this workload, B64 with `skin=0.5 A` is the measured choice. At 46 atoms,
cached B64 and B128 are effectively tied (123.282 versus 123.005 seconds), so
the extra B128 memory has no meaningful return.

The next stage is refill-policy measurement with the cache setting fixed per
workload. MACE remains on its native graph fallback until a generic topology
translation adapter is implemented and separately validated.
