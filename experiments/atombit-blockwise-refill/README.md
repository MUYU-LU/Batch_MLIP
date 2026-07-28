# AtomBit blockwise refill

This experiment isolates two different meanings of `steps_between_swaps` for
variable-cell AtomBit BFGS on signed 256-job pools at resident B64:

- **Frozen K5:** detect convergence every step, freeze completed residents, and
  mutate the resident batch every five scheduler steps.
- **Block K5:** detect convergence and mutate the resident batch only every five
  scheduler steps, allowing up to four additional BFGS steps per job.

Active drain and immediate refill are the controls. Every case uses the same
smooth-rms fp32 checkpoint, float64 optimizer state, deterministic settings,
and one synchronized timing trial.

## Results

| Workload | Mode | Wall (s) | Speedup vs active | Speedup vs immediate | Reserved (GiB) | Model evals | Graph evals | BFGS steps |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| H46 | Active | 80.18 | 1.000x | 0.873x | 7.94 | 1,076 | 30,348 | 30,092 |
| H46 | Immediate | 69.97 | 1.146x | 1.000x | 23.04 | 663 | 30,380 | 30,124 |
| H46 | Frozen K5 | **68.41** | **1.172x** | **1.023x** | 23.04 | 670 | 30,870 | 30,124 |
| H46 | Block K5 | 83.64 | 0.959x | 0.837x | 38.79 | 766 | 37,474 | 37,218 |
| STEPVAR-H276 | Active | 107.59 | 1.000x | 0.945x | 42.78 | 740 | 18,720 | 18,464 |
| STEPVAR-H276 | Immediate | **101.72** | **1.058x** | **1.000x** | 78.34 | 416 | 18,738 | 18,482 |
| STEPVAR-H276 | Frozen K5 | 102.96 | 1.045x | 0.988x | 78.26 | 421 | 19,211 | 18,554 |
| STEPVAR-H276 | Block K5 | 143.82 | 0.748x | 0.707x | 78.16 | 656 | 26,184 | 25,928 |

All 2,048 jobs converged according to the variable-cell generalized-force
criterion. Physical atomic-force maxima can slightly exceed `0.05 eV/A`
because `smax=None` uses the Frechet generalized force, not the atomic-force
maximum, as the stopping quantity.

## Mechanism

Block K5 reduces convergence checks and refill events, but a check is much
cheaper than an AtomBit force/stress evaluation. It does not skip model calls:
acceptable residents keep moving until the next boundary. Relative to
immediate refill, this adds 7,094 H46 and 7,446 STEPVAR optimizer steps. Maximum
endpoint energy differences are `20.17` and `7.81 meV/atom`, respectively, so
block K5 fails both the speed and endpoint gates.

Frozen K5 isolates swap overhead without deliberately overshooting convergence.
For H46 it reduces refill events from 176 to 95 and is 2.3% faster than
immediate refill. It is 1.2% slower on STEPVAR-H276, and both H276 refill modes
reserve about 98.8% of the H100 memory, above the automatic policy's 85% limit.

## Decision

Reject blockwise convergence/refill K5. Keep `convergence_check_interval=1` as
the default. Retain delayed physical refill as an explicit experimental
parameter, but do not enable it automatically based on one H46 timing trial.
The production planner remains active drain because it is memory-safe across
these cases. A future H276 refill policy must either choose a smaller resident
capacity or reuse variable-shape graph and autograd buffers well enough to keep
the CUDA allocator below the 85% reserved-memory gate; pending neighbors are
already constructed lazily.

Raw outputs are archived in `results/raw-results.tar.gz`; derived values are in
`results/summary.json` and `results/summary.csv`.
