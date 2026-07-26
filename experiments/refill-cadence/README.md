# Periodic refill cadence

This experiment tests whether accumulating completed resident slots before a
physical refill mutation can outperform immediate refill. Finished jobs are
detected and frozen every step; only their physical removal or replacement is
deferred to intervals of 2 or 5 scheduler steps. Interval 1 is the existing
immediate-refill control.

## Results

Every point optimized the same signed 256-job H46 pool once. All jobs converged.
Speedup is relative to interval 1 for the same model, optimizer, and resident
batch.

| Model | Optimizer | Batch | K2 speedup | K5 speedup | K5 frozen graphs | K1/K5 refill events | K1/K5 scheduler time (s) |
|:--|:--|--:|--:|--:|--:|:--|:--|
| AtomBit | BFGS | 64 | 1.013x | 0.960x | 490 | 176 / 95 | 1.153 / 0.658 |
| AtomBit | BFGS | 128 | 1.012x | **1.059x** | 487 | 123 / 59 | 2.231 / 0.849 |
| MACE | BFGS | 64 | 0.977x | 0.996x | 630 | 185 / 102 | 1.002 / 0.577 |
| MACE | BFGS | 128 | 1.009x | 1.012x | 630 | 120 / 67 | 1.650 / 0.838 |
| AtomBit | FIRE | 64 | 0.958x | 1.008x | 420 | 181 / 125 | 1.167 / 0.803 |
| AtomBit | FIRE | 128 | 0.985x | 0.995x | 429 | 122 / 85 | 2.289 / 1.542 |
| MACE | FIRE | 64 | 1.007x | 1.002x | 477 | 195 / 115 | 1.062 / 0.687 |
| MACE | FIRE | 128 | 0.996x | 0.999x | 477 | 117 / 70 | 1.726 / 0.976 |

Periodic refill does reduce mutation count and scheduler time. Usually the
extra inference on frozen graphs consumes that saving. The sole point above the
2% performance gate is AtomBit BFGS, B128, interval 5: `59.69 s` versus
`63.21 s`, or `1.059x`. It converges 256/256 and differs by at most
`0.065 meV/atom` at the endpoint, but its converged step counts differ by up to
4 because the delayed admission order changes floating-point graph reductions.
It therefore fails the predeclared exact-step gate.

MACE trajectories preserve converged steps exactly at both intervals, but no
MACE point reaches the speed gate. AtomBit FIRE and the other AtomBit BFGS
points also change step counts.

## Decision

Keep `refill_interval=1` as the default and do not add periodic cadence to the
automatic planner. Retain the explicit parameter as an experimental mechanism.
Stage 2 H276/MIX expansion was not run because no candidate passed both
predeclared gates.

The 24 raw outputs are retained in `results/raw-results.tar.gz`. Derived rows
are in `results/summary.json` and `results/summary.csv`.
