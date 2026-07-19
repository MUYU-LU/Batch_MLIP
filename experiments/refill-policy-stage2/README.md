# Stage 2 refill-policy benchmark

## Implementation

Full batched BFGS now accepts three generic pending-queue policies:

- `drain`: finish the current resident wave before admitting another wave;
- `immediate`: fill every vacancy after active compaction;
- `threshold`: refill to capacity only after occupancy crosses a low-water mark
  and at least a minimum chunk of slots is available.

Immediate refill remains the default. The measured threshold used an 80% low
water mark and a minimum chunk of 8 for resident B64. Finished systems are
always actively compacted; no masked chunks are included.

## Matrix

Each point is one deterministic timing over the same 256-job workload.

| model | atoms | resident | graph policy | refill policies |
|:--|--:|--:|:--|:--|
| AtomBit | 276 | 64 | corrected `skin=0.5 A` cache | drain, immediate, threshold |
| MACE-OFF-Small | 46 | 64 | native MACE graphs | drain, immediate, threshold |

The mixed-workload extension was conditional on threshold improving a
representative case by at least 5%. It was not run after that gate failed.

## Performance

| model | policy | time (s) | systems/s | evaluations | mean occupancy | triggered refills | mean insertion | repack (s) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| AtomBit | drain | 341.753 | 0.749 | 1,160 | 17.59 | 3 | 64.00 | 1.259 |
| AtomBit | immediate | **334.485** | **0.765** | 514 | 39.17 | 123 | 1.56 | 1.890 |
| AtomBit | threshold | 341.155 | 0.750 | 553 | 36.78 | 15 | 12.80 | 1.945 |
| MACE | drain | 145.439 | 1.760 | 1,240 | 27.31 | 3 | 64.00 | 1.451 |
| MACE | immediate | 130.401 | 1.963 | 725 | 46.70 | 157 | 1.22 | 2.510 |
| MACE | threshold | **129.376** | **1.979** | 779 | 43.47 | 15 | 12.80 | 2.102 |

For MACE, threshold is 0.8% faster than immediate and 12.4% faster than drain.
For AtomBit, threshold is 2.0% slower than immediate and only 0.2% faster than
drain. Immediate is 2.2% faster than drain for AtomBit and 11.5% faster for
MACE.

## Interpretation

Threshold refill successfully changes insertion granularity: 123 to 15
triggered refills for AtomBit and 157 to 15 for MACE. It does not eliminate the
dominant convergence-triggered operation. Every exit still compacts survivors,
reconstructs the resident state, and invalidates packed graph state. AtomBit
therefore sees no repack-time reduction, while lower occupancy adds 39 model
evaluations. MACE saves 0.41 seconds of repacking, too little to produce a 5%
end-to-end improvement.

All structures converge under every policy. MACE step counts match exactly;
its maximum policy-dependent position difference is `4.87e-5 A`. AtomBit uses
a float32 model, and changed batch schedules send 46-50 of 256 structures along
different numerical trajectories. Convergence flags still match, but step and
final-minimum differences mean policies must be compared as scheduling modes,
not as bitwise trajectory-preserving transformations.

## Decision

The threshold policy fails the 5% retention gate and is not selected as the
default. Immediate refill remains the production policy for both measured
workloads. Threshold and drain remain explicit interfaces for workload testing
and reproducible baselines.

The mixed matrix is stopped. A future threshold attempt first requires
topology-preserving or in-place active compaction so that deferring insertion
also avoids graph repacking. The planned next stage is memory-aware resident
capacity and bucketing.
