# BFGS active refill

## Setup

- Same 32 fixed T2 structures per 46, 92, 184, and 276-atom group,
  repeated in order to a 256-job workload.
- Full variable-cell BFGS with the Frechet cell filter, `fmax=0.05 eV/A`,
  `alpha=70 eV/A^2`, `maxstep=0.2 A`, and a 500-step per-system limit.
- AtomBit uses float32 inference with float64 BFGS/Frechet state. MACE-OFF-Small
  uses float64 inference and optimizer state.
- One measured run per point. Active-drain B64 and B128 measure one resident
  batch and multiply time by four and two. Refill directly optimizes all 256
  jobs. ASE measures 32 jobs and multiplies by eight.

## Results

Seconds are for the equivalent 256-job workload. `gain` is active-drain time
divided by active-refill time.

| model | atoms | ASE | drain B64 | refill B64 | gain | drain B128 | refill B128 | gain |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| AtomBit | 46 | 1026.95 | 125.21 | 115.40 | 1.085x | 112.93 | 109.43 | 1.032x |
| AtomBit | 92 | 1204.06 | 230.69 | 213.50 | 1.081x | 206.23 | 209.84 | 0.983x |
| AtomBit | 184 | 1774.37 | 297.86 | 284.62 | 1.047x | 282.34 | 283.05 | 0.997x |
| AtomBit | 276 | 2429.03 | 326.21 | 324.75 | 1.004x | 320.33 | 321.89 | 0.995x |
| MACE | 46 | 1327.63 | 151.21 | 129.91 | 1.164x | 130.03 | 123.65 | 1.052x |
| MACE | 92 | 1309.91 | 235.03 | 208.46 | 1.127x | 213.91 | 203.67 | 1.050x |
| MACE | 184 | 1795.54 | 296.37 | 276.94 | 1.070x | 277.49 | 270.48 | 1.026x |
| MACE | 276 | 3094.05 | 340.95 | 315.66 | 1.080x | 329.46 | 313.41 | 1.051x |

All refill points converged 256/256 jobs. Refill B64 improved every point,
from 0.4% to 16.4%. B128 improved every MACE point by 2.6-5.2%, but AtomBit
92-276 was within 1.7% of active drain and slightly slower. The result indicates
that refill helps when convergence imbalance leaves useful GPU capacity, but it
cannot overcome an already saturated resident batch.

Refill B64/B128 peak allocated memory ranged from 4.31/8.54 GiB at MACE 46
atoms to 25.61/50.94 GiB at MACE 276 atoms. Every requested capacity completed
without an out-of-memory result.

## Correctness

The exact CPU variable-cell test confirms that refill preserves each survivor's
Hessian, previous coordinate/force vectors, Frechet deformation state, local
step count, and original output index. The default suite passes with 44 tests
and two optional MACE tests skipped; the opt-in MACE suite passes both FIRE and
BFGS integration tests.

Every long production comparison has identical convergence flags. Most final
state gates pass, but AtomBit 46-B64, 92-B64, 276-B64/B128 and MACE 276-B128
select different nearby minima. This is consistent with the existing BFGS
audit: full BFGS amplifies microscopic batch-composition differences over long
nonconvex trajectories. These endpoint failures are retained in `results.json`
and are not used as evidence of a refill state-equation error.

## Limitation

The structure-level API currently creates graph state for all 256 pending jobs
before selecting the resident 64 or 128. This eager queue construction adds
CPU neighbor-list time and makes queued graph tensors coexist with the resident
model batch. A lazy pending-state queue is the next refill optimization.
