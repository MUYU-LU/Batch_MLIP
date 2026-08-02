# OMC-CSP compatible refill v1

This experiment isolates deterministic compatibility-aware fixed-slot refill
from refill cadence and neighbor-backend selection.

The first implementation matches atom count because variable-cell full BFGS
has dimension `(3N + 9)^2`. Candidate-edge count is measured and reported, but
is not a correctness constraint: incoming candidate graphs are rebuilt exactly,
while survivor graph blocks and optimizer state remain unchanged.

## Result

The mechanism is correct but fails the declared 2% performance gate. Each row
is one deterministic H100 timing of the same signed 256-unique-CIF pool.

| Workload | Batch | Active (s) | FIFO slots (s) | Compatible slots (s) | Compatible/FIFO | Complete matches |
|:--|--:|--:|--:|--:|--:|--:|
| SOXLEX48 | 64 | 28.811 | 20.497 | 20.167 | 1.016x | 96 |
| SOXLEX48 | 128 | 22.636 | 19.256 | 19.322 | 0.997x | 35 |
| ROFA-MIX | 32 | 44.529 | 43.419 | 43.157 | 1.006x | 26 |
| ROFA-MIX | 64 | 39.137 | 37.015 | 37.406 | 0.990x | 0 |
| ROFA-MIX | 128 | 36.509 | 35.485 | 35.614 | 0.996x | 0 |

All compatible-slot cases converged 256/256 and passed the 5 meV/atom endpoint
gate against active drain. The SOXLEX results show that refill itself remains
valuable for a shape-homogeneous variable-duration pool: 1.429x at B64 and
1.172x at B128. Compatible search adds no material benefit because ordinary
FIFO slots are already compatible.

ROFA-MIX is ordered in four exact atom-count blocks of 64 jobs. At B64/B128,
released slots and the next pending block have different full-BFGS dimensions,
so no complete fixed-slot cohort exists. At B32, compatible replacement occurs
26 times but improves FIFO by only 0.6%. The fallback prevents dead slots and
preserves correctness, but searching across an incompatible queue does not
solve structural heterogeneity.

Threshold/arena refill was not selected. It was slower than immediate slots on
SOXLEX, and its SOXLEX B64 endpoint exceeded the declared 5 meV/atom gate.

## Scheduler Decision

The OMC-CSP outer scheduler must construct refill queues with exact optimizer
shape compatibility. For full variable-cell BFGS this initially means equal
atom count and therefore equal `(3N + 9)` dimension. Candidate-edge variation
remains allowed and is handled by the graph cache and automatic neighbor
backend.

Within each compatible queue:

- use immediate fixed slots only while compatible pending jobs remain;
- preserve survivor BFGS, Frechet, and candidate-graph state;
- initialize incoming optimizer state and rebuild only its invalid graph;
- drain the tail after the compatible queue is exhausted.

Across incompatible queues, assign complete micro-pools through the outer
scheduler. Do not use fixed-slot refill across atom-count boundaries. Arena or
repack remains an explicit fallback, not the accepted OMC-CSP default.

The online capacity controller now enforces this boundary: low measured
occupancy cannot enable refill for a mixed-atom-count bucket. The offline
evidence policy already had the same restriction.

The profile also shows why a more elaborate slot matcher is low priority.
Model forward/autograd and BFGS dominate runtime; recorded refill mutation is a
small fraction of the full optimization. The automatic outer scheduler now
performs planner-level compatible micro-pool construction: it groups by exact
`(atom_count, dof_squared)`, derives a conservative resident capacity from the
already memory-safe chunks, and extracts a fixed-slot queue only if packaged
evidence accepts that subgroup. If no subgroup is accepted, the original
active-drain chunks are preserved exactly. The one-by-one compatible pending
slot search remains diagnostic-only.

Machine-readable results are in `results/summary.json` and `results/summary.csv`.
Raw JSON and logs are retained in `results/raw-results.tar.gz`.
