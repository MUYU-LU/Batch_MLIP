# OMC-CSP bounded streaming refill v1

## Question

Can source-backed micro-pools retain the useful inner refill behavior measured
at eight GPUs without the startup, static-assignment, and large private-queue
penalties observed for `omc-csp-multigpu-local-refill-v1`?

This experiment keeps pending structures as manifest references until a bounded
micro-pool is requested. Each micro-pool contains at most two resident waves,
uses the worst-case memory-safe capacity derived by the signed planner, and is
executed by one persistent GPU worker using fixed-slot refill. The outer
scheduler retains undispatched micro-pool ownership and work stealing. A worker
keeps all live BFGS, cell-filter, and candidate-graph state; no state is moved
between GPUs.

This v1 mechanism is source-backed at micro-pool granularity. It does not yet
implement a live slot-by-slot worker request channel: one complete two-wave
micro-pool is materialized before its task is dispatched.

## Contract

The held-out homogeneous OBEQIX P2048 manifest, checkpoint, BFGS/Frechet cell
contract, deterministic settings, 85%-memory capacity model, and convergence
criteria are unchanged from the paired active-drain and private-refill runs.
Every method receives the same 2,048 source structures.

The planner derives resident capacity B131 from the globally most expensive
first wave. It constructs eight deterministic, cost-balanced micro-pools of 256
structures. Thus each task contains one resident wave plus at most 125 pending
replacements. The same eight tasks are used at G2, G4, and G8.

## Results

Each row is one deterministic H100 execution. `Speedup active` and `Speedup
private` use execution time. Peak memory is the maximum reserved memory on any
worker. The predefined gates are at least 1.02x speedup over active drain, at
most 85% peak memory, 2,048 converged structures, and at most 5 meV/atom maximum
endpoint-energy difference from the paired active-drain run.

| GPUs | Active (s) | Private (s) | Streaming (s) | Speedup active | Speedup private | Production speedup | Peak streaming | Max dE (meV/atom) | Result |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 2 | 339.143 | 351.185 | 337.127 | 1.006x | 1.042x | 1.006x | 51.28 GB | 5.043 | speed and endpoint fail |
| 4 | 190.983 | 197.009 | 186.561 | 1.024x | 1.056x | 1.024x | 51.28 GB | 5.043 | endpoint fail |
| 8 | 138.980 | 124.379 | 122.415 | 1.135x | 1.016x | 1.168x | 51.28 GB | 5.043 | endpoint fail |

All runs converged 2,048/2,048 structures. Streaming uses 60.3% of the
85.02-GB H100, compared with approximately 77.9% for active drain. It performs
2,413 model evaluations and 165,903 graph evaluations at every GPU count, with
1,000 fixed-slot insertions. Active drain performs 2,613, 2,613, and 3,528
model evaluations at G2, G4, and G8.

The endpoint failure is one shared outlier out of 2,048. Its difference is
5.0425 meV/atom; the 99th percentiles are 0.0329, 0.0329, and 0.0845 meV/atom.
The convergence flags are identical. The predefined maximum gate is not
relaxed after observing the result.

## Interpretation

Bounded micro-pools remove the private-refill regressions: outer work stealing
balances four, two, and one tasks per worker at G2, G4, and G8, respectively,
and dispatch wait after the initial task is below 0.3 ms. This recovers 4.2-5.6%
relative to private refill at G2/G4 while preserving its reduced memory and
model-call count.

The G2 gain over active drain is only 0.6%, below the 2% gate. Active drain can
already keep two GPUs occupied with its 12 edge-aware B131-B217 chunks, and
larger edge-aware batches offset its additional draining calls. At G8, active
drain subdivides into 16 execution tasks and repeats more drain tails. Eight
two-wave refill tasks avoid 31.6% of its model calls and improve production
time by 16.8%.

## Decision

Do not promote bounded streaming refill into the automatic OMC-CSP default:

- G2 does not pass the predefined performance gate;
- the strict endpoint gate fails by 0.0425 meV/atom for one structure;
- this evidence covers one homogeneous family and not mixed-shape micro-pools;
- v1 materializes a complete two-wave task rather than admitting one source
  reference into an empty slot on demand.

Keep `manifest_multi_gpu_refill_policy="streaming_compatible"` explicit-only.
The accepted automatic policy remains active drain. The mechanism establishes
that bounded source-backed scheduling solves the static private-queue problem,
but automatic promotion requires an in-flight eligibility rule plus validation
on additional homogeneous and mixed OMC-CSP workloads.

Machine-readable results are in `results/summary.json`. Raw result JSON and
logs remain on the H100 experiment host.
