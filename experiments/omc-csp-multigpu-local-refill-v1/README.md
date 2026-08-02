# OMC-CSP multi-GPU local refill v1

## Question

Can the compatible empty slots measured by
`omc-csp-outer-inner-opportunity-v1` be recovered without a live cross-GPU
control channel?

The experimental policy assigns one deterministic, cost-balanced private queue
to each GPU. Each inner variable-cell BFGS optimizer uses immediate fixed-slot
refill and drains after its private queue is exhausted. Optimizer state is never
migrated between GPUs.

The workload is the held-out homogeneous OBEQIX P2048 manifest. Active drain is
the unchanged automatic baseline with bucket-stratified, cost-descending work
stealing. Both methods use the signed 85%-memory hardware policy and the same
AtomBit, BFGS, Frechet cell, graph, and convergence contract.

## Capacity

OBEQIX has fixed atom count and BFGS dimension but variable candidate-edge
counts. The signed active-drain plan packs memory-safe resident chunks from
B131 to B217. B217 is not safe for arbitrary fixed-slot replacement because a
later high-edge structure can enter that slot population.

The local policy therefore derives B131 from the first planner wave. The
planner orders systems by decreasing incremental memory, so this is the largest
capacity already proven safe for the globally most expensive structures. No
batch size is hard-coded.

## Results

Each row is one deterministic H100 execution. `Speedup` is active-drain time
divided by local-refill time. Peak memory is the maximum reserved memory on any
worker GPU. The predefined gates are at least 1.02x execution speedup, at most
85% peak memory, complete convergence, and at most 5 meV/atom endpoint-energy
difference from the paired active-drain run.

| GPUs | Active (s) | Local (s) | Speedup | Production speedup | Model calls active/local | Peak active/local | Max dE (meV/atom) | Result |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 2 | 339.143 | 351.185 | 0.966x | 0.964x | 2613 / 1549 | 66.25 / 52.08 GB | 2.835 | speed fail |
| 4 | 190.983 | 197.009 | 0.969x | 0.969x | 2613 / 1819 | 66.25 / 51.58 GB | 3.315 | speed fail |
| 8 | 138.980 | 124.379 | 1.117x | 1.160x | 3528 / 2413 | 66.23 / 51.28 GB | 5.043 | endpoint fail |

All three local runs converged 2,048/2,048 structures. Local peak memory is
60.3-61.3% of the 85.02-GB H100, compared with 77.9% for active drain. Fixed
slot insertion counts were 1,786, 1,524, and 1,000 at G2, G4, and G8.

Endpoint differences are highly concentrated. At G8, only one structure
exceeds 5 meV/atom and the 95th percentile is 0.000196 meV/atom, but the
predeclared maximum gate is not changed after observing the result.

## Interpretation

At G2/G4, the active plan already contains 12 full edge-aware resident chunks,
enough to feed the GPUs without additional subdivision. Local refill eliminates
draining model calls, but forcing every queue to conservative B131 reduces model
throughput. It also materializes 1,024 or 512 structures before dispatch and
removes outer work stealing. The net regressions are 3.6% and 3.2%.

At G8, the active outer policy subdivides 12 resident chunks into 16 execution
tasks to maintain two tasks per GPU. Every task develops its own drain tail.
Eight B131 refill queues avoid those repeated tails and reduce model calls by
31.6%, producing the measured 11.7% execution speedup. This is a real regime
boundary, not evidence for universal refill.

## Decision

Do not promote private local refill into the automatic OMC-CSP policy:

- G2/G4 fail the speed gate;
- G8 fails the strict endpoint gate;
- large private tasks increase initial materialization and prevent outer work
  stealing;
- a single conservative capacity cannot exploit edge-aware B131-B217 packing.

The implementation remains explicit-only through
`manifest_multi_gpu_refill_policy="local_compatible"`. The production default
remains active drain.

The follow-up `omc-csp-multigpu-streaming-refill-v1` uses bounded two-wave
source-backed micro-pools and retains outer work stealing. It removes the G2/G4
private-queue regressions and preserves the G8 gain, but it is not promoted:
G2 remains below the 1.02x performance gate and the unchanged endpoint gate has
one 5.043-meV/atom outlier.

Machine-readable results are in `results/summary.json`. Raw JSON and logs remain
on the H100 experiment host.
