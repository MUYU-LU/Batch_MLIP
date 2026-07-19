# Workload-aware performance strategy

## Objective

Maximize completed-structure throughput without assuming that one batch size,
neighbor skin, refill rule, or GPU count is optimal for every task. The runtime
policy must account for:

- task type: single-point evaluation, optimization, or MD;
- pending-pool size;
- atom- and edge-count distribution;
- convergence-time distribution;
- graph-construction cost;
- model and optimizer memory;
- the number of available GPUs.

The optimizer remains independent of the MLIP. Model-specific graph translation
belongs in calculator adapters, while cache validity, workload planning, and
queue scheduling are generic runtime concerns.

## Current evidence

The existing BFGS experiments establish the following baseline:

- Active refill helps when convergence imbalance would otherwise leave a GPU
  underfilled. MACE B64 improved by 6.5-14.1% in elapsed time, while AtomBit
  B128 was between a 1.8% regression and a 3.1% improvement.
- Lazy construction of pending common graphs changed timing by only
  0.983x-1.011x and reduced peak allocation by less than 0.4%. It is useful
  cleanup, not an acceleration target.
- AtomBit and MACE both ultimately use
  `matscipy.neighbours.neighbour_list` on the periodic T2 structures. AtomBit
  calls it through the common neighbor wrapper; MACE calls it while constructing
  `AtomicData`.
- Preliminary one-shot measurements estimate raw neighbor search at 18-25% of
  AtomBit BFGS time for 92-276 atom structures. For MACE, raw search is about
  7-9%, while complete `AtomicData` graph preparation is about 17-22%.
- The corrected AtomBit variable-cell cache is bitwise exact against `skin=0`
  on paired 256-job B64 and B128 workloads. At B64 it improves wall time by
  5.4% for 46 atoms and 14.1% for 276 atoms.
- B128 is not automatically better: cached 276-atom B128 is 6.8% slower than
  cached B64 and uses 54.1 GB rather than 27.3 GB of peak allocated memory.

The phase percentages are estimates from isolated graph timings multiplied by
logged graph-evaluation counts. The next benchmark must add direct phase timing
before using them as optimization claims.

## Three coupled controls

### Neighbor topology cache

A cache trades less CPU graph construction for cache maintenance and possibly
more candidate edges. A useful policy must measure both sides:

```text
cache benefit = avoided rebuild time
              - candidate update/filter time
              - cache packing and transfer time
              - any added model edge work
```

The generic cache stores per-structure candidate topology at `cutoff + skin`,
integer periodic shifts, reference positions and cells, and validity metadata.
Current distances are updated on GPU. Edges outside the physical model cutoff
are filtered before model evaluation so that skin does not silently increase
message-passing work.

For fixed cells, the usual displacement bound is sufficient. For changing
cells, validity must conservatively include both atomic displacement and the
change in periodic image displacement induced by cell deformation. Cache
invalidation and rebuilding must be per structure; one dirty structure must not
force a rebuild of the complete resident batch.

Cache use is adaptive. It should be disabled when the observed reuse interval
is too short to repay filtering and maintenance costs. `skin=0` remains the
exact baseline and fallback.

### Active refill

Refill trades better resident occupancy for state insertion, graph packing, and
changing-batch overhead. It is useful when all of the following hold:

- the pending queue is substantially larger than the resident batch;
- systems leave the batch at different optimizer steps;
- the recovered model throughput exceeds refill overhead;
- enough work remains after a refill event.

The current policy fills available slots after each convergence check. When one
system finishes at a time, this is effectively one-job refill and can repeatedly
repack state. The next scheduler will support:

- `none`: drain the current resident batch;
- `immediate`: fill all vacancies at every convergence check;
- `threshold`: refill only below a low-water occupancy, then fill a chunk to the
  resident target.

The threshold and chunk are policy parameters, not public optimizer semantics.
An initial policy can use a low-water mark of 80% and a minimum chunk of
`max(8, resident_capacity // 8)`, but measured packing and inference times must
decide the final values.

Refill should be disabled for a pool that fits in one resident batch, for
single-point evaluation, and normally for fixed-length MD trajectories.

### Resident capacity and GPU use

Resident capacity is a memory budget, not a structure count. The planner must
estimate at least:

- atoms and directed edges;
- model activations and temporary tensors;
- graph packing and transfer buffers;
- optimizer state;
- full-BFGS Hessians, which scale approximately as `(3 * atoms + 9)^2` per
  variable-cell structure.

Small homogeneous structures can use a high resident count. Large structures
may require a much lower count even when the model graph fits, because the BFGS
Hessian becomes dominant. Mixed workloads require memory/edge-aware packing;
atom-count-only bucketing is an approximation.

GPU memory is not pooled across devices. Each GPU owns an independent resident
batch, optimizer histories, and pending queue. A process-per-GPU design avoids
cross-device model-batch communication. A central dispatcher may assign new
jobs or permit job-level work stealing, but active optimizer histories remain
on their owning GPU.

## Workload policy

| Workload | Cache | Refill | Resident and GPU policy |
|:--|:--|:--|:--|
| Single-point, small pool | No reuse opportunity | Off | One memory-safe packed batch |
| Single-point, large pool | No reuse opportunity | Off | Memory/edge-aware batches across GPU workers |
| Optimization, few small systems | Use if multiple steps | Off | Usually one GPU and one batch |
| Optimization, few large systems | Use if valid long enough | Off | Small batches; shard independent systems if useful |
| Many homogeneous systems, similar convergence | Use if profitable | Usually off | Large static resident batches |
| Many homogeneous systems, varied convergence | Use if profitable | Threshold refill | Keep each GPU near its memory-safe target |
| Many mixed systems | Per-structure policy | Threshold refill within compatible buckets | Edge/Hessian-aware packing and work stealing |
| Fixed-length MD replicas | High priority | Off | Persistent replica batches across GPUs |

## Generic runtime design

The runtime should expose capabilities rather than MLIP names:

1. `WorkloadProfiler` records atoms, edges, phase timings, memory, convergence
   exits, cache reuse, and refill events.
2. `BatchPlanner` predicts memory cost and packs compatible jobs into a target
   resident budget.
3. `NeighborCachePolicy` owns generic candidate topology and per-system
   validity decisions.
4. `RefillPolicy` selects `none`, `immediate`, or `threshold` scheduling.
5. One `GpuWorker` owns a calculator, resident optimizer state, and queue.

AtomBit can consume generic graph tensors directly. A MACE adapter may translate
the same cached topology into MACE tensors, but `AtomicData` must not enter the
optimizer, scheduler, or generic cache API. Calculators unable to accept an
external cached topology retain their native rebuild fallback.

## Staged experiments

Experiments are decision-gated. A stage that fails its correctness or
performance criterion stops expansion of that approach.

### Stage 0: direct phase instrumentation

**Hypothesis:** direct timers will identify whether graph preparation, model
evaluation, BFGS updates, or scheduling dominates each representative workload.

Instrument synchronized wall time for neighbor search, graph conversion and
transfer, model evaluation, BFGS update/eigensolve, compaction, and refill.
Also record active systems, atoms, edges, cache rebuilds, refill sizes, and peak
GPU memory per optimizer evaluation.

Instrumentation overhead must remain below 1% on a warmed representative run.
This stage changes measurement only, not numerical behavior.

### Stage 1: generic variable-cell neighbor cache

**Hypothesis:** per-structure topology reuse with exact-cutoff GPU filtering
reduces wall time when a candidate graph survives multiple BFGS evaluations.

Correctness pilots use B1 and a small heterogeneous batch for fixed and variable
cells. Every evaluation is compared with `skin=0` for active edges, energy,
forces, and stress. Optimization gates include convergence flags, step counts,
and final observables; nearby-minimum divergence is reported separately from
per-step force/stress disagreement.

Performance screening uses four paired scenarios:

| MLIP | atoms | workload | resident | comparison |
|:--|--:|--:|--:|:--|
| AtomBit | 46 | 256 | 64 | `skin=0` versus cached `skin=0.5 A` |
| AtomBit | 276 | 256 | 64 | `skin=0` versus cached `skin=0.5 A` |
| MACE-OFF-Small | 46 | 256 | 64 | native rebuild versus generic cache adapter |
| MACE-OFF-Small | 276 | 256 | 64 | native rebuild versus generic cache adapter |

Proceed to B128 confirmation only if the cache is correct and either reduces
wall time by at least 5% or removes at least half of graph-preparation time
without a wall-time regression. Record candidate-edge inflation, physical-edge
count, mean evaluations per rebuild, and dirty-system distribution.

**Result:** AtomBit passes the gate. Paired physical graphs and final records
are exact after matching ASE's float64 cutoff decision. B64 speedups are 5.4%
at 46 atoms and 14.1% at 276 atoms. B128 confirms a 3.1% and 6.3% cache benefit
against its own baseline, but cached B128 does not beat cached B64. Use B64 with
`skin=0.5 A` for the measured 276-atom workload and proceed to Stage 2. The MACE
cache adapter remains unimplemented and is not included in this conclusion.

### Stage 2: refill policy

**Hypothesis:** threshold/chunk refill retains the occupancy benefit of immediate
refill while reducing repeated insertion and packing overhead.

Use cases are selected from existing evidence rather than a full matrix:

- MACE 46 atoms, B64: known positive immediate-refill case.
- AtomBit 276 atoms, B128: known neutral immediate-refill case.
- A 256-job 50:50 mixture of 46- and 276-atom structures for each MLIP.

For the two homogeneous cases, add threshold refill and reuse matched drain and
immediate policies under the new instrumentation. For mixed cases, compare
`none`, `immediate`, and `threshold`. Record occupancy over time, refill-event
count and size, packing time, graph/model time, and completed systems per second.

Retain threshold refill only when it improves a representative case by at least
5% without degrading the known positive MACE case. A pool no larger than the
resident capacity must select `none` without a timing experiment.

### Stage 3: memory-aware resident planning and bucketing

**Hypothesis:** predicted edge, model, and Hessian memory permits higher safe
occupancy for small jobs and prevents large or mixed jobs from forcing a poor
global structure-count batch size.

Calibrate memory on 46-, 92-, 184-, and 276-atom samples, then predict a target
with explicit safety headroom. Evaluate one planned run for each MLIP on:

- a small 32-job 50:50 mixture of 46- and 276-atom structures;
- 256 homogeneous 46-atom jobs;
- 256 homogeneous 276-atom jobs;
- the 256-job 50:50 mixed workload.

The small pool tests whether planning and bucketing overhead should be bypassed.
The large homogeneous endpoints test the highest and lowest useful resident
counts, and the mixed case tests memory-aware packing. Compare with the best
existing fixed B64/B128 policy where it applies. Require no OOM, bounded
prediction error, unchanged numerical gates, and at least a 5% throughput gain
in one regime without a material regression in the others.

### Stage 4: multi-GPU sharding

**Hypothesis:** independent memory-aware GPU workers scale aggregate completed
systems per second without changing per-job results.

Only the winning single-GPU policies advance to this stage. Measure 1, 4, and 8
GPU workers on one homogeneous and one mixed 256-job workload for each MLIP.
Report aggregate throughput, per-GPU occupancy and memory, load imbalance,
scheduler overhead, and parallel efficiency. Do not split one structure or one
BFGS history across GPUs.

Also compare 1 versus 8 workers for the small 32-job mixed pool. This is a
deliberate negative-control case: the planner should use fewer GPUs when
dispatch and model-replication overhead would dominate. Correctness is checked
by original job identifier, independent of completion order or owning GPU.

## Measurement and decision rules

- Use the fixed T2 manifest and record the exact filename sequence.
- Warm model, stress/autograd, and graph paths before timing.
- Synchronize CUDA around direct phase timers.
- Screening uses one complete run per point. Differences below 2% are
  inconclusive, not speedups.
- Only policies selected for a performance claim receive confirmatory timing
  repeats, satisfying the repository experiment protocol without repeating the
  full screening matrix.
- Report systems/s, atoms/s, edges/s, phase times, peak memory, active-batch
  distribution, rebuild count, and convergence distribution.
- Preserve B1, batch-versus-single, stress, optimizer, graph-isolation, and
  deterministic validation gates.
- Keep negative results. A policy that is useful only in a specific workload
  remains conditional rather than becoming the default.

## Execution order

The next code change is Stage 0 instrumentation. The first acceleration tested
afterward is the generic variable-cell neighbor cache. Refill-policy changes,
memory-aware planning, and multi-GPU scheduling follow only after their required
measurements are available.
