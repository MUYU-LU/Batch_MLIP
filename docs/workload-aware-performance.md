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
- Low-water chunk refill reduces triggered insertions from 123 to 15 for the
  AtomBit 276/B64 workload, but is 2.0% slower than immediate refill. For MACE
  46/B64 it is only 0.8% faster. Immediate remains the selected policy.
- The atom/edge/Hessian memory model predicts held-out B128 peaks within 1.30%
  for AtomBit and 0.36% for MACE. Mixed-workload bucketing is memory-safe but
  improves throughput by only 4.82% and 3.99%, below the 5% gate.

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
| Many homogeneous systems, varied convergence | Use if profitable | Immediate refill; threshold is experimental | Keep each GPU near its memory-safe target |
| Many mixed systems | Per-structure policy | Refill only with matched target evidence | Edge/Hessian-aware packing; conservative drain otherwise |
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
- AtomBit 276 atoms, B64 with `skin=0.5 A`: selected Stage 1 operating point.
- A 256-job 50:50 mixture of 46- and 276-atom structures for each MLIP.

For the two homogeneous cases, add threshold refill and reuse matched drain and
immediate policies under the new instrumentation. For mixed cases, compare
`none`, `immediate`, and `threshold`. Record occupancy over time, refill-event
count and size, packing time, graph/model time, and completed systems per second.

Retain threshold refill only when it improves a representative case by at least
5% without degrading the known positive MACE case. A pool no larger than the
resident capacity must select `none` without a timing experiment.

**Result:** the threshold policy does not pass. It reduces triggered refill
insertions to 15 for both workloads, but finished systems still force active
compaction and resident-state reconstruction. AtomBit threshold is 2.0% slower
than immediate; MACE is only 0.8% faster. Immediate remains the default and the
conditional mixed matrix was not run. Another threshold attempt requires
topology-preserving or in-place compaction before chunking can remove the
dominant repeated work.

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

**Result:** memory prediction and safety pass, but automatic execution does not
pass the throughput gate. A 32 GiB mixed-workload plan selects B128 for 46-atom
systems and B69/B79 for 276-atom AtomBit/MACE systems. All runs converge and
stay within budget. Speedups versus fixed B64 are 4.82% and 3.99%, so the
conditional endpoint matrix is stopped. Keep `BatchPlanner` as an explicit
OOM-prevention and inspection interface rather than applying it automatically.

### Stage 4: multi-GPU sharding

**Hypothesis:** independent memory-aware GPU workers scale aggregate completed
systems per second without changing per-job results.

Only the winning single-GPU policies advance to this stage. Measure 1, 4, and 7
GPU workers on one homogeneous and one mixed 256-job workload for each MLIP.
Report aggregate throughput, per-GPU occupancy and memory, load imbalance,
scheduler overhead, and parallel efficiency. Do not split one structure or one
BFGS history across GPUs.

Also compare 1 versus 7 workers for the small 32-job mixed pool. This is a
deliberate negative-control case: the planner should use fewer GPUs when
dispatch and model-replication overhead would dominate. Correctness is checked
by original job identifier, independent of completion order or owning GPU.

### MACE tensor-state cache result

The MACE adapter can now consume the generic cached topology without exposing
`AtomicData` to the optimizer or scheduler. MACE-OFF-Small B64 immediate-refill
BFGS improves by 3.3% at 46 atoms and 9.7% at 276 atoms versus native graph
rebuilding. Peak allocated memory changes by less than 0.2%.

Cached B1/B2 predictions, B1 BFGS, and short NVE drift pass against the native
path. All 256 production structures converge. The 276-atom workload has 83
step-count differences and local-minimum outliers because canonical cached-edge
ordering changes floating-point reductions relative to MACE's matscipy order.
For that reason cached mode is explicit rather than the compatibility default.
The multi-GPU stage should measure both the default rebuild path and the cached
276-atom operating point.

### MPS comparison and dense-BFGS result

The matched 256-structure MPS32 comparison exposed a size-specific optimizer
bottleneck. For H184 variable-cell BFGS, runtime profiling assigned 36.60 s of
a 75.55 s B64 run to Hessian update and linear algebra, including 34.79 s in
`torch.linalg.eigh`. Neighbor updates cost 6.85 s and cell-filter displacement
cost 0.86 s. Parallel CPU eigensolves were rejected after an end-to-end run
increased the 64-system time to 111.72 s.

CUDA `auto` BFGS now uses a batched Cholesky solve for positive-definite
Hessians and falls back to ASE's absolute-eigenvalue expression per failed
system. This preserves the BFGS update and displacement while avoiding a full
eigendecomposition when both expressions are mathematically identical. The
profiled H184/B64 workload had zero fallbacks in 440 optimizer updates and
dropped from 75.55 s to 40.27 s.

Expandable CUDA allocator segments expose the usable resident-memory frontier:
H184 B224 uses 63.15 GiB allocated and 78.30 GiB reserved; B256 is OOM. Active
refill then avoids a separate B32 tail for the 256-job pool. One complete
screening run per point gives:

| Workload | Batched policy | Batch s | MPS32 s | Batch speedup | Peak alloc/reserved |
|---|---|---:|---:|---:|---:|
| H46 BFGS | active B256 | 55.65 | 71.46 | 1.284x | 17.31/43.55 GiB |
| H92 BFGS | active B256 | 89.28 | 126.35 | 1.415x | 37.26/77.80 GiB |
| H184 BFGS | refill B224 | 125.34 | 124.02 | 0.989x | 63.31/78.32 GiB |
| H276 FIRE | active B128 | 89.55 | 90.86 | 1.015x | 47.31/76.90 GiB |

H46 and H92 pass the performance gate. H184 is parity under the below-2%
decision rule, not evidence that either scheduler is faster. The result
supports a workload-aware policy: use tensor batching for model throughput,
Cholesky with eigen fallback for full BFGS, expandable segments near the memory
frontier, and refill only when the pool exceeds resident capacity by a modest
tail. MPS remains a valid fallback for large dense-Hessian jobs.

### Cross-family robustness result

The same comparison now covers four signed 256-job pools selected from the
independent CSP test set: GUFJOG44, XATMOV88, OBEQIX220, and heterogeneous
ROFA-MIX. Both smooth-rms AtomBit and MACE-OFF-Small converge all jobs with
variable-cell BFGS.

With one CPU thread per ASE process or MPS worker, selected tensor policies are
12.79-24.22x faster than serial ASE. Tensor batching is 1.29-1.86x faster than
MPS32 on seven of eight model/workload pairs. MACE OBEQIX220 is parity at
0.99x, inside the 2% inconclusive band. B256 wins for the small, medium, and
heterogeneous pools. Dense OBEQIX220 B128/B192/refill points are within 2%, so
B128 is selected for its lower live memory. Refill helps only when a resident
tail would otherwise require another drain cycle.

Short deterministic ASE/eigen/Cholesky trajectory gates pass, but full
GUFJOG44 and OBEQIX220 relaxations can reach different local minima after tiny
numerical perturbations. The effect also appears in tensor B1 and, less often,
MACE under MPS. The result therefore supports a converged-throughput claim, not
identical minimum identity. Application validation must compare low-energy hit
rates, energy/ranking distributions, duplicate minima, and target observables.

The frozen matrix, failed job IDs, endpoint distributions, memory, artifact
hashes, and B1 diagnostic are in
`experiments/cross-family-robustness/results/optimization_summary.json`.

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

Stages 0-4, topology-preserving compaction, the conditional MACE tensor cache,
and public process-worker integration are complete. Process isolation improves
the measured steady worker phase but spawn startup loses on short queues, so
automatic execution retains threads below the validated amortization boundary.
The next mechanism experiment is the `torch.compile`/CUDA-graph audit on stable
MD and fixed-resident optimization shapes.
