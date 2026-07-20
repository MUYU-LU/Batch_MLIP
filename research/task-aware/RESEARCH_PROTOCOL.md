# Task-Aware Batched Atomistic Simulation
## Research, Benchmarking, and Engineering Protocol

**Version:** 1.0-draft  
**Date:** 2026-07-20  
**Stage:** Controlled-test stage; not yet an application-validation paper  
**Target operations:** static MLIP evaluation, geometry optimization, and molecular dynamics (MD)

---

## 1. Scientific position

The project should not claim that batching, caching, refill, larger resident batches, or more GPUs are universally beneficial. The defensible central thesis is:

> For an independent-structure atomistic workload and a specified hardware environment, a task-aware runtime can select a validated combination of topology caching, batch packing, active-set management, refill, optimizer or integrator, and multi-GPU execution that improves goodput while satisfying memory, numerical-correctness, and reproducibility constraints.

This is a **constrained execution-planning problem**. Performance is optimized only after scientific correctness and memory safety are satisfied.

The project has three levels:

1. **Controlled component studies** isolate cache, refill, memory, batch size, and GPU-count effects.
2. **Planner evaluation** tests whether an automatic policy selects configurations close to an empirical oracle on held-out workloads.
3. **Scientific applications** later demonstrate utility in crystal-structure prediction (CSP), finite-displacement phonons, conformer searches, adsorption searches, multi-temperature MD, melting workflows, and equation-of-state calculations.

At the current stage, only Levels 1 and the engineering preparation for Level 2 should be claimed.

---

## 2. Core definitions

### 2.1 Job

A **job** is one independent atomistic calculation with immutable identity. Examples are one geometry relaxation, one displaced phonon supercell, or one MD replica.

Each job \(j\) has descriptors

\[
x_j = (N_j, E_j, P_j, q_j, \tau_j, a_j, S_j, \chi_j),
\]

where:

- \(N_j\): number of atoms;
- \(E_j\): active or candidate edge count;
- \(P_j\): periodicity and cell information;
- \(q_j\): additional cell degrees of freedom, usually 0 for fixed-cell and 9 for a full variable-cell representation;
- \(\tau_j\): task type;
- \(a_j\): arrival time or release event;
- \(S_j\): number of force evaluations or optimization/MD steps;
- \(\chi_j\): topology-stability and other model-specific features.

For an unfinished optimization, \(S_j\) is unknown and is treated as a random variable estimated from pilot steps and historical telemetry.

### 2.2 Pool

A **pool** is the set of jobs to be executed. A pool may be:

- **closed:** all jobs are available at time zero;
- **streaming/open:** jobs arrive over time;
- **finite:** the total number of jobs is known;
- **persistent:** jobs continue to arrive during a long-running service.

### 2.3 Resident batch

A **resident batch** is the set of jobs currently represented in one GPU worker's tensors. Resident capacity should be measured by predicted memory and compute cost, not only by graph count.

### 2.4 Micro-pool

A **micro-pool** is an immutable group of jobs that is assigned as one scheduling unit to a persistent GPU worker. Its internal order, random seeds, batching, and refill schedule are fixed. Dynamic scheduling may move whole micro-pools between GPUs without changing their internal numerical path.

### 2.5 Candidate and active graphs

- The **candidate graph** is constructed at \(r_\mathrm{cut}+r_\mathrm{skin}\).
- The **active graph** contains only edges satisfying the physical model cutoff at the current geometry.

Inactive skin edges must not alter model normalization. For a model whose degree normalization counts all edges in `edge_index`, inactive candidate edges must be removed before the forward pass or masked in both message construction and degree calculation.

### 2.6 Cache

A topology cache stores candidate edges, shifts, reference positions, reference cell, and rebuild metadata. Distances and active-cutoff decisions are still updated every force evaluation.

### 2.7 Refill policies

- **Drain:** no new jobs are admitted until the current resident batch finishes.
- **Immediate refill:** a free slot is filled as soon as a compatible pending job is available.
- **Threshold refill:** physical compaction/refill occurs only after free capacity exceeds a threshold.
- **Cohort refill:** several vacancies are filled together at a scheduled refill window.
- **Fixed-slot refill:** a completed job is replaced in preallocated tensors without changing model input shapes.

Logical completion and physical compaction are distinct events.

### 2.8 Compaction

- **Ordinary compaction:** remove completed jobs and reconstruct the resident graph.
- **Topology-preserving compaction:** retain surviving jobs' cached graph state and only update indices/offsets needed to admit replacements.

### 2.9 Goodput

Throughput counts work attempted. **Goodput** counts scientifically accepted work:

\[
G = \frac{\sum_j w_j I_j^{\mathrm{accepted}}}{T_{\mathrm{wall}}},
\]

where \(w_j\) may be one job, one atom-step, or one edge-step and \(I_j^{\mathrm{accepted}}\) indicates that correctness and convergence criteria were met.

---

## 3. The coupled control problem

The original three balances are coupled:

1. **Topology amortization:** skin, cache, filtering, and rebuild policy.
2. **Temporal occupancy:** compaction and refill policy for jobs that finish at different times.
3. **Spatial capacity:** resident batch packing and GPU-memory target.
4. **Resource parallelism:** number of persistent GPU workers and micro-pool assignment.

They cannot be optimized independently:

\[
r_\mathrm{skin}
\rightarrow E_\mathrm{candidate}
\rightarrow M_\mathrm{job}
\rightarrow B_\mathrm{resident}
\rightarrow \text{refill frequency}
\rightarrow \text{worker load balance}.
\]

For geometry optimization, the optimizer adds another coupled cost. A dense BFGS state for system \(j\) scales approximately as

\[
M^{\mathrm{BFGS}}_j \propto d_j^2,
\qquad d_j = 3N_j + q_j.
\]

Therefore, maximizing graph count or allocated GPU memory is not equivalent to maximizing useful work per second.

The formal planning problem is

\[
P^* = \arg\min_{P \in \mathcal{P}} T(W,P,R)
\]

subject to

\[
M_{\mathrm{peak}}(W,P,R) \le \rho M_{\mathrm{available}},
\]

\[
\epsilon_E \le \epsilon_E^{\max},\quad
\epsilon_F \le \epsilon_F^{\max},\quad
\epsilon_\sigma \le \epsilon_\sigma^{\max},
\]

and a declared equivalence/reproducibility tier. Here \(W\) is the workload, \(R\) the hardware/software environment, and \(P\) the execution plan.

The optimization is lexicographic:

1. pass numerical and scientific correctness gates;
2. avoid OOM and runtime failure;
3. satisfy reproducibility and latency requirements;
4. maximize goodput or minimize makespan.

---

## 4. Task taxonomy

Every workload is first assigned to one execution family.

| Family | Defining property | Examples | Default execution pattern |
|---|---|---|---|
| Static one-shot | One energy/force/stress call per sample | finite-displacement phonons, energy screening, EOS single points | large fixed microbatches; no refill |
| Fixed-horizon persistent | Same requested step count for all jobs | ensemble NVE/NVT MD | persistent resident batch; cache; no refill |
| Variable-horizon closed | Jobs terminate after different numbers of steps | CSP, conformer relaxation, adsorption-state relaxation | active mask/compaction plus refill |
| Dynamic-arrival service | Jobs arrive while execution continues | MC/adaptive adsorption, active learning | persistent workers, queue, staged screening/relaxation |
| Small-pool latency | Few jobs; startup dominates | one structure, small EOS scan | ASE or one GPU; no multi-GPU |

A planner must not apply refill to a one-shot phonon workload or enlarge a batch merely because memory remains available.

---

## 5. Research questions and falsifiable hypotheses

### RQ1 — Topology caching

**Question:** Under which combinations of system size, edge density, skin, cell motion, and reuse interval does caching reduce wall time without changing forces?

**H1:** Cache benefit grows with neighbor-list construction cost and mean reuse interval, but decreases with candidate-edge inflation and edge-dependent model cost.

A cache is beneficial only if approximately

\[
(R-1)T_\mathrm{build}
>
R\,\Delta T_\mathrm{model}(E_\mathrm{candidate})
+T_\mathrm{filter}+T_\mathrm{validation},
\]

where \(R\) is the mean number of model evaluations between rebuilds.

### RQ2 — Refill

**Question:** When does replacing converged jobs improve makespan rather than introduce repacking and reconstruction overhead?

**H2:** Refill is beneficial only when completion-time heterogeneity creates substantial inactive compute and survivor topology can be preserved.

A practical break-even condition is

\[
H(1-u)T_\mathrm{step}
>
T_\mathrm{compact}+T_\mathrm{admit}+T_\mathrm{compile},
\]

where \(u\) is active cost fraction and \(H\) the expected steps until the next natural scheduling event.

### RQ3 — Batch size and memory

**Question:** Is the largest memory-fitting resident batch the throughput optimum?

**H3:** Goodput is non-monotonic in batch size because model saturation, optimizer-state scaling, graph management, and cache-edge inflation can offset higher occupancy.

### RQ4 — Multi-GPU execution

**Question:** When do additional GPUs overcome model startup, underfilled pools, and load imbalance?

**H4:** Persistent workers and cost-balanced micro-pools improve large-pool scaling, while cold multi-GPU execution has weak or negative benefit for small pools.

### RQ5 — Task-aware planning

**Question:** Can a planner select near-optimal execution plans on workloads and models not used for calibration?

**H5:** A planner using atoms, edges, optimizer-state dimension, topology reuse, pilot step time, and completion-step variance achieves median runtime regret below 10% relative to an empirical oracle on held-out workloads.

### RQ6 — Reproducibility

**Question:** Can performance improvements preserve declared numerical or scientific equivalence?

**H6:** Invariant micro-pools preserve each job's internal numerical schedule across GPU counts, reducing schedule-induced changes in BFGS trajectories and final minima.

---

## 6. Status of current evidence

All values below are **preliminary engineering observations** from the recorded experiments. They are not publication-grade until rerun with frozen manifests, the final unified implementation, independent repetitions, and uncertainty estimates.

### 6.1 Cache

For AtomBit on 256-job workloads, the recorded cache improvements were:

| Atoms/system | Resident batch | Time reduction |
|---:|---:|---:|
| 46 | 64 | 5.4% |
| 276 | 64 | 14.1% |
| 46 | 128 | 3.1% |
| 276 | 128 | 6.3% |

The preliminary selection was B64 with a 0.5 Å skin. B128 increased memory for large structures without improving throughput.

For MACE, the recorded time reductions were 3.3% for 46 atoms and 9.7% for 276 atoms. Cached MACE should remain opt-in until canonical edge ordering and endpoint-equivalence behavior are validated, because changed numerical ordering can alter a BFGS trajectory.

### 6.2 Refill and topology-preserving compaction

Recorded makespans:

| Model/workload | Drain | Immediate | Threshold |
|---|---:|---:|---:|
| AtomBit, 276 atoms | 341.75 s | 334.49 s | 341.16 s |
| MACE, 46 atoms | 145.44 s | 130.40 s | 129.38 s |

The MACE threshold advantage over immediate refill was only 0.8%, below the chosen practical-effect threshold. More importantly, grouping refill did not remove the dominant graph-reconstruction cost.

Topology-preserving compaction reduced the AtomBit B64 time from 334.49 s to 280.28 s, corresponding to a recorded 19.3% throughput improvement. This supports the mechanism that survivor preservation, not refill frequency alone, is the main source of benefit.

### 6.3 Memory planning and bucketing

A memory model using atom count, candidate-edge count, variable-cell degrees of freedom, and dense BFGS state had recorded held-out B128 errors of 1.30% for AtomBit and 0.36% for MACE.

For a mixed 128×46 plus 128×276 workload, planned buckets improved time by 4.82% for AtomBit and 3.99% for MACE. The planner should therefore be retained immediately for OOM prevention and diagnostics, while automatic performance bucketing remains an experimental policy until a stronger benefit is shown.

### 6.4 Multi-GPU scaling

Recorded 256-job results:

| Model/workload | 1 GPU | 4 GPUs | 7 GPUs | 7-GPU speedup | 7-GPU efficiency |
|---|---:|---:|---:|---:|---:|
| AtomBit, 276 atoms | 282.32 s | 83.56 s | 54.20 s | 5.21× | 74.4% |
| AtomBit, mixed | 199.46 s | 60.52 s | 41.44 s | 4.81× | 68.7% |
| MACE, 276 atoms | 291.77 s | 101.27 s | 53.56 s | 5.45× | 77.9% |
| MACE, mixed | 209.07 s | 69.76 s | 45.32 s | 4.61× | 65.9% |

For only 32 mixed cold-start jobs, recorded 7-GPU speedups were 1.06× for AtomBit and 1.34× for MACE. These observations support a pool-size-dependent policy but do not yet define a universal GPU-count threshold.

### 6.5 What remains unimplemented or incompletely validated

- automatic joint selection of skin, cache, batch size, refill, bucketing, and GPU count;
- persistent invariant micro-pool execution;
- pilot-based prediction of remaining optimization cost;
- batched L-BFGS;
- exact restart for every optimizer/integrator and scheduler state;
- production validation of NPT/NPH and barostat/stress behavior;
- application-level CSP, phonon, conformer, adsorption, melting, and bulk-modulus benchmarks.

---

## 7. Controlled workload suite

### 7.1 Naming convention

Canonical IDs encode operation, size distribution, uniqueness, pool size, and version.

- `R`: replicated-control workload; repeated structures are technical replicates.
- `U`: unique-structure workload.
- `H`: homogeneous size.
- `MIX`: heterogeneous or bimodal size.
- `CLOSED`: all jobs available at time zero.
- `STREAM`: deterministic or calibrated arrivals.

Existing shorthand IDs may be retained as aliases, but publications should use versioned canonical IDs.

### 7.2 Optimization workloads

| Canonical ID | Existing alias | Definition | Cell | Primary purpose |
|---|---|---|---|---|
| OPT-H46-R256-v1 | OPT-H46-256 | 32 unique 46-atom structures, each repeated 8× | variable | small homogeneous cache/batch study |
| OPT-H276-R256-v1 | OPT-H276-256 | 32 unique 276-atom structures, each repeated 8× | variable | large homogeneous cache/batch study |
| OPT-MIX-R256-v1 | OPT-MIX-256 | 128 46-atom plus 128 276-atom jobs | variable | size/edge heterogeneity and packing |
| OPT-STEPVAR-R256-v1 | OPT-STEPVAR-256 | 128 easy plus 128 hard 276-atom jobs | variable | completion heterogeneity and refill |
| OPT-H276-FIXED-R256-v1 | OPT-FIXED-256 | same starts as OPT-H276-R256-v1 | fixed | isolate cell and variable-cell-cache cost |
| OPT-MIX-R32-v1 | OPT-SMALL-32 | 16 46-atom plus 16 276-atom jobs | variable | cold-start/small-pool control |
| OPT-MIX-STREAM256-v1 | OPT-STREAM-256 | same jobs as mixed workload released in calibrated waves | variable | persistent workers and online admission |
| OPT-H46-U256-v1 | new | 256 unique 46-atom starts | variable | publication generalization workload |
| OPT-H276-U256-v1 | new | 256 unique 276-atom starts | variable | publication generalization workload |

For `OPT-STEPVAR`, easy and hard strata are defined once using reference single-system B1 BFGS step counts:

- easy: lowest quartile;
- hard: highest quartile.

The assignments are frozen in a manifest and never recomputed using the candidate scheduler.

Repeated structures are not 256 independent scientific samples. Statistical analysis must cluster by unique structure; repeats are technical replicates.

### 7.3 Fixed-work kernel controls

Two controls separate implementation speed from convergence behavior:

1. **EVAL-REPLAY50:** evaluate 50 frozen reference frames per structure. This gives exact model/graph work independent of optimizer divergence.
2. **OPT-FIXED50:** run exactly 50 optimizer steps with convergence stopping disabled. This measures runtime execution but may still follow schedule-dependent trajectories.

The two controls answer different questions and must not be combined.

### 7.4 Static finite-displacement workloads

| ID | Definition | Samples | Purpose |
|---|---|---:|---|
| FD-H46-FULL276-v1 | all ± Cartesian 0.01 Å single-atom displacements for a 46-atom periodic cell | 276 | homogeneous one-shot batching |
| FD-H276-FULL1656-v1 | all ± Cartesian 0.01 Å single-atom displacements for a 276-atom cell | 1656 | large static pool and multi-GPU streaming |
| FD-H276-RAND1024-v1 | 1024 frozen random all-atom displacements with 0.01 Å RMS amplitude | 1024 | regression-style IFC dataset and static throughput |
| FD-SYM-v1 | symmetry-reduced displaced supercells generated by a frozen Phonopy configuration | dataset-specific | application-like validation after static engine is trusted |

Finite-displacement jobs receive one force evaluation. Refill is disabled. The candidate plan is a fixed-shape microbatch stream with shared or validated cached topology.

### 7.5 MD workloads

| ID | Definition | Purpose |
|---|---|---|
| MD-NVE-H64-v1 | 64 replicas of one fixed-cell crystal, independent frozen velocity seeds | integrator parity and energy drift |
| MD-NVT-T64-v1 | 8 temperatures × 8 seeds, fixed cell | per-system thermostat parameters and canonical statistics |
| MD-MIX64-v1 | 32 46-atom plus 32 276-atom systems, equal step count | heterogeneous persistent-batch efficiency |
| MD-CACHE64-v1 | 32 low-temperature plus 32 high-temperature replicas of the same crystal | cache rebuild behavior versus atomic motion |
| MD-MGPU256-v1 | 256 persistent replicas on 1, 4, and 7 GPUs | multi-GPU persistent scaling |

Default controlled settings, subject to physical-stability checks:

- timestep: 0.5 fs;
- warm-up: 100 steps;
- performance interval: 10,000 steps;
- long NVE validation: at least 100,000 steps;
- NVE initial temperature: 300 K;
- NVT temperatures: 100, 150, 200, 250, 300, 350, 400, and 450 K, eight independent seeds each;
- Langevin friction: 0.01 fs⁻¹ unless the reference integrator uses a different declared convention.

NPT/NPH are excluded until stress, barostat equations, invariant measure, and restart behavior are validated.

### 7.6 Small equation-of-state control

`EOS-V21-v1` contains 21 isotropically strained configurations with linear volume factors spanning a frozen range, for example 0.94–1.06. It is used as a small-pool latency and stress/energy correctness case, not a principal throughput benchmark.

---

## 8. Frozen workload manifest

Every workload version must have a machine-readable manifest containing:

- workload ID and version;
- exact structure file and frame index;
- SHA-256 of each source file and normalized structure representation;
- immutable `system_id`, `group_id`, and `duplicate_group`;
- atom count and species composition;
- exact active-edge count at the physical cutoff;
- candidate-edge count for each tested skin;
- periodicity, cell, volume, and constraints;
- reference B1 step count and convergence status;
- easy/hard stratum, if applicable;
- arrival time/wave;
- random seed;
- expected output order.

A new structure selection, displacement seed, or difficulty classification creates a new workload version.

---

## 9. Common optimization protocol

Unless an experiment explicitly varies one of these factors:

```text
optimizer              ASE-compatible BFGS
cell filter            FrechetCellFilter for variable-cell workloads
force tolerance        0.05 eV/Å
maximum steps          500
maximum step           0.2 Å
initial alpha           70
optimizer state dtype  float64
AtomBit model dtype    float32
MACE model dtype       float64
determinism            enabled where supported
```

Any deviation must be encoded in the run configuration and not introduced implicitly by the planner.

BFGS, L-BFGS, FIRE, and FIRE2 are distinct algorithms. Classic ASE FIRE must preserve the reference update order, including use of the current alpha for velocity mixing and reduction of alpha only for the next step.

---

## 10. Correctness and equivalence hierarchy

A speedup is reportable only with an explicit equivalence tier.

### Tier K0 — Kernel isolation

- no cross-system edges;
- energy, force, and stress shape correctness;
- B1 batch result matches single-system reference;
- active skin-edge filtering produces exact-cutoff parity.

### Tier K1 — Step/trajectory parity

For deterministic algorithms and fixed schedules:

- per-step energy, force, positions, cell, velocity, and optimizer state match within declared tolerances;
- classic FIRE ordering and BFGS state updates are regression-tested;
- restart from a checkpoint reproduces the uninterrupted trajectory.

### Tier K2 — Endpoint equivalence

For schedule-sensitive optimization:

- both runs converge;
- final force and stress tolerances are satisfied;
- final energy per atom, volume, lattice metric, and symmetry-aware structure distance are compared;
- different local minima are identified rather than averaged away.

### Tier K3 — Distributional MD equivalence

- NVE drift, temperature, momentum, and structural observables agree statistically;
- NVT kinetic-energy distribution and target-temperature behavior are validated;
- random seeds and RNG state are checkpointed.

### Tier K4 — Application equivalence

Examples:

- CSP candidate ranking and recovered polymorphs;
- phonon frequencies, acoustic-sum-rule residual, DOS, and thermodynamic observables;
- bulk modulus and fitted equation of state;
- adsorption-state ranking;
- melting observable or coexistence outcome.

If two BFGS schedules reach different minima, their runtimes are not an exact-work comparison. They may still be compared at K2/K4 as solver goodput, with basin outcomes reported.

---

## 11. Performance metrics

### 11.1 Timing scopes

Three scopes must be reported separately:

1. **Kernel:** model forward/backward only.
2. **Warm steady state:** includes neighbor management, packing, optimization/integration, and transfers, but excludes process/model startup and first compilation.
3. **Cold end-to-end:** includes worker startup, model load, compilation, input, and output.

### 11.2 Primary metrics

- makespan, seconds;
- converged jobs/second;
- useful atom-steps/second;
- useful active-edge-steps/second;
- time to first accepted result;
- GPU-seconds per accepted job;
- peak allocated and reserved GPU memory;
- OOM/failure rate.

Useful edge-step goodput is

\[
G_E = \frac{\sum_t E_{\mathrm{active}}(t)}{T_{\mathrm{wall}}}.
\]

### 11.3 Cache metrics

- candidate/active edge inflation \(E_\mathrm{candidate}/E_\mathrm{active}\);
- cache-hit rate;
- rebuild count and mean rebuild interval;
- neighbor-build, filter, and model time;
- maximum force/energy deviation relative to exact rebuild;
- count of missed active edges, which must be zero.

For variable cells, a conservative invalidation bound should combine non-affine atomic displacement and cell deformation:

\[
b_g = 2\delta_g^{\mathrm{nonaffine}} + \beta_g^{\mathrm{cell}}.
\]

Rebuild when \(b_g \ge r_\mathrm{skin}\). The implementation-specific \(\beta_g^{\mathrm{cell}}\) must be documented and validated against exact neighbor reconstruction.

### 11.4 Refill metrics

- active-cost fraction versus time;
- fraction of model work spent on inactive/completed jobs;
- refill event count;
- insertion-size distribution;
- compaction, admission, and graph-update time;
- mean queue delay;
- model-call shape/compile variants;
- survivor-cache retention rate.

### 11.5 Packing and memory metrics

- total atoms, active edges, candidate edges, and BFGS dimensions per resident batch;
- predicted versus measured peak memory;
- fill ratio relative to safety budget;
- goodput versus batch size;
- number of OOM backoffs;
- tail-batch overhead.

### 11.6 Multi-GPU metrics

\[
S_p = \frac{T_1}{T_p}, \qquad
\eta_p = \frac{S_p}{p}.
\]

Also report:

- load imbalance \(\max_g T_g / \operatorname{mean}_g T_g\);
- worker startup and idle time;
- micro-pools per GPU;
- GPU-seconds/job;
- cold and persistent-worker results separately.

---

## 12. Statistical protocol

### 12.1 Repetitions

- At least five independent process-level timing repetitions for long runs.
- At least ten repetitions for short or highly variable timings.
- Warm-up runs are discarded and not counted as repetitions.
- GPU synchronization is required at timed boundaries.

### 12.2 Paired design

Configurations are compared on the same frozen jobs and seeds. Run order is randomized or counterbalanced. Structure identity is a blocking factor.

Repeated copies of one structure are technical replicates. Confidence intervals must cluster or bootstrap by unique structure, not treat all 256 copies as independent scientific observations.

### 12.3 Reporting

Report:

- median and mean wall time;
- bootstrap 95% confidence interval;
- coefficient of variation;
- paired speedup/effect distribution;
- failure and convergence counts.

The current 5% selection rule is an **engineering practical-significance threshold**, not a statistical-significance test. A performance feature becomes a default only when:

1. median paired improvement is at least 5%;
2. the 95% confidence interval excludes no improvement, or evidence is otherwise compelling and replicated;
3. correctness and memory gates pass.

Safety features such as an OOM-prevention memory planner may be retained even when the speed effect is below 5%.

### 12.4 Avoiding planner leakage

- Fit memory/time models and policy thresholds on calibration workloads.
- Freeze the planner.
- Evaluate on held-out structures, pool sizes, and preferably a held-out MLIP.
- Compare against an empirical oracle obtained by bounded exhaustive search over feasible plans.

Planner regret is

\[
\mathcal{R} = \frac{T_\mathrm{planner}-T_\mathrm{oracle}}{T_\mathrm{oracle}}.
\]

Primary planner targets:

- zero OOMs on held-out workloads;
- median regret below 10%;
- 90th-percentile regret below 20%;
- all required equivalence gates passed.

---

## 13. Experiment blocks

### E0 — Correctness gates

Compare batch B1, static batch, cache, compaction/refill, and multi-GPU results against the single-system reference. No performance claim proceeds if E0 fails.

### E1 — Cache/skin factorial study

Workloads:

- OPT-H46-R256-v1;
- OPT-H276-R256-v1;
- OPT-H276-FIXED-R256-v1;
- MD-CACHE64-v1;
- FD-H46-FULL276-v1.

Factors:

- skin: 0, 0.25, 0.5, 1.0 Å;
- batch: 16, 32, 64, 128 where feasible;
- topology policy: exact rebuild, cached candidate plus filtering;
- cell: fixed versus variable.

Primary analysis: interaction of skin × size × batch × task. The output is a cache decision model, not a universal skin.

### E2 — Refill study

Primary workload: OPT-STEPVAR-R256-v1.

Policies:

- drain;
- immediate refill with full reconstruction;
- threshold refill with full reconstruction;
- immediate refill with topology-preserving compaction;
- cohort refill with topology-preserving compaction;
- fixed-slot refill for shape-compatible homogeneous jobs.

Report occupancy, inactive work, compaction cost, shape changes, and makespan. Refill conclusions drawn from homogeneous-step workloads are invalid.

### E3 — Batch-size and memory study

Workloads:

- homogeneous small;
- homogeneous large;
- mixed/bimodal;
- static finite displacement;
- MD persistent replicas.

Candidate resident capacities are selected by measured memory and include B16, B32, B64, and B128 when feasible. The best batch is the one maximizing goodput, not memory use.

### E4 — Optimizer-state study

Compare ASE-compatible BFGS, L-BFGS, and FIRE on large structures.

Separate:

- fixed 50-step cost;
- convergence-to-tolerance performance;
- memory and eigensolve/update costs;
- endpoint basins.

### E5 — Multi-GPU scaling

#### E5a. Invariant-work strong scaling

- eight fixed B32 micro-pools for 256 jobs;
- identical micro-pool contents and internal schedules on 1, 4, and 7 GPUs;
- dynamic assignment of whole micro-pools only.

This is the scientifically clean speedup result.

#### E5b. Best-configuration resource scaling

The planner may change batch size, bucketing, refill, and GPU count. This is reported as best-achieved throughput, not strong scaling.

#### E5c. Weak scaling

Hold jobs or predicted work per GPU approximately constant and increase total pool size with GPU count. This tests persistent-worker service capacity.

### E6 — Planner evaluation

Baselines:

1. ASE sequential B1;
2. static B64, no cache/refill;
3. largest memory-fitting batch;
4. hand-tuned current default;
5. simple atom-count bucketing;
6. task-aware planner;
7. empirical oracle.

Evaluate decision quality, runtime regret, OOMs, and correctness on held-out workloads.

---

## 14. Planner logic

### 14.1 Task profiling

The profiler records:

- pool size and arrival mode;
- distribution of atoms and active/candidate edges;
- edge density and its variation;
- fixed/variable cell and constraints;
- optimizer/integrator state dimension;
- model dtype and force/stress mode;
- topology-cache support;
- pilot model, neighbor, and optimizer times;
- pilot completion-step variance;
- GPU memory, count, and worker startup cost;
- required equivalence tier.

### 14.2 Execution-family selection

```text
one-shot evaluation          -> static_full or static_stream
fixed-horizon MD             -> persistent_ensemble
variable-horizon closed pool -> active_refill
open/dynamic arrival         -> service_queue
few jobs                     -> single_gpu_latency
```

### 14.3 Cache decision

Enable cache only if:

- exact active-edge semantics are preserved;
- measured or predicted rebuild amortization is positive;
- candidate-edge inflation does not reduce the selected resident capacity enough to erase benefit;
- variable-cell invalidation is validated.

### 14.4 Refill decision

Disable refill for one-shot tasks and synchronized fixed-length MD.

For variable-horizon jobs, use refill when:

- pending jobs remain;
- completion-step variation is material;
- predicted payback exceeds compaction/admission cost;
- survivor topology can be preserved or fixed slots are shape-compatible.

Immediate one-by-one refill is allowed only for low-cost fixed-slot replacement. Otherwise use refill windows with hysteresis.

Suggested initial hysteresis:

```text
check interval              10 optimizer steps
refill below active cost    70%
fill target                 90%
minimum released cost       15%
```

These are calibration defaults, not scientific constants.

### 14.5 Batch selection

Fit workload/model-specific memory and time models:

\[
M_\mathrm{peak}
\approx M_0 + aN_\Sigma + bE_\Sigma + c\sum_j d_j^2,
\]

\[
T_\mathrm{step}
\approx t_0 + \alpha N_\Sigma + \beta E_\Sigma + \gamma T_\mathrm{optimizer}.
\]

Enumerate feasible candidate packs under a safety fraction \(\rho\), profile the Pareto frontier, and select the plan maximizing measured or predicted goodput.

### 14.6 Multi-GPU decision

Use persistent multi-GPU workers when the predicted service work exceeds startup and underfill costs. Jobs are grouped into invariant micro-pools and complete micro-pools are assigned dynamically to idle workers.

A micro-pool cost estimate should include

\[
C_m = \sum_{j\in m} \hat S_j\,
\hat t_\mathrm{step}(N_j,E_j,\tau_j)
+T_\mathrm{cache/refill},
\]

where \(\hat S_j\) is updated from pilot steps and completed jobs.

### 14.7 Optimizer decision

- BFGS: smaller systems where curvature information is useful and dense state fits comfortably.
- L-BFGS: large structures or memory-constrained resident batches.
- FIRE: robust screening, difficult starts, or when low optimizer-state memory is more valuable than rapid local convergence.

The planner may recommend but must not silently change the requested scientific algorithm unless the user allows optimizer selection.

---

## 15. Engineering architecture

### TaskProfiler

Computes immutable workload features and pilot estimates.

### TopologyManager

Owns candidate graphs, active masks/filtering, skin validity, cell deformation bounds, and rebuild telemetry.

### CostModel

Predicts peak memory, model time, neighbor time, optimizer time, and remaining job cost. It is updated online but versioned for reproducibility.

### BatchPacker

Packs jobs under atom, edge, optimizer-state, and memory constraints. It supports fixed slots, homogeneous buckets, and heterogeneous best-fit packs.

### ActiveSetManager

Maintains active/completed masks, convergence, logical completion, survivor state, and compaction.

### RefillController

Applies drain, immediate, threshold, cohort, or fixed-slot refill policies with explicit payback logic.

### PersistentWorker

Loads one MLIP once per GPU and executes micro-pools. It records timing scopes, peak memory, compile variants, and failures.

### GlobalScheduler

Assigns whole micro-pools to workers, uses telemetry to update cost estimates, and preserves original result order.

### ResultValidator

Applies K0–K4 correctness/equivalence gates and marks outputs accepted or rejected for goodput accounting.

---

## 16. Application mapping after the controlled stage

| Application | Task family | Expected plan |
|---|---|---|
| Molecular-crystal CSP | variable-horizon, variable-cell, large pool | size/edge bucketing, cache, topology-preserving refill, persistent multi-GPU micro-pools |
| Finite-displacement phonons | static one-shot, highly homogeneous | large fixed microbatch stream, shared/validated topology, no refill |
| Molecular conformer search | variable-horizon, nonperiodic | fixed-cell active optimization, refill, size buckets, likely L-BFGS/FIRE options |
| MC adsorption-state search | dynamic-arrival pipeline | batched energy screening, selected relaxation, persistent queue workers |
| Multi-temperature MD | fixed-horizon persistent | resident replica batches, per-system thermostat parameters, cache, multi-GPU sharding |
| Melting workflow | long MD and possibly NPT/coexistence | only after NPT/stress/barostat validation |
| Bulk modulus/EOS | small static or fixed-volume optimization pool | one GPU, batched strain points, stress/energy validation |

The first three full applications should be CSP, finite-displacement phonons, and multi-temperature MD because they deliberately span three different execution families.

---

## 17. Paper claim policy

### Safe preliminary statements

- A correct batch representation and optimizer/MD engine have been implemented and tested against single-system references.
- Preliminary experiments show that cache, refill, batch size, and GPU count interact strongly with task and system size.
- Topology-preserving compaction appears to be a major mechanism behind refill benefit.
- Large cold batches and many GPUs are not universally faster.

### Claims that require final reruns

- numerical values for speedup or cache benefit;
- B64 or skin=0.5 Å as a general default;
- superiority of immediate over threshold refill;
- near-linear multi-GPU scaling;
- automatic planner optimality.

### Claims that are currently out of scope

- production NPT/NPH accuracy;
- melting-point prediction;
- full CSP, phonon, adsorption, conformer, or bulk-modulus application speedups;
- universal superiority over other batch runtimes.

---

## 18. Proposed paper structure

1. **Introduction:** independent-structure MLIP workloads and the absence of a universal batching policy.
2. **Method:** batch state, optimizer/integrator, topology cache, active set/refill, packing, persistent micro-pools, planner.
3. **Correctness:** K0–K3 validation against ASE/reference implementations.
4. **Controlled workloads:** frozen optimization, static displacement, and MD suites.
5. **Ablations:** cache, refill, batch/memory, optimizer state, and multi-GPU.
6. **Planner:** held-out oracle-regret evaluation.
7. **Scientific applications:** later CSP, phonons, and multi-temperature MD.
8. **Generality:** AtomBit and MACE, ideally a third MLIP/backend.
9. **Limitations:** local-minimum sensitivity, edge ordering, cold start, BFGS scaling, NPT status.

The strongest eventual contribution is not a single fast optimizer; it is a validated planner that chooses different execution strategies for different atomistic task classes.

---

## 19. Immediate implementation sequence

1. Freeze manifests for the existing controlled workloads.
2. Add complete telemetry and three-scope timing.
3. Retain classic FIRE/BFGS parity tests as release gates.
4. Implement persistent GPU workers.
5. Implement invariant micro-pools and whole-micro-pool dynamic assignment.
6. Implement topology-preserving compaction for all supported MLIPs.
7. Add pilot-based remaining-step and micro-pool cost estimation.
8. Add batched L-BFGS.
9. Rerun E0–E5 with repetitions and confidence intervals.
10. Implement and evaluate the task-aware planner against a held-out empirical oracle.
11. Add finite-displacement static batching as the first contrasting non-optimization task.
12. Complete long-time NVE/NVT validation before application MD.

---

## 20. Reproducibility checklist

Every reported result must archive:

- source commit and dirty-state hash;
- workload manifest and SHA-256 files;
- model checkpoint hash;
- Python, PyTorch, CUDA, driver, ASE, MACE, and compiler versions;
- GPU model, count, memory, clocks/power mode where available;
- CPU model, thread count, and affinity;
- all random seeds and deterministic settings;
- exact optimizer/integrator and parameters;
- exact cache, skin, refill, batch, bucket, and micro-pool settings;
- cold/warm timing scope;
- raw per-run telemetry;
- convergence and correctness outputs;
- peak memory and OOM/backoff events;
- final structure/trajectory checksums;
- planner version, calibration data version, and selected-plan explanation.

