# Changelog

## 2026-07-28

- Add an explicit BFGS convergence-check cadence and reject blockwise K5 after
  it increased H46/STEPVAR work by 7,094/7,446 optimizer steps and changed
  endpoint energies by up to 20.17/7.81 meV per atom.
- Correct the cadence benchmark's allocator environment: the production
  dual-variable configuration reduces STEPVAR-H276 refill peak reservation
  from 78.22 to 29.48 GiB with zero allocation retries.
- Validate immediate B64 refill at 67.92/101.55 s versus 78.67/105.70 s for
  active drain on signed H46/STEPVAR-H276 R256 pools. Retain stepwise
  convergence and expose both allocator variable values in benchmark telemetry.

## 2026-07-26

- Add an allocator-aware one-or-more-GPU process launcher. It selects
  expandable segments for measured AtomBit variable-cell BFGS workloads,
  preserves native allocation elsewhere, sets both PyTorch compatibility
  variables before child CUDA initialization, and reports worker telemetry.
- Add spawn-isolated multi-GPU task workers to the plug-and-play relaxation API,
  with persistent per-worker calculators, deterministic first-wave assignment,
  pending-chunk stealing, CPU result offload, and child-lifetime handshakes.
- Add a serializable MACE adapter reconstruction path and explicit
  `auto`/`process`/`thread` worker controls. Auto retains threads for fewer than
  eight chunks per GPU because measured process startup outweighs the steady
  worker speedup on short pools.
- Add plug-and-play `scheduling="auto"` without a user-authored planner or
  pilot, using production-only cold-start capacity ramping and a persistent
  hardware/model/optimizer policy cache.
- Add conservative allocated/reserved-memory growth gates, online refill
  admission, homogeneous multi-GPU calculator cloning, compatible chunking,
  and pending-work stealing through `devices=[...]`.
- Validate warm policies on AtomBit, MACE, T2 H92, and independent XATMOV88;
  retain the rejected 77.44 GiB allocator-frontier result alongside the safe
  B128 policy and multi-GPU scaling artifacts.
- Separate refill behavior from atom count with a signed BFGS factorial over
  H46/H276, B32/B64/B128, AtomBit/MACE, drain/repack/slots, and MPS32.
- Show that BFGS refill is strong at B32, weakens at B64, and is neutral or
  harmful at B128; the best memory-safe B128 tensor modes beat MPS32 in all
  four BFGS workloads.
- Add optimizer-safe fixed- and variable-cell FIRE active refill, preserving
  velocity, time step, mixing, positive-power count, Frechet state, and local
  step count across pending-job admission.
- Validate FIRE refill against active drain and MPS32. Wide-tail H46 gains
  2.44x-2.55x at B32 and 1.24x-1.27x at B128, while narrow-tail H276 rejects
  refill at B128.
- Benchmark BFGSLineSearch B128 against MPS32. Tensor trial batching wins three
  of four cases; MACE H276 remains an MPS regime.
- Replace atom-count-only refill reasoning with a measured policy over
  optimizer, convergence spread, resident batch, graph/model scaling, memory,
  pending-pool size, and MPS parity.

## 2026-07-25

- Add opt-in fixed-slot BFGS refill inspired by TorchSim in-flight admission,
  with partial neighbor invalidation and safe repack fallback.
- Measure fixed-slot refill against active drain, repack refill, and MPS32:
  AtomBit H46 gains 1.21x/1.06x/1.09x, while MACE H276 rejects refill.
- Identify PyTorch allocator fragmentation in long AtomBit refill and reduce
  peak reserved memory from 78.22 to 29.48 GiB with expandable segments.
- Generalize the CUDA MPS ASE pool reference to signed one-shot evaluation and
  NVE workloads in addition to variable-cell optimization.
- Compare the tensor engine directly with 32-worker CUDA MPS on equal H46/H276
  MIX evaluation and NVE pools and model-specific H276 step-variance pools.
- Record 1.30x/1.87x EVAL and 1.28x/1.09x NVE gains for AtomBit/MACE,
  respectively; AtomBit active-drain BFGS gains 1.41x while MACE is at parity.
- Reject active refill as a universal throughput default on STEPVAR: it reduces
  model calls but is 2.9-3.5% slower and increases reserved memory.
- Add a reproducible mechanism-atlas summarizer, normalized results, explicit
  evidence boundaries, and a ranked untested-acceleration backlog.

## 2026-07-18

- Audited BFGS B1 against common ASE and found the original single-run
  final-minimum comparison was invalid under nondeterministic GPU reductions.
- Added deterministic benchmark control and an optional independent BFGS
  optimizer dtype; the production default remains the calculator state dtype.
- Added a mixed-precision variable-cell regression test and retained all
  negative float64 and GPU-initialization artifacts.
- Repeated BFGS scaling with float32 model inference, float64 optimizer state,
  and deterministic CUDA controls. B32 is 4.68x-6.89x faster than common ASE;
  deterministic three-step validation passes across all 128 structures.

## 0.2.0 - Unreleased

- Add opt-in `relax(..., scheduling="auto", planner=...)` execution that selects
  a calibrated whole pool or restores input order across memory-safe queues.
- Add explicit `relax_ase()` native-ASE reference execution rather than
  presenting the sequential batch-calculator adapter as strict ASE semantics.
- Validate automatic scheduling on the signed six-family CROSS-MIX R192
  variable-cell workload with AtomBit and MACE-OFF-Small; recommended whole-pool
  execution is 1.63x and 1.24x faster than 32-worker CUDA MPS references.
- Add opt-in deferred-CUDA runtime phase profiling for graph construction,
  model evaluation, BFGS updates, compaction, refill, and occupancy events.
- Add generic per-structure variable-cell neighbor caching, exact-cutoff GPU
  edge filtering, and canonical neighbor ordering.
- Match ASE's float64 physical-cutoff decision for cached float32 geometries,
  preventing boundary-edge loss without changing model precision.
- Validate exact AtomBit B64/B128 cache scaling on 256-job variable-cell BFGS
  workloads; B64 with a 0.5 A skin is preferred for the 276-atom case.
- Add explicit BFGS `drain`, `immediate`, and low-water `threshold` refill
  policies with profiler-visible insertion decisions.
- Benchmark refill policies on AtomBit and MACE; threshold fails the 5% gate,
  so immediate refill remains the default and selected production policy.
- Add a generic atom/edge/Hessian-aware `BatchPlanner`, non-negative memory
  calibration, heterogeneous bucketing, and original-order queue metadata.
- Validate held-out B128 memory prediction and a 32 GiB mixed workload; memory
  safety passes, while 4.82%/3.99% speedups remain below the automatic-use gate.
- Fix active-refill state aliasing when resident capacity equals pool size.
- Add an opt-in MACE-OFF-Small variable-cell optimization test covering common
  ASE, masked batching, and active batching with both FIRE and BFGS.
- Add MACE-OFF-Small ASE/masked/active variable-cell FIRE scaling on the same
  fixed 32-structure B1-B32 pools used for AtomBit.
- Rename the canonical distribution/package to `batch-mlip`/`batch_mlip` and
  retain `atombit_batch` as a thin compatibility namespace.
- Rename the canonical AtomBit adapter and cell filter to
  `AtomBitBatchCalculator` and `FrechetCellFilter`; preserve the former class
  names as aliases.
- Add `MACEBatchCalculator.from_off()` as the named MACE-OFF constructor while
  preserving `load_mace_off_batch()`.
- Organize implementation modules into readable `core`, `optimization`,
  `dynamics`, `models`, and `interfaces` subpackages while preserving root
  exports and legacy import aliases.
- Add ASE-compatible full batched BFGS for fixed and Frechet variable-cell
  coordinates, including `FixAtoms` and active Hessian compaction.
- Register `BatchedBFGS` as `optimizer="bfgs"` in Python and YAML interfaces.
- Add the runtime-checkable `BatchOptimizer` and `OptimizerFactory` protocols.
- Add `BatchedFIRE` and `BatchedGradientDescent` optimizer objects.
- Add the optimizer registry, `create_optimizer()`, and direct-object dispatch
  through the Python and YAML relaxation interfaces.
- Add capability validation for variable-cell relaxation and active compaction.
- Add a model-independent `BatchCalculator` contract shared by FIRE and MD.
- Add calculator-style `evaluate`, `relax`, and `molecular_dynamics` functions.
- Add a sequential `ASECalculatorAdapter` for compatibility and references.
- Add optional batched Frechet cell degrees of freedom to FIRE relaxation.
- Add active-batch compaction for variable-cell FIRE, including cell state and full-order restoration.
- Validate graph-model stress by finite differences and variable-cell FIRE against ASE.
- Reserve explicit `npt`/`npt_mtk` API ensemble names for a future validated barostat.
- Preserve `BatchedPotential` and the existing low-level API.

## 0.1.0 — initial project packet

- Added heterogeneous graph batching without a runtime PyTorch Geometric dependency.
- Added cached ASE/matscipy neighbour lists with a configurable skin.
- Added autograd and direct-force model adapters, per-graph E0 offsets, and stress evaluation.
- Added batched FIRE and steepest-descent relaxation.
- Added NVE velocity-Verlet and NVT Langevin BAOAB dynamics.
- Added `FixAtoms`, extxyz/JSONL reporters, tensor checkpoints, YAML CLI, validation, tests, benchmarks, and an agent experiment protocol.
- Preserved the uploaded `src.*` namespace for checkpoint compatibility.
