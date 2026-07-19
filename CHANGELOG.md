# Changelog

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

- Add opt-in deferred-CUDA runtime phase profiling for graph construction,
  model evaluation, BFGS updates, compaction, refill, and occupancy events.
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
