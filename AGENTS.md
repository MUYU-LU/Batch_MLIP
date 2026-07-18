# Agent Operating Protocol

This repository is intended for iterative work by coding and research agents. The goal is not merely to make code faster; it is to improve throughput **without breaking graph isolation, force correctness, integration stability, or reproducibility**.

## Prime directive

Never claim an improvement from wall-clock speed alone. A change is acceptable only when it passes the relevant correctness gates and its benchmark is reproducible from a committed experiment specification.

## Immutable provenance

Do not edit files under `original_uploads/`. Compatibility changes belong under `src/`; batching changes belong under `atombit_batch/`.

## Required loop

For each experiment:

1. Create a new experiment directory or copy `experiments/baseline/experiment.yaml`.
2. State one falsifiable hypothesis and the expected mechanism.
3. Record the baseline commit or source packet hash.
4. Run `pytest -q` before changing code.
5. Make the smallest change that tests the hypothesis.
6. Add or tighten a regression test for any changed behavior.
7. Run batch-vs-single validation with the intended model and dataset.
8. Run an NVE drift test for changes affecting forces, neighbours, precision, or integration.
9. Benchmark with warm-up, repeated trials, and synchronized GPU timing.
10. Store raw logs and a machine-readable manifest.
11. Report negative and inconclusive outcomes; do not delete them.

## Non-negotiable invariants

- Every edge connects atoms belonging to the same graph.
- There is exactly one total energy per graph.
- Force rows preserve the concatenated atom order.
- Batch-size-one agrees with the existing single-structure path within declared tolerances.
- Batched and individually evaluated systems agree within declared tolerances.
- Fixed atoms do not move and have zero integration velocity.
- Autograd forces equal `-dE/dr` by construction; direct-force results must be separately validated.
- E0 is added exactly once.
- PBC shifts use the same sign convention as the model.
- Neighbour skins never omit a pair inside the physical cutoff.
- NVE changes include an energy-drift comparison at several time steps.
- Randomized MD records seeds and deterministic settings.

## Performance measurement

Report at least:

- systems per second;
- atoms per second;
- model-forward time;
- neighbour-list time;
- host-to-device transfer time when applicable;
- peak GPU memory;
- batch-size distribution and atom-count distribution;
- neighbour rebuild count;
- precision and compilation settings.

Warm up before timing. Synchronize CUDA before and after timed regions. Compare identical structures and identical requested steps.

## Scientific validation ladder

1. Tensor shapes and graph isolation.
2. Batch-size-one equivalence.
3. Batch-versus-single equivalence.
4. Finite-difference force checks on small systems.
5. Translation, rotation, and permutation tests appropriate to the model.
6. Stress finite differences for periodic systems.
7. Optimizer convergence against ASE on the same potential.
8. NVE drift and time-reversibility tests.
9. Thermostat distribution checks.
10. Domain-specific production validation.

Do not skip directly to production trajectories.

## High-value experiment backlog

1. GPU-native PBC cell-list neighbour builder.
2. Active-batch compaction for converged relaxations.
3. Size-aware bucketing and dynamic autobatching.
4. `torch.compile` modes and graph-break reduction.
5. Mixed precision with force/energy error controls.
6. Fused scatter and radial basis operations.
7. Multi-GPU sharding with one process per GPU.
8. Variable-cell Frechet/log-strain FIRE.
9. NPT integrators and stress validation.
10. General constraints with projection or RATTLE.
11. Conservative direct-force regularization/evaluation.
12. Long-trajectory restart and exact state restoration.

## Change discipline

- Keep public API changes backward compatible or document a migration.
- Prefer typed, testable functions over notebook-only work.
- Do not silently change units.
- Do not catch and suppress numerical errors.
- Do not weaken tolerances to make a failing change pass without a quantitative justification.
- Avoid global mutable state.
- Keep model-specific translation in adapters, not integrators.
- Add comments explaining physics or tensor semantics, not obvious syntax.

## Completion report

Every agent change should end with:

- hypothesis;
- files changed;
- commands run;
- tests and validation results;
- performance result with uncertainty;
- scientific limitations;
- recommended next experiment.
