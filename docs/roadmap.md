# Roadmap

## P0 — production correctness

- Validate real checkpoints against the existing ASE calculator.
- Add finite-difference force and stress utilities to the CLI.
- Add exact restart/resume for optimizer and thermostat internal state.
- Add dataset-level failure summaries and NaN/overflow guards.

## P1 — throughput

- Follow the decision-gated [workload-aware performance strategy](workload-aware-performance.md)
  for neighbor caching, refill, resident planning, and multi-GPU experiments.
- GPU-native periodic cell-list neighbour construction.
- Active-batch compaction after graphs converge.
- Atom-count/edge-count bucketing and adaptive batch sizing.
- CUDA timing and peak-memory instrumentation.
- `torch.compile` compatibility and graph-break reports.

## P2 — broader simulation capability

- Frechet/log-strain variable-cell FIRE.
- NPT dynamics.
- General constraints and RATTLE.
- Multi-GPU sharding.
- Replica and ensemble workflows.

## P3 — model/science experiments

- Conservative-versus-direct force comparisons.
- Mixed-precision error budgets.
- Cutoff smoothness and neighbour-skin sensitivity.
- Long-time stability across chemistries and phases.
- Uncertainty-driven active learning hooks.

Each roadmap item should enter through the experiment protocol in `AGENTS.md`, not as an unbenchmarked rewrite.
