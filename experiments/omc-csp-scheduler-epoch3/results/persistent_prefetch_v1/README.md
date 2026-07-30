# Persistent manifest executor and bounded prefetch

This experiment separates two execution mechanisms on the frozen ROF-A P2048
AtomBit/H100 variable-cell BFGS workload.

1. `BatchExecutor.relax_manifest` retains the seven GPU worker PIDs, model
   instances, CUDA contexts, and a compatible CPU loader generation across
   calls.
2. A global host-side source keeps at most one additional materialized chunk
   per active worker. Chunks remain unassigned until a worker completes its
   current task, so work stealing is retained.

| Path | Call 1 (s) | Call 2 (s) | Two calls (s) |
|---|---:|---:|---:|
| Best prior one-shot, four loaders | 100.727 | 100.727 | 201.454 |
| Persistent, no prefetch | 85.626 | 58.063 | 143.689 |
| Persistent, prefetch depth 1 | 83.435 | 57.492 | 140.927 |

Persistence plus the global loader is `1.429x` faster than two independent
best one-shot calls. The second call has zero worker startup and is `1.752x`
faster than the one-shot reference. Prefetch is a smaller, separately measured
effect on this two-wave workload: `1.020x` across both calls and `1.010x` on
the warm second call. All seven second-wave chunks were ready when requested.

All 2,048 jobs converged. Source order, chunk plans, optimizer steps, energies,
and forces match the best one-shot result exactly. Peak worker reservation was
`54.06 GB`, or `63.45%` of the H100 memory reported by PyTorch.

Each point is one run. The authoritative raw outputs remain on the execution
host as `executor_prefetch_rofa_p2048_v1.json` and
`executor_no_prefetch_rofa_p2048_v1.json`.
