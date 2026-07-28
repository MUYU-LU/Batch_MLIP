# Prompt BatchExecutor shutdown

`PersistentTaskPool.close()` previously joined four workers sequentially with
a 30-second timeout. CUDA workers also selected PyTorch's `file_system`
sharing strategy, whose detached `torch_shm_manager` processes retained the
parent resource-tracker pipe. The combined behavior added about 120 seconds
after useful computation had finished and could keep the interpreter alive
after pytest printed its summary.

The implementation now:

- sends every shutdown sentinel before waiting;
- records one shared-memory acknowledgment per worker;
- applies one global two-second deadline instead of a timeout per worker;
- concurrently terminates only workers that remain after acknowledging that
  no useful task remains;
- uses bounded `file_descriptor` tensor sharing for persistent task results;
- exposes shutdown wall time, acknowledged worker IDs, and forced-worker count
  through `BatchExecutor.shutdown_metadata`.

## Held-out results

The test reuses the exact signed R256 and R1024 mixed-family manifests,
four GPUs, deterministic AtomBit float32, variable-cell BFGS, `fmax=0.05`,
and `max_steps=500` from `heldout-auto-vs-mps`.

| Pool | Close | External before | External after | MPS external | After vs MPS | Production before/after | Peak reserved before/after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.783 s | 193.080 s | 71.788 s | 131.699 s | 1.835x | 33.208 / 32.864 s | 9.368 / 9.366 GB |
| 1024 | 1.056 s | 298.881 s | 180.701 s | 341.654 s | 1.891x | 91.047 / 90.726 s | 46.890 / 46.890 GB |

All four workers acknowledged shutdown in both runs. All 256 and all 1024
structures converged. Compute time and peak memory are unchanged within the
declared gates, so the process-level gain comes from eliminating teardown
wait rather than changing optimization work.

Independent dynamic-work-stealing runs are not bitwise reproducible, even
with deterministic CUDA kernels, because chunks may execute on a different
GPU. Therefore final-state hashes are retained in raw records for audit but
are not used as a cross-run shutdown gate. Shutdown begins after all results
have already returned to the parent.

Validation: `PYTHONPATH=. pytest -q` completed normally with 250 passed and
10 skipped in 32.45 seconds. Ruff also passed.
