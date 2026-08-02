# T2 P6000 strict BFGS: automatic batching versus MPS16

## Question

Does the accepted OMC-CSP active-drain workflow retain its advantage over
ASE/CUDA-MPS when the production relaxation contract is tightened from
`fmax=0.05`, `max_steps=500` to `fmax=0.01`, `max_steps=3000`?

The comparison uses all 6,000 unique T2 CIFs exactly once, the same signed
manifest, AtomBit checkpoint, eight H100 GPUs, deterministic settings,
float64 BFGS state, Frechet cell relaxation, and no post-run tail recovery.
Automatic batching uses the unchanged 31 memory-safe resident chunks,
bucket-stratified outer work stealing, active compaction, graph caching, and
explicitly disabled multi-GPU refill. MPS16 uses 16 persistent ASE BFGS
workers per GPU, 128 workers total, with static cost-balanced LPT assignment.

## Results

Each method was run once, as requested.

| Method | Execution/makespan (s) | Full script (s) | Systems/s | Converged | Peak memory |
|:--|--:|--:|--:|--:|--:|
| Accepted active-drain batching | 2,245.09 | 2,248.97 | 2.672 | 5,986/6,000 | 76.68 GB reserved |
| ASE/CUDA-MPS, 16 workers/GPU | 4,706.15 | 4,775.35 | 1.275 | 5,985/6,000 | 77.70 GB sampled |

MPS makespan divided by batched execution time is **2.096x**. Comparing the
inner production regions gives 2.130x, and full scripts give 2.123x. Automatic
batching performs 69,957 batched model calls for 3,827,420 structure
evaluations. MPS performs 3,807,979 single-structure model calls.

The benchmark's `script_seconds` is sampled immediately before writing the
large result JSON. It includes model loading, worker startup, materialization,
execution, reassembly, and record construction, but not final artifact I/O.

The workload is genuinely long-horizon. Batched median, p90, and p99 step
counts are 520, 1,092, and 2,222; 14 structures reach 3,000 steps. The MPS
values are 518, 1,072, and 2,206; 15 structures reach 3,000 steps.

## Numerical outcomes

Both methods preserve all 6,000 source IDs, but they are not endpoint
equivalent. They disagree on convergence status for 27 sources. Only one
source is nonconverged in both methods; the totals happen to differ by one.

Across all sources, the absolute energy difference has median 0.0029,
p95 4.06, p99 14.24, and maximum 30.84 meV/atom. This distribution is
consistent with mostly close trajectories plus a chemically relevant tail of
schedule-sensitive alternative minima. The speed result therefore applies to
executing the fixed optimization contract; it must not be presented as proof
that batched and ASE BFGS always select the same minimum.

## Production findings

The accepted initial-state capacity model is not conservative enough for this
strict horizon. Batched peak reserved memory reaches 76.68 GB, 90.2% of the
85.02-GB H100, despite a nominal 85% policy. MPS reaches a 77.70-GB sampled
peak, 91.4%, because 16 independent allocators retain high-water blocks.
These values use different measurement APIs, so their small difference is not
a defensible method ranking; both demonstrate inadequate long-horizon margin.

The framework hotspot also changes relative to the earlier `fmax=0.05` run.
Summed batched worker time is distributed as follows:

| Inclusive phase | Worker-time share |
|:--|--:|
| Model forward and autograd | 64.6% |
| BFGS update | 27.1% |
| Neighbor update | 2.9% |
| Active compaction | 0.9% |

Within BFGS update, eigensolver fallback consumes 20.2% of worker time and is
invoked on 63,091 of 69,926 updates. Neighbor construction is no longer the
dominant framework-controlled target for this contract.

Outer work stealing remains effective: batched worker max/mean time is 1.031,
with a 65.2-s ideal-balance upper bound. MPS shard max/mean time is 1.043, but
its final minutes show several idle GPUs because static atom/edge cost cannot
predict individual strict convergence duration.

## Decision

Keep the original active-drain chunks for production OMC-CSP. They are 2.096x
faster than MPS16 on the strict P6000 contract and use comparable peak memory.

Do not freeze the strict production policy unchanged. The next engineering
epoch must add:

- a horizon-aware memory margin or allocator high-water guard;
- a tail-safe BFGS linear-algebra path that reduces repeated eigensolver
  fallback;
- an explicit policy for max-step structures and endpoint reporting;
- comparable device-level memory sampling for both methods.

Raw result JSON and logs remain on the H100 experiment host. Machine-readable
metrics are in `results/summary.json`.
