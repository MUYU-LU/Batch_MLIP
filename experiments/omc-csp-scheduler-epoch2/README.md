# OMC-CSP Scheduler Performance Epoch 2

Epoch 1 accepted persistent ASE-BFGS tail recovery but found that eager
parent-side structure materialization consumed `59.03%` of aggregate
untouched-test automatic time. Automatic batching won all `P64/P512`
workloads but only three of five `P2048` workloads.

## Hypothesis

Planning from the existing signed layered sidecar and materializing production
structures inside process-owned GPU workers will reduce full-process makespan
without changing the deterministic resident plan, execution chunk order,
optimizer, graph/cache policy, compaction policy, numerical endpoints, or
memory safety.

This is one data-path change. It does not introduce hardware-model capacity
planning, prefetch, refill changes, or a new optimizer.

## Causal Matrix

The first ablation reuses the five frozen epoch-1 test `P2048` manifests:
JAYDUI, OBEQIX, ROF-A, ROF-C, and XULDUD. Each source-backed run executes
once on GPUs 0-6. The matched eager automatic and four-workers-per-GPU MPS
records are the immutable epoch-1 artifacts; they are not rerun.

The source-backed run must use:

- the same manifest and source order;
- the matching signed planning sidecar;
- the same AtomBit smooth-rms fp32 checkpoint;
- float64 BatchedBFGS with `FrechetCellFilter`;
- 6.0 A cutoff and 0.5 A skin;
- `fmax=0.05`, `max_steps=500`, `max_step=0.2`, and `alpha=70`;
- the same seven GPUs and deterministic process scheduler;
- ASE-BFGS tail recovery under the accepted epoch-1 contract.

## Gates

- Exact source identity and output coverage.
- Identical resident-plan and execution-chunk signatures versus eager.
- Identical requested numerical contract.
- No OOM and conservative peak reserved memory at most 85% per GPU.
- Convergence count no lower than eager or MPS.
- Endpoint comparison retained by immutable source ID.
- External full-process makespan improvement over eager.
- Report speedup over the existing MPS record, including failures.

This matrix is a causal engineering ablation, not a new chemically untouched
test split. The held-out validation below uses families not selected from
these five results before updating the default production policy.

## Causal Result

All five source-backed runs completed and passed the exact plan-signature,
identity, numerical-contract, no-OOM, 85% memory, and convergence gates.
Source-backed and eager endpoints were exactly identical for every stored
energy, force, stress, position, cell, convergence state, and step record.
All three methods converged all 10,240 jobs.

| Family | Source-backed (s) | Eager (s) | MPS (s) | vs eager | vs MPS |
|---|---:|---:|---:|---:|---:|
| JAYDUI | 47.633 | 115.340 | 74.124 | 2.421x | 1.556x |
| OBEQIX | 210.563 | 536.598 | 656.298 | 2.548x | 3.117x |
| ROF-A | 138.887 | 418.934 | 441.257 | 3.016x | 3.177x |
| ROF-C | 93.645 | 215.919 | 212.013 | 2.306x | 2.264x |
| XULDUD | 79.933 | 138.719 | 160.823 | 1.735x | 2.012x |

Summed full-process makespan fell from `1,425.51 s` eager and `1,544.51 s`
MPS to `570.66 s` source-backed. This is an aggregate `2.498x` speedup over
eager and `2.707x` over MPS. Source-backed won all five workloads, with a
median `2.421x` speedup over eager and `2.264x` over MPS. Conservative peak
memory was at most `72.01%` of one H100.

The eager structure/model setup/output component was `946.97 s`, or `66.43%`
of eager aggregate time. Source-backed reduced the corresponding component to
`13.95 s`, or `2.44%`. Worker execution increased because it now includes
parallel CIF materialization, but this work is distributed across the seven
process workers rather than serialized in the parent.

The complete report is
[source_backed_analysis.md](results/source_backed_analysis/source_backed_analysis.md),
with machine-readable JSON and CSV beside it.

## Decision

The mechanism is accepted as an effective OMC-CSP P2048 acceleration, and the
measured epoch-1 large-pool bottleneck is solved for this causal matrix. These
five families selected the mechanism, so the implementation was frozen before
a second workload-regime-held-out validation.

## Held-Out Validation

The unchanged source-backed path was evaluated on BOQWIN, GUFJOG, KONTIQ,
ROF-B, and XAFPAY `P2048`. These families were not used to select lazy
materialization. Tail recovery was disabled for both lazy and frozen eager
automatic execution in this isolation, so the data path was required to
reproduce the raw tensor endpoint exactly. The independently accepted epoch-1
recovery layer remains composable afterward.

All five validation workloads passed exact plan, source identity, endpoint,
contract, no-OOM, and 85% memory gates:

| Family | Source-backed (s) | Eager (s) | MPS (s) | vs eager | vs MPS |
|---|---:|---:|---:|---:|---:|
| BOQWIN | 159.829 | 265.881 | 368.016 | 1.664x | 2.303x |
| GUFJOG | 164.653 | 189.050 | 465.587 | 1.148x | 2.828x |
| KONTIQ | 247.119 | 310.120 | 716.373 | 1.255x | 2.899x |
| ROF-B | 488.909 | 1,052.815 | 2,071.279 | 2.153x | 4.237x |
| XAFPAY | 177.668 | 332.024 | 439.887 | 1.869x | 2.476x |

Across these 10,240 additional jobs, source-backed makespan was `1,238.18 s`
versus `2,149.89 s` eager and `4,061.14 s` MPS: aggregate speedups of `1.736x`
and `3.280x`. It won every workload; median speedups were `1.664x` over eager
and `2.828x` over MPS. Peak conservative memory was `76.17%`.

Source-backed and eager tensor results were exactly identical. Both converged
10,146 jobs before recovery. MPS converged 10,183, which is consistent with
the numerical-tail problem already handled by the accepted recovery layer and
is not attributed to materialization.

The validation report is
[source_backed_analysis.md](results/source_backed_validation/source_backed_analysis.md).

## Final Decision

Manifest-backed worker materialization is accepted as the default data path
for signed, large-pool OMC-CSP workloads. Ordinary callers that already hold
ASE `Atoms` continue to use `relax(...)`; file-backed workloads use
`relax_manifest(...)` and the matching signed planning sidecar.

This decision does not yet enable offline hardware-calibrated capacity
integration or asynchronous prefetch. Those remain separate changes with
their own causal experiments.
