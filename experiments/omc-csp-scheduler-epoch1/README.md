# OMC-CSP Scheduler Refinement Epoch 1

This experiment separates scheduler method development from scheduler
validation. It derives immutable `v2` manifests from the signed OMC-CSP `v1`
P2048 family pools; it does not rescan, regenerate, or replicate CIFs.

## Family Split

| Split | Families | Role |
|---|---|---|
| Development | BOQWIN, GUFJOG, HAMTIZ, KONTIQ, NACJAF, OBEQUJ, PAHYON, SOXLEX, UJIRIO, XAFPAY, XATMOV, ROF-B | Diagnose one bottleneck and implement one refinement. |
| Validation | AXOSOW, BOQQUT, WICZUF, WIDBAO, XAFQIH | Accept or reject the refinement. |
| Test | JAYDUI, OBEQIX, XULDUD, ROF-A, ROF-C | Final one-time scheduler evaluation. |

All `P64`, `P512`, and `P2048` descendants of a family remain in its split.
`XIFZOF` is excluded because it contains Si, which the frozen AtomBit
smooth-rms checkpoint does not support. `OBEQET` has no candidates.

## Matrix

Development runs all 12 families at `P512`. Five of those same intra-family
workloads, GUFJOG, KONTIQ, BOQWIN, XAFPAY, and ROF-B, additionally run at the
extreme pool sizes `P64/P2048`. Validation follows the same pattern for
BOQQUT and WIDBAO. The test runs all five test families at every pool size.
Each split also has one balanced, split-local mixed-family `P512` pool.

The current automatic scheduler and MPS are measured on development first.
One bottleneck-specific scheduler refinement is then made. The refined policy
must pass validation before the test manifests are executed; test results never
select another change.

## Execution Contract

The development runner executes 23 workloads, each once, without timing
repeats. `P64` uses GPU 0, `P512` uses GPUs 0--3, and `P2048` uses GPUs 0--6.
The current automatic scheduler uses deterministic BatchedBFGS with a
`FrechetCellFilter`, float64 optimizer state, 6.0 A cutoff, and 0.5 A skin.
Its process-owned production chunks retain phase telemetry for neighbor work,
model/autograd, BFGS linear algebra/state updates, and active compaction.

The reference is ASE BFGS with ASE `FrechetCellFilter`, run as four workers
per GPU through one CUDA MPS daemon. Complete structures are statically
cost-balanced across the selected GPUs using only atoms, candidate edges, and
the BFGS dimension; no timing labels are used for that baseline assignment.
Both methods retain every final endpoint record and report an external
full-process makespan in addition to their production timing.

## Artifacts

- Host workload directory:
  `/public/home/lmy/Batch_imple_project/omc_csp_scheduler_workloads_v2`
- Construction hash:
  `16337463225baf362840bea21e598d295c7dde41c7aa3819ec1c226bf7f7469a`
- Local index snapshot: [workload_index.json](results/workload_index.json)

The `v2` directory contains 66 nested intra-family manifests and three
balanced split-local `P512` mixed manifests.

Matching layered planning sidecars are at
`/public/home/lmy/Batch_imple_project/omc_csp_scheduler_planning_profiles_v2`.
Their index hash is
`4b99107315f3b61970dc6525460216aedd22d4e07e718d21664bdb86f549ed6d` and the
validator checked 59,264 manifest-bound system references.

## Development Result

All 23 automatic runs were faster than the matched MPS baseline. The median
external full-process speedup was `2.096x`; pool-specific medians were
`2.097x`, `2.008x`, and `1.967x` for `P64`, `P512`, and `P2048`.
Identity, coverage, contract, no-OOM, and conservative 85% memory gates
passed.

The convergence-count gate failed. Across 17,216 jobs, automatic tensor BFGS
converged 17,072 and MPS converged 17,132. The methods did not fail on exactly
the same jobs: tensor alone converged 56, MPS alone converged 116, and the
automatic nonconverged tail contained 144 jobs.

Development-only solver diagnostics rejected a global linear-algebra switch.
Batched eigen algebra recovered KONTIQ `P64` from 62 to 63 converged jobs but
left ROF-B at 62; serial eigen left both at 62. Tensor `B1` retries converged
only one of four representative MPS-only failures.

The selected refinement is therefore a source-stable, persistent ASE-BFGS
recovery pass over only the tensor-nonconverged tail. It retains the user's
BFGS, Frechet, cutoff, precision, threshold, and per-attempt step contract.
The recovery endpoint replaces the tensor endpoint unconditionally for every
retried source, so the scheduler does not choose between two final energies.
The full retry cost is included in makespan and model-evaluation telemetry.

The complete report is
[development_analysis.md](results/development_analysis/development_analysis.md),
with machine-readable JSON and CSV beside it. The refinement contract is
[refinement_decision.json](refinement_decision.json).

## Validation Result

The persistent ASE-BFGS tail recovery passed the frozen ten-workload
validation split. All identity, numerical-contract, no-OOM, 85% memory, and
convergence non-regression gates passed. The refined automatic method won all
ten external full-process comparisons, with a median `1.468x` speedup over
MPS. It converged 7,295 of 7,296 jobs versus 7,294 for MPS; recovery attempted
three tensor-tail jobs and converged two.

The complete validation report is
[validation_analysis.md](results/validation_analysis/validation_analysis.md).
Passing validation accepted the one epoch-1 method change and froze the
implementation before test.

## Untouched Test Result

The frozen implementation then ran once on all 16 untouched test workloads:
five `P64`, six `P512`, and five `P2048`. Exact coverage, requested-contract,
no-OOM, 85% memory, and convergence non-regression gates all passed. Both
methods converged all 13,632 jobs, so no tail-recovery work was triggered.

Automatic batching won 14 of 16 full-process comparisons. It won every `P64`
and `P512` workload. Median external speedups were `2.159x` at `P64`,
`1.403x` at `P512`, and `1.053x` at `P2048`. The `P2048` result is a genuine
family-dependent boundary:

| Family | Automatic (s) | MPS (s) | MPS / automatic |
|---|---:|---:|---:|
| JAYDUI | 115.340 | 74.124 | 0.643x |
| OBEQIX | 536.598 | 656.298 | 1.223x |
| ROF-A | 418.934 | 441.257 | 1.053x |
| ROF-C | 215.919 | 212.013 | 0.982x |
| XULDUD | 138.719 | 160.823 | 1.159x |

The automatic full-process decomposition attributes `59.03%` of aggregate
test time to structure materialization, model setup, and output, compared
with `24.60%` for worker execution and `10.08%` for worker startup. This
identifies a data-path bottleneck rather than a BFGS, graph-cache,
compaction, refill, or GPU-memory failure. Peak conservative memory remained
below 85% for every workload.

The complete untouched report is
[test_analysis.md](results/test_analysis/test_analysis.md), with
machine-readable JSON and CSV beside it.

## Epoch Decision

Epoch 1 is complete. Tail recovery is accepted because it passed validation
and retained scientific parity on untouched test families. Its performance
claim is bounded: the current eager parent-side input path is not reliably
faster than MPS for `P2048` short-to-moderate OMC-CSP workloads.

The single next performance hypothesis is manifest-backed lazy
materialization. The parent will plan from signed layered sidecars and pass
only immutable source descriptors to persistent GPU workers; workers will
load their assigned structures. The experiment must retain the same job
order, resident plan, GPU assignment, BFGS/Frechet contract, and MPS baseline
so that only the data path changes.
