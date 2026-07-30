# OMC-CSP Scheduler Performance Epoch 3

Epoch 2 removed eager parent-side CIF materialization but retained one
representative model forward to estimate each device's resident capacity.
Epoch 3 integrates the existing signed H100 reserved-memory calibration into
the public manifest-backed execution path.

## Hypothesis

For an exact model, task, graph, allocator, software, and hardware contract,
the signed offline byte model can replace the production memory probe while
retaining the outer scheduler's workload buckets, satisfying every predicted
chunk bound and the 85% device-memory limit.

This changes capacity selection only. It does not select the optimizer,
modify BFGS or `FrechetCellFilter`, change compaction/refill, or infer an OMC
application from atomic structures.

## Runtime Policy

`relax_manifest(...)` now:

1. verifies the workload and signed planning-profile binding;
2. matches the complete runtime contract against the packaged signed capacity
   policy;
3. preserves the task profiler's outer cost buckets;
4. applies the calibrated reserved-memory model and the existing 1.10
   optimization-growth guard inside each bucket;
5. packs chunks under 85% of one H100 and dispatches them through the existing
   multi-GPU worker scheduler;
6. performs zero model probes when the contract matches;
7. falls back to the representative probe for any contract mismatch.

The packaged policy is specific to AtomBit smooth-rms fp32, float64
`BatchedBFGS`, `FrechetCellFilter`, 6.0 A cutoff, 0.5 A skin, autograd forces,
expandable CUDA allocator segments, PyTorch 2.9.1/CUDA 12.8, and NVIDIA H100
80GB HBM3 GPUs.

## Validation Matrix

The non-fit matrix contains three frozen `P2048` workloads and 6,144
variable-cell optimization jobs:

- ROF-A and ROF-C are coefficient-fit-excluded calibration-validation
  families.
- XULDUD is a chemically untouched final-test family.

Each offline run executes once on GPUs 0-6. Probe-backed tensor and
four-workers-per-GPU ASE CUDA MPS timings are immutable epoch-2/epoch-1
references over the same manifests and numerical contract.

## Diagnostic And Correction

The first integration passed the global 85% limit but underpredicted seven
individual chunks by at most 1.50%. It exposed two implementation defects:

- the offline path omitted the scheduler's existing 1.10 memory-growth guard;
- the inner capacity planner re-bucketed systems instead of preserving the
  outer scheduler's workload buckets.

Both were corrected as architectural invariants rather than by refitting the
calibration. The failed diagnostic is retained under `results/diagnostic_v1`.

## Refined Result

| Family | Offline (s) | Probe (s) | MPS (s) | vs probe | vs MPS | Peak | Bound ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROF-A | 132.860 | 138.887 | 441.257 | 1.045x | 3.321x | 63.58% | 0.9160 |
| ROF-C | 94.165 | 93.645 | 212.013 | 0.994x | 2.251x | 51.20% | 0.9104 |
| XULDUD | 74.482 | 79.933 | 160.823 | 1.073x | 2.159x | 50.71% | 0.8996 |

Aggregate offline makespan was `301.51 s`, versus `312.46 s` for the
probe-backed tensor path and `814.09 s` for MPS. This is `1.036x` and
`2.700x` faster, respectively. Every job converged, all capacity gates passed,
the maximum reserved-memory fraction was `63.58%`, and the worst actual to
predicted chunk-memory ratio was `0.9160`.

The complete report is
[capacity_integration.md](results/refined_v2/capacity_integration.md), with
machine-readable JSON and CSV beside it.

## Numerical Limitation

Changing resident chunk composition changes floating-point reduction shapes.
The previous exact endpoint equality therefore does not hold for every job.
Against the probe-backed tensor path, the project's established variable-cell
tolerances fail for three ROF-C energy records, two ROF-C position records,
and one XULDUD energy record. All force, stress, cell, convergence, source
identity, and coverage checks pass; the 95th-percentile difference is zero or
near numerical noise for every reported endpoint metric.

The corresponding tensor-versus-MPS differences are substantially broader.
This does not make the sparse offline-versus-probe differences disappear, so
capacity acceptance and strict endpoint reproducibility are reported as
separate gates.

## Decision

The signed offline model is accepted as the capacity mechanism for the exact
packaged AtomBit/H100 manifest contract. It eliminates the representative
production probe and retains automatic probe fallback outside that contract.

The experiment does not establish schedule-invariant endpoints, policy
transfer to MACE or other hardware, or a universal throughput model. Those
remain separate validation problems.

## CPU Materialization Follow-up

The next measured hotspot was worker-side CIF parsing. A bounded process loader
and its no-pilot selection rule are documented in
[`results/materialization_v1`](results/materialization_v1/README.md). On the
heavy ROF-A P2048 workload it reduced critical-worker parsing by `3.23x` and
worker runtime by `1.41x`; the policy retains serial loading for the lighter
XULDUD workload where four loaders regressed external time.

The follow-on
[`persistent_prefetch_v1`](results/persistent_prefetch_v1/README.md) integrates
source-backed pools with `BatchExecutor`. Reusing GPU/model workers across two
ROF-A P2048 calls gives `1.429x` over two independent best one-shot runs.
Bounded global prefetch adds a separate `1.020x` over persistence alone.

## Scheduler v1 Freeze Validation

The final
[`freeze_validation_v1`](results/freeze_validation_v1/README.md) runs 7,680
unique test-split jobs from six consecutive homogeneous and mixed P512/P2048
pools through one persistent seven-H100 executor. All jobs converge, every
endpoint is bitwise identical to its independent same-plan tensor control,
peak reservation is `63.61%`, and GPU workers survive the expected
`28 -> 7` CPU-loader transition.

Internal script makespan is `244.36 s`, versus `1260.10 s` for the summed
frozen MPS references (`5.157x`); production-only speedup is `5.043x`.
OMC-CSP Scheduler v1 is frozen for the exact signed AtomBit/H100
variable-cell BFGS contract. GPU-count selection, multi-GPU refill, other
MLIPs/hardware, conformer workloads, and MD remain outside this freeze.
