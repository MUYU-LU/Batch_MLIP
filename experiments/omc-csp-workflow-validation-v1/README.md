# OMC-CSP workflow validation v1

This experiment validates the current plug-and-play OMC-CSP path without
retuning it for the selected workloads. The three signed manifests are from
the chemically held-out scheduler test split and contain unique CIFs.

The matrix covers a small homogeneous P64 pool, a chemically mixed P512 pool,
and a dense homogeneous P2048 pool. The P512 case is compared with ASE BFGS
under CUDA MPS using eight workers per physical GPU. One deterministic timing
trial is used, as requested; conclusions must therefore be treated as measured
points rather than timing distributions.

The validation distinguishes correctness from policy reachability. In
particular, assigning one device must not silently bypass an inner policy that
would be eligible when the calculator device is used implicitly.

## Initial validation

| Workload | GPUs | Converged | Execution | Peak worker reserved | Startup fraction |
|---|---:|---:|---:|---:|---:|
| JAYDUI P64 | 1 | 64/64 | 8.800 s | 5.0% | 42.0% |
| MIX P512 | 8 | 512/512 | 50.316 s | 9.6% | 50.1% |
| OBEQIX P2048 | 8 | 2048/2048 | 155.323 s | 77.9% | 16.8% |

Every automatic result preserved exact manifest coverage and order. The signed
offline hardware-capacity model matched all three runs, the automatic loader
tier selected 1/2/4 loader processes per worker, and no timing pilot ran. The
neighbour policy selected matscipy, cuda_dense, and cuda_cell in the measured
regimes. The P2048 run used 2,092 rebuilds for 3,754 model evaluations; model
forward plus autograd remained the dominant worker cost.

For MIX P512, automatic worker execution was 2.260x faster than the MPS8
production region and the complete automatic script was 1.684x faster than the
complete MPS script. Comparing cold automatic execution, which includes worker
startup, with warmed MPS production gives 0.998x and is intentionally retained
as a mixed-scope diagnostic. Current automatic endpoints remained within 0.499
meV/atom of the frozen automatic result. Against ASE BFGS under MPS, 11/512
endpoints exceeded 5 meV/atom, with a maximum of 19.221 meV/atom.

## Issues

1. `relax_manifest` with one supplied GPU enters the multi-GPU path, bypasses
   the offline refill selector, and reports a multi-GPU refill fallback.
2. The one-GPU P64 workload split one already-safe resident batch into two
   execution chunks.
3. Cold process/model startup consumed 42.0% of P64 and 50.1% of MIX P512
   execution, eroding the batched compute advantage.
4. MIX P512 expanded four resident plans into 16 execution chunks. Child chunks
   retained parent peak predictions, making per-execution memory predictions
   misleading.
5. Source-backed top-level `peak_memory` measured the parent CUDA context rather
   than process-worker peaks. Worker telemetry was used for the table above.
6. The nonpersistent source-backed result omitted the explicit
   `optimization_pilot_runs` telemetry field.
7. The OBEQIX P2048 slowest worker was 24.2% above the worker mean, leaving a
   19.5% end-of-run tail despite cost-aware work stealing.
8. Batched BFGS and ASE BFGS reached materially different minima for 11 MIX
   jobs, although the automatic scheduler itself remained stable against its
   frozen reference.

Raw outputs remain in the isolated H100 staging checkout under this
experiment's `results/` directory. `results/summary.json` preserves the
pre-fix diagnosis.

## Accepted fixes

The production convergence contract now sets `smax=None`, matching ASE BFGS
with `FrechetCellFilter`. Source-backed one-GPU execution uses the true inner
scheduler without outer subdivision. Multi-GPU CUDA manifest execution now
uses the bounded global-prefetch executor automatically. Worker CUDA contexts,
not the parent context, provide execution peak memory; a subdivided child
retains its parent only as `capacity_bound_bytes`.

| Workload | Final path | Execution | Peak worker reserved | Change |
|---|---|---:|---:|---:|
| JAYDUI P64, G1 | direct one-GPU B64 | 3.547 s | 8.46 GB | 2.481x faster |
| MIX P512, G8 | global prefetch, target2 | 50.242 s | 8.19 GB | 1.037x vs fixed standalone |
| OBEQIX P2048, G8 | global prefetch, target2 | 137.227 s | 66.23 GB | 1.084x vs worker-local loading |

The final MIX full script is 1.682x faster than the MPS8 full script (51.786
versus 87.093 s). Cold automatic execution is effectively tied with warmed
MPS production (50.242 versus 50.205 s); persistent `BatchExecutor` reuse
removes worker startup from subsequent pools.

Correcting `smax` reduced MIX endpoints above 5 meV/atom versus ASE from 11 to
2. B1 isolation of all 12 remaining >1 meV/atom cases had zero failures above
5 meV/atom; the two aggregate outliers matched ASE at B1 within 2 micro-eV/atom
and with identical step counts. The aggregate difference is therefore
float32 batch-order trajectory sensitivity in a nonconvex problem, not a BFGS
single-system implementation mismatch.

## Policy decisions

The P512 target1 candidate was slower and less balanced than target2. The
P2048 target4 candidate was 1.5% faster and better balanced, but changed two
endpoints by more than 5 meV/atom, so it was rejected. Refill plumbing is now
reachable in the source-backed one-GPU path, but the planner did not
extrapolate B64/B128 evidence to the unmeasured B256 descriptor regime.

Remaining bounded limitations are multi-GPU cold startup, float32 trajectory
sensitivity across batch partitions, workload-dependent outer tail balance,
and deliberately narrow refill evidence. `results/fix-summary.json` is the
machine-readable remediation conclusion.
