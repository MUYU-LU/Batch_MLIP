# Hybrid refill and tail compaction

This experiment separates queue admission from end-of-pool compaction and
measures CUDA allocator fragmentation directly.

## Tail compaction

Every point optimized the same signed 256-job H46 pool. Pending jobs were
always admitted immediately. Tail-75% and tail-50% deferred physical
compaction only after the pending queue became empty.

| Model | Optimizer | Batch | Tail-75% speedup | Tail-50% speedup | Tail-75% frozen graphs |
|:--|:--|--:|--:|--:|--:|
| AtomBit | BFGS | 64 | 0.989x | 0.952x | 615 |
| AtomBit | BFGS | 128 | **1.055x** | **1.020x** | 1,437 |
| MACE | BFGS | 64 | 0.982x | 0.973x | 610 |
| MACE | BFGS | 128 | 0.950x | 0.931x | 1,375 |
| AtomBit | FIRE | 64 | 1.002x | 0.978x | 1,654 |
| AtomBit | FIRE | 128 | 0.934x | 0.918x | 3,298 |
| MACE | FIRE | 64 | 0.975x | 0.957x | 1,814 |
| MACE | FIRE | 128 | 0.951x | 0.851x | 4,748 |

All points converged 256/256 and stayed within `0.2 meV/atom` of immediate
refill. MACE trajectories preserve step counts exactly. AtomBit graph
reordering changes step counts, as in the earlier refill experiments.

AtomBit BFGS B128 was the only initial performance candidate. It did not
reproduce in matched allocator controls:

| Allocator | Immediate (s) | Tail-75% (s) | Tail speedup |
|:--|--:|--:|--:|
| Native | 60.69 | 60.35 | 1.006x |
| Effective expandable segments | 60.39 | 60.38 | 1.000x |

The initial `1.055x` result is therefore inconclusive timing variation rather
than a planner-safe scheduler improvement. Immediate compaction remains the
default; the threshold parameter remains explicit and experimental.

## Allocation and reserve

The allocated/reserved gap is real for AtomBit variable-cell BFGS. It comes
from changing-shape autograd/model workspaces, not the small persistent BFGS
Hessians. Effective expandable segments are the useful solution:

| Model/workload | Allocator | Time (s) | Peak allocated (GiB) | Peak reserved (GiB) | Retries |
|:--|:--|--:|--:|--:|--:|
| AtomBit H46 B128 | Native | 60.69 | 9.05 | 78.15 | 1 |
| AtomBit H46 B128 | Expandable | 60.39 | 9.04 | **10.07** | 0 |
| AtomBit H46 B128 | cudaMallocAsync | 65.26 | 9.04 | **9.69** | 0 |
| AtomBit H46 B128 | GC threshold 0.8 | 60.35 | 9.05 | 78.15 | 1 |
| AtomBit H276 B176 | Native | 90.16 | 64.03 | 78.03 | 8 |
| AtomBit H276 B176 | Expandable | **88.99** | 64.02 | **71.12** | 0 |
| MACE H276 B176 | Native | **143.10** | 69.50 | **76.66** | 1 |
| MACE H276 B176 | Expandable | 144.85 | 69.48 | 77.97 | 5 |

`cudaMallocAsync` controls memory but loses about 8% on H46. Proactive garbage
collection does not prevent the peak. Repeated `empty_cache()` calls are
therefore not justified.

On this PyTorch 2.9.1 build, the documented replacement
`PYTORCH_ALLOC_CONF` did not activate the requested settings. The deprecated
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did activate them, despite
emitting a warning. Launchers should set both variables to the same value for
cross-version compatibility until this environment is upgraded.

The resulting policy is:

1. AtomBit variable-cell BFGS: effective expandable segments.
2. MACE BFGS: native allocator.
3. MACE H276 B176 exceeds the 85% allocated-memory safety gate; choose a
   smaller resident batch even though the run completes.

Machine-readable summaries are under `results/`. All raw JSON traces are
retained in `results/raw-results.tar.gz`.
