# MACE OMC-CSP P6000: automatic batching versus MPS16

## Question

Does the frozen AtomBit OMC-CSP execution architecture transfer to a different
MLIP, MACE-OFF23-Small, and remain faster than 16 ASE/CUDA-MPS processes per
GPU on the same strict variable-cell BFGS workload?

The benchmark uses all 6,000 unique T2 CIFs exactly once. Both paths use the
same MACE model file, eight H100 GPUs, float64 model and optimizer state,
`FrechetCellFilter`, `fmax=0.01 eV/A`, `max_steps=3000`, and no tail recovery.
Each path was timed once, as requested.

## Execution paths

Both automatic tensor controls use 31 offline-planned resident chunks
containing 109-256 structures. The outer scheduler uses model-specific
atom/edge/BFGS profiles, bucket-stratified initial placement, and
complete-chunk work stealing. Each GPU uses active compaction and active drain;
multi-GPU refill is disabled.

The original control rebuilds MACE `AtomicData` on every evaluation and uses
the native CUDA allocator. The improved path projects the common tensor state
directly into a persistent MACE candidate graph, uses expandable CUDA segments,
and releases completed chunk state at the persistent-worker boundary. Its
resident capacities come from a signed cached-graph/allocator-specific H100
capacity model; no timing or memory pilot is run for the P6000 workload.

MPS16 uses 16 persistent ASE BFGS processes per GPU, 128 processes in total.
Complete structures are assigned by static cost-balanced LPT. A worker-scoped
warning filter suppresses only repeated SciPy `logm` accuracy warnings; it does
not change the numerical calculation.

## Results

| Method | Execution/makespan (s) | Full script (s) | Systems/s | Converged |
|:--|--:|--:|--:|--:|
| Automatic tensor, rebuild control | 3,190.83 | 3,200.59 | 1.880 | 5,963/6,000 |
| Automatic tensor, cached accepted policy | 2,340.53 | 2,350.13 | 2.564 | 5,946/6,000 |
| ASE/CUDA-MPS, 16 workers/GPU | 4,563.54 | 4,652.83 | 1.315 | 5,960/6,000 |

The accepted automatic MACE path is **1.950x faster** than MPS16 using
execution/makespan and **1.980x faster** using full-script time. It is also
**1.363x faster** than the original tensor rebuild control. Worker balance is
effectively unchanged (`max/mean=1.053`).

MPS16 has similar per-GPU aggregate balance (`max/mean=1.058`), but static
assignment leaves a terminal tail: complete ASE jobs cannot move after their
unknown convergence durations become visible.

## Numerical gate

The three-step P64 control matches ASE/MPS16 to `2.91e-11 eV` in energy,
`1.18e-11 A` in positions, `1.07e-11 A` in cells, and
`1.09e-12 eV/A^3` in stress. This validates the model adapter, force/stress
mapping, cell filter, and early BFGS trajectory.

The 3,000-step production endpoints are not identical. The convergence totals
differ by only three systems, or 0.05 percentage points, but 73 paired flags
differ: tensor alone converges 38 and MPS16 alone converges 35. Only two fail
in both paths. Absolute endpoint energy differences have median 0.0059, p95
6.88, p99 15.75, and maximum 36.49 meV/atom.

Four difficult structures were rerun through tensor B1. By maximum coordinate
difference, two B1 endpoints are closer to the production tensor result and
two are closer to ASE/MPS16. This rules out a simple batch-size error and is
evidence of local-minimum bifurcation over long, floating-point-sensitive BFGS
trajectories. The timing result therefore measures the same optimization
contract; it does not claim identical minimum selection.

The cached graph has a separate three-step B244 control against graph rebuild:
energy is identical and maximum differences are `3.54e-13 eV/A` in forces,
`3.55e-15 A` in positions, `1.78e-15 A` in cells, and
`2.24e-16 eV/A^3` in stress. Over complete relaxations, schedule-sensitive
minimum selection remains: cached versus rebuild has median 0.0053, p95 6.65,
p99 15.47, and maximum 33.19 meV/atom across P6000. The cached path converges
5,946 systems versus 5,963 for rebuild and 5,960 for MPS16. These differences
remain inside the envelope already observed between rebuild batching and ASE.

Bounded production chunks isolate throughput from that endpoint variation.
Cached execution is `1.726x` faster on homogeneous H46 B244 and `1.281x`
faster on homogeneous H92 B256. Native and expandable allocation produce
identical endpoint hashes in both rebuild and cached modes.

## Memory and hotspots

The original capacity model is rejected because sequential chunks retained an
82.16-GB allocator high-water mark. The persistent worker omitted cyclic
garbage collection and cache release after offloading a completed result. The
fixed boundary returns reserved memory to 90 MB after every chunk. Combined
with expandable segments and the new signed capacity model, the production
peak is 67.47 GB, or 79.36% of the H100 and below the 85% budget. Peak allocated
memory is 59.58 GB. The held-out B256 capacity point is conservatively
overpredicted by 10.83%.

Selected worker phases are below; graph subphases are nested in neighbour
update and should not be summed with it:

| Phase | Worker-time share |
|:--|--:|
| BFGS update | 46.51% |
| MACE model forward | 45.19% |
| Neighbour update, including search | 2.77% |
| Cache-validity check | 0.99% |
| Active compaction | 0.90% |
| MACE tensor-state projection | 0.19% |

Repeated `AtomicData` construction is no longer a material hotspot. Dense BFGS
linear algebra is now the largest framework-controlled phase, narrowly ahead
of MACE forward evaluation; changing optimizer mathematics is outside this
freeze.

## Decision

Freeze this MACE OMC-CSP policy for the exact MACE-OFF23-Small, float64 BFGS,
`FrechetCellFilter`, H100 contract. Cached tensor graphs, persistent chunk
cleanup, the signed 85%-budget capacity model, and automatic expandable
segments are accepted. MACE fixed-cell and FIRE allocator policies remain
native because they have no matched evidence. The frozen AtomBit policy is
unchanged.

Machine-readable metrics and raw-artifact hashes are in
`results/improvement-summary.json` and
`results/bounded-improvement-summary.json`. The raw production results remain
on the H100 host at
`/public/home/lmy/Batch_imple_project/mace_omc_csp_p6000_v1`.
