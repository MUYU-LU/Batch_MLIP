# Topology-preserving active compaction

## Implementation

Compaction now carries candidate neighbor topology and reference geometry for
surviving systems. Newly admitted pending systems have an explicit cache-cold
validity bit, so the next AtomBit evaluation rebuilds only those systems that
are new or have independently invalidated their skin bound.

Edge blocks are remapped into the compact atom numbering without sending the
survivor graph to the CPU. Partial rebuilds transfer only dirty geometry to ASE
neighbor construction, retain clean edge blocks on the GPU, and concatenate the
new packed graph on-device.

The first prototype preserved topology but copied the complete resident graph
GPU-to-CPU on every partial rebuild. It was rejected after regressing immediate
refill from `334.485 s` to `371.754 s`. The retained implementation removes that
host round-trip.

## Matrix

Each point is one deterministic timing over the same 256 fixed 276-atom
structures used by Stage 2. All runs use AtomBit float32 inference, float64 BFGS
state, `FrechetCellFilter`, `skin=0.5 A`, B64, and at most 500 steps.

| policy | old rebuild-resident (s) | topology-preserving (s) | time reduction | throughput increase |
|:--|--:|--:|--:|--:|
| immediate | 334.485 | **280.282** | **16.2%** | **19.3%** |
| threshold | 341.155 | **285.801** | **16.2%** | **19.4%** |

Immediate remains 2.0% faster than threshold after both receive the same cache
optimization, so immediate remains the default refill policy.

## Why it works

For immediate refill, rebuilt-system work falls from 9,350 to 1,430 systems.
Neighbor-search time falls from `70.131 s` to `10.503 s`. Rebuild call count
rises from 217 to 368 because invalid systems are handled in smaller groups,
but each partial rebuild now scales with dirty systems rather than resident
batch size.

Packing itself becomes more expensive (`1.890 s` to `3.439 s`) because survivor
edges and cache references must be remapped. The `59.628 s` saved in neighbor
search is much larger than the `1.548 s` added to packing.

## Correctness and scope

Both optimized policies converge all 256 structures. Against their original
Stage 2 policy records, convergence flags, step counts, final positions, final
cells, and final energies match exactly.

This result applies to calculators that consume `AseGraphBatch` topology, such
as AtomBit. MACE was not retimed because the current native adapter reconstructs
MACE `AtomicData` graphs every evaluation and does not consume this cache. MACE
requires a separate native tensor-state cache rather than pretending this
generic graph optimization accelerates it.

Raw production outputs remain under `runs/topology_preserving_compaction/` on
the benchmark host.
