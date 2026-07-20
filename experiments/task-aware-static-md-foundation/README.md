# Static and fixed-horizon MD execution foundation

## Purpose

This stage turns the task-aware protocol into executable workloads. It freezes
single-point force evaluation and NVE trajectory work at small (`R32`) and large
(`R256`) pool sizes for homogeneous 46-atom, homogeneous 276-atom, and mixed
structure distributions. It makes no speedup claim.

## Implementation

- Added six `EVAL` and six `MD-NVE` signed manifests with exact source and
  normalized-structure hashes.
- Assigned each MD replica an independent deterministic seed, so initial
  velocities do not depend on resident batch partitioning.
- Added calculator-independent materialization and ordered microbatch execution
  for any native `BatchCalculator`.
- Added a YAML entry point that writes telemetry JSON, runtime phases, summary
  JSON, final structures, and an optional append-only registry row.
- Defined synchronized measured time, end-to-end time, throughput, and measured
  peak allocated/reserved GPU memory without mixing timing boundaries.

## Correctness

Static energies, forces, and job order are compared across resident chunkings.
NVE positions and velocities are compared across equivalent chunkings. The
manifest validator checks workload identities, task metadata, and per-job seeds.

## Commands

```bash
PYTHONPATH=. python tools/generate_controlled_workloads.py
PYTHONPATH=. python tools/validate_controlled_workloads.py \
  --output experiments/task-aware-static-md-foundation/validation.json
pytest -q
python -m ruff check batch_mlip tests tools
```

Generation was repeated and SHA-256 checksums matched for all 81 files. Semantic
validation passed for all 20 workloads. The full suite passed with 94 tests and
6 optional MACE integration tests skipped by their explicit environment gate.

A real MACE-OFF-Small `EVAL-MIX-R32-v1` YAML smoke run on one GPU completed all
32 jobs in manifest order and wrote every declared output. Its measured-region
time was 0.429 s, end-to-end time was 5.938 s, and peak allocated/reserved memory
was 7.882/8.521 GB. This single run validates execution and instrumentation only;
it is not used as a speed or policy comparison.

## Next Measurement

Run the frozen manifests with sequential ASE, AtomBit native batching, and MACE
native batching. Screen safe resident sizes using peak memory, then report both
measured and end-to-end speedup against ASE B1. Cache, refill, packing, and
multi-GPU policies enter only where the workload profile makes them applicable.
