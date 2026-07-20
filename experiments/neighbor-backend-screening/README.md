# Neighbor backend screening

## Question

Does replacing the compiled matscipy neighbor list with a GPU implementation
actually reduce graph-construction time, or do CUDA launch, allocation, and
transfer overhead eliminate the benefit?

## Protocol

The benchmark uses the frozen H46, H276, and MIX EVAL manifests. It measures
the complete boundary from ASE structures to packed `edge_index` and integer
periodic-shift tensors resident on an H100. CIF I/O is excluded. Each method is
warmed up once and measured once, following the screening convention used by
the task-aware matrix.

The compared methods are serial matscipy, an eight-worker matscipy thread pool,
and an exact PyTorch CUDA screen. The CUDA prototype is dense O(N^2), evaluates
all 27 periodic images in `[-1, 1]^3`, and buckets systems by atom count. Every
directed edge and integer periodic shift is sorted and compared exactly with
serial matscipy. All 24 measured points pass this gate.

The serial matscipy path is also timed in two phases. Raw compiled neighbor
search accounts for 94.5%-97.9% of its time; NumPy packing and host-to-device
transfer account for only 2.1%-5.5%. The CUDA result is therefore not explained
by avoiding transfer alone.

## Fixed-matrix results

`T8` and `CUDA` are speedups over serial matscipy. Memory is CUDA peak allocated
memory for the complete returned graph pool, including the PyTorch CUDA context.

| Distribution | Pool | Resident B | Cutoff (A) | Serial (s) | T8 | CUDA (s) | CUDA | Search share | CUDA GB |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| H46 | 32 | 32 | 6.0 | 0.03706 | 0.60x | 0.00477 | 7.77x | 96.3% | 0.040 |
| H46 | 32 | 32 | 6.5 | 0.03793 | 0.66x | 0.00443 | 8.56x | 96.6% | 0.041 |
| H46 | 256 | 128 | 6.0 | 0.27402 | 0.65x | 0.00908 | 30.18x | 97.4% | 0.069 |
| H46 | 256 | 128 | 6.5 | 0.29845 | 0.62x | 0.00972 | 30.71x | 97.6% | 0.073 |
| H276 | 32 | 16 | 6.0 | 0.10147 | 0.79x | 0.01099 | 9.24x | 96.3% | 0.143 |
| H276 | 32 | 16 | 6.5 | 0.14308 | 0.80x | 0.01057 | 13.54x | 97.0% | 0.145 |
| H276 | 256 | 16 | 6.0 | 0.80594 | 0.82x | 0.07546 | 10.68x | 96.2% | 0.229 |
| H276 | 256 | 16 | 6.5 | 1.14349 | 0.85x | 0.07548 | 15.15x | 97.1% | 0.240 |
| MIX | 32 | 32 | 6.0 | 0.08495 | 0.83x | 0.00960 | 8.85x | 97.3% | 0.138 |
| MIX | 32 | 32 | 6.5 | 0.12494 | 0.89x | 0.00973 | 12.84x | 97.9% | 0.139 |
| MIX | 256 | 32 | 6.0 | 0.53677 | 0.79x | 0.06776 | 7.92x | 96.4% | 0.188 |
| MIX | 256 | 32 | 6.5 | 0.72996 | 0.83x | 0.06729 | 10.85x | 97.1% | 0.196 |

## Batch crossover at 6.5 A

This scan fixes each pool at 32 structures, so serial matscipy performs the same
work at every point while the CUDA resident batch changes.

| Resident B | H46 CUDA | H46 CUDA GB | H276 CUDA | H276 CUDA GB |
|--:|--:|--:|--:|--:|
| 1 | 0.42x | 0.036 | 1.53x | 0.053 |
| 2 | 0.72x | 0.036 | 2.76x | 0.059 |
| 4 | 1.34x | 0.037 | 5.02x | 0.071 |
| 8 | 2.51x | 0.037 | 8.98x | 0.096 |
| 16 | 4.50x | 0.038 | 13.30x | 0.145 |
| 32 | 8.62x | 0.041 | 17.90x | 0.242 |

Eight-way matscipy threading is slower at every point (0.57x-0.89x). For H46,
CUDA launch overhead loses at B1-B2 and the measured crossover is B4. For H276,
the denser graph amortizes CUDA overhead at B1. Larger resident batches improve
CUDA throughput but increase temporary dense memory, most visibly for H276.

## Decision

Matscipy remains the safe backend for small H46 batches and unsupported cell
geometries. A GPU backend is justified for batched rebuild-heavy evaluation and
optimization, but the dense prototype is not yet a general production backend.
The next implementation should expose an adaptive backend boundary and retain
exact matscipy fallback. End-to-end AtomBit and MACE measurements are required
after integration because neighbor-only speedup is bounded by the remaining
model and optimizer work.

The current prototype is limited to fully periodic cells whose required image
shifts fit `[-1, 1]^3`. Its O(N^2) temporary work is unsuitable as a universal
large-system solution. A production implementation should use a GPU cell-list
or spatial-binning algorithm and be tested for nonperiodic, partially periodic,
small-cell, mixed-size, variable-cell, skin-cache, and active-compaction cases.
