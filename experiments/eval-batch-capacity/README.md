# EVAL resident-batch capacity

## Question

How far can the resident batch be increased for homogeneous 46-atom and
276-atom structures on one H100 80GB, and where does full-model throughput
stop improving before out-of-memory?

The frozen R32 H46/H276 manifest order is repeated cyclically to the requested
resident batch. Every point runs in a fresh process with one B1 warmup, two
same-shape warmups, one synchronized measurement, and `neighbor_backend=auto`.
The two same-shape warmups are required to exclude MACE lazy-kernel setup.

## Recommended frontier

The production frontier uses the fastest validated point below the strict 85%
reserved-memory gate.

| Model | Distribution | Recommended B | structures/s | atoms/s | Allocated GB | Reserved GB | Memory | Neighbor fraction |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| AtomBit | H46 | 768 | 1074.1 | 49,409 | 54.35 | 62.89 | 74.0% | 16.7% |
| AtomBit | H276 | 160 | 262.6 | 72,483 | 60.50 | 67.21 | 79.1% | 14.2% |
| MACE-OFF-Small | H46 | 768 | 1190.5 | 54,761 | 53.94 | 58.67 | 69.0% | 19.1% |
| MACE-OFF-Small | H276 | 128 | 282.1 | 77,864 | 53.55 | 58.26 | 68.5% | 15.4% |

## H46 results

Both models execute B1024, but it is above 90% reserved memory. AtomBit is
already slower at B1024 than B768; MACE gains only 1.8% throughput for 33%
more resident jobs and crosses the safety gate.

| Model | B | Status | structures/s | Allocated / reserved GB | Memory | Neighbor fraction |
|:--|--:|:--|--:|:--|--:|--:|
| AtomBit | 256 | pass | 1016.2 | 18.17 / 20.02 | 23.6% | 18.6% |
| AtomBit | 512 | pass | 1042.9 | 36.26 / 40.05 | 47.1% | 17.2% |
| AtomBit | 768 | pass | 1074.1 | 54.35 / 62.89 | 74.0% | 16.7% |
| AtomBit | 1024 | pass, unsafe | 987.1 | 72.45 / 79.97 | 94.1% | 23.7% |
| MACE-OFF-Small | 256 | pass | 1136.6 | 18.04 / 19.63 | 23.1% | 20.4% |
| MACE-OFF-Small | 512 | pass | 1177.5 | 35.98 / 39.15 | 46.1% | 19.4% |
| MACE-OFF-Small | 768 | pass | 1190.5 | 53.94 / 58.67 | 69.0% | 19.1% |
| MACE-OFF-Small | 1024 | pass, unsafe | 1211.6 | 71.89 / 78.19 | 92.0% | 18.3% |

## H276 results

AtomBit reaches a throughput plateau at B128-B192. B192 executes but reserves
95.0% of memory; B224 is OOM. MACE is flat at B128-B160, but B160 is just over
the strict memory gate and B192 is OOM.

| Model | B | Status | structures/s | Allocated / reserved GB | Memory | Neighbor fraction |
|:--|--:|:--|--:|:--|--:|--:|
| AtomBit | 32 | pass | 243.2 | 12.17 / 13.69 | 16.1% | 16.1% |
| AtomBit | 64 | pass | 256.4 | 24.25 / 26.94 | 31.7% | 14.7% |
| AtomBit | 96 | pass | 255.9 | 36.33 / 40.16 | 47.2% | 15.6% |
| AtomBit | 128 | pass | 260.7 | 48.43 / 54.02 | 63.5% | 14.5% |
| AtomBit | 160 | pass | 262.6 | 60.50 / 67.21 | 79.1% | 14.2% |
| AtomBit | 192 | pass, unsafe | 262.3 | 72.59 / 80.73 | 95.0% | 14.6% |
| AtomBit | 224 | OOM | - | - | - | - |
| AtomBit | 256 | OOM | - | - | - | - |
| MACE-OFF-Small | 32 | pass | 258.5 | 13.45 / 14.84 | 17.5% | 17.2% |
| MACE-OFF-Small | 64 | pass | 275.1 | 26.81 / 29.53 | 34.7% | 15.6% |
| MACE-OFF-Small | 96 | pass | 274.7 | 40.18 / 44.00 | 51.8% | 16.8% |
| MACE-OFF-Small | 128 | pass | 282.1 | 53.55 / 58.26 | 68.5% | 15.4% |
| MACE-OFF-Small | 160 | pass, unsafe | 282.0 | 66.91 / 72.53 | 85.3% | 15.1% |
| MACE-OFF-Small | 192 | OOM | - | - | - | - |
| MACE-OFF-Small | 224 | OOM | - | - | - | - |
| MACE-OFF-Small | 256 | OOM | - | - | - | - |

## Correctness and interpretation

All 19 executable points use `cuda_dense` and pass batch-versus-B1 validation.
AtomBit maxima are `2.08e-7 eV/atom` and `8.35e-6 eV/A`; MACE maxima are
`1.27e-12 eV/atom` and `1.07e-14 eV/A`.

The CUDA neighbor temporary workspace is not the capacity limiter. Full-model
activations and force autograd dominate, causing approximately linear memory
growth with resident batch size. Consequently, maximizing occupied memory is
not the correct policy: the recommended point is the throughput plateau below
the safety gate, not the largest batch that happens to execute.

This is a one-measurement capacity screen. Raw JSON and logs, including the
superseded warmup-protocol screens, are retained under
`runs/eval_batch_capacity/` on the benchmark server. Complete machine-readable
results are in `results.csv`.
