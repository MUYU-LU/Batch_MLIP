# Production batch scaling

## Hypothesis

True graph batching will increase production energy-plus-autograd-force throughput over sequential ASE inference. Speedup should grow with batch size until GPU memory pressure or kernel saturation limits scaling.

## Controls

- Production checkpoint `model_epoch_15.pt`, loaded strictly.
- Float32 inference with conservative autograd forces and a 6.0 A cutoff.
- Four exact atom counts from T2: 46, 92, 184, and 276.
- A persisted, hash-ranked pool of 64 structures per atom count.
- Every batch size processes the same 64 structures, chunked into batches of 1, 2, 4, 8, 16, 32, or 64.
- Identical structures for batched and sequential ASE paths.
- CUDA synchronization, warm-up, repeated trials, and peak-memory recording.

## Acceptance gates

- Batch/single residual energy passes `1e-5 eV` with `rtol=1e-6`, or the maximum absolute error is no greater than `5e-7 eV/atom` for large systems.
- Maximum batch/single force error no greater than `1e-4 eV/A` with `rtol=1e-5`.
- No cross-system graph edges.
- End-to-end speedup is reported separately from model-only speedup.
- OOM and validation failures remain in the machine-readable result.

## Results

All 28 atom-count/batch-size points passed validation and completed without OOM on one H100 80GB GPU. End-to-end speedup relative to sequential ASE energy-plus-autograd-force inference was:

| atoms | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 | batch 32 | batch 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 0.97x | 1.69x | 2.95x | 4.50x | 6.32x | 7.41x | 7.71x |
| 92 | 0.94x | 1.63x | 2.62x | 3.68x | 4.40x | 4.63x | 4.74x |
| 184 | 0.94x | 1.59x | 2.33x | 2.90x | 3.11x | 3.22x | 3.26x |
| 276 | 0.93x | 1.52x | 1.99x | 2.27x | 2.38x | 2.43x | 2.44x |

At batch 64, end-to-end throughput was 565.0, 329.6, 217.1, and 148.2 systems/s for 46, 92, 184, and 276 atoms. Peak allocated GPU memory was 4.66, 9.95, 19.09, and 25.27 GB, respectively.

The largest observed batch/single discrepancies were `5.72e-5 eV` total residual energy (`2.07e-7 eV/atom`) and `6.44e-6 eV/A` force. These satisfy the declared mixed float32 energy tolerance and force tolerance.

The hypothesis is supported for batch sizes of at least two. Batch size one is slightly slower than the ASE reference because the batch engine adds state-management overhead without exposing parallelism. Gains decrease as atom and edge counts grow.

## Commands

```bash
CUDA_VISIBLE_DEVICES='' /public/home/lmy/.conda/envs/lmy/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 /public/home/lmy/.conda/envs/lmy/bin/python benchmarks/benchmark_production.py --output runs/production_batch_scaling.json
/public/home/lmy/.conda/envs/lmy/bin/python -m ruff check atombit_batch src benchmarks/benchmark_production.py
/public/home/lmy/.conda/envs/lmy/bin/python -m compileall -q atombit_batch src benchmarks/benchmark_production.py
```

## Limitations

- Results cover one checkpoint, float32, one H100, fixed-cell P1 structures, and initial single-point energy/force evaluations.
- E0 is excluded from timing and numerical comparison because it is constant, coordinate-independent bookkeeping.
- The end-to-end batch path uses CPU matscipy neighbor lists; production trajectory performance will also depend on neighbor rebuild frequency and skin.
- This experiment does not measure full FIRE relaxation, MD integration, multi-GPU execution, or `torch.compile`.
- Atom count does not uniquely determine cost because cell density and edge count vary among structures.

## Next experiment

Evaluate edge-count-aware dynamic batching on full FIRE relaxations. Use the same 256 fixed structures, compare equal total structures and convergence tolerances, and record active-system compaction, neighbor rebuilds, total steps, failures, and final structures against sequential ASE relaxation.
