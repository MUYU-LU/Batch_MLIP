# Active-batch compaction

## Hypothesis

Removing converged systems from subsequent AtomBit forward calls will eliminate
the measured graph-evaluation waste and improve large-batch FIRE wall time while
preserving convergence and final-structure validation against ASE FIRE.

## Method

- Production checkpoint: `../AtomBit-OMC-s/model_epoch_15.pt`.
- Fixed manifest: `benchmarks/t2_fixed_samples.json`, first 16 samples per atom group.
- Atom groups: 46, 92, 184, and 276 atoms.
- Batch sizes: 1, 2, 4, 8, and 16; three synchronized timing repeats.
- FIRE: `fmax=0.05 eV/A`, `max_steps=500`, fixed cells, float32 autograd forces.
- Eight workers split small and large batch sizes by atom group. Each worker ran
  masked and active modes on the same H100.

## Results

Active compaction made actual graph evaluations equal useful graph evaluations;
the uncompacted counterfactual and paired median timings were:

| atoms | B | avoided graph work | speedup vs masked |
|---:|---:|---:|---:|
| 46 | 16 | 59.2% | 1.43x |
| 92 | 16 | 68.7% | 2.08x |
| 184 | 16 | 52.1% | 1.82x |
| 276 | 16 | 37.4% | 1.52x |

At `B8`, speedups were 1.19x, 1.71x, 1.46x, and 1.40x in the
same atom-count order. `B1` avoided no graph work and stayed within 1-11% of
the masked timings, as expected.

All active candidates converged. Fifteen of 20 independently evaluated points
passed every ASE final-state gate in the vectorized matrix. The five failures
were threshold-branch variations: 92/B2, 92/B16, and 276/B1-B4. In particular,
276/B1 has no compaction but exhibited the same alternate minimum as 276/B2-B4,
and a targeted 92/B16 revalidation passed. Earlier paired runs also passed the
276 points. These raw failures are retained rather than discarded.

## Conclusion

The hypothesis is supported for computational work and wall time. Active
compaction eliminates the measured inactive-graph work and provides its largest
benefit when relaxation lengths are heterogeneous and batches are large.
CUDA scatter-order sensitivity near the FIRE threshold remains a scientific
reproducibility limitation independent of compaction.

## Artifacts

- `runs/active_batch_compaction_summary.json`: merged paired metrics.
- `runs/compaction_paired_*_masked.json`: masked timing shards.
- `runs/compaction_paired_*_active_vectorized.json`: final active timing shards.
- `runs/active_compaction_92_b16_revalidation.json`: passing targeted rerun.
- `runs/compaction_paired_*_active.json`: retained pre-vectorization results.
