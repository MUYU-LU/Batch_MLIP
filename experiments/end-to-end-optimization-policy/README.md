# End-to-end optimization policy

## Decision

Stage 1 is complete; stage 2 is blocked by its predeclared gates. The automatic
engine beats four-worker ASE CUDA MPS in all 16 production comparisons and
also when planning plus first-worker startup are charged. It stays within the
conservative 85% memory bound after changing the representative-forward
calibration from allocated to reserved memory.

The stricter claim does not pass. Automatic production is within 5% of the
manual active-drain oracle in 14/16 cases, not 16/16. Separate R64 final-tensor
diagnostics pass a 1 meV/atom gate in 6/8 cases; both failures are AXOSOW full
BFGS, consistent with the previously documented local-minimum sensitivity of
long float32/GPU BFGS trajectories.

Geometric means across the 16 cases:

- automatic production/manual oracle: `1.063x`
- automatic production/MPS: `2.982x`
- automatic total API/MPS: `2.377x`
- median planning time: `0.566 s`
- median first-worker startup: `8.796 s`
- maximum conservative device-memory fraction: `82.61%`

## Full matrix

`Auto/manual` is the production throughput fraction. MPS columns are speedups
of automatic production and total API time, respectively. Memory is the
conservative worker-reserved plus parent-probe-reserved bound.

| Case | Chunks | Auto/manual | Prod/MPS | Total/MPS | Memory | Converged |
|:--|:--|--:|--:|--:|--:|--:|
| AXOSOW-R64-atombit-fire | 58,6 | 1.09x | 3.07x | 2.87x | 13.8% | 62/64 |
| AXOSOW-R64-atombit-bfgs | 64 | 1.01x | 2.65x | 2.07x | 13.9% | 64/64 |
| AXOSOW-R64-mace-fire | 56,8 | 1.05x | 1.69x | 1.15x | 7.9% | 64/64 |
| AXOSOW-R64-mace-bfgs | 64 | 0.97x | 2.83x | 1.12x | 10.4% | 64/64 |
| AXOSOW-R256-atombit-fire | 232,24 | 1.23x | 3.54x | 3.44x | 51.6% | 248/256 |
| AXOSOW-R256-atombit-bfgs | 256 | 1.03x | 3.40x | 3.03x | 54.0% | 256/256 |
| AXOSOW-R256-mace-fire | 224,32 | 1.08x | 3.62x | 3.31x | 37.2% | 256/256 |
| AXOSOW-R256-mace-bfgs | 256 | 0.88x | 3.37x | 2.73x | 39.0% | 256/256 |
| XAFPAY-R64-atombit-fire | 40,24 | 0.93x | 2.40x | 1.83x | 17.7% | 64/64 |
| XAFPAY-R64-atombit-bfgs | 64 | 1.02x | 3.98x | 3.11x | 24.7% | 64/64 |
| XAFPAY-R64-mace-fire | 64 | 1.09x | 1.64x | 1.23x | 25.4% | 64/64 |
| XAFPAY-R64-mace-bfgs | 64 | 1.05x | 3.70x | 2.44x | 26.5% | 64/64 |
| XAFPAY-R256-atombit-fire | 160,96 | 1.22x | 2.97x | 2.76x | 61.6% | 256/256 |
| XAFPAY-R256-atombit-bfgs | 162,94 | 1.12x | 4.73x | 4.42x | 65.1% | 256/256 |
| XAFPAY-R256-mace-fire | 215,41 | 1.24x | 2.30x | 2.12x | 82.6% | 256/256 |
| XAFPAY-R256-mace-bfgs | 187,69 | 1.07x | 3.64x | 3.28x | 75.8% | 256/256 |

The two automatic/manual misses are
`AXOSOW-R256-mace-bfgs` and `XAFPAY-R64-atombit-fire`. AtomBit FIRE on
AXOSOW has the same incomplete convergence count under automatic, manual, and
MPS execution, so it is an optimizer/workload outcome rather than omitted
work.

## Reproduction

```bash
bash experiments/end-to-end-optimization-policy/run_stage1.sh
bash experiments/end-to-end-optimization-policy/run_correctness.sh
python benchmarks/summarize_end_to_end_policy.py \
  --raw-dir experiments/end-to-end-optimization-policy/raw \
  --correctness-dir experiments/end-to-end-optimization-policy/raw/correctness \
  --output experiments/end-to-end-optimization-policy/results.json
```

Raw timing and correctness artifacts remain on the benchmark server and under
the git-ignored experiment `raw/` directory.
