# Variable-cell scaling to batch 32

## Hypothesis

Native batching and variable-cell active compaction should reduce end-to-end
wall time relative to sequential ASE `FIRE(FrechetCellFilter)` while preserving
ASE convergence decisions and final states within declared tolerances.

## Setup

- Model: production AtomBit checkpoint, float32, uncompiled.
- Hardware: NVIDIA H100 80 GB; one process and one GPU per benchmark worker.
- Data: the same 32 unique fixed-manifest T2 structures for each atom-count
  group (46, 92, 184, and 276 atoms); no replication or synthesis.
- Methods: common sequential ASE, masked batched Frechet-FIRE, and
  active-compacted batched Frechet-FIRE.
- Cell filter: full anisotropic Frechet/log-deformation filter at zero pressure.
- Optimizer: `fmax=0.05 eV/A`, ASE generalized filter-force convergence,
  500-step cap, `dt=0.1`, `dtmax=1.0`, and `maxstep=0.2 A`.
- Potential: 6.0 A cutoff, zero skin, autograd forces and stress.
- Timing: CUDA synchronized, one warm-up, three batch repeats. ASE is one
  sequential full-pool timing, so no ASE uncertainty estimate is available.

## Results

End-to-end time in seconds for the 32-structure pool (batch values are medians):

| atoms | method | B1 | B2 | B4 | B8 | B16 | B32 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 46 | masked | 99.37 | 73.08 | 60.18 | 54.07 | 41.57 | 33.13 |
| 46 | active | 96.87 | 69.85 | 52.42 | 40.66 | 26.49 | 16.97 |
| 92 | masked | 99.50 | 76.17 | 67.04 | 62.74 | 59.83 | 54.14 |
| 92 | active | 107.63 | 78.05 | 57.58 | 42.22 | 29.99 | 20.85 |
| 184 | masked | 82.67 | 72.10 | 71.42 | 67.33 | 61.76 | 82.45 |
| 184 | active | 82.54 | 65.24 | 51.56 | 38.42 | 27.17 | 24.33 |
| 276 | masked | 54.76 | 37.11 | 30.85 | 28.36 | 26.89 | 28.54 |
| 276 | active | 49.77 | 33.94 | 25.06 | 19.53 | 16.58 | 15.40 |

Direct B32 comparison:

| atoms | ASE s | masked s | active s | masked/ASE | active/ASE | active/masked | ASE converged | active graph reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 117.39 | 33.13 | 16.97 | 3.54x | 6.92x | 1.95x | 30/32 | 69.8% |
| 92 | 123.08 | 54.14 | 20.85 | 2.27x | 5.90x | 2.60x | 28/32 | 69.8% |
| 184 | 96.08 | 82.45 | 24.33 | 1.17x | 3.95x | 3.39x | 29/32 | 76.4% |
| 276 | 57.17 | 28.54 | 15.40 | 2.00x | 3.71x | 1.85x | 32/32 | 45.7% |

B32 active timing ranges across the three repeats were 16.92-16.99, 20.69-20.95,
24.26-24.43, and 15.40-15.90 seconds in increasing atom-count order. Peak
allocated GPU memory was 2.18, 4.68, 9.00, and 11.35 GiB. Active compaction
reduced graph evaluations from masked counts of 16,032, 16,032, 16,032, and
4,000 to 4,844, 4,841, 3,780, and 2,173, respectively. Peak memory is not
reduced because every active run starts with the full B32 batch.

## Correctness

At B32, masked and active match ASE's per-structure convergence flags in all
four groups. The methods agree on the naturally incomplete cases: 2/32,
4/32, and 3/32 structures reach the 500-step cap for 46, 92, and 184 atoms;
all 276-atom structures converge.

The strict final-state gate requires at most 0.1 meV/atom energy error, 0.03
eV/A final maximum-force error, 0.02 A position/cell RMSD, 0.01 eV/A^3 stress
error, and 25 steps difference. At B32, active passes all gates for 184 atoms.
The other active groups fail the strict energy gate with worst errors of 0.271,
0.373, and 0.242 meV/atom for 46, 92, and 276 atoms. The 92-atom group also
has a 0.198 eV/A maximum-force difference, dominated by a structure that both
methods classify as nonconverged at the step cap. Across all B32 active groups,
the worst position RMSD is 0.0147 A, cell RMSD is 0.00871 A, stress error is
0.000526 eV/A^3, and step-count difference is 13. These are float32 FIRE
trajectory/stopping-threshold differences, not mismatched convergence flags;
the raw final states remain available for audit.

Masked B32 also matches every ASE convergence flag but fails the 0.1
meV/atom energy gate in all four groups; its worst energy error is 0.566
meV/atom. Its worst force difference is likewise in the capped 92-atom group.

Verification:

```text
python -m pytest -q
19 passed in 32.19s

python -m ruff check atombit_batch tests benchmarks
All checks passed!

python -m pip wheel . --no-deps --wheel-dir runs/variable_cell_scaling32_checks/dist
Successfully built atombit-batch-lab
```

## Reproduction

```bash
python benchmarks/benchmark_variable_cell_scaling.py \
  --method METHOD --atom-count ATOMS --pool-size 32 \
  --batch-sizes 1,2,4,8,16,32 --repeats 3 --device cuda:0 \
  --output runs/variable_cell_scaling32_raw/METHOD_atomsATOMS.json

python benchmarks/summarize_variable_cell_scaling.py \
  --raw-dir runs/variable_cell_scaling32_raw \
  --output runs/variable_cell_scaling32_summary.json
```

`raw/` contains the 12 machine-readable benchmark artifacts and `results.json`
contains the merged results and validation. The source packet hash, fixed
manifest, parameters, and exact worker command are in `experiment.yaml`.

## Limitations and next experiment

ASE has only one timing sample. Model-forward, CPU neighbor-list, and
host-to-device times were not isolated, so only synchronized end-to-end
relaxation speedups are claimed. The current variable-cell path supports full
3D periodic, right-handed cells and not `FixAtoms` combined with changing cells.
The next performance experiment should replace the per-step CPU ASE neighbor
builder with a GPU-native PBC cell-list builder, then add edge-count-aware
bucketing/autobatching around the existing calculator-like relaxation API.
