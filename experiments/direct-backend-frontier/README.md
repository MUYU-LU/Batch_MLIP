# Direct ASE/matscipy/auto frontier

## Question

This experiment directly measures common sequential ASE, native batching with
matscipy neighbors, and native batching with the adaptive `auto` backend. It
closes the previous measurement gap: separate ASE/batching and
CUDA/matscipy ratios are not multiplied.

All runs use one H100 per process, frozen T2 manifests, identical checkpoints,
one measured timing, synchronized CUDA, one CPU thread, and the selected batch
plus one adjacent frontier batch. Selection requires successful execution,
the task-specific validation gate, and less than 85% peak memory use.

## Selected frontier

| Task | Model | Direct speedup vs ASE | Gain vs matscipy |
|:--|:--|--:|--:|
| EVAL | AtomBit | 5.99-18.09x | 2.90-5.02x |
| EVAL | MACE-OFF-Small | 8.49-66.80x | 1.69-2.02x |
| NVE R32 | AtomBit | 3.98-10.62x | 1.01-1.07x |
| NVE R32 | MACE-OFF-Small | 6.09-16.27x | 1.00-1.03x |
| Variable-cell BFGS | AtomBit | 6.96-8.73x | 1.21-1.38x |
| Variable-cell BFGS | MACE-OFF-Small | 8.81-8.92x | 1.04-1.11x |
| Variable-cell FIRE | AtomBit | 3.62-6.37x | 1.03-1.70x |

The full 24-row selected table, including workload, batch, peak memory,
observed backend, and neighbor fraction, is in `results.md`. `results.csv` and
`results.json` retain all 120 rows, not only winners.

## Interpretation

EVAL is rebuild-dominated. `auto` selected CUDA dense at every selected point,
raising total throughput substantially rather than merely accelerating an
isolated neighbor microbenchmark.

NVE uses a 0.5 A skin. More than 90% of replica-steps reuse candidate graphs,
so CUDA construction affects only rebuild steps. The selected CUDA/matscipy
gain is consequently 0-7%; MACE H46 correctly selects matscipy. Auto can
observe both backends in one NVE run because large simultaneous rebuilds use
CUDA while small selective rebuilds fall back to matscipy.

Variable-cell optimization rebuilds frequently. CUDA provides a useful total
gain for AtomBit H276 FIRE and both AtomBit BFGS workloads, but only a modest
gain for MACE BFGS because model and optimizer work dominate. AtomBit H46 FIRE
selects B16. Both B32 backends disagreed with ASE on one convergence flag, so
the selector excluded them even though the discrepancy was independent of the
neighbor backend.

## Correctness

All EVAL points pass exact ordering and declared energy/force gates. Selected
EVAL maxima are `1.94e-7 eV/atom` and `1.08e-5 eV/A`.

All NVE points pass the real-model short-horizon parity gate and have finite
long-horizon endpoints. The selected maximum endpoint position and velocity
RMSDs are `2.68e-5 A` and `6.69e-6 A/fs`; long-horizon endpoint differences are
descriptive rather than trajectory-identity gates.

Every selected optimization point matches ASE convergence flags. FIRE endpoint
differences remain small: at most `0.114 meV/atom`, `0.00338 A` position RMSD,
and `0.00216 A` cell RMSD. Full BFGS amplifies microscopic model/batch
differences and can select another minimum; selected maxima reach
`8.08 meV/atom` and `0.444 A` position RMSD. These BFGS timings are valid
throughput measurements, but the endpoints are not claimed to be the same
minimum. The deterministic forced-step controls remain the algorithmic
equivalence evidence.

## Limitations

- Each frontier point has one measured timing, so differences below 2% are
  inconclusive and the values are screening results rather than uncertainty-
  qualified paper estimates.
- R256 NVE is not directly timed against sequential ASE in this stage.
- The H100-derived auto thresholds and batch frontier require recalibration on
  other GPU models.
- NVT, NPT, and production application workloads remain future measurements.
