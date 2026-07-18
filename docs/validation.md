# Validation Guide

## Before using a trained model

Run the configured batch/single comparison:

```bash
atombit-batch validate configs/your_run.yaml
```

This checks one batched evaluation against the concatenation of individual evaluations. Repeat with representative periodic, nonperiodic, small, large, and chemically diverse structures.

## Recommended tolerances

Set tolerances according to precision and model operations rather than hiding discrepancies:

| Check | float64 starting point | float32 starting point |
|---|---:|---:|
| batch/single energy, absolute | 1e-9 eV | 1e-5 eV |
| batch/single force, absolute | 1e-8 eV/Å | 1e-4 eV/Å |
| finite-difference force | 1e-5 to 1e-4 | 1e-3 to 1e-2 |

Large systems may require mixed absolute/relative tolerances.

## Finite-difference forces

For atom coordinate `x_i`, compare autograd force to:

```text
F_i ≈ -[E(x_i+h)-E(x_i-h)]/(2h)
```

Use several `h` values, typically `1e-3` to `1e-5 Å` in float64, to separate truncation and roundoff errors. Rebuild neighbours consistently; do not place a pair exactly at the cutoff.

## Stress

For a periodic graph, compare the returned stress to central differences under symmetric strain. Verify sign and volume conventions against the intended downstream code. Nonperiodic stress is returned as NaN deliberately.

## Optimizer comparison

For a small dataset and identical potential:

1. Relax each structure with the existing ASE calculator and optimizer.
2. Relax the batch with the same force tolerance and broadly comparable FIRE parameters.
3. Compare final energies, force maxima, structures modulo symmetry, and convergence failures.

Identical trajectories are not expected because per-system FIRE parameters evolve independently.

## NVE validation

Use autograd forces first. Record:

- initial and final total energy;
- maximum absolute energy drift;
- linear drift slope versus simulated time;
- temperature behavior;
- center-of-mass momentum;
- results at several time steps.

Energy oscillation that shrinks quadratically with time step is expected for velocity Verlet. Monotonic drift may indicate inconsistent forces, a discontinuous neighbour/cutoff implementation, unit errors, or too-large time steps.

## NVT validation

Check the temperature distribution over a sufficiently long equilibrated trajectory, not only the final value. Small systems have large temperature fluctuations. Verify that the friction value is interpreted in fs⁻¹.

## Direct-force head

Compare direct and autograd forces on a held-out set. For NVE, test energy drift with both. A low force MAE does not guarantee a conservative vector field.
