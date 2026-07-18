# BFGS B1 numerical audit

## Question

The first scaling experiment reported large final-state differences between
native B1 and common ASE BFGS despite both converging. This audit tests whether
the BFGS update, Frechet forces, calculator gradients, precision, or GPU
nondeterminism causes the discrepancy.

## Calculator and algorithm isolation

For `Aven12_1_295_z1.cif`, native B1 and the ASE calculator agree initially to
`1.67e-6 eV/A` in forces and `9.31e-9 eV/A^3` in stress. On all 112 states of
an ASE trajectory for `Aven12_1_62_z1.cif`, the maximum calculator differences
are `3.81e-6 eV`, `2.44e-6 eV/A`, and `2.05e-8 eV/A^3`. There is no large
gradient or stress error.

Float64 full-BFGS/Frechet state did not restore the complete B1 pools, so
optimizer precision alone is not the cause. The feature remains optional;
production defaults continue to follow the calculator dtype.

## Nondeterminism

Without deterministic CUDA controls, repeated optimization of the same
structure produces multiple valid outcomes:

| path | repeated step counts | final energy range (eV) |
|:---|:---|---:|
| native B1 | 110, 111, and 185 | -4.32378 to -3.94784 |
| common ASE BFGS | 110, 111, and 185 | -4.32377 to -3.94835 |

With `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`, three native repetitions are
bitwise identical and three ASE repetitions are bitwise identical. Both paths
then converge in 111 steps; final energies differ by only `6.2e-6 eV`.

Thus, the earlier benchmark incorrectly treated one nondeterministic common
ASE trajectory as a unique scientific reference. Common ASE itself can select
the alternative minimum.

## Deterministic full-pool check

Determinism makes each implementation reproducible but does not force native
and ASE paths to select the same basin for every structure. Tiny deterministic
differences in reduction order, float32 model inputs, and the SciPy-versus-
Torch Frechet implementation can still be amplified by full BFGS.

| atoms | ASE steps | native B1 steps | ASE time (s) | native time (s) | worst energy difference (meV/atom) |
|---:|---:|---:|---:|---:|---:|
| 46 | 3830 | 3843 | 132.88 | 134.35 | 1.178 |
| 92 | 3886 | 3897 | 152.21 | 152.82 | 6.158 |
| 184 | 3381 | 3407 | 222.81 | 133.33 | 3.625 |
| 276 | 2335 | 2388 | 298.97 | 109.86 | 4.507 |

Every convergence flag still matches. These final-minimum differences must not
be interpreted as direct force errors.

## Same-calculator control

`ASECalculatorAdapter` removes the native calculator path and supplies the
identical AtomBit ASE calculator to batched BFGS. On five worst-case controls:

- two match ASE cells and positions within `1e-10 A` and have identical steps;
- one matches the basin within `0.0012 A` position RMSD and identical steps;
- two select a different basin after small SciPy/Torch Frechet numerical
  differences, with position RMSD `0.045-0.090 A`.

Exact float64 LJ and quadratic references remain matched within `3e-11 A`.
The BFGS and Frechet equations are therefore validated, but long nonconvex
trajectory identity is not a stable correctness criterion.

## Conclusion

FIRE also uses gradients, but it carries velocity rather than an accumulated
dense curvature estimate. BFGS repeatedly divides by curvature inferred from
force differences, so micro-level reduction noise can rotate later steps and
select another basin.

Future validation must use deterministic fixed-step comparisons, calculator
force/stress errors on identical structures, convergence, and final energy
distributions over repeated runs. Exact final-coordinate equality is only
appropriate when both paths use the identical calculator and numerical filter.
The previous BFGS scaling timings remain measurements, but its single-run
final-state interpretation is superseded by this audit.

Verification: `40 passed in 30.07s`; Ruff clean. Raw deterministic,
float64-negative, failed-worker, and trajectory artifacts are retained beside
`results.json`.
