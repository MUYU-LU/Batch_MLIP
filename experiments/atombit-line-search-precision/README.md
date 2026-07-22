# AtomBit BFGSLineSearch precision diagnostic

## Question

Does converting the complete AtomBit inference and optimization path from
float32 to float64 resolve the excessive strong-Wolfe trial evaluations seen
for H46 variable-cell optimization?

## Protocol

Stage 1 isolates `Aven12_1_367_z1.cif`, reported as system 13 in the failed
H46 R32 active-batch run. Sequential ASE and active batched B1 are each run in
float32 and float64. All optimizer state is float64 so the controlled variable
is model, graph, autograd, energy, force, and stress evaluation precision.

If float64 passes Stage 1, Stage 2 compares sequential ASE and active B32 on
the original frozen H46 R32 pool. Runs use one deterministic timing observation,
`fmax=0.05 eV/A`, at most 500 accepted steps, `alpha=10`, and `maxstep=0.2 A`.

Float64 evaluation does not recover information absent from checkpoint weights;
it tests whether higher-precision execution and reductions stabilize the line
search.

## Results

The dtype audit confirmed that every floating model parameter and buffer, the
positions and cells, and the energy, force, and stress outputs were float64.

| Method | Model dtype | Time (s) | Model evals | Evals/step | Steps | Converged | Final fmax (eV/A) |
|---|---|---:|---:|---:|---:|---|---:|
| ASE | float32 | 357.80 | 19,623 | 39.25 | 500 | no | 0.484 |
| ASE | float64 | 349.00 | 18,903 | 37.81 | 500 | no | 0.703 |
| Active B1 | float32 | 414.25 | 13,299 | 26.60 | 500 | no | 0.999 |
| Active B1 | float64 | 307.69 | 9,661 | 19.32 | 500 | no | 0.824 |

Float64 reduced active-B1 model evaluations by 27.4% and wall time by 25.7%,
but did not restore convergence. Both ASE and batched implementations exhausted
500 accepted steps well above `fmax=0.05 eV/A`. In contrast, standard batched
BFGS converged this structure in 139 steps at float32.

The precision hypothesis is therefore rejected: float32 rounding contributes
to excessive trial work, but is not the root cause. Stage 2 R32 timing was
skipped because its declared B1 convergence gate failed; timing a known
non-convergent method would not be a valid speed comparison.
