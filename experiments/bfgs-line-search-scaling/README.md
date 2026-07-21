# BFGS line-search scaling

## Question

For full variable-cell relaxation, is batched ASE `QuasiNewton`/
`BFGSLineSearch` faster than sequential ASE, and does it outperform the current
batched standard BFGS?

## Matrix

The frozen R32 H46 and H276 pools are run with AtomBit and MACE-OFF-Small.
Every model/workload pair compares sequential ASE BFGSLineSearch, active B32
BFGSLineSearch, and active B32 standard BFGS. The convergence target is
`fmax=0.05 eV/A` on ASE Frechet generalized forces with at most 500 accepted
steps. Standard BFGS uses `alpha=70`; BFGSLineSearch uses `alpha=10`.

One synchronized timing observation is recorded per point, as requested for
the project screening stages. Results below 2% are inconclusive. Every timing
must also report model evaluations, summed accepted steps, convergence count,
and peak allocated GPU memory.
