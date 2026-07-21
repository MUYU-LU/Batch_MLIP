# Batched BFGS line-search integration

## Hypothesis

ASE-compatible inverse-Hessian and strong-Wolfe state can remain independent
per structure while all requested trial energies and gradients are evaluated in
one tensor batch.

## Correctness gates

- `QuasiNewton` and `BFGSLineSearch` resolve to the same batched optimizer.
- Fixed-cell B1 follows ASE `BFGSLineSearch` at float64 precision.
- A heterogeneous batch matches independent batched runs.
- Active compaction preserves final results while reducing graph evaluations.
- Frechet variable-cell B1 follows ASE when both use generalized-force
  convergence (`smax=None`).

## Scope

This experiment makes no performance claim. It records line-search model-call
counts because one accepted optimizer step can require multiple evaluations.
Active refill and real AtomBit/MACE benchmarks are follow-up gates.
