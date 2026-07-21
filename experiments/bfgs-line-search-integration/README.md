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

## Results

The analytic float64 gates pass on ASE 3.26 and 3.27. Fixed-cell B1 agrees with
ASE to `2e-12 A`; the four-step Frechet trajectory agrees within
`6.3e-12 A`. Heterogeneous batch-versus-independent execution and active
compaction are identical at the declared tensor tolerances.

Real-model forced-step validation used H46 structures, three accepted steps,
and float64 optimizer state:

| Model/path | Control | Result | Maximum diagnostic |
|:--|:--|:--|:--|
| AtomBit B1 fixed cell | ASE | pass | force `2.82e-4 eV/A` |
| AtomBit B1 variable cell | ASE | pass | stress `7.54e-8 eV/A^3` |
| AtomBit B2 fixed cell | ASE and B1 | pass | force `1.71e-4 eV/A` |
| AtomBit B2 variable cell | B1 | pass | stress `1.10e-7 eV/A^3` |
| AtomBit B2 variable cell | ASE | old direct gate missed | stress `2.72e-7` versus `2.0e-7 eV/A^3` |
| MACE-OFF-Small B1 variable cell | ASE | pass | existing MACE float64 gates |

For AtomBit B1, B2, and ASE, positions, cells, energies, forces, accepted-step
counts, and convergence flags pass. The B2-to-ASE stress excess is retained as
a negative gate rather than hidden by increasing the tolerance: ASE-to-B1 and
B1-to-B2 pass independently, so the direct excess is accumulated float32 graph
reduction and trajectory variation, not a B1 optimizer-equation mismatch.

The three-step AtomBit paths require seven model evaluations: one initial call
and six trial calls, or two trial evaluations per accepted step. This confirms
why BFGSLineSearch throughput cannot be inferred from accepted-step counts.

Raw records remain on the benchmark server under
`runs/bfgs_line_search_integration/`:

- `atombit_h46_b1_fixed3.json`: `29d8fdd240eaa7bf211000efa26c1d11225853c2a40c6576a33a401e6e8f34e9`
- `atombit_h46_b2_fixed3.json`: `559a7f04f5abfd17bbbdb7634b8d2b138a498106864e8817397b399e36ae773b`

## Decision

Retain the optimizer as an explicit production option under `quasinewton` and
`bfgslinesearch`. Do not make it the default and do not claim a speed advantage
until a direct ASE/BFGS/BFGSLineSearch throughput experiment is complete.
Active refill remains a separate follow-up because admission during nested
line searches needs a line-search-aware scheduler.
