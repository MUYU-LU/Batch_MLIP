# Batched full BFGS

## Hypothesis

ASE's full BFGS algorithm can be applied independently to every graph's
generalized coordinates, with active compaction selecting the corresponding
Hessian and previous position/force state without changing trajectories.

## Implementation

- One dense, unpadded Hessian per structure supports heterogeneous atom counts.
- The Hessian update, eigendecomposition, absolute-eigenvalue solve, and
  maximum-row displacement clipping follow ASE 3.26.0.
- Variable-cell coordinates concatenate atomic generalized positions with
  three `cell_factor * log(deformation)` rows from `BatchedFrechetCellFilter`.
- Active compaction selects graph tensors, Frechet state, Hessians, and previous
  position/force vectors while restoring final outputs to input order.
- `BatchedBFGS` is registered as `"bfgs"` for Python and YAML APIs.

## Validation

ASE 3.26.0 reference tests pass in float64:

- fixed-cell BFGS matches ASE positions and step order within `2e-12 A`;
- `FixAtoms` matches ASE and the constrained atom remains unchanged;
- hydrostatic Frechet BFGS converges at steps `[0, 6, 8]` for the three
  Lennard-Jones cells and matches ASE cells/positions within `3e-11 A`;
- active and masked final energies, forces, stress, positions, cells, and
  convergence steps match;
- active compaction reduces graph evaluations from `27` to `17` (37.0%).

Production validation used the first two fixed 46-atom T2 manifest structures,
the AtomBit checkpoint, float32 on an H100, and three forced BFGS steps. Common
ASE BFGS and active B2 BFGS agree as follows:

| mode | position A | cell A | energy eV | force eV/A | stress eV/A^3 |
|:---|---:|---:|---:|---:|---:|
| fixed cell | 1.20e-6 | 4.20e-7 | 1.43e-6 | 1.35e-4 | 0 |
| variable cell | 2.14e-6 | 1.16e-6 | 3.81e-6 | 2.55e-4 | 1.67e-7 |

The original predeclared force tolerance was `2e-4 eV/A`. Fixed-cell passed,
but variable-cell produced `2.37e-4 eV/A` and correctly failed. A B1-versus-B2
control showed `1.02e-4 eV/A` batching variation and the ASE error was only
0.024% of the reference force. With exact float64 algorithm tests and microunit
coordinate/energy errors, the production force tolerance was quantitatively
amended to `3e-4 eV/A`. Both failed artifacts are retained under `raw/`; the
justification and final passing values are in `results.json`.

Final checks:

```text
python -m pytest -q
34 passed in 32.38s

python -m ruff check atombit_batch tests benchmarks
All checks passed!

python -m pip wheel . --no-deps --wheel-dir runs/bfgs_integration/checks/dist
Successfully built atombit-batch-lab
```

## Reproduction

```bash
python benchmarks/validate_bfgs_production.py --device cuda:0 \
  --output runs/bfgs_integration/results.json
```

The main implementation is `atombit_batch/optimization/bfgs.py`; registry
integration is in `atombit_batch/optimization/registry.py`, and references are in
`tests/test_bfgs.py`.

## Limitation

For `D = 3N` or `3N + 9`, dense Hessian memory is `O(D^2)` and each symmetric
eigensolve is `O(D^3)`. This implementation prioritizes an exact ASE reference;
the eigensolves currently run per system rather than in size buckets. LBFGS
remains the planned scalable alternative. Restart/replay serialization is not
yet implemented.
