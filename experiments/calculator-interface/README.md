# Generic calculator interface

## Hypothesis

FIRE and MD can be MLIP-independent if all model-specific translation is
confined to a `BatchCalculator` adapter. A custom calculator must therefore run
single-point evaluation, FIRE, and NVE through the same public API without any
AtomBit dependency.

## Scope

- Add `BatchCalculator`, `ASECalculatorAdapter`, and structure-level functions.
- Preserve the existing `BatchedPotential` and low-level APIs.
- Do not change FIRE or MD numerical update equations.
- Make no speed claim for the sequential ASE compatibility adapter.

## Validation

Run the commands recorded in `experiment.yaml`. The full regression suite is
the numerical non-regression gate; the dedicated calculator tests exercise a
custom batched implementation and an ordinary ASE calculator.

## Result

- Baseline: 11 tests passed before the change.
- Final: 13 tests passed, including the NVE drift regression.
- Ruff: all checks passed.
- Batch versus individual: zero maximum energy and force error on the toy
  three-system input; no cross-system edges.
- Public API smoke test: all three FIRE relaxations converged and the same
  calculator completed a three-step NVE run.
- Performance: not benchmarked because model execution and numerical update
  paths did not change. The ASE adapter remains a sequential reference path.
