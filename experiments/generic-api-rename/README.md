# Generic API Rename

This migration makes `batch_mlip` the canonical package and gives each MLIP
adapter an explicit calculator class. It does not change numerical kernels.

| Before 0.2 | Canonical 0.2 API |
|---|---|
| `atombit_batch` | `batch_mlip` |
| `BatchedPotential` | `AtomBitBatchCalculator` |
| `BatchedFrechetCellFilter` | `FrechetCellFilter` |
| `load_mace_off_batch(...)` | `MACEBatchCalculator.from_off(...)` |

All former imports remain supported by aliases and are covered by tests.
