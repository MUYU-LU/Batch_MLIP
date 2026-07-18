# MACE-OFF-Small native batch adapter

## Scope

- Environment: `/public/home/lmy/.conda/envs/MACE_clean`.
- MACE Torch 0.3.14, MACE-OFF23-Small, float64, H100.
- Model: 694,320 parameters, 4.5 A cutoff, cached foundation checkpoint.
- Data: eight fixed 46-atom T2 manifest structures containing H/C/N/O.
- Single point: energy, forces, and full stress; B1/B2/B4/B8 versus MACE's
  official sequential ASE calculator.
- Optimization: four structures, three forced full Frechet BFGS steps, three
  timing repeats.
- MD: two-structure, one-step NVE API smoke test.

MACE-OFF is distributed under the Academic Software License and does not
permit commercial use.

## Results

| method | median seconds | speedup vs MACE ASE | max energy error | max force error | max stress error |
|:---|---:|---:|---:|---:|---:|
| MACE ASE | 0.15398 | 1.00x | - | - | - |
| Native B1 | 0.15417 | 1.00x | 5.09e-11 eV | 5.11e-15 eV/A | 2.78e-17 eV/A^3 |
| Native B2 | 0.08862 | 1.74x | 5.82e-11 eV | 5.33e-15 eV/A | 4.25e-17 eV/A^3 |
| Native B4 | 0.05575 | 2.76x | 5.82e-11 eV | 4.88e-15 eV/A | 4.51e-17 eV/A^3 |
| Native B8 | 0.03370 | 4.57x | 5.09e-11 eV | 6.22e-15 eV/A | 3.30e-17 eV/A^3 |

The first ASE timing sample included a one-time 2.43-second backend event;
the other samples were 0.15357 and 0.15398 seconds, so the reported median is
not controlled by that outlier.

Three-step variable-cell BFGS took 0.44105 seconds through common MACE ASE and
0.12972 seconds through the native B4 path, a 3.40x speedup. Maximum differences
were 8.00e-11 eV in energy, 8.43e-11 eV/A in force, 1.40e-12 eV/A^3 in stress,
7.11e-12 A position RMSD, and 7.12e-12 A cell RMSD. Step counts were identical.

The one-step NVE smoke test returned finite energy and temperature for both
systems.

## Design

`MACEBatchCalculator` implements the same `BatchCalculator` contract used by
AtomBit. It converts the common state back to input-ordered ASE structures,
uses MACE's official `config_from_atoms` and `AtomicData.from_config` graph
construction, collates all systems into one MACE PyG batch, and returns MACE's
direct energy, forces, and stress tensors.

MACE remains an optional dependency. `load_mace_off_batch()` loads the
foundation model, validates its cutoff and supported elements, selects its
head, and includes a CUDA-initialization workaround required by the supplied
MACE/PyTorch environment before importing legacy serialized MACE models.

## Limitation

This first adapter rebuilds MACE graphs on CPU and round-trips positions/cells
through ASE every call. The measured speedup therefore comes from native model
batching despite that overhead. The next MACE acceleration should construct or
update MACE `AtomicData` directly from the common tensor state and add a
validated neighbor-skin cache.

## Artifacts

- `results.json`: timing, numerical errors, BFGS comparison, and NVE smoke data.
- `benchmarks/validate_mace_off_adapter.py`: repeatable validation command.
