# Optional variable-cell FIRE

## Hypothesis

A batched Frechet log-deformation filter should reproduce ASE
`FrechetCellFilter` plus FIRE while leaving `cell_filter=None` numerically
identical to the existing fixed-cell implementation.

## Scope

- Full-rank, fully periodic, right-handed cells.
- Hydrostatic or full anisotropic log strain and a symmetric component mask.
- Optional external pressure in GPa and separate force/stress convergence.
- No variable-cell active compaction or `FixAtoms` interaction yet.
- NPT remains an explicit unimplemented API slot.

## Validation

The native graph stress is checked against central strain finite differences.
Two different initial hydrostatic volumes are optimized in one batch and
compared with separate ASE references. A sheared anisotropic cell provides a
full Frechet transformation regression.

## Results

- Native graph stress finite-difference maximum error: `3.4e-12 eV/A^3`.
- Hydrostatic batch convergence steps: `[52, 80]`, identical to ASE.
- Anisotropic final cell maximum error against ASE: below `2e-12 A`.
- Positive `1 GPa` pressure: batch and ASE both converge in 61 steps.
- Production AtomBit/T2 batch-versus-single maximum errors: `3.34e-6 eV`,
  `3.81e-6 eV/A`, and `1.40e-8 eV/A^3`.
- Final suite: 18 passed; Ruff passed; package wheel built successfully.

No speed claim is made. A changed cell currently invalidates the CPU neighbor
list every step, and variable-cell active compaction remains future work.
