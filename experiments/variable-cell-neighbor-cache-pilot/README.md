# Variable-cell neighbor-cache pilot

## Change

The generic `AseGraphBatch` now separates candidate topology from the physical
model graph. CPU `matscipy` builds candidates through `cutoff + skin`; the GPU
filters current distances at the exact physical cutoff before model evaluation.
The model therefore does not perform message passing on skin-only edges.

Cache validity is decided per structure. For a fully periodic changing cell,
reference fractional coordinates separate affine deformation from non-affine
atomic motion. A cache is safe while

```text
(cutoff + 2 * max_non_affine_motion)
    * spectral_norm(inverse(current_cell) @ reference_cell)
<= cutoff + skin
```

For a fixed cell this reduces to the standard `max_motion <= skin / 2` rule.
Nonperiodic structures use that displacement rule directly. Changing partially
periodic cells retain the conservative rebuild fallback.

Dirty structures rebuild independently; clean structures retain their
candidate blocks and reference states. Neighbor tuples are canonicalized by
`(i, j, Sx, Sy, Sz)` so skin-zero and cached physical graphs have identical
accumulation order.

The original cell filter erased neighbor references after every variable-cell
step. That unconditional invalidation was removed; cache lifecycle now belongs
to the generic neighbor policy.

## Correctness tests

Tests cover:

- a pair entering the cutoff while remaining in cached candidates;
- safe affine compression without rebuilding;
- compression that starts beyond `cutoff + skin` and must rebuild;
- a heterogeneous batch where only one structure becomes invalid;
- exact physical-edge indices and periodic shifts versus a fresh skin-zero list.

The targeted fixed-cell, variable-cell, and BFGS suites pass.

## Production pilot

The first fixed-manifest structure from each 46-, 92-, 184-, and 276-atom group
was run for three deterministic variable-cell full-BFGS steps. AtomBit used
float32 inference and float64 optimizer state. The comparison was performed as
four independent B1 runs and as one heterogeneous B4 run.

| mode | skin-zero rebuilt systems | cached rebuilt systems | candidate/physical edges | exact outputs |
|:--|--:|--:|--:|:--|
| B1 total | 20 | 4 | 1.180x | yes |
| heterogeneous B4 | 20 | 4 | 1.180x | yes |

Cells, positions, energies, forces, stresses, convergence flags, and step counts
are bitwise identical. The cache builds only the four initial candidate lists
and reuses them for all subsequent evaluations.

For heterogeneous B4, neighbor-update time falls from 0.0468 to 0.0122 seconds
and one-shot total time falls from 0.2605 to 0.2048 seconds. These short timings
are screening evidence only, not a performance claim.

## Negative intermediate result

Before canonical edge ordering, the initial force difference was
`2.38e-6 eV/A`; full BFGS amplified it to approximately `1.27e-4 eV/A` after
three steps. Sorting both cached and skin-zero neighbor tuples removed the
difference completely. The failed intermediate result is retained in the raw
run directory.

## Scope and decision

This pilot validates the generic cache through the AtomBit adapter. MACE still
uses its native graph-construction fallback and is not included in the speed or
reuse result.

The correctness gate passes and neighbor work falls by more than half, so the
next experiment is a paired AtomBit B64 screen at 46 and 276 atoms. B128 and a
MACE cache adapter remain conditional on that result.
