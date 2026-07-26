# NVIDIA ALCHEMI neighbor backend

This experiment evaluates an optional maintained Warp implementation before
adding another custom neighbor kernel. The only primary variable is neighbor
construction. Model, optimizer, cutoff, precision, structures, requested steps,
and deterministic settings remain fixed.

The candidate must reproduce canonical directed edge and integer-shift tensors.
It enters production policy only if both AtomBit and MACE show a synchronized
end-to-end improvement of at least 2% in a representative regime without more
than 10% additional peak allocated GPU memory. Negative and unsupported
geometry results are retained.

## Result

The adapter matches Matscipy exactly in directed edge order and integer shifts
at all 14 neighbor-only points. Targeted CUDA tests also cover selective
triclinic batches, multiple periodic images, and both ALCHEMI naive and cell
dispatch.

At matched batch sizes the native `cuda_cell` implementation is faster at every
neighbor-only point. ALCHEMI requires much less neighbor scratch memory for
large systems: at 6 A, H2208 B2 uses 57 MB peak allocated memory versus 227 MB
for `cuda_cell`, but takes 5.09 ms versus 4.65 ms.

| Production point | `cuda_cell` (s) | ALCHEMI (s) | ALCHEMI speedup | Exact state |
|:--|--:|--:|--:|:--|
| AtomBit H2208 B2 EVAL | 0.06835 | 0.06842 | 0.999x | yes |
| MACE H2208 B2 EVAL | 0.06395 | 0.06331 | 1.010x | yes |
| AtomBit H368 B4 BFGS, 3 steps | 0.21064 | 0.21308 | 0.989x | yes |
| MACE H368 B4 BFGS, 3 steps | 0.18607 | 0.19435 | 0.957x | yes |
| AtomBit H1242 B1 NVE, 10 steps | 0.35998 | 0.35482 | 1.015x | yes |
| MACE H1242 B1 NVE, 10 steps | 0.27574 | 0.28852 | 0.956x | yes |

AtomBit and MACE NVE drift is identical between backends. End-to-end peak
allocated memory is also effectively identical because model intermediates,
not neighbor scratch, set the production peak.

## Decision

The performance hypothesis fails. `nvalchemi` remains an optional explicit
experimental backend and is not selected by `auto`. Its maintained Warp API and
low neighbor-only scratch footprint remain useful controls for future
memory-frontier work, but the current native cell list remains the production
choice.
