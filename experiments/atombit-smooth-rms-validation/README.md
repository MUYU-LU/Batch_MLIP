# AtomBit smooth-RMS validation

## Scope

This experiment validates the two epoch-5 smooth-RMS fine-tuned checkpoints
against the cutoff discontinuity and BFGSLineSearch failure identified for the
original hard-degree checkpoint. It does not evaluate accuracy against an
independent DFT test set.

Both serialized configurations declare `degree_norm="smooth_rms"`. The fp32
checkpoint contains only float32 model state and the fp64 checkpoint only
float64 state. The loader constructs the model in the stored dtype before
loading, preventing silent fp64-to-fp32 rounding.

## Cutoff continuity

The same-geometry edge removal was repeated for three fixed structures at each
of H46, H92, H184, and H276.

| Checkpoint | Maximum energy effect | Maximum atom-force effect |
|---|---:|---:|
| smooth-RMS fp32 | 3.81e-6 eV | 1.67e-6 eV/A |
| smooth-RMS fp64 | 0.0 eV | 3.72e-15 eV/A |
| old hard degree, fp64 | 4.885e-3 eV | 5.945e-3 eV/A |

The residual fp32 values are reduction-order precision, not a degree jump. The
fp64 control is continuous to numerical precision across all 12 structures.

## Original failing optimizer case

`Aven12_1_367_z1.cif` was relaxed with a variable cell, `fmax=0.05 eV/A`,
`alpha=10`, and at most 500 accepted BFGSLineSearch steps.

| Precision | Method | Time (s) | Steps | Model evaluations | Converged |
|---|---|---:|---:|---:|---|
| fp32 | ASE | 8.80 | 98 | 449 | yes |
| fp32 | Active B1 | 7.07 | 98 | 176 | yes |
| fp64 | ASE | 8.82 | 105 | 466 | yes |
| fp64 | Active B1 | 7.04 | 105 | 181 | yes |

The active implementation is 1.245x faster than ASE for fp32 and 1.253x for
fp64 in this single observation. The fp64 endpoints agree to `4.53e-10 eV`,
`8.65e-9 A` maximum atomic-position difference, and `1.38e-8 A` maximum
cell-component difference. The fp32 endpoint differences are `5.72e-6 eV`,
`8.95e-4 A`, and `9.80e-5 A`, respectively.

The hard-degree active runs previously exhausted 500 steps without convergence,
using 13,299 evaluations in fp32 and 9,661 in float64 execution. Smooth RMS
therefore resolves the diagnosed optimizer failure on the target structure.

## Remaining validation

The architectural and optimizer gates pass. Before treating either checkpoint
as production-ready, evaluate held-out energy, force, and stress errors against
the original hard checkpoint and DFT labels, then test natural cutoff-crossing
frequency and NVE energy drift. Larger pooled BFGSLineSearch scaling is also a
separate performance experiment.
