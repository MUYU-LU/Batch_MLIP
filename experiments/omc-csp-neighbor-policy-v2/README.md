# OMC-CSP neighbor policy v2

## Scope

This experiment calibrates automatic neighbor construction for AtomBit OMC-CSP
on an NVIDIA H100 80GB. It changes graph construction only; model arithmetic,
BFGS, Frechet cell coordinates, forces, stress, and convergence criteria are
unchanged.

The scheduling unit is the subset of resident structures whose candidate graph
must be rebuilt. Every timing includes backend selection, neighbor search, and
replacement of the selected edge blocks in the full resident graph.

## Data split

- Fit: GUFJOG, XATMOV, BOQWIN, XAFPAY, OBEQIX, rof-b, AXOSOW, HAMTIZ, WICZUF, rof-a.
- Validation: JAYDUI, NACJAF, rof-c, KONTIQ, PAHYON, OBEQUJ.
- Held-out test: UJIRIO, XAFQIH, BOQQUT, SOXLEX, XULDUD, WIDBAO.

Every family uses 64 evenly spaced unique CIFs. Rebuilt subsets are nested B1,
B2, B4, B8, B16, B32, and B64 at candidate cutoffs 6.0 and 6.5 A. Each backend
uses one warmup and three synchronized timings.

## Negative result

Moving the occupied-span estimator from CPU/NumPy to Torch did not improve the
complete operation. Across 84 initial points, the median baseline/new auto
rebuild-time ratio was 0.995x. The tensor implementation was therefore reverted;
its raw results remain under `results/tensor-selector`.

## Accepted rule

Cold graph construction retains the conservative pair-work and cell-estimator
fallback. Once a valid previous candidate graph exists, `auto` uses its measured
work without transferring positions to the CPU:

```text
E <= 10,000:
    cuda_cell only for mean volume/atom <= 10.5 A^3; otherwise matscipy
10,000 < E <= 35,000:
    cuda_cell when E/N > 62 or rebuilt systems <= 3; otherwise cuda_dense
E > 35,000:
    cuda_cell when E/sum(N_i^2) > 1.3; otherwise cuda_dense
```

Unsupported CUDA-cell geometry still falls back to CUDA dense and then
Matscipy/ASE. The previous graph is used only as a performance descriptor; the
new graph is reconstructed exactly from current positions and cells.

## Held-out result

All 168 held-out points matched Matscipy exactly in ordered directed edges and
integer shifts.

| Metric | Previous auto | Candidate-graph auto |
|:--|--:|--:|
| Exact fastest-backend matches | 104 / 168 | 147 / 168 |
| Median regret | 11.8% | 0.7% |
| Mean regret | 26.8% | 2.9% |
| P95 regret | 122.0% | 15.0% |
| Maximum regret | 220.4% | 68.8% |

## Integrated validation

AtomBit variable-cell BFGS and ten-step NVE were tested with Matscipy,
`cuda_dense`, `cuda_cell`, and `auto`. Replicated H46/H276 controls and unique
SOXLEX/BOQQUT candidate batches produced exactly identical positions, cells,
energies, forces, velocities, and NVE drift across backends.

On unique BOQQUT, auto matched CUDA-dense total time within noise. On unique
SOXLEX, auto remained 4-6% slower than explicit CUDA cell. This residual is
retained as a held-out limitation rather than adding a family-specific rule.

## Limitations

- The thresholds are calibrated for the recorded H100 software contract.
- Cold builds do not have a previous candidate graph and use the older fallback.
- Fixed-slot refill invalidates incoming slots, so the warm rule applies after
  their first rebuilt graph unless candidate-edge metadata is preserved later.
- The study covers fully periodic molecular crystals; partial and nonperiodic
  systems retain correctness fallbacks but are not performance-calibrated here.
