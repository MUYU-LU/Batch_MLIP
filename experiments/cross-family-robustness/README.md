# Cross-family robustness

This experiment replaces single-chemistry scaling claims with fixed workloads
selected from the supplied multi-family CSP test set.

Each positive family first contributes 256 deterministic candidates. Thirty-two
unique structures are then selected uniformly across their directed 6 A
edge-per-atom ranking and repeated eight times to form a matched 256-job pool.
`rof-a` is additionally balanced across 74, 148, 222, and 296 atoms before
density stratification.

The positive controls cover small, heavy-element, dense, anisotropic, large,
porous, and heterogeneous workloads. XIFZOF is an applicability control because
it contains Si; it cannot support an AtomBit performance claim unless a
checkpoint explicitly declaring Si support is supplied.

The experiment advances to optimization timing only after finite prediction,
B1 equivalence, and batch-versus-single equivalence checks for both AtomBit and
MACE-OFF-Small. Performance comparisons use identical ordered structures,
variable-cell BFGS, warm-up, synchronized CUDA timing, and recorded peak memory.

## Compatibility result

Both models pass finite prediction and mixed-size batch-versus-B1 equivalence
on the low-, median-, and high-edge representatives from all six positive
families. AtomBit's maximum energy, force, and stress differences are
`2.29e-5 eV`, `3.81e-6 eV/A`, and `1.77e-8 eV/A^3`. MACE-OFF-Small differences
are `1.75e-10 eV`, `1.18e-14 eV/A`, and `5.55e-17 eV/A^3`.

Both checkpoint element tables exclude Si. XIFZOF is therefore excluded from
performance and optimizer conclusions rather than being treated as a batching
failure.

## Reduced timing matrix

The full optimization comparison advances four workloads for both models:

- `GUFJOG44`: small-structure throughput and edge-density variation.
- `XATMOV88`: medium structures with the widest positive-control edge range.
- `OBEQIX220`: large dense full-BFGS Hessians and memory capacity.
- `ROFA-MIX`: heterogeneous 74/148/222/296-atom porous scheduling.

`SOXLEX48` remains a heavy-halogen correctness control because its batching
shape is redundant with `GUFJOG44`. `ROFB296` remains a large-capacity control
and advances only if `OBEQIX220` and `ROFA-MIX` give conflicting memory-policy
conclusions. The cross-family R192 pool is reserved for the final bucketing
confirmation after homogeneous capacity is measured.
