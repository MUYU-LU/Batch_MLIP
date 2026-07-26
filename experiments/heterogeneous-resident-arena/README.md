# Heterogeneous resident arena

This experiment tests an allocation mechanism, not a new scheduling policy.
Generic graph MLIPs require compact atom tensors, so heterogeneous slots cannot
contain padding or inactive holes. The candidate implementation therefore uses
two reusable compact storage banks: each refill copies selected survivor and
new-job segments into the inactive bank, then swaps banks. This prevents live
source overwrite while removing intermediate `select_systems` batches and
repeated `torch.cat` allocation.

The first integration target is full BFGS. Per-job Hessian histories are already
persistent and remain indexed by global job ID. FIRE will retain its existing
repack behavior until separately validated.

## Results

The candidate preserves output ordering and is bitwise identical to repack for
convergence flags, step counts, energies, forces, stresses, positions, and
cells in all three 256-job production comparisons.

| MLIP | workload | batch | storage | time (s) | systems/s | refill time (s) | events | allocated (GiB) | reserved (GiB) |
|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| AtomBit | ROFA-MIX-R256 | 128 | repack | 49.155 | 5.208 | 0.554 | 25 | 29.950 | 77.893 |
| AtomBit | ROFA-MIX-R256 | 128 | arena | 49.733 | 5.147 | 0.426 | 25 | 29.977 | 77.875 |
| AtomBit | MIX4-R256 | 64 | repack | 111.523 | 2.295 | 4.395 | 209 | 14.008 | 52.365 |
| AtomBit | MIX4-R256 | 64 | arena | 110.673 | 2.313 | 3.883 | 209 | 14.050 | 52.365 |
| MACE | MIX4-R256 | 64 | repack | 127.251 | 2.012 | 2.613 | 197 | 13.796 | 17.297 |
| MACE | MIX4-R256 | 64 | arena | 125.835 | 2.034 | 3.024 | 197 | 13.796 | 17.299 |

ROFA packing improves by 23.1%, but total runtime regresses by 1.2%. The more
refill-heavy MIX4 case improves packing by 11.6% and total runtime by 0.77%
(`1.008x`). Repack itself is only 3.94% of MIX4 wall time, so even eliminating
it entirely would cap speedup at `1.041x`.

MACE wall time is 1.13% lower with arena, but the target phase regresses by
15.8%. Its roughly 1.4 s apparent total gain comes from variation in
`graph.mace_atomic_data`, graph collation, and model-forward time. It is not
evidence of arena acceleration.

## Decision

The 30% pack-time and 5% end-to-end gates are not met. The arena remains an
explicit experimental BFGS option because it is correct and provides a small
measured AtomBit gain in the high-refill regime. It is not used by automatic
planning, and FIRE continues to reject it. The result rules out further generic
work on allocation reuse until a workload profile shows packing above 5% of
wall time.

The next generic optimization target remains model-independent neighbor-list
construction: a sparse GPU cell list can address the measured 25% neighbor
update contribution without changing optimizer mathematics. Persistent worker
services are separately useful only for repeated submissions because process
startup dominated one-wave tests.
