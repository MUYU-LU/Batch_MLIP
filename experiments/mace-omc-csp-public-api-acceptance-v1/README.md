# MACE OMC-CSP public API acceptance v1

This experiment is the release gate for the one-call `optimize_pool(...)`
interface. It uses the chemically untouched XULDUD test family: 2,048 signed,
unique OMC-CSP structures with no replication. The numerical contract is
MACE-OFF23-Small float64, batched BFGS float64, `FrechetCellFilter`,
`fmax=0.01 eV/A`, and at most 3,000 steps.

The runner verifies every source file and normalized structure against the
sealed manifest before execution. Acceptance requires exact output coverage
and ordering, finite endpoints, at least 99% convergence, the exact packaged
MACE/H100 capacity policy with zero probe forwards, peak reserved memory no
greater than 85% of an H100, and acknowledged shutdown from every active
worker. Relaxed structures and full endpoint hashes remain on the H100 host.

Status: planned.
