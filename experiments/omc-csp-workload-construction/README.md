# OMC-CSP Workload Construction

## Hypothesis

The existing CSP candidates can be frozen into exact, nested benchmark pools
without repeating structures. Pool cardinality, family composition, and
computational cost spread remain separate workload factors.

## Contract

- Source data are read without modifying or cleaning the CIF collection.
- Files containing `_exp_` are catalogued as references and excluded from
  optimization pools.
- Empty family directories remain visible in the construction index.
- Candidate ordering is a seeded SHA-256 rank over relative paths.
- Content and normalized-structure hashes reject duplicates within a pool.
- Every nonempty family receives nested `P64`/`P512`/`P2048` pools.
- Balanced all-family pools exercise inter-family wide-cost scheduling.
- Eligible OMC variable-cell BFGS proxy-cost bands receive balanced
  inter-family narrow-cost pools.
- Candidate edges are recorded at 6.0 and 6.5 Angstrom for the AtomBit
  cutoff and the current 0.5 Angstrom cache skin.

`P` denotes a pool of specific unique CIFs. It never denotes replication.
Resident batch size `B` remains an independent scheduler decision.

The exact CIF ordering and normalized-structure hashes define the reusable
structure workload. The v1 serialized field named `planning_cost` is retained
only for construction-hash provenance; it means the historical OMC
variable-cell BFGS proxy
`16*D^2 + 256*N + 64*E_candidate`, not a universal cost. New planning uses
separate model/task/policy sidecars and hardware-bound coefficients.

## Scope

This stage constructs data identities only. It does not load an MLIP, optimize
a structure, or claim a speedup.

## Commands

```bash
PYTHONPATH=. python benchmarks/build_omc_csp_workloads.py \
  --dataset /path/to/test_set \
  --output /path/to/omc_csp_workloads_v1 \
  --workers 24 \
  --pool-sizes 64 512 2048 \
  --no-csv

PYTHONPATH=. python benchmarks/validate_omc_csp_workloads.py \
  --workloads /path/to/omc_csp_workloads_v1
```
