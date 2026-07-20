# Task-aware static and NVE performance

## Hypothesis

The throughput-optimal resident batch depends on model, task, pool size, and
atom/edge distribution. Native batching should improve useful-work throughput
over ordinary ASE B1 without violating numerical or memory gates.

## Preflight findings

The ASE boundary originally copied velocity numbers directly even though ASE
uses its internal time unit and tensor-state MD uses Angstrom/fs. The boundary
now converts in both directions and preserves ASE kinetic energy on round trip.

AtomBit B1 and B8 initially exceeded the frozen total-energy tolerance because
E0 constants were accumulated in float32; forces agreed within
`3.82e-6 eV/Angstrom`. Accumulating coordinate-independent E0 in float64 reduces
the B8 error to `1.04e-7 eV/atom`, below the original `5e-7 eV/atom` gate.

Real-model 10-step NVE parity on two mixed structures passes:

| Model | max position error (A) | max velocity error (A/fs) |
|:--|--:|--:|
| AtomBit | 5.59e-6 | 2.05e-6 |
| MACE-OFF-Small | 7.05e-10 | 4.69e-10 |

## Measurement status

The performance matrix is decision-gated. EVAL runs first for all frozen
R32/R256 H46/H276/MIX workloads. Exact R32 NVE follows; exact R256 NVE points
are selected only after measured cost and peak memory are known. Screening uses
one run per point, so differences below 2% are inconclusive and no uncertainty-
qualified performance claim is made at this stage.
