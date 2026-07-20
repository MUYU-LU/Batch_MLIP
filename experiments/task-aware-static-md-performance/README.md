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

## Measurement protocol

All points use frozen manifests and ordinary sequential ASE calculator calls as
the B1 reference. EVAL measures ASE and native batches on both R32 and R256.
NVE measures 100 warmup plus 1000 timed Velocity-Verlet steps; R32 compares ASE
and native batches directly, while decision-gated R256 runs measure native
capacity only. Because every R256 manifest is an exact eightfold repeat of its
R32 manifest, R256 NVE speedups are derived from measured R32 ASE replica-step
throughput and are labeled as such. They are not R256 end-to-end measurements.

This is a one-run screening matrix. Differences below 2% are treated as
inconclusive. The selected policy is therefore the smallest resident batch
within 2% of the highest throughput that also keeps the larger of peak allocated
and peak reserved GPU memory below 85% of the H100's capacity.

## Results

The complete selected-point table is in `results.md`; all points and validation
fields are in `results.csv` and `results.json`.

Exact EVAL validation passes for both models on all workloads. H276 B256 is OOM
for both models. Selected EVAL speedups over measured same-workload ASE B1 range
from 2.08x to 5.79x for AtomBit and from 2.87x to 45.61x for MACE.

Exact R32 NVE results are:

| Model | Distribution | Selected B | Measured speedup | Peak reserved GB |
|:--|:--|--:|--:|--:|
| AtomBit | H46 | 32 | 10.64x | 4.24 |
| AtomBit | H276 | 16 | 4.19x | 45.69 |
| AtomBit | MIX | 32 | 5.94x | 7.94 |
| MACE-OFF-Small | H46 | 32 | 16.30x | 3.24 |
| MACE-OFF-Small | H276 | 32 | 5.90x | 16.23 |
| MACE-OFF-Small | MIX | 32 | 8.28x | 11.00 |

The R256 NVE capacity selections are:

| Model | Distribution | Selected B | Derived speedup | Peak reserved GB |
|:--|:--|--:|--:|--:|
| AtomBit | H46 | 128 | 12.21x | 22.95 |
| AtomBit | H276 | 16 | 4.22x | 45.69 |
| AtomBit | MIX | 32 | 6.05x | 49.09 |
| MACE-OFF-Small | H46 | 128 | 18.71x | 11.65 |
| MACE-OFF-Small | H276 | 64 | 6.14x | 37.74 |
| MACE-OFF-Small | MIX | 128 | 8.74x | 38.16 |

These speedups use measured R32 ASE throughput on exact repeated structures;
they are capacity-screening values, not measured R256 ASE wall-time ratios.

Every NVE endpoint comparison passes. Maximum endpoint RMSD is
`4.31e-5 A` for AtomBit and `2.44e-8 A` for MACE. Energy drift agrees with the
ASE trajectory at the reported precision; it is a property of the integrator,
timestep, model, and starting state rather than a batch/ASE discrepancy.

## Interpretation

There is no model-independent best batch size. Small structures sustain larger
resident batches, whereas AtomBit's H276 graph reaches the memory-safety limit
much earlier. A larger job pool improves scheduling opportunity but does not
make an unsafe resident batch desirable. For repeated dynamics, batching
amortizes Python and launch overhead across every step and gives larger gains
than one-shot evaluation in most workloads.

The edge metric also has to be model-specific: the frozen H46/H276 structures
average about 1,629/8,695 directed AtomBit edges but 859/4,942 directed MACE
edges because the models use different cutoffs. Atom count alone is therefore
not a transferable proxy for memory or optimal resident size.

With `skin=0.5 A`, only 6.0-8.2% of measured NVE replica-steps rebuild a
system's candidate graph. Cached graphs therefore avoid more than 91% of full
neighbor reconstructions while geometry, distances, cutoff weights, and forces
are still updated on every step.

The matrix supports a task-aware policy based on model, task, graph size,
available pool, measured throughput, and peak reserved memory. It does not yet
support uncertainty-qualified performance claims because each point has one
timing observation. Confirmation repeats should be restricted to the policy
frontier used in a paper, not applied to the entire screening matrix.
