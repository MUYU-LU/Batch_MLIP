# omc-csp-scheduler-v2-loader-tier

## Scope

This experiment retains Scheduler v1's signed memory plan, 14 execution
chunks, seven persistent GPU workers, BFGS, `FrechetCellFilter`, AtomBit,
neighbor policy, and input order. It changes only parent-side CIF parsing:

- one loader process per GPU is the frozen P512 baseline;
- two loader processes per GPU are the medium-pool candidate;
- four loader processes per GPU locate the CPU/storage saturation point.

## Evidence

The paired validation matrix contains six chemically separate P512 workloads
and 3,072 jobs. The held-out matrix contains three test P512 workloads and
1,536 jobs. Every execution uses one persistent sequence, one timing trial,
the same signed manifests and profiles, and seven H100 GPUs.

Two loaders improve validation makespan from 260.04 to 253.17 seconds (2.7%)
and held-out makespan from 100.49 to 94.81 seconds (6.0%). Four loaders take
253.13 seconds on validation, indistinguishable from two while consuming twice
the loader processes. Loader changes produce exact endpoint and step equality.

## Decision

Use a tiered offline rule:

- below P512 or below 3,000 atom records per active GPU: one loader per GPU;
- P512 and above with sufficient CPU capacity: two loaders per GPU;
- P2048 with at least 32,000 atom records per active GPU: four loaders per GPU.

This is an I/O policy, not a change to numerical scheduling.
