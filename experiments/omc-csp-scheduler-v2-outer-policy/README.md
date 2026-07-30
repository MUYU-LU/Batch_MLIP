# omc-csp-scheduler-v2-outer-policy

## Scope

This experiment starts Scheduler v2 from the frozen
`omc-csp-scheduler-v1` tag. It changes only outer multi-GPU dispatch:

- `subdivide` retains the frozen v1 behavior;
- `preserve_resident` keeps memory-safe resident batches intact and starts
  only the GPU workers required to execute them.

It does not change AtomBit, neighbor construction, BFGS,
`FrechetCellFilter`, active compaction, or the refill implementation.

## Hypothesis

When v1 creates extra work-stealing tasks by splitting already-safe resident
batches, preserving those batches will improve useful work per active GPU and
GPU-seconds per job without changing scientific results or exceeding the
memory bound. It may lose makespan on strongly imbalanced pools; therefore it
remains an explicit candidate until paired H100 validation identifies the
selection boundary.

## Required gates

- exact job coverage and manifest order;
- identical convergence flags and same-plan endpoint records;
- peak reserved memory at most 85%;
- report both makespan and active-GPU-seconds per job;
- compare cold and persistent timing scopes separately;
- do not promote the candidate if paired persistent makespan regresses by more
  than 5%.

## Status

`preserve_resident` is rejected as the universal persistent-throughput
policy by the first paired seven-H100 sequence. The next bounded candidate is
`subdivide` with one execution chunk per GPU, which retains persistent workers
while measuring whether the second work-stealing wave is worth its smaller
resident batches.

Changing outer chunk depth is also rejected for automatic promotion because
the OMC-CSP endpoints are schedule-sensitive. The safe CIF-loader follow-up is
tracked separately in `omc-csp-scheduler-v2-loader-tier`.
