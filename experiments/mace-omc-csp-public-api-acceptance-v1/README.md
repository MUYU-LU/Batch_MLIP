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

## Result

The clean acceptance run used source commit `ed6fe9a` and GPUs
`0,1,4,5,6,7`. All predefined gates passed.

| Metric | Result |
|:--|--:|
| Unique jobs returned in input order | 2,048/2,048 |
| Converged at 3,000 steps | 2,046/2,048 (99.902%) |
| Production execution | 461.098 s |
| `optimize_pool` including startup and shutdown | 515.236 s |
| Full script including verified CIF loading and output | 594.627 s |
| Production throughput | 4.442 structures/s |
| Peak allocated memory | 42.67 GB |
| Peak reserved memory | 47.14 GB (55.45% of H100) |
| Execution chunks | 12 |
| Worker-time maximum/mean | 1.177 |

The exact packaged MACE capacity policy was selected with zero probe systems
and zero probe model forwards. The allocator was expandable segments, the
inner mode was active compaction plus active drain, and the six workers pulled
complete chunks from the outer cost-ordered queue. Multi-GPU refill remained
disabled under the frozen scientific policy.

The two nonconverged systems are jobs `01499` and `01860`; their final maximum
forces are 0.16638 and 0.03600 eV/A, respectively. Every converged job satisfies
the requested force threshold, with a maximum of 0.0099996 eV/A. Converged-step
statistics are median 277, p95 853, p99 1,440, and maximum 2,840. A 99%
convergence gate was fixed before execution because nonconvergence of difficult
physical jobs is distinct from scheduler failure.

The endpoint digest is
`4389dcb1ef8b59cbe844c10662cde7afd387a3e9d37827675f829d195ed77afb`.
The persisted gzip artifact independently passes integrity checks and contains
2,048 frame headers and workload IDs.

## Diagnostics

Two attempts are retained remotely but excluded from acceptance. The first
completed optimization, then exposed a nullable-field bug in the experiment
reporter. The second launch accidentally overlapped two identical processes
through a shell-control error and one copy OOMed under combined MPS memory.
After terminating both process trees and verifying all selected GPUs at their
74 MB baseline, the accepted attempt ran as one foreground-controlled process.
Its peak reservation is therefore uncontaminated, and the duplicate-launch OOM
is not attributed to the capacity model.

## Decision

Accept and tag the public MACE OMC-CSP workflow for the exact documented
MACE-OFF23-Small/float64 BFGS/`FrechetCellFilter`/H100 contract. This validates
the plug-and-play in-memory API, policy selection, multi-GPU execution, result
reassembly, memory bound, and shutdown. It does not establish a MPS speedup for
XULDUD, select a different optimizer, or generalize the signed capacity model
to other checkpoints or GPUs.

The compact result is `results/acceptance-result.json`; the 35 MB raw schedule
and 5.8 MB relaxed structure archive remain at the remote artifact root named
in `experiment.yaml`.
