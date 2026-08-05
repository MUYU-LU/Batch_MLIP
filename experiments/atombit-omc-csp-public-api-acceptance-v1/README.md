# AtomBit OMC-CSP public API acceptance v1

This is the release gate for direct use of the frozen AtomBit OMC-CSP path.
It invokes `AtomBitBatchCalculator.from_checkpoint(...)` and
`optimize_pool(...)` rather than benchmark-only loaders or manifest execution
internals.

The small gate uses the signed, unique XULDUD P64 test workload on one H100.
The large gate uses the nested signed XULDUD P2048 workload on eight H100s once
all requested devices are externally idle. Both retain the AtomBit smooth-RMS
fp32, float64 BFGS state, `FrechetCellFilter`, 6.0 A cutoff, 0.5 A skin,
maximum resident batch size 256, `fmax=0.01 eV/A`, and 3,000-step production
contract.

Acceptance requires exact packaged-policy selection with no memory-probe model
forwards, immutable output coverage and ordering, finite endpoints, at least
99% convergence, predicted chunks within the 85% planning budget, observed
reserved memory within the frozen 91% runtime high-water bound, and explicit
worker shutdown acknowledgements.

Full relaxed structures and run JSON remain outside Git. A compact signed
summary is committed after both gates complete.

Status: planned.
