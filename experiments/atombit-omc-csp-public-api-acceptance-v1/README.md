# AtomBit OMC-CSP public API acceptance v1

This is the release gate for direct use of the frozen AtomBit OMC-CSP path.
It invokes `AtomBitBatchCalculator.from_checkpoint(...)` and
`optimize_pool(...)` rather than benchmark-only loaders or manifest execution
internals.

The small gate uses the signed, unique `rof-c` P64 test workload on one H100.
The large gate uses the nested signed `rof-c` P2048 workload on the six H100s
that were externally idle. Both retain the AtomBit smooth-RMS
fp32, float64 BFGS state, `FrechetCellFilter`, 6.0 A cutoff, 0.5 A skin,
maximum resident batch size 256, `fmax=0.01 eV/A`, and 3,000-step production
contract.

The first large-pool diagnostic preserved all numerical gates but found a
14.2% reserved-memory underprediction. Capacity policy v2 therefore binds a
1.30 model-specific growth margin; it does not lower the user's 85% planning
budget or change MACE defaults.

Acceptance requires exact packaged-policy selection with no memory-probe model
forwards, immutable output coverage and ordering, finite endpoints, at least
99% convergence, predicted chunks within the 85% planning budget, observed
reserved memory within the frozen 91% runtime high-water bound, and explicit
worker shutdown acknowledgements.

Full relaxed structures and run JSON remain outside Git. The compact committed
summary binds their hashes and the prior eight-H100 P6000/MPS16 reference.

Status: accepted. Both P64/G1 and P2048/G6 gates passed with 100% convergence,
exact input ordering, exact policy-v2 selection, zero probe forwards, and peak
reserved fractions of 11.76% and 58.35%, respectively.
