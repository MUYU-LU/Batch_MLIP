# Formal MACE Batch Optimization Test

This opt-in integration test applies the same correctness protocol used for
AtomBit variable-cell optimization to MACE-OFF-Small. Four fixed 46-atom T2
structures are optimized for three forced steps with common ASE, masked batch,
and active batch execution. Both FIRE and BFGS use the full Frechet cell filter.

The gate compares step counts, energies, forces, full stress tensors, atomic
positions, and cells. It also checks masked/active tensor identity and model and
graph evaluation accounting. Performance scaling remains a benchmark concern;
this pytest test is a deterministic correctness gate.
