# Allocator-aware process launcher

This experiment tests whether the public automatic scheduler can apply an
evidence-based CUDA allocator policy before either cold tuning or production
workers initialize CUDA.

## Result

The implementation passes the experiment.

- AtomBit variable-cell BFGS automatically selected expandable segments.
- Both PyTorch compatibility variables were present in each worker before
  calculator preparation; `torch.cuda.is_initialized()` was false at that
  boundary.
- A cold-cache H46 B128 run converged 256/256 structures in 57.32 s with
  8.58 GiB peak allocated and 9.27 GiB peak reserved memory.
- The explicit-expandable one-GPU control had the same 9.27 GiB peak reserve.
- Two warm workers completed the same pool in 38.88 s, a measured 1.35x
  end-to-end speedup over the isolated 52.62 s one-GPU explicit control.

The automatic and explicit policies are numerically equivalent at the existing
AtomBit trajectory tolerance scale. Over two BFGS steps on all 256 structures,
maximum differences were 0.95 microangstrom in positions and cells,
6.68 microelectronvolt in total energy, and 1.46e-4 eV/A in force. Convergence
flags and step arrays matched. One- versus two-GPU errors were no larger.

## Negative control

MACE was launched from a parent process with both allocator variables set to
expandable segments. Its spawned worker selected native allocation, removed
both variables, reported CUDA uninitialized before preparation, and reported
the native backend. The AtomBit rule therefore does not leak into other MLIPs.

## Interpretation

The validated interface is:

```python
result = relax(
    structures,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    devices=["cuda:0", "cuda:1"],
    cell_filter=FrechetCellFilter(),
)
```

The first call is protected as well as warm-cache calls because cold tuning is
moved into a configured child for the expandable policy. Calls without
`devices` remain in-process and inherit the allocator chosen before Python
started.

Performance timings have one sample, so the 1.35x multi-GPU result is a
screening measurement rather than an uncertainty-qualified scaling claim.
The machine-readable conclusion is `results/summary.json`; all raw runs and
full tensor traces are retained in `results/raw-results.tar.gz`.
