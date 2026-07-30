# Unified Automatic Scheduling

## Hypothesis

Ordinary relaxation calls can default to the deterministic memory scheduler
without changing optimizer results, while explicit manual refill and
`scheduling="single_batch"` calls retain their previous behavior.

## Baseline

- Commit: `4ee8b18`
- This is an API and reporting cleanup, not a throughput experiment.
- No speedup is claimed and no GPU benchmark is required.

## Policy

The public decision is now:

1. Use deterministic automatic scheduling when the calculator exposes a
   cutoff.
2. Keep explicit `single_batch` and manual refill controls unmanaged.
3. Use one compact `metadata["scheduling"]["summary"]` schema for automatic
   single-GPU, automatic multi-GPU, and manual execution.
4. Keep timing-based `autotune`, calibrated planners, and detailed scientific
   evidence as advanced interfaces.

## Validation

The regression tests cover the default automatic path, the explicit manual
path, manual refill compatibility, and the multi-device summary. Existing
optimizer and scientific-equivalence tests remain the behavioral gate.
