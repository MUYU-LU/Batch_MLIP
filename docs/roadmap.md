# Roadmap

The authoritative project logic and evidence boundary are maintained in
[`research/project/`](../research/project/README.md). This roadmap lists only
work that remains after scheduler v1; completed chronological experiments stay
in `experiments/`.

## P0 — Freeze scheduler v1

- Commit and tag the validated automatic-default interface.
- Preserve the refill-policy artifact hash and complete evidence registry.
- Keep CUDA MPS as the primary independent-job baseline.
- Do not import historical timings into the next planner fit.

## P1 — OMC-CSP workload benchmark

- Freeze exact unique CIF identities without modifying the source collection.
- Construct nested `P64/P512/P2048` pools for every nonempty CSP family.
- Construct balanced inter-family wide-cost pools and eligible narrow-cost
  pools from the same selected CIFs.
- Validate hashes, uniqueness, nesting, family coverage, and workload
  descriptors.
- Collect paired scheduler-v1 and CUDA-MPS labels under one
  hardware/model/numerical contract.

## P2 — OMC-CSP policy validation

- Validate memory, work, horizon, and topology-volatility decisions across
  held-out CSP families.
- Add productive-wave adaptation without repeating optimization jobs.
- Select only execution parameters; keep the user’s MLIP and optimizer fixed.
- Compare scheduler v1, CUDA MPS, zero-shot, adaptive, and a bounded oracle.
- Report family-level regret, total makespan, peak reserved memory, failures,
  and endpoint equivalence.

## P3 — Conformer and application extension

- Construct the nonperiodic molecular conformer workload separately; do not
  infer its policy from OMC-CSP labels.
- Revisit chemical parent selection only for transfer beyond available
  workload families.
- Extend the same planner contract separately to static phonons and
  equation-of-state workloads.
- Validate task-aware NVE/NVT/NPT scheduling only after long-horizon drift,
  thermostat/barostat, restart, and stress gates pass.

New acceleration mechanisms enter through `AGENTS.md`; they do not alter the
frozen comparison baseline during held-out planner evaluation.
