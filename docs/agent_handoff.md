# Agent Handoff Prompt

Use the following as the starting instruction for an autonomous coding/research agent:

> Work inside this repository. Read `README.md`, `AGENTS.md`, `docs/architecture.md`, and `docs/validation.md` before changing code. Preserve `original_uploads/`. Establish the baseline with `pytest -q` and the baseline experiment. Select one item from `docs/roadmap.md`, create an experiment with `python tools/new_experiment.py <id>`, state a falsifiable hypothesis, implement the smallest testable change, add regression tests, run batch/single validation, run NVE checks when forces/neighbours/precision/integration change, and benchmark reproducibly. Store all commands and raw results. Do not claim a speedup if scientific tolerances regress. End with a written comparison against the baseline and the next recommended experiment.

For real-model work, provide the agent with the checkpoint, matching architecture configuration, representative extxyz validation set, and accepted numerical tolerances. Do not place private checkpoints in a public repository.
