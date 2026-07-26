# Process-worker autoscheduler

This experiment integrates the repository's spawn-isolated multi-GPU mechanism
into the public automatic relaxation path. The current public implementation
uses threads and measured 57% four-GPU AtomBit efficiency because calculator
graph preparation and Python orchestration contend inside one process.

The process backend must preserve the normal interface:

```python
relax(
    structures,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
)
```

Startup/model replication time is reported separately and remains part of
end-to-end time.

## Result

All AtomBit H92-R256 and MACE H92-R32 jobs converged. Every planned input index
was returned exactly once, children exited cleanly, and peak reserved memory per
GPU was effectively unchanged.

| Model/workload | Backend | Worker run (s) | Startup (s) | End-to-end (s) | Peak reserved |
|:--|:--|--:|--:|--:|--:|
| AtomBit H92-R256, 4 GPU | thread | 35.49 | 0.00 | **36.47** | 17.13 GiB |
| AtomBit H92-R256, 4 GPU | process | **24.02** | 11.79 | 37.41 | 17.10 GiB |
| MACE H92-R32, 2 GPU | thread | 17.17 | 0.00 | **17.44** | 3.02 GiB |
| MACE H92-R32, 2 GPU | process | **14.16** | 17.88 | 34.06 | 2.98 GiB |

Process isolation improves the steady worker phase by 1.48x for AtomBit and
1.21x for MACE. Against the previously committed 81.04 s warm single-GPU
AtomBit point, the 24.02 s process worker phase is 3.37x faster on four GPUs,
or 84% parallel efficiency, passing the experiment's 70% steady-state gate.

The tested pools contain only one chunk per GPU. Python spawn, imports, model
reconstruction, and warm-up therefore outweigh the worker-phase gain. The
automatic backend conservatively requires eight pending chunks per GPU before
selecting processes; explicit process execution remains available for research.
A future cross-call worker service should pay startup only once.

Full BFGS endpoints are not bitwise invariant between thread and process
execution. All jobs converge, but small execution-order perturbations alter
step counts and can select different minima, as already observed for tensor B1
and MPS in the cross-family robustness study. These results support throughput
and convergence equivalence, not identical-minimum identity.

Raw accepted and negative runs, including the ineffective shared-CPU-model
experiment and the superseded pre-fix thread timings, are retained under
`results/raw`.
