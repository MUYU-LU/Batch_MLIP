# Batched NVT and Isotropic NPT

## Outcome

The hypothesis passed. Fixed-cell Langevin BAOAB NVT and isotropic
Martyna-Tobias-Klein NPT now use persistent batched tensor state through the
same model-independent calculator interface. Isotropic NPT reproduces ASE at
B1, matches independent runs in a heterogeneous batch, restarts exactly, and
has the expected timestep-refinement behavior.

NPT owns one log-volume coordinate, volume momentum, Nose-Hoover thermostat
chain, and barostat chain per replica. Cell shape is fixed while volume
changes. Partial periodicity, `FixAtoms`, and nonpositive thermodynamic
parameters fail explicitly.

## Performance

All formal measurements used one H100, 20 warm-up steps, 100 measured steps,
`dt=0.5 fs`, `T=300 K`, one repeat, and signed T2 workloads. The MPS baseline
used four ASE worker processes per GPU and the same pool and measured horizon.
NPT uses isotropic MTK on both paths. The NVT comparison is ensemble- and
parameter-matched, but ASE uses its Langevin propagator while the tensor path
uses BAOAB; stepwise trajectory identity is therefore not claimed for NVT.

| Model | Workload | Ensemble | Batch B32 | ASE+MPS | Batch/MPS | Batch memory | MPS memory |
|---|---|---:|---:|---:|---:|---:|---:|
| AtomBit | H46 | NVT | 726.13 | 250.07 | 2.90x | 2.34 GB | 5.08 GB |
| AtomBit | H46 | NPT | 626.60 | 121.99 | 5.14x | 2.34 GB | 5.10 GB |
| MACE | H46 | NVT | 876.83 | 188.00 | 4.66x | 2.32 GB | 5.10 GB |
| MACE | H46 | NPT | 750.00 | 179.59 | 4.18x | 2.32 GB | 5.10 GB |
| AtomBit | H276 | NVT | 263.03 | 167.16 | 1.57x | 12.18 GB | 7.59 GB |
| AtomBit | H276 | NPT | 241.51 | 79.12 | 3.05x | 12.19 GB | 7.59 GB |
| MACE | H276 | NVT | 278.87 | 161.53 | 1.73x | 13.45 GB | 6.99 GB |
| MACE | H276 | NPT | 272.06 | 155.71 | 1.75x | 13.45 GB | 6.99 GB |

Throughput units are replica-steps/s. Batch memory is PyTorch peak allocated;
MPS memory is process-total `nvidia-smi` memory, so the memory columns are
operational rather than allocator-identical.

The large capacity points were H46 B256 and H276 B128. H46 reached
935-1042 replica-steps/s at about 18 GB allocated. H276 saturated much earlier:
B128 reached 274-306 replica-steps/s at 49-54 GB allocated, with AtomBit NVT
reserving 81.89 decimal GB. The planner should not maximize resident batch
blindly: B32 to B128 adds only 10-13% H276 throughput while consuming roughly
four times the allocated memory.

## Validation And Limits

```text
PYTHONPATH=. pytest -q
python -m ruff check batch_mlip tests/test_md.py tests/test_cli.py benchmarks/benchmark_batched_ensembles.py
python benchmarks/benchmark_batched_ensembles.py ... --warmup-steps 20 --measured-steps 100
python benchmarks/benchmark_mps_ase_pool.py ... --workers 4 --warmup-steps 20 --measured-steps 100
```

Results were `227 passed, 10 skipped` locally, `14 passed` in the focused
remote MD/CLI suite, `6 passed` for the MPS harness, and Ruff passed. Raw logs
and JSON are under `raw/`.

The 50 fs AtomBit float32 runs had maximum absolute isotropic-NPT conserved
drift of `3.99e-3 eV/atom`; MACE float64 reached `1.51e-4 eV/atom`. This does
not invalidate the batching implementation, but it prevents claiming that a
single timestep is production-safe across MLIPs. Long melting or phase-change
studies need an equilibration/statistics experiment and a model-specific
timestep/precision drift gate. Fully anisotropic NPT, constraints, partial PBC,
and cross-replica methods remain future work.
