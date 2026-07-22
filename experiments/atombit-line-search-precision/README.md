# AtomBit BFGSLineSearch precision diagnostic

## Question

Does converting the complete AtomBit inference and optimization path from
float32 to float64 resolve the excessive strong-Wolfe trial evaluations seen
for H46 variable-cell optimization?

## Protocol

Stage 1 isolates `Aven12_1_367_z1.cif`, reported as system 13 in the failed
H46 R32 active-batch run. Sequential ASE and active batched B1 are each run in
float32 and float64. All optimizer state is float64 so the controlled variable
is model, graph, autograd, energy, force, and stress evaluation precision.

If float64 passes Stage 1, Stage 2 compares sequential ASE and active B32 on
the original frozen H46 R32 pool. Runs use one deterministic timing observation,
`fmax=0.05 eV/A`, at most 500 accepted steps, `alpha=10`, and `maxstep=0.2 A`.

Float64 evaluation does not recover information absent from checkpoint weights;
it tests whether higher-precision execution and reductions stabilize the line
search.
