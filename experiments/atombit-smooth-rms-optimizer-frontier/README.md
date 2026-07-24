# Smooth-RMS AtomBit optimizer frontier

Status: planned.

This experiment measures the maximum safe single-GPU throughput of native-fp32
smooth-RMS AtomBit variable-cell relaxation for FIRE, full BFGS, and
BFGSLineSearch across H46, H92, H184, and H276 structures.

The R256 workload repeats the frozen R32 manifest order exactly eight times.
Common ASE is measured on R32 for every optimizer and size; its measured
throughput is therefore the exact sequential reference for the repeated R256
workload. Batch screening starts at the two sizes specified in
`experiment.yaml` and extends downward only when needed to bracket a maximum.

This is a one-observation frontier screen, not an uncertainty-qualified paper
benchmark. Selected paper points would require confirmation repeats.
