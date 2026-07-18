# Lazy pending graphs for BFGS refill

## Change

The structure-level refill path now creates the 256-job output state without a
common neighbor list. AtomBit builds neighbors only after a structure enters
the resident B64/B128 batch. MACE builds its native `AtomicData` only for the
resident structures and no longer constructs an unused common pending graph.

## Results

The table compares the previous eager refill with the lazy implementation.
Ratios above one favor lazy construction. Every point is one complete 256-job
variable-cell BFGS relaxation without timing repeats.

| model | atoms | eager B64 | lazy B64 | ratio | eager B128 | lazy B128 | ratio |
|:---|---:|---:|---:|---:|---:|---:|---:|
| AtomBit | 46 | 115.40 | 115.84 | 0.996x | 109.43 | 108.71 | 1.007x |
| AtomBit | 92 | 213.50 | 212.33 | 1.006x | 209.84 | 207.58 | 1.011x |
| AtomBit | 184 | 284.62 | 285.59 | 0.997x | 283.05 | 282.64 | 1.001x |
| AtomBit | 276 | 324.75 | 323.97 | 1.002x | 321.88 | 320.51 | 1.004x |
| MACE | 46 | 129.91 | 130.34 | 0.997x | 123.65 | 125.03 | 0.989x |
| MACE | 92 | 208.46 | 208.79 | 0.998x | 203.67 | 203.91 | 0.999x |
| MACE | 184 | 276.94 | 276.30 | 1.002x | 270.48 | 274.42 | 0.986x |
| MACE | 276 | 315.66 | 313.87 | 1.006x | 313.41 | 318.76 | 0.983x |

The timing ratio spans 0.983x-1.011x, so the change is neutral under the
single-run protocol. Peak allocated memory falls by 8.7-90.1 MiB, less than
0.4% of the resident workloads. Common pending edges are therefore measurable
waste but not a performance bottleneck for full BFGS.

## Correctness

All 16 comparisons are bitwise identical for the complete 256-job output:
convergence flags, step counts, energies, forces, stresses, positions, and
cells. Unit tests also verify that a three-job B1 refill builds exactly three
size-one neighbor lists and never builds a size-three pending graph.

## Conclusion

Retain lazy construction because it removes unnecessary work and lowers memory
without changing results. Do not claim a speedup. Further BFGS acceleration
should target the independent dense eigensolves or replace full BFGS with a
batched limited-memory method; MACE can additionally avoid rebuilding native
`AtomicData` from ASE structures on every optimizer step.
