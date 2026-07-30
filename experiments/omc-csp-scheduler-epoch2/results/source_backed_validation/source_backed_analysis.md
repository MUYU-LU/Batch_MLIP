# OMC-CSP Source-Backed P2048 Ablation

| Workload | Lazy (s) | Eager (s) | MPS (s) | vs eager | vs MPS | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P2048-INTRA-NARROW-v2 | 159.829 | 265.881 | 368.016 | 1.664x | 2.303x | 76.2% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P2048-INTRA-NARROW-v2 | 164.653 | 189.050 | 465.587 | 1.148x | 2.828x | 19.8% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P2048-INTRA-NARROW-v2 | 247.119 | 310.120 | 716.373 | 1.255x | 2.899x | 37.6% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P2048-INTRA-NARROW-v2 | 488.909 | 1052.815 | 2071.279 | 2.153x | 4.237x | 69.5% |
| OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P2048-INTRA-WIDE-v2 | 177.668 | 332.024 | 439.887 | 1.869x | 2.476x | 70.5% |

## Aggregate

- Scientific/resource gates: `True`
- Plan signatures equal: `True`
- Wins over eager: `5/5`
- Wins over MPS: `5/5`
- Aggregate speedup over eager: `1.736x`
- Aggregate speedup over MPS: `3.280x`
- Convergence: `10146/10240` source-backed, `10146/10240` eager, `10183/10240` MPS
