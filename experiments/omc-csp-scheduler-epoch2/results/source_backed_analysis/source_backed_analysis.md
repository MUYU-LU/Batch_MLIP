# OMC-CSP Source-Backed P2048 Ablation

| Workload | Lazy (s) | Eager (s) | MPS (s) | vs eager | vs MPS | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| OPT-OMC-SCHED-E1-TEST-JAYDUI-P2048-INTRA-NARROW-v2 | 47.633 | 115.340 | 74.124 | 2.421x | 1.556x | 31.7% |
| OPT-OMC-SCHED-E1-TEST-OBEQIX-P2048-INTRA-NARROW-v2 | 210.563 | 536.598 | 656.298 | 2.548x | 3.117x | 72.0% |
| OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2 | 138.887 | 418.934 | 441.257 | 3.016x | 3.177x | 64.4% |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2 | 93.645 | 215.919 | 212.013 | 2.306x | 2.264x | 52.5% |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2 | 79.933 | 138.719 | 160.823 | 1.735x | 2.012x | 55.0% |

## Aggregate

- Scientific/resource gates: `True`
- Plan signatures equal: `True`
- Wins over eager: `5/5`
- Wins over MPS: `5/5`
- Aggregate speedup over eager: `2.498x`
- Aggregate speedup over MPS: `2.707x`
- Convergence: `10240/10240` source-backed, `10240/10240` eager, `10240/10240` MPS
