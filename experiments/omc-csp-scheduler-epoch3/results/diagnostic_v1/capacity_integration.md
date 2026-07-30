# OMC-CSP Offline Capacity Integration

| Workload | Offline (s) | Probe (s) | MPS (s) | vs probe | vs MPS | Peak | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2 | 139.623 | 138.887 | 441.257 | 0.995x | 3.160x | 59.56% | fail |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2 | 92.868 | 93.645 | 212.013 | 1.008x | 2.283x | 51.20% | fail |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2 | 76.513 | 79.933 | 160.823 | 1.045x | 2.102x | 64.86% | fail |

Aggregate speedup over probe-backed source: `1.011x`.
Aggregate speedup over MPS: `2.635x`.
Maximum peak reserved fraction: `64.86%`.
