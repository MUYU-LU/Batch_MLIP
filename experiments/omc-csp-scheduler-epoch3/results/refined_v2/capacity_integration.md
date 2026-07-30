# OMC-CSP Offline Capacity Integration

| Workload | Offline (s) | Probe (s) | MPS (s) | vs probe | vs MPS | Peak | Capacity | Strict endpoint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2 | 132.860 | 138.887 | 441.257 | 1.045x | 3.321x | 63.58% | pass | pass |
| OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2 | 94.165 | 93.645 | 212.013 | 0.994x | 2.251x | 51.20% | pass | fail |
| OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2 | 74.482 | 79.933 | 160.823 | 1.073x | 2.159x | 50.71% | pass | fail |

Aggregate speedup over probe-backed source: `1.036x`.
Aggregate speedup over MPS: `2.700x`.
Maximum peak reserved fraction: `63.58%`.
Maximum actual/predicted memory ratio: `0.9160`.
Capacity gates: `pass`.
Capacity plus strict endpoint gates: `fail`.
