# CPU materialization policy

The source-backed path originally parsed each assigned CIF sequentially inside
its GPU worker. Threads regressed on the same files. Independent process
loading scaled, so the accepted implementation gives each persistent GPU
worker a deterministic `spawn` pool and closes it with that worker.

Four loader processes improved external P2048 makespan by `1.079x-1.207x` on
ROF-A, ROF-C, XAFPAY, and BOQWIN. It regressed XULDUD by `2.35%`. The offline
policy therefore selects four processes only for a pool of at least 2,048
structures, at least 32,000 signed-manifest atom-records per active worker, and
enough host CPUs for four loaders plus one compute thread per worker. All other
cases retain one process. A positive integer remains an explicit override.

The integrated automatic check selected four processes for ROF-A and one for
XULDUD without a pilot. ROF-A critical-worker materialization decreased from
`57.59 s` to `17.82 s` (`3.23x`) and worker runtime decreased from `98.05 s` to
`69.55 s` (`1.41x`). Its cold external run was `130.64 s` because fresh-source
worker startup took `46.52 s`; the earlier warm four-process external run was
`110.09 s`. XULDUD stayed serial and completed in `75.62 s`, close to the
`74.48 s` frozen serial reference.

Every point is one run. The raw authoritative outputs remain on the execution
host under `omc_csp_scheduler_epoch3/results/materialization_*`.
