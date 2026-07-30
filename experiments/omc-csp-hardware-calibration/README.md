# OMC-CSP Hardware Calibration

This experiment binds the layered OMC-CSP profile to the exact execution
contract:

`AtomBit smooth-rms fp32 + float64 BatchedBFGS + FrechetCellFilter + H100 80GB + expandable_segments`.

It is an offline project calibration, not a pilot performed for each user
workload. The expanded design contains 42 fit points from 12 OMC families and
20 validation points from six chemically held-out families. Four additional
untouched families provide eight final tests. Resident sizes span B8-B512;
every point has one warm execution and one measured execution, with no timing
repeat.

## Accepted Results

- Expanded held-out allocated-memory MARE/max error: 0.81% / 0.94%.
- Expanded held-out reserved-memory MARE/max error: 3.10% / 7.25%.
- Expanded held-out per-evaluation runtime MARE/max error: 23.44% / 66.17%.
- All eight untouched final tests completed without OOM: UJIRIO B128/B286,
  WIDBAO B128/B364, XAFQIH B128/B293, and XULDUD B128/B207.
- On the untouched final families, reserved memory was conservatively
  overpredicted by 1.49-7.21% (MARE 3.05%).

The reserved-memory model is accepted for capacity planning. The runtime model
is only a coarse ordering prior and is not accepted as a precise time
predictor. It does not replace the measured throughput frontier because GPU
saturation and allocator effects are nonlinear in batch size.

The default allocator is rejected for this contract. In the negative control,
ROF-B B64 allocated 25.92 GiB but reserved 77.92 GiB. With
`expandable_segments`, held-out reserve prediction and the three capacity
checks pass.

## Public Use

```python
from batch_mlip import (
    HardwareCalibratedBatchPlanner,
    load_hardware_cost_model,
    relax,
)

memory_model = load_hardware_cost_model(
    "experiments/omc-csp-hardware-calibration/results/expanded/calibration.json",
    model_name="peak_reserved_bytes",
)
planner = HardwareCalibratedBatchPlanner(memory_model)
result = relax(
    structures,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    planner=planner,
    cell_filter=cell_filter,
)
```

This automatic path profiles atoms and active/candidate edges once, adds the
BFGS/Frechet task state, and selects memory-safe queues without trial
optimizations or timing sweeps.

The original 18-fit/11-validation calibration remains in
`results/calibration.json` as the seed study. It is superseded for production
capacity planning by `results/expanded/calibration.json`.
