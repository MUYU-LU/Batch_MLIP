#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/lmy/Batch_imple_project/Batch_MLIP_process_stage
DATA=/public/home/lmy/Batch_imple_project/test_set
PYTHON=/public/home/lmy/.conda/envs/lmy/bin/python
CHECKPOINT=/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt
EXP="$ROOT/experiments/prompt-executor-shutdown"
SOURCE="$ROOT/experiments/heldout-auto-vs-mps"
RAW="$EXP/results/raw"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$RAW"
cd "$ROOT"

for pool in 256 1024; do
    output="$RAW/auto_r${pool}.json"
    external="$RAW/auto_r${pool}_external.json"
    manifest="$SOURCE/workloads/manifests/OPT-HOLDOUT-MIX-R${pool}-v1.json"
    [[ -s "$output" && -s "$external" ]] && continue
    started=$(date +%s%N)
    "$PYTHON" benchmarks/benchmark_persistent_executor.py \
        --mlip atombit --mode persistent --optimizer bfgs \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 --calls 1 \
        --automatic-capacity --max-steps 500 --fmax 0.05 \
        --deterministic --workload-manifest "$manifest" \
        --dataset-dir "$DATA" \
        --cache-path "$RAW/auto_cache_r${pool}.json" --clear-cache \
        --checkpoint "$CHECKPOINT" --output "$output" \
        > "$RAW/auto_r${pool}.log" 2>&1
    ended=$(date +%s%N)
    "$PYTHON" - "$started" "$ended" "$external" <<'PY'
import json
import sys

started, ended, output = sys.argv[1:]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(
        {"external_process_wall_seconds": (int(ended) - int(started)) / 1e9},
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
done
