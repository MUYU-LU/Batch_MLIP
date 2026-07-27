#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/lmy/Batch_imple_project/Batch_MLIP_process_stage
DATA=/public/home/lmy/Batch_imple_project/test_set
PYTHON=/public/home/lmy/.conda/envs/lmy/bin/python
CHECKPOINT=/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt
EXP="$ROOT/experiments/heldout-auto-vs-mps"
RAW="$EXP/results/raw"
UUIDS=(
    GPU-2cdc5de1-68c5-b482-f42a-f3be8539fa75
    GPU-687cb62d-55f3-2b6a-7826-94e3b9eeb9d7
    GPU-5224a82c-031d-260b-e039-19963cc0e69c
    GPU-75664b50-96bc-d831-8932-45ef9bb09fcc
)

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$RAW"
cd "$ROOT"

write_elapsed() {
    local started=$1 ended=$2 output=$3
    "$PYTHON" - "$started" "$ended" "$output" <<'PY'
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
}

run_auto() {
    local pool=$1 manifest output started ended
    manifest="$EXP/workloads/manifests/OPT-HOLDOUT-MIX-R${pool}-v1.json"
    output="$RAW/auto_r${pool}.json"
    [[ -s "$output" ]] && return
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
    write_elapsed "$started" "$ended" "$RAW/auto_r${pool}_external.json"
}

run_mps_gpu() {
    local pool=$1 gpu=$2 per_gpu manifest output
    per_gpu=$((pool / 4))
    manifest="$EXP/workloads/mps_r${pool}/OPT-HOLDOUT-MIX-R${pool}-v1-MPS-G${gpu}.json"
    output="$RAW/mps_r${pool}_gpu${gpu}.json"
    [[ -s "$output" ]] && return
    CUDA_VISIBLE_DEVICES="${UUIDS[$gpu]}" \
    CUDA_MPS_PIPE_DIRECTORY="/public/home/lmy/.cuda-mps/gpu${gpu}/pipe" \
    CUDA_MPS_LOG_DIRECTORY="/public/home/lmy/.cuda-mps/gpu${gpu}/log" \
        "$PYTHON" benchmarks/benchmark_mps_ase_pool.py \
        --mlip atombit --task optimization --optimizer bfgs \
        --pool-size "$per_gpu" --workers 4 --device cuda:0 \
        --gpu-index "$gpu" --workload-manifest "$manifest" \
        --dataset-dir "$DATA" --checkpoint "$CHECKPOINT" \
        --fmax 0.05 --max-steps 500 --alpha 70.0 --max-step 0.2 \
        --model-dtype float32 --optimizer-dtype float64 \
        --cpu-threads-per-worker 1 --deterministic \
        --output "$output" > "$RAW/mps_r${pool}_gpu${gpu}.log" 2>&1
}

run_mps() {
    local pool=$1 started ended pids=()
    [[ -s "$RAW/mps_r${pool}_external.json" ]] && return
    started=$(date +%s%N)
    for gpu in 0 1 2 3; do
        run_mps_gpu "$pool" "$gpu" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    ended=$(date +%s%N)
    write_elapsed "$started" "$ended" "$RAW/mps_r${pool}_external.json"
}

for pool in 256 1024; do
    run_auto "$pool"
    run_mps "$pool"
done
