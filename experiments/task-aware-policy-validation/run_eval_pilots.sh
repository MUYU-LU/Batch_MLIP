#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_COUNT="${GPU_COUNT:-7}"
OUT="runs/task_aware_policy/eval_timing"
LOG="logs/task_aware_policy/eval_timing"
mkdir -p "$OUT" "$LOG"

ATOMBIT_ENV="/public/home/lmy/.conda/envs/lmy"
MACE_ENV="/public/home/lmy/.conda/envs/MACE_clean"
ATOMBIT_CHECKPOINT="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
ATOMBIT_E0="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/meta_e0_data_OMC_r6_single.pt"
COMMIT="$(git rev-parse HEAD)"

run_worker() {
    local gpu="$1"
    local task_index=0
    local model atom_count batch_size env_path
    for model in atombit mace; do
        for atom_count in 46 276; do
            for batch_size in 1 8 16 32 64 128; do
                if (( task_index % GPU_COUNT == gpu )); then
                    env_path="$ATOMBIT_ENV"
                    extra=(
                        --atombit-checkpoint "$ATOMBIT_CHECKPOINT"
                        --atombit-e0 "$ATOMBIT_E0"
                    )
                    if [[ "$model" == "mace" ]]; then
                        env_path="$MACE_ENV"
                        extra=()
                    fi
                    CUDA_VISIBLE_DEVICES="$gpu" \
                    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                    PYTORCH_ALLOC_CONF=expandable_segments:True \
                    PYTHONPATH=. \
                    "$env_path/bin/python" \
                        benchmarks/benchmark_eval_capacity_point.py \
                        --model "$model" \
                        --distribution "H${atom_count}" \
                        --batch-size "$batch_size" \
                        --compute-stress \
                        --validation-count 1 \
                        --device cuda:0 \
                        --code-commit "$COMMIT" \
                        "${extra[@]}" \
                        --output "$OUT/${model}_h${atom_count}_b${batch_size}.json" \
                        > "$LOG/${model}_h${atom_count}_b${batch_size}.log" 2>&1
                fi
                task_index=$((task_index + 1))
            done
        done
    done
}

pids=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    run_worker "$gpu" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
