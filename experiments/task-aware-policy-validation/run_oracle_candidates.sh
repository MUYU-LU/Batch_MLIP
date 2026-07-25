#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FIRST_GPU="${FIRST_GPU:-4}"
GPU_COUNT="${GPU_COUNT:-3}"
OUT="runs/task_aware_policy/candidates"
LOG="logs/task_aware_policy/candidates"
mkdir -p "$OUT" "$LOG"

ATOMBIT_ENV="/public/home/lmy/.conda/envs/lmy"
MACE_ENV="/public/home/lmy/.conda/envs/MACE_clean"
CHECKPOINT="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"

run_homogeneous() {
    local gpu="$1" model="$2" optimizer="$3" atom_count="$4"
    local pool_size="$5" method="$6" batch_size="$7"
    local env_path="$ATOMBIT_ENV"
    local script="benchmarks/benchmark_variable_cell_scaling.py"
    local max_steps=500
    local extra=(
        --checkpoint "$CHECKPOINT"
        --model-dtype float32
        --optimizer-dtype float64
    )
    if [[ "$model" == "mace" ]]; then
        env_path="$MACE_ENV"
        script="benchmarks/benchmark_mace_variable_cell_scaling.py"
        extra=()
    fi
    if [[ "$optimizer" == "fire" ]]; then
        max_steps=2000
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=. \
    "$env_path/bin/python" "$script" \
        --method "$method" \
        --optimizer "$optimizer" \
        --atom-count "$atom_count" \
        --pool-size "$pool_size" \
        --batch-sizes "$batch_size" \
        --repeats 1 \
        --skin 0 \
        --max-steps "$max_steps" \
        --refill-storage slots \
        --deterministic \
        --device cuda:0 \
        "${extra[@]}" \
        --output \
        "$OUT/${model}_${optimizer}_H${atom_count}_r${pool_size}_${method}_b${batch_size}.json" \
        > "$LOG/${model}_${optimizer}_H${atom_count}_r${pool_size}_${method}_b${batch_size}.log" 2>&1
}

run_mixed() {
    local gpu="$1" model="$2" pool_size="$3" batch_size="$4"
    local env_path="$ATOMBIT_ENV"
    local extra=(--checkpoint "$CHECKPOINT")
    if [[ "$model" == "mace" ]]; then
        env_path="$MACE_ENV"
        extra=()
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=. \
    "$env_path/bin/python" benchmarks/benchmark_mixed_scheduling.py \
        --mlip "$model" \
        --mode refill \
        --batch-size "$batch_size" \
        --workload-manifest \
        "benchmarks/workloads/manifests/OPT-MIX4-R${pool_size}-v1.json" \
        --dataset-dir data/T2_test/structures \
        --deterministic \
        --device cuda:0 \
        "${extra[@]}" \
        --output \
        "$OUT/${model}_bfgs_MIX4_r${pool_size}_refill_b${batch_size}.json" \
        > "$LOG/${model}_bfgs_MIX4_r${pool_size}_refill_b${batch_size}.log" 2>&1
}

run_worker() {
    local worker="$1"
    local gpu=$((FIRST_GPU + worker))
    local index=0 model optimizer atom_count pool_size batch_size
    while read -r model optimizer atom_count; do
        for specification in "64 32" "256 64"; do
            read -r pool_size batch_size <<< "$specification"
            if (( index % GPU_COUNT == worker )); then
                run_homogeneous \
                    "$gpu" "$model" "$optimizer" "$atom_count" \
                    "$pool_size" refill "$batch_size"
            fi
            index=$((index + 1))
        done
    done <<'EOF'
atombit bfgs 92
mace bfgs 92
atombit fire 184
mace fire 184
EOF
    for model in atombit mace; do
        for specification in "64 32" "256 64"; do
            read -r pool_size batch_size <<< "$specification"
            if (( index % GPU_COUNT == worker )); then
                run_mixed "$gpu" "$model" "$pool_size" "$batch_size"
            fi
            index=$((index + 1))
        done
    done
    if (( index % GPU_COUNT == worker )); then
        run_homogeneous  \
            "$gpu" atombit bfgs 92 256 active 128
    fi
}

pids=()
for ((worker = 0; worker < GPU_COUNT; worker++)); do
    run_worker "$worker" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
