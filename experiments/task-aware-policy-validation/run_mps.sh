#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_COUNT="${GPU_COUNT:-4}"
OUT="runs/task_aware_policy/mps"
LOG="logs/task_aware_policy/mps"
mkdir -p "$OUT" "$LOG"

ATOMBIT_ENV="/public/home/lmy/.conda/envs/lmy"
MACE_ENV="/public/home/lmy/.conda/envs/MACE_clean"
ATOMBIT_CHECKPOINT="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
ATOMBIT_E0="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/meta_e0_data_OMC_r6_single.pt"

run_worker() {
    local gpu="$1"
    local task_index=0
    local model optimizer distribution pool_size env_path max_steps cutoff
    local gpu_uuid output_path
    gpu_uuid="$(
        nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i "$gpu"
    )"
    while read -r model optimizer distribution; do
        for pool_size in 32 64 256; do
            if (( task_index % GPU_COUNT == gpu )); then
                output_path="$OUT/${model}_${optimizer}_${distribution}_r${pool_size}.json"
                if [[ -f "$output_path" ]]; then
                    task_index=$((task_index + 1))
                    continue
                fi
                env_path="$ATOMBIT_ENV"
                cutoff=6.0
                extra=(
                    --checkpoint "$ATOMBIT_CHECKPOINT"
                    --atombit-e0 "$ATOMBIT_E0"
                    --model-dtype float32
                    --optimizer-dtype float64
                )
                if [[ "$model" == "mace" ]]; then
                    env_path="$MACE_ENV"
                    cutoff=5.0
                    extra=(--mace-model small)
                fi
                max_steps=500
                if [[ "$optimizer" == "fire" ]]; then
                    max_steps=2000
                fi
                # UUIDs avoid ordinal remapping ambiguity between independently
                # pinned MPS daemons and their client processes.
                CUDA_VISIBLE_DEVICES="$gpu_uuid" \
                CUDA_MPS_PIPE_DIRECTORY="/public/home/lmy/.cuda-mps/gpu${gpu}/pipe" \
                CUDA_MPS_LOG_DIRECTORY="/public/home/lmy/.cuda-mps/gpu${gpu}/log" \
                CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                PYTORCH_ALLOC_CONF=expandable_segments:True \
                PYTHONPATH=. \
                "$env_path/bin/python" benchmarks/benchmark_mps_ase_pool.py \
                    --mlip "$model" \
                    --task optimization \
                    --optimizer "$optimizer" \
                    --pool-size "$pool_size" \
                    --workers 32 \
                    --workload-manifest \
                    "benchmarks/workloads/manifests/OPT-${distribution}-R${pool_size}-v1.json" \
                    --dataset-dir data/T2_test/structures \
                    --cutoff "$cutoff" \
                    --max-steps "$max_steps" \
                    --deterministic \
                    --device cuda:0 \
                    --gpu-index "$gpu" \
                    --cpu-threads-per-worker 1 \
                    --worker-start-interval 0.02 \
                    "${extra[@]}" \
                    --output \
                    "$output_path" \
                    > "$LOG/${model}_${optimizer}_${distribution}_r${pool_size}.log" 2>&1
            fi
            task_index=$((task_index + 1))
        done
    done <<'EOF'
atombit bfgs H92
mace bfgs H92
atombit fire H184
mace fire H184
atombit bfgs MIX4
mace bfgs MIX4
EOF
}

pids=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    run_worker "$gpu" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
