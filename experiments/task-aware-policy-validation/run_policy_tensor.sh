#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FIRST_GPU="${FIRST_GPU:-4}"
GPU_COUNT="${GPU_COUNT:-3}"
OUT="runs/task_aware_policy/policy"
LOG="logs/task_aware_policy/policy"
mkdir -p "$OUT" "$LOG"

ATOMBIT_ENV="/public/home/lmy/.conda/envs/lmy"
MACE_ENV="/public/home/lmy/.conda/envs/MACE_clean"

run_worker() {
    local worker="$1"
    local gpu=$((FIRST_GPU + worker))
    local task_index=0
    local model optimizer distribution pool_size env_path max_steps output_path
    while read -r model optimizer distribution; do
        for pool_size in 32 64 256; do
            if (( task_index % GPU_COUNT == worker )); then
                output_path="$OUT/${model}_${optimizer}_${distribution}_r${pool_size}.json"
                if [[ -f "$output_path" ]]; then
                    task_index=$((task_index + 1))
                    continue
                fi
                env_path="$ATOMBIT_ENV"
                if [[ "$model" == "mace" ]]; then
                    env_path="$MACE_ENV"
                fi
                max_steps=500
                if [[ "$optimizer" == "fire" ]]; then
                    max_steps=2000
                fi
                CUDA_VISIBLE_DEVICES="$gpu" \
                CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                PYTORCH_ALLOC_CONF=expandable_segments:True \
                PYTHONPATH=. \
                "$env_path/bin/python" \
                    benchmarks/benchmark_task_aware_policy.py \
                    --mlip "$model" \
                    --optimizer "$optimizer" \
                    --workload-manifest \
                    "benchmarks/workloads/manifests/OPT-${distribution}-R${pool_size}-v1.json" \
                    --pilot \
                    "runs/task_aware_policy/policy_inputs/${model}_${optimizer}.json" \
                    --dataset-dir data/T2_test/structures \
                    --max-steps "$max_steps" \
                    --deterministic \
                    --device cuda:0 \
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
for ((worker = 0; worker < GPU_COUNT; worker++)); do
    run_worker "$worker" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
