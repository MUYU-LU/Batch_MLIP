#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/lmy/Batch_imple_project/Batch_MLIP_process_stage
DATA=/public/home/lmy/Batch_imple_project/test_set
ATOMBIT_ENV=/public/home/lmy/.conda/envs/lmy
MACE_ENV=/public/home/lmy/.conda/envs/MACE_clean
CHECKPOINT=/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt
OUT="$ROOT/experiments/end-to-end-optimization-policy/correctness"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
cd "$ROOT"

run_case() {
    local spec=$1 gpu=$2
    IFS=: read -r mlip optimizer family <<<"$spec"
    local python manifest steps output
    python=$ATOMBIT_ENV/bin/python
    [[ "$mlip" == mace ]] && python=$MACE_ENV/bin/python
    steps=500
    [[ "$optimizer" == fire ]] && steps=2000
    case "$family" in
        AXOSOW)
            manifest=benchmarks/workloads/manifests/OPT-RB-AXOSOW64-R256-v1.json
            ;;
        XAFPAY)
            manifest=benchmarks/workloads/manifests/OPT-RB-XAFPAY172-R256-v1.json
            ;;
    esac
    output="$OUT/auto_${mlip}_${optimizer}_${family}_R64.json"
    [[ -s "$output" ]] && return
    CUDA_VISIBLE_DEVICES=$gpu "$python" \
        benchmarks/benchmark_persistent_executor.py \
        --mlip "$mlip" --mode persistent --optimizer "$optimizer" \
        --devices cuda:0 --calls 1 --job-limit 64 \
        --resident-batch-size 256 --max-steps "$steps" --fmax 0.05 \
        --deterministic --store-final-tensors \
        --workload-manifest "$manifest" --dataset-dir "$DATA" \
        --cache-path experiments/end-to-end-optimization-policy/auto-cache.json \
        --checkpoint "$CHECKPOINT" --output "$output"
}

tasks=()
for family in AXOSOW XAFPAY; do
    for mlip in atombit mace; do
        for optimizer in fire bfgs; do
            tasks+=("$mlip:$optimizer:$family")
        done
    done
done

devices=(4 5 6)
index=0
while (( index < ${#tasks[@]} )); do
    pids=()
    for gpu in "${devices[@]}"; do
        (( index >= ${#tasks[@]} )) && break
        run_case "${tasks[index]}" "$gpu" &
        pids+=("$!")
        ((index += 1))
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
done
