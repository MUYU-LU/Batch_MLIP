#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/lmy/Batch_imple_project/Batch_MLIP_process_stage
DATA=/public/home/lmy/Batch_imple_project/test_set
ATOMBIT_ENV=/public/home/lmy/.conda/envs/lmy
MACE_ENV=/public/home/lmy/.conda/envs/MACE_clean
CHECKPOINT=/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt
RAW="$ROOT/experiments/end-to-end-optimization-policy/raw"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$RAW"
cd "$ROOT"

manifest_for() {
    case "$1" in
        AXOSOW)
            echo benchmarks/workloads/manifests/OPT-RB-AXOSOW64-R256-v1.json
            ;;
        XAFPAY)
            echo benchmarks/workloads/manifests/OPT-RB-XAFPAY172-R256-v1.json
            ;;
        *)
            return 1
            ;;
    esac
}

manual_capacity() {
    local mlip=$1 optimizer=$2 family=$3 pool=$4
    if [[ "$pool" == 64 ]]; then
        echo 64
        return
    fi
    case "$mlip:$optimizer:$family" in
        atombit:fire:AXOSOW) echo 128 ;;
        atombit:bfgs:AXOSOW) echo 256 ;;
        mace:fire:AXOSOW) echo 256 ;;
        mace:bfgs:AXOSOW) echo 256 ;;
        atombit:fire:XAFPAY) echo 64 ;;
        atombit:bfgs:XAFPAY) echo 128 ;;
        mace:fire:XAFPAY) echo 128 ;;
        mace:bfgs:XAFPAY) echo 128 ;;
        *) return 1 ;;
    esac
}

run_batch_case() {
    local spec=$1 gpu=$2
    IFS=: read -r method mlip optimizer family pool <<<"$spec"
    local manifest python max_steps output capacity allocator
    manifest=$(manifest_for "$family")
    python=$ATOMBIT_ENV/bin/python
    [[ "$mlip" == mace ]] && python=$MACE_ENV/bin/python
    max_steps=500
    [[ "$optimizer" == fire ]] && max_steps=2000
    output="$RAW/${method}_${mlip}_${optimizer}_${family}_R${pool}_G1.json"
    [[ -s "$output" ]] && return

    if [[ "$method" == auto ]]; then
        CUDA_VISIBLE_DEVICES=$gpu "$python" \
            benchmarks/benchmark_persistent_executor.py \
            --mlip "$mlip" --mode persistent --optimizer "$optimizer" \
            --devices cuda:0 --calls 1 --job-limit "$pool" \
            --resident-batch-size 256 --max-steps "$max_steps" --fmax 0.05 \
            --deterministic --workload-manifest "$manifest" \
            --dataset-dir "$DATA" \
            --cache-path experiments/end-to-end-optimization-policy/auto-cache.json \
            --checkpoint "$CHECKPOINT" --output "$output"
        return
    fi

    capacity=$(manual_capacity "$mlip" "$optimizer" "$family" "$pool")
    allocator=
    [[ "$mlip" == atombit ]] && allocator=expandable_segments:True
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTORCH_ALLOC_CONF=$allocator \
    PYTORCH_CUDA_ALLOC_CONF=$allocator \
    "$python" \
        benchmarks/benchmark_robustness_optimization.py \
        --mlip "$mlip" --method active --optimizer "$optimizer" \
        --workload-manifest "$manifest" --dataset-dir "$DATA" \
        --batch-size "$capacity" --job-limit "$pool" --cpu-threads 1 \
        --deterministic --device cuda:0 --fmax 0.05 \
        --max-steps "$max_steps" --skin 0.5 --checkpoint "$CHECKPOINT" \
        --output "$output"
}

run_mps_case() {
    local spec=$1 gpu=$2
    IFS=: read -r mlip optimizer family pool <<<"$spec"
    local manifest python max_steps output uuid cutoff
    manifest=$(manifest_for "$family")
    python=$ATOMBIT_ENV/bin/python
    cutoff=6.0
    if [[ "$mlip" == mace ]]; then
        python=$MACE_ENV/bin/python
        cutoff=5.0
    fi
    max_steps=500
    [[ "$optimizer" == fire ]] && max_steps=2000
    output="$RAW/mps_${mlip}_${optimizer}_${family}_R${pool}_G1.json"
    [[ -s "$output" ]] && return
    case "$gpu" in
        0) uuid=GPU-2cdc5de1-68c5-b482-f42a-f3be8539fa75 ;;
        1) uuid=GPU-687cb62d-55f3-2b6a-7826-94e3b9eeb9d7 ;;
        2) uuid=GPU-5224a82c-031d-260b-e039-19963cc0e69c ;;
        3) uuid=GPU-75664b50-96bc-d831-8932-45ef9bb09fcc ;;
        *) return 1 ;;
    esac

    CUDA_VISIBLE_DEVICES=$uuid \
    CUDA_MPS_PIPE_DIRECTORY=/public/home/lmy/.cuda-mps/gpu"$gpu"/pipe \
    CUDA_MPS_LOG_DIRECTORY=/public/home/lmy/.cuda-mps/gpu"$gpu"/log \
        "$python" benchmarks/benchmark_mps_ase_pool.py \
        --mlip "$mlip" --task optimization --optimizer "$optimizer" \
        --pool-size "$pool" --workers 4 --device cuda:0 \
        --workload-manifest "$manifest" --dataset-dir "$DATA" \
        --checkpoint "$CHECKPOINT" --gpu-index "$gpu" \
        --cpu-threads-per-worker 1 --deterministic \
        --model-dtype float32 --optimizer-dtype float64 \
        --cutoff "$cutoff" --fmax 0.05 --max-steps "$max_steps" \
        --output "$output"
}

run_batch_matrix() {
    local tasks=() devices=(4 5 6) pids=() index=0
    local family pool mlip optimizer method task
    for family in AXOSOW XAFPAY; do
        for pool in 64 256; do
            for mlip in atombit mace; do
                for optimizer in fire bfgs; do
                    for method in auto manual; do
                        task="$method:$mlip:$optimizer:$family:$pool"
                        if [[ "$task" == auto:atombit:bfgs:AXOSOW:64 ]] ||
                           [[ "$task" == manual:atombit:bfgs:AXOSOW:64 ]]; then
                            continue
                        fi
                        tasks+=("$task")
                    done
                done
            done
        done
    done
    while (( index < ${#tasks[@]} )); do
        pids=()
        for gpu in "${devices[@]}"; do
            (( index >= ${#tasks[@]} )) && break
            run_batch_case "${tasks[index]}" "$gpu" &
            pids+=("$!")
            ((index += 1))
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    done
}

run_mps_matrix() {
    local tasks=() devices=(0 1 2 3) pids=() index=0
    local family pool mlip optimizer task
    for family in AXOSOW XAFPAY; do
        for pool in 64 256; do
            for mlip in atombit mace; do
                for optimizer in fire bfgs; do
                    task="$mlip:$optimizer:$family:$pool"
                    [[ "$task" == atombit:bfgs:AXOSOW:64 ]] && continue
                    tasks+=("$task")
                done
            done
        done
    done
    while (( index < ${#tasks[@]} )); do
        pids=()
        for gpu in "${devices[@]}"; do
            (( index >= ${#tasks[@]} )) && break
            run_mps_case "${tasks[index]}" "$gpu" &
            pids+=("$!")
            ((index += 1))
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    done
}

run_batch_matrix &
batch_pid=$!
run_mps_matrix &
mps_pid=$!
wait "$batch_pid"
wait "$mps_pid"
