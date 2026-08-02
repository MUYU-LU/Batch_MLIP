#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
DATASET="${DATASET:-/public/home/lmy/Batch_imple_project/test_set}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/omc-csp-compatible-refill-v1/results/raw}"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"
MANIFEST_ROOT="${ROOT}/experiments/refill-offline-predictor/workloads/manifests"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"
git -C "${ROOT}" rev-parse HEAD >"${OUTPUT_DIR}/baseline-commit.txt"
git -C "${ROOT}" diff --binary | sha256sum >"${OUTPUT_DIR}/dirty-tree.sha256"

run_case() {
    local family="$1"
    local batch="$2"
    local mode="$3"
    local gpu="$4"
    local method storage policy min_chunk

    case "${mode}" in
        active)
            method=active
            storage=repack
            policy=immediate
            ;;
        fifo_slots)
            method=refill
            storage=slots
            policy=immediate
            ;;
        compatible_slots)
            method=refill
            storage=compatible_slots
            policy=immediate
            ;;
        threshold_arena)
            method=refill
            storage=arena
            policy=threshold
            ;;
        *)
            printf 'unknown mode: %s\n' "${mode}" >&2
            return 2
            ;;
    esac
    min_chunk=$(( (batch * 15 + 99) / 100 ))

    "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest \
        "${MANIFEST_ROOT}/OPT-RF-U256-${family}-R256-v1.json" \
        --dataset-dir "${DATASET}" \
        --batch-size "${batch}" \
        --refill-storage "${storage}" \
        --refill-policy "${policy}" \
        --refill-low-watermark 0.7 \
        --refill-min-chunk "${min_chunk}" \
        --refill-interval 1 \
        --convergence-check-interval 1 \
        --device "cuda:${gpu}" \
        --fmax 0.05 \
        --max-steps 500 \
        --skin 0.5 \
        --linear-algebra-backend auto \
        --cpu-threads 1 \
        --deterministic \
        --profile-runtime \
        --output "${OUTPUT_DIR}/${family}_B${batch}_${mode}.json" \
        >"${OUTPUT_DIR}/${family}_B${batch}_${mode}.log" 2>&1
}

run_family_wave() {
    local family="$1"
    local -a pids=()
    local gpu=0
    local batch mode
    for batch in 64 128; do
        for mode in active fifo_slots compatible_slots threshold_arena; do
            run_case "${family}" "${batch}" "${mode}" "${gpu}" &
            pids+=("$!")
            gpu=$((gpu + 1))
        done
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

run_family_wave SOXLEX48
run_family_wave ROFA-MIX
