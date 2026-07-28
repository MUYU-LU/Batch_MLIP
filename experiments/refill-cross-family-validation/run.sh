#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/refill_cross_family_validation}"
DATASET="${DATASET:-/public/home/lmy/Batch_imple_project/test_set}"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"

run_case() {
    local family="$1"
    local method="$2"
    local device="$3"
    local storage manifest

    case "${family}" in
        GUFJOG44|XATMOV88|XAFPAY172|OBEQIX220|ROFB296)
            storage=slots
            ;;
        ROFA-MIX)
            storage=repack
            ;;
        *)
            echo "unknown family: ${family}" >&2
            return 2
            ;;
    esac
    manifest="${ROOT}/benchmarks/workloads/manifests/OPT-RB-${family}-R256-v1.json"

    "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest "${manifest}" \
        --dataset-dir "${DATASET}" \
        --batch-size 64 \
        --refill-storage "${storage}" \
        --refill-interval 1 \
        --convergence-check-interval 1 \
        --device "cuda:${device}" \
        --fmax 0.05 \
        --max-steps 500 \
        --skin 0.5 \
        --linear-algebra-backend auto \
        --cpu-threads 1 \
        --deterministic \
        --output "${OUTPUT_DIR}/${family}_${method}.json" \
        >"${OUTPUT_DIR}/${family}_${method}.log" 2>&1
}

run_wave() {
    local first="$1"
    local second="$2"
    local third="$3"
    local -a pids=()

    run_case "${first}" active 0 & pids+=("$!")
    run_case "${first}" refill 1 & pids+=("$!")
    run_case "${second}" active 2 & pids+=("$!")
    run_case "${second}" refill 3 & pids+=("$!")
    run_case "${third}" active 4 & pids+=("$!")
    run_case "${third}" refill 5 & pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

run_wave GUFJOG44 XATMOV88 XAFPAY172
run_wave OBEQIX220 ROFB296 ROFA-MIX
