#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
DATASET="${DATASET:-/public/home/lmy/Batch_imple_project/test_set}"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"
SPLIT="${SPLIT:-all}"
WORKLOAD_VARIANT="${WORKLOAD_VARIANT:-unique}"

case "${WORKLOAD_VARIANT}" in
    unique)
        OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/refill_offline_predictor_unique}"
        MANIFEST_DIR="${ROOT}/experiments/refill-offline-predictor/workloads/manifests"
        MANIFEST_PREFIX=OPT-RF-U256
        ;;
    repeated)
        OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/refill_offline_predictor}"
        MANIFEST_DIR="${ROOT}/benchmarks/workloads/manifests"
        MANIFEST_PREFIX=OPT-RB
        ;;
    *)
        echo "WORKLOAD_VARIANT must be unique or repeated" >&2
        exit 2
        ;;
esac

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
    local batch_size="$3"
    local device="$4"
    local storage manifest

    case "${family}" in
        GUFJOG44|XATMOV88|XAFPAY172|OBEQIX220|ROFB296|SOXLEX48|AXOSOW64|BOQWIN116)
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
    manifest="${MANIFEST_DIR}/${MANIFEST_PREFIX}-${family}-R256-v1.json"

    "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest "${manifest}" \
        --dataset-dir "${DATASET}" \
        --batch-size "${batch_size}" \
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
        --output "${OUTPUT_DIR}/${family}_B${batch_size}_${method}.json" \
        >"${OUTPUT_DIR}/${family}_B${batch_size}_${method}.log" 2>&1
}

run_wave() {
    local batch_size="$1"
    local swap_methods="$2"
    local first="$3"
    local second="$4"
    local third="$5"
    local active_device refill_device
    local -a pids=()

    if [[ "${swap_methods}" == "true" ]]; then
        active_device=1
        refill_device=0
    else
        active_device=0
        refill_device=1
    fi
    run_case "${first}" active "${batch_size}" "${active_device}" & pids+=("$!")
    run_case "${first}" refill "${batch_size}" "${refill_device}" & pids+=("$!")
    run_case "${second}" active "${batch_size}" "$((active_device + 2))" & pids+=("$!")
    run_case "${second}" refill "${batch_size}" "$((refill_device + 2))" & pids+=("$!")
    run_case "${third}" active "${batch_size}" "$((active_device + 4))" & pids+=("$!")
    run_case "${third}" refill "${batch_size}" "$((refill_device + 4))" & pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

if [[ "${SPLIT}" != "all" && "${SPLIT}" != "fit" && "${SPLIT}" != "heldout" ]]; then
    echo "SPLIT must be all, fit, or heldout" >&2
    exit 2
fi

for batch_size in 32 64 128; do
    swap_methods=false
    if [[ "${batch_size}" == "64" ]]; then
        swap_methods=true
    fi
    if [[ "${SPLIT}" == "all" || "${SPLIT}" == "fit" ]]; then
        run_wave "${batch_size}" "${swap_methods}" GUFJOG44 XATMOV88 XAFPAY172
        run_wave "${batch_size}" "${swap_methods}" OBEQIX220 ROFB296 ROFA-MIX
    fi
    if [[ "${SPLIT}" == "all" || "${SPLIT}" == "heldout" ]]; then
        run_wave "${batch_size}" "${swap_methods}" SOXLEX48 AXOSOW64 BOQWIN116
    fi
done
