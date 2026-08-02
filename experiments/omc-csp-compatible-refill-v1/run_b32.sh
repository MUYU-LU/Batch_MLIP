#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
OUTPUT_DIR="${ROOT}/experiments/omc-csp-compatible-refill-v1/results/raw"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"
MANIFEST="${ROOT}/experiments/refill-offline-predictor/workloads/manifests/OPT-RF-U256-ROFA-MIX-R256-v1.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTPUT_DIR}"

run_case() {
    local mode="$1" gpu="$2" method storage policy
    case "${mode}" in
        active) method=active; storage=repack; policy=immediate ;;
        fifo_slots) method=refill; storage=slots; policy=immediate ;;
        compatible_slots) method=refill; storage=compatible_slots; policy=immediate ;;
        threshold_arena) method=refill; storage=arena; policy=threshold ;;
    esac
    "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit --method "${method}" --optimizer bfgs \
        --workload-manifest "${MANIFEST}" \
        --dataset-dir /public/home/lmy/Batch_imple_project/test_set \
        --batch-size 32 --refill-storage "${storage}" --refill-policy "${policy}" \
        --refill-low-watermark 0.7 --refill-min-chunk 5 \
        --device "cuda:${gpu}" --fmax 0.05 --max-steps 500 --skin 0.5 \
        --linear-algebra-backend auto --cpu-threads 1 --deterministic \
        --profile-runtime \
        --output "${OUTPUT_DIR}/ROFA-MIX_B32_${mode}.json" \
        >"${OUTPUT_DIR}/ROFA-MIX_B32_${mode}.log" 2>&1
}

pids=()
gpu=0
for mode in active fifo_slots compatible_slots threshold_arena; do
    run_case "${mode}" "${gpu}" &
    pids+=("$!")
    gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done
