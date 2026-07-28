#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/atombit_blockwise_refill}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/T2_test/structures}"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"
CHECKPOINT="/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"

run_case() {
    local workload="$1"
    local mode="$2"
    local device="$3"
    local manifest method refill_interval check_interval

    case "${workload}" in
        H46)
            manifest="${ROOT}/benchmarks/workloads/manifests/OPT-H46-R256-v1.json"
            ;;
        STEPVAR)
            manifest="${ROOT}/benchmarks/workloads/manifests/OPT-STEPVAR-ATOMBIT-R256-v1.json"
            ;;
        *)
            echo "unknown workload: ${workload}" >&2
            return 2
            ;;
    esac

    case "${mode}" in
        active)
            method=active
            refill_interval=1
            check_interval=1
            ;;
        immediate)
            method=refill
            refill_interval=1
            check_interval=1
            ;;
        frozen_k5)
            method=refill
            refill_interval=5
            check_interval=1
            ;;
        block_k5)
            method=refill
            refill_interval=5
            check_interval=5
            ;;
        *)
            echo "unknown mode: ${mode}" >&2
            return 2
            ;;
    esac

    "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest "${manifest}" \
        --dataset-dir "${DATASET_DIR}" \
        --batch-size 64 \
        --refill-storage slots \
        --refill-interval "${refill_interval}" \
        --convergence-check-interval "${check_interval}" \
        --device "cuda:${device}" \
        --fmax 0.05 \
        --max-steps 500 \
        --skin 0.5 \
        --linear-algebra-backend auto \
        --checkpoint "${CHECKPOINT}" \
        --cpu-threads 1 \
        --deterministic \
        --profile-runtime \
        --output "${OUTPUT_DIR}/${workload}_${mode}.json" \
        >"${OUTPUT_DIR}/${workload}_${mode}.log" 2>&1
}

pids=()
run_case H46 active 0 & pids+=("$!")
run_case H46 immediate 1 & pids+=("$!")
run_case H46 frozen_k5 2 & pids+=("$!")
run_case H46 block_k5 3 & pids+=("$!")
run_case STEPVAR active 4 & pids+=("$!")
run_case STEPVAR immediate 5 & pids+=("$!")
run_case STEPVAR frozen_k5 6 & pids+=("$!")

for pid in "${pids[@]}"; do
    wait "${pid}"
done

run_case STEPVAR block_k5 0
