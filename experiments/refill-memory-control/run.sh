#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/refill_memory_control}"
BENCHMARK="${ROOT}/benchmarks/benchmark_robustness_optimization.py"
DATASET="${ROOT}/data/T2_test/structures"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_DIR}"

run_case() {
    local name="$1"
    local workload="$2"
    local device="$3"
    local method="$4"
    local batch_size="$5"
    local backend="$6"
    local allocator_mode="$7"
    local manifest
    local -a allocator_environment

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

    case "${allocator_mode}" in
        dual)
            allocator_environment=(
                PYTORCH_ALLOC_CONF=expandable_segments:True
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        new_gc80)
            allocator_environment=(
                PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
            )
            ;;
        new_expandable)
            allocator_environment=(
                PYTORCH_ALLOC_CONF=expandable_segments:True
            )
            ;;
        old_expandable)
            allocator_environment=(
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        new_native_gc80)
            allocator_environment=(
                PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
            )
            ;;
        *)
            echo "unknown allocator mode: ${allocator_mode}" >&2
            return 2
            ;;
    esac

    env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF \
        "${allocator_environment[@]}" \
        "${PYTHON}" "${BENCHMARK}" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest "${manifest}" \
        --dataset-dir "${DATASET}" \
        --batch-size "${batch_size}" \
        --refill-storage slots \
        --refill-interval 1 \
        --convergence-check-interval 1 \
        --device "cuda:${device}" \
        --fmax 0.05 \
        --max-steps 500 \
        --skin 0.5 \
        --linear-algebra-backend "${backend}" \
        --cpu-threads 1 \
        --deterministic \
        --output "${OUTPUT_DIR}/${name}.json" \
        >"${OUTPUT_DIR}/${name}.log" 2>&1
}

pids=()
run_case dual_active_B64 STEPVAR 0 active 64 auto dual & pids+=("$!")
run_case dual_refill_B64 STEPVAR 1 refill 64 auto dual & pids+=("$!")
run_case allocator_gc80_B64 STEPVAR 2 refill 64 auto new_gc80 & pids+=("$!")
run_case auto_B48 STEPVAR 3 refill 48 auto new_expandable & pids+=("$!")
run_case deprecated_alias_B64 STEPVAR 4 refill 64 auto old_expandable & pids+=("$!")
run_case native_gc80_B64 STEPVAR 5 refill 64 auto new_native_gc80 & pids+=("$!")
run_case serial_B64 STEPVAR 6 refill 64 serial new_expandable & pids+=("$!")

for pid in "${pids[@]}"; do
    wait "${pid}"
done

pids=()
run_case auto_B32 STEPVAR 0 refill 32 auto new_expandable & pids+=("$!")
run_case dual_H46_active_B64 H46 1 active 64 auto dual & pids+=("$!")
run_case dual_H46_refill_B64 H46 2 refill 64 auto dual & pids+=("$!")

for pid in "${pids[@]}"; do
    wait "${pid}"
done
