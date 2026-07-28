#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
DATASET="${DATASET:-/public/home/lmy/Batch_imple_project/test_set}"
CHECKPOINT="${CHECKPOINT:-/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt}"
OUTPUT="${OUTPUT:-${ROOT}/runs/refill_pool_multigpu_transfer}"
MANIFEST_DIR="${ROOT}/experiments/refill-pool-multigpu-transfer/workloads/manifests"
SPLIT="${SPLIT:-all}"
PHASE="${PHASE:-fit}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}"

run_single() {
    local family="$1"
    local pool="$2"
    local capacity="$3"
    local method="$4"
    local device="$5"
    local stem="${family}_R${pool}_B${capacity}_G1_${method}"
    "${PYTHON}" "${ROOT}/benchmarks/benchmark_robustness_optimization.py" \
        --mlip atombit \
        --method "${method}" \
        --optimizer bfgs \
        --workload-manifest "${MANIFEST_DIR}/OPT-RFT-${family}-R${pool}-v1.json" \
        --dataset-dir "${DATASET}" \
        --batch-size "${capacity}" \
        --refill-storage slots \
        --refill-interval 1 \
        --convergence-check-interval 1 \
        --device "cuda:${device}" \
        --fmax 0.05 \
        --max-steps 500 \
        --skin 0.5 \
        --linear-algebra-backend auto \
        --cpu-threads 1 \
        --deterministic \
        --checkpoint "${CHECKPOINT}" \
        --output "${OUTPUT}/${stem}.json" \
        >"${OUTPUT}/${stem}.log" 2>&1
}

run_single_wave() {
    local pool="$1"
    local capacity="$2"
    local swap="$3"
    local active_offset=0
    local refill_offset=1
    local -a pids=()
    if [[ "${swap}" == "true" ]]; then
        active_offset=1
        refill_offset=0
    fi
    local -a families
    if [[ "${PHASE}" == "fit" ]]; then
        families=(XATMOV88 XAFPAY172)
    elif [[ "${PHASE}" == "heldout" ]]; then
        families=(SOXLEX48)
    else
        echo "PHASE must be fit or heldout" >&2
        return 2
    fi
    local family base=0
    for family in "${families[@]}"; do
        case "${family}" in
            XATMOV88) base=0 ;;
            XAFPAY172) base=2 ;;
            SOXLEX48) base=0 ;;
        esac
        run_single "${family}" "${pool}" "${capacity}" active "$((base + active_offset))" &
        pids+=("$!")
        run_single "${family}" "${pool}" "${capacity}" refill "$((base + refill_offset))" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

run_multi() {
    local family="$1"
    local pool="$2"
    local capacity="$3"
    local method="$4"
    local devices="$5"
    local gpu_count
    gpu_count="$(awk -F, '{print NF}' <<<"${devices}")"
    local stem="${family}_R${pool}_B${capacity}_G${gpu_count}_${method}"
    "${PYTHON}" \
        "${ROOT}/experiments/refill-pool-multigpu-transfer/benchmark_multigpu.py" \
        --method "${method}" \
        --manifest "${MANIFEST_DIR}/OPT-RFT-${family}-R${pool}-v1.json" \
        --dataset-dir "${DATASET}" \
        --checkpoint "${CHECKPOINT}" \
        --devices "${devices}" \
        --resident-capacity "${capacity}" \
        --output "${OUTPUT}/${stem}.json" \
        >"${OUTPUT}/${stem}.log" 2>&1
}

if [[ "${SPLIT}" == "all" || "${SPLIT}" == "single" ]]; then
    run_single_wave 128 32 false
    run_single_wave 128 64 true
    run_single_wave 512 64 false
    run_single_wave 512 128 true
fi

if [[ "${SPLIT}" == "all" || "${SPLIT}" == "multi" ]]; then
    if [[ "${PHASE}" == "fit" ]]; then
        run_multi XATMOV88 512 64 active 0,1 &
        active_pid="$!"
        run_multi XATMOV88 512 64 refill 2,3 &
        refill_pid="$!"
        wait "${active_pid}"
        wait "${refill_pid}"
        run_multi XATMOV88 1024 64 active 0,1,2,3
        run_multi XATMOV88 1024 64 refill 0,1,2,3
    elif [[ "${PHASE}" == "heldout" ]]; then
        run_multi BOQWIN116 512 64 active 2,3 &
        active_pid="$!"
        run_multi BOQWIN116 512 64 refill 0,1 &
        refill_pid="$!"
        wait "${active_pid}"
        wait "${refill_pid}"
        run_multi BOQWIN116 1024 64 refill 0,1,2,3
        run_multi BOQWIN116 1024 64 active 0,1,2,3
    else
        echo "PHASE must be fit or heldout" >&2
        exit 2
    fi
fi
