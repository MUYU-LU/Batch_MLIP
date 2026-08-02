#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/public/home/lmy/.conda/envs/lmy/bin/python}"
PROJECT="/public/home/lmy/Batch_imple_project"
WORKLOADS="${PROJECT}/omc_csp_scheduler_workloads_v2"
PROFILES="${PROJECT}/omc_csp_scheduler_planning_profiles_v2"
DATASET="${PROJECT}/test_set"
CHECKPOINT="${PROJECT}/AtomBit-OMC-s/checkpoints/smooth_rms_finetune/AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
MPS_CONFIG="${ROOT}/experiments/t2-unique6000-auto-bfgs-vs-mps8/mps_sessions.json"
OUTPUT="${ROOT}/experiments/omc-csp-workflow-validation-v1/results"
AUTO="${ROOT}/benchmarks/benchmark_omc_csp_scheduler_auto.py"
MPS="${ROOT}/benchmarks/run_omc_csp_scheduler_mps.py"

export BATCH_MLIP_DETERMINISTIC_ALGORITHMS=1
export BATCH_MLIP_DETERMINISTIC_WARN_ONLY=0
export BATCH_MLIP_REPRODUCIBILITY_SEED=20260729
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=20260729
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/mps-mix-runtime"
git -C "${ROOT}" rev-parse HEAD >"${OUTPUT}/baseline-commit.txt"
git -C "${ROOT}" diff --binary | sha256sum >"${OUTPUT}/dirty-tree.sha256"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader >"${OUTPUT}/nvidia-smi-before.csv"

run_auto() {
    local workload="$1"
    local devices="$2"
    local output_name="$3"
    "${PYTHON}" "${AUTO}" \
        --manifest "${WORKLOADS}/manifests/${workload}.json" \
        --dataset-dir "${DATASET}" \
        --checkpoint "${CHECKPOINT}" \
        --planning-profile "${PROFILES}/${workload}.json" \
        --devices "${devices}" \
        --output "${OUTPUT}/${output_name}.json" \
        --materialization manifest_lazy \
        --tail-recovery none \
        >"${OUTPUT}/logs/${output_name}.log" 2>&1
}

run_auto \
    OPT-OMC-SCHED-E1-TEST-JAYDUI-P64-INTRA-NARROW-v2 \
    0 \
    auto-jaydui-p64-g1

run_auto \
    OPT-OMC-SCHED-E1-TEST-MIX-ALL-P512-INTER-WIDE-v2 \
    0,1,2,3,4,5,6,7 \
    auto-mix-p512-g8

"${PYTHON}" "${MPS}" \
    --manifest "${WORKLOADS}/manifests/OPT-OMC-SCHED-E1-TEST-MIX-ALL-P512-INTER-WIDE-v2.json" \
    --dataset-dir "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --gpus 0,1,2,3,4,5,6,7 \
    --runtime-dir "${OUTPUT}/mps-mix-runtime" \
    --output "${OUTPUT}/mps-mix-p512-g8-w8.json" \
    --workers-per-gpu 8 \
    --mps-session-config "${MPS_CONFIG}" \
    >"${OUTPUT}/logs/mps-mix-p512-g8-w8.log" 2>&1

run_auto \
    OPT-OMC-SCHED-E1-TEST-OBEQIX-P2048-INTRA-NARROW-v2 \
    0,1,2,3,4,5,6,7 \
    auto-obeqix-p2048-g8

nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader >"${OUTPUT}/nvidia-smi-after.csv"

"${PYTHON}" "${ROOT}/experiments/omc-csp-workflow-validation-v1/summarize.py" \
    --results "${OUTPUT}" \
    --manifest-dir "${WORKLOADS}/manifests" \
    --frozen-auto-reference \
    "${PROJECT}/omc_csp_scheduler_epoch3/results/freeze_validation_v1/standalone/OPT-OMC-SCHED-E1-TEST-MIX-ALL-P512-INTER-WIDE-v2.json" \
    --output "${OUTPUT}/summary.json"
