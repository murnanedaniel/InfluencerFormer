#!/usr/bin/env bash
# Launch all COCO scaling + tau ablation runs on a single interactive 4-GPU node.
# This script is meant to run ON the compute node (via srun from salloc).
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
SCRIPT="${REPO}/benchmarks/coco_detr/train.py"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

cd "${REPO}"
mkdir -p "${LOG_DIR}"

run_batch() {
  local -n configs=$1
  local batch_name=$2
  echo "=== ${batch_name} (${#configs[@]} runs) ==="
  local pids=()
  for idx in "${!configs[@]}"; do
    config="${configs[$idx]}"
    gpu=$((idx % 4))
    name="${config%.json}"
    log_path="${LOG_DIR}/${name}.log"
    echo "  GPU ${gpu}: ${name} -> ${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${SCRIPT}" \
      --config "${CONFIG_DIR}/${config}" \
      > "${log_path}" 2>&1 &
    pids+=("$!")
  done
  wait "${pids[@]}"
  echo "=== ${batch_name} complete ==="
}

# Batch 1: scaling study (4 runs, 4 GPUs)
BATCH1=(
  baseline_hungarian_medium.json
  pm3_softmatch_medium.json
  baseline_hungarian_large.json
  pm3_softmatch_large.json
)
run_batch BATCH1 "Batch 1: Scaling study"

# Batch 2: tau ablation (3 runs, 3 GPUs)
BATCH2=(
  pm3_tau005_small.json
  pm3_tau008_small.json
  pm3_tau020_small.json
)
run_batch BATCH2 "Batch 2: Tau ablation"

echo "All runs finished!"
