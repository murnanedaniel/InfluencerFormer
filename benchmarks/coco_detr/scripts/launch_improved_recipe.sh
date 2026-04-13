#!/usr/bin/env bash
# Improved recipe: 20 epochs, separate backbone LR, cosine schedule.
# Run ON the compute node via srun from salloc.
set -uo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
SCRIPT="${REPO}/benchmarks/coco_detr/train.py"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

cd "${REPO}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  hungarian_medium_20ep.json
  pm3_medium_20ep.json
  hungarian_large_20ep.json
  pm3_large_20ep.json
)

echo "=== Improved recipe: 20ep, backbone LR, cosine (4 runs) ==="
pids=()
for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu=$((idx % 4))
  name="${config%.json}"
  log_path="${LOG_DIR}/${name}.log"
  echo "  GPU ${gpu}: ${name} -> ${log_path}"
  ( CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${SCRIPT}" \
    --config "${CONFIG_DIR}/${config}" || true ) \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
echo "=== All improved recipe runs complete ==="
