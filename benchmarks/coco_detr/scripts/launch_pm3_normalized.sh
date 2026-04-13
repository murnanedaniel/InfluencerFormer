#!/usr/bin/env bash
# PM3 with Lp-norm normalization runs. Run ON the compute node.
set -uo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="/global/homes/d/danieltm/.conda/envs/influencer/bin/python"
SCRIPT="${REPO}/benchmarks/coco_detr/train.py"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

cd "${REPO}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  pm3_norm_medium_20ep.json
  pm3_norm_large_20ep.json
)

echo "=== PM3 normalized runs (2 runs) ==="
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
echo "=== PM3 normalized runs complete ==="
