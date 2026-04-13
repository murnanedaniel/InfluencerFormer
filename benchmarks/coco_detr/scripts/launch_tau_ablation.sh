#!/usr/bin/env bash
# Submit PM3 temperature ablation runs (tau=0.05, 0.08, 0.20) on small subset.
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  pm3_tau005_small.json
  pm3_tau008_small.json
  pm3_tau020_small.json
)

for config in "${CONFIGS[@]}"; do
  name="${config%.json}"
  echo "Submitting ${name}..."
  sbatch \
    --qos=shared \
    --constraint=gpu \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --time=00:30:00 \
    --account=m4958_g \
    --job-name="${name}" \
    --output="${LOG_DIR}/${name}.log" \
    --wrap="cd ${REPO} && /global/homes/d/danieltm/.conda/envs/influencer/bin/python -u benchmarks/coco_detr/train.py --config ${CONFIG_DIR}/${config}"
done

echo "All tau ablation jobs submitted."
