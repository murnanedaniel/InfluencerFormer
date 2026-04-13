#!/usr/bin/env bash
# Submit 4 independent COCO scaling runs, each as a shared 1-GPU job.
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  baseline_hungarian_medium.json
  pm3_softmatch_medium.json
  baseline_hungarian_large.json
  pm3_softmatch_large.json
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
    --time=01:00:00 \
    --account=m4958_g \
    --job-name="${name}" \
    --output="${LOG_DIR}/${name}.log" \
    --wrap="cd ${REPO} && /global/homes/d/danieltm/.conda/envs/influencer/bin/python -u benchmarks/coco_detr/train.py --config ${CONFIG_DIR}/${config}"
done

echo "All jobs submitted. Check with: squeue -u \$USER"
