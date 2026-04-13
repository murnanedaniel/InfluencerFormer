#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --qos=interactive
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m4958_g
#SBATCH --job-name=coco_scaling
#SBATCH --output=benchmarks/coco_detr/runs/logs/slurm_%j.out
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="${PYTHON:-/global/homes/d/danieltm/.conda/envs/influencer/bin/python}"
SCRIPT="${REPO}/benchmarks/coco_detr/train.py"
CONFIG_DIR="${REPO}/benchmarks/coco_detr/configs"
LOG_DIR="${REPO}/benchmarks/coco_detr/runs/logs"

cd "${REPO}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  baseline_hungarian_medium.json
  pm3_softmatch_medium.json
  baseline_hungarian_large.json
  pm3_softmatch_large.json
)

pids=()
for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu=$((idx % 4))
  name="${config%.json}"
  log_path="${LOG_DIR}/${name}.log"
  echo "Launching ${name} on GPU ${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${SCRIPT}" \
    --config "${CONFIG_DIR}/${config}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

echo "Waiting for all ${#pids[@]} runs to finish..."
wait "${pids[@]}"
echo "All scaling runs complete."
