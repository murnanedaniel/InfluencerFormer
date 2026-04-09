#!/usr/bin/env bash
#SBATCH -A m4958
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH -J clevr_ablations
#SBATCH -o /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer/results/slurm_%j.out
#SBATCH -e /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer/results/slurm_%j.err

set -euo pipefail

REPO=/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer
cd "${REPO}"

export PYTHON=/global/homes/d/danieltm/.conda/envs/influencer/bin/python
export N_EPOCHS=${N_EPOCHS:-100}
export BATCH_SIZE=${BATCH_SIZE:-4096}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/results}

echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4 | tr '\n' '  ')"
echo "Epochs:     ${N_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

bash scripts/launch_4gpu.sh
