#!/bin/bash
#SBATCH -A m4958
#SBATCH -q shared_interactive
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J clevr_bench
#SBATCH -o logs/clevr_bench_%j.out
#SBATCH -e logs/clevr_bench_%j.err

cd /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer
mkdir -p logs

/pscratch/sd/d/danieltm/conda_envs/envs/influencerformer/bin/python examples/run_clevr_boxes.py
