#!/usr/bin/env bash
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="${PYTHON:-/global/homes/d/danieltm/.conda/envs/influencer/bin/python}"
SCRIPT="${REPO}/studies/clevr/scripts/train_pm3_grid_point.py"
RESULTS_DIR="${OUTPUT_DIR:-${REPO}/studies/clevr/results/pm3_grid}"
LOG_DIR="${RESULTS_DIR}/logs"
N_EPOCHS="${N_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PATIENCE="${PATIENCE:-30}"
VAL_SAMPLES="${VAL_SAMPLES:-3000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"

mkdir -p "${LOG_DIR}"

POINTS=(
  "0.12 0 0"
  "0.12 0 1"
  "0.12 1 0"
  "0.12 1 1"
  "0.15 0 0"
  "0.15 0 1"
  "0.15 1 0"
  "0.15 1 1"
)

pids=()
for idx in "${!POINTS[@]}"; do
  read -r tau slotted warmup <<< "${POINTS[$idx]}"
  gpu=$((idx % 4))
  stem="pm3_tau${tau}_slots${slotted}_warmup${warmup}"
  log_path="${LOG_DIR}/${stem}.log"
  echo "Launching ${stem} on GPU ${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}"     --tau "${tau}"     --slotted "${slotted}"     --warmup_cosine "${warmup}"     --n_epochs "${N_EPOCHS}"     --batch_size "${BATCH_SIZE}"     --patience "${PATIENCE}"     --val_samples "${VAL_SAMPLES}"     --train_samples "${TRAIN_SAMPLES}"     --output_dir "${RESULTS_DIR}"     > "${log_path}" 2>&1 &
  pids+=("$!")
  if (( (${#pids[@]} % 4) == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi
