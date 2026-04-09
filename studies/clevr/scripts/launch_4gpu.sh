#!/usr/bin/env bash
set -euo pipefail

REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
PYTHON="${PYTHON:-/global/homes/d/danieltm/.conda/envs/influencer/bin/python}"
SCRIPT="${REPO}/studies/clevr/scripts/train_study.py"
RESULTS_DIR="${OUTPUT_DIR:-${REPO}/studies/clevr/results}"
LOG_DIR="${RESULTS_DIR}/logs"
N_EPOCHS="${N_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PATIENCE="${PATIENCE:-30}"
VAL_SAMPLES="${VAL_SAMPLES:-3000}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"

mkdir -p "${LOG_DIR}"

STUDIES=(
  tau_sweep
  power_sweep
  data_scale
  slot_embeddings
  lr_schedule
  warmstart
  softdcd
  model_scale
  hungarian_baseline
  cost_comparison
  combined_recipe
  pm3_best_recipe
)

pids=()
for idx in "${!STUDIES[@]}"; do
  study="${STUDIES[$idx]}"
  gpu=$((idx % 4))
  log_path="${LOG_DIR}/${study}.log"
  echo "Launching ${study} on GPU ${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}"     --study "${study}"     --n_epochs "${N_EPOCHS}"     --batch_size "${BATCH_SIZE}"     --patience "${PATIENCE}"     --val_samples "${VAL_SAMPLES}"     --train_samples "${TRAIN_SAMPLES}"     --output_dir "${RESULTS_DIR}"     > "${log_path}" 2>&1 &
  pids+=("$!")
  if (( (${#pids[@]} % 4) == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi
