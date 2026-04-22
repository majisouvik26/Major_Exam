#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_train_eval.sh
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES=2 EPOCHS=15 RUN_NAME=unet-23class bash run_train_eval.sh

DATA_ROOT="${DATA_ROOT:-MLDLOPs_2026_Major_Exam}"
EPOCHS="${EPOCHS:-15}"
PROJECT="${PROJECT:-major-exam-segmentation}"
RUN_NAME="${RUN_NAME:-unet-23class}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/unet_segmentation_23cls.pt}"

mkdir -p logs "${OUTPUT_DIR}"

TRAIN_LOG="logs/train_${RUN_NAME}.log"
EVAL_LOG="logs/eval_${RUN_NAME}.log"

# If CUDA_VISIBLE_DEVICES is not set, default to GPU 2 as in your examples.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

echo "[INFO] Training started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee "${TRAIN_LOG}"
uv run python train.py \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --use_wandb \
  --wandb_project "${PROJECT}" \
  --wandb_run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${TRAIN_LOG}"

echo "[INFO] Sequential evaluation using checkpoint: ${CHECKPOINT}" | tee "${EVAL_LOG}"
uv run python train.py \
  --data_root "${DATA_ROOT}" \
  --eval_only \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${EVAL_LOG}"

echo "[INFO] Done."
echo "[INFO] Train log: ${TRAIN_LOG}"
echo "[INFO] Eval log: ${EVAL_LOG}"
