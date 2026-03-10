#!/usr/bin/env bash
set -euo pipefail

# Phase 5 Track A2 (Globalized) -- 3-GPU DDP
# A2 lock: batch-size 1 + gradient accumulation 4 (collision-safe pressure lane).
# GPUs: 0,3,4

cd /home/nate/GemmPali
source /home/nate/grpo-env/bin/activate
source ./scripts/train_env_nvme.sh >/dev/null

RUN_NAME="phase5_trackA2_global_3gpu"
PAIRS="/home/nate/GemmPali/nvme_cache/processed/phase5_global/phase5_unrolled_mix.jsonl"
MODEL="/home/nate/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
INIT_HEAD="/mnt/ripped_media/GemmPali/checkpoint_archive/phase5_trackA1_global_3gpu/head_step_0002000.pt"

SAVE_DIR="/home/nate/GemmPali/checkpoints/${RUN_NAME}"
ARCHIVE_DIR="/mnt/ripped_media/GemmPali/checkpoint_archive/${RUN_NAME}"
LOG="/home/nate/GemmPali/reports/${RUN_NAME}.log"
mkdir -p "$SAVE_DIR" "$ARCHIVE_DIR" /home/nate/GemmPali/reports

export CUDA_VISIBLE_DEVICES=0,3,4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=3 scripts/train_phase4_vision.py \
  --pairs "$PAIRS" \
  --model "$MODEL" \
  --steps 8000 \
  --max-len 4096 \
  --dtype bf16 \
  --load-in-4bit \
  --lr 2e-5 \
  --scheduler cosine_with_warmup \
  --warmup-steps 300 \
  --temperature 0.05 \
  --max-grad-norm 1.0 \
  --max-grad-value 0.1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --negatives-per-query 3 \
  --intra-doc-negatives true \
  --in-batch-negatives true \
  --eval-every 250 \
  --eval-batches 8 \
  --log-every 10 \
  --save-every 500 \
  --save-dir "$SAVE_DIR" \
  --keep-top-k 2 \
  --archive-dir "$ARCHIVE_DIR" \
  --cache-dir /home/nate/GemmPali/nvme_cache/vision_cache \
  --debug-vision-every 25 \
  --init-head "$INIT_HEAD" \
  | tee "$LOG"

echo "Run complete: $RUN_NAME"