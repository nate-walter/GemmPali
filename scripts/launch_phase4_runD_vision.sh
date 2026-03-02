#!/usr/bin/env bash
set -euo pipefail

# Phase4 Vision Rescue (Run D)
# Conservative defaults for 2x3090 to avoid OOM.
# Start here first, then scale with comments at bottom.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="phase4_runD_vision"
PAIRS="nvme_cache/processed/phase3_3_runB/phase3_3_pairs.jsonl"
MODEL="google/gemma-3-4b-it"

SAVE_DIR="checkpoints/${RUN_NAME}"
ARCHIVE_DIR="/mnt/ripped_media/GemmPali/checkpoint_archive/${RUN_NAME}"
mkdir -p "$SAVE_DIR" "$ARCHIVE_DIR" reports

export CUDA_VISIBLE_DEVICES=3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="reports/${RUN_NAME}.log"

torchrun --nproc_per_node=2 scripts/train_phase4_vision.py \
  --pairs "$PAIRS" \
  --model "$MODEL" \
  --steps 8000 \
  --max-len 4096 \
  --dtype bf16 \
  --lr 4e-6 \
  --scheduler cosine_with_warmup \
  --warmup-steps 300 \
  --temperature 0.05 \
  --max-grad-norm 1.0 \
  --max-grad-value 0.1 \
  --batch-size 1 \
  --negatives-per-query 1 \
  --intra-doc-negatives true \
  --in-batch-negatives true \
  --eval-every 250 \
  --eval-batches 8 \
  --log-every 10 \
  --save-every 500 \
  --save-dir "$SAVE_DIR" \
  --keep-top-k 2 \
  --archive-dir "$ARCHIVE_DIR" \
  --cache-dir nvme_cache/vision_cache \
  --debug-vision-every 25 \
  | tee "$LOG"

cat <<'EOF'
Run completed.

Escalation path (only after stable run at defaults):
1) Increase negatives per query first:
   --negatives-per-query 3
2) Then cautiously increase batch size:
   --batch-size 2
3) If OOM appears, immediately revert to:
   --batch-size 1 --negatives-per-query 1
EOF
