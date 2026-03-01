#!/usr/bin/env bash
set -euo pipefail

cd /home/nate/GemmPali
source /home/nate/ColPali/colpali-us-vlm-multipage/.venv-usvlm/bin/activate
source ./scripts/train_env_nvme.sh >/dev/null

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="phase2_run1"
ARCHIVE_DIR="/mnt/ripped_media/GemmPali/checkpoint_archive/${RUN_NAME}"
GEMMA_LOCAL="/home/nate/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
PAIRS="/home/nate/GemmPali/nvme_cache/processed/phase2/phase2_pairs.jsonl"

# Phase 2: multi-page corpora curriculum mix (MP-DocVQA + DUDE + sampled warmup)
torchrun --standalone --nproc_per_node=2 scripts/train_phase1_smoke.py \
  --pairs "$PAIRS" \
  --model "$GEMMA_LOCAL" \
  --steps 8000 \
  --max-len 128 \
  --dtype bf16 \
  --lr 8e-5 \
  --log-every 20 \
  --eval-every 100 \
  --eval-batches 3 \
  --save-every 500 \
  --save-dir /home/nate/GemmPali/checkpoints/${RUN_NAME} \
  --keep-top-k 2 \
  --archive-dir "$ARCHIVE_DIR" \
  --load-in-4bit
