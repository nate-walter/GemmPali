#!/usr/bin/env bash
set -euo pipefail

cd /home/nate/GemmPali
source /home/nate/ColPali/colpali-us-vlm-multipage/.venv-usvlm/bin/activate
source ./scripts/train_env_nvme.sh >/dev/null

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="phase2_run3"
ARCHIVE_DIR="/mnt/ripped_media/GemmPali/checkpoint_archive/${RUN_NAME}"
GEMMA_LOCAL="/home/nate/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
PAIRS="/home/nate/GemmPali/nvme_cache/processed/phase2/phase2_pairs.jsonl"
INIT_HEAD="/home/nate/GemmPali/checkpoints/phase2_run2/head_step_0005000.pt"

# Long-horizon phase2 pattern run: start from best run2 checkpoint and watch 20k dynamics
# Newest fixes in effect:
# - fixed eval behavior
# - higher eval batch count
# - top-2 checkpoint retention with HDD archive
torchrun --standalone --nproc_per_node=2 scripts/train_phase1_smoke.py \
  --pairs "$PAIRS" \
  --model "$GEMMA_LOCAL" \
  --steps 20000 \
  --max-len 128 \
  --dtype bf16 \
  --lr 5e-5 \
  --log-every 20 \
  --eval-every 100 \
  --eval-batches 12 \
  --save-every 500 \
  --save-dir /home/nate/GemmPali/checkpoints/${RUN_NAME} \
  --keep-top-k 2 \
  --archive-dir "$ARCHIVE_DIR" \
  --init-head-checkpoint "$INIT_HEAD" \
  --fixed-eval \
  --load-in-4bit
