#!/usr/bin/env bash
set -euo pipefail

cd /home/nate/GemmPali
source /home/nate/ColPali/colpali-us-vlm-multipage/.venv-usvlm/bin/activate
source ./scripts/train_env_nvme.sh >/dev/null

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GEMMA_LOCAL="/home/nate/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

# Phase 1 warmup run (head-focused)
torchrun --standalone --nproc_per_node=2 scripts/train_phase1_smoke.py \
  --pairs /home/nate/GemmPali/nvme_cache/processed/phase1/phase1_warmup_pairs.jsonl \
  --model "$GEMMA_LOCAL" \
  --steps 5000 \
  --max-len 128 \
  --dtype bf16 \
  --lr 1e-4 \
  --log-every 20 \
  --save-every 500 \
  --save-dir /home/nate/GemmPali/checkpoints/phase1 \
  --load-in-4bit
