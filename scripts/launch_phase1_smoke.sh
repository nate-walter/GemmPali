#!/usr/bin/env bash
set -euo pipefail

cd /home/nate/GemmPali
source /home/nate/ColPali/colpali-us-vlm-multipage/.venv-usvlm/bin/activate
source ./scripts/train_env_nvme.sh >/dev/null

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONPATH=.

torchrun --standalone --nproc_per_node=2 scripts/train_phase1_smoke.py \
  --pairs /home/nate/GemmPali/nvme_cache/processed/phase1/phase1_warmup_pairs.jsonl \
  --model google/gemma-3-12b-it \
  --steps 100 \
  --max-len 128 \
  --dtype bf16 \
  --lr 1e-4 \
  --log-every 10
