#!/usr/bin/env bash
set -euo pipefail

NVME_ROOT="${GEMMPALI_NVME_ROOT:-/home/nate/GemmPali/nvme_cache}"
mkdir -p "$NVME_ROOT/cache/hf_home" "$NVME_ROOT/cache/hf_datasets" "$NVME_ROOT/cache/hf_transformers"

export GEMMPALI_NVME_ROOT="$NVME_ROOT"
export HF_HOME="$NVME_ROOT/cache/hf_home"
export HF_DATASETS_CACHE="$NVME_ROOT/cache/hf_datasets"
export TRANSFORMERS_CACHE="$NVME_ROOT/cache/hf_transformers"

# Optional: dataset consumers can use this as canonical root
export GEMMPALI_DATA_ROOT="$NVME_ROOT/raw"

echo "[env] GEMMPALI_NVME_ROOT=$GEMMPALI_NVME_ROOT"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[env] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "[env] GEMMPALI_DATA_ROOT=$GEMMPALI_DATA_ROOT"
