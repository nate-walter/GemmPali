#!/usr/bin/env bash
set -euo pipefail

# Exact DeepThink dataset setup (canonical locked)
# Persistent root: HDD on Sigma
ROOT="${GEMMPALI_HDD_ROOT:-/mnt/ripped_media/GemmPali/datasets/raw}"
HF_BIN="${HF_BIN:-$HOME/.local/bin/hf}"

mkdir -p "$ROOT"

download_ds () {
  local repo="$1"
  local dir="$2"
  mkdir -p "$ROOT/$dir"
  echo "[download] $repo -> $ROOT/$dir"
  "$HF_BIN" download "$repo" --repo-type dataset --local-dir "$ROOT/$dir"
}

# DeepThink-required + approved canonical set
# 1) ViDoRe base warmup
# 2) ViDoRe DocVQA training source (single dataset repo)
# 3) AHS-uni MP-DocVQA split IR format
# 4) AHS-uni DUDE split IR format

download_ds "vidore/colpali_train_set" "vidore-colpali-train-set"
download_ds "vidore/docvqa_train" "vidore-docvqa-train"
download_ds "AHS-uni/mpdocvqa-corpus" "mpdocvqa-corpus"
download_ds "AHS-uni/mpdocvqa-qa" "mpdocvqa-qa"
download_ds "AHS-uni/dude-corpus" "dude-corpus"
download_ds "AHS-uni/dude-qa" "dude-qa"

echo "[done] Exact DeepThink datasets are present under: $ROOT"
