#!/usr/bin/env bash
set -euo pipefail

# GemmPali mandatory preflight dataset staging script.
# HDD (source): /mnt/ripped_media/GemmPali/datasets
# NVMe (target): /home/nate/GemmPali/nvme_cache (default)

HDD_ROOT="${HDD_ROOT:-/mnt/ripped_media/GemmPali/datasets}"
NVME_ROOT="${NVME_ROOT:-/home/nate/GemmPali/nvme_cache}"
PHASE="${1:-phase1}"

mkdir -p "$NVME_ROOT/raw" "$NVME_ROOT/processed" "$NVME_ROOT/cache"

stage_dir() {
  local src="$1"
  local dst="$2"
  echo "[stage] $src -> $dst"
  mkdir -p "$(dirname "$dst")"
  rsync -a --delete "$src" "$dst"
}

case "$PHASE" in
  phase1)
    stage_dir "$HDD_ROOT/raw/vidore-colpali-train-set/" "$NVME_ROOT/raw/vidore-colpali-train-set/"
    stage_dir "$HDD_ROOT/raw/vidore-docvqa-train/" "$NVME_ROOT/raw/vidore-docvqa-train/"
    ;;
  phase2)
    stage_dir "$HDD_ROOT/raw/mpdocvqa-corpus/" "$NVME_ROOT/raw/mpdocvqa-corpus/"
    stage_dir "$HDD_ROOT/raw/mpdocvqa-qa/" "$NVME_ROOT/raw/mpdocvqa-qa/"
    stage_dir "$HDD_ROOT/raw/dude-corpus/" "$NVME_ROOT/raw/dude-corpus/"
    stage_dir "$HDD_ROOT/raw/dude-qa/" "$NVME_ROOT/raw/dude-qa/"
    stage_dir "$HDD_ROOT/raw/vidore-colpali-train-set/" "$NVME_ROOT/raw/vidore-colpali-train-set/"
    ;;
  phase3)
    stage_dir "$HDD_ROOT/raw/mpdocvqa-corpus/" "$NVME_ROOT/raw/mpdocvqa-corpus/"
    stage_dir "$HDD_ROOT/raw/mpdocvqa-qa/" "$NVME_ROOT/raw/mpdocvqa-qa/"
    stage_dir "$HDD_ROOT/raw/dude-corpus/" "$NVME_ROOT/raw/dude-corpus/"
    stage_dir "$HDD_ROOT/raw/dude-qa/" "$NVME_ROOT/raw/dude-qa/"
    # CGI supplemental hard-negative stress set (optional copy when needed)
    if [[ -d "$HDD_ROOT/raw/cgi-corpus" ]]; then
      stage_dir "$HDD_ROOT/raw/cgi-corpus/" "$NVME_ROOT/raw/cgi-corpus/"
    fi
    ;;
  all)
    for d in "$HDD_ROOT/raw"/*; do
      [[ -d "$d" ]] || continue
      bn="$(basename "$d")"
      stage_dir "$d/" "$NVME_ROOT/raw/$bn/"
    done
    ;;
  *)
    echo "Unknown phase: $PHASE"
    echo "Usage: $0 [phase1|phase2|phase3|all]"
    exit 1
    ;;
esac

cat <<EOF

[done] Staging complete.
Set these env vars before training:
  export GEMMPALI_NVME_ROOT="$NVME_ROOT"
  export HF_HOME="$NVME_ROOT/cache/hf_home"
  export HF_DATASETS_CACHE="$NVME_ROOT/cache/hf_datasets"
  export TRANSFORMERS_CACHE="$NVME_ROOT/cache/hf_transformers"

EOF
