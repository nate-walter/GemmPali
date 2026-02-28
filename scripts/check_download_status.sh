#!/usr/bin/env bash
set -euo pipefail
ROOT="${GEMMPALI_HDD_ROOT:-/mnt/ripped_media/GemmPali/datasets/raw}"

echo "== sizes =="
for p in mpdocvqa-corpus mpdocvqa-qa dude-corpus dude-qa vidore-colpali-train-set vidore-docvqa-train; do
  if [[ -d "$ROOT/$p" ]]; then
    du -sh "$ROOT/$p"
  else
    echo "MISSING: $ROOT/$p"
  fi
done

echo "== active hf download processes =="
pgrep -af "hf download" || true
