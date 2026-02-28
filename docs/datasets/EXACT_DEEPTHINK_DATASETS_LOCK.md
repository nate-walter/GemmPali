# Exact DeepThink Datasets Lock (2026-02-28)

This run is locked to the exact approved set for GemmPali Phase-2:

## Locked datasets

1. `vidore/colpali_train_set`
2. `vidore/docvqa_train`
3. `AHS-uni/mpdocvqa-corpus`
4. `AHS-uni/mpdocvqa-qa`
5. `AHS-uni/dude-corpus`
6. `AHS-uni/dude-qa`

## Important note on DocVQA split naming

DeepThink suggested `vidore/docvqa_train_corpus` and `vidore/docvqa_train_queries`.
We verified on HuggingFace Hub that these exact repo IDs are not available.
Canonical ViDoRe source in this environment is:

- `vidore/docvqa_train`

(Any corpus/query splitting is handled in our preprocessing layer, not by separate HF repos.)

## Storage policy

- Persistent datasets: HDD (`/mnt/ripped_media/GemmPali/datasets/raw/*`)
- Active training preflight: stage required phase data to NVMe (`/home/nate/GemmPali/nvme_cache`) using `scripts/stage_phase_to_nvme.sh`

## Setup command

```bash
scripts/setup_exact_deepthink_datasets.sh
```
