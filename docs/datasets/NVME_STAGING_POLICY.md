# GemmPali NVMe Staging Policy (Mandatory)

## Rule
Persistent datasets live on HDD:
- `/mnt/ripped_media/GemmPali/datasets/raw/*`

Active training datasets must be staged to NVMe before training:
- default NVMe root: `/home/nate/GemmPali/nvme_cache`

## Why
Phase 2/3 multi-page batches (8-16 pages per sequence) require high random read throughput.
Streaming directly from HDD can starve GPUs and collapse utilization.

## Preflight
1. Run staging script for phase:
   - `scripts/stage_phase_to_nvme.sh phase1|phase2|phase3`
2. Load NVMe env vars:
   - `source scripts/train_env_nvme.sh`
3. Ensure training configs read data from `$GEMMPALI_DATA_ROOT` and cache from HF vars.

## Post-run cleanup
- Keep HDD canonical data unchanged.
- NVMe cache/data may be pruned after run completion unless marked for immediate reuse.
