# GemmPali (Multi-Page Gemma Retrieval Prototype)

**GemmPali** (two Ms = Multi-page) is a reproducible prototype showing that a US VLM stack (Gemma 3 12B) can be patched and verified for **multi-page retrieval behavior**.

## What this repo contains

- `src/`
  - `model_wrapper.py` (patched wrapper wiring)
  - `masks.py` (doc-index omni mask + query causal mask)
  - `retriever_head.py` (pool + 128D projection + L2 normalize)
  - `rope3d.py` (Volumetric3DRoPE scaffold hook)
- `scripts/`
  - `smoke_forward.py` (forward-pass smoke)
  - `multipage_probe_test.py` (multi-page retrieval verification harness)
- `configs/`
  - `exp_A.yaml`, `exp_B.yaml`, `exp_C.yaml`
- `docs/verification/`
  - JSON evidence from successful runs
- `docs/deepthink/`
  - DeepThink docs that guided the implementation (PDF exports)
- `environment/`
  - `.env.example`
  - `requirements-freeze.txt`

## Upstream implementation repo

Primary implementation landed in:

- `nate-walter/colpali-us-vlm-multipage` (branch: `main`)
- Commit: `cddd8ea`
- Commit title: `feat: DeepThink multi-page patch + GemPali verification harness`

## Reproduction

### 1) Environment

- Python 3.10.x
- Install deps from `environment/requirements-freeze.txt`

### 2) GPU pinning

Use free GPU explicitly:

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=.
```

### 3) Run deepthink smoke checks

```bash
python scripts/smoke_forward.py \
  --model google/gemma-3-12b-it \
  --dtype bf16 \
  --seq-len 64 \
  --pages 4 \
  --batch-size 1 \
  --allow-dummy \
  --out docs/verification/deepthink_patch_gpu3_p4_seq64.json
```

### 4) Run GemmPali multi-page verification harness

```bash
python scripts/multipage_probe_test.py \
  --model google/gemma-3-12b-it \
  --dtype bf16 \
  --tokens-per-page 64 \
  --trials 2 \
  --pages 2 4 6 \
  --out docs/verification/multipage_probe_report_gpu3.json
```

## Key result

From `docs/verification/multipage_probe_report_gpu3.json`:

- `using_dummy = false`
- overall accuracy = **1.0 (6/6)**
- page-level:
  - 2 pages: 100%
  - 4 pages: 100%
  - 6 pages: 100%

## Source links

- DeepThink shared conversation:
  - https://g.co/gemini/share/cc72325be710
- DeepThink response doc #1:
  - https://docs.google.com/document/d/1szArxsvLtTvj37d7lbFudaRS7RLm3e6NVmX3xqaZFa0/edit?usp=drivesdk
- DeepThink response doc #2:
  - https://docs.google.com/document/d/1tg-b5t41aMj3XDR0NSzN26dItvZh5BEKpMkkEpN6hSs/edit?usp=drivesdk
