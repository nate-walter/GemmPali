# ColPali US VLM Multi-Page

Frontier project to build an American VLM contender for multi-page ColPali-style retrieval and analysis.

## Goal
Beat current multi-page retrieval limitations in US VLMs via architecture/training adaptation.

## Structure
- `Documents/DeepThink-docs/` — DeepThink strategy docs and related references
- `src/` — implementation code
- `scripts/` — training/eval utilities
- `configs/` — experiment configs
- `data/` — local dataset metadata/manifests (no large blobs committed)

## Setup
```bash
python3 -m venv .venv-usvlm
source .venv-usvlm/bin/activate
pip install -U pip
cp .env.example .env
# fill Gemini + provider keys in .env
```

### Gemini project wiring
The project is wired to use environment variables:
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`
- `GEMINI_API_KEY_NAME`
- `GOOGLE_PROJECT_NAME` (format: `projects/<number>`)
- `GOOGLE_PROJECT_NUMBER`

## Notes
Initial DeepThink response reference is stored in:
- `Documents/DeepThink-docs/DeepThink-response-link.md`

## Active execution docs
- CHECKLIST.md — live execution checklist (phase-by-phase)
- Documents/DeepThink-docs/IMPLEMENTATION-PLAN-2026-02-27.md — DeepThink-aligned build contract for code


## Progress update (2026-02-27)
- ✅ Environment + dependency setup complete (`.venv-usvlm`, transformers/accelerate stack).
- ✅ Torch integration scaffolding landed:
  - `src/model_wrapper.py`
  - `scripts/smoke_forward.py`
  - `scripts/setup_env.sh`
- ✅ Hugging Face access confirmed and **Gemma 3 12B** weights pulled to cache (~23G):
  - `~/.cache/huggingface/hub/models--google--gemma-3-12b-it`
- ✅ Gemini API key + project wiring added to `.env`/`.env.example` docs.
- ⏸️ Real GPU smoke load is intentionally deferred while training workloads occupy GPUs.



## Progress update (2026-02-28)
- ✅ Resumed smoke-test track with real GPU run window.
- ⚠️ Two blockers found in smoke path:
  1) OOM at higher token settings,
  2) dtype mismatch (`Half` vs `float`) in retriever path.
- 🔧 Active fix sequence: lower-memory smoke profile first, then dtype-consistent retriever forward, then baseline capture for 1/2/4 pages.


### GPU3 smoke results (2026-02-28)
- Ran on free GPU3 only (`CUDA_VISIBLE_DEVICES=3`).
- `seq-len=128`, `bf16`, Gemma 3 12B:
  - pages=1 ✅ (13.88s, peak 22.88 GB)
  - pages=2 ✅ (13.57s, peak 22.94 GB)
  - pages=4 ❌ OOM (needs memory reduction/offload/quantization for this profile)


## DeepThink exact patch set + verification (2026-02-28)
Implemented DeepThink-aligned module split and wrapper wiring:
- `src/rope3d.py` (Volumetric3DRoPE scaffold/hook)
- `src/masks.py` (doc-index omni mask + query causal mask)
- `src/retriever_head.py` (pool + 128D projection + L2 norm)
- `src/model_wrapper.py` wired to those modules

Verification (Gemma 3 12B, non-dummy, GPU3 only):
- 4 pages @ seq-len 64: PASS (`Reports/deepthink_patch_gpu3_p4_seq64.json`)
- 6 pages @ seq-len 64: PASS (`Reports/deepthink_patch_gpu3_p6_seq64.json`)

## GemPali verification harness (2026-02-28)
Built `scripts/multipage_probe_test.py` to verify true multi-page retrieval behavior after DeepThink patching.

Run command:
`PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 python scripts/multipage_probe_test.py --model google/gemma-3-12b-it --dtype bf16 --tokens-per-page 64 --trials 2 --pages 2 4 6 --out Reports/multipage_probe_report_gpu3.json`

Result:
- non-dummy execution (`using_dummy=false`)
- overall accuracy: 1.0 (6/6)
- page-level: 2p=100%, 4p=100%, 6p=100%
