# ColPali US VLM Multi-Page — Execution Checklist

Updated: 2026-02-27

## Phase 0 — Foundation (today)
- [x] Re-read DeepThink response + build contract docs
- [x] Confirm repo scaffold and experiment configs exist
- [x] Lock implementation contract (what ships in v0)

## Phase 1 — Baseline bring-up (Gemma 3 without surgery)
- [x] Create `scripts/setup_env.sh` for deterministic env setup
- [x] Create `scripts/smoke_forward.py` (single-page + multi-page forward pass)
- [~] Install base model and verify local load/inference (Gemma 3 12B fully pulled to local HF cache; non-dummy GPU load deferred until GPU window opens)
- [x] Record baseline memory + latency on 1/2/4 pages (GPU3 verified after DeepThink patch; 4p pass at seq-len 64)

## Phase 2 — Torch surgery scaffolding
- [x] Create `src/rope3d.py` (Volumetric3DRoPE module)
- [x] Create `src/masks.py` (omnidirectional doc-index mask + causal query mask)
- [x] Create `src/retriever_head.py` (128D projection + pooling + L2 norm)
- [x] Create `src/model_wrapper.py` (Gemma load + patch injection points)
- [ ] Build `scripts/validate_shapes.py` and assert all expected tensor shapes

## Phase 3 — Index/query path split
- [x] Implement `is_document_indexing=True` path (omni mask)
- [x] Implement query path (causal-safe)
- [ ] Add `scripts/index_docs.py` prototype for multi-page embeddings
- [ ] Add `scripts/query_maxsim.py` prototype for retrieval scoring

## Phase 4 — Training plan kickoff
- [ ] Add contrastive warmup script skeleton (`scripts/train_contrastive.py`)
- [ ] Add RL stub (`scripts/train_grpo.py`) with reward decomposition placeholders
- [ ] Add ablation runner (`scripts/run_ablation.py`) for 1/2/4/10 page degradation

## Phase 5 — Go/No-Go gates
- [ ] Gate 1: 10-page forward pass under target VRAM
- [ ] Gate 2: Baseline retrieval sanity above random with stable gradients
- [ ] Gate 3: Cross-page benchmark improvement vs internal baseline

## Immediate next actions (Nate-approved)
- [~] Install model + dependencies for first smoke run (deps installed; model cached locally; real GPU smoke started 2026-02-28, follow-up fixes in progress)
- [x] Wire and test torch script integration path (`src/model_wrapper.py` + `scripts/smoke_forward.py`)


## Progress update (2026-02-28)
- ✅ Re-opened project docs (`README.md`, `CHECKLIST.md`) and resumed in strict task order.
- ✅ Real GPU smoke execution started.
- ⚠️ Blockers observed during smoke path:
  - OOM at higher token settings.
  - `Half` vs `float` dtype mismatch in retriever-head path.
- 🔧 Next patch order: stabilize smoke config (token/memory), then fix retriever dtype path, then re-run and capture baseline metrics (1/2/4 pages).


### GPU3 results (2026-02-28)
- Device lock: `CUDA_VISIBLE_DEVICES=3`
- Model: `google/gemma-3-12b-it`, dtype `bf16`, seq-len 128
- 1 page: PASS — 13.88s, peak VRAM 22.88 GB
- 2 pages: PASS — 13.57s, peak VRAM 22.94 GB
- 4 pages: FAIL — OOM (additional 16 MiB requested at ~23.52 GiB used)


### DeepThink patch verification (2026-02-28)
- Patched files: `src/rope3d.py`, `src/masks.py`, `src/retriever_head.py`, `src/model_wrapper.py`
- GPU lock: `CUDA_VISIBLE_DEVICES=3`
- Verified non-dummy multi-page forward with Gemma 3 12B:
  - 4 pages @ seq-len 64 ✅ (`peak_vram_gb` 22.94)
  - 6 pages @ seq-len 64 ✅ (`peak_vram_gb` 23.00)
- Reports:
  - `Reports/deepthink_patch_gpu3_p4_seq64.json`
  - `Reports/deepthink_patch_gpu3_p6_seq64.json`

### Verification harness (2026-02-28)
- ✅ Added `scripts/multipage_probe_test.py` (GemPali multi-page retrieval probe).
- ✅ Ran on GPU3 (`CUDA_VISIBLE_DEVICES=3`), Gemma 3 12B, non-dummy.
- ✅ Report: `Reports/multipage_probe_report_gpu3.json`
- ✅ Result: 6/6 correct (2p, 4p, 6p; 2 trials each)
