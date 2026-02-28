# GemmPali CHECKLIST (SOTA Multi-Page Program)

Updated: 2026-02-28

## Mission lock
- [x] GemmPali is **NOT CGI-centric**.
- [x] Primary objective: beat ColQwen on **general multi-page indexing/retrieval** across diverse document types.
- [x] CGI corpus is treated as supplemental hard-negative / continuity stress data only.
- [x] AHS-uni MP-DocVQA/DUDE corpus+qa splits are locked as canonical.

## Phase 0 — Proven baseline (already complete)
- [x] DeepThink patch path implemented (masks/retriever_head/rope3d/wrapper wiring).
- [x] Multi-page non-dummy verification passed.
- [x] Fresh-repo reproducibility verification passed.

## Phase 1 — Dataset acquisition (HDD-first)
- [x] Create canonical HDD dataset root for GemmPali training corpora.
- [x] Download MP-DocVQA to HDD dataset root (AHS-uni split: corpus+qa complete).
- [x] Download DUDE to HDD dataset root (AHS-uni split: corpus+qa complete).
- [x] Confirm ViDoRe + DocVQA availability and normalize location map (locked: `vidore/colpali_train_set` + `vidore/docvqa_train`; corpus/query split IDs not present on HF).
- [x] Record dataset provenance + SHAs in manifest.
- [x] Stamp final dataset manifest snapshot (`dataset_manifest_final_2026-02-28.json`).
- [x] Document **mandatory** NVMe staging policy (copy-on-train preflight).
- [x] Run Phase 1 NVMe staging (only `vidore-colpali-train-set` + `vidore-docvqa-train`).

## Phase 2 — Data contract + preprocessing
- [x] Define Phase-1 canonical records via JSONL preprocessor script (`scripts/prepare_phase1_data.py`).
- [x] Build initial converter for ViDoRe/DocVQA Phase-1 (`docvqa_corpus.jsonl`, `docvqa_queries.jsonl`, `phase1_warmup_pairs.jsonl`).
- [ ] Extend converters for MP-DocVQA/DUDE canonical schema.
- [ ] Build CGI corpus converter as supplemental stress set only.
- [ ] Implement intra-document hard-negative mining (N vs N±1 same document).
- [ ] Implement temporal hard-negatives where layout is similar but values differ.
- [ ] Add quality gates (dedupe, leakage, malformed records, class/domain balance).

## Phase 1.5 — Distribution plan (locked)
- [x] Training distribution plan updated: **start on GPUs 3+4 (2-GPU DDP)** for Phase 1 stability.
- [x] Single-GPU fallback acknowledged (GPU 3 only) but not default.
- [ ] Run 100-step Phase 1 smoke on 2-GPU plan.

## Phase 3 — Training pipeline
- [ ] Add/author `train_gemmpali_phase2.py` (or equivalent) with DeepThink phase structure.
- [ ] Add config for DeepSpeed ZeRO-3 + bf16 + grad checkpointing.
- [ ] Implement phase curriculum:
  - [ ] Phase 1 warmup (ViDoRe + DocVQA)
  - [ ] Phase 2 volumetric expansion (MP-DocVQA + DUDE + balanced supplemental CGI)
  - [ ] Phase 3 hard-negative crucible (multi-page stress with strict hard negatives)
- [ ] Add Gate 1 and Gate 2 go/no-go checks and failover logic.

## Phase 4 — Evaluation vs ColQwen (general, not CGI-only)
- [ ] Build apples-to-apples benchmark harness for multi-page retrieval.
- [ ] Include mixed-domain holdout (scientific/docs/forms/reports), not only CGI.
- [ ] Track: Hit@k, MRR, NDCG, latency, VRAM, CPCR-like cross-page consistency.
- [ ] Add failure taxonomy + remediation loop.
- [ ] Produce scoreboard artifacts and reproducible run cards.

## Ops / guardrails
- [x] Check in after each major step.
- [x] Keep all long-term datasets on HDD.
- [x] Use NVMe only as temporary training cache, staged per phase.
- [x] Keep dataset manifest + provenance in `docs/datasets/`.
- [x] Maintain exact DeepThink dataset lock file (`docs/datasets/EXACT_DEEPTHINK_DATASETS_LOCK.md`).
- [x] Never reframe mission into domain-specialist tuning.
