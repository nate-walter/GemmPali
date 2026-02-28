# GemmPali CHECKLIST (SOTA Multi-Page Program)

Updated: 2026-02-28

## Mission lock
- [x] GemmPali is **NOT CGI-centric**.
- [x] Primary objective: beat ColQwen on **general multi-page indexing/retrieval** across diverse document types.
- [x] CGI corpus is treated as supplemental hard-negative / continuity stress data only.

## Phase 0 — Proven baseline (already complete)
- [x] DeepThink patch path implemented (masks/retriever_head/rope3d/wrapper wiring).
- [x] Multi-page non-dummy verification passed.
- [x] Fresh-repo reproducibility verification passed.

## Phase 1 — Dataset acquisition (HDD-first)
- [ ] Create canonical HDD dataset root for GemmPali training corpora.
- [ ] Download MP-DocVQA to HDD dataset root.
- [ ] Download DUDE to HDD dataset root.
- [ ] Confirm ViDoRe + DocVQA availability and normalize location map.
- [ ] Record storage footprint and integrity checksums/manifests.
- [ ] Document ad-hoc NVMe staging policy (copy-on-train only).

## Phase 2 — Data contract + preprocessing
- [ ] Define canonical schema: `[query, image_sequence, target_page_idx, hard_negatives...]`.
- [ ] Build converters for ViDoRe/DocVQA/MP-DocVQA/DUDE to canonical schema.
- [ ] Build CGI corpus converter as supplemental stress set only.
- [ ] Implement intra-document hard-negative mining (N vs N±1 same document).
- [ ] Implement temporal hard-negatives where layout is similar but values differ.
- [ ] Add quality gates (dedupe, leakage, malformed records, class/domain balance).

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
- [ ] Keep all long-term datasets on HDD.
- [ ] Use NVMe only as temporary ad-hoc training cache.
- [ ] Keep dataset manifest + provenance in `docs/datasets/`.
- [ ] Never reframe mission into domain-specialist tuning.
