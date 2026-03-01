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
- [x] Run 100-step Phase 1 smoke on 2-GPU plan.
- [x] Launch full Phase 1 warmup run on GPUs 3+4 (run1 completed 5k/5k at `checkpoints/phase1/head_step_0005000.pt`; run2 completed 5k/5k at `checkpoints/phase1_run2/head_step_0005000.pt`).

## Phase 2.5 — Next phase execution
- [x] Stage Phase-2 datasets to NVMe (`stage_phase_to_nvme.sh phase2`).
- [x] Build Phase-2 training pairs from AHS MP-DocVQA + DUDE + warmup carryover (`prepare_phase2_data.py`).
- [x] Launch Phase 2 training run on GPUs 3+4 (`phase2_run1`, 8000 steps) with top-2 checkpoint retention.
- [x] Monitor early-run quality and complete full run to 8k.
- [x] Phase 2 run1 post-run assessment recorded (best retained checkpoint = step 4500).
- [x] Launch Phase 2 refinement run (`phase2_run2`) from step-4500 head init with stronger eval reliability (completed 6000/6000; best retained checkpoint = step 5000).
- [~] Launch Phase 2 long-horizon run (`phase2_run3`) from run2 best checkpoint with 20k steps for pattern analysis.
- [ ] Validate long-run settings for `phase2_run3`: lr=5e-5, eval_batches=12, fixed_eval, top-2 checkpoint retention.

## Phase 2.6 — Contingency gates (if Phase 2 underperforms)
- [ ] If eval spikes >2x floor for 3+ eval windows, continue to next checkpoint boundary before intervention.
- [ ] Roll back to best checkpoint using `checkpoints/<run>/checkpoint_index.json` (top-2 policy).
- [ ] Rebalance phase2 pair mix (lower warmup carryover, increase MP-DocVQA/DUDE hard examples) and regenerate pairs.
- [ ] Lower LR step-down (e.g., `8e-5 -> 6e-5`) if grad_norm bursts persist.
- [ ] Increase eval reliability (`--eval-batches` up, fixed holdout) before go/no-go decisions.
- [ ] Promote phase only when eval trend is stable/improving across multiple windows.

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

## Phase 1.6 — Observability / dashboard
- [x] Build dedicated GemmPali metrics dashboard with cyberpunk UI and metric hints.
- [x] Parse live run telemetry from sigma logs + checkpoints + GPU 3/4 runtime stats.
- [x] Add fast-change config file (`dashboard/config/run.json`) for path/process edits during crashes/restarts.
- [x] Add one-command start/stop scripts for quick spin-up.
- [x] Added dedicated ColQwen dashboard start/stop scripts (`start_colqwen_dashboard.sh`, `stop_colqwen_dashboard.sh`).
- [x] Build companion iPhone Expo app (`GemmPaliMobile`) with compact KPI cards + smooth horizontal trend lines.
- [x] Copy/retheme a separate ColQwen app (`ColQwenMonsterOpsMobile`) with independent API target.
- [x] Add axis-labeled mobile charts and expanded metrics panels for train loss, eval loss, learning rate, and grad norm (telemetry appears when run logs include those fields).

## Ops / guardrails
- [x] Check in after each major step.
- [x] Expo iPhone launch note recorded: paste `exp://<LAN-IP>:8081` in Safari, then open in Expo Go.
- [x] Keep all long-term datasets on HDD.
- [x] Use NVMe only as temporary training cache, staged per phase.
- [x] Keep dataset manifest + provenance in `docs/datasets/`.
- [x] Maintain exact DeepThink dataset lock file (`docs/datasets/EXACT_DEEPTHINK_DATASETS_LOCK.md`).
- [x] Enforce checkpoint retention policy (best top-2 in active run dir; archive extras to HDD).
- [x] Never reframe mission into domain-specialist tuning.
