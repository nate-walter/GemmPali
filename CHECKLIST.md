# GemmPali CHECKLIST (SOTA Multi-Page Program)

Updated: 2026-03-02

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
- [x] Launch Phase 2 long-horizon run (`phase2_run3`) from run2 best checkpoint with 20k steps for pattern analysis.
- [x] Validate long-run settings for `phase2_run3`: lr=5e-5, eval_batches=12, fixed_eval, top-2 checkpoint retention.

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
  - [x] Phase 3 hard-negative crucible data pass prepared (MP-DocVQA/DUDE-heavy + warmup cap).
  - [x] Phase 3 hard-negative crucible run completed (`phase3_run1`, 16000/16000) and champion promoted (step 12500).
- [ ] Add Gate 1 and Gate 2 go/no-go checks and failover logic.

## Phase 4 — Evaluation vs ColQwen (general, not CGI-only)
- [ ] Build apples-to-apples benchmark harness for multi-page retrieval.
- [ ] Include mixed-domain holdout (scientific/docs/forms/reports), not only CGI.
- [ ] Track: Hit@k, MRR, NDCG, latency, VRAM, CPCR-like cross-page consistency.
- [ ] Add failure taxonomy + remediation loop.
- [~] Produce scoreboard artifacts and reproducible run cards (bootstrap scoreboard created; retrieval metrics harness run pending).

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



## 2026-03-01 execution burst
- [x] Promote Phase 2 champion (`phase2_run3` step 18000) to `checkpoints/champions/current_champion.pt`.
- [x] Build Phase 3 crucible pairs (`nvme_cache/processed/phase3/phase3_pairs.jsonl`).
- [x] Launch Phase 3 training from champion (`scripts/launch_phase3_crucible.sh`, log `/tmp/gemmpali/phase3_run1.log`).
- [x] Stand up head-to-head scoreboard anchor (`reports/head_to_head_scoreboard.{json,md}`).


## 2026-03-01 phase3 closeout
- [x] Confirm Phase 3 run completion at 16,000/16,000 (`SMOKE_OK`).
- [x] Promote best Phase 3 checkpoint to current champion (`phase3_run1` step 12500).
- [x] Hard harness confirms Phase 2 checkpoint-18000 remains best retrieval champion.
- [x] Refresh README status + CHECKLIST + scoreboard artifacts.
- [~] Run full retrieval harness (Hit@k, MRR, NDCG, CPCR, latency, VRAM) vs ColQwen and publish results. (surrogate harness complete for Phase2/3/3.1/3.2; true ColQwen apples-to-apples still pending (next after phase3_3_runB_long))


## 2026-03-01 phase3.1 corrective + verdict
- [x] Build corrective dataset (nvme_cache/processed/phase3_1/phase3_pairs.jsonl, warmup 18.03%).
- [x] Run Phase 3.1 corrective (phase3_1_run1, 4000 steps, LR 2e-5).
- [x] Run hard harness comparison across Phase2 / Phase3 / Phase3.1.
- [x] Re-point current_champion.pt to Phase2 step 18000 based on hard-harness performance.


## 2026-03-01 phase3.2 long-run + verdict
- [x] Launch Phase 3.2 long run (phase3_2_run1) at 20,000 steps from Phase2 champion init.
- [x] Confirm run completion at 20,000/20,000 (SMOKE_OK).
- [x] Evaluate retained best checkpoint (step 11500, eval_loss 0.0007724) with hard harness.
- [x] Compare hard harness against Phase2 / Phase3 / Phase3.1 baselines.
- [x] Keep current_champion.pt on Phase2 step 18000 (best quality+latency tradeoff).


## 2026-03-01 deepthink-r3 corrective launch
- [x] Read and applied docs/deepthink/deepthink-response-3.md directives.
- [x] Updated scripts/train_phase1_smoke.py with temperature, intra-doc negatives, scheduler/warmup, explicit grad clamp args.
- [x] Built Run-B dataset (nvme_cache/processed/phase3_3_runB/phase3_3_pairs.jsonl) at 80k pairs.
- [x] Mix locked: 25% ViDoRe anchor + 75% intra-doc crucible.
- [x] Launched phase3_3_runB_long (20,000 steps) from Phase2 champion init.
- [x] Complete run + benchmark with scaled raw-image harness and evaluate promotion gate. (Run-B benchmark completed; no promotion)


## 2026-03-01 deepthink-r3 runB completion
- [x] Run-B long completed (phase3_3_runB_long, 20000/20000, SMOKE_OK).
- [x] Retained best checkpoint set captured (step16000, step5500).
- [x] Hard harness comparison vs Phase2 champion completed.
- [x] Promotion decision: rejected (Phase2 remains champion).


## 2026-03-01 deepthink-r4 runC launch
- [x] Consolidated Run-B failure packet and asked DeepThink for hard-call follow-up.
- [x] Implemented new trainer: scripts/train_phase3_full.py (hybrid loss + grad value clipping + in-batch negatives).
- [x] Added launcher: scripts/launch_phase3_4_runC_unshackled.sh.
- [x] Applied exact run config from response-4 (5e-6, cosine warmup, negatives-per-query=7, max_len=8192).
- [x] Launched phase3_4_runC_unshackled from Phase2 champion init.
- [x] Complete run + run hard harness vs champion + make promote/hold decision.
- [x] Decision: HOLD Phase2 champion (phase2_run3 step 18000). RunC underperformed on Hit@1/MRR/NDCG in hard harness.

## 2026-03-02 deepthink-r5 phase4 vision rescue
- [x] Read and accepted DeepThink response-5 forensic diagnosis.
- [x] Added Phase4 vision trainer (`scripts/train_phase4_vision.py`) with real image pipeline + MaxSim + LoRA.
- [x] Added conservative launcher (`scripts/launch_phase4_runD_vision.sh`) for 2x3090.
- [x] Added raw-image retrieval harness (`scripts/run_retrieval_harness_raw_image.py`) with no surrogate text docs.
- [x] Added operator runbook (`reports/phase4_vision_rescue_runbook.md`).
- [x] Static compile validation passed for new scripts.
- [x] README/CHECKLIST updated and committed.
- [x] Launch Phase4 Run-D vision training and monitor first 250-step smoke gate.

