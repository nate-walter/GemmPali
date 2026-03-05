# GemmPali (Multi-Page Gemma Retrieval Program)

GemmPali is a Gemma-based multi-page retrieval system pursuing **general-document** SOTA in ColPali-style indexing/retrieval.

## Mission statement

GemmPali is **not a CGI-specialist model**.

Our objective is broad multi-page dominance vs ColQwen across diverse document regimes (reports, technical docs, forms, layouts, cross-page tables/charts), with CGI annual reports used only as one supplemental stress domain.

## Current status


## Fresh update (2026-03-05, 08:05 EST)


## Seed stability update (2026-03-05, 09:55 EST)

Corrected raw-image harness seed sweep completed (3 seeds, 100 samples each) on `head_step_0008000.pt`:

- Hit@1 mean: **0.1967** (std 0.0330, min 0.15, max 0.22)
- Hit@5 mean: **0.4267** (std 0.0499, min 0.36, max 0.48)
- MRR@10 mean: **0.2994** (std 0.0251, min 0.2705, max 0.3316)
- NDCG@10 mean: **0.3708** (std 0.0217, min 0.3540, max 0.4015)
- CPCR mean: **0.4267** (std 0.0499, min 0.36, max 0.48)
- Latency mean: **10.69s/query** (std 1.21s)

Interpretation: quality gains are stable across seeds on the corrected test path; latency remains the primary optimization target.


- Phase4 Run-D vision completed at 8000/8000 and passed corrected raw-image harness evaluation.
- Corrected harness result (step 8000) vs prior step-1000 baseline:
  - Hit@1: **0.185** (from 0.050, +0.135)
  - Hit@5: **0.445** (from 0.150, +0.295)
  - MRR@10: **0.29796** (from 0.10681, +0.19115)
  - NDCG@10: **0.37096** (from 0.15374, +0.21721)
  - CPCR: **0.445** (from 0.150, +0.295)
- Net: substantial retrieval-quality lift confirmed on the corrected test path.
- Caveat: latency is materially higher in this heavy raw-image setting (~9.93s/query avg).

## Fresh update (2026-03-02, 11:20 EST)

- DeepThink response-5 forensic diagnosis accepted: Phase3 surrogate-text path was bypassing true vision learning.
- Phase4 vision rescue implemented (additive, non-destructive):
  - scripts/train_phase4_vision.py
  - scripts/launch_phase4_runD_vision.sh
  - scripts/run_retrieval_harness_raw_image.py
  - reports/phase4_vision_rescue_runbook.md
- Core architectural corrections in Phase4 trainer:
  - real image loading via PIL + AutoProcessor (no query+answer surrogate docs in forward path)
  - sequence-preserving late interaction (no mean-pool collapse)
  - MaxSim-style scoring for positives + in-batch + hard negatives
  - LoRA adapters on backbone attention modules (q_proj/k_proj/v_proj/o_proj) with trainable head
- Phase2 champion remains current retrieval champion until raw-image harness promotion gate is cleared.
- Next action: launch Phase4 Run-D vision training (conservative defaults), then evaluate on raw-image harness.

## Fresh update (2026-03-01, 21:47 EST)

- DeepThink response-4 directives applied in full (Run-C unshackled path).
- New trainer implemented: scripts/train_phase3_full.py
  - hybrid objective: positive + in-batch negatives + hard negatives
  - new controls: max_grad_value, in_batch_negatives, negatives_per_query, cosine warmup scheduler
  - keeps checkpoint top-k retention + archive behavior
- New launcher implemented: scripts/launch_phase3_4_runC_unshackled.sh
  - init: Phase2 champion (phase2_run3/head_step_0018000.pt)
  - run data: nvme_cache/processed/phase3_3_runB/phase3_3_pairs.jsonl
  - steps: 8,000
  - lr: 5e-6, warmup 500, cosine_with_warmup
  - max_len: 8192, temperature 0.05, max_grad_norm 1.0, max_grad_value 0.1
  - negatives: intra-doc=true, in-batch=true, hard negatives/query=7
- Run status: phase3_4_runC_unshackled launched; promotion decision remains benchmark-gated.

### Completed
- DeepThink-driven patch path is implemented and running:
  - doc/query mask split
  - retriever head pooling + 128D + L2
  - 3D-RoPE scaffold integration
  - wrapper wiring + dtype consistency
- Multi-page behavior validated in non-dummy path.
- Fresh-repo re-verification passed from clean checkout/venv.

### Active program
- Transition from prototype validation -> full SOTA training/eval campaign.
- Primary dataset strategy now centered on open multi-page corpora with strict hard-negative training.

### Dataset milestone (2026-02-28)
- Exact DeepThink-locked 6-dataset set downloaded to HDD and stamped.
- Final manifest snapshot:
  - `/mnt/ripped_media/GemmPali/datasets/manifests/dataset_manifest_final_2026-02-28.json`
- Phase 1 NVMe staging completed (strict payload only):
  - `vidore-colpali-train-set` (~50G)
  - `vidore-docvqa-train` (~6.7G)
- Phase 1 preprocessing started/completed on staged NVMe data:
  - `docvqa_corpus.jsonl` (10,189 docs)
  - `docvqa_queries.jsonl` (39,463 queries)
  - `phase1_warmup_pairs.jsonl` (118,695 pairs)
  - output root: `/home/nate/GemmPali/nvme_cache/processed/phase1`

## Dataset policy (critical)

### HDD-first storage policy
All training corpora are stored under HDD roots for space management and reproducibility.

- Long-term datasets: **HDD only**
- NVMe usage: **mandatory preflight stage for active training phase only**
- Run phase staging script before each training launch (Phase 1/2/3).
- After training run: clear staged NVMe copies unless explicitly retained.
- **Checkpoint retention policy:** keep only best **top-2** checkpoints in active run directory; archive the rest to HDD under `/mnt/ripped_media/GemmPali/checkpoint_archive/<run_name>/`.

This policy is non-negotiable for operational stability.

## Dataset strategy baseline

Canonical dataset plan is now locked to:
- `vidore/colpali_train_set`
- `vidore/docvqa_train`
- `AHS-uni/mpdocvqa-corpus` + `AHS-uni/mpdocvqa-qa`
- `AHS-uni/dude-corpus` + `AHS-uni/dude-qa`
- Internal CGI annual-report corpus (supplemental stress domain, not central identity)

(We are intentionally not using alternate MP-DocVQA/DUDE distributions unless explicitly re-approved.)

## Distribution plan

Phase 1 warmup default is now **2-GPU DDP on GPUs 3+4** for stability and throughput.

Current run profile:
- 4-bit backbone loading + frozen-backbone no-grad forward
- train head path with checkpoint saves every 500 steps
- initial full warmup target: 5000 steps
- top-2 checkpoint retention active (best 2 kept in run dir, others archived to HDD)
- status:
  - Phase 1 run1 completed at 5000/5000 (active dir retains best: `head_step_0003000.pt`, `head_step_0004000.pt`)
  - Phase 1 run2 completed at 5000/5000 (active dir retains best: `head_step_0003000.pt`, `head_step_0004000.pt`)
  - Phase 2 run1 completed at 8000/8000 on GPUs 3+4 using MP-DocVQA + DUDE + warmup carryover (`nvme_cache/processed/phase2/phase2_pairs.jsonl`)
  - Phase 2 run1 highlights:
    - final step metrics: train `0.0002`, eval `0.0014`, grad_norm `0.0123`, lr `8e-5`
    - best retained checkpoint: `head_step_0004500.pt` (eval `0.000121`)
    - eval was low overall but noisy/spiky at times (expected for early multi-page hard-negative curriculum)
  - Phase 2 run2 completed at 6000/6000 (refinement from run1 best checkpoint)
    - final step line had eval spike; retention correctly kept best checkpoints
    - best retained checkpoint: `phase2_run2/head_step_0005000.pt` (eval `0.00338`)
  - Phase 3.2 run1 completed at 20000/20000 (long-run corrective from Phase2 champion)
    - best retained checkpoint: phase3_2_run1/head_step_0011500.pt (eval 0.0007724)
    - second retained checkpoint: phase3_2_run1/head_step_0014500.pt (eval 0.0019670)
  - Phase 3.3 Run B long started from Phase2 champion (DeepThink r3 contract)
    - run: phase3_3_runB_long (target 20,000 steps)
    - data: phase3_3_runB/phase3_3_pairs.jsonl (25% ViDoRe + 75% intra-doc)
  - Phase 3.3 Run B long completed at 20000/20000
    - best retained checkpoint: phase3_3_runB_long/head_step_0016000.pt (eval 0.0116015)
    - second retained checkpoint: phase3_3_runB_long/head_step_0005500.pt (eval 0.0117919)

- Default launch lane: `CUDA_VISIBLE_DEVICES=3,4`
- Single-GPU mode (`GPU 3` only) is fallback-only for constrained windows.

## Training philosophy

To beat ColQwen at multi-page retrieval, GemmPali training emphasizes:
- cross-page continuity routing,
- strict in-document hard negatives,
- temporal/value-sensitive negatives for similar layouts,
- measurable multi-page consistency metrics.

## Evaluation philosophy

No single-domain overfitting claims.

Head-to-head success must be demonstrated on mixed-domain multi-page retrieval tasks with apples-to-apples controls and explicit cross-page metrics.

## Contingency plan (if current phase underperforms)

If Phase 2 does not hold quality, apply this in order (do not freestyle):

1. **Stability gate (during run)**
   - Trigger: eval loss rises >2x from recent floor for 3+ eval windows.
   - Action: continue until next checkpoint boundary; do not panic-stop on single spikes.

2. **Checkpoint rollback**
   - Action: promote best checkpoint from `checkpoint_index.json` (top-2 policy), not latest step by default.

3. **Data rebalance pass**
   - Trigger: persistent eval degradation after rollback.
   - Action: reduce warmup carryover share, increase MP-DocVQA/DUDE hard examples; regenerate phase pairs.

4. **Learning-rate/grad control**
   - Trigger: repeated grad_norm bursts + unstable eval.
   - Action: lower LR (e.g., 8e-5 -> 6e-5), keep clip norm, rerun from best checkpoint.

5. **Eval reliability hardening**
   - Trigger: train/eval disagreement with noisy point estimates.
   - Action: increase eval batches and fixed holdout slice before making promote/kill decisions.

6. **Promotion criteria**
   - Only advance to next phase if:
     - eval trend is stable/improving over multiple windows,
     - no sustained instability pattern,
     - top checkpoint beats prior phase baseline on held-out retrieval checks.

## Next attack plan (Phase 2 refinement)

Immediate follow-up long-horizon run is locked as `phase2_run3`:
- initialize head from best checkpoint: `checkpoints/phase2_run2/head_step_0005000.pt`
- keep same multi-page dataset mix (MP-DocVQA + DUDE + warmup carryover)
- run long pattern window: **20,000 steps**
- strengthen eval reliability (`eval_batches=12`, fixed holdout behavior)
- lower LR for long stability (`5e-5`)
- keep top-2 checkpoint retention and HDD archive policy
- objective: observe long-run train/eval/grad patterns and promote based on stable trend, not single-point wins

## GemmPali dashboard (new)

A dedicated cyberpunk training dashboard now exists under:
- `dashboard/` (Express backend + neon frontend)
- `GemmPaliMobile/` (Expo iPhone app for compact GemmPali telemetry)
- `colqwen-dashboard/` (separate dashboard API/frontend for ColQwen CGI Monster run)
- `ColQwenMonsterOpsMobile/` (Expo iPhone app themed for ColQwen Monster telemetry)

### What it tracks
- live train loss stream from run log
- eval loss (when run logging includes eval passes)
- learning rate and grad norm (when run logging includes optimizer telemetry)
- rolling-10 and rolling-50 loss curves
- checkpoint cadence + latest checkpoint
- spike count and low-loss streak
- GPU 3/4 utilization, memory, temperature, power
- run status (active / completed / idle)

### Fast changeability (for crashes/restarts)
- Edit `dashboard/config/run.json` for log path, checkpoint dir, process pattern, max steps, and target GPUs.
- No frontend rebuild needed for path changes — backend reads config each request.

### One-command control
- Start: `scripts/start_gemmpali_dashboard.sh`
- Stop: `scripts/stop_gemmpali_dashboard.sh`
- Default URL: `http://10.46.150.108:3472`

### iPhone app (Expo)
- Directory: `GemmPaliMobile/`
- Start dev server: `cd GemmPaliMobile && npx expo start --lan --clear`
- iOS app uses the same backend API (`http://10.46.150.108:3472/api/metrics`) with ATS relaxed in `app.json` for LAN HTTP.
- ✅ Working launch flow (confirmed):
  1. Copy Expo URL from terminal (`exp://<LAN-IP>:8081`)
  2. Paste URL into Safari on iPhone
  3. Tap **Open in Expo Go**


### ColQwen dashboard + iPhone app
- Dashboard URL: `http://10.46.150.108:3473`
- Dashboard scripts:
  - Start: `scripts/start_colqwen_dashboard.sh`
  - Stop: `scripts/stop_colqwen_dashboard.sh`
- Mobile app directory: `ColQwenMonsterOpsMobile/`
- Start dev server: `cd ColQwenMonsterOpsMobile && npx expo start --lan --clear`
- Open on iPhone: paste `exp://<LAN-IP>:8081` into Safari and tap **Open in Expo Go**

## Key reports

- `reports/DeepThink-Prompt-GemmPali-SOTA-MultiPage-Training-2026-02-28.md`
- `reports/DeepThink-Response-GemmPali-SOTA-MultiPage-2026-02-28.txt`
- `reports/GemmPali-Accomplishment-Report-2026-02-28.md`

## Operational checklist

See `CHECKLIST.md` for the active execution plan and gates.

## Fresh update (2026-03-02, 21:35 EST)

- DeepThink response-6 ingested and applied as a hard reset over Phase4 Run-D.
- Root findings from response-6 (confirmed):
  - wrapper path was being bypassed in parts of the training flow,
  - multimodal prompt/template + wrapper interface mismatch,
  - MaxSim masking/normalization stability risks,
  - eval-mode discipline required for trustworthy telemetry.
- Phase4 code-level reset applied:
  - `src/model_wrapper.py` forward updated to accept multimodal kwargs (`input_ids`, `attention_mask`, `pixel_values`, etc.)
  - `scripts/train_phase4_vision.py` updated with wrapper-based forward path, L2 normalization, masked MaxSim utilities, FP32 logits before temperature division, and eval `model.eval()/model.train()` discipline.
  - `scripts/launch_phase4_runD_vision.sh` reset config aligned to response-6 direction (`lr=2e-5`, `batch_size=2`, `negatives_per_query=1`, fallback notes).
- Run restarted from scratch:
  - New live log: `/home/nate/GemmPali/reports/phase4_runD_vision_current.log`
  - New run nohup: `phase4_runD_vision_nohup_20260302-212937.log`
- Existing iPhone Expo console remains the same URL/port (no app switch required):
  - Dashboard/API: `http://10.46.150.108:3474`
  - Expo Go: `exp://10.46.150.108:8084`

## Fresh update (2026-03-02, 23:36 EST)

- Phase4 Run-D (DeepThink r6 reset) encountered repeated OOM during multimodal vision tower forward.
- Confirmed fallback path execution:
  - `max_len` reduced from 4096 -> 2048.
  - `batch_size` reduced from 2 -> 1.
  - hard negatives remain enabled (`negatives_per_query=1`), intra-doc negatives remain enabled.
  - in-batch negatives still configured but effectively inactive at batch_size=1.
- Relaunched with fallback settings:
  - log: `/home/nate/GemmPali/reports/phase4_runD_vision_nohup_20260302-233422.log`
  - live pointer: `/home/nate/GemmPali/reports/phase4_runD_vision_current.log`
- Expo console remains unchanged:
  - Dashboard/API: `http://10.46.150.108:3474`
  - Expo Go: `exp://10.46.150.108:8084`

## Fresh update (2026-03-03, 17:50 EST)

- DeepThink response-7 applied (forensic follow-up on r6):
  - gradient checkpointing enabled in k-bit prep + explicit PEFT backbone checkpointing enable.
  - MaxSim score normalization by valid query length added in `batched_maxsim` and `hardneg_maxsim`.
- Attempted aggressive smoke per r7 target config:
  - `max_len=4096`, `batch_size=2`, `negatives_per_query=3`, in-batch negatives enabled.
- Outcome:
  - run failed with CUDA OOM during vision-tower forward path in smoke gate.
  - confirms this exact aggressive tuple currently exceeds practical memory budget on active hardware context.
- Current state:
  - Phase4 run not active after OOM fail.
  - next planned move: fallback from r7 aggression while preserving its math fixes (checkpointing + normalized MaxSim).

