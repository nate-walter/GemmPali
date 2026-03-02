# GemmPali (Multi-Page Gemma Retrieval Program)

GemmPali is a Gemma-based multi-page retrieval system pursuing **general-document** SOTA in ColPali-style indexing/retrieval.

## Mission statement

GemmPali is **not a CGI-specialist model**.

Our objective is broad multi-page dominance vs ColQwen across diverse document regimes (reports, technical docs, forms, layouts, cross-page tables/charts), with CGI annual reports used only as one supplemental stress domain.

## Current status

## Fresh update (2026-03-01, 21:05 EST)

- DeepThink Run-B long benchmark decision completed (hard harness, 80 samples / 48 candidates / seed 7).
- Head-to-head on Run-B dataset (`phase3_3_pairs`) results:
  - Phase2 champion (step 18000): Hit@1 0.8625, Hit@5 0.9750, MRR@10 0.90375, NDCG@10 0.92694, CPCR 0.9750, latency 1952.05ms
  - Phase3.3 Run-B best (step 16000): Hit@1 0.6875, Hit@5 0.9500, MRR@10 0.80104, NDCG@10 0.84178, CPCR 0.9500, latency 1953.64ms
- Decision: keep Phase2 step18000 as champion; Run-B does not pass promotion gate.

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
