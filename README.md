# GemmPali (Multi-Page Gemma Retrieval Program)

GemmPali is a Gemma-based multi-page retrieval system pursuing **general-document** SOTA in ColPali-style indexing/retrieval.

## Mission statement

GemmPali is **not a CGI-specialist model**.

Our objective is broad multi-page dominance vs ColQwen across diverse document regimes (reports, technical docs, forms, layouts, cross-page tables/charts), with CGI annual reports used only as one supplemental stress domain.

## Current status

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
- status: Phase 1 full warmup completed at 5000/5000 with checkpoints through `checkpoints/phase1/head_step_0005000.pt`

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

## GemmPali dashboard (new)

A dedicated cyberpunk training dashboard now exists under:
- `dashboard/` (Express backend + neon frontend)
- `GemmPaliMobile/` (Expo iPhone app for compact mobile telemetry)

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

## Key reports

- `reports/DeepThink-Prompt-GemmPali-SOTA-MultiPage-Training-2026-02-28.md`
- `reports/DeepThink-Response-GemmPali-SOTA-MultiPage-2026-02-28.txt`
- `reports/GemmPali-Accomplishment-Report-2026-02-28.md`

## Operational checklist

See `CHECKLIST.md` for the active execution plan and gates.
