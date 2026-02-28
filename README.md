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

## Key reports

- `reports/DeepThink-Prompt-GemmPali-SOTA-MultiPage-Training-2026-02-28.md`
- `reports/DeepThink-Response-GemmPali-SOTA-MultiPage-2026-02-28.txt`
- `reports/GemmPali-Accomplishment-Report-2026-02-28.md`

## Operational checklist

See `CHECKLIST.md` for the active execution plan and gates.
