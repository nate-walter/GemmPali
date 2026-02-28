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

## Dataset policy (critical)

### HDD-first storage policy
All training corpora are stored under HDD roots for space management and reproducibility.

- Long-term datasets: **HDD only**
- NVMe usage: **ad-hoc copy/stage only when actively training**
- After training run: clear staged NVMe copies unless explicitly retained

This policy is non-negotiable for operational stability.

## Dataset strategy baseline

We currently plan around:
- ViDoRe
- DocVQA
- MP-DocVQA (to add/download)
- DUDE (to add/download)
- Internal CGI annual-report corpus (supplemental stress domain, not central identity)

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
