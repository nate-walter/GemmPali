# GemmPali Multipage Validation Milestone
## Date: 2026-03-05

## Why this matters
This checkpoint documents the first clear evidence that GemmPali exhibits **non-zero multipage retrieval behavior** on our controlled CGI annual-report benchmark flow, and that this behavior is meaningfully above vanilla Gemma control.

---

## 3-Step Validation Protocol (the exact process we ran)

### Step 1 — GemmPali (trained model under test)
- Script: `/home/nate/GemmPali/scripts/validate_multipage_cgi2019.py`
- Output: `/home/nate/GemmPali/reports/multipage_validation/gemmpali_multipage_cgi2019_step8000.json`
- Model:
  - Backbone: `google/gemma-3-4b-it`
  - Retrieval head checkpoint: `head_step_0008000.pt`

Result summary:
- hit@1: 0.10
- hit@3: 0.15
- hit@5: 0.15
- hit@10: 0.20
- multipage_hit@5: 0.25 (1/4 multipage questions)

### Step 2 — Vanilla Gemma3 control (no training magic)
- Same script/dataset/document objective
- Control head: random/untrained retrieval head snapshot
- Output: `/home/nate/GemmPali/reports/multipage_validation/vanilla_gemma3_randomhead_multipage_cgi2019.json`

Result summary:
- hit@1: 0.00
- hit@3: 0.05
- hit@5: 0.10
- hit@10: 0.15
- multipage_hit@5: 0.00 (0/4)

### Step 3 — ColQwen baseline comparison (existing benchmark artifacts)
- ColQwen2.5 baseline result file:
  - `/home/nate/ColPali/upgrading_team_qdrant_colpali_NBs/Reports/byaldi-validation/ab_cgi2019_fullpdf_colqwen25_v1_gpu0.json`
- ColQwen2 baseline result file:
  - `/home/nate/ColPali/upgrading_team_qdrant_colpali_NBs/Reports/byaldi-validation/ab_cgi2019_fullpdf_colqwen2_v1_gpu0.json`

Result summary (from these files):
- ColQwen2.5: hit@1 0.10, hit@3 0.15, hit@5 0.20, hit@10 0.35
- ColQwen2:   hit@1 0.10, hit@3 0.15, hit@5 0.20, hit@10 0.35

---

## A/B/C Interpretation

- GemmPali > vanilla Gemma control on all major retrieval metrics.
- GemmPali shows **some multipage functionality** (non-zero multipage hit@5), while vanilla control shows none.
- GemmPali is not yet at ColQwen hit@10 performance on this benchmark slice.

This is therefore a **functional signal milestone**, not final SOTA declaration.

---

## Exact Dataset / QA Genesis (source of truth)

### Document used
- PDF: `/home/nate/ColPali/CGI-Annual-Reports-PDFs/cgi-2019-annual-report.pdf`

### QA set used
- Primary wrapped file: `/home/nate/ColPali/upgrading_team_qdrant_colpali_NBs/DeepThink/2026-02-24-cgi2019-qa-full.json`
  - contains `qa_pairs`
- Flat extraction copy: `/home/nate/ColPali/upgrading_team_qdrant_colpali_NBs/DeepThink/2026-02-24-cgi2019-qa-full.qa_pairs.json`

### QA schema fields relevant to multipage validation
- `id`
- `question` (or `query` in some scripts)
- `expected_pages_any` (critical multipage target set)
- `expected_pages_primary`
- optional evidence metadata fields

---

## Milestone statement (internal)
GemmPali has now cleared the key gate from “zero multipage signal” to “measurable multipage signal above vanilla control.”

Next work is to increase reliability and depth recall while preserving the validated functionality direction.
