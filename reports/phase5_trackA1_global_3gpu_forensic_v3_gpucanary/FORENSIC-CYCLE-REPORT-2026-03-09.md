# Phase5 A1 Forensic Cycle Report (2k / 4k / 8k)

Date: 2026-03-09  
Run family: `phase5_trackA1_global_3gpu_forensic_v3_gpucanary`  
Scope: forensic gate evaluation at checkpoints `0002000`, `0004000`, `0008000`

---

## 1) Executive summary

This cycle confirms stable multipage retrieval behavior across all three checkpoints.

- Multipage consistency (`CPCR@10`) is strong at every checkpoint and above gate threshold (`>= 0.35`).
- ViDoRe floor is held at all checkpoints, with strongest value at 2k.
- CGI rows are included in the scorecards (`cgi_rows=36` each), but CGI TRR remains non-evaluable (`cgi_adjacent_available=0`) under current candidate-pool composition.
- Final winner call for this cycle (TRR excluded): **2k checkpoint**.

---

## 2) Scorecard table

| Checkpoint | Hit@1 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 | CPCR@10 | ViDoRe Hit@10 | CGI Hit@10 | CGI rows | CGI TRR | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2k | 0.205 | 0.465 | **0.605** | 0.3079 | 0.3775 | **0.570** | **0.7333** | 0.8333 | 36 | null | 20083.9 |
| 4k | **0.210** | 0.465 | 0.595 | **0.3146** | **0.3806** | 0.560 | 0.7167 | 0.8333 | 36 | null | 20175.1 |
| 8k | 0.190 | **0.475** | 0.590 | 0.3074 | 0.3748 | 0.550 | 0.7167 | 0.8333 | 36 | null | **20052.5** |

Source artifacts:
- `scorecard_step_0002000.json`
- `scorecard_step_0004000.json`
- `scorecard_step_0008000.json`

---

## 3) What changed during this cycle (critical debugging history)

### A) CGI exclusion bug fixed
Initial forensic passes showed `cgi_rows=0` because resolver eligibility ignored explicit row image fields (`image_path`, etc.) and relied on target_doc/parquet matching only.

Fix applied:
- Resolver `has_candidate()` and `resolve()` now honor direct image-path fields before parquet fallback.

Result:
- CGI rows now appear in scorecards (`cgi_rows=36`).

### B) TRR evaluability investigation
Even with CGI rows present, `cgi_trr` remained null due to:
- `cgi_adjacent_available=0`
- Current sampled/candidate pool has no adjacent non-expected CGI trap pages in-eval.

Decision for this cycle:
- finalize gate interpretation without CGI TRR.

---

## 4) Interpretation by objective

### Multipage functionality
Confirmed.

Evidence:
- CPCR@10 = 0.57 / 0.56 / 0.55 (all above gate floor 0.35)
- Stable top-10 retrieval quality across checkpoints

### General retrieval robustness (anti-forgetting)
Maintained.

Evidence:
- ViDoRe Hit@10 remains high at all checkpoints (0.7333 / 0.7167 / 0.7167)

### Ranking sharpness
Best at 4k.

Evidence:
- Highest MRR@10 and NDCG@10 at 4k.

### Latency
Best at 8k by a small margin.

Evidence:
- Lowest average query latency at 8k (~20052 ms).

---

## 5) Winner determination

### Winner (overall, TRR excluded): **2k checkpoint**

Reasoning:
- Best Hit@10
- Best CPCR@10 (multipage consistency)
- Best ViDoRe Hit@10 (anti-forgetting)
- Near-best latency (difference vs 8k is minor)

When to choose alternatives:
- Choose 4k if ranking-shape (`MRR/NDCG`) is explicitly prioritized over top-10 recall/CPCR.
- Choose 8k only if minimal latency and hit@5 are the top priorities.

---

## 6) Gate verdict (for this cycle)

With CGI TRR excluded by explicit operator decision:
- CPCR gate: PASS
- ViDoRe floor gate: PASS
- Overall forensic objective: PASS

---

## 7) Recommended next steps

1. Promote/use 2k checkpoint as working winner for subsequent downstream evaluation and planning.
2. If CGI TRR must be restored later, run a dedicated TRR protocol with guaranteed adjacent-trap candidate injection from a full CGI page universe.
3. Keep the current winner board and summary scripts as standard reporting outputs for future forensic cycles.

---

## 8) Related artifacts

- `README.md` (dedicated final summary section)
- `CHECKLIST.md` (updated completion state)
- `WINNER-BOARD.md`
- `GATE-SUMMARY.md`
- `scripts/winner_board_phase5.py`
- `scripts/summarize_phase5_forensic_gates.py`
