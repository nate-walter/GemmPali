# DeepThink Continuation Prompt — GemmPali Multi-Page SOTA Push

DeepThink — continuing our same thread that produced your Gemma-3-12B Torch patch plan.

We executed your patch direction and got the prototype working. Now we need your **next-step execution contract** to turn GemmPali into a true behemoth that beats ColQwen on multi-page indexing/retrieval.

## Continuation context (same mission, now post-patch)

From this same conversation lineage:
- We started with the US-model multi-page gap and refusal to accept single-page limitations.
- You provided architecture guidance and a Torch patch strategy for `gemma-3-12b-it`.
- We implemented that direction and now have verified multi-page behavior in our prototype path.

This is not a restart. This is phase-2 continuation.

## What we now need from you

We need the **best dataset + training plan** to make GemmPali dominant on multi-page retrieval quality.

### Known dataset availability right now
We already have access to:
- **ViDoRe**
- **DocVQA**

We also have a proprietary internal **CGI annual-report PDF corpus**:
- multi-year annual reports (financial + narrative docs)
- naturally rich in cross-page tables/charts/section continuity
- useful for real-world long-document multi-page retrieval stress

### Important instruction
Do **not** assume prior local repos/scripts from earlier failed attempts are part of this contract.
Treat this as:
1) your proven Gemma patch direction has worked,
2) we currently rely on ViDoRe + DocVQA + internal CGI corpus,
3) now we need the best path to SOTA.

---

## Required output

Return an execution-grade answer with these sections only:

1. `dataset_strategy`
2. `additional_datasets_to_add_or_skip`
3. `data_mixing_and_curriculum`
4. `exact_training_plan`
5. `multi_page_eval_protocol_vs_colqwen`
6. `risk_mitigation`
7. `14_day_operator_timeline`

---

## Section requirements (strict)

## 1) dataset_strategy
- How to use ViDoRe + DocVQA + internal CGI corpus together.
- Exact role of each dataset (what skill each should teach the model).
- Whether these are enough to win alone; if not, say exactly why.

## 2) additional_datasets_to_add_or_skip
- If additional datasets are required, list exact names and why they help specifically for **multi-page indexing/retrieval**.
- Separate into:
  - must-add
  - optional-add
  - skip
- Include exact acquisition commands where possible.

## 3) data_mixing_and_curriculum
- Exact mixture ratios and staging across phases.
- How to avoid overfitting to enterprise/financial style from CGI corpus.
- Hard-negative strategy for cross-page confusion robustness.

## 4) exact_training_plan
- Full runnable command-level plan (not conceptual only).
- Hyperparameters, schedule, precision, batching, eval cadence.
- Any architecture deltas beyond current patch path.
- Clear branch logic and go/no-go thresholds.

## 5) multi_page_eval_protocol_vs_colqwen
- Apples-to-apples head-to-head against ColQwen variants.
- Multi-page-specific benchmark design (cross-page tables/charts, continuity queries).
- Metrics: Hit@k, MRR, NDCG, latency, VRAM, plus cross-page consistency.
- Failure taxonomy + remediation loop.

## 6) risk_mitigation
- Top risks, detection signals, thresholds, intervention actions.

## 7) 14_day_operator_timeline
- Day-by-day milestones and expected artifacts.

---

## Hard constraints
- No generic advice.
- No hand-wavy “it depends” without measurable branch criteria.
- Optimize for one goal: **GemmPali beats ColQwen on multi-page indexing/retrieval quality**.

---

## Provenance links (same conversation chain)
- https://g.co/gemini/share/cc72325be710
- https://docs.google.com/document/d/1szArxsvLtTvj37d7lbFudaRS7RLm3e6NVmX3xqaZFa0/edit
- https://docs.google.com/document/d/1tg-b5t41aMj3XDR0NSzN26dItvZh5BEKpMkkEpN6hSs/edit
