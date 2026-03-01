#!/usr/bin/env python3
import argparse
import ast
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet_rows(pattern: str):
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        d = t.to_pydict()
        n = t.num_rows
        for i in range(n):
            yield {k: d[k][i] for k in d.keys()}


def normalize_answer(ans_obj):
    if not isinstance(ans_obj, dict):
        return ""
    variants = ans_obj.get("variants") or []
    if not variants:
        return ""
    v = variants[0]
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    if isinstance(v, str):
        s = v.strip()
        # DUDE sometimes stores python-list-like strings
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return " | ".join(str(x) for x in parsed)
            except Exception:
                pass
        return s
    return str(v)


def qa_to_pair(row, source_name: str, idx: int):
    q_obj = row.get("question") or {}
    d_obj = row.get("document") or {}
    e_obj = row.get("evidence") or {}

    query = (q_obj.get("text") or "").strip()
    answer = normalize_answer(row.get("answer") or {})
    doc_id = d_obj.get("id")
    pages = e_obj.get("pages") or []
    target_page = pages[0] if pages else None

    # keep text-heavy representation for current training script
    return {
        "pair_id": f"{source_name}::{idx}",
        "query": query,
        "answer": answer,
        "target_doc_id": doc_id,
        "target_page": target_page,
        "source": source_name,
        "meta": {
            "count_pages": d_obj.get("count_pages"),
            "question_type": q_obj.get("type"),
            "answer_type": (row.get("answer") or {}).get("type"),
        },
    }


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpdocvqa-glob", required=True)
    ap.add_argument("--dude-glob", required=True)
    ap.add_argument("--warmup-jsonl", default="")
    ap.add_argument("--warmup-sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    mp_n = 0
    for i, row in enumerate(iter_parquet_rows(args.mpdocvqa_glob)):
        p = qa_to_pair(row, "AHS-uni/mpdocvqa-qa", i)
        if p["query"]:
            pairs.append(p)
            mp_n += 1

    dude_n = 0
    for i, row in enumerate(iter_parquet_rows(args.dude_glob)):
        p = qa_to_pair(row, "AHS-uni/dude-qa", i)
        if p["query"]:
            pairs.append(p)
            dude_n += 1

    warmup_n = 0
    if args.warmup_jsonl:
        warm_rows = load_jsonl(Path(args.warmup_jsonl))
        if warm_rows:
            random.shuffle(warm_rows)
            take = min(args.warmup_sample, len(warm_rows))
            warm_rows = warm_rows[:take]
            for i, r in enumerate(warm_rows):
                pairs.append({
                    "pair_id": f"warmup::{i}",
                    "query": r.get("query", ""),
                    "answer": r.get("answer", ""),
                    "target_doc_id": r.get("target_doc_id"),
                    "target_page": r.get("meta", {}).get("page"),
                    "source": "vidore/colpali_train_set",
                    "meta": r.get("meta", {}),
                })
            warmup_n = take

    random.shuffle(pairs)

    out_jsonl = out_dir / "phase2_pairs.jsonl"
    with out_jsonl.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    summary = {
        "mpdocvqa_pairs": mp_n,
        "dude_pairs": dude_n,
        "warmup_pairs": warmup_n,
        "total_pairs": len(pairs),
        "out_jsonl": str(out_jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
