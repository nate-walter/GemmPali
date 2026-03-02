#!/usr/bin/env python3
import argparse
import glob
import json
import math
import random
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from src.model_wrapper import WrapperConfig, MultiPageRetrieverWrapper, maxsim_score

try:
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    pq = None
    _pq_err = e


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


class ParquetImageResolver:
    def __init__(self, parquet_globs, cache_dir):
        if pq is None:
            raise RuntimeError(f"pyarrow is required for raw-image evaluation: {_pq_err}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.materialized = {}
        self._build(parquet_globs)

    @staticmethod
    def _to_int(v):
        try:
            if v is None or v == "":
                return None
            return int(v)
        except Exception:
            return None

    def _build(self, parquet_globs):
        files = []
        for g in parquet_globs:
            files.extend(glob.glob(g))
        files = sorted(set(files))
        for f in files:
            try:
                t = pq.read_table(f)
            except Exception:
                continue
            cols = set(t.column_names)
            d = t.to_pydict()
            n = t.num_rows
            if {"document_id", "page_number", "image"}.issubset(cols):
                for i in range(n):
                    did = d["document_id"][i]
                    pg = self._to_int(d["page_number"][i])
                    self.index[(did, pg)] = (f, i)
                    self.index.setdefault((did, None), (f, i))
            if "image_filename" in cols and "image" in cols:
                pages = d.get("page", [None] * n)
                for i in range(n):
                    did = d["image_filename"][i]
                    pg = self._to_int(pages[i])
                    self.index[(did, pg)] = (f, i)
                    self.index.setdefault((did, None), (f, i))

    def _materialize(self, key, rec):
        if key in self.materialized and Path(self.materialized[key]).exists():
            return self.materialized[key]
        f, i = rec
        t = pq.read_table(f)
        d = t.to_pydict()
        img = d["image"][i]
        b = img.get("bytes") if isinstance(img, dict) else None
        if not b:
            raise RuntimeError(f"missing image bytes at {f}:{i}")
        out = self.cache_dir / f"eval_{abs(hash((f, i)))}.png"
        if not out.exists():
            Image.open(BytesIO(b)).convert("RGB").save(out)
        self.materialized[key] = str(out)
        return str(out)

    def resolve(self, row):
        did = row.get("target_doc_id")
        pg = self._to_int(row.get("target_page"))

        if isinstance(did, str):
            p = Path(did)
            if p.exists():
                return str(p)

        keys = [(did, pg), (did, None)]
        if isinstance(did, str):
            keys.extend([(did.split("/")[-1], pg), (did.split("/")[-1], None)])
        for key in keys:
            rec = self.index.get(key)
            if rec:
                return self._materialize(key, rec)
        return None


def encode_query_seq(wrapper, tok, query, max_len, device, dtype):
    q = tok([query], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    q = {k: v.to(device) for k, v in q.items() if torch.is_tensor(v)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        out = wrapper.backbone(**q)
        hs = out.last_hidden_state.to(wrapper.head.proj.weight.dtype)
        return wrapper.head(hs)


def encode_doc_seq_from_image(wrapper, processor, image_paths, max_len, device, dtype):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompts = ["Index this document page for retrieval."] * len(images)
    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

    allowed = {
        "input_ids", "attention_mask", "pixel_values", "image_sizes", "aspect_ratio_ids",
        "aspect_ratio_mask", "token_type_ids", "position_ids",
    }
    fw = {k: v for k, v in batch.items() if k in allowed}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        out = wrapper.backbone(**fw)
        hs = out.last_hidden_state.to(wrapper.head.proj.weight.dtype)
        return wrapper.head(hs)


def pairwise_query_vs_docs(q_seq, docs_seq):
    sims = torch.einsum("bqd,cpd->bcqp", q_seq, docs_seq)
    return sims.max(dim=-1).values.sum(dim=-1).squeeze(0)


def dcg_at_k(rels, k):
    out = 0.0
    for i, r in enumerate(rels[:k], start=1):
        out += (2 ** r - 1) / math.log2(i + 1)
    return out


def ndcg_at_k(rank_is_hit, k):
    rels = [1.0 if x else 0.0 for x in rank_is_hit]
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if idcg == 0 else dcg / idcg


def run_eval(args):
    random.seed(args.seed)
    rows = load_jsonl(args.pairs)

    device = "cuda:0"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    wrapper = MultiPageRetrieverWrapper(
        WrapperConfig(
            model_name=args.model,
            embed_dim=128,
            pool_factor=4,
            dtype=dtype,
            device=device,
            load_in_4bit=args.load_in_4bit,
            allow_dummy=False,
        )
    ).to(device)

    if wrapper.processor is None:
        wrapper.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    cp = torch.load(args.checkpoint, map_location="cpu")
    wrapper.head.load_state_dict(cp["head"], strict=True)
    if "lora" in cp and cp["lora"]:
        wrapper.backbone.load_state_dict(cp["lora"], strict=False)

    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad = False

    default_globs = [
        "nvme_cache/raw/mpdocvqa-corpus/**/*.parquet",
        "nvme_cache/raw/dude-corpus/**/*.parquet",
        "nvme_cache/raw/vidore-colpali-train-set/**/*.parquet",
        "nvme_cache/raw/vidore-docvqa-train/**/*.parquet",
    ]
    resolver = ParquetImageResolver(args.parquet_glob or default_globs, cache_dir=args.cache_dir)

    usable = [r for r in rows if resolver.resolve(r)]
    sample_rows = random.sample(usable, k=min(args.samples, len(usable)))

    hit1 = hit5 = 0
    mrr10 = 0.0
    ndcg10 = 0.0
    latencies = []
    cpcr_hits = 0

    all_docs = [(r.get("target_doc_id"), r.get("target_page"), resolver.resolve(r)) for r in usable]
    all_docs = [x for x in all_docs if x[2]]

    for r in sample_rows:
        q = r.get("query", "")
        pos_doc_id = r.get("target_doc_id")
        pos_page = r.get("target_page")
        pos_path = resolver.resolve(r)

        negs = []
        tries = 0
        while len(negs) < max(1, args.candidates - 1) and tries < args.candidates * 60:
            did, pg, p = random.choice(all_docs)
            tries += 1
            if not p:
                continue
            if did == pos_doc_id and str(pg) == str(pos_page):
                continue
            negs.append((did, pg, p))

        doc_paths = [pos_path] + [x[2] for x in negs]
        doc_ids = [(pos_doc_id, pos_page)] + [(x[0], x[1]) for x in negs]

        t0 = time.time()
        q_seq = encode_query_seq(wrapper, tok, q, args.max_len, device, dtype)
        d_seq = encode_doc_seq_from_image(wrapper, wrapper.processor, doc_paths, args.max_len, device, dtype)
        scores = pairwise_query_vs_docs(q_seq, d_seq).detach().float().cpu().tolist()
        latencies.append((time.time() - t0) * 1000.0)

        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        ranks = [idx for idx, _ in ranked]
        pos_rank = ranks.index(0) + 1

        if pos_rank == 1:
            hit1 += 1
        if pos_rank <= 5:
            hit5 += 1
        if pos_rank <= 10:
            mrr10 += 1.0 / pos_rank

        rank_is_hit = [(i == 0) for i in ranks[:10]]
        ndcg10 += ndcg_at_k(rank_is_hit, 10)

        top5_ids = [doc_ids[i] for i in ranks[:5]]
        if (pos_doc_id, pos_page) in top5_ids:
            cpcr_hits += 1

    n = len(sample_rows)
    out = {
        "harness": "phase4_raw_image_retrieval",
        "note": "Uses real images resolved from pair target_doc_id/target_page via parquet image bytes (no query+answer surrogate docs).",
        "samples": n,
        "candidates_per_query": args.candidates,
        "metrics": {
            "Hit@1": hit1 / n if n else 0.0,
            "Hit@5": hit5 / n if n else 0.0,
            "MRR@10": mrr10 / n if n else 0.0,
            "NDCG@10": ndcg10 / n if n else 0.0,
            "CPCR": cpcr_hits / n if n else 0.0,
            "latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        },
        "model": {
            "checkpoint": args.checkpoint,
            "model_name": args.model,
            "load_in_4bit": bool(args.load_in_4bit),
            "dtype": args.dtype,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--cache-dir", default="nvme_cache/vision_cache_eval")
    ap.add_argument("--parquet-glob", action="append", default=[])
    args = ap.parse_args()

    out = run_eval(args)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
