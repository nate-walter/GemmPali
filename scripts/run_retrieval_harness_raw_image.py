#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import torch.nn.functional as F

from src.model_wrapper import WrapperConfig, MultiPageRetrieverWrapper, maxsim_score

try:
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    pq = None
    _pq_err = e

try:
    from torchvision.io import decode_image, decode_jpeg
except Exception:
    decode_image = None
    decode_jpeg = None


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


class ParquetImageResolver:
    def __init__(self, parquet_globs, cache_dir, gpu_preprocess=False, gpu_preprocess_device="cuda:0"):
        if pq is None:
            raise RuntimeError(f"pyarrow is required for raw-image evaluation: {_pq_err}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.materialized = {}
        self.gpu_preprocess = bool(gpu_preprocess)
        self.gpu_preprocess_device = gpu_preprocess_device
        self.preprocess_stats = {
            "gpu_enabled": self.gpu_preprocess,
            "gpu_device": self.gpu_preprocess_device,
            "gpu_decode_count": 0,
            "gpu_decode_seconds": 0.0,
            "gpu_resize_count": 0,
            "gpu_resize_seconds": 0.0,
            "cpu_decode_count": 0,
            "cpu_decode_seconds": 0.0,
            "cpu_resize_count": 0,
            "cpu_resize_seconds": 0.0,
        }
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

    def _decode_with_gpu(self, b):
        if not self.gpu_preprocess or not torch.cuda.is_available():
            return None
        if decode_jpeg is None and decode_image is None:
            return None
        t0 = time.time()
        try:
            bt = torch.tensor(bytearray(b), dtype=torch.uint8)
            # Fast path for JPEG byte streams.
            if decode_jpeg is not None and len(b) >= 3 and b[0] == 0xFF and b[1] == 0xD8 and b[2] == 0xFF:
                x = decode_jpeg(bt, device=self.gpu_preprocess_device)  # [C,H,W] uint8 on GPU
            else:
                # Generic decode fallback; move to GPU for consistency.
                x = decode_image(bt, mode="RGB")
                x = x.to(self.gpu_preprocess_device, non_blocking=True)
            self.preprocess_stats["gpu_decode_count"] += 1
            self.preprocess_stats["gpu_decode_seconds"] += (time.time() - t0)
            return x
        except Exception as e:
            print(f"[preprocess] WARN gpu decode fallback -> cpu ({e})", flush=True)
            return None

    def _resize_with_gpu(self, b, nw, nh):
        if not self.gpu_preprocess:
            return None
        x = self._decode_with_gpu(b)
        if x is None:
            return None
        t0 = time.time()
        try:
            x = x.float().unsqueeze(0) / 255.0
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = (x.clamp(0.0, 1.0) * 255.0).byte().squeeze(0).cpu()  # [C,H,W]
            arr = x.permute(1, 2, 0).numpy()
            self.preprocess_stats["gpu_resize_count"] += 1
            self.preprocess_stats["gpu_resize_seconds"] += (time.time() - t0)
            return Image.fromarray(arr, mode="RGB")
        except Exception as e:
            print(f"[preprocess] WARN gpu resize fallback -> cpu ({e})", flush=True)
            return None

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

        policy = os.environ.get("GEMMPALI_EVAL_POLICY", "v1")
        max_pixels = int(os.environ.get("GEMMPALI_MAX_PAGE_PIXELS", "89478485"))
        stable_key = hashlib.sha1(f"{f}:{i}:{policy}:{max_pixels}".encode("utf-8")).hexdigest()
        out = self.cache_dir / f"eval_{policy}_{max_pixels}_{stable_key}.png"

        if not out.exists():
            t0 = time.time()

            x_gpu = self._decode_with_gpu(b)
            if x_gpu is not None:
                _, h, w = x_gpu.shape
                pix = w * h
                if pix > max_pixels:
                    scale = (max_pixels / float(pix)) ** 0.5
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    im = self._resize_with_gpu(b, nw, nh)
                    if im is None:
                        t_resize = time.time()
                        arr = x_gpu.cpu().permute(1, 2, 0).numpy()
                        im = Image.fromarray(arr, mode="RGB").resize((nw, nh), Image.Resampling.BILINEAR)
                        self.preprocess_stats["cpu_resize_count"] += 1
                        self.preprocess_stats["cpu_resize_seconds"] += (time.time() - t_resize)
                    print(f"[preprocess] capped page {w}x{h} -> {nw}x{nh} via GPU (policy={policy}, max_pixels={max_pixels})", flush=True)
                else:
                    arr = x_gpu.cpu().permute(1, 2, 0).numpy()
                    im = Image.fromarray(arr, mode="RGB")
                    print(f"[preprocess] decoded page {w}x{h} via GPU", flush=True)
            else:
                t_decode = time.time()
                im = Image.open(BytesIO(b))
                self.preprocess_stats["cpu_decode_count"] += 1
                self.preprocess_stats["cpu_decode_seconds"] += (time.time() - t_decode)
                w, h = im.size
                pix = w * h
                if pix > max_pixels:
                    scale = (max_pixels / float(pix)) ** 0.5
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    try:
                        im.draft("RGB", (nw, nh))
                    except Exception:
                        pass
                    t_resize = time.time()
                    im = im.convert("RGB")
                    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
                    self.preprocess_stats["cpu_resize_count"] += 1
                    self.preprocess_stats["cpu_resize_seconds"] += (time.time() - t_resize)
                    print(f"[preprocess] capped page {w}x{h} -> {nw}x{nh} via CPU (policy={policy}, max_pixels={max_pixels})", flush=True)
                else:
                    im = im.convert("RGB")

            im.save(out)
            dt = time.time() - t0
            if dt > 1.0:
                print(f"[preprocess] materialized {out.name} in {dt:.2f}s", flush=True)
        self.materialized[key] = str(out)
        return str(out)

    def resolve(self, row):
        did = row.get("target_doc_id")
        pg = self._to_int(row.get("target_page"))

        # First-class direct image path support (needed for CGI rows in phase5 mix).
        for k in ("image_path", "target_image_path", "doc_image_path", "path"):
            v = row.get(k)
            if isinstance(v, str) and v:
                p = Path(v)
                if p.exists():
                    return str(p)

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

    def has_candidate(self, row):
        """Fast existence check without materializing image bytes to disk."""
        did = row.get("target_doc_id")
        pg = self._to_int(row.get("target_page"))

        # First-class direct image path support (needed for CGI rows in phase5 mix).
        for k in ("image_path", "target_image_path", "doc_image_path", "path"):
            v = row.get(k)
            if isinstance(v, str) and v and Path(v).exists():
                return True

        if isinstance(did, str):
            p = Path(did)
            if p.exists():
                return True

        keys = [(did, pg), (did, None)]
        if isinstance(did, str):
            leaf = did.split("/")[-1]
            keys.extend([(leaf, pg), (leaf, None)])
        return any(k in self.index for k in keys)


def encode_query_seq(wrapper, tok, query, max_len, device, dtype):
    q = tok([query], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    q = {k: v.to(device) for k, v in q.items() if torch.is_tensor(v)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        out = wrapper.backbone(**q)
        hs = out.last_hidden_state.to(wrapper.head.proj.weight.dtype)
        return wrapper.head(hs)


def encode_doc_seq_from_image(wrapper, processor, image_paths, max_len, device, dtype):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompts = ["<image> Index this document page for retrieval."] * len(images)

    allowed = {
        "input_ids", "attention_mask", "pixel_values", "image_sizes", "aspect_ratio_ids",
        "aspect_ratio_mask", "token_type_ids", "position_ids",
    }

    def _forward(batch_dict):
        batch_dict = {k: v.to(device) for k, v in batch_dict.items() if torch.is_tensor(v)}
        fw = {k: v for k, v in batch_dict.items() if k in allowed}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            out = wrapper.backbone(**fw)
            hs = out.last_hidden_state.to(wrapper.head.proj.weight.dtype)
            return wrapper.head(hs)

    batch_attempts = [
        {"images": [[img] for img in images], "text": prompts},
        {"images": images, "text": prompts},
    ]

    last_err = None
    for attempt in batch_attempts:
        try:
            batch = processor(
                images=attempt["images"],
                text=attempt["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            return _forward(batch)
        except ValueError as e:
            last_err = e

    print(f"[{datetime.now().isoformat()}] WARN batch doc encode failed ({last_err}); fallback to per-image encode", flush=True)
    chunks = []
    for i, img in enumerate(images):
        # Gemma3 processor can be strict about image-token alignment.
        # Retry with alternate prompt token forms + image list shape to avoid
        # "Prompt contained 0 image tokens but received 1 images" crashes.
        one = None
        per_image_err = None
        prompt_candidates = [
            prompts[i],
            "<start_of_image> Index this document page for retrieval.",
            "Index this document page for retrieval. <image>",
        ]
        image_shapes = ([img], [[img]])

        for pc in prompt_candidates:
            for im in image_shapes:
                try:
                    one = processor(
                        images=im,
                        text=[pc],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                    )
                    break
                except ValueError as e:
                    per_image_err = e
                    if "image tokens" in str(e).lower():
                        continue
                    raise
            if one is not None:
                break

        if one is None:
            raise ValueError(f"per-image processor fallback failed at idx={i}: {per_image_err}")

        chunks.append(_forward(one))
        if (i + 1) % 8 == 0 or (i + 1) == len(images):
            print(f"[{datetime.now().isoformat()}] doc-fallback progress {i+1}/{len(images)}", flush=True)
    return torch.cat(chunks, dim=0)


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


def _to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _domain_tag(row):
    src = (row.get("source") or "").lower()
    if "cgi" in src:
        return "cgi"
    if "vidore" in src:
        return "vidore"
    if "dude" in src:
        return "dude"
    if "docvqa" in src:
        return "mp_docvqa"
    return "other"


def run_eval(args):
    random.seed(args.seed)
    rows = load_jsonl(args.pairs)

    os.environ["GEMMPALI_MAX_PAGE_PIXELS"] = str(args.max_page_pixels)
    os.environ.setdefault("GEMMPALI_EVAL_POLICY", "dude-safe-v1")

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
    resolver = ParquetImageResolver(
        args.parquet_glob or default_globs,
        cache_dir=args.cache_dir,
        gpu_preprocess=args.gpu_preprocess,
        gpu_preprocess_device=args.gpu_preprocess_device,
    )

    usable = [r for r in rows if resolver.has_candidate(r)]
    sample_rows = random.sample(usable, k=min(args.samples, len(usable)))

    print(f"[{datetime.now().isoformat()}] STAGE usable_filter_done rows={len(rows)} usable={len(usable)}", flush=True)

    print(f"[{datetime.now().isoformat()}] START eval rows={len(rows)} usable={len(usable)} samples={len(sample_rows)} candidates={args.candidates} max_len={args.max_len}", flush=True)
    print(f"[{datetime.now().isoformat()}] STAGE model_loaded checkpoint={args.checkpoint}", flush=True)
    print(f"[{datetime.now().isoformat()}] STAGE preprocess_mode gpu_preprocess={args.gpu_preprocess} gpu_device={args.gpu_preprocess_device} torchvision_decode={decode_image is not None}", flush=True)

    hit1 = hit5 = hit10 = 0
    mrr10 = 0.0
    ndcg10 = 0.0
    latencies = []
    cpcr10_hits = 0

    split_totals = {"vidore": 0, "mp_docvqa": 0, "dude": 0, "cgi": 0, "other": 0}
    split_hit10 = {"vidore": 0, "mp_docvqa": 0, "dude": 0, "cgi": 0, "other": 0}

    cgi_rows = 0
    cgi_adjacent_available = 0
    cgi_trr_hits = 0

    all_doc_keys = list({
        (r.get("target_doc_id"), _to_int(r.get("target_page")))
        for r in usable
    })

    for idx, r in enumerate(sample_rows, start=1):
        q = r.get("query", "")
        pos_doc_id = r.get("target_doc_id")
        pos_page = _to_int(r.get("target_page"))
        pos_path = resolver.resolve(r)

        expected_pages = r.get("expected_pages_all") or r.get("expected_pages_any") or []
        expected_pages = {_to_int(x) for x in expected_pages if _to_int(x) is not None}
        if not expected_pages and pos_page is not None:
            expected_pages = {pos_page}

        dtag = _domain_tag(r)

        negs = []
        seen_neg_keys = set()

        # For CGI forensic rows, inject same-doc adjacent-page traps first
        # so CGI TRR is evaluable (without altering scoring math).
        if dtag == "cgi" and pos_doc_id is not None and expected_pages:
            adj_pool = []
            for did, pg_i in all_doc_keys:
                if did != pos_doc_id or pg_i is None:
                    continue
                if pg_i in expected_pages:
                    continue
                if any(abs(pg_i - ep) <= 2 for ep in expected_pages):
                    adj_pool.append((did, pg_i))
            random.shuffle(adj_pool)
            for did, pg_i in adj_pool:
                if len(negs) >= max(1, args.candidates - 1):
                    break
                neg_key = (did, pg_i)
                if neg_key in seen_neg_keys:
                    continue
                p = resolver.resolve({"target_doc_id": did, "target_page": pg_i})
                if not p:
                    continue
                negs.append((did, pg_i, p))
                seen_neg_keys.add(neg_key)

        tries = 0
        while len(negs) < max(1, args.candidates - 1) and tries < args.candidates * 200:
            did, pg_i = random.choice(all_doc_keys)
            tries += 1
            if did == pos_doc_id and pg_i == pos_page:
                continue
            neg_key = (did, pg_i)
            if neg_key in seen_neg_keys:
                continue
            p = resolver.resolve({"target_doc_id": did, "target_page": pg_i})
            if not p:
                continue
            negs.append((did, pg_i, p))
            seen_neg_keys.add(neg_key)

        doc_paths = [pos_path] + [x[2] for x in negs]
        doc_ids = [(pos_doc_id, pos_page)] + [(x[0], x[1]) for x in negs]

        t0 = time.time()
        try:
            q_seq = encode_query_seq(wrapper, tok, q, args.max_len, device, dtype)
            d_seq = encode_doc_seq_from_image(wrapper, wrapper.processor, doc_paths, args.max_len, device, dtype)
            scores = pairwise_query_vs_docs(q_seq, d_seq).detach().float().cpu().tolist()
            latencies.append((time.time() - t0) * 1000.0)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] ERROR sample={idx}/{len(sample_rows)} qlen={len(q)} docs={len(doc_paths)} pos_doc={pos_doc_id} pos_page={pos_page} err={e}", flush=True)
            raise

        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        ranks = [rid for rid, _ in ranked]
        rank_pos = {rid: ridx + 1 for ridx, rid in enumerate(ranks)}

        true_candidate_ids = [
            rid for rid, (did, pg) in enumerate(doc_ids)
            if did == pos_doc_id and pg in expected_pages
        ]
        if not true_candidate_ids:
            true_candidate_ids = [0]

        first_true_rank = min(rank_pos[rid] for rid in true_candidate_ids)

        if first_true_rank == 1:
            hit1 += 1
        if first_true_rank <= 5:
            hit5 += 1
        if first_true_rank <= 10:
            hit10 += 1
            mrr10 += 1.0 / first_true_rank

        rank_is_hit = [rid in true_candidate_ids for rid in ranks[:10]]
        ndcg10 += ndcg_at_k(rank_is_hit, 10)

        top10_ids = [doc_ids[rid] for rid in ranks[:10]]
        top10_pages_for_doc = {pg for did, pg in top10_ids if did == pos_doc_id and pg is not None}
        if expected_pages.issubset(top10_pages_for_doc):
            cpcr10_hits += 1

        split_totals[dtag] += 1
        if first_true_rank <= 10:
            split_hit10[dtag] += 1

        if dtag == "cgi":
            cgi_rows += 1
            adjacent_ids = [
                rid for rid, (did, pg) in enumerate(doc_ids)
                if did == pos_doc_id and pg is not None and pg not in expected_pages and any(abs(pg - ep) <= 2 for ep in expected_pages)
            ]
            if adjacent_ids:
                cgi_adjacent_available += 1
                best_adj_rank = min(rank_pos[rid] for rid in adjacent_ids)
                if best_adj_rank < first_true_rank:
                    cgi_trr_hits += 1

        if idx == 1 or idx % max(1, args.log_every) == 0:
            elapsed_s = sum(latencies) / 1000.0 if latencies else 0.0
            rate = idx / elapsed_s if elapsed_s > 0 else 0.0
            remain = max(0, len(sample_rows) - idx)
            eta_s = (remain / rate) if rate > 0 else float("inf")
            eta_txt = f"{eta_s/60:.1f}m" if eta_s != float("inf") else "?"
            print(f"[{datetime.now().isoformat()}] progress {idx}/{len(sample_rows)} | hit1={hit1} hit5={hit5} hit10={hit10} mrr10={mrr10/max(1,idx):.4f} ndcg10={ndcg10/max(1,idx):.4f} | eta={eta_txt}", flush=True)

    n = len(sample_rows)
    split_hit10_rate = {
        k: (split_hit10[k] / split_totals[k] if split_totals[k] else None)
        for k in split_totals
    }

    out = {
        "harness": "phase4_raw_image_retrieval",
        "note": "Uses real images resolved from pair target_doc_id/target_page via parquet image bytes (no query+answer surrogate docs).",
        "samples": n,
        "candidates_per_query": args.candidates,
        "metrics": {
            "Hit@1": hit1 / n if n else 0.0,
            "Hit@5": hit5 / n if n else 0.0,
            "Hit@10": hit10 / n if n else 0.0,
            "MRR@10": mrr10 / n if n else 0.0,
            "NDCG@10": ndcg10 / n if n else 0.0,
            "CPCR@10": cpcr10_hits / n if n else 0.0,
            "latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        },
        "split_metrics": {
            "hit10_rate": split_hit10_rate,
            "totals": split_totals,
        },
        "forensic_metrics": {
            "vidore_hit10": split_hit10_rate.get("vidore"),
            "cgi_trr": (cgi_trr_hits / cgi_adjacent_available) if cgi_adjacent_available else None,
            "cgi_trr_count": cgi_trr_hits,
            "cgi_adjacent_available": cgi_adjacent_available,
            "cgi_rows": cgi_rows,
            "global_cpcr10": cpcr10_hits / n if n else 0.0,
        },
        "model": {
            "checkpoint": args.checkpoint,
            "model_name": args.model,
            "load_in_4bit": bool(args.load_in_4bit),
            "dtype": args.dtype,
        },
        "preprocess": {
            "gpu_preprocess": bool(args.gpu_preprocess),
            "gpu_preprocess_device": args.gpu_preprocess_device,
            "max_page_pixels": int(args.max_page_pixels),
            "stats": resolver.preprocess_stats,
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
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-page-pixels", type=int, default=89478485)
    ap.add_argument("--gpu-preprocess", action="store_true", help="Use GPU decode/resize lane for oversized-page preprocessing when available.")
    ap.add_argument("--gpu-preprocess-device", default="cuda:0", help="CUDA device for preprocess offload (after CUDA_VISIBLE_DEVICES remap).")
    args = ap.parse_args()

    out = run_eval(args)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[{datetime.now().isoformat()}] STAGE write_output path={p}", flush=True)
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
