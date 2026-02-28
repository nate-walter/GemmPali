#!/usr/bin/env python3
import argparse
import json
import time

import torch

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model_wrapper import MultiPageRetrieverWrapper, WrapperConfig, maxsim_score


def pick_dtype(name: str):
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--pool-factor", type=int, default=4)
    ap.add_argument("--allow-dummy", action="store_true")
    ap.add_argument("--force-dummy", action="store_true")
    ap.add_argument("--out", default="Reports/smoke_forward_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)

    cfg = WrapperConfig(
        model_name=args.model,
        embed_dim=args.embed_dim,
        pool_factor=args.pool_factor,
        dtype=dtype,
        device=str(device),
        allow_dummy=args.allow_dummy,
        force_dummy=args.force_dummy,
    )

    t0 = time.time()
    wrapper = MultiPageRetrieverWrapper(cfg).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Synthetic stand-in for multipage: token length scales with pages.
    total_tokens = args.seq_len * max(args.pages, 1)
    input_ids = torch.randint(0, 32000, (args.batch_size, total_tokens), device=device)

    doc_vecs = wrapper(input_ids=input_ids, is_document_indexing=True)
    qry_len = min(512, total_tokens)
    q_ids = input_ids[:, :qry_len]
    qry_vecs = wrapper(input_ids=q_ids, is_document_indexing=False)
    score = maxsim_score(qry_vecs, doc_vecs)

    elapsed = time.time() - t0
    peak_gb = None
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    report = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "using_dummy": bool(getattr(wrapper, "using_dummy", False)),
        "pages": args.pages,
        "input_shape": list(input_ids.shape),
        "doc_vec_shape": list(doc_vecs.shape),
        "qry_vec_shape": list(qry_vecs.shape),
        "maxsim": [float(x) for x in score.detach().cpu().tolist()],
        "elapsed_sec": elapsed,
        "peak_vram_gb": peak_gb,
    }

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
