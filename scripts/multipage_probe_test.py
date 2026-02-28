import argparse
import json
import random
import time
from pathlib import Path

import torch

from src.model_wrapper import WrapperConfig, MultiPageRetrieverWrapper


def page_scores(query_vecs: torch.Tensor, page_vecs: torch.Tensor) -> float:
    sims = torch.matmul(query_vecs, page_vecs.T)
    return float(sims.max(dim=-1).values.sum().item())


def run_case(wrapper, num_pages: int, tokens_per_page: int, vocab_lo: int, vocab_hi: int):
    target_page = random.randint(0, num_pages - 1)

    pages = []
    for p in range(num_pages):
        if p == target_page:
            toks = torch.randint(vocab_hi - 200, vocab_hi - 1, (tokens_per_page,), dtype=torch.long)
        else:
            toks = torch.randint(vocab_lo, vocab_hi - 500, (tokens_per_page,), dtype=torch.long)
        pages.append(toks)

    doc_ids = torch.cat(pages).unsqueeze(0).to(wrapper.head.proj.weight.device)
    qry_ids = pages[target_page].unsqueeze(0).to(wrapper.head.proj.weight.device)

    with torch.no_grad():
        doc_vecs = wrapper(doc_ids, is_document_indexing=True)[0]
        qry_vecs = wrapper(qry_ids, is_document_indexing=False)[0]

    pool = wrapper.cfg.pool_factor
    vecs_per_page = tokens_per_page // pool
    page_chunks = [doc_vecs[i * vecs_per_page:(i + 1) * vecs_per_page] for i in range(num_pages)]

    scores = [page_scores(qry_vecs, pv) for pv in page_chunks]
    pred = int(max(range(num_pages), key=lambda i: scores[i]))

    return {
        "num_pages": num_pages,
        "target_page": target_page,
        "pred_page": pred,
        "correct": pred == target_page,
        "scores": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens-per-page", type=int, default=64)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--pages", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="Reports/multipage_probe_report.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dmap = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dmap[args.dtype]

    t0 = time.time()
    wrapper = MultiPageRetrieverWrapper(
        WrapperConfig(
            model_name=args.model,
            embed_dim=128,
            pool_factor=4,
            dtype=dtype,
            device=args.device,
            allow_dummy=False,
            force_dummy=False,
        )
    ).to(args.device)
    load_sec = time.time() - t0

    results = []
    for p in args.pages:
        for _ in range(args.trials):
            results.append(run_case(wrapper, p, args.tokens_per_page, 100, 32000))

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_pages = {}
    for p in args.pages:
        sub = [r for r in results if r["num_pages"] == p]
        by_pages[str(p)] = {
            "cases": len(sub),
            "correct": sum(1 for r in sub if r["correct"]),
            "accuracy": (sum(1 for r in sub if r["correct"]) / len(sub)) if sub else 0.0,
        }

    report = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "tokens_per_page": args.tokens_per_page,
        "trials_per_page": args.trials,
        "pages_tested": args.pages,
        "load_sec": load_sec,
        "overall_accuracy": correct / total if total else 0.0,
        "overall": {"correct": correct, "cases": total},
        "by_pages": by_pages,
        "results": results,
        "using_dummy": bool(wrapper.using_dummy),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
