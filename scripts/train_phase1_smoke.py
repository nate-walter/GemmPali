#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from src.model_wrapper import WrapperConfig, MultiPageRetrieverWrapper


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def mean_pool(x):
    return x.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = f"cuda:{local_rank}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_jsonl(args.pairs)
    if not rows:
        raise RuntimeError("No training rows")

    wrapper = MultiPageRetrieverWrapper(
        WrapperConfig(model_name=args.model, embed_dim=128, pool_factor=4, dtype=dtype, device=device)
    ).to(device)

    # Phase-1 smoke: freeze backbone, train retriever head only
    for p in wrapper.backbone.parameters():
        p.requires_grad = False
    for p in wrapper.rope3d.parameters():
        p.requires_grad = False

    model = DDP(wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = [p for p in model.module.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # shard rows by rank
    my_rows = rows[local_rank::world_size]
    if not my_rows:
        my_rows = rows

    for step in range(1, args.steps + 1):
        r = my_rows[(step - 1) % len(my_rows)]
        q = r.get("query", "")
        d = (r.get("query", "") + " \n " + (r.get("answer", "") or ""))

        batch_rows = [r]
        # simple in-batch negatives from local shard
        for _ in range(1):
            batch_rows.append(random.choice(my_rows))

        q_texts = [x.get("query", "") for x in batch_rows]
        d_texts = [x.get("query", "") + " \n " + (x.get("answer", "") or "") for x in batch_rows]

        q_ids = tokenizer(q_texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").input_ids.to(device)
        d_ids = tokenizer(d_texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").input_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=dtype if dtype != torch.float32 else torch.bfloat16, enabled=(dtype != torch.float32)):
            q_vecs = model(q_ids, is_document_indexing=False)
            d_vecs = model(d_ids, is_document_indexing=True)
            q_emb = mean_pool(q_vecs)
            d_emb = mean_pool(d_vecs)
            sims = torch.matmul(q_emb, d_emb.T)
            labels = torch.arange(sims.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(sims / 0.05, labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0 and local_rank == 0:
            print(f"step={step} loss={loss.item():.4f}", flush=True)

    if local_rank == 0:
        out = Path("reports/phase1_smoke_result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"steps": args.steps, "status": "ok"}, indent=2))
        print("SMOKE_OK", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
