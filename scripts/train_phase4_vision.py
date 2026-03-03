#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, AutoTokenizer

from src.model_wrapper import WrapperConfig, MultiPageRetrieverWrapper, maxsim_score

try:
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    pq = None
    _pq_err = e


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_doc_index(rows):
    by_doc = defaultdict(list)
    for r in rows:
        did = r.get("target_doc_id")
        if did:
            by_doc[did].append(r)
    return by_doc


def choose_neg(anchor, pool, by_doc, intra_doc=False):
    if intra_doc:
        did = anchor.get("target_doc_id")
        if did and did in by_doc and len(by_doc[did]) > 1:
            for _ in range(20):
                cand = random.choice(by_doc[did])
                if cand.get("pair_id") != anchor.get("pair_id"):
                    return cand
    return random.choice(pool)


def set_lr(optimizer, step, total_steps, base_lr, scheduler, warmup_steps):
    if warmup_steps > 0 and step <= warmup_steps:
        lr = base_lr * (step / float(max(1, warmup_steps)))
    elif scheduler in {"cosine", "cosine_with_warmup"}:
        rem = max(1, total_steps - warmup_steps)
        p = (step - warmup_steps) / float(rem)
        p = min(max(p, 0.0), 1.0)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * p))
    else:
        lr = base_lr
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


class ParquetImageResolver:
    def __init__(self, parquet_globs, cache_dir):
        if pq is None:
            raise RuntimeError(f"pyarrow is required for raw-image loading: {_pq_err}")
        self.parquet_globs = parquet_globs
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.materialized = {}
        self._indexed = False

    @staticmethod
    def _to_int(v):
        try:
            if v is None or v == "":
                return None
            return int(v)
        except Exception:
            return None

    @staticmethod
    def _is_image_path(s):
        if not isinstance(s, str):
            return False
        l = s.lower()
        return any(l.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]) or "/" in s

    def _ensure_index(self):
        if self._indexed:
            return
        self._build_index()
        self._indexed = True

    def _build_index(self):
        import glob

        files = []
        for g in self.parquet_globs:
            files.extend(glob.glob(g))
        files = sorted(set(files))
        for f in files:
            try:
                table = pq.read_table(f, columns=None)
            except Exception:
                continue
            cols = set(table.column_names)
            d = table.to_pydict()
            n = table.num_rows

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

    def _materialize_from_parquet(self, key, rec):
        if key in self.materialized and Path(self.materialized[key]).exists():
            return self.materialized[key]

        f, i = rec
        table = pq.read_table(f)
        d = table.to_pydict()
        img_obj = d.get("image", [None])[i]
        if not isinstance(img_obj, dict):
            raise RuntimeError(f"image payload missing in parquet row: {f}:{i}")

        path_hint = img_obj.get("path")
        if path_hint:
            p = Path(path_hint)
            if p.exists():
                self.materialized[key] = str(p)
                return str(p)

        img_bytes = img_obj.get("bytes")
        if not img_bytes:
            raise RuntimeError(f"image bytes missing in parquet row: {f}:{i}")

        out = self.cache_dir / f"img_{abs(hash((f, i)))}.png"
        if not out.exists():
            im = Image.open(BytesIO(img_bytes)).convert("RGB")
            im.save(out)
        self.materialized[key] = str(out)
        return str(out)

    def resolve_path(self, row):
        for k in ["image_path", "doc_image_path", "target_image", "target_image_path"]:
            v = row.get(k)
            if isinstance(v, str) and Path(v).exists():
                return v
        meta = row.get("meta") or {}
        for k in ["image_path", "doc_image_path", "target_image_path", "path"]:
            v = meta.get(k)
            if isinstance(v, str) and Path(v).exists():
                return v

        did = row.get("target_doc_id")
        pg = self._to_int(row.get("target_page", None))

        if self._is_image_path(did):
            p = Path(did)
            if p.exists():
                return str(p)

        self._ensure_index()

        keys = []
        if did is not None:
            keys.extend([(did, pg), (did, None)])
        if isinstance(did, str):
            keys.extend([(did.split("/")[-1], pg), (did.split("/")[-1], None)])

        for key in keys:
            rec = self.index.get(key)
            if rec is not None:
                return self._materialize_from_parquet(key, rec)

        return None

    def load_pil(self, row):
        p = self.resolve_path(row)
        if not p:
            raise RuntimeError(f"cannot resolve image for pair_id={row.get('pair_id')} doc={row.get('target_doc_id')} page={row.get('target_page')}")
        return Image.open(p).convert("RGB")


def encode_queries(model, tokenizer, texts, max_len, device, dtype):
    m = model.module if hasattr(model, "module") else model
    # Gemma-3 text path: use chat template so token routing matches model expectations.
    messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in texts]
    prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]

    q = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    q = {k: v.to(device) for k, v in q.items() if torch.is_tensor(v)}

    with torch.autocast(device_type="cuda", dtype=dtype if dtype != torch.float32 else torch.bfloat16, enabled=(dtype != torch.float32)):
        q_seq = m(**q, is_document_indexing=False)
        q_seq = F.normalize(q_seq, p=2, dim=-1)
    return q_seq, q["attention_mask"]


def encode_docs_from_images(model, processor, images, max_len, device, dtype):
    m = model.module if hasattr(model, "module") else model
    # Gemma-3 multimodal routing requires structured chat template with explicit image content.
    messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Index this document page for retrieval."}]}] for _ in images]
    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]

    # Gemma-3 processor expects per-sample image lists aligned to each text prompt.
    image_inputs = [[im] for im in images]
    batch = processor(images=image_inputs, text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

    with torch.autocast(device_type="cuda", dtype=dtype if dtype != torch.float32 else torch.bfloat16, enabled=(dtype != torch.float32)):
        d_seq = m(**batch, is_document_indexing=True)
        d_seq = F.normalize(d_seq, p=2, dim=-1)
    return d_seq, batch["attention_mask"], batch


def batched_maxsim(q_seq, d_seq, q_mask, d_mask):
    # q_seq: [B,Q,D], d_seq: [C,P,D], q_mask:[B,Q0], d_mask:[C,P0]
    q_mask = _align_mask_2d(q_mask, q_seq.size(1))
    d_mask = _align_mask_2d(d_mask, d_seq.size(1))

    sims = torch.einsum("bqd,cpd->bcqp", q_seq, d_seq)
    d_mask_ext = d_mask.view(1, d_seq.size(0), 1, d_seq.size(1)).bool()
    sims = sims.masked_fill(~d_mask_ext, float("-inf"))
    max_sims = sims.max(dim=-1).values
    q_mask_ext = q_mask.view(q_seq.size(0), 1, q_seq.size(1)).bool()
    max_sims = max_sims.masked_fill(~q_mask_ext, 0.0)
    return max_sims.sum(dim=-1)




def _align_mask_2d(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Align [B, T] token mask to pooled sequence length target_len."""
    if mask is None:
        raise ValueError('mask is None')
    if mask.dim() != 2:
        raise ValueError(f'expected 2D mask [B,T], got {tuple(mask.shape)}')
    B, T = mask.shape
    if T == target_len:
        return mask.bool()
    m = mask.float().unsqueeze(1)  # [B,1,T]
    # adaptive max-pool preserves any-valid-token semantics under pooling/downsampling
    m2 = F.adaptive_max_pool1d(m, target_len).squeeze(1)
    return (m2 > 0.5)


def _align_mask_3d(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Align [B, N, T] token mask to pooled sequence length target_len."""
    if mask is None:
        raise ValueError('mask is None')
    if mask.dim() != 3:
        raise ValueError(f'expected 3D mask [B,N,T], got {tuple(mask.shape)}')
    B, N, T = mask.shape
    if T == target_len:
        return mask.bool()
    m = mask.float().view(B * N, 1, T)
    m2 = F.adaptive_max_pool1d(m, target_len).view(B, N, target_len)
    return (m2 > 0.5)

def hardneg_maxsim(q_seq, hard_seq, q_mask, hard_mask):
    # q_seq:[B,Q,D], hard_seq:[B,N,P,D], q_mask:[B,Q0], hard_mask:[B,N,P0]
    q_mask = _align_mask_2d(q_mask, q_seq.size(1))
    hard_mask = _align_mask_3d(hard_mask, hard_seq.size(2))

    sims = torch.einsum("bqd,bnpd->bnqp", q_seq, hard_seq)
    h_mask_ext = hard_mask.unsqueeze(2).bool()
    sims = sims.masked_fill(~h_mask_ext, float("-inf"))
    max_sims = sims.max(dim=-1).values
    q_mask_ext = q_mask.unsqueeze(1).bool()
    max_sims = max_sims.masked_fill(~q_mask_ext, 0.0)
    return max_sims.sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--lr", type=float, default=4e-6)
    ap.add_argument("--scheduler", default="cosine_with_warmup")
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-grad-value", type=float, default=0.1)
    ap.add_argument("--intra-doc-negatives", type=str2bool, default=True)
    ap.add_argument("--in-batch-negatives", type=str2bool, default=True)
    ap.add_argument("--negatives-per-query", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--keep-top-k", type=int, default=2)
    ap.add_argument("--archive-dir", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--cache-dir", default="nvme_cache/vision_cache")
    ap.add_argument("--parquet-glob", action="append", default=[])
    ap.add_argument("--debug-vision-every", type=int, default=25)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
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

    rows = load_jsonl(args.pairs)
    if not rows:
        raise RuntimeError("no rows")

    eval_holdout = min(max(128, len(rows) // 20), max(1, len(rows) - 1)) if len(rows) > 512 else max(1, len(rows) // 10)
    train_rows = rows[:-eval_holdout] if len(rows) > eval_holdout else rows
    eval_rows = rows[-eval_holdout:] if len(rows) > eval_holdout else rows

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    for p in wrapper.backbone.parameters():
        p.requires_grad = False
    for p in wrapper.rope3d.parameters():
        p.requires_grad = False
    for p in wrapper.head.parameters():
        p.requires_grad = True

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: peft. Install with `pip install peft` in the training env before running phase4 vision trainer."
        ) from e

    if args.load_in_4bit:
        wrapper.backbone = prepare_model_for_kbit_training(wrapper.backbone, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg)

    for n, p in wrapper.backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    model = DDP(wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    params = [p for p in model.module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    if local_rank == 0:
        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in model.module.parameters())
        print(f"trainable_params={trainable} total_params={total} pct={(100.0*trainable/max(1,total)):.4f}", flush=True)
        print("step=0 loss=0.0000 eval_loss=na grad_norm=0.0000 lr=0.00000000", flush=True)

    default_globs = [
        "nvme_cache/raw/mpdocvqa-corpus/**/*.parquet",
        "nvme_cache/raw/dude-corpus/**/*.parquet",
        "nvme_cache/raw/vidore-colpali-train-set/**/*.parquet",
        "nvme_cache/raw/vidore-docvqa-train/**/*.parquet",
    ]
    globs = args.parquet_glob or default_globs
    resolver = ParquetImageResolver(globs, cache_dir=args.cache_dir)

    my_rows = train_rows[local_rank::world_size] or train_rows
    my_eval = eval_rows[local_rank::world_size] or eval_rows
    by_doc = build_doc_index(my_rows)
    by_doc_eval = build_doc_index(my_eval)

    checkpoint_meta = []
    last_eval = None

    for step in range(1, args.steps + 1):
        lr_now = set_lr(optimizer, step, args.steps, args.lr, args.scheduler, args.warmup_steps)

        anchors = [my_rows[(step * args.batch_size + i) % len(my_rows)] for i in range(args.batch_size)]
        q_texts = [a.get("query", "") for a in anchors]
        q_seq, q_mask = encode_queries(model, tokenizer, q_texts, args.max_len, device, dtype)

        pos_images = [resolver.load_pil(a) for a in anchors]
        pos_seq, pos_mask, pos_batch = encode_docs_from_images(model, model.module.processor, pos_images, args.max_len, device, dtype)

        hard_imgs = []
        for a in anchors:
            this_h = []
            for _ in range(max(0, args.negatives_per_query)):
                picked = None
                for _try in range(20):
                    cand = choose_neg(a, my_rows, by_doc, intra_doc=args.intra_doc_negatives)
                    rp = resolver.resolve_path(cand)
                    if rp:
                        picked = cand
                        break
                if picked is None:
                    picked = choose_neg(a, my_rows, by_doc, intra_doc=False)
                this_h.append(resolver.load_pil(picked))
            hard_imgs.extend(this_h)

        if hard_imgs:
            hard_seq_flat, hard_mask_flat, _ = encode_docs_from_images(model, model.module.processor, hard_imgs, args.max_len, device, dtype)
            hard_seq = hard_seq_flat.view(args.batch_size, args.negatives_per_query, hard_seq_flat.shape[1], hard_seq_flat.shape[2])
            hard_mask = hard_mask_flat.view(args.batch_size, args.negatives_per_query, hard_mask_flat.shape[1])
        else:
            hard_seq = None
            hard_mask = None

        all_sims = batched_maxsim(q_seq, pos_seq, q_mask, pos_mask)
        pos_scores = torch.diag(all_sims).unsqueeze(1)
        logits_parts = [pos_scores]

        if args.in_batch_negatives and args.batch_size > 1:
            mask = torch.eye(args.batch_size, dtype=torch.bool, device=device)
            inb = all_sims.masked_fill(mask, float("-inf"))
            logits_parts.append(inb)

        if hard_seq is not None:
            hard_scores = hardneg_maxsim(q_seq, hard_seq, q_mask, hard_mask)
            logits_parts.append(hard_scores)

        logits = torch.cat(logits_parts, dim=1).to(torch.float32) / args.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if args.max_grad_value is not None:
            torch.nn.utils.clip_grad_value_(params, args.max_grad_value)
        optimizer.step()

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for bi in range(max(1, args.eval_batches)):
                    a = my_eval[(step + bi) % len(my_eval)]
                    qe, qme = encode_queries(model.module, tokenizer, [a.get("query", "")], args.max_len, device, dtype)
                    pe, pme, _ = encode_docs_from_images(model.module, model.module.processor, [resolver.load_pil(a)], args.max_len, device, dtype)

                    n_rows = [choose_neg(a, my_eval, by_doc_eval, intra_doc=args.intra_doc_negatives) for _ in range(max(1, args.negatives_per_query))]
                    n_imgs = [resolver.load_pil(nr) for nr in n_rows]
                    ne_flat, nme_flat, _ = encode_docs_from_images(model.module, model.module.processor, n_imgs, args.max_len, device, dtype)
                    ne = ne_flat.unsqueeze(0)
                    nme = nme_flat.unsqueeze(0)

                    pos = batched_maxsim(qe, pe, qme, pme).diag().unsqueeze(1)
                    neg = hardneg_maxsim(qe, ne, qme, nme)
                    lg = torch.cat([pos, neg], dim=1).to(torch.float32) / args.temperature
                    lb = torch.zeros(1, dtype=torch.long, device=device)
                    e_loss = torch.nn.functional.cross_entropy(lg, lb)
                    eval_losses.append(e_loss.detach())

                eval_local = torch.stack(eval_losses).mean()
                eval_global = eval_local.clone()
                dist.all_reduce(eval_global, op=dist.ReduceOp.SUM)
                eval_global = eval_global / world_size
                last_eval = float(eval_global.item())
            model.train()

        if step % args.log_every == 0 and local_rank == 0:
            eval_str = f"{last_eval:.4f}" if last_eval is not None else "na"
            grad_now = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            print(f"step={step} loss={loss.item():.4f} eval_loss={eval_str} grad_norm={grad_now:.4f} lr={lr_now:.8f}", flush=True)

        if step % args.debug_vision_every == 0 and local_rank == 0:
            px = pos_batch.get("pixel_values", None)
            px_shape = tuple(px.shape) if px is not None else None
            hshape = tuple(hard_seq.shape) if hard_seq is not None else None
            print(
                f"vision_debug step={step} pixel_values_shape={px_shape} q_seq={tuple(q_seq.shape)} pos_seq={tuple(pos_seq.shape)} hard_seq={hshape}",
                flush=True,
            )

        if local_rank == 0 and step % args.save_every == 0:
            sdir = Path(args.save_dir)
            sdir.mkdir(parents=True, exist_ok=True)
            ckpt = sdir / f"head_step_{step:07d}.pt"
            train_loss_now = float(loss.item())
            eval_loss_now = float(last_eval) if last_eval is not None else None
            score = eval_loss_now if eval_loss_now is not None else train_loss_now

            lora_state = {k: v.detach().cpu() for k, v in model.module.backbone.state_dict().items() if "lora_" in k}
            torch.save(
                {
                    "step": step,
                    "head": model.module.head.state_dict(),
                    "lora": lora_state,
                    "train_loss": train_loss_now,
                    "eval_loss": eval_loss_now,
                    "score": score,
                    "cfg": vars(args),
                },
                ckpt,
            )
            print(f"checkpoint_saved={ckpt}", flush=True)

            checkpoint_meta.append(
                {
                    "step": step,
                    "path": str(ckpt),
                    "train_loss": train_loss_now,
                    "eval_loss": eval_loss_now,
                    "score": score,
                }
            )
            checkpoint_meta.sort(key=lambda x: (x["score"], x["step"]))
            keep = checkpoint_meta[: max(1, args.keep_top_k)]
            remove = checkpoint_meta[max(1, args.keep_top_k):]
            archive_dir = Path(args.archive_dir) if args.archive_dir else None
            for item in remove:
                pth = Path(item["path"])
                if pth.exists():
                    if archive_dir is not None:
                        archive_dir.mkdir(parents=True, exist_ok=True)
                        dst = archive_dir / pth.name
                        try:
                            shutil.move(str(pth), str(dst))
                            print(f"checkpoint_archived={dst}", flush=True)
                        except Exception:
                            pth.unlink(missing_ok=True)
                    else:
                        pth.unlink(missing_ok=True)
            checkpoint_meta = [x for x in keep if Path(x["path"]).exists()]
            (sdir / "checkpoint_index.json").write_text(json.dumps({"top_k": args.keep_top_k, "checkpoints": checkpoint_meta}, indent=2))

    if local_rank == 0:
        out = Path("reports/phase4_vision_result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"steps": args.steps, "status": "ok", "vision_path": "active"}, indent=2))
        print("SMOKE_OK", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
