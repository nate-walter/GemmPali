import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks import build_mask
from src.retriever_head import RetrieverHead
from src.rope3d import Volumetric3DRoPE, RoPE3DConfig

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoProcessor = None


@dataclass
class WrapperConfig:
    model_name: str = "google/gemma-3-4b-it"
    embed_dim: int = 128
    pool_factor: int = 4
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    allow_dummy: bool = True
    force_dummy: bool = False
    load_in_4bit: bool = False


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(32000, hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids: torch.Tensor, **kwargs):
        h = self.ff(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=h)


class MultiPageRetrieverWrapper(nn.Module):
    """Gemma wrapper with ColPali-style projection head for smoke/integration tests."""

    def __init__(self, cfg: WrapperConfig):
        super().__init__()
        self.cfg = cfg
        self.processor = None
        self.backbone = None
        self.using_dummy = False
        self._load_backbone()
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self.backbone.config, "text_config", object()), "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Could not resolve hidden_size from backbone config")
        self.head = RetrieverHead(hidden_size, cfg.embed_dim, cfg.pool_factor)
        self.head = self.head.to(device=cfg.device, dtype=cfg.dtype)
        self.rope3d = Volumetric3DRoPE(RoPE3DConfig(enabled=False))

    def _load_backbone(self):
        if self.cfg.force_dummy:
            self.backbone = _DummyBackbone()
            self.using_dummy = True
            return

        if AutoModelForCausalLM is None:
            if not self.cfg.allow_dummy:
                raise RuntimeError("transformers not installed and allow_dummy=False")
            self.backbone = _DummyBackbone()
            self.using_dummy = True
            return

        try:
            self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if self.cfg.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.cfg.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["torch_dtype"] = self.cfg.dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                **load_kwargs,
            )
            self.backbone = model.get_model() if hasattr(model, "get_model") else model.model
        except Exception:
            if not self.cfg.allow_dummy:
                raise
            self.backbone = _DummyBackbone()
            self.using_dummy = True

    def generate_mask(self, seq_len: int, is_document_indexing: bool, device: torch.device) -> torch.Tensor:
        return build_mask(seq_len, is_document_indexing, device=device, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor = None, is_document_indexing: bool = True, **kwargs) -> torch.Tensor:
        # Accept full multimodal kwargs (input_ids, attention_mask, pixel_values, etc.)
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("MultiPageRetrieverWrapper.forward requires input_ids")

        attn = self.generate_mask(input_ids.shape[1], is_document_indexing, input_ids.device)
        _ = attn  # reserved for future custom attention hooks

        fw = dict(kwargs)
        fw["input_ids"] = input_ids

        if any(p.requires_grad for p in self.backbone.parameters()):
            out = self.backbone(**fw)
        else:
            with torch.no_grad():
                out = self.backbone(**fw)
        hidden_states = out.last_hidden_state
        hidden_states = self.rope3d(hidden_states)
        hidden_states = hidden_states.to(self.head.proj.weight.dtype)
        return self.head(hidden_states)


def maxsim_score(query_vecs: torch.Tensor, doc_vecs: torch.Tensor) -> torch.Tensor:
    # query_vecs: [B, Q, D], doc_vecs: [B, P, D]
    sims = torch.einsum("bqd,bpd->bqp", query_vecs, doc_vecs)
    return sims.max(dim=-1).values.sum(dim=-1)
