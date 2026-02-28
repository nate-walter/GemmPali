import torch


def omni_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones((1, 1, seq_len, seq_len), dtype=dtype, device=device)


def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=dtype, device=device))


def build_mask(seq_len: int, is_document_indexing: bool, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if is_document_indexing:
        return omni_mask(seq_len, device, dtype)
    return causal_mask(seq_len, device, dtype)
