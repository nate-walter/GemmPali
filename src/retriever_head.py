import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrieverHead(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int = 128, pool_factor: int = 4):
        super().__init__()
        self.proj = nn.Linear(hidden_size, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.pool_factor = pool_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, T/pool, D] -> [B, T/pool, 128]
        if self.pool_factor > 1 and hidden_states.shape[1] >= self.pool_factor:
            x = hidden_states.transpose(1, 2)
            x = F.avg_pool1d(x, kernel_size=self.pool_factor, stride=self.pool_factor)
            hidden_states = x.transpose(1, 2)
        vecs = self.proj(hidden_states)
        return F.normalize(vecs, p=2, dim=-1)
