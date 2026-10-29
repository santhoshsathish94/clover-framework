"""HybridLink: a lightweight residual adapter that maps a last-layer hidden
state back into the input-embedding space, R(h) = h + W2 σ(W1 h), so that the agent
can continue generating in latent mode without decoding tokens.

Layer norms around the residual block keep the mapping stable; the parameter names
match the released checkpoints (`adapter.pt`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Tag stored in `adapter_config.json` next to the HybridLink weights.
HYBRIDLINK_CONFIG = {"adapter_type": "ln_res_adapter"}


class HybridLink(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.proj2(self.act(self.proj1(self.pre_ln(h))))
        return self.post_ln(h + out)
