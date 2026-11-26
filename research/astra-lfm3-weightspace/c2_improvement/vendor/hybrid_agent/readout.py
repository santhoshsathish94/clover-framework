"""Interpretable Latent Mode: latent thoughts are rendered as
pseudo-text by mapping each HybridLink output to its nearest token embedding under
cosine similarity, so a latent span can be inspected without being decoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_BANK_CACHE: dict[tuple, torch.Tensor] = {}


def normalized_embeddings(embed_weight: torch.Tensor) -> torch.Tensor:
    """Unit-norm fp32 copy of the input-embedding table, built once per table."""
    key = (embed_weight.data_ptr(), tuple(embed_weight.shape), str(embed_weight.device))
    bank = _BANK_CACHE.get(key)
    if bank is None:
        bank = F.normalize(embed_weight.detach().float(), dim=-1)
        _BANK_CACHE.clear()
        _BANK_CACHE[key] = bank
    return bank


def nearest_token_ids(latent_inputs: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Nearest vocabulary token of each latent input [n, D] by cosine similarity."""
    z = F.normalize(latent_inputs.detach().float(), dim=-1)
    return (z @ bank.T).argmax(dim=-1)


def latent_readout(
    latent_inputs: torch.Tensor, embed_weight: torch.Tensor, tokenizer, top_k: int = 1
) -> list[list[tuple[str, int, float]]]:
    """Top-k (token, id, cosine) per latent step for offline inspection."""
    bank = normalized_embeddings(embed_weight)
    z = F.normalize(latent_inputs.detach().float(), dim=-1)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    sims = z @ bank.T
    values, indices = torch.topk(sims, k=top_k, dim=-1)
    out = []
    for vals, ids in zip(values.tolist(), indices.tolist()):
        out.append([(tokenizer.decode([i]), int(i), float(v)) for v, i in zip(vals, ids)])
    return out


@dataclass
class LatentSpanReadout:
    """Pseudo-text of one latent span and how it ended ("anchor" or "max_steps")."""
    steps: int
    exit: str
    token_ids: list[int] = field(default_factory=list)
    pseudo_text: str = ""

    def to_dict(self) -> dict:
        return {"steps": self.steps, "exit": self.exit, "pseudo_text": self.pseudo_text}


def pseudo_text(token_ids: list[int], tokenizer) -> str:
    return "".join(tokenizer.decode([tok]) for tok in token_ids)


def render_with_readout(output: str, spans: list[LatentSpanReadout]) -> str:
    """Fill the k-th empty `<latent></latent>` marker of a decoded output with the
    k-th span's pseudo-text."""
    marker = "<latent></latent>"
    parts = output.split(marker)
    rendered = parts[0]
    for k, part in enumerate(parts[1:]):
        text = spans[k].pseudo_text if k < len(spans) else ""
        rendered += f"<latent>{text}</latent>" + part
    return rendered
