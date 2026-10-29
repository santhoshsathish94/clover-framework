"""Batched decoding primitives for the hybrid generation engine: fan a prefilled KV cache
out to B rows, run one lockstep forward with per-row attention masks and positions, and
sample text tokens under the mode-dependent token masks.
"""
from __future__ import annotations

import torch


def _cache_layers(past_kv) -> list:
    layers = getattr(past_kv, "layers", None)
    if layers is None:
        raise TypeError(f"unsupported cache object {type(past_kv).__name__!r}: no .layers")
    return list(layers)


def _layer_tensor_attrs(layer) -> list[tuple[str, torch.Tensor]]:
    """Every tensor attribute of a cache layer that carries a batch dimension."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, value in vars(layer).items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            out.append((name, value))
    return out


def expand_cache_batch(past_kv, batch_size: int) -> None:
    """Expand a batch-1 prefill cache to `batch_size` rows in place."""
    for layer in _cache_layers(past_kv):
        for name, tensor in _layer_tensor_attrs(layer):
            if tensor.shape[0] == batch_size:
                continue
            if tensor.shape[0] != 1:
                raise ValueError(f"cannot expand cache attr {name!r} with batch {tensor.shape[0]}")
            setattr(layer, name, tensor.repeat_interleave(batch_size, dim=0).contiguous())
        if hasattr(layer, "batch_size") and isinstance(getattr(layer, "batch_size"), int):
            layer.batch_size = batch_size


def forward_step(
    lm,
    *,
    input_embeds: torch.Tensor,        # [B, 1, D]
    past_key_values,
    attention_mask: torch.Tensor,      # [B, L_total] incl. the new slot
    position_ids: torch.Tensor,        # [B, 1] per-row true RoPE positions
    cache_position: torch.Tensor,      # [1] global cache slot index
):
    """One lockstep decode forward; returns (cache, next logits [B, V], last hidden [B, D])."""
    outputs = lm(
        inputs_embeds=input_embeds,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return (
        outputs.past_key_values,
        outputs.logits[:, -1, :],
        outputs.hidden_states[-1][:, -1, :],
    )


def token_log_probs(
    logits: torch.Tensor,              # [B, V]
    temperature: float,
    top_p: float,
    top_k: int,
    blocked_rows: torch.Tensor,        # [B, V] bool, True = blocked
) -> torch.Tensor:
    """Log-probs of the sampling distribution: block -> temperature -> top-k -> softmax ->
    top-p -> renormalize. Excluded tokens get -inf."""
    x = logits.float().masked_fill(blocked_rows, float("-inf"))

    if temperature == 0.0:
        selected = x.argmax(dim=-1, keepdim=True)
        out = torch.full_like(x, float("-inf"))
        return out.scatter(1, selected, 0.0)

    scaled = x / max(temperature, 1e-5)
    if top_k and 0 < top_k < scaled.shape[-1]:
        kth = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumulative - sorted_probs) <= top_p
        keep[..., 0] = True
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return torch.log(probs.clamp_min(1e-45)).masked_fill(probs <= 0, float("-inf"))


def sample_rows(
    log_probs: torch.Tensor,           # [B, V]
    rows: list[int],
    rng: torch.Generator | None,
) -> dict[int, int]:
    """Sample one token per listed row, consuming the generator in ascending row order."""
    out: dict[int, int] = {}
    for row in sorted(rows):
        probs = log_probs[row].exp()
        out[row] = int(torch.multinomial(probs, 1, generator=rng).item())
    return out
