"""Exactness test for LFM3 candidate contextual representations (needed by the embedding-synonym method).
Reference: one full forward per candidate over [prefix + candidate]. Fast: one prefix forward with a cache, the cache
repeated across the candidates, one batched 1-token step. Also reports the cache layer types.
Usage: python lfm3_cache_test.py <model dir>"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
bos = model.config.bos_token_id
text = "The Urban Knife Caddy by Kapoosh holds the same amount of cutlery as a traditional knife block."
ids = torch.tensor([[bos] + tok(text, add_special_tokens=False).input_ids], device="cuda")
print("tokenizer class:", type(tok).__name__, "| DynamicCache.batch_repeat_interleave:",
      hasattr(DynamicCache, "batch_repeat_interleave"))


def repeat_cache(cache, n):
    """Repeat a filled LFM3 cache n times in the batch dimension: attention layers with their own method, conv
    layers (LinearAttentionLayer, no such method in transformers 5.22) by repeating their state tensors."""
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
            continue
        for states in (layer.conv_states, layer.recurrent_states):
            for k, v in states.items():
                if v is not None:
                    states[k] = v.repeat_interleave(n, dim=0)

with torch.inference_mode():
    full = model(ids, output_hidden_states=True, use_cache=False)
for idx in (3, 9, ids.shape[1] - 2):
    cands = torch.topk(full.logits[0, idx - 1].float(), 24).indices
    n = cands.shape[0]
    with torch.inference_mode():
        t0 = time.time()
        ref_in = torch.cat([ids[:, :idx].expand(n, -1), cands[:, None]], dim=1)
        ref = model(ref_in, output_hidden_states=True, use_cache=False).hidden_states[-1][:, idx].float()
        torch.cuda.synchronize(); t_ref = time.time() - t0
        t0 = time.time()
        pre = model(ids[:, :idx], use_cache=True)
        cache = pre.past_key_values
        if idx == 3:
            kinds = sorted({type(layer).__name__ for layer in cache.layers})
            print("cache:", type(cache).__name__, "layer kinds:", kinds)
        repeat_cache(cache, n)
        out = model(input_ids=cands[:, None], attention_mask=torch.ones(n, idx + 1, device="cuda", dtype=torch.long),
                    position_ids=torch.full((n, 1), idx, device="cuda"), past_key_values=cache,
                    output_hidden_states=True, use_cache=True)
        fast = out.hidden_states[-1][:, 0].float()
        torch.cuda.synchronize(); t_fast = time.time() - t0
    cos = torch.nn.functional.cosine_similarity(ref, fast, dim=-1)
    same_rank = (torch.argsort(cos) is not None) and torch.equal(
        torch.argsort(torch.nn.functional.cosine_similarity(ref, full.hidden_states[-1][0, idx].float()[None], -1)),
        torch.argsort(torch.nn.functional.cosine_similarity(fast, full.hidden_states[-1][0, idx].float()[None], -1)))
    print(f"idx {idx:3d}: {n} candidates | max|ref-fast| {(ref - fast).abs().max().item():.4f} "
          f"(ref norm {ref.norm(dim=-1).mean().item():.1f}) | min cos(ref,fast) {cos.min().item():.5f} | "
          f"same ranking by context cosine: {same_rank} | ref {t_ref * 1000:.0f} ms, fast {t_fast * 1000:.0f} ms")
