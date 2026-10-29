#!/usr/bin/env python3
"""
Stage-by-Stage Mathematical & Empirical Study of Liquid AI LFM3-1.7B (`LiquidAI/LFM3-1.7B`).

This script inspects, reconstructs, and verifies every stage of `LiquidAI/LFM3-1.7B`:
  - Stage 0: Tokenizer & ChatML formatting (`<|startoftext|><|im_start|>...`)
  - Stage 1: Token Embedding (`model.embed_tokens`, [128000, 2048], tied with `lm_head`)
  - Stage 2: Tri-Hybrid Rotary Position Embeddings (`Lfm3RotaryEmbedding`):
      * `sliding_attention`: theta=10,000.0, full rotation (64 frequency pairs)
      * `full_attention`: theta=1,000,000.0, zero-tail partial rotation (`partial_rotary_factor=0.5`,
        32 rotated pairs + 32 zero pairs)
  - Stage 3: All 24 Tri-Hybrid Decoder Layers (`Lfm3DecoderLayer`, layers 0..23):
      * 13 Gated Short Convolution layers (`conv`):
        operator_norm -> in_proj (2048->6144) -> chunk(B, C, x) -> pre-gate (B*x)
        -> depthwise causal conv1d (width K=7) -> post-gate (C*conv) -> out_proj (2048->2048)
        -> post_operator_norm -> residual_1
      * 5 Sliding-Window Attention layers (`sliding_attention`, window=256) &
        6 Full Global Attention layers (`full_attention`, 256K window):
        operator_norm -> q_proj/k_proj/v_proj (GQA 16Q:4KV, head_dim=128)
        -> per-head q_norm & k_norm (RMSNorm(128)) -> float32 RoPE -> causal SDPA
        -> per-head sigmoid gate (`sigmoid(gate_proj(u))` [B, T, 16, 1]) -> o_proj
        -> post_operator_norm -> residual_1
      * Dense SwiGLU MLP in all 24 layers:
        ffn_norm -> silu(w1(x)) * w3(x) (2048->7680) -> w2 (7680->2048)
        -> post_ffn_norm -> residual_2
  - Stage 4: Final RMSNorm (`model.embedding_norm`, [2048])
  - Stage 5: Tied Output Projection (`lm_head`, linear with `model.embed_tokens.weight` [128000, 2048])
"""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers.cache_utils as _tf_cache
import transformers.integrations as _tf_integrations
import transformers.masking_utils as _tf_masking

if hasattr(_tf_cache, "LinearAttentionLayer"):
    if not hasattr(_tf_cache.LinearAttentionLayer, "record_past"):
        _tf_cache.LinearAttentionLayer.record_past = False
    _orig_update_conv = getattr(_tf_cache.DynamicCache, "update_conv_state", None)
    if _orig_update_conv is not None:
        def _compat_update_conv_state(self, conv_states, layer_idx, **kwargs):
            res = _orig_update_conv(self, conv_states, layer_idx, **kwargs)
            layer_cache = self.layers[layer_idx]
            if (
                hasattr(layer_cache, "conv_states")
                and isinstance(layer_cache.conv_states, torch.Tensor)
                and layer_cache.conv_states.ndim == 3
            ):
                layer_cache.conv_states = [layer_cache.conv_states]
            return res

        _tf_cache.DynamicCache.update_conv_state = _compat_update_conv_state

if not hasattr(_tf_integrations, "use_kernel_func_from_hub_with_fallback"):
    def _noop_kernel_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

    _tf_integrations.use_kernel_func_from_hub_with_fallback = _noop_kernel_decorator

_orig_create_causal_mask = _tf_masking.create_causal_mask
if "layer_idx" not in inspect.signature(_orig_create_causal_mask).parameters:
    def _compat_create_causal_mask(*args, **kwargs):
        kwargs.pop("layer_idx", None)
        return _orig_create_causal_mask(*args, **kwargs)

    _tf_masking.create_causal_mask = _compat_create_causal_mask

_orig_create_sliding_mask = _tf_masking.create_sliding_window_causal_mask
if "layer_idx" not in inspect.signature(_orig_create_sliding_mask).parameters:
    def _compat_create_sliding_mask(*args, **kwargs):
        kwargs.pop("layer_idx", None)
        return _orig_create_sliding_mask(*args, **kwargs)

    _tf_masking.create_sliding_window_causal_mask = _compat_create_sliding_mask

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("/opt/clover-k3/models/LFM3-1.7B")
REPORT_PATH = Path("/tmp/lfm3_stage_study_report.json")


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Exact reference implementation of Lfm3RMSNorm."""
    in_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(var + eps)
    return weight * normed.to(in_dtype)


def rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_ref(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_u = cos.unsqueeze(1).float()
    sin_u = sin.unsqueeze(1).float()
    q_embed = (q.float() * cos_u) + (rotate_half_ref(q.float()) * sin_u)
    k_embed = (k.float() * cos_u) + (rotate_half_ref(k.float()) * sin_u)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def compute_rope_cos_sin_ref(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reference for Lfm3RotaryEmbedding zero-tail partial rotation."""
    rope_angles = int(partial_rotary_factor * head_dim) // 2
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device) / head_dim)
    )
    inv_freq = torch.cat(
        (inv_freq, torch.zeros(head_dim // 2 - rope_angles, dtype=torch.float32, device=device))
    )
    freqs = position_ids[..., None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def inspect_safetensors(model_dir: Path) -> dict[str, Any]:
    st_path = model_dir / "model.safetensors"
    tensors_info = []
    total_params = 0
    total_bytes = 0
    by_category: dict[str, int] = {
        "embedding_tied": 0,
        "conv_operators": 0,
        "sliding_attention_operators": 0,
        "full_attention_operators": 0,
        "swiglu_mlp": 0,
        "norms": 0,
    }
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layer_types = config["layer_types"]

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        for name in keys:
            sl = f.get_slice(name)
            shape = list(sl.get_shape())
            dtype = sl.get_dtype()
            numel = math.prod(shape)
            bpe = 2 if dtype in ("BF16", "F16") else 4
            nbytes = numel * bpe
            total_params += numel
            total_bytes += nbytes

            if name == "model.embed_tokens.weight":
                by_category["embedding_tied"] += numel
            elif ".feed_forward." in name:
                by_category["swiglu_mlp"] += numel
            elif ".conv." in name:
                by_category["conv_operators"] += numel
            elif ".self_attn." in name:
                layer_idx = int(name.split(".")[2])
                ltype = layer_types[layer_idx]
                if ltype == "sliding_attention":
                    by_category["sliding_attention_operators"] += numel
                else:
                    by_category["full_attention_operators"] += numel
            elif "norm" in name:
                by_category["norms"] += numel

            if len(tensors_info) < 35 or ".layers.2." in name or ".layers.4." in name:
                tensors_info.append(
                    {"name": name, "shape": shape, "dtype": dtype, "numel": numel}
                )

    return {
        "num_tensors": len(keys),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "by_category": by_category,
        "sample_tensors": tensors_info,
    }


@torch.no_grad()
def run_manual_stage_by_stage_verification(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokens_str: list[str],
) -> dict[str, Any]:
    """
    Runs each stage manually using explicit tensor equations and compares every stage's
    output against the live `Lfm3ForCausalLM` forward pass.
    """
    config = model.config
    dtype = next(model.parameters()).dtype
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    official_layer_outputs: list[torch.Tensor] = []
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def make_hook(layer_i: int):
            def _hook(_mod, _inp, out):
                official_layer_outputs.append(out.detach().clone())
            return _hook
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    t_official_0 = time.perf_counter()
    official_out = model(input_ids=input_ids, use_cache=False)
    t_official_ms = (time.perf_counter() - t_official_0) * 1000.0
    for h in hooks:
        h.remove()

    t0 = time.perf_counter()
    h = F.embedding(input_ids, model.model.embed_tokens.weight, padding_idx=config.pad_token_id)
    embed_ms = (time.perf_counter() - t0) * 1000.0
    embed_rms_per_pos = [round(float(h[0, p].float().pow(2).mean().sqrt()), 6) for p in range(seq_len)]

    cos_full, sin_full = compute_rope_cos_sin_ref(
        position_ids,
        head_dim=config.head_dim,
        rope_theta=config.rope_parameters["full_attention"]["rope_theta"],
        partial_rotary_factor=config.rope_parameters["full_attention"].get("partial_rotary_factor", 1.0),
        dtype=dtype,
        device=device,
    )
    cos_sliding, sin_sliding = compute_rope_cos_sin_ref(
        position_ids,
        head_dim=config.head_dim,
        rope_theta=config.rope_parameters["sliding_attention"]["rope_theta"],
        partial_rotary_factor=config.rope_parameters["sliding_attention"].get("partial_rotary_factor", 1.0),
        dtype=dtype,
        device=device,
    )

    off_cos_full, off_sin_full = model.model.rotary_emb(h, position_ids=position_ids, layer_type="full_attention")
    off_cos_slid, off_sin_slid = model.model.rotary_emb(h, position_ids=position_ids, layer_type="sliding_attention")
    rope_full_max_diff = float((cos_full - off_cos_full).abs().max() + (sin_full - off_sin_full).abs().max())
    rope_sliding_max_diff = float((cos_sliding - off_cos_slid).abs().max() + (sin_sliding - off_sin_slid).abs().max())

    layer_records = []
    for i in range(config.num_hidden_layers):
        layer_t0 = time.perf_counter()
        layer = model.model.layers[i]
        ltype = config.layer_types[i]

        in_norm_val = float(h.float().pow(2).mean().sqrt())
        residual_1 = h
        op_in = rms_norm_ref(h, layer.operator_norm.weight, eps=config.norm_eps)

        substage_stats: dict[str, Any] = {}
        if ltype == "conv":
            conv_mod = layer.conv
            bcx = F.linear(op_in, conv_mod.in_proj.weight, conv_mod.in_proj.bias).transpose(-1, -2)
            B_gate, C_gate, x_branch = bcx.chunk(3, dim=-2)
            pre_gated = B_gate * x_branch
            conv_raw = conv_mod.conv(pre_gated.to(conv_mod.conv.weight.dtype))[..., :seq_len].to(op_in.dtype)
            post_gated = (C_gate * conv_raw).transpose(-1, -2).contiguous()
            op_out_raw = F.linear(post_gated, conv_mod.out_proj.weight, conv_mod.out_proj.bias)

            substage_stats = {
                "B_gate_rms": round(float(B_gate.float().pow(2).mean().sqrt()), 5),
                "C_gate_rms": round(float(C_gate.float().pow(2).mean().sqrt()), 5),
                "pre_gated_rms": round(float(pre_gated.float().pow(2).mean().sqrt()), 5),
                "conv_raw_rms": round(float(conv_raw.float().pow(2).mean().sqrt()), 5),
                "post_gated_rms": round(float(post_gated.float().pow(2).mean().sqrt()), 5),
            }
        else:
            attn_mod = layer.self_attn
            head_dim = attn_mod.head_dim
            n_heads = config.num_attention_heads
            n_kv = config.num_key_value_heads
            n_groups = n_heads // n_kv

            q = F.linear(op_in, attn_mod.q_proj.weight).view(batch_size, seq_len, n_heads, head_dim)
            k = F.linear(op_in, attn_mod.k_proj.weight).view(batch_size, seq_len, n_kv, head_dim)
            v = F.linear(op_in, attn_mod.v_proj.weight).view(batch_size, seq_len, n_kv, head_dim)

            q = rms_norm_ref(q, attn_mod.q_norm.weight, eps=config.norm_eps).transpose(1, 2)
            k = rms_norm_ref(k, attn_mod.k_norm.weight, eps=config.norm_eps).transpose(1, 2)
            v = v.transpose(1, 2)

            cos, sin = (cos_full, sin_full) if ltype == "full_attention" else (cos_sliding, sin_sliding)
            q, k = apply_rope_ref(q, k, cos, sin)

            k_exp = k[:, :, None, :, :].expand(batch_size, n_kv, n_groups, seq_len, head_dim).reshape(batch_size, n_heads, seq_len, head_dim)
            v_exp = v[:, :, None, :, :].expand(batch_size, n_kv, n_groups, seq_len, head_dim).reshape(batch_size, n_heads, seq_len, head_dim)

            scores = torch.matmul(q, k_exp.transpose(2, 3)) * (head_dim ** -0.5)
            row_idx = torch.arange(seq_len, device=device)[:, None]
            col_idx = torch.arange(seq_len, device=device)[None, :]
            mask = col_idx <= row_idx
            if ltype == "sliding_attention":
                mask = mask & ((row_idx - col_idx) < config.sliding_window)
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_val = torch.matmul(probs, v_exp).transpose(1, 2).contiguous()

            head_gate = torch.sigmoid(F.linear(op_in, attn_mod.gate_proj.weight)).unsqueeze(-1)
            gated_attn = (attn_val * head_gate).reshape(batch_size, seq_len, n_heads * head_dim)
            op_out_raw = F.linear(gated_attn, attn_mod.o_proj.weight)

            probs_fp32 = probs.float().clamp_min(1e-12)
            entropy = float(-(probs_fp32 * probs_fp32.log()).sum(-1).mean())
            substage_stats = {
                "head_gate_mean": round(float(head_gate.float().mean()), 5),
                "head_gate_min": round(float(head_gate.float().min()), 5),
                "head_gate_max": round(float(head_gate.float().max()), 5),
                "attn_entropy_nats": round(entropy, 5),
                "attn_val_rms": round(float(attn_val.float().pow(2).mean().sqrt()), 5),
            }

        op_raw_rms = float(op_out_raw.float().pow(2).mean().sqrt())
        op_out_normed = rms_norm_ref(op_out_raw, layer.post_operator_norm.weight, eps=config.norm_eps)
        h_mid = residual_1 + op_out_normed

        ffn_in = rms_norm_ref(h_mid, layer.ffn_norm.weight, eps=config.norm_eps)
        w1_out = F.linear(ffn_in, layer.feed_forward.w1.weight)
        w3_out = F.linear(ffn_in, layer.feed_forward.w3.weight)
        swiglu_act = F.silu(w1_out) * w3_out
        ffn_raw = F.linear(swiglu_act, layer.feed_forward.w2.weight)
        ffn_raw_rms = float(ffn_raw.float().pow(2).mean().sqrt())
        ffn_normed = rms_norm_ref(ffn_raw, layer.post_ffn_norm.weight, eps=config.norm_eps)
        h = h_mid + ffn_normed

        layer_ms = (time.perf_counter() - layer_t0) * 1000.0

        off_h = official_layer_outputs[i]
        max_abs_diff = float((h.float() - off_h.float()).abs().max())
        cos_sim = float(
            F.cosine_similarity(h.float().reshape(1, -1), off_h.float().reshape(1, -1), dim=-1).item()
        )
        out_rms = float(h.float().pow(2).mean().sqrt())
        tok0_rms = float(h[0, 0].float().pow(2).mean().sqrt())
        last_tok_rms = float(h[0, -1].float().pow(2).mean().sqrt())
        op_branch_rms = float(op_out_normed.float().pow(2).mean().sqrt())
        ffn_branch_rms = float(ffn_normed.float().pow(2).mean().sqrt())

        layer_records.append({
            "layer_idx": i,
            "layer_type": ltype,
            "in_rms": round(in_norm_val, 5),
            "op_raw_rms": round(op_raw_rms, 5),
            "op_branch_rms": round(op_branch_rms, 5),
            "ffn_raw_rms": round(ffn_raw_rms, 5),
            "ffn_branch_rms": round(ffn_branch_rms, 5),
            "out_rms": round(out_rms, 5),
            "tok0_rms": round(tok0_rms, 5),
            "last_tok_rms": round(last_tok_rms, 5),
            "max_abs_diff_vs_official": max_abs_diff,
            "cos_sim_vs_official": round(cos_sim, 8),
            "layer_ms": round(layer_ms, 3),
            "substage": substage_stats,
        })

    h_final = rms_norm_ref(h, model.model.embedding_norm.weight, eps=config.norm_eps)
    final_norm_rms = float(h_final.float().pow(2).mean().sqrt())

    tied_exact = bool(torch.equal(model.lm_head.weight, model.model.embed_tokens.weight))
    logits = F.linear(h_final, model.model.embed_tokens.weight)

    off_logits = official_out.logits
    logits_max_diff = float((logits.float() - off_logits.float()).abs().max())
    logits_cos = float(
        F.cosine_similarity(logits.float().reshape(1, -1), off_logits.float().reshape(1, -1), dim=-1).item()
    )

    last_probs = F.softmax(logits[0, -1].float(), dim=-1)
    top_probs, top_ids = torch.topk(last_probs, k=5)

    return {
        "seq_len": seq_len,
        "tokens": tokens_str,
        "official_forward_ms": round(t_official_ms, 3),
        "embed_ms": round(embed_ms, 3),
        "embed_rms_per_pos": embed_rms_per_pos,
        "rope_full_max_diff": rope_full_max_diff,
        "rope_sliding_max_diff": rope_sliding_max_diff,
        "tied_weights_exact": tied_exact,
        "final_norm_rms": round(final_norm_rms, 5),
        "logits_max_abs_diff_vs_official": logits_max_diff,
        "logits_cos_sim_vs_official": round(logits_cos, 8),
        "top5_next_token_ids": top_ids.tolist(),
        "top5_next_token_probs": [round(float(p), 5) for p in top_probs],
        "layers": layer_records,
    }


@torch.no_grad()
def verify_causal_prefix_invariance(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    prefix_len = input_ids.shape[1] // 2
    full_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    pref_out = model(input_ids=input_ids[:, :prefix_len], output_hidden_states=True, use_cache=False)

    max_gap_across_layers = 0.0
    min_cos_across_layers = 1.0
    for h_full, h_pref in zip(full_out.hidden_states, pref_out.hidden_states):
        slice_full = h_full[:, :prefix_len, :].float()
        slice_pref = h_pref.float()
        gap = float((slice_full - slice_pref).abs().max())
        cos = float(F.cosine_similarity(slice_full.reshape(1, -1), slice_pref.reshape(1, -1), dim=-1).item())
        max_gap_across_layers = max(max_gap_across_layers, gap)
        min_cos_across_layers = min(min_cos_across_layers, cos)

    return {
        "full_len": int(input_ids.shape[1]),
        "prefix_len": int(prefix_len),
        "max_gap_across_all_stages": max_gap_across_layers,
        "min_cos_across_all_stages": round(min_cos_across_layers, 8),
    }


def compute_state_memory_comparison() -> dict[str, Any]:
    conv_total_bytes = 13 * 2048 * 6 * 2
    comparisons = {}
    for seq_len in [1024, 4096, 32768, 131072, 262144]:
        slid_tokens = min(seq_len, 256)
        slid_bytes = 5 * slid_tokens * 4 * 128 * 2 * 2
        full_bytes = 6 * seq_len * 4 * 128 * 2 * 2
        lfm3_total_mb = (conv_total_bytes + slid_bytes + full_bytes) / (1024 * 1024)
        all_full_24_mb = (24 * seq_len * 4 * 128 * 2 * 2) / (1024 * 1024)
        comparisons[str(seq_len)] = {
            "lfm3_state_mb": round(lfm3_total_mb, 3),
            "standard_24_layer_gqa_mb": round(all_full_24_mb, 3),
            "reduction_factor": round(all_full_24_mb / lfm3_total_mb, 2),
        }
    return {
        "conv_13_layers_fixed_kb": round(conv_total_bytes / 1024, 2),
        "sliding_5_layers_max_mb": round((5 * 256 * 4 * 128 * 2 * 2) / (1024 * 1024), 2),
        "by_context_length": comparisons,
    }


def main():
    torch.set_num_threads(8)
    st_summary = inspect_safetensors(MODEL_DIR)
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    load_s = time.perf_counter() - t_load0

    probe_text = "Clover verifies every stage of a model on real inputs before drawing architectural conclusions."
    encoded = tokenizer(probe_text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    stage_verification = run_manual_stage_by_stage_verification(model, input_ids, tokens_str)
    stage_verification["top5_next_tokens_decoded"] = [
        tokenizer.decode([tid]) for tid in stage_verification["top5_next_token_ids"]
    ]
    causal_check = verify_causal_prefix_invariance(model, input_ids)

    report = {
        "model_id": "LiquidAI/LFM3-1.7B",
        "load_seconds": round(load_s, 3),
        "safetensors_summary": st_summary,
        "state_memory_analysis": compute_state_memory_comparison(),
        "stage_by_stage_verification": stage_verification,
        "causal_prefix_invariance": causal_check,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
