#!/usr/bin/env python3
"""
GPU-Resident Differentiable Stage Backbone for NVIDIA Nemotron 3.5 Super VL
(`nemotron_gpu_backbone.py`).

Loads the exact precomputed Clover containers from `/dev/shm/clover-train/nemotron/pruned/`:
  - `global-0/embeddings.bf16.bin` [131072, 4096]
  - `trunk-0/` (Mamba-2 SSM layer, d=4096, d_inner=8192, n_heads=128, d_head=64, d_state=128)
  - `trunk-1/` + `expert-1/experts.direct` (Latent ReLU^2 MoE layer: 512 routed experts 1024->2688->1024,
    top-22 sigmoid routing with group mask, fc1 4096->1024, fc2 1024->4096, shared_experts 4096->5376->4096)
  - `trunk-7/` (NoPE 16:1 GQA Attention layer: 32 Q heads, 2 KV heads, head_dim=128)
  - `global-0/norm_f.bf16.bin` [4096]
  - `global-0/lm_head.bf16.bin` [131072, 4096]

Provides a HuggingFace-compatible `CausalLMOutputWithPast` interface (`input_ids`, `inputs_embeds`,
`attention_mask`, `labels`, `output_hidden_states=True`, `get_input_embeddings()`) so that both
3-stage `HybridSFT` (`HybridSFTWrapper` + `HybridLink(d=4096)`) and 11-level `NCM` training run natively
across 8x RTX 3090 GPUs!
"""

from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.nemotron_h.modeling_nemotron_h import (
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemotron_precomputed_engine import (
    DTYPE_BF16,
    DTYPE_FP32,
    OPERATOR_MAGIC,
    NemotronPrunedTrunkReader,
    bytes_to_tensor,
    gated_rms_norm_op,
    relu2_op,
    rms_norm_op,
)


@dataclass
class NemotronStageConfig:
    vocab_size: int = 131072
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 11
    pad_token_id: int = 11
    think_start_id: int = 12
    think_end_id: int = 13
    latent_start_id: int = 30
    latent_end_id: int = 31


class NemotronGPUMambaStage(nn.Module):
    """GPU-resident differentiable Mamba-2 SSM stage (`trunk-0`)."""

    def __init__(self, trunk_dir: Path):
        super().__init__()
        trunk = NemotronPrunedTrunkReader(trunk_dir)
        op_buf = trunk.operator_bytes
        assert op_buf is not None and op_buf[:8] == OPERATOR_MAGIC
        _, kind, layer_id, len_norm, len_in, len_cw, len_cb, len_dtb = struct.unpack_from("<8sIIIIIII", op_buf, 0)
        len_A, len_D = struct.unpack_from("<II", op_buf, 36)
        assert kind == 0

        mv = memoryview(op_buf)
        cur = 64
        norm_w = bytes_to_tensor(mv[cur : cur + len_norm], DTYPE_BF16, (4096,)); cur += len_norm
        in_proj_w = bytes_to_tensor(mv[cur : cur + len_in], DTYPE_BF16, (18560, 4096)); cur += len_in
        conv_w = bytes_to_tensor(mv[cur : cur + len_cw], DTYPE_BF16, (10240, 1, 4)); cur += len_cw
        conv_b = bytes_to_tensor(mv[cur : cur + len_cb], DTYPE_BF16, (10240,)); cur += len_cb
        dt_bias = bytes_to_tensor(mv[cur : cur + len_dtb], DTYPE_BF16, (128,)); cur += len_dtb
        A_precomp = bytes_to_tensor(mv[cur : cur + len_A], DTYPE_FP32, (128,)); cur += len_A
        D_vec = bytes_to_tensor(mv[cur : cur + len_D], DTYPE_BF16, (128,)); cur += len_D

        mixer_norm_w = trunk.get_record_tensor(1, (8192,))
        out_proj_w = trunk.get_record_tensor(2, (4096, 8192))

        self.register_buffer("norm_w", norm_w, persistent=False)
        self.register_buffer("in_proj_w", in_proj_w, persistent=False)
        self.register_buffer("conv_w", conv_w, persistent=False)
        self.register_buffer("conv_b", conv_b, persistent=False)
        self.register_buffer("dt_bias", dt_bias, persistent=False)
        self.register_buffer("A_precomp", A_precomp, persistent=False)
        self.register_buffer("D_vec", D_vec, persistent=False)
        self.register_buffer("mixer_norm_w", mixer_norm_w, persistent=False)
        self.register_buffer("out_proj_w", out_proj_w, persistent=False)

    def forward(self, hidden_states: torch.Tensor, chunk_size: int = 64, eps: float = 1e-5) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        u = rms_norm_op(hidden_states, self.norm_w, eps=eps)
        proj = F.linear(u, self.in_proj_w)
        gate, conv_in, dt = torch.split(proj, [8192, 10240, 128], dim=-1)

        conv_T = conv_in.transpose(1, 2)
        conv_raw = F.conv1d(conv_T, self.conv_w, bias=self.conv_b, padding=3, groups=10240)[..., :seq_len]
        conv_act = F.silu(conv_raw).transpose(1, 2)
        x_ssm, B_ssm, C_ssm = torch.split(conv_act, [8192, 1024, 1024], dim=-1)

        dt_f = torch.clamp(F.softplus(dt.float() + self.dt_bias.float()), min=0.001)
        x_h = x_ssm.reshape(bsz, seq_len, 128, 64).float()
        B_h = B_ssm.reshape(bsz, seq_len, 8, 128).float().repeat_interleave(16, dim=2, output_size=128)
        C_h = C_ssm.reshape(bsz, seq_len, 8, 128).float().repeat_interleave(16, dim=2, output_size=128)
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        D_residual = self.D_vec.float()[..., None] * pad_tensor_by_size(x_h, pad_size)

        x_disc = x_h * dt_f[..., None]
        A_disc = self.A_precomp.to(x_disc.dtype) * dt_f
        x_c, A_c, B_c, C_c = [reshape_into_chunks(t, pad_size, chunk_size) for t in (x_disc, A_disc, B_h, C_h)]
        A_c = A_c.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A_c, dim=-1)
        L = torch.exp(segment_sum(A_c))
        G = (C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]).sum(dim=-1)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * x_c[:, :, None]).sum(3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B_c * decay_states.permute(0, 2, 3, 1)[..., None]
        states = (
            (B_decay.permute(0, 1, 3, 2, 4)[..., None] * x_c.permute(0, 1, 3, 2, 4)[..., None, :])
            .sum(dim=3)
            .permute(0, 1, 2, 4, 3)
        )
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = (
            (decay_chunk[..., None, None] * states.permute(0, 2, 1, 3, 4)[:, :, None, ...])
            .sum(dim=2)
            .permute(0, 2, 1, 3, 4)
        )
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum).permute(0, 2, 3, 1)
        Y_off = (C_c[..., None, :] * states[:, :, None, ...]).sum(-1) * state_decay_out[..., None]
        y = (Y_diag + Y_off).reshape(bsz, -1, 128, 64) + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(bsz, seq_len, 8192)
        scan_out = gated_rms_norm_op(y, gate, self.mixer_norm_w, group_size=1024, eps=eps)
        out = F.linear(scan_out.to(hidden_states.dtype), self.out_proj_w)
        return hidden_states + out


class NemotronGPULatentMoEStage(nn.Module):
    """
    GPU-resident differentiable Latent ReLU^2 MoE stage (`trunk-1` + `expert-1/experts.direct`).
    Keeps all 512 routed experts directly in GPU VRAM (`5.25 GiB` BF16 out of `24 GiB` RTX 3090)
    or shared `/dev/shm` RAM, and executes top-22 routed experts + shared experts on GPU!
    """

    def __init__(self, trunk_dir: Path, expert_dir: Path, experts_on_gpu: bool = True):
        super().__init__()
        trunk = NemotronPrunedTrunkReader(trunk_dir)
        self.register_buffer("norm_w", trunk.get_record_tensor(0, (4096,)), persistent=False)
        self.register_buffer("gate_w_fp32", trunk.get_record_tensor(29, (512, 4096)), persistent=False)
        self.register_buffer("gate_bias_fp32", trunk.get_record_tensor(30, (512,)), persistent=False)
        self.register_buffer("fc1_w", trunk.get_record_tensor(31, (1024, 4096)), persistent=False)
        self.register_buffer("fc2_w", trunk.get_record_tensor(32, (4096, 1024)), persistent=False)
        self.register_buffer("sh_up_w", trunk.get_record_tensor(34, (5376, 4096)), persistent=False)
        self.register_buffer("sh_down_w", trunk.get_record_tensor(36, (4096, 5376)), persistent=False)

        exp_path = expert_dir / "experts.direct"
        raw_np = np.memmap(exp_path, dtype=np.uint16, mode="r", shape=(512, 2 * 2688 * 1024))
        exp_all = torch.from_numpy(raw_np).view(torch.bfloat16)
        self.experts_on_gpu = experts_on_gpu
        if experts_on_gpu:
            self.register_buffer(
                "experts_up", exp_all[:, : 2688 * 1024].reshape(512, 2688, 1024).contiguous(), persistent=False
            )
            self.register_buffer(
                "experts_down", exp_all[:, 2688 * 1024 :].reshape(512, 1024, 2688).contiguous(), persistent=False
            )
        else:
            self.experts_up_cpu = exp_all[:, : 2688 * 1024].reshape(512, 2688, 1024)
            self.experts_down_cpu = exp_all[:, 2688 * 1024 :].reshape(512, 1024, 2688)

    def forward(
        self, hidden_states: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        u = rms_norm_op(hidden_states, self.norm_w, eps=eps)
        flat_u = u.reshape(-1, 4096)

        with torch.no_grad():
            router_logits = F.linear(flat_u.float(), self.gate_w_fp32)
            scores = torch.sigmoid(router_logits)
            scores_choice = scores + self.gate_bias_fp32
            group_scores = scores_choice.view(-1, 1, 512).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=1, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, 1, 512).reshape(-1, 512)
            scores_choice = scores_choice.masked_fill(~score_mask.bool(), float("-inf"))
            topk_indices = torch.topk(scores_choice, k=22, dim=-1, sorted=False)[1]
            topk_weights = scores.gather(1, topk_indices)
            topk_weights = (topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)) * 5.0

        latent_in = F.linear(flat_u, self.fc1_w)  # [N_tok, 1024]
        latent_accum = torch.zeros_like(latent_in, dtype=torch.float32)

        expert_mask = F.one_hot(topk_indices, num_classes=512).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for exp_tensor in expert_hit:
            e_id = int(exp_tensor.item())
            top_k_pos, tok_idx = torch.where(expert_mask[e_id])
            if self.experts_on_gpu:
                up_w = self.experts_up[e_id]
                down_w = self.experts_down[e_id]
            else:
                up_w = self.experts_up_cpu[e_id].to(hidden_states.device, non_blocking=True)
                down_w = self.experts_down_cpu[e_id].to(hidden_states.device, non_blocking=True)
            cur_in = latent_in[tok_idx]
            cur_h = F.linear(relu2_op(F.linear(cur_in, up_w)), down_w)
            cur_h = cur_h.float() * topk_weights[tok_idx, top_k_pos, None]
            latent_accum.index_add_(0, tok_idx, cur_h)

        routed_4096 = F.linear(latent_accum.to(hidden_states.dtype), self.fc2_w).view(bsz, seq_len, 4096)
        shared_4096 = F.linear(relu2_op(F.linear(u, self.sh_up_w)), self.sh_down_w)
        out = hidden_states + (routed_4096 + shared_4096)
        return out, routed_4096


class NemotronGPUAttentionStage(nn.Module):
    """GPU-resident differentiable NoPE 16:1 GQA Attention stage (`trunk-7`)."""

    def __init__(self, trunk_dir: Path):
        super().__init__()
        trunk = NemotronPrunedTrunkReader(trunk_dir)
        op_buf = trunk.operator_bytes
        assert op_buf is not None and op_buf[:8] == OPERATOR_MAGIC
        _, kind, layer_id, len_norm, len_q, len_k, len_v = struct.unpack_from("<8sIIIIII", op_buf, 0)
        assert kind == 1

        mv = memoryview(op_buf)
        cur = 64
        norm_w = bytes_to_tensor(mv[cur : cur + len_norm], DTYPE_BF16, (4096,)); cur += len_norm
        q_w = bytes_to_tensor(mv[cur : cur + len_q], DTYPE_BF16, (4096, 4096)); cur += len_q
        k_w = bytes_to_tensor(mv[cur : cur + len_k], DTYPE_BF16, (256, 4096)); cur += len_k
        v_w = bytes_to_tensor(mv[cur : cur + len_v], DTYPE_BF16, (256, 4096)); cur += len_v
        o_w = trunk.get_record_tensor(1, (4096, 4096))

        self.register_buffer("norm_w", norm_w, persistent=False)
        self.register_buffer("q_w", q_w, persistent=False)
        self.register_buffer("k_w", k_w, persistent=False)
        self.register_buffer("v_w", v_w, persistent=False)
        self.register_buffer("o_w", o_w, persistent=False)

    def forward(self, hidden_states: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        u = rms_norm_op(hidden_states, self.norm_w, eps=eps)
        q = F.linear(u, self.q_w).view(bsz, seq_len, 32, 128).transpose(1, 2)
        k = F.linear(u, self.k_w).view(bsz, seq_len, 2, 128).transpose(1, 2)
        v = F.linear(u, self.v_w).view(bsz, seq_len, 2, 128).transpose(1, 2)

        k_rep = k[:, :, None, :, :].expand(bsz, 2, 16, seq_len, 128).reshape(bsz, 32, seq_len, 128)
        v_rep = v[:, :, None, :, :].expand(bsz, 2, 16, seq_len, 128).reshape(bsz, 32, seq_len, 128)

        ctx = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, 4096)
        return hidden_states + F.linear(ctx, self.o_w)


class NemotronGPUStageModel(nn.Module):
    """
    Complete GPU-resident Nemotron 3.5 Super VL Stage Model (`d=4096`, `V=131072`)
    compatible with `HybridSFTWrapper` (`HybridLink(d=4096)`) and 11-level `NCM` training.
    """

    def __init__(
        self,
        pruned_root: Path = Path("/dev/shm/clover-train/nemotron/pruned"),
        experts_on_gpu: bool = True,
        init_latent_special_tokens: bool = True,
    ):
        super().__init__()
        self.config = NemotronStageConfig()
        global_dir = pruned_root / "global-0"

        emb_raw = np.fromfile(global_dir / "embeddings.bf16.bin", dtype=np.uint16)
        emb_t = torch.from_numpy(emb_raw).view(torch.bfloat16).reshape(131072, 4096).clone()

        lm_raw = np.fromfile(global_dir / "lm_head.bf16.bin", dtype=np.uint16)
        lm_t = torch.from_numpy(lm_raw).view(torch.bfloat16).reshape(131072, 4096).clone()

        norm_f_raw = np.fromfile(global_dir / "norm_f.bf16.bin", dtype=np.uint16)
        norm_f_t = torch.from_numpy(norm_f_raw).view(torch.bfloat16).reshape(4096).clone()

        if init_latent_special_tokens:
            with torch.no_grad():
                lat_start_src = [12, 1060, 12902, 1297, 1062]
                lat_end_src = [13, 1885, 12902, 1297, 1062]
                emb_t[30] = emb_t[lat_start_src].float().mean(dim=0).to(torch.bfloat16)
                emb_t[31] = emb_t[lat_end_src].float().mean(dim=0).to(torch.bfloat16)
                lm_t[30] = lm_t[lat_start_src].float().mean(dim=0).to(torch.bfloat16)
                lm_t[31] = lm_t[lat_end_src].float().mean(dim=0).to(torch.bfloat16)

        self.embed_tokens = nn.Embedding(131072, 4096)
        self.embed_tokens.weight = nn.Parameter(emb_t, requires_grad=False)

        self.mamba_stage = NemotronGPUMambaStage(pruned_root / "trunk-0")
        self.moe_stage = NemotronGPULatentMoEStage(
            pruned_root / "trunk-1", pruned_root / "expert-1", experts_on_gpu=experts_on_gpu
        )
        self.attn_stage = NemotronGPUAttentionStage(pruned_root / "trunk-7")

        self.norm_f = nn.Parameter(norm_f_t, requires_grad=False)
        self.lm_head = nn.Linear(4096, 131072, bias=False)
        self.lm_head.weight = nn.Parameter(lm_t, requires_grad=False)

        self.last_moe_routed: Optional[torch.Tensor] = None
        self.last_moe_out: Optional[torch.Tensor] = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        h0 = inputs_embeds.to(torch.bfloat16)
        h1 = self.mamba_stage(h0, eps=self.config.rms_norm_eps)
        h2, routed_branch = self.moe_stage(h1, eps=self.config.rms_norm_eps)
        self.last_moe_routed = routed_branch
        self.last_moe_out = h2
        h3 = self.attn_stage(h2, eps=self.config.rms_norm_eps)
        h_final = rms_norm_op(h3, self.norm_f, eps=self.config.rms_norm_eps)

        logits = self.lm_head(h_final)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        hidden_states = (h0, h1, h2, h3, h_final) if output_hidden_states else None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    device = torch.device("cuda:0")
    model = NemotronGPUStageModel(experts_on_gpu=True).to(device)
    load_s = time.perf_counter() - t0
    vram_gb = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"[NemotronGPUStageModel] Loaded on {device} in {load_s:.2f}s | VRAM allocated: {vram_gb:.2f} GiB")

    ids = torch.randint(100, 5000, (2, 128), device=device)
    embeds = model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
    t1 = time.perf_counter()
    out = model(inputs_embeds=embeds, labels=ids, output_hidden_states=True)
    out.loss.backward()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t1) * 1000.0
    print(
        f"[NemotronGPUStageModel] B=2, T=128 forward+backward: {step_ms:.1f} ms | "
        f"loss={out.loss.item():.4f} | grad_norm={embeds.grad.float().norm().item():.4f} | "
        f"h_final={out.hidden_states[-1].shape}"
    )
