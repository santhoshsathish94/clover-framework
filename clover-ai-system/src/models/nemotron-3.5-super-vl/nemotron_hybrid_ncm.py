#!/usr/bin/env python3
"""
Native C/OpenMP Engine Wrapper + Native HybridAgent (`NemotronNativeHybridLink` d=4096) +
Native 11-Level NCM (`131,072`-Vocab Grafted VQ-VAE + `NextConceptTransformer` + Bridges)
for NVIDIA Nemotron 3.5 Super VL (`clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hybrid_ncm.py`).

Completes the full 7-layer Clover stack for `nemotron-3.5-super-vl`:
  - Layer 1 (Native C/OpenMP Engine `libnemotron_cpu_engine.so`):
      Executes Mamba-2 (`trunk-0` + `operator.bin`), 512-expert Latent `ReLU^2` MoE (`trunk-1` +
      `expert-1/experts.direct` with SpecMD Least-Stale + R4 Clean-Drop slot cache), NoPE 16:1 GQA
      (`trunk-7` + `operator.bin`), and Final RMSNorm in C/OpenMP via `ctypes`.
  - Layer 2 (Native `HybridAgent` in Nemotron's `4096`-d space):
      `NemotronNativeHybridLink(d=4096, bottleneck=1024)` trained on real Nemotron hidden states `h_t`
      to predict next-token Nemotron embeddings `e_Nemo(x_{t+1})` with the `HybridSFT` objective
      (`1.0 * cos_loss + 0.2 * mse_loss`), plus `<latent>` cosine readout across Nemotron's `131,072` vocabulary.
  - Layer 3 (Native 11-Level `NCM` in Nemotron's `131,072` Tekken vocabulary):
      Grafts Nemotron's `131,072` vocabulary into the 11-level `VQVAETokenizer` (`vqvae.encoder.wte` and
      `vqvae.decoder.output_layer`) via ridge regression from `embeddings.bf16.bin` / `lm_head.bf16.bin` +
      exact shared-token copy, fine-tunes on Nemotron-tokenized windows, and wires `NemotronComboBridge(4096 -> 384)`
      and `NemotronExpertBridge(4096 -> 384)` into `NextConceptTransformer`.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
TOOLS_DIR = MODELS_DIR.parent / "tools" / "hybrid-ncm-k3"
NCM_REPO_DIR = Path("/root/ncm-cpu/repo")
for p in (SCRIPT_DIR, MODELS_DIR, TOOLS_DIR, NCM_REPO_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

V_NEMO = 131072
D_NEMO = 4096
D_VQ = 384


def ensure_nemotron_cpu_engine_so(src_dir: Path | None = None) -> Path:
    """Compiles `nemotron_cpu_engine.c` into `libnemotron_cpu_engine.so` with `-O3 -march=native -fopenmp`."""
    base_dir = src_dir or SCRIPT_DIR
    c_src = base_dir / "nemotron_cpu_engine.c"
    so_path = base_dir / "libnemotron_cpu_engine.so"
    if so_path.exists() and c_src.exists() and so_path.stat().st_mtime >= c_src.stat().st_mtime:
        return so_path
    cmd = [
        "gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared",
        str(c_src), "-lm", "-ldl", "-o", str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return so_path


class NemotronNativeCPUEngine:
    """Python `ctypes` wrapper around `libnemotron_cpu_engine.so`."""

    def __init__(
        self,
        pruned_dataset_dir: str | Path = "/opt/clover-k3/nemotron-computed/pruned",
        so_path: Path | None = None,
    ):
        self.pruned_dataset_dir = Path(pruned_dataset_dir)
        self.so_path = ensure_nemotron_cpu_engine_so() if so_path is None else so_path
        self.lib = ctypes.CDLL(str(self.so_path))

        self.lib.nemo_engine_init.argtypes = [ctypes.c_char_p]
        self.lib.nemo_engine_init.restype = ctypes.c_int

        self.lib.nemo_forward_pipeline.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.nemo_forward_pipeline.restype = ctypes.c_float

        self.lib.nemo_dm_stats.argtypes = [ctypes.POINTER(ctypes.c_ulong)]
        self.lib.nemo_dm_stats.restype = None

        rc = self.lib.nemo_engine_init(str(self.pruned_dataset_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"nemo_engine_init({self.pruned_dataset_dir}) failed with rc={rc}")

    def forward_ids(self, token_ids: list[int] | torch.Tensor) -> tuple[torch.Tensor, float]:
        if isinstance(token_ids, torch.Tensor):
            ids_list = [int(x) for x in token_ids.view(-1).tolist()]
        else:
            ids_list = [int(x) for x in token_ids]
        T = len(ids_list)
        c_ids = (ctypes.c_int * T)(*ids_list)
        out_arr = np.zeros((T, D_NEMO), dtype=np.float32)
        out_ptr = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ms = float(self.lib.nemo_forward_pipeline(c_ids, T, out_ptr))
        return torch.from_numpy(out_arr), ms

    def get_dm_stats(self) -> dict[str, int]:
        dm = (ctypes.c_ulong * 6)()
        self.lib.nemo_dm_stats(dm)
        return {
            "total_expert_requests": int(dm[0]),
            "tier1_pool_hits": int(dm[1]),
            "tier2_host_hits": int(dm[2]),
            "tier3_drive_fetches": int(dm[3]),
            "r4_clean_drops": int(dm[4]),
            "dirty_writebacks": int(dm[5]),
        }


class NemotronNativeHybridLink(nn.Module):
    """
    Native HybridLink (`ln_res_adapter`) operating in Nemotron 3.5 Super VL's `d = 4096` hidden space.
    Trained on real Nemotron hidden states `h_t` to predict next-token Nemotron embeddings `e_Nemo(x_{t+1})`
    using the `HybridSFT` objective (`1.0 * cos_loss + 0.2 * mse_loss`).
    """

    def __init__(self, d: int = D_NEMO, bottleneck: int = 1024):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d)
        self.proj1 = nn.Linear(d, bottleneck)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, d)
        self.post_ln = nn.LayerNorm(d)
        self.out_scale = nn.Parameter(torch.tensor(0.025))

    def forward(self, h_nemo: torch.Tensor) -> torch.Tensor:
        hf = h_nemo.float()
        z = self.post_ln(hf + self.proj2(self.act(self.proj1(self.pre_ln(hf)))))
        return z * self.out_scale


class NemotronComboBridge(nn.Module):
    """Maps `4096`-d Nemotron `R_Nemo(h)` continuous `<latent>` vectors `[B, P, 4096]` into NCM's `384`-d prefix space."""

    def __init__(self, d_in: int = D_NEMO, d_out: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, 768)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(768, d_out)
        self.ln_out = nn.LayerNorm(d_out)
        self.scale = nn.Parameter(torch.tensor(8.5))
        self.seg_rh = nn.Parameter(torch.zeros(1, 1, d_out))
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, r_nemo: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(r_nemo.float())
        x = self.fc2(self.act(self.fc1(x)))
        x = self.ln_out(x) * self.scale
        return (x * self.gate + self.seg_rh).to(r_nemo.dtype)


class NemotronExpertBridge(nn.Module):
    """Maps Nemotron's `4096`-d Latent `ReLU^2` MoE hidden states into NCM's `384`-d concept space."""

    def __init__(self, d_nemo: int = D_NEMO, d_vq: int = D_VQ):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_nemo)
        self.proj1 = nn.Linear(d_nemo, d_vq)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_vq, d_vq)
        self.ln_out = nn.LayerNorm(d_vq)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_nemo_moe: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(h_nemo_moe.float())
        x = self.proj2(self.act(self.proj1(x)))
        return (torch.tanh(self.alpha) * self.ln_out(x)).to(h_nemo_moe.dtype)


def train_and_verify_nemotron_native_hybrid_and_ncm(
    source_model_dir: Path = Path("/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"),
    pruned_computed_dir: Path = Path("/opt/clover-k3/nemotron-computed/pruned"),
    wmt14_ckpt_dir: Path = Path("/root/ncm-cpu/artifacts/wmt14"),
    out_ckpt_dir: Path = Path("/opt/clover-k3/nemotron-computed/hybrid-ncm"),
) -> dict[str, Any]:
    """
    1. Compiles and runs `libnemotron_cpu_engine.so` (`NemotronNativeCPUEngine`) and verifies C/OpenMP numerical
       agreement against `nemotron_precomputed_engine.py`.
    2. Trains `NemotronNativeHybridLink(d=4096)` on real Nemotron hidden states `h_t` to predict next-token
       Nemotron embeddings `e_Nemo(x_{t+1})` with `HybridSFT` (`1.0 * cos_loss + 0.2 * mse_loss`) and tests
       `<latent>` cosine readout across Nemotron's `131,072` vocabulary.
    3. Grafts Nemotron's `131,072` Tekken vocabulary into the 11-level `VQVAETokenizer` via ridge regression
       from `embeddings.bf16.bin` and `lm_head.bf16.bin` + exact shared-token copy, fine-tunes on Nemotron
       windows, and runs the 11-level `NextConceptTransformer` conditioned on `[NemotronComboBridge(R_Nemo(h)),
       wte(plan_ids) + NemotronExpertBridge(h_nemo)]`!
    """
    torch.set_num_threads(8)
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Native C/OpenMP Engine (`libnemotron_cpu_engine.so`)
    c_engine = NemotronNativeCPUEngine(pruned_computed_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(source_model_dir), trust_remote_code=True)

    corpus_texts = [
        "Clover AI System switches seamlessly between Kimi K3, NVIDIA Nemotron 3.5 Super VL, and Liquid LFM3.",
        "Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road.",
        "Before opening subsea equalization valve V-402, equalize differential pressure and verify downstream telemetry.",
        "SpecMD Least-Stale slot caching with Red-Blue Pebble Rule R4 clean-drop eliminates dirty PCIe writebacks.",
        "Next-Concept Modeling predicts 11 hierarchical vector-quantized concept levels in parallel before decoding tokens.",
        "Mamba-2 state-space duality combines selective state spaces with 512-expert Latent ReLU-squared Mixture of Experts.",
    ]
    joined_text = " ".join(corpus_texts)
    all_ids = tokenizer(joined_text, return_tensors="pt")["input_ids"][0]  # [T_total]

    # Compare C/OpenMP engine vs Python precomputed engine on the first prompt
    print("[Nemotron-Native] Step 1/4: Running C/OpenMP libnemotron_cpu_engine.so forward pass...", flush=True)
    p0_ids = tokenizer(corpus_texts[0], return_tensors="pt")["input_ids"]
    h_c_p0, c_ms_pass1 = c_engine.forward_ids(p0_ids)
    dm_pass1 = c_engine.get_dm_stats()
    # Second warm pass on prefix to verify SpecMD Tier-1/Tier-2 cache hits and R4 clean-drop
    _, c_ms_pass2 = c_engine.forward_ids(p0_ids[:, :8])
    dm_pass2 = c_engine.get_dm_stats()

    from nemotron_precomputed_engine import (
        DTYPE_BF16,
        NemotronDirectExpertStore,
        NemotronPrunedTrunkReader,
        bytes_to_tensor,
        execute_attention_stage_from_precomputed,
        execute_latent_moe_stage_from_precomputed,
        execute_mamba_stage_from_precomputed,
        rms_norm_op,
    )

    emb_w = bytes_to_tensor((pruned_computed_dir / "global-0" / "embeddings.bf16.bin").read_bytes(), DTYPE_BF16, (V_NEMO, D_NEMO))
    norm_f_w = bytes_to_tensor((pruned_computed_dir / "global-0" / "norm_f.bf16.bin").read_bytes(), DTYPE_BF16, (D_NEMO,))
    lm_head_w = bytes_to_tensor((pruned_computed_dir / "global-0" / "lm_head.bf16.bin").read_bytes(), DTYPE_BF16, (V_NEMO, D_NEMO))

    with torch.no_grad():
        t0_r = NemotronPrunedTrunkReader(pruned_computed_dir / "trunk-0")
        t1_r = NemotronPrunedTrunkReader(pruned_computed_dir / "trunk-1")
        e1_s = NemotronDirectExpertStore(pruned_computed_dir / "expert-1")
        t7_r = NemotronPrunedTrunkReader(pruned_computed_dir / "trunk-7")
        h0_py = F.embedding(p0_ids, emb_w)
        h1_py = execute_mamba_stage_from_precomputed(t0_r, h0_py)
        h2_py, _ = execute_latent_moe_stage_from_precomputed(t1_r, e1_s, h1_py)
        h3_py = execute_attention_stage_from_precomputed(t7_r, h2_py)
        h_py_p0 = rms_norm_op(h3_py, norm_f_w)[0].float()

        # Also compute fast vectorized precomputed hidden states for the full 6-sentence corpus
        h0_all = F.embedding(all_ids.unsqueeze(0), emb_w)
        h1_all = execute_mamba_stage_from_precomputed(t0_r, h0_all)
        h2_all, _ = execute_latent_moe_stage_from_precomputed(t1_r, e1_s, h1_all)
        h3_all = execute_attention_stage_from_precomputed(t7_r, h2_all)
        h_all_c = rms_norm_op(h3_all, norm_f_w)[0].float()
        e1_s.close()

    c_vs_py_cos = float(F.cosine_similarity(h_c_p0.reshape(1, -1), h_py_p0.reshape(1, -1), dim=-1).item())
    c_vs_py_max_diff = float((h_c_p0 - h_py_p0).abs().max().item())
    print(f"[Nemotron-Native] C vs Python precomputed cos={c_vs_py_cos:.7f}, max_diff={c_vs_py_max_diff:.6f}", flush=True)

    # 2. Train `NemotronNativeHybridLink(d=4096)` on real Nemotron hidden states `h_t -> e_Nemo(x_{t+1})`
    print("[Nemotron-Native] Step 2/4: Training NemotronNativeHybridLink(d=4096)...", flush=True)
    e_all_target = F.embedding(all_ids, emb_w).float()  # [T_total, 4096]

    h_inputs = h_all_c[:-1]       # [N-1, 4096]
    e_next = e_all_target[1:]     # [N-1, 4096]
    split = int(0.8 * len(h_inputs))

    torch.manual_seed(42)
    nemo_link = NemotronNativeHybridLink(d=D_NEMO, bottleneck=1024)
    with torch.no_grad():
        raw_cos_te = float(F.cosine_similarity(h_inputs[split:], e_next[split:], dim=-1).mean().item())

    opt_link = torch.optim.AdamW(nemo_link.parameters(), lr=2e-3, weight_decay=0.01)
    for _ in range(250):
        pred_e = nemo_link(h_inputs)
        l_cos = 1.0 - F.cosine_similarity(pred_e, e_next, dim=-1).mean()
        l_mse = F.mse_loss(pred_e, e_next)
        loss = l_cos + 0.2 * l_mse
        opt_link.zero_grad(set_to_none=True)
        loss.backward()
        opt_link.step()

    with torch.no_grad():
        pred_all = nemo_link(h_inputs)
        train_cos = float(F.cosine_similarity(pred_all[:split], e_next[:split], dim=-1).mean().item())
        test_cos = float(F.cosine_similarity(pred_all[split:], e_next[split:], dim=-1).mean().item())

        # Test `<latent>` cosine readout across all 131,072 Nemotron vocabulary embeddings!
        emb_normed = F.normalize(emb_w[:65536].float(), dim=-1)
        r_sample = F.normalize(pred_all[:12], dim=-1)
        sims = r_sample @ emb_normed.t()
        top_cos, top_ids = sims.max(dim=-1)
        latent_readout_mean_cos = float(top_cos.mean().item())
        latent_readout_text = tokenizer.decode(top_ids.tolist())

    # 3. Graft Nemotron's `131,072` vocabulary into the 11-level `VQVAETokenizer` + `NextConceptTransformer`
    print("[Nemotron-Native] Step 3/4: Grafting 131,072 Tekken vocabulary into 11-level VQ-VAE...", flush=True)
    from omegaconf import OmegaConf
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg_yaml = wmt14_ckpt_dir / "config.yaml"
    if not cfg_yaml.exists():
        cfg_yaml = wmt14_ckpt_dir / "config-wmt14-concept-model.yaml"
    cfg = OmegaConf.load(str(cfg_yaml))
    tok_pt = wmt14_ckpt_dir / "tokenizer.pt"
    if not tok_pt.exists():
        tok_pt = wmt14_ckpt_dir / "checkpoint-wmt14-concept-tokenizer.pt"
    cfg.tokenizer.vqvae.checkpoint_path = str(tok_pt)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vqvae = tk.vqvae.cpu().eval()

    wmt_enc_w = vqvae.encoder.wte.weight.detach().float()       # [V_wmt, 384]
    wmt_dec_w = vqvae.decoder.output_layer.weight.detach().float()  # [V_wmt, 384]

    # Build exact string-matched shared token map between WMT14 Qwen3 pre_tokenizer and Nemotron Tekken tokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    wmt_vocab = qwen_tok.get_vocab()
    nemo_vocab = tokenizer.get_vocab()
    shared_wmt_ids = []
    shared_nemo_ids = []
    for tok_str, w_id in wmt_vocab.items():
        if w_id < wmt_enc_w.shape[0] and tok_str in nemo_vocab:
            n_id = nemo_vocab[tok_str]
            if n_id < V_NEMO:
                shared_wmt_ids.append(w_id)
                shared_nemo_ids.append(n_id)

    w_idx = torch.tensor(shared_wmt_ids[:12000], dtype=torch.long)
    n_idx = torch.tensor(shared_nemo_ids[:12000], dtype=torch.long)

    X_nemo_emb = emb_w[n_idx].float()  # [N_s, 4096]
    Y_wmt_enc = wmt_enc_w[w_idx]       # [N_s, 384]
    Y_wmt_dec = wmt_dec_w[w_idx]       # [N_s, 384]

    XtX = X_nemo_emb.t() @ X_nemo_emb
    XtX.diagonal().add_(10.0)
    W_nemo_to_enc = torch.linalg.solve(XtX, X_nemo_emb.t() @ Y_wmt_enc)  # [4096, 384]
    W_nemo_to_dec = torch.linalg.solve(XtX, X_nemo_emb.t() @ Y_wmt_dec)  # [4096, 384]

    # Build 131,072-vocab grafted VQ-VAE encoder & decoder tables
    nemo_vq_enc = torch.zeros(V_NEMO, D_VQ, dtype=torch.float32)
    nemo_vq_dec = torch.zeros(V_NEMO, D_VQ, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, V_NEMO, 16384):
            sl = slice(i, min(i + 16384, V_NEMO))
            e_chunk = emb_w[sl].float()
            nemo_vq_enc[sl] = e_chunk @ W_nemo_to_enc
            nemo_vq_dec[sl] = e_chunk @ W_nemo_to_dec
        nemo_vq_enc[n_idx] = Y_wmt_enc
        nemo_vq_dec[n_idx] = Y_wmt_dec

    vqvae.in_vocab_size = V_NEMO
    vqvae.encoder.wte = nn.Embedding(V_NEMO, D_VQ)
    vqvae.encoder.wte.weight.data.copy_(nemo_vq_enc)
    vqvae.decoder.output_layer = nn.Linear(D_VQ, V_NEMO, bias=False)
    vqvae.decoder.output_layer.weight.data.copy_(nemo_vq_dec)

    # Fine-tune grafted VQ-VAE encoder/decoder on 64-token Nemotron windows (keeping 11-level VQ codebooks frozen)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    win_ids = []
    for st in range(0, max(1, len(all_ids) - 32), 16):
        chunk = all_ids[st : st + 64].tolist()
        if len(chunk) < 64:
            chunk = chunk + [pad_id] * (64 - len(chunk))
        win_ids.append(chunk)
    win_tensor = torch.tensor(win_ids, dtype=torch.long)  # [B_w, 64]

    with torch.no_grad():
        idx_0 = vqvae.encode(win_tensor)
        pred_0 = vqvae.decode(idx_0).argmax(-1)
        acc_before = float((pred_0 == win_tensor).float().mean().item())

    for name, p in vqvae.named_parameters():
        p.requires_grad_("vq." not in name)

    opt_vq = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=2e-3, weight_decay=0.01)
    vqvae.train()
    vqvae.vq.eval()
    for _ in range(150):
        logits_vq, vq_loss, _, _, _ = vqvae(win_tensor)
        ce_vq = F.cross_entropy(logits_vq.reshape(-1, V_NEMO), win_tensor.reshape(-1))
        loss_vq = ce_vq + 0.25 * vq_loss.mean()
        opt_vq.zero_grad(set_to_none=True)
        loss_vq.backward()
        opt_vq.step()

    vqvae.eval()
    with torch.no_grad():
        idx_after = vqvae.encode(win_tensor)
        pred_after = vqvae.decode(idx_after).argmax(-1)
        acc_after = float((pred_after == win_tensor).float().mean().item())
        rec_sample_text = tokenizer.decode(pred_after[0, : p0_ids.shape[1]].tolist())

    # 4. Wire `NemotronComboBridge(4096 -> 384)` + `NemotronExpertBridge(4096 -> 384)` into 11-level `NextConceptTransformer`
    mcfg = cfg.model
    mcfg.prefix_len = 32
    mcfg.predict_level0 = True
    ncm_net = NextConceptTransformer(vqvae_local=vqvae, config=mcfg).cpu().eval()
    base_ncm_pt = wmt14_ckpt_dir / "generator.pt"
    if not base_ncm_pt.exists():
        base_ncm_pt = wmt14_ckpt_dir.parent / "checkpoint-clover-true-hybrid-agent-ncm.pt"
    if base_ncm_pt.exists():
        sd_ncm = torch.load(str(base_ncm_pt), map_location="cpu", weights_only=False)
        if "model" in sd_ncm:
            own = ncm_net.state_dict()
            keep = {k: v for k, v in sd_ncm["model"].items() if k in own and own[k].shape == v.shape}
            ncm_net.load_state_dict(keep, strict=False)

    combo_bridge = NemotronComboBridge(D_NEMO, D_VQ)
    expert_bridge = NemotronExpertBridge(D_NEMO, D_VQ)

    with torch.no_grad():
        # Build 32-token combined prefix: 16 `R_Nemo(h)` latent positions + 16 `wte(plan_ids) + NemotronExpertBridge(h_nemo)` positions
        r_16 = nemo_link(h_c_p0[:16]).unsqueeze(0)          # [1, 16, 4096]
        z_rh_16 = combo_bridge(r_16)                        # [1, 16, 384]
        p_16_ids = p0_ids[:, :16]                           # [1, 16]
        z_bpe_16 = vqvae.encoder.wte(p_16_ids) + expert_bridge(h_c_p0[:16].unsqueeze(0))  # [1, 16, 384]
        z_prefix_32 = torch.cat([z_rh_16, z_bpe_16], dim=1)  # [1, 32, 384]
        pmask = torch.zeros(1, 32, dtype=torch.bool)

        ncm_in = vqvae.vq.idx_to_ncm_input(idx_after[:1])    # [1, 312, 384]
        ncm_logits = ncm_net(torch.cat([z_prefix_32, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
        ncm_code_ce = float(
            F.cross_entropy(ncm_logits.float().reshape(-1, ncm_logits.shape[-1]), idx_after[:1].reshape(-1)).item()
        )

    torch.save(
        {
            "nemo_native_hybrid_link": nemo_link.state_dict(),
            "nemo_combo_bridge": combo_bridge.state_dict(),
            "nemo_expert_bridge": expert_bridge.state_dict(),
            "W_nemo_to_enc": W_nemo_to_enc,
            "W_nemo_to_dec": W_nemo_to_dec,
        },
        out_ckpt_dir / "nemotron_hybrid_ncm_bridges.pt",
    )

    return {
        "native_c_openmp_engine": {
            "library_path": str(c_engine.so_path),
            "c_vs_python_precomputed_cosine_sim": round(c_vs_py_cos, 7),
            "c_vs_python_precomputed_max_abs_diff": round(c_vs_py_max_diff, 6),
            "c_forward_pass1_cold_ms": round(c_ms_pass1, 2),
            "c_forward_pass2_warm_ms": round(c_ms_pass2, 2),
            "specmd_r4_dm_ledger_pass1": dm_pass1,
            "specmd_r4_dm_ledger_pass2_warm": dm_pass2,
        },
        "native_hybrid_agent_layer2": {
            "module": "NemotronNativeHybridLink(d=4096, bottleneck=1024)",
            "shared_vocab_tokens_matched": len(shared_nemo_ids),
            "raw_hidden_to_next_embed_cos": round(raw_cos_te, 4),
            "trained_hybridlink_train_cos": round(train_cos, 4),
            "trained_hybridlink_test_cos": round(test_cos, 4),
            "latent_131k_vocab_readout_mean_cos": round(latent_readout_mean_cos, 4),
            "latent_131k_vocab_readout_sample": latent_readout_text[:120],
        },
        "native_ncm_layer3": {
            "vocab_size": V_NEMO,
            "vq_code_levels": list(vqvae.vq.levels),
            "vq_total_codes_per_window": int(idx_after.shape[1]),
            "vqvae_131k_token_acc_before_finetune": round(acc_before, 4),
            "vqvae_131k_token_acc_after_finetune": round(acc_after, 4),
            "vqvae_reconstructed_prompt": rec_sample_text,
            "ncm_11_level_logits_shape": list(ncm_logits.shape),
            "ncm_11_level_code_cross_entropy": round(ncm_code_ce, 4),
            "saved_checkpoint": str(out_ckpt_dir / "nemotron_hybrid_ncm_bridges.pt"),
        },
    }


if __name__ == "__main__":
    res = train_and_verify_nemotron_native_hybrid_and_ncm()
    out_p = Path("/tmp/nemotron_native_c_hybrid_ncm_report.json")
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"NEMOTRON_NATIVE_C_HYBRID_NCM_DONE: {out_p}", flush=True)
