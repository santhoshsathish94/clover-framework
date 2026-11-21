#!/usr/bin/env python3
"""Complete 3-Way Quality, Footprint, Speed & Prompt->Response Benchmark on DiffuSpace Clover-35B-A3B.

Compares `Clover-35B-A3B-Preview` (`BD-GDN + Gated GQA + SwiGLU MoE + 248,320 Vocab`) on `k3` (`16` CPU threads):
  1. Reference `BF16` (`2,264.9 MiB`: BF16 trunks + BF16 248K vocab + BF16 SwiGLU MoE)
  2. Shipped `Clover v5.0.7` (`2,123.5 MiB`: PyTorch BF16 trunks/vocab + planar Group-64 4-bit MoE)
  3. NEW `Clover HG4/HG8-128` (`858.4 MiB`: Native C11 AVX-512 VNNI+VBMI engine with Register-Fused SwiGLU MoE,
     SwiGLU Equilibrium `pre_diag`, Native BD-GDN Causal & Bidirectional Preview, and 1-Row Inverse-`FWHT` 248K Lookup)
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)

for p in (
    Path("/opt/clover-k3/web-studio/clover"),
    Path("/opt/clover-k3/clover-framework/porting-models"),
    Path("/opt/clover-k3/clover-framework/clover-ai-system/src/models/clover-35b-a3b"),
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clover_35b_gpu_backbone as cb  # noqa: E402
import omni_c2_c3  # noqa: E402


def dir_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    pruned_dir = Path("/opt/clover-k3/clover-35b-computed/pruned")
    hf_stage = Path("/opt/clover-k3/clover-35b-computed/hf-stage")
    hg4_dir = Path("/opt/clover-k3/clover35b-hg4")

    rt = omni_c2_c3.OmniC2C3Runtime(hf_stage, model_key="clover-35b-a3b")
    tok = rt.tok

    # Exact footprint of active C1 stage files:
    g0 = hf_stage / "trunks" / "global-0"
    base_trunks_b = (
        (g0 / "embed_tokens.bf16.bin").stat().st_size
        + (g0 / "lm_head.bf16.bin").stat().st_size
        + (g0 / "norm.bf16.bin").stat().st_size
        + dir_bytes(hf_stage / "trunks" / "trunk-0")
        + dir_bytes(hf_stage / "trunks" / "trunk-3")
    )
    bf16_exp_b = (
        (hf_stage / "experts" / "expert-0" / "experts.direct").stat().st_size
        + (hf_stage / "experts" / "expert-3" / "experts.direct").stat().st_size
    )
    q4_exp_b = (
        (hf_stage / "experts" / "expert-0" / "experts.4bit.direct").stat().st_size
        + (hf_stage / "experts" / "expert-3" / "experts.4bit.direct").stat().st_size
    )
    sz_bf16_mib = (base_trunks_b + bf16_exp_b) / (1024**2)
    sz_v507_mib = (base_trunks_b + q4_exp_b) / (1024**2)
    sz_hg4_mib = (sum(f.stat().st_size for f in hg4_dir.glob("*.hg4")) + (hg4_dir / "meta.bin").stat().st_size) / (1024**2)
    tot_params = (base_trunks_b + bf16_exp_b) / 2.0

    # Load 1. Reference BF16 & 2. Shipped Clover v5.0.7 (4-bit Group-64 MoE)
    mod_bf16 = cb.Clover35BGPUStageModel(pruned_dir, device="cpu", use_4bit_experts=False, from_raw_records=False)
    mod_v507 = cb.Clover35BGPUStageModel(pruned_dir, device="cpu", use_4bit_experts=True, from_raw_records=False)

    # Load 3. NEW Native C11 Clover HG4/HG8-128 Engine
    c_lib = ctypes.CDLL(str(hg4_dir / "libclover35b_hg4_engine.so"))
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_flt_p = ctypes.POINTER(ctypes.c_float)
    c_lib.clover35b_hg4_isa_name.argtypes = []
    c_lib.clover35b_hg4_isa_name.restype = ctypes.c_char_p
    c_lib.clover35b_hg4_init.argtypes = [ctypes.c_char_p]
    c_lib.clover35b_hg4_init.restype = ctypes.c_int
    c_lib.clover35b_hg4_forward_ids.argtypes = [c_int_p, ctypes.c_int, ctypes.c_int, c_flt_p]
    c_lib.clover35b_hg4_forward_ids.restype = ctypes.c_float
    c_lib.clover35b_hg4_all_logits.argtypes = [c_flt_p, ctypes.c_int, c_flt_p]
    c_lib.clover35b_hg4_all_logits.restype = ctypes.c_float
    c_lib.clover35b_hg4_lm_head_top1.argtypes = [c_flt_p, c_flt_p]
    c_lib.clover35b_hg4_lm_head_top1.restype = ctypes.c_int
    assert c_lib.clover35b_hg4_init(str(hg4_dir).encode()) == 0
    isa_str = c_lib.clover35b_hg4_isa_name().decode()

    # Held-Out 64-Token Multi-Domain Corpus
    corpus = (
        "Clover-35B-A3B combines Bidirectional Gated DeltaNet linear-attention layers, Gated 16Q/2KV Grouped-Query "
        "Attention with partial rotary embeddings, and 256-expert SwiGLU Mixture-of-Experts with block coreset routing:\n"
        "def block_coreset(probs, top_p=0.9):\n    order = probs.argsort(descending=True)\n    return order\n"
    )
    eval_ids = tok.encode(corpus)[:64]
    T = len(eval_ids)
    xb = torch.tensor([eval_ids], dtype=torch.long)
    targets = torch.tensor(eval_ids[1:], dtype=torch.long)
    c_ids = (ctypes.c_int * T)(*eval_ids)

    # Warmup
    with torch.no_grad():
        mod_bf16.forward_ids(xb[:, :4], preview=False)
        mod_v507.forward_ids(xb[:, :4], preview=False)
    w_h = np.zeros((4, 2048), dtype=np.float32)
    c_lib.clover35b_hg4_forward_ids((ctypes.c_int * 4)(*eval_ids[:4]), 4, 0, w_h.ctypes.data_as(c_flt_p))

    with torch.no_grad():
        t0 = time.perf_counter()
        out_bf16 = mod_bf16.forward_ids(xb, preview=False)
        ms_bf16 = (time.perf_counter() - t0) * 1000.0
        out_bf16_prev = mod_bf16.forward_ids(xb, preview=True)

        t0 = time.perf_counter()
        out_v507 = mod_v507.forward_ids(xb, preview=False)
        ms_v507 = (time.perf_counter() - t0) * 1000.0
        out_v507_prev = mod_v507.forward_ids(xb, preview=True)

        # Single-token step on v5.0.7
        t_s1 = time.perf_counter()
        mod_v507.forward_ids(xb[:, -1:], preview=False)
        s1_v507 = (time.perf_counter() - t_s1) * 1000.0

        t_s1b = time.perf_counter()
        mod_bf16.forward_ids(xb[:, -1:], preview=False)
        s1_bf16 = (time.perf_counter() - t_s1b) * 1000.0

    # Evaluate NEW Native C11 Clover HG4/HG8-128 (Causal Commit & Bidirectional Preview!)
    h_hg4 = np.zeros((T, 2048), dtype=np.float32)
    for _ in range(2):
        ms_hg4_h = float(c_lib.clover35b_hg4_forward_ids(c_ids, T, 0, h_hg4.ctypes.data_as(c_flt_p)))
    lg_hg4_np = np.zeros((T - 1, 248320), dtype=np.float32)
    ms_hg4_lm = float(c_lib.clover35b_hg4_all_logits(h_hg4[:-1].ctypes.data_as(c_flt_p), T - 1, lg_hg4_np.ctypes.data_as(c_flt_p)))

    h_hg4_prev = np.zeros((T, 2048), dtype=np.float32)
    for _ in range(2):
        ms_hg4_prev = float(c_lib.clover35b_hg4_forward_ids(c_ids, T, 1, h_hg4_prev.ctypes.data_as(c_flt_p)))

    # Single-token T=1 step + 248K LM-Head on HG4-128
    h1_hg4 = np.zeros((1, 2048), dtype=np.float32)
    lg1_hg4 = np.zeros(248320, dtype=np.float32)
    c1_id = (ctypes.c_int * 1)(eval_ids[-1])
    for _ in range(2):
        t_s1h = time.perf_counter()
        c_lib.clover35b_hg4_forward_ids(c1_id, 1, 0, h1_hg4.ctypes.data_as(c_flt_p))
        c_lib.clover35b_hg4_lm_head_top1(h1_hg4.ctypes.data_as(c_flt_p), lg1_hg4.ctypes.data_as(c_flt_p))
        s1_hg4 = (time.perf_counter() - t_s1h) * 1000.0

    h_ref = out_bf16["h_final"][0].float()
    lg_ref = out_bf16["logits"][0, :-1].float()
    h_ref_prev = out_bf16_prev["h_final"][0].float()

    def metrics(lg: torch.Tensor, h_c: torch.Tensor, h_c_prev: torch.Tensor):
        ppl = math.exp(float(F.cross_entropy(lg, targets).item()))
        p_ref = F.softmax(lg_ref, dim=-1)
        kl = float((p_ref * (F.log_softmax(lg_ref, dim=-1) - F.log_softmax(lg, dim=-1))).sum(-1).mean().item())
        top1 = float((lg.argmax(-1) == lg_ref.argmax(-1)).float().mean().item() * 100.0)
        h_cos = float(F.cosine_similarity(h_c, h_ref, dim=-1).mean().item())
        prev_cos = float(F.cosine_similarity(h_c_prev, h_ref_prev, dim=-1).mean().item())
        return ppl, kl, top1, h_cos, prev_cos

    ppl_ref = math.exp(float(F.cross_entropy(lg_ref, targets).item()))
    ppl_v, kl_v, t1_v, hc_v, pc_v = metrics(
        out_v507["logits"][0, :-1].float(), out_v507["h_final"][0].float(), out_v507_prev["h_final"][0].float()
    )
    ppl_h, kl_h, t1_h, hc_h, pc_h = metrics(
        torch.from_numpy(lg_hg4_np), torch.from_numpy(h_hg4), torch.from_numpy(h_hg4_prev)
    )

    bpw_b = (sz_bf16_mib * 1024**2 * 8.0) / tot_params
    bpw_v = (sz_v507_mib * 1024**2 * 8.0) / tot_params
    bpw_h = (sz_hg4_mib * 1024**2 * 8.0) / tot_params

    print("\n" + "=" * 118)
    print(f"  CLOVER-35B-A3B 3-WAY QUALITY, SIZE & BD-GDN PARITY ({T}-Token Corpus, 16 CPU Threads, {isa_str})")
    print("=" * 118)
    print(f"  1. Reference BF16      | {sz_bf16_mib:7.1f} MiB ({bpw_b:5.2f} bpw) | PPL {ppl_ref:9.2f} | KL 0.00000 | Top1 100.0% | Causal-Cos 1.000000 | Preview-Cos 1.000000")
    print(f"  2. Clover v5.0.7 Q4    | {sz_v507_mib:7.1f} MiB ({bpw_v:5.2f} bpw) | PPL {ppl_v:9.2f} | KL {kl_v:.5f} | Top1  {t1_v:5.1f}% | Causal-Cos {hc_v:.6f} | Preview-Cos {pc_v:.6f}")
    print(f"  3. NEW Clover HG4-128  | {sz_hg4_mib:7.1f} MiB ({bpw_h:5.2f} bpw) | PPL {ppl_h:9.2f} | KL {kl_h:.5f} | Top1  {t1_h:5.1f}% | Causal-Cos {hc_h:.6f} | Preview-Cos {pc_h:.6f}")

    print("\n" + "=" * 118)
    print("  CLOVER-35B-A3B C1 HIDDEN-STATE PREFILL, FULL +248K LOGITS & T=1 DECODE STEP THROUGHPUT (16 Threads)")
    print("=" * 118)
    tot_hg4 = ms_hg4_h + ms_hg4_lm
    print(f"  1. Reference BF16      | C1+248K ({T}t): {ms_bf16:7.1f}ms ({T/(ms_bf16/1000):6.1f} tok/s) | T=1 Step+248K LMHead: {s1_bf16:6.1f}ms ({1000/s1_bf16:5.2f} tok/s)")
    print(f"  2. Clover v5.0.7 Q4    | C1+248K ({T}t): {ms_v507:7.1f}ms ({T/(ms_v507/1000):6.1f} tok/s) | T=1 Step+248K LMHead: {s1_v507:6.1f}ms ({1000/s1_v507:5.2f} tok/s)")
    print(f"  3. NEW Clover HG4-128  | C1-Only ({T}t): {ms_hg4_h:7.1f}ms ({T/(ms_hg4_h/1000):6.1f} tok/s) | C1+248K: {tot_hg4:6.1f}ms ({T/(tot_hg4/1000):5.1f} tok/s) | T=1 Step+248K: {s1_hg4:5.1f}ms ({1000/s1_hg4:6.2f} tok/s) | BD-Preview: {ms_hg4_prev:5.1f}ms")

    # End-to-End Prompt -> Response via Full Clover C1 -> C2 HybridAgent -> C3 11-Level NCM (248,320 Vocab)!
    print("\n" + "=" * 118)
    print("  CLOVER-35B-A3B FULL PROMPT -> RESPONSE (C1 -> C2 HybridAgent -> C3 11-Level NCM 248,320-Vocab Decoder)")
    print("=" * 118)

    def fn_v507(seq_ids: list[int]) -> np.ndarray:
        ids_c = [max(0, min(248319, int(x))) for x in (seq_ids or [248044])]
        with torch.no_grad():
            return mod_v507.forward_ids(torch.tensor([ids_c], dtype=torch.long), preview=False)["h_final"][0].float().numpy()

    def fn_hg4(seq_ids: list[int]) -> np.ndarray:
        ids_c = [max(0, min(248319, int(x))) for x in (seq_ids or [248044])]
        n = len(ids_c)
        out = np.zeros((n, 2048), dtype=np.float32)
        c_lib.clover35b_hg4_forward_ids((ctypes.c_int * n)(*ids_c), n, 0, out.ctypes.data_as(c_flt_p))
        return out

    prompts = [
        ("Math", "Solar capacity check: 160 panels at 450 W each generates 72 kW total, yielding $54.00 per hour at $0.75/kWh."),
        ("Code", "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2"),
        ("Coord", "Clover-35B-A3B coordinates multi-step workflows using Bidirectional Gated DeltaNet preview blocks and 256-expert SwiGLU coreset routing."),
    ]
    for tag, p_txt in prompts:
        t0 = time.perf_counter()
        c1_s_old, c1_i_old = rt.c1_infer_states(p_txt, states_fn=fn_v507, model_name="Clover-35B-v5.0.7")
        c2_s_old, _ = rt.c2_plan(p_txt, c1_state=c1_s_old)
        ans_old, _ = rt.c3_decode(c2_s_old, full_ncm=True)
        ms_old = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        c1_s_new, c1_i_new = rt.c1_infer_states(p_txt, states_fn=fn_hg4, model_name="Clover-35B-HG4-128")
        c2_s_new, _ = rt.c2_plan(p_txt, c1_state=c1_s_new)
        ans_new, _ = rt.c3_decode(c2_s_new, full_ncm=True)
        ms_new = (time.perf_counter() - t1) * 1000.0

        print(f"\n=== [{tag}] Prompt: {p_txt} ===")
        print(f"  v5.0.7 PyTorch Q4 ({sz_v507_mib:6.0f} MiB) | C1={c1_i_old.get('forward_ms')}ms | Total={ms_old:6.1f}ms -> {ans_old!r}")
        print(f"  NEW C11 HG4-128   ({sz_hg4_mib:6.0f} MiB) | C1={c1_i_new.get('forward_ms')}ms | Total={ms_new:6.1f}ms -> {ans_new!r}")


if __name__ == "__main__":
    main()
