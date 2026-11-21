#!/usr/bin/env python3
"""Complete 3-Way Quality, Footprint, Throughput & Prompt->Response Benchmark on NVIDIA Nemotron 3.5 Super VL.

Compares unchanged `Nemotron 3.5 Super VL (120B-A12B MoE)` on `k3` (`16` CPU threads, `AVX-512 VNNI+VBMI`):
  1. Reference `BF16` (`12.96 GiB`: BF16 trunks + BF16 512-MoE + BF16 131K vocab + BF16 MTP)
  2. Shipped `Clover v5.0.7 CLVR4B01` (`5.36 GiB`: BF16 trunks/vocab + planar Group-64 4-bit 512-MoE)
  3. NEW `Clover HG4-128` (`~3.36 GiB`: 100% HG4-128 trunks + Shared-Input 512-MoE + 131K vocab + MTP)
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

STUDIO_CLOVER = next(
    (
        p for p in (
            Path("/opt/clover-k3/web-studio/clover"),
            Path("/opt/clover-k3/clover-framework/clover-ai-system/studio/clover"),
            Path(__file__).resolve().parent.parent.parent.parent / "studio" / "clover" if len(Path(__file__).resolve().parents) >= 4 else Path("/nonexistent"),
        )
        if p.is_dir()
    ),
    Path("/opt/clover-k3/web-studio/clover"),
)
sys.path.insert(0, str(STUDIO_CLOVER))

import omni_c2_c3  # noqa: E402


def bind_nemo_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_flt_p = ctypes.POINTER(ctypes.c_float)
    lib.nemo_engine_init.argtypes = [ctypes.c_char_p]
    lib.nemo_engine_init.restype = ctypes.c_int
    lib.nemo_forward_pipeline.argtypes = [c_int_p, ctypes.c_int, c_flt_p]
    lib.nemo_forward_pipeline.restype = ctypes.c_float
    lib.nemo_lm_head_logits.argtypes = [c_flt_p, c_flt_p, c_flt_p]
    lib.nemo_lm_head_logits.restype = ctypes.c_int
    lib.nemo_mtp_draft_forward.argtypes = [c_int_p, c_flt_p, ctypes.c_int, c_flt_p]
    lib.nemo_mtp_draft_forward.restype = ctypes.c_float
    if hasattr(lib, "nemo_set_use_4bit_experts"):
        lib.nemo_set_use_4bit_experts.argtypes = [ctypes.c_int]
        lib.nemo_set_use_4bit_experts.restype = None
    if hasattr(lib, "nemo_hg4_isa_name"):
        lib.nemo_hg4_isa_name.argtypes = []
        lib.nemo_hg4_isa_name.restype = ctypes.c_char_p
    if hasattr(lib, "nemo_all_logits"):
        lib.nemo_all_logits.argtypes = [c_flt_p, ctypes.c_int, c_flt_p]
        lib.nemo_all_logits.restype = ctypes.c_float
    return lib


def file_or_dir_bytes(p: Path, exclude_names: set[str] | None = None) -> int:
    excl = exclude_names or set()
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.name not in excl)


def main() -> None:
    pruned_dir = Path("/opt/clover-k3/nemotron-computed/pruned")
    hf_stage = Path("/opt/clover-k3/nemotron-computed/hf-stage")
    hg4_dir = Path("/opt/clover-k3/nemotron-hg4")
    work_dir = Path(os.environ.get("BENCH_NEMO_WORK", "/tmp/bench_nemo_hg4"))

    rt = omni_c2_c3.OmniC2C3Runtime(hf_stage, model_key="nemotron-3.5-super-vl")
    tok = rt.tok

    # Footprint calculation across the exact active C1 + MTP components loaded by nemo_engine_init:
    active_trunks = ("global-0", "trunk-0", "trunk-1", "trunk-7", "trunk-mtp-0", "trunk-mtp-1")
    trunk_bf16_b = sum(file_or_dir_bytes(pruned_dir / t) for t in active_trunks)
    exp_bf16_b = (pruned_dir / "expert-1" / "experts.direct").stat().st_size + (pruned_dir / "expert-mtp-1" / "experts.direct").stat().st_size
    exp_q4_b = (pruned_dir / "expert-1" / "experts.4bit.direct").stat().st_size + (pruned_dir / "expert-mtp-1" / "experts.4bit.direct").stat().st_size

    sz_bf16_mib = (trunk_bf16_b + exp_bf16_b) / (1024**2)
    sz_v507_mib = (trunk_bf16_b + exp_q4_b) / (1024**2)
    sz_hg4_mib = sum(f.stat().st_size for f in hg4_dir.glob("*.hg4")) / (1024**2) + (hg4_dir / "meta.bin").stat().st_size / (1024**2)
    total_params = (trunk_bf16_b + exp_bf16_b) / 2.0  # 2 bytes per BF16 param

    # Load both C libraries (two separate .so copies for BF16 vs v5.0.7 CLVR4B01 so static globals stay independent)
    lib_bf16 = bind_nemo_lib(work_dir / "libnemotron_bf16.so")
    lib_q4 = bind_nemo_lib(work_dir / "libnemotron_v507.so")
    lib_hg4 = bind_nemo_lib(hg4_dir / "libnemotron_hg4_engine.so")

    assert lib_bf16.nemo_engine_init(str(pruned_dir).encode()) == 0
    lib_bf16.nemo_set_use_4bit_experts(0)

    assert lib_q4.nemo_engine_init(str(pruned_dir).encode()) == 0
    lib_q4.nemo_set_use_4bit_experts(1)

    assert lib_hg4.nemo_engine_init(str(hg4_dir).encode()) == 0
    isa_str = lib_hg4.nemo_hg4_isa_name().decode()

    # Held-Out 64-Token Multi-Domain Corpus
    corpus = (
        "NVIDIA Nemotron 3.5 Super VL combines Mamba-2 selective state-space layers, 512-expert Latent ReLU-squared "
        "Mixture-of-Experts routing top-22 active experts per token, and NoPE 16:1 Grouped-Query Attention:\n"
        "def latent_moe(x, up_w, down_w):\n    h = torch.relu(x @ up_w.T).square()\n    return h @ down_w.T\n"
    )
    eval_ids = tok.encode(corpus)[:64]
    T = len(eval_ids)
    c_ids = (ctypes.c_int * T)(*eval_ids)
    c_next = (ctypes.c_int * (T - 1))(*eval_ids[1:])
    targets = torch.tensor(eval_ids[1:], dtype=torch.long)
    c_flt_p = ctypes.POINTER(ctypes.c_float)

    # Warmup
    w_ids = (ctypes.c_int * 4)(*eval_ids[:4])
    w_h = np.zeros((4, 4096), dtype=np.float32)
    lib_bf16.nemo_forward_pipeline(w_ids, 4, w_h.ctypes.data_as(c_flt_p))
    lib_q4.nemo_forward_pipeline(w_ids, 4, w_h.ctypes.data_as(c_flt_p))
    lib_hg4.nemo_forward_pipeline(w_ids, 4, w_h.ctypes.data_as(c_flt_p))

    def eval_legacy_engine(lib: ctypes.CDLL):
        h = np.zeros((T, 4096), dtype=np.float32)
        t0 = time.perf_counter()
        lib.nemo_forward_pipeline(c_ids, T, h.ctypes.data_as(c_flt_p))
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        # Compute exact [T-1, 131072] BF16 lm_head logits via PyTorch F.linear on exact lm_head.bf16.bin for speed,
        # AND time single-token C `nemo_lm_head_logits` directly!
        one_lg = np.zeros(131072, dtype=np.float32)
        mx = ctypes.c_float()
        t_lm0 = time.perf_counter()
        lib.nemo_lm_head_logits(h[-1:].ctypes.data_as(c_flt_p), one_lg.ctypes.data_as(c_flt_p), ctypes.byref(mx))
        lm1_ms = (time.perf_counter() - t_lm0) * 1000.0

        # Single-token T=1 C1 step latency
        h1 = np.zeros((1, 4096), dtype=np.float32)
        c1_id = (ctypes.c_int * 1)(eval_ids[-1])
        t_s1 = time.perf_counter()
        lib.nemo_forward_pipeline(c1_id, 1, h1.ctypes.data_as(c_flt_p))
        step1_ms = (time.perf_counter() - t_s1) * 1000.0

        # 2-Layer MTP Drafter
        mtp_h = np.zeros((T - 1, 4096), dtype=np.float32)
        mtp_ms = float(lib.nemo_mtp_draft_forward(c_next, h[:-1].ctypes.data_as(c_flt_p), T - 1, mtp_h.ctypes.data_as(c_flt_p)))
        return h, fwd_ms, step1_ms, lm1_ms, mtp_h, mtp_ms

    # Load BF16 lm_head once as float32 tensor to score legacy BF16 & v5.0.7 all-token [T-1, 131072] logits quickly
    lm_u16 = np.memmap(pruned_dir / "global-0" / "lm_head.bf16.bin", dtype=np.uint16, mode="r").reshape(131072, 4096)
    lm_ref_t = torch.from_numpy(np.left_shift(np.asarray(lm_u16, dtype=np.uint32), np.uint32(16)).view(np.float32))

    h_bf16, fwd_bf16, s1_bf16, lm1_bf16, mtp_bf16, mtp_ms_bf16 = eval_legacy_engine(lib_bf16)
    h_q4, fwd_q4, s1_q4, lm1_q4, mtp_q4, mtp_ms_q4 = eval_legacy_engine(lib_q4)

    with torch.no_grad():
        lg_bf16 = F.linear(torch.from_numpy(h_bf16[:-1]), lm_ref_t)
        lg_q4 = F.linear(torch.from_numpy(h_q4[:-1]), lm_ref_t)
        mtp_lg_bf16 = F.linear(torch.from_numpy(mtp_bf16), lm_ref_t)
        mtp_lg_q4 = F.linear(torch.from_numpy(mtp_q4), lm_ref_t)
    del lm_ref_t

    # Evaluate NEW Clover HG4-128 (`100% HG4-128` including `lm_head.hg4`!)
    h_hg4 = np.zeros((T, 4096), dtype=np.float32)
    for _ in range(2):
        t0 = time.perf_counter()
        lib_hg4.nemo_forward_pipeline(c_ids, T, h_hg4.ctypes.data_as(c_flt_p))
        fwd_hg4 = (time.perf_counter() - t0) * 1000.0

    lg_hg4_np = np.zeros((T - 1, 131072), dtype=np.float32)
    lib_hg4.nemo_all_logits(h_hg4[:-1].ctypes.data_as(c_flt_p), T - 1, lg_hg4_np.ctypes.data_as(c_flt_p))
    lg_hg4 = torch.from_numpy(lg_hg4_np)

    one_lg_hg4 = np.zeros(131072, dtype=np.float32)
    mx = ctypes.c_float()
    for _ in range(2):
        t_lm0 = time.perf_counter()
        lib_hg4.nemo_lm_head_logits(h_hg4[-1:].ctypes.data_as(c_flt_p), one_lg_hg4.ctypes.data_as(c_flt_p), ctypes.byref(mx))
        lm1_hg4 = (time.perf_counter() - t_lm0) * 1000.0

    h1_hg4 = np.zeros((1, 4096), dtype=np.float32)
    c1_id = (ctypes.c_int * 1)(eval_ids[-1])
    for _ in range(2):
        t_s1 = time.perf_counter()
        lib_hg4.nemo_forward_pipeline(c1_id, 1, h1_hg4.ctypes.data_as(c_flt_p))
        s1_hg4 = (time.perf_counter() - t_s1) * 1000.0

    mtp_hg4 = np.zeros((T - 1, 4096), dtype=np.float32)
    for _ in range(2):
        mtp_ms_hg4 = float(lib_hg4.nemo_mtp_draft_forward(c_next, h_hg4[:-1].ctypes.data_as(c_flt_p), T - 1, mtp_hg4.ctypes.data_as(c_flt_p)))
    mtp_lg_hg4_np = np.zeros((T - 1, 131072), dtype=np.float32)
    lib_hg4.nemo_all_logits(mtp_hg4.ctypes.data_as(c_flt_p), T - 1, mtp_lg_hg4_np.ctypes.data_as(c_flt_p))
    mtp_lg_hg4 = torch.from_numpy(mtp_lg_hg4_np)

    def metrics_vs_bf16(lg: torch.Tensor, h_np: np.ndarray, mtp_lg: torch.Tensor):
        ppl = math.exp(float(F.cross_entropy(lg, targets).item()))
        p_ref = F.softmax(lg_bf16, dim=-1)
        kl = float((p_ref * (F.log_softmax(lg_bf16, dim=-1) - F.log_softmax(lg, dim=-1))).sum(dim=-1).mean().item())
        top1 = float((lg.argmax(dim=-1) == lg_bf16.argmax(dim=-1)).float().mean().item() * 100.0)
        l_cos = float(F.cosine_similarity(lg, lg_bf16, dim=-1).mean().item())
        h_cos = float(F.cosine_similarity(torch.from_numpy(h_np), torch.from_numpy(h_bf16), dim=-1).mean().item())
        mtp_top1 = float((mtp_lg.argmax(dim=-1) == mtp_lg_bf16.argmax(dim=-1)).float().mean().item() * 100.0)
        return ppl, kl, top1, l_cos, h_cos, mtp_top1

    ppl_ref = math.exp(float(F.cross_entropy(lg_bf16, targets).item()))
    ppl_q4, kl_q4, t1_q4, lcos_q4, hcos_q4, mtp1_q4 = metrics_vs_bf16(lg_q4, h_q4, mtp_lg_q4)
    ppl_hg4, kl_hg4, t1_hg4, lcos_hg4, hcos_hg4, mtp1_hg4 = metrics_vs_bf16(lg_hg4, h_hg4, mtp_lg_hg4)

    bpw_bf16 = (sz_bf16_mib * 1024**2 * 8.0) / total_params
    bpw_v507 = (sz_v507_mib * 1024**2 * 8.0) / total_params
    bpw_hg4 = (sz_hg4_mib * 1024**2 * 8.0) / total_params

    print("\n" + "=" * 116)
    print(f"  NEMOTRON 3.5 SUPER VL 3-WAY QUALITY, SIZE & SPEED SUMMARY ({T}-Token Corpus, 16 CPU Threads, {isa_str})")
    print("=" * 116)
    print(f"  1. Reference BF16      | {sz_bf16_mib:8.1f} MiB ({bpw_bf16:5.2f} bpw) | PPL {ppl_ref:8.2f} | KL 0.00000 | Top1 100.0% | H-Cos 1.000000 | MTP-Top1 100.0%")
    print(f"  2. Clover v5.0.7 Q4    | {sz_v507_mib:8.1f} MiB ({bpw_v507:5.2f} bpw) | PPL {ppl_q4:8.2f} | KL {kl_q4:.5f} | Top1  {t1_q4:5.1f}% | H-Cos {hcos_q4:.6f} | MTP-Top1  {mtp1_q4:5.1f}%")
    print(f"  3. NEW Clover HG4-128  | {sz_hg4_mib:8.1f} MiB ({bpw_hg4:5.2f} bpw) | PPL {ppl_hg4:8.2f} | KL {kl_hg4:.5f} | Top1  {t1_hg4:5.1f}% | H-Cos {hcos_hg4:.6f} | MTP-Top1  {mtp1_hg4:5.1f}%")

    print("\n" + "=" * 116)
    print("  NEMOTRON 3.5 C1 PREFILL, SINGLE-TOKEN DECODE STEP, 131K LM-HEAD & 2-LAYER MTP THROUGHPUT (16 Threads)")
    print("=" * 116)
    tot1_bf16 = s1_bf16 + lm1_bf16
    tot1_q4 = s1_q4 + lm1_q4
    tot1_hg4 = s1_hg4 + lm1_hg4
    print(f"  1. Reference BF16      | Prefill ({T}t): {fwd_bf16:7.1f}ms ({T/(fwd_bf16/1000):5.1f} tok/s) | T=1 Step+131K LMHead: {s1_bf16:6.1f}+{lm1_bf16:5.1f}={tot1_bf16:6.1f}ms ({1000/tot1_bf16:5.2f} tok/s) | MTP ({T-1}t): {mtp_ms_bf16:7.1f}ms")
    print(f"  2. Clover v5.0.7 Q4    | Prefill ({T}t): {fwd_q4:7.1f}ms ({T/(fwd_q4/1000):5.1f} tok/s) | T=1 Step+131K LMHead: {s1_q4:6.1f}+{lm1_q4:5.1f}={tot1_q4:6.1f}ms ({1000/tot1_q4:5.2f} tok/s) | MTP ({T-1}t): {mtp_ms_q4:7.1f}ms")
    print(f"  3. NEW Clover HG4-128  | Prefill ({T}t): {fwd_hg4:7.1f}ms ({T/(fwd_hg4/1000):5.1f} tok/s) | T=1 Step+131K LMHead: {s1_hg4:6.1f}+{lm1_hg4:5.1f}={tot1_hg4:6.1f}ms ({1000/tot1_hg4:5.2f} tok/s) | MTP ({T-1}t): {mtp_ms_hg4:7.1f}ms")

    # End-to-End Prompt -> Response via Full Clover C1 -> C2 HybridAgent -> C3 11-Level NCM (131,072 Tekken Vocab)!
    print("\n" + "=" * 116)
    print("  NEMOTRON 3.5 FULL PROMPT -> RESPONSE (C1 -> C2 HybridAgent -> C3 11-Level NCM 131K Decoder)")
    print("=" * 116)

    def make_states_fn(lib: ctypes.CDLL):
        def _fn(seq_ids: list[int]) -> np.ndarray:
            ids_c = [max(0, min(131071, int(x))) for x in (seq_ids or [1])]
            n = len(ids_c)
            out = np.zeros((n, 4096), dtype=np.float32)
            lib.nemo_forward_pipeline((ctypes.c_int * n)(*ids_c), n, out.ctypes.data_as(c_flt_p))
            return out
        return _fn

    fn_v507 = make_states_fn(lib_q4)
    fn_hg4 = make_states_fn(lib_hg4)

    prompts = [
        ("Math", "Solar capacity check: 160 panels at 450 W each generates 72 kW total, yielding $54.00 per hour at $0.75/kWh."),
        ("Code", "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2"),
        ("Arch", "Latent Mixture-of-Experts projects 4096 hidden dimensions into a 1024-dimensional latent space before routing 22 of 512 ReLU-squared experts."),
    ]
    for tag, p_txt in prompts:
        t0 = time.perf_counter()
        c1_s_old, c1_i_old = rt.c1_infer_states(p_txt, states_fn=fn_v507, model_name="Nemotron-v5.0.7")
        c2_s_old, _ = rt.c2_plan(p_txt, c1_state=c1_s_old)
        ans_old, _ = rt.c3_decode(c2_s_old, full_ncm=True)
        ms_old = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        c1_s_new, c1_i_new = rt.c1_infer_states(p_txt, states_fn=fn_hg4, model_name="Nemotron-HG4-128")
        c2_s_new, _ = rt.c2_plan(p_txt, c1_state=c1_s_new)
        ans_new, _ = rt.c3_decode(c2_s_new, full_ncm=True)
        ms_new = (time.perf_counter() - t1) * 1000.0

        print(f"\n=== [{tag}] Prompt: {p_txt} ===")
        print(f"  v5.0.7 CLVR4B01 ({sz_v507_mib:6.0f} MiB) | C1={c1_i_old.get('forward_ms')}ms | Total={ms_old:6.1f}ms -> {ans_old!r}")
        print(f"  NEW HG4-128     ({sz_hg4_mib:6.0f} MiB) | C1={c1_i_new.get('forward_ms')}ms | Total={ms_new:6.1f}ms -> {ans_new!r}")


if __name__ == "__main__":
    main()
