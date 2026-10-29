#!/usr/bin/env python3
"""
4-Bit Planar Group-64 Direct Expert Container Builder (`experts.4bit.direct`) & Native 2-Layer MTP C Verification
for NVIDIA Nemotron 3.5 Super VL (`clover-ai-system/src/models/nemotron-3.5-super-vl/build_and_verify_nemotron_4bit_mtp.py`).

1. Builds `experts.4bit.direct` inside `/opt/clover-k3/nemotron-computed/pruned/expert-1/` and
   `/opt/clover-k3/nemotron-computed/pruned/expert-mtp-1/`:
   - Each of the 512 routed non-gated `ReLU^2` experts (`up_proj: [2688, 1024]`, `down_proj: [1024, 2688]`)
     is quantized into Clover's planar 4-bit group-64 format (`quantize_matrix_4bit_group64`):
       * `up_proj` (`rows = 2688, cols = 1024, groups = 16`):
         - `up_nibbles`: `2688 * 16 * 32 = 1,376,256 B`
         - `up_steps`: `2688 * 16 = 43,008 B`
         - `up_scales_fp32`: `2688 * 4 = 10,752 B`
         (Total `up_proj` 4-bit = `1,430,016 B` vs `5,505,024 B` BF16)
       * `down_proj` (`rows = 1024, cols = 2688, groups = 42`):
         - `down_nibbles`: `1024 * 42 * 32 = 1,376,256 B`
         - `down_steps`: `1024 * 42 = 43,008 B`
         - `down_scales_fp32`: `1024 * 4 = 4,096 B`
         (Total `down_proj` 4-bit = `1,423,360 B` vs `5,505,024 B` BF16)
       * Total 4-bit expert stride `EXPERT_4BIT_STRIDE = 2,853,376 B` (`2.721 MiB` vs `11,010,048 B` = `10.500 MiB` BF16,
         a **3.859x memory & bandwidth reduction**; full 512-expert layer = **`1.393 GiB`** vs `5.250 GiB` BF16!).
2. Compiles and verifies the upgraded `libnemotron_cpu_engine.so` (`nemotron_cpu_engine.c`) supporting:
   - `nemo_forward_pipeline_4bit` / `nemo_set_use_4bit_experts(1)` (executing 8-token register-blocked 4-bit group-64
     planar nibble unpacking directly in C/OpenMP without dequantizing the whole matrix to RAM!).
   - `nemo_mtp_draft_forward` (executing the complete 2-layer Native Multi-Token Prediction speculative drafter
     `enorm(e_{t+1}) || hnorm(h_t) -> eh_proj -> mtp.layers.0 [NoPE 16:1 GQA] -> mtp.layers.1 [512-expert Latent ReLU^2 MoE] -> final_layernorm`
     directly in C/OpenMP from `trunk-mtp-0`, `trunk-mtp-1`, and `expert-mtp-1/experts.4bit.direct` or `experts.direct`!).
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nemotron_precomputed_engine import (
    DTYPE_BF16,
    NemotronDirectExpertStore,
    NemotronPrunedTrunkReader,
    bytes_to_tensor,
    execute_native_mtp_draft_from_precomputed,
)
from prepare_nemotron_datasets import quantize_matrix_4bit_group64

EXPERT_BF16_STRIDE = 11010048
UP_ELEMS = 2688 * 1024
DOWN_ELEMS = 1024 * 2688

UP_4BIT_BYTES = (2688 * 16 * 32) + (2688 * 16) + (2688 * 4)  # 1,430,016 B
DOWN_4BIT_BYTES = (1024 * 42 * 32) + (1024 * 42) + (1024 * 4)  # 1,423,360 B
EXPERT_4BIT_STRIDE = UP_4BIT_BYTES + DOWN_4BIT_BYTES  # 2,853,376 B


def build_4bit_direct_expert_file(expert_dir: Path) -> dict:
    """Converts `expert_dir/experts.direct` (`5.25 GiB` BF16) into `expert_dir/experts.4bit.direct` (`1.39 GiB` 4-bit group-64)."""
    bf16_path = expert_dir / "experts.direct"
    out_4bit_path = expert_dir / "experts.4bit.direct"
    if out_4bit_path.exists() and out_4bit_path.stat().st_size == 512 * EXPERT_4BIT_STRIDE:
        return {
            "expert_dir": str(expert_dir),
            "bf16_bytes": bf16_path.stat().st_size,
            "fourbit_bytes": out_4bit_path.stat().st_size,
            "compression_ratio": round(bf16_path.stat().st_size / out_4bit_path.stat().st_size, 4),
            "cached": True,
        }

    t0 = time.perf_counter()
    raw_mm = np.memmap(str(bf16_path), dtype=np.uint16, mode="r")
    with open(out_4bit_path, "wb") as f_out:
        for e in range(512):
            base_u16 = e * (EXPERT_BF16_STRIDE // 2)
            up_u16 = raw_mm[base_u16 : base_u16 + UP_ELEMS]
            down_u16 = raw_mm[base_u16 + UP_ELEMS : base_u16 + UP_ELEMS + DOWN_ELEMS]

            up_t = torch.from_numpy(np.array(up_u16, copy=True)).view(torch.bfloat16).view(2688, 1024)
            down_t = torch.from_numpy(np.array(down_u16, copy=True)).view(torch.bfloat16).view(1024, 2688)

            u_nib, u_stp, u_scl = quantize_matrix_4bit_group64(up_t)
            d_nib, d_stp, d_scl = quantize_matrix_4bit_group64(down_t)

            f_out.write(u_nib)
            f_out.write(u_stp)
            f_out.write(u_scl)
            f_out.write(d_nib)
            f_out.write(d_stp)
            f_out.write(d_scl)
            if (e + 1) % 128 == 0:
                print(f"  [{expert_dir.name}] Quantized {e + 1}/512 experts to 4-bit group-64...", flush=True)

    elapsed = round(time.perf_counter() - t0, 2)
    bf16_sz = bf16_path.stat().st_size
    q4_sz = out_4bit_path.stat().st_size
    return {
        "expert_dir": str(expert_dir),
        "bf16_bytes": bf16_sz,
        "fourbit_bytes": q4_sz,
        "compression_ratio": round(bf16_sz / q4_sz, 4),
        "build_seconds": elapsed,
        "cached": False,
    }


def main():
    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"
    pruned_dir = Path("/opt/clover-k3/nemotron-computed/pruned")
    src_dir = Path("/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16")

    print("=== Step 1/3: Building 4-Bit Planar Group-64 Direct Expert Stores (`experts.4bit.direct`) ===", flush=True)
    e1_info = build_4bit_direct_expert_file(pruned_dir / "expert-1")
    emtp_info = build_4bit_direct_expert_file(pruned_dir / "expert-mtp-1")
    print(json.dumps({"expert_1": e1_info, "expert_mtp_1": emtp_info}, indent=2), flush=True)

    print("=== Step 2/3: Compiling & Verifying C/OpenMP 4-Bit Expert Kernel + 2-Layer Native MTP Drafter ===", flush=True)
    from nemotron_hybrid_ncm import D_NEMO, V_NEMO, NemotronNativeCPUEngine, ensure_nemotron_cpu_engine_so

    # Force recompile of `nemotron_cpu_engine.c`
    so_path = SCRIPT_DIR / "libnemotron_cpu_engine.so"
    if so_path.exists():
        so_path.unlink()
    ensure_nemotron_cpu_engine_so(SCRIPT_DIR)
    eng = NemotronNativeCPUEngine(pruned_dir)

    tok = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)
    prompt = "Clover AI System switches seamlessly between Kimi K3, NVIDIA Nemotron 3.5 Super VL, and Liquid LFM3."
    ids = tok.encode(prompt, add_special_tokens=False)
    T = len(ids)

    # 1. BF16 C forward pass
    eng.set_use_4bit_experts(False)
    h_bf16, ms_bf16 = eng.forward_ids(ids)

    # 2. 4-Bit Group-64 C forward pass
    eng.set_use_4bit_experts(True)
    h_4bit, ms_4bit = eng.forward_ids(ids)

    cos_4bit_vs_bf16 = float(F.cosine_similarity(h_bf16.reshape(1, -1), h_4bit.reshape(1, -1), dim=-1).item())
    print(
        f"[C/OpenMP 4-Bit vs BF16] T={T} tokens: BF16={ms_bf16:.2f}ms, 4-Bit={ms_4bit:.2f}ms "
        f"({ms_bf16/max(0.1, ms_4bit):.2f}x speedup), cosine={cos_4bit_vs_bf16:.6f}",
        flush=True,
    )

    # 3. Verify C/OpenMP 2-Layer Native MTP Speculative Drafter against Python `execute_native_mtp_draft_from_precomputed`
    print("=== Step 3/3: Verifying C/OpenMP 2-Layer Native MTP Speculative Drafter vs Official Precomputed Reference ===", flush=True)
    emb_w = bytes_to_tensor((pruned_dir / "global-0" / "embeddings.bf16.bin").read_bytes(), DTYPE_BF16, (V_NEMO, D_NEMO))
    lm_head_w = bytes_to_tensor((pruned_dir / "global-0" / "lm_head.bf16.bin").read_bytes(), DTYPE_BF16, (V_NEMO, D_NEMO))

    mtp0_trk = NemotronPrunedTrunkReader(pruned_dir / "trunk-mtp-0")
    mtp1_trk = NemotronPrunedTrunkReader(pruned_dir / "trunk-mtp-1")
    mtp1_exp = NemotronDirectExpertStore(pruned_dir / "expert-mtp-1", pool_slots=64, host_slots=512)

    next_ids = ids[1:] + [ids[-1]]
    next_embeds = F.embedding(torch.tensor([next_ids], dtype=torch.long), emb_w)
    h_bf16_3d = h_bf16.to(torch.bfloat16).unsqueeze(0)

    t_py0 = time.perf_counter()
    mtp_logits_py, _ = execute_native_mtp_draft_from_precomputed(
        mtp0_trk, mtp1_trk, mtp1_exp, next_embeds, h_bf16_3d, lm_head_w
    )
    py_mtp_ms = (time.perf_counter() - t_py0) * 1000.0

    eng.set_use_4bit_experts(False)
    h_mtp_c_bf16, c_mtp_bf16_ms = eng.forward_mtp_draft(next_ids, h_bf16)
    logits_mtp_c_bf16 = F.linear(h_mtp_c_bf16.to(torch.bfloat16), lm_head_w).float()

    eng.set_use_4bit_experts(True)
    h_mtp_c_4bit, c_mtp_4bit_ms = eng.forward_mtp_draft(next_ids, h_bf16)
    logits_mtp_c_4bit = F.linear(h_mtp_c_4bit.to(torch.bfloat16), lm_head_w).float()

    mtp_bf16_cos = float(
        F.cosine_similarity(mtp_logits_py[0, -1:].float(), logits_mtp_c_bf16[-1:].float(), dim=-1).item()
    )
    mtp_4bit_cos = float(
        F.cosine_similarity(mtp_logits_py[0, -1:].float(), logits_mtp_c_4bit[-1:].float(), dim=-1).item()
    )
    top1_py = int(mtp_logits_py[0, -1].argmax().item())
    top1_c_bf16 = int(logits_mtp_c_bf16[-1].argmax().item())
    top1_c_4bit = int(logits_mtp_c_4bit[-1].argmax().item())

    report = {
        "expert_4bit_containers": {
            "expert_1": e1_info,
            "expert_mtp_1": emtp_info,
            "expert_bf16_stride_bytes": EXPERT_BF16_STRIDE,
            "expert_4bit_stride_bytes": EXPERT_4BIT_STRIDE,
            "bandwidth_reduction_factor": round(EXPERT_BF16_STRIDE / EXPERT_4BIT_STRIDE, 4),
        },
        "backbone_c_engine_4bit_vs_bf16": {
            "prompt_tokens": T,
            "bf16_ms": round(ms_bf16, 2),
            "fourbit_ms": round(ms_4bit, 2),
            "speedup_ratio": round(ms_bf16 / max(0.1, ms_4bit), 3),
            "cosine_similarity_4bit_vs_bf16": round(cos_4bit_vs_bf16, 6),
        },
        "native_2_layer_mtp_speculative_drafter_c_engine": {
            "python_reference_ms": round(py_mtp_ms, 2),
            "c_bf16_mtp_ms": round(c_mtp_bf16_ms, 2),
            "c_4bit_mtp_ms": round(c_mtp_4bit_ms, 2),
            "c_4bit_speedup_vs_bf16": round(c_mtp_bf16_ms / max(0.1, c_mtp_4bit_ms), 3),
            "c_bf16_vs_python_logits_cosine": round(mtp_bf16_cos, 6),
            "c_4bit_vs_python_logits_cosine": round(mtp_4bit_cos, 6),
            "top1_token_id_python": top1_py,
            "top1_token_id_c_bf16": top1_c_bf16,
            "top1_token_id_c_4bit": top1_c_4bit,
            "top1_exact_match": (top1_py == top1_c_bf16 == top1_c_4bit),
        },
    }
    out_json = SCRIPT_DIR.parents[3] / "research" / "nemotron_4bit_and_native_mtp_c_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {out_json}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
