import os
import math
import sys
import traceback

os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

def attention_ref_exact(q, k, v, causal=False, softmax_scale=None, bias=None, compute_dtype=torch.float64):
    """
    Reference attention matching tests/test_flash_attn.py:attention_ref.
    q: (B, Sq, H, D), k: (B, Sk, H, D), v: (B, Sk, H, D)
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(D))
    qf = q.to(compute_dtype).transpose(1, 2)  # (B, H, Sq, D)
    kf = k.to(compute_dtype).transpose(1, 2)  # (B, H, Sk, D)
    vf = v.to(compute_dtype).transpose(1, 2)  # (B, H, Sk, D)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # (B, H, Sq, Sk)
    if bias is not None:
        scores = scores + bias.to(compute_dtype)
    if causal:
        row_idx = torch.arange(Sq, device=q.device)[:, None]
        col_idx = torch.arange(Sk, device=q.device)[None, :]
        mask = row_idx >= col_idx
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (B, H, Sq)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vf).transpose(1, 2).contiguous()  # (B, Sq, H, D)
    return out, lse

def test_unmodified(work_dir):
    sys.path.insert(0, work_dir)
    import orig_flash_attn_triton as orig_triton

    print("=== 1. Test unmodified flash_attn_triton.py ===")
    q32 = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    for dt_name, q_in in [("float32", q32), ("float16", q32.half())]:
        try:
            orig_triton._flash_attn_forward(q_in, q_in, q_in, causal=False)
            print(f"  unmodified flash_attn_triton ({dt_name}): SUCCEEDED unexpectedly")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"  unmodified flash_attn_triton ({dt_name}): {type(e).__name__} at line {tb.lineno}: {e}")

    print("=== 2. Test unmodified flash_attn_triton_og.py ===")
    try:
        import orig_flash_attn_triton_og as orig_og
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"  unmodified import of flash_attn_triton_og: {type(e).__name__} at line {tb.lineno}: {e}")
        import types
        sys.modules["pytest"] = types.ModuleType("pytest")
        import orig_flash_attn_triton_og as orig_og

    # flash_attn_triton_og expects shape (Z, H, N_CTX, D)
    q_og = torch.randn(1, 2, 128, 64, dtype=torch.float16)
    try:
        out_og = orig_og.attention(q_og, q_og, q_og, 1.0 / math.sqrt(64))
        print(f"  flash_attn_triton_og (unmodified kernel, fp16): SUCCEEDED unexpectedly")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"  flash_attn_triton_og (unmodified kernel, fp16): {type(e).__name__} at line {tb.lineno}: {e}")

    import host_patched_flash_attn_triton as host_triton
    try:
        host_triton._flash_attn_forward(q32, q32, q32, causal=False)
        print("  host_patched_flash_attn_triton (unmodified _fwd_kernel): SUCCEEDED unexpectedly")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"  host_patched_flash_attn_triton (unmodified _fwd_kernel): {type(e).__name__} at line {tb.lineno}: {e}")

    import patched_flash_attn_triton_og as p_og
    out_og = p_og.attention(q_og, q_og, q_og, 1.0 / math.sqrt(64))
    q_bshd = q_og.transpose(1, 2).contiguous()
    ref_og, _ = attention_ref_exact(q_bshd, q_bshd, q_bshd, causal=True, compute_dtype=torch.float64)
    ref_og_zhsd = ref_og.transpose(1, 2)
    diff_og = (out_og.double() - ref_og_zhsd).abs().max().item()
    rel_og = diff_og / (ref_og_zhsd.abs().max().item() + 1e-12)
    print(f"  patched_flash_attn_triton_og (fp16, Z=1,H=2,N=128,D=64): max_abs_diff={diff_og:.3e} max_rel_diff={rel_og:.3e}")

def test_patched(work_dir):
    sys.path.insert(0, work_dir)
    import patched_flash_attn_triton as p_triton

    print("=== 3. Test patched_flash_attn_triton._flash_attn_forward ===")
    torch.manual_seed(42)
    shapes = [
        (1, 64, 2, 32),
        (1, 64, 2, 64),
        (1, 113, 2, 32),
        (1, 113, 2, 64),
        (1, 128, 2, 32),
        (1, 128, 2, 64),
        (1, 192, 2, 32),
        (1, 256, 2, 32),
        (1, 256, 2, 64),
    ]
    for dtype in [torch.float32, torch.float16]:
        dt_str = "fp32" if dtype == torch.float32 else "fp16"
        max_abs_all = 0.0
        max_rel_all = 0.0
        for (B, N, H, D) in shapes:
            for causal in [False, True]:
                q = torch.randn(B, N, H, D, dtype=dtype)
                k = torch.randn(B, N, H, D, dtype=dtype)
                v = torch.randn(B, N, H, D, dtype=dtype)
                try:
                    out, lse, _ = p_triton._flash_attn_forward(q, k, v, bias=None, causal=causal)
                    ref64, lse64 = attention_ref_exact(q, k, v, causal=causal, compute_dtype=torch.float64)
                    ref32, _ = attention_ref_exact(q, k, v, causal=causal, compute_dtype=torch.float32)
                    abs_d64 = (out.double() - ref64).abs().max().item()
                    rel_d64 = (abs_d64 / (ref64.abs().max().item() + 1e-12))
                    abs_d32 = (out.float() - ref32).abs().max().item()
                    lse_diff = (lse[:, :, :N].double() - lse64).abs().max().item()
                    max_abs_all = max(max_abs_all, abs_d64)
                    max_rel_all = max(max_rel_all, rel_d64)
                    print(f"  [{dt_str}] B={B} N={N:3d} H={H} D={D:2d} causal={str(causal):5s} | "
                          f"abs_vs_f64={abs_d64:.3e} rel_vs_f64={rel_d64:.3e} abs_vs_f32={abs_d32:.3e} lse_diff={lse_diff:.3e}")
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    print(f"  [{dt_str}] B={B} N={N:3d} H={H} D={D:2d} causal={str(causal):5s} | "
                          f"FAIL {type(e).__name__} at line {tb.lineno}: {e}")
        print(f"  --> Summary [{dt_str}]: max_abs_vs_f64={max_abs_all:.3e} max_rel_vs_f64={max_rel_all:.3e}")

    print("=== 4. Negative controls on real Triton _fwd_kernel ===")
    q = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    k = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    v = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    out_causal, _, _ = p_triton._flash_attn_forward(q, k, v, causal=True)
    ref_noncausal, _ = attention_ref_exact(q, k, v, causal=False)
    nc1_diff = (out_causal.double() - ref_noncausal).abs().max().item()
    out_badscale, _, _ = p_triton._flash_attn_forward(q, k, v, causal=False, softmax_scale=1.0)
    nc2_diff = (out_badscale.double() - ref_noncausal).abs().max().item()
    print(f"  [neg_ctrl_1] causal kernel vs non-causal ref: max_abs_diff={nc1_diff:.3e} (expected > 0.1)")
    print(f"  [neg_ctrl_2] softmax_scale=1.0 vs 1/sqrt(64): max_abs_diff={nc2_diff:.3e} (expected > 0.1)")

if __name__ == "__main__":
    work_dir = sys.argv[1]
    test_unmodified(work_dir)
    test_patched(work_dir)
