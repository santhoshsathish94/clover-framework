import os
import math
import sys
import traceback

os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
import triton.runtime.interpreter as t_interp

# --- Hook Triton interpreter's low-level load/store/atomic_rmw ---
IO_STATS = {"loads": 0, "load_elems": 0, "stores": 0, "store_elems": 0, "atomics": 0, "atomic_elems": 0}

def reset_io_stats():
    for k in IO_STATS:
        IO_STATS[k] = 0

_orig_load = t_interp.interpreter_builder.create_masked_load
_orig_store = t_interp.interpreter_builder.create_masked_store
_orig_atomic = getattr(t_interp.interpreter_builder, "create_atomic_rmw", None)

def _hooked_load(ptr, mask, other, cache_modifier, eviction_policy, is_volatile):
    IO_STATS["loads"] += 1
    IO_STATS["load_elems"] += int(ptr.data.size)
    return _orig_load(ptr, mask, other, cache_modifier, eviction_policy, is_volatile)

def _hooked_store(ptr, val, mask, cache_modifier, eviction_policy):
    IO_STATS["stores"] += 1
    IO_STATS["store_elems"] += int(val.data.size)
    return _orig_store(ptr, val, mask, cache_modifier, eviction_policy)

t_interp.interpreter_builder.create_masked_load = _hooked_load
t_interp.interpreter_builder.create_masked_store = _hooked_store

if _orig_atomic is not None:
    def _hooked_atomic(op, ptr, val, mask, *args, **kwargs):
        IO_STATS["atomics"] += 1
        IO_STATS["atomic_elems"] += int(val.data.size)
        return _orig_atomic(op, ptr, val, mask, *args, **kwargs)
    t_interp.interpreter_builder.create_atomic_rmw = _hooked_atomic

def attention_autograd_ref(q, k, v, do, causal=False, compute_dtype=torch.float64):
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    qd = q.detach().to(compute_dtype).requires_grad_(True)
    kd = k.detach().to(compute_dtype).requires_grad_(True)
    vd = v.detach().to(compute_dtype).requires_grad_(True)
    qf = qd.transpose(1, 2)
    kf = kd.transpose(1, 2)
    vf = vd.transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        row_idx = torch.arange(Sq, device=q.device)[:, None]
        col_idx = torch.arange(Sk, device=q.device)[None, :]
        scores = scores.masked_fill(~(row_idx >= col_idx), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vf).transpose(1, 2).contiguous()
    out.backward(do.to(compute_dtype))
    return out, qd.grad, kd.grad, vd.grad

@triton.jit
def _calib_kernel(X, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(Y + offs, x)

def calibrate_counter():
    x = torch.zeros(32, dtype=torch.float32)
    y = torch.zeros(32, dtype=torch.float32)
    reset_io_stats()
    _calib_kernel[(2,)](x, y, BLOCK=16)
    print(f"=== 0. Counter calibration (2 blocks of 16): {IO_STATS} ===")
    assert IO_STATS["loads"] == 2 and IO_STATS["load_elems"] == 32
    assert IO_STATS["stores"] == 2 and IO_STATS["store_elems"] == 32

def main(work_dir):
    calibrate_counter()
    sys.path.insert(0, work_dir)
    import patched_flash_attn_triton as p_e2

    print("=== 1. Test E2-patched _flash_attn_backward (with @triton.autotune 2 configs) ===")
    torch.manual_seed(0)
    q = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    k = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    v = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    do = torch.randn(1, 128, 2, 64, dtype=torch.float32)
    out, lse, _ = p_e2._flash_attn_forward(q, k, v, causal=False)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    try:
        p_e2._flash_attn_backward(do, q, k, v, out, lse, dq, dk, dv, causal=False)
        print("  E2-patched _flash_attn_backward: SUCCEEDED unexpectedly")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"  E2-patched _flash_attn_backward failed at {tb.filename}:{tb.lineno} ({tb.name}): {type(e).__name__}: {e}")

    import bwd_patched_flash_attn_triton as p_bwd
    print("=== 2. Test backward pass accuracy across shapes, dtypes, causal, SEQUENCE_PARALLEL ===")
    shapes = [(1, 64, 2, 32), (1, 64, 2, 64), (1, 113, 2, 32), (1, 113, 2, 64), (1, 128, 2, 32), (1, 128, 2, 64), (1, 256, 2, 64)]
    for dtype in [torch.float32, torch.float16]:
        dt_str = "fp32" if dtype == torch.float32 else "fp16"
        for seq_par in [False, True]:
            max_dq, max_dk, max_dv = 0.0, 0.0, 0.0
            for (B, N, H, D) in shapes:
                for causal in [False, True]:
                    q = torch.randn(B, N, H, D, dtype=dtype)
                    k = torch.randn(B, N, H, D, dtype=dtype)
                    v = torch.randn(B, N, H, D, dtype=dtype)
                    do = torch.randn(B, N, H, D, dtype=dtype)
                    out, lse, _ = p_bwd.run_fwd_With_blocks(q, k, v, causal=causal, block_m=128, block_n=128)
                    dq, dk, dv = p_bwd.run_bwd_with_blocks(do, q, k, v, out, lse, causal=causal, block_m=128, block_n=128, seq_par=seq_par)
                    _, ref_dq, ref_dk, ref_dv = attention_autograd_ref(q, k, v, do, causal=causal, compute_dtype=torch.float64)
                    edq = (dq.double() - ref_dq).abs().max().item()
                    edk = (dk.double() - ref_dk).abs().max().item()
                    edv = (dv.double() - ref_dv).abs().max().item()
                    max_dq, max_dk, max_dv = max(max_dq, edq), max(max_dk, edk), max(max_dv, edv)
                    if dtype == torch.float32 and not seq_par:
                        print(f"  [{dt_str} seq_par={seq_par}] N={N:3d} D={D:2d} causal={str(causal):5s} | "
                              f"dQ={edq:.3e} dK={edk:.3e} dV={edv:.3e}")
            print(f"  --> Summary [{dt_str} seq_par={seq_par}]: max_abs dQ={max_dq:.3e} dK={max_dk:.3e} dV={max_dv:.3e}")

    print("=== 3. Sweep (BLOCK_M, BLOCK_N) and count real tl.load / tl.store in interpreter (B=1, H=1, D=64) ===")
    for N in [64, 113, 128, 256]:
        for (bm, bn) in [(128, 128), (64, 64), (64, 128), (128, 64), (32, 32)]:
            for causal in [False, True]:
                q = torch.randn(1, N, 1, 64, dtype=torch.float32)
                k = torch.randn(1, N, 1, 64, dtype=torch.float32)
                v = torch.randn(1, N, 1, 64, dtype=torch.float32)
                do = torch.randn(1, N, 1, 64, dtype=torch.float32)
                reset_io_stats()
                out, lse, _ = p_bwd.run_fwd_With_blocks(q, k, v, causal=causal, block_m=bm, block_n=bn)
                fwd_stats = dict(IO_STATS)
                reset_io_stats()
                dq, dk, dv = p_bwd.run_bwd_with_blocks(do, q, k, v, out, lse, causal=causal, block_m=bm, block_n=bn, seq_par=False)
                bwd_stats = dict(IO_STATS)
                ref_out, ref_dq, ref_dk, ref_dv = attention_autograd_ref(q, k, v, do, causal=causal, compute_dtype=torch.float64)
                e_out = (out.double() - ref_out).abs().max().item()
                e_dq = (dq.double() - ref_dq).abs().max().item()
                e_dk = (dk.double() - ref_dk).abs().max().item()
                e_dv = (dv.double() - ref_dv).abs().max().item()
                if N in [113, 128, 256] and (bm, bn) in [(128, 128), (64, 64), (32, 32)]:
                    print(f"  N={N:3d} BM={bm:3d} BN={bn:3d} causal={str(causal):5s} | "
                          f"FWD loads={fwd_stats['loads']:3d} ({fwd_stats['load_elems']:6d} el) stores={fwd_stats['stores']:2d} ({fwd_stats['store_elems']:5d} el) | "
                          f"BWD loads={bwd_stats['loads']:3d} ({bwd_stats['load_elems']:6d} el) stores={bwd_stats['stores']:2d} ({bwd_stats['store_elems']:5d} el) | "
                          f"err out={e_out:.2e} dQ={e_dq:.2e} dK={e_dk:.2e} dV={e_dv:.2e}")

    print("=== 4. Negative control on real Triton _bwd_kernel (corrupt D_i = rowsum(dO * O) to zero) ===")
    q = torch.randn(1, 128, 1, 64, dtype=torch.float32)
    k = torch.randn(1, 128, 1, 64, dtype=torch.float32)
    v = torch.randn(1, 128, 1, 64, dtype=torch.float32)
    do = torch.randn(1, 128, 1, 64, dtype=torch.float32)
    out, lse, _ = p_bwd.run_fwd_With_blocks(q, k, v, causal=False, block_m=64, block_n=64)
    dq_bad, dk_bad, dv_bad = p_bwd.run_bwd_with_blocks(do, q, k, v, out, lse, causal=False, block_m=64, block_n=64, seq_par=False, zero_delta=True)
    _, ref_dq, ref_dk, ref_dv = attention_autograd_ref(q, k, v, do, causal=False, compute_dtype=torch.float64)
    print(f"  [neg_ctrl_bwd] zero_delta=True: dQ_err={(dq_bad.double()-ref_dq).abs().max().item():.3e} "
          f"dK_err={(dk_bad.double()-ref_dk).abs().max().item():.3e} "
          f"dV_err={(dv_bad.double()-ref_dv).abs().max().item():.3e} (dQ, dK must fail > 0.1; dV unaffected)")

if __name__ == "__main__":
    main(sys.argv[1])
