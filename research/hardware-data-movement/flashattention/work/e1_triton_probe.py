import os
import sys
import traceback

# Record versions and environment
print("=== E1: Triton Interpreter Probe on CPU ===")
print(f"Python: {sys.version.split()[0]}")
print(f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}")

import numpy as np
import torch
import triton
import triton.language as tl

print(f"numpy version: {np.__version__}")
print(f"torch version: {torch.__version__}")
print(f"triton version: {triton.__version__}")
print(f"triton path: {triton.__file__}")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run_tests():
    torch.manual_seed(42)
    # 1. Vector add (including non-multiple of BLOCK_SIZE: n=1000, BLOCK_SIZE=256)
    for n in [1024, 1000]:
        x = torch.randn(n, dtype=torch.float32, device="cpu")
        y = torch.randn(n, dtype=torch.float32, device="cpu")
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 256),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
        ref = x + y
        max_abs = (out - ref).abs().max().item()
        print(f"[vec_add] n={n} max_abs_diff={max_abs:.3e} exact={torch.equal(out, ref)}")

    # 2. Matmul with tl.dot across dtypes
    for dtype_name, dt in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        try:
            M, N, K = 64, 64, 64
            a = torch.randn((M, K), dtype=dt, device="cpu")
            b = torch.randn((K, N), dtype=dt, device="cpu")
            c = torch.empty((M, N), dtype=torch.float32, device="cpu")
            grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
            )
            ref = torch.matmul(a.float(), b.float())
            max_abs = (c - ref).abs().max().item()
            max_rel = ((c - ref).abs() / (ref.abs() + 1e-8)).max().item()
            print(f"[matmul tl.dot] dtype={dtype_name} shape=({M},{N},{K}) max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")
        except Exception as e:
            print(f"[matmul tl.dot] dtype={dtype_name} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if "--neg-control" in sys.argv:
        x = torch.ones(16, dtype=torch.float32, device="cpu")
        y = torch.ones(16, dtype=torch.float32, device="cpu")
        out = torch.empty_like(x)
        try:
            add_kernel[(1,)](x, y, out, 16, BLOCK_SIZE=16)
            print("NEG_CONTROL_UNEXPECTED_PASS")
        except Exception as e:
            print(f"NEG_CONTROL_EXPECTED_FAIL: {type(e).__name__}: {e}")
    else:
        run_tests()

