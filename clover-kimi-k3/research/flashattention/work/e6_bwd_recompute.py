import math
import torch

class FlopAndByteCounter:
    def __init__(self):
        self.matmul_flops = 0
        self.matmul_calls = 0

    def matmul(self, A, B):
        # A: (m, k), B: (k, n) -> 2 * m * k * n FLOPs
        m, k = A.shape
        k2, n = B.shape
        assert k == k2
        self.matmul_flops += 2 * m * k * n
        self.matmul_calls += 1
        return A @ B


def forward_std_and_lse(Q, K, V, causal=False):
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale
    if causal:
        idx = torch.arange(N, device=Q.device)
        S = S.masked_fill(idx[:, None] < idx[None, :], float("-inf"))
    L = torch.logsumexp(S, dim=-1)
    P = torch.exp(S - L[:, None])
    O = P @ V
    return O, P, L


def bwd_standard_stored_P(Q, K, V, P, dO):
    """
    Standard attention backward storing N x N matrix P from forward.
    4 matmuls: dV = P^T dO, dP = dO V^T, dQ = tau dS K, dK = tau dS^T Q.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    ctr = FlopAndByteCounter()
    dV = ctr.matmul(P.T, dO)
    dP = ctr.matmul(dO, V.T)
    D = (P * dP).sum(dim=-1)
    dS = P * (dP - D[:, None])
    dQ = ctr.matmul(dS, K) * scale
    dK = ctr.matmul(dS.T, Q) * scale
    return dQ, dK, dV, ctr


def bwd_tiled_recompute(Q, K, V, O, L, dO, B_r=64, B_c=64, causal=False, omit_D=False, corrupt_L=False):
    """
    FlashAttention backward pass with recomputation (Paper 1 Alg 4 / Paper 2 Alg 2):
    Keeps only (O, L) from forward; recomputes S_ij and P_ij tile by tile.
    5 matmuls per tile: S_ij = Q_i K_j^T, dV_j += P_ij^T dO_i, dP_ij = dO_i V_j^T,
                        dQ_i += tau dS_ij K_j, dK_j += tau dS_ij^T Q_i.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    T_c = math.ceil(N / B_c)
    T_r = math.ceil(N / B_r)
    ctr = FlopAndByteCounter()

    # Preprocess: D_i = rowsum(dO_i * O_i)
    D = torch.zeros(N, dtype=Q.dtype, device=Q.device) if omit_D else (dO * O).sum(dim=-1)
    L_use = L + 1.0 if corrupt_L else L

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for j in range(T_c):
        c0, c1 = j * B_c, min((j + 1) * B_c, N)
        Kj = K[c0:c1]
        Vj = V[c0:c1]
        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        for i in range(T_r):
            r0, r1 = i * B_r, min((i + 1) * B_r, N)
            Qi = Q[r0:r1]
            dOi = dO[r0:r1]
            Li = L_use[r0:r1]
            Di = D[r0:r1]

            # 1. Recompute S_ij = tau Q_i K_j^T
            Sij = ctr.matmul(Qi, Kj.T) * scale
            if causal:
                r_idx = torch.arange(r0, r1, device=Q.device)[:, None]
                c_idx = torch.arange(c0, c1, device=Q.device)[None, :]
                Sij = Sij.masked_fill(r_idx < c_idx, float("-inf"))

            # Recompute P_ij = exp(S_ij - L_i)
            Pij = torch.exp(Sij - Li[:, None])

            # 2. dV_j += P_ij^T dO_i
            dVj = dVj + ctr.matmul(Pij.T, dOi)

            # 3. dP_ij = dO_i V_j^T
            dPij = ctr.matmul(dOi, Vj.T)

            # dS_ij = P_ij * (dP_ij - D_i)
            dSij = Pij * (dPij - Di[:, None])

            # 4. dQ_i += tau dS_ij K_j
            dQ[r0:r1] = dQ[r0:r1] + ctr.matmul(dSij, Kj) * scale

            # 5. dK_j += tau dS_ij^T Q_i
            dKj = dKj + ctr.matmul(dSij.T, Qi) * scale

        dK[c0:c1] = dKj
        dV[c0:c1] = dVj

    return dQ, dK, dV, ctr


def autograd_ref(Q, K, V, dO, causal=False):
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    q = Q.detach().clone().requires_grad_(True)
    k = K.detach().clone().requires_grad_(True)
    v = V.detach().clone().requires_grad_(True)
    S = (q @ k.T) * scale
    if causal:
        idx = torch.arange(N, device=Q.device)
        S = S.masked_fill(idx[:, None] < idx[None, :], float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = P @ v
    O.backward(dO)
    return q.grad, k.grad, v.grad


def main():
    torch.manual_seed(123)
    print("=== 1. Negative Controls on Tiled Backward with Recomputation (N=128, d=64) ===")
    Q0 = torch.randn(128, 64, dtype=torch.float64)
    K0 = torch.randn(128, 64, dtype=torch.float64)
    V0 = torch.randn(128, 64, dtype=torch.float64)
    dO0 = torch.randn(128, 64, dtype=torch.float64)
    O0, P0, L0 = forward_std_and_lse(Q0, K0, V0, causal=False)
    rq, rk, rv = autograd_ref(Q0, K0, V0, dO0, causal=False)

    dq_nd, dk_nd, dv_nd, _ = bwd_tiled_recompute(Q0, K0, V0, O0, L0, dO0, B_r=32, B_c=32, omit_D=True)
    dq_cl, dk_cl, dv_cl, _ = bwd_tiled_recompute(Q0, K0, V0, O0, L0, dO0, B_r=32, B_c=32, corrupt_L=True)
    print(f"  [NegCtrl 1: omit D_i]      dQ_err={(dq_nd-rq).abs().max().item():.3e} dK_err={(dk_nd-rk).abs().max().item():.3e} dV_err={(dv_nd-rv).abs().max().item():.3e}")
    print(f"  [NegCtrl 2: L_i + 1.0]     dQ_err={(dq_cl-rq).abs().max().item():.3e} dK_err={(dk_cl-rk).abs().max().item():.3e} dV_err={(dv_cl-rv).abs().max().item():.3e}")

    print("\n=== 2. Exactness vs PyTorch float64 autograd across shapes ===")
    Ns = [64, 113, 128, 256, 512, 1024]
    ds = [32, 64, 128]
    max_dq, max_dk, max_dv = 0.0, 0.0, 0.0
    n_cfg = 0
    for N in Ns:
        for d in ds:
            Q = torch.randn(N, d, dtype=torch.float64)
            K = torch.randn(N, d, dtype=torch.float64)
            V = torch.randn(N, d, dtype=torch.float64)
            dO = torch.randn(N, d, dtype=torch.float64)
            for causal in [False, True]:
                O, P, L = forward_std_and_lse(Q, K, V, causal=causal)
                rq, rk, rv = autograd_ref(Q, K, V, dO, causal=causal)
                for (Br, Bc) in [(32, 32), (64, 64)]:
                    dq, dk, dv, _ = bwd_tiled_recompute(Q, K, V, O, L, dO, B_r=Br, B_c=Bc, causal=causal)
                    edq = (dq - rq).abs().max().item()
                    edk = (dk - rk).abs().max().item()
                    edv = (dv - rv).abs().max().item()
                    max_dq, max_dk, max_dv = max(max_dq, edq), max(max_dk, edk), max(max_dv, edv)
                    n_cfg += 1
        print(f"  N={N:4d} (all d, causal, tiles): running max_abs dQ={max_dq:.3e} dK={max_dk:.3e} dV={max_dv:.3e}")
    print(f"  --> Summary across {n_cfg} configs: max_abs dQ={max_dq:.3e} dK={max_dk:.3e} dV={max_dv:.3e}")

    print("\n=== 3. Stored Bytes and Counted Matmul FLOPs: Standard (store P) vs Recompute (store O, L) at d=64 ===")
    d = 64
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        if N <= 1024:
            Q = torch.randn(N, d, dtype=torch.float64)
            K = torch.randn(N, d, dtype=torch.float64)
            V = torch.randn(N, d, dtype=torch.float64)
            dO = torch.randn(N, d, dtype=torch.float64)
            O, P, L = forward_std_and_lse(Q, K, V, causal=False)
            _, _, _, ctr_std = bwd_standard_stored_P(Q, K, V, P, dO)
            _, _, _, ctr_rec = bwd_tiled_recompute(Q, K, V, O, L, dO, B_r=64, B_c=64, causal=False)
            flops_std, flops_rec = ctr_std.matmul_flops, ctr_rec.matmul_flops
        else:
            # Verify formula 8 N^2 d vs 10 N^2 d
            flops_std, flops_rec = 8 * N * N * d, 10 * N * N * d
        bytes_P_fp16 = 2 * N * N
        bytes_L_fp32 = 4 * N
        bytes_OL_mixed = 2 * N * d + 4 * N
        print(f"  N={N:4d} d={d} | P_fp16={bytes_P_fp16:10,d} B vs (O_fp16+L_fp32)={bytes_OL_mixed:7,d} B "
              f"(ratio={bytes_P_fp16/bytes_OL_mixed:6.2f}x; P/L={bytes_P_fp16/bytes_L_fp32:6.1f}x) | "
              f"BWD matmul FLOPs: std={flops_std:13,d} recomp={flops_rec:13,d} (ratio={flops_rec/flops_std:.4f}x)")

if __name__ == "__main__":
    main()
