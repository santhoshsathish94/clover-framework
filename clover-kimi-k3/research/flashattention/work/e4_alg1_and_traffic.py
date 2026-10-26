import math
import torch

class HBMCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mat_reads = 0
        self.mat_writes = 0
        self.vec_reads = 0
        self.vec_writes = 0
        self.init_writes = 0

    def read_mat(self, t):
        self.mat_reads += t.numel()
        return t.clone()

    def write_mat(self, dst, src, is_init=False):
        if is_init:
            self.init_writes += src.numel()
        else:
            self.mat_writes += src.numel()
        dst.copy_(src)

    def read_vec(self, t):
        self.vec_reads += t.numel()
        return t.clone()

    def write_vec(self, dst, src, is_init=False):
        if is_init:
            self.init_writes += src.numel()
        else:
            self.vec_writes += src.numel()
        dst.copy_(src)

    @property
    def total_mat(self):
        return self.mat_reads + self.mat_writes

    @property
    def total_all_with_init(self):
        return self.mat_reads + self.mat_writes + self.vec_reads + self.vec_writes + self.init_writes


def standard_attention_counted(Q, K, V, causal=False, counter=None):
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    if counter is None:
        counter = HBMCounter()
    # Pass 1: Read Q, K from HBM, compute S, write S to HBM
    q_sram = counter.read_mat(Q)
    k_sram = counter.read_mat(K)
    S_sram = (q_sram @ k_sram.T) * scale
    if causal:
        idx = torch.arange(N, device=Q.device)
        S_sram = S_sram.masked_fill(idx[:, None] < idx[None, :], float("-inf"))
    S_hbm = torch.empty((N, N), dtype=Q.dtype, device=Q.device)
    counter.write_mat(S_hbm, S_sram)

    # Pass 2: Read S from HBM, compute P = softmax(S), write P to HBM
    s_in = counter.read_mat(S_hbm)
    P_sram = torch.softmax(s_in, dim=-1)
    P_hbm = torch.empty((N, N), dtype=Q.dtype, device=Q.device)
    counter.write_mat(P_hbm, P_sram)

    # Pass 3: Read P, V from HBM, compute O = P V, write O to HBM
    p_in = counter.read_mat(P_hbm)
    v_sram = counter.read_mat(V)
    O_sram = p_in @ v_sram
    O_hbm = torch.empty((N, d), dtype=Q.dtype, device=Q.device)
    counter.write_mat(O_hbm, O_sram)
    return O_hbm, counter


def flash_attention_alg1_counted(Q, K, V, M_sram, causal=False, counter=None, drop_last_outer=False):
    """
    Paper 1 Algorithm 1 (2205.14135/src/algo.tex lines 78-107):
    B_c = ceil(M / (4d)), B_r = min(B_c, d)
    Outer loop j = 1..T_c over K_j, V_j; inner loop i = 1..T_r over Q_i, O_i, l_i, m_i.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    B_c = math.ceil(M_sram / (4 * d))
    B_r = min(B_c, d)
    T_c = math.ceil(N / B_c)
    T_r = math.ceil(N / B_r)
    if counter is None:
        counter = HBMCounter()

    O_hbm = torch.empty((N, d), dtype=Q.dtype, device=Q.device)
    l_hbm = torch.empty((N,), dtype=Q.dtype, device=Q.device)
    m_hbm = torch.empty((N,), dtype=Q.dtype, device=Q.device)
    # Line 3: Initialize O = 0, l = 0, m = -inf in HBM
    counter.write_mat(O_hbm, torch.zeros_like(O_hbm), is_init=True)
    counter.write_vec(l_hbm, torch.zeros_like(l_hbm), is_init=True)
    counter.write_vec(m_hbm, torch.full_like(m_hbm, float("-inf")), is_init=True)

    outer_limit = max(1, T_c - 1) if drop_last_outer else T_c
    for j in range(outer_limit):
        c0, c1 = j * B_c, min((j + 1) * B_c, N)
        # Line 6: Load K_j, V_j from HBM to on-chip SRAM
        Kj = counter.read_mat(K[c0:c1])
        Vj = counter.read_mat(V[c0:c1])
        for i in range(T_r):
            r0, r1 = i * B_r, min((i + 1) * B_r, N)
            # Line 8: Load Q_i, O_i, l_i, m_i from HBM to on-chip SRAM
            Qi = counter.read_mat(Q[r0:r1])
            Oi = counter.read_mat(O_hbm[r0:r1])
            li = counter.read_vec(l_hbm[r0:r1])
            mi = counter.read_vec(m_hbm[r0:r1])

            # Line 9: On chip, compute S_ij = tau Q_i K_j^T (plus causal mask if causal)
            Sij = (Qi @ Kj.T) * scale
            if causal:
                r_idx = torch.arange(r0, r1, device=Q.device)[:, None]
                c_idx = torch.arange(c0, c1, device=Q.device)[None, :]
                Sij = Sij.masked_fill(r_idx < c_idx, float("-inf"))

            # Line 10: m_tilde_ij = rowmax(S_ij), P_tilde_ij = exp(S_ij - m_tilde_ij), l_tilde_ij = rowsum(P_tilde_ij)
            m_tilde = Sij.max(dim=-1).values
            # Handle rows where all keys in this block are causally masked (-inf)
            all_neg_inf = torch.isinf(m_tilde) & (m_tilde < 0)
            m_tilde_safe = torch.where(all_neg_inf, torch.zeros_like(m_tilde), m_tilde)
            P_tilde = torch.where(all_neg_inf[:, None], torch.zeros_like(Sij), torch.exp(Sij - m_tilde_safe[:, None]))
            l_tilde = P_tilde.sum(dim=-1)

            # Line 11: m_new = max(m_i, m_tilde_ij), l_new = exp(m_i - m_new)*l_i + exp(m_tilde_ij - m_new)*l_tilde_ij
            m_new = torch.maximum(mi, m_tilde)
            m_new_safe = torch.where(torch.isinf(m_new) & (m_new < 0), torch.zeros_like(m_new), m_new)
            alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(mi), torch.exp(mi - m_new_safe))
            beta = torch.where(all_neg_inf, torch.zeros_like(m_tilde), torch.exp(m_tilde - m_new_safe))
            l_new = alpha * li + beta * l_tilde

            # Line 12: Write O_i <- diag(l_new)^(-1) (diag(l_i)*exp(m_i - m_new)*O_i + exp(m_tilde_ij - m_new)*P_tilde_ij V_j)
            denom = torch.where(l_new == 0, torch.ones_like(l_new), l_new)
            Oi_new = ((li * alpha)[:, None] * Oi + (beta[:, None] * P_tilde) @ Vj) / denom[:, None]

            counter.write_mat(O_hbm[r0:r1], Oi_new)
            counter.write_vec(l_hbm[r0:r1], l_new)
            counter.write_vec(m_hbm[r0:r1], m_new)

    return O_hbm, l_hbm, m_hbm, counter, (B_c, B_r, T_c, T_r)


def main():
    torch.manual_seed(2026)
    print("=== 1. Hand-countable verification: N=4, d=2 at M=16 and M=8 ===")
    Q4 = torch.randn(4, 2, dtype=torch.float64)
    K4 = torch.randn(4, 2, dtype=torch.float64)
    V4 = torch.randn(4, 2, dtype=torch.float64)
    O_std, c_std = standard_attention_counted(Q4, K4, V4, causal=False)
    print(f"  [Std N=4,d=2] mat_reads={c_std.mat_reads} mat_writes={c_std.mat_writes} total_mat={c_std.total_mat} (hand: 56, 40, 96)")
    assert (c_std.mat_reads, c_std.mat_writes, c_std.total_mat) == (56, 40, 96)

    for M_hand, exp_qkvo, exp_all in [(16, 64, 112), (8, 112, 192)]:
        O_fa, _, _, c_fa, tiles = flash_attention_alg1_counted(Q4, K4, V4, M_sram=M_hand, causal=False)
        diff = (O_fa - O_std).abs().max().item()
        print(f"  [FA1 N=4,d=2 M={M_hand:2d}] B_c={tiles[0]} B_r={tiles[1]} T_c={tiles[2]} T_r={tiles[3]} | "
              f"mat_reads={c_fa.mat_reads} mat_writes={c_fa.mat_writes} total_qkvo={c_fa.total_mat} (hand: {exp_qkvo}) | "
              f"total_all={c_fa.total_all_with_init} (hand: {exp_all}) | max_diff={diff:.2e}")
        assert c_fa.total_mat == exp_qkvo and c_fa.total_all_with_init == exp_all

    # Negative control on counter + algorithm: drop last outer iteration at M=16
    O_bad, _, _, c_bad, _ = flash_attention_alg1_counted(Q4, K4, V4, M_sram=16, causal=False, drop_last_outer=True)
    print(f"  [NegCtrl drop_last_outer N=4,d=2,M=16] total_qkvo={c_bad.total_mat} (!= 64) max_diff={(O_bad - O_std).abs().max().item():.3e} (must fail)")

    print("=== 2. Exactness sweep across shapes (N in 64..2048, d in 32/64/128, causal, M) ===")
    Ns = [64, 113, 128, 256, 512, 517, 1024, 2048]
    ds = [32, 64, 128]
    Ms = [1024, 3000, 4096, 16384, 65536]
    max_abs = 0.0
    max_rel = 0.0
    count_cfg = 0
    for N in Ns:
        for d in ds:
            Q = torch.randn(N, d, dtype=torch.float64)
            K = torch.randn(N, d, dtype=torch.float64)
            V = torch.randn(N, d, dtype=torch.float64)
            for causal in [False, True]:
                O_ref, _ = standard_attention_counted(Q, K, V, causal=causal)
                for M in Ms:
                    Bc_est = math.ceil(M / (4 * d))
                    Br_est = min(Bc_est, d)
                    if math.ceil(N / Bc_est) * math.ceil(N / Br_est) > 4096:
                        continue
                    O_fa, _, _, _, _ = flash_attention_alg1_counted(Q, K, V, M_sram=M, causal=causal)
                    ad = (O_fa - O_ref).abs().max().item()
                    rd = ad / (O_ref.abs().max().item() + 1e-12)
                    max_abs = max(max_abs, ad)
                    max_rel = max(max_rel, rd)
                    count_cfg += 1
            if N in [113, 517, 2048] and d == 64:
                print(f"  checked N={N:4d} d={d:3d} (causal/non-causal): running max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    print(f"  --> Exactness summary across {count_cfg} configs: max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")

    print("=== 3. Traffic ratio and crossover M at N=4096, d=64 (counted by HBMCounter) ===")
    N_big, d_big = 4096, 64
    Q_b = torch.randn(N_big, d_big, dtype=torch.float64)
    K_b = torch.randn(N_big, d_big, dtype=torch.float64)
    V_b = torch.randn(N_big, d_big, dtype=torch.float64)
    O_std_b, c_std_b = standard_attention_counted(Q_b, K_b, V_b, causal=False)
    print(f"  Standard attention (N=4096, d=64): E_std = {c_std_b.total_mat:,} elements (reads={c_std_b.mat_reads:,}, writes={c_std_b.mat_writes:,})")
    for M_test in [8192, 12032, 12033, 12288, 12289, 16384, 32768, 65536, 98304, 131072]:
        O_f, _, _, c_f, (Bc, Br, Tc, Tr) = flash_attention_alg1_counted(Q_b, K_b, V_b, M_sram=M_test, causal=False)
        ratio_qkvo = c_std_b.total_mat / c_f.total_mat
        ratio_all = c_std_b.total_mat / c_f.total_all_with_init
        err = (O_f - O_std_b).abs().max().item()
        print(f"  M={M_test:6d} | B_c={Bc:3d} B_r={Br:2d} T_c={Tc:3d} T_r={Tr:3d} | "
              f"E_qkvo={c_f.total_mat:10,d} (ratio={ratio_qkvo:8.6f}x) | "
              f"E_all={c_f.total_all_with_init:10,d} (ratio={ratio_all:8.6f}x) | err={err:.2e}")

if __name__ == "__main__":
    main()
