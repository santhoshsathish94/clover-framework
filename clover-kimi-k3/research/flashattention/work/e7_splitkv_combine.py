import math
import torch

def run_single_pass_ref(Q, K, V, causal=False, causal_offset=0):
    Nq, d = Q.shape
    Nk = K.shape[0]
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale
    if causal:
        r_idx = torch.arange(Nq, device=Q.device)[:, None] + causal_offset
        c_idx = torch.arange(Nk, device=Q.device)[None, :]
        S = S.masked_fill(c_idx > r_idx, float("-inf"))
    L = torch.logsumexp(S, dim=-1)
    all_inf = torch.isinf(L) & (L < 0)
    L_safe = torch.where(all_inf, torch.zeros_like(L), L)
    P = torch.where(all_inf[:, None], torch.zeros_like(S), torch.exp(S - L_safe[:, None]))
    O = P @ V
    return O, L


def compute_chunk_partial(Q, K_chunk, V_chunk, col_start, causal=False, causal_offset=0):
    """
    Compute partial (O^(s), L^(s), m^(s), l^(s)) for a KV chunk [col_start, col_start + len(K_chunk)).
    Handles empty chunks or chunks where all keys are causally masked for some query rows.
    """
    Nq, d = Q.shape
    ck = K_chunk.shape[0]
    if ck == 0:
        O_s = torch.zeros((Nq, d), dtype=Q.dtype, device=Q.device)
        L_s = torch.full((Nq,), float("-inf"), dtype=Q.dtype, device=Q.device)
        m_s = torch.full((Nq,), float("-inf"), dtype=Q.dtype, device=Q.device)
        l_s = torch.zeros((Nq,), dtype=Q.dtype, device=Q.device)
        return O_s, L_s, m_s, l_s

    scale = 1.0 / math.sqrt(d)
    S_s = (Q @ K_chunk.T) * scale
    if causal:
        r_idx = torch.arange(Nq, device=Q.device)[:, None] + causal_offset
        c_idx = torch.arange(col_start, col_start + ck, device=Q.device)[None, :]
        S_s = S_s.masked_fill(c_idx > r_idx, float("-inf"))

    m_s = S_s.max(dim=-1).values
    all_inf = torch.isinf(m_s) & (m_s < 0)
    m_safe = torch.where(all_inf, torch.zeros_like(m_s), m_s)
    P_tilde = torch.where(all_inf[:, None], torch.zeros_like(S_s), torch.exp(S_s - m_safe[:, None]))
    l_s = P_tilde.sum(dim=-1)
    L_s = torch.where(all_inf, torch.full_like(m_s, float("-inf")), m_s + torch.log(l_s))
    denom = torch.where(l_s == 0, torch.ones_like(l_s), l_s)
    O_s = (P_tilde @ V_chunk) / denom[:, None]
    return O_s, L_s, m_s, l_s


def combine_splits_lse(O_list, L_list, uniform_bad=False, unguarded_bad=False):
    """
    Combine S partial outputs O_list [S, Nq, d] and logsumexp L_list [S, Nq]
    matching combine_attn_seqk_parallel in csrc/flash_attn/src/flash_fwd_kernel.h:1138-1300.
    """
    O_stack = torch.stack(O_list, dim=0)  # (S, Nq, d)
    L_stack = torch.stack(L_list, dim=0)  # (S, Nq)
    S_splits = O_stack.shape[0]

    if uniform_bad:
        return O_stack.mean(dim=0), L_stack.max(dim=0).values

    if unguarded_bad:
        # Unguarded: if L_max == -inf or L_s == -inf without guard
        L_max = L_stack.max(dim=0).values
        l_sum = torch.exp(L_stack - L_max[None, :]).sum(dim=0)
        L_total = L_max + torch.log(l_sum)
        w = torch.exp(L_stack - L_total[None, :])
        O_total = (w[:, :, None] * O_stack).sum(dim=0)
        return O_total, L_total

    L_max = L_stack.max(dim=0).values  # (Nq,)
    row_all_inf = torch.isinf(L_max) & (L_max < 0)
    L_max_safe = torch.where(row_all_inf, torch.zeros_like(L_max), L_max)
    exp_shifted = torch.where(
        torch.isinf(L_stack) & (L_stack < 0),
        torch.zeros_like(L_stack),
        torch.exp(L_stack - L_max_safe[None, :])
    )
    l_sum = exp_shifted.sum(dim=0)
    L_total = torch.where(row_all_inf, torch.full_like(L_max, float("-inf")), L_max_safe + torch.log(l_sum))
    L_total_safe = torch.where(row_all_inf, torch.zeros_like(L_total), L_total)
    w = torch.where(
        torch.isinf(L_stack) & (L_stack < 0),
        torch.zeros_like(L_stack),
        torch.exp(L_stack - L_total_safe[None, :])
    )
    O_total = (w[:, :, None] * O_stack).sum(dim=0)
    return O_total, L_total


def combine_splits_ml(O_list, m_list, l_list):
    """
    Combine S partial outputs using (m^(s), l^(s), O^(s)) representation.
    """
    O_stack = torch.stack(O_list, dim=0)
    m_stack = torch.stack(m_list, dim=0)
    l_stack = torch.stack(l_list, dim=0)
    m_max = m_stack.max(dim=0).values
    row_all_inf = torch.isinf(m_max) & (m_max < 0)
    m_max_safe = torch.where(row_all_inf, torch.zeros_like(m_max), m_max)
    scales = torch.where(
        torch.isinf(m_stack) & (m_stack < 0),
        torch.zeros_like(m_stack),
        torch.exp(m_stack - m_max_safe[None, :])
    )
    l_resc = scales * l_stack
    l_total = l_resc.sum(dim=0)
    denom = torch.where(l_total == 0, torch.ones_like(l_total), l_total)
    O_total = ((l_resc / denom[None, :])[:, :, None] * O_stack).sum(dim=0)
    L_total = torch.where(row_all_inf, torch.full_like(m_max, float("-inf")), m_max_safe + torch.log(denom))
    return O_total, L_total


def run_split_kv(Q, K, V, chunk_sizes, causal=False, causal_offset=0):
    O_list, L_list, m_list, l_list = [], [], [], []
    col = 0
    for sz in chunk_sizes:
        K_c = K[col:col + sz]
        V_c = V[col:col + sz]
        O_s, L_s, m_s, l_s = compute_chunk_partial(Q, K_c, V_c, col, causal=causal, causal_offset=causal_offset)
        O_list.append(O_s)
        L_list.append(L_s)
        m_list.append(m_s)
        l_list.append(l_s)
        col += sz
    assert col == K.shape[0]
    O_lse, L_lse = combine_splits_lse(O_list, L_list)
    O_ml, L_ml = combine_splits_ml(O_list, m_list, l_list)
    return O_lse, L_lse, O_ml, L_ml, (O_list, L_list)


def main():
    torch.manual_seed(99)
    d = 64

    print("=== 1. Equal chunks: N_k=1024, d=64, N_q in {1, 8, 64}, S in {1, 2, 4, 8, 16} ===")
    K = torch.randn(1024, d, dtype=torch.float64)
    V = torch.randn(1024, d, dtype=torch.float64)
    max_e_lse, max_e_ml, max_e_L = 0.0, 0.0, 0.0
    for Nq in [1, 8, 64]:
        Q = torch.randn(Nq, d, dtype=torch.float64)
        O_ref, L_ref = run_single_pass_ref(Q, K, V, causal=False)
        for S in [1, 2, 4, 8, 16]:
            csz = [1024 // S] * S
            O_lse, L_lse, O_ml, L_ml, _ = run_split_kv(Q, K, V, csz, causal=False)
            max_e_lse = max(max_e_lse, (O_lse - O_ref).abs().max().item())
            max_e_ml = max(max_e_ml, (O_ml - O_ref).abs().max().item())
            max_e_L = max(max_e_L, (L_lse - L_ref).abs().max().item())
    print(f"  [Equal chunks summary] max_O_lse_err={max_e_lse:.3e} max_O_ml_err={max_e_ml:.3e} max_L_err={max_e_L:.3e}")

    print("\n=== 2. Uneven & prime chunks: N_k=517, d=64 ===")
    K517 = torch.randn(517, d, dtype=torch.float64)
    V517 = torch.randn(517, d, dtype=torch.float64)
    Q16 = torch.randn(16, d, dtype=torch.float64)
    O_ref517, L_ref517 = run_single_pass_ref(Q16, K517, V517, causal=False)
    partitions = [
        ("S=3 ceil", [173, 173, 171]),
        ("S=5 ceil", [104, 104, 104, 104, 101]),
        ("S=7 ceil", [74, 74, 74, 74, 74, 74, 73]),
        ("S=11 with empty tail", [52]*9 + [49, 0]),
        ("irregular [1,3,19,100,394]", [1, 3, 19, 100, 394]),
        ("irregular [250,1,1,265]", [250, 1, 1, 265]),
    ]
    for name, csz in partitions:
        O_lse, L_lse, O_ml, _, _ = run_split_kv(Q16, K517, V517, csz, causal=False)
        print(f"  [{name:26s}] O_lse_err={(O_lse - O_ref517).abs().max().item():.3e} "
              f"O_ml_err={(O_ml - O_ref517).abs().max().item():.3e} L_err={(L_lse - L_ref517).abs().max().item():.3e}")

    print("\n=== 3. Causally masked chunks (including fully masked chunks with L^(s) = -inf) ===")
    for (Nq, Nk, offset, csz) in [
        (256, 256, 0, [64, 64, 64, 64]),
        (64, 517, 0, [64, 64, 128, 128, 133]),
        (16, 517, 100, [50, 50, 50, 150, 217]),
    ]:
        Q_c = torch.randn(Nq, d, dtype=torch.float64)
        K_c = torch.randn(Nk, d, dtype=torch.float64)
        V_c = torch.randn(Nk, d, dtype=torch.float64)
        O_r, L_r = run_single_pass_ref(Q_c, K_c, V_c, causal=True, causal_offset=offset)
        O_lse, L_lse, O_ml, _, (O_list, L_list) = run_split_kv(Q_c, K_c, V_c, csz, causal=True, causal_offset=offset)
        n_neg_inf = sum(int((torch.isinf(Ls) & (Ls < 0)).sum().item()) for Ls in L_list)
        print(f"  Nq={Nq:3d} Nk={Nk:3d} offset={offset:3d} splits={len(csz)} (-inf entries={n_neg_inf:4d}) | "
              f"O_lse_err={(O_lse - O_r).abs().max().item():.3e} O_ml_err={(O_ml - O_r).abs().max().item():.3e} L_err={(L_lse - L_r).abs().max().item():.3e}")

    print("\n=== 4. Negative controls on Split-KV combine ===")
    _, _, _, _, (O_list_u, L_list_u) = run_split_kv(Q16, K517, V517, [1, 3, 19, 100, 394], causal=False)
    O_unif, _ = combine_splits_lse(O_list_u, L_list_u, uniform_bad=True)
    print(f"  [NegCtrl 1: uniform 1/S average] max_abs_diff = {(O_unif - O_ref517).abs().max().item():.3e} (must fail > 0.1)")

    # Row with all splits masked (causal_offset = -1 on row 0) -> L_max = -inf -> unguarded exp(-inf - (-inf)) = NaN
    Q_m = torch.randn(4, d, dtype=torch.float64)
    _, _, _, _, (O_list_m, L_list_m) = run_split_kv(Q_m, K517, V517, [100, 200, 217], causal=True, causal_offset=-1)
    O_unguarded, _ = combine_splits_lse(O_list_m, L_list_m, unguarded_bad=True)
    O_guarded, _ = combine_splits_lse(O_list_m, L_list_m, unguarded_bad=False)
    print(f"  [NegCtrl 2: unguarded -inf row]  unguarded has_nan={bool(torch.isnan(O_unguarded).any())} "
          f"vs guarded has_nan={bool(torch.isnan(O_guarded).any())}")

if __name__ == "__main__":
    main()
