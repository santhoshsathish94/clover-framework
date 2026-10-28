import math
import torch

def trace_one_row_online_softmax(s_row, V, B_c=4, omit_rescale=False):
    """
    Process 1D score row s_row of length N and value matrix V of shape (N, d)
    in key blocks of size B_c.
    """
    N, d = V.shape
    T_c = math.ceil(N / B_c)
    m = float("-inf")
    l = 0.0
    O_fa1 = torch.zeros(d, dtype=torch.float64)
    O_tilde_fa2 = torch.zeros(d, dtype=torch.float64)

    mode_str = "OMIT exp(m_old - m_new) [NEG CTRL A]" if omit_rescale else "EXACT ONLINE SOFTMAX"
    print(f"--- Trace mode: {mode_str} (N={N}, d={d}, B_c={B_c}, T_c={T_c}) ---")
    for j in range(T_c):
        c0, c1 = j * B_c, min((j + 1) * B_c, N)
        s_blk = s_row[c0:c1]
        V_blk = V[c0:c1]

        m_tilde = s_blk.max().item()
        p_tilde = torch.exp(s_blk - m_tilde)
        l_tilde = p_tilde.sum().item()

        m_old = m
        l_old = l
        m_new = max(m_old, m_tilde)
        if m_old == float("-inf"):
            alpha = 0.0
        else:
            alpha = 1.0 if omit_rescale else math.exp(m_old - m_new)
        beta = 1.0 if omit_rescale else math.exp(m_tilde - m_new)

        l_new = alpha * l_old + beta * l_tilde
        # FA1 update (normalized every step)
        O_fa1 = ((l_old * alpha) * O_fa1 + (beta * p_tilde) @ V_blk) / l_new
        # FA2 update (unnormalized until epilogue)
        O_tilde_fa2 = alpha * O_tilde_fa2 + (beta * p_tilde) @ V_blk
        O_fa2_curr = O_tilde_fa2 / l_new

        # Exact prefix softmax over keys 0..c1
        p_prefix = torch.softmax(s_row[:c1], dim=0)
        O_prefix_ref = p_prefix @ V[:c1]
        prefix_err = (O_fa1 - O_prefix_ref).abs().max().item()

        m, l = m_new, l_new
        vec_str = ", ".join(f"{v:+.6f}" for v in O_fa1.tolist())
        print(f"  Block {j+1}/{T_c} [keys {c0:2d}..{c1-1:2d}]: "
              f"m_blk={m_tilde:+.4f} l_blk={l_tilde:.4f} | "
              f"m={m:+.4f} l={l:8.4f} alpha={alpha:.4f} beta={beta:.4f} | "
              f"O=[{vec_str}] | err_vs_prefix_ref={prefix_err:.2e}")

    O_fa2_final = O_tilde_fa2 / l
    lse_online = m + math.log(l)
    return O_fa1, O_fa2_final, lse_online


def test_causal_mask_shift_neg_control(N=64, d=32, B_c=16, B_r=16):
    torch.manual_seed(7)
    Q = torch.randn(N, d, dtype=torch.float64)
    K = torch.randn(N, d, dtype=torch.float64)
    V = torch.randn(N, d, dtype=torch.float64)
    scale = 1.0 / math.sqrt(d)

    def run_tiled(mask_shift=0):
        T_c = math.ceil(N / B_c)
        T_r = math.ceil(N / B_r)
        O = torch.zeros((N, d), dtype=torch.float64)
        l = torch.zeros(N, dtype=torch.float64)
        m = torch.full((N,), float("-inf"), dtype=torch.float64)
        for j in range(T_c):
            c0, c1 = j * B_c, min((j + 1) * B_c, N)
            Kj, Vj = K[c0:c1], V[c0:c1]
            for i in range(T_r):
                r0, r1 = i * B_r, min((i + 1) * B_r, N)
                Qi = Q[r0:r1]
                Sij = (Qi @ Kj.T) * scale
                r_idx = torch.arange(r0, r1)[:, None]
                c_idx = torch.arange(c0, c1)[None, :]
                # Correct causal: c_idx <= r_idx. Shifted by +1: c_idx <= r_idx + mask_shift
                Sij = Sij.masked_fill(c_idx > (r_idx + mask_shift), float("-inf"))
                m_tilde = Sij.max(dim=-1).values
                all_inf = torch.isinf(m_tilde) & (m_tilde < 0)
                m_tilde_s = torch.where(all_inf, torch.zeros_like(m_tilde), m_tilde)
                P_tilde = torch.where(all_inf[:, None], torch.zeros_like(Sij), torch.exp(Sij - m_tilde_s[:, None]))
                l_tilde = P_tilde.sum(dim=-1)
                mi, li, Oi = m[r0:r1], l[r0:r1], O[r0:r1]
                m_new = torch.maximum(mi, m_tilde)
                m_new_s = torch.where(torch.isinf(m_new) & (m_new < 0), torch.zeros_like(m_new), m_new)
                alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(mi), torch.exp(mi - m_new_s))
                beta = torch.where(all_inf, torch.zeros_like(m_tilde), torch.exp(m_tilde - m_new_s))
                l_new = alpha * li + beta * l_tilde
                denom = torch.where(l_new == 0, torch.ones_like(l_new), l_new)
                O[r0:r1] = ((li * alpha)[:, None] * Oi + (beta[:, None] * P_tilde) @ Vj) / denom[:, None]
                l[r0:r1], m[r0:r1] = l_new, m_new
        return O

    # Plain reference causal attention
    S_ref = (Q @ K.T) * scale
    idx = torch.arange(N)
    S_ref = S_ref.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
    O_ref = torch.softmax(S_ref, dim=-1) @ V

    O_exact = run_tiled(mask_shift=0)
    O_shift_plus1 = run_tiled(mask_shift=1)
    O_shift_minus1 = run_tiled(mask_shift=-1)

    err_exact = (O_exact - O_ref).abs().max().item()
    err_plus1 = (O_shift_plus1 - O_ref).abs().max().item()
    err_plus1_row0 = (O_shift_plus1[0] - O_ref[0]).abs().max().item()
    err_plus1_last = (O_shift_plus1[-1] - O_ref[-1]).abs().max().item()
    err_minus1 = (O_shift_minus1 - O_ref).abs().max().item()

    print("=== 3. Negative Control B: Causal mask shifted by +1 and -1 (N=64, d=32, B_c=16, B_r=16) ===")
    print(f"  [exact causal mask (shift=0)]   max_abs_diff = {err_exact:.3e}")
    print(f"  [shifted causal mask (shift=+1)] max_abs_diff = {err_plus1:.3e} "
          f"(row 0 err={err_plus1_row0:.3e}, last row N-1 err={err_plus1_last:.3e})")
    print(f"  [shifted causal mask (shift=-1)] max_abs_diff = {err_minus1:.3e}")


def main():
    torch.manual_seed(42)
    N, d, B_c = 16, 4, 4
    q = torch.randn(d, dtype=torch.float64)
    K = torch.randn(N, d, dtype=torch.float64)
    V = torch.randn(N, d, dtype=torch.float64)
    # Add a growing offset across blocks so m_new strictly increases in blocks 2 and 4
    s_row = (q @ K.T) / math.sqrt(d)
    s_row[4:8] += 1.5
    s_row[12:16] += 3.0

    O_plain = torch.softmax(s_row, dim=0) @ V
    lse_plain = torch.logsumexp(s_row, dim=0).item()
    plain_str = ", ".join(f"{v:+.6f}" for v in O_plain.tolist())
    print(f"=== Plain full-row softmax reference: O=[{plain_str}], LSE={lse_plain:+.6f} ===")

    print("\n=== 1. Exact Online Softmax Block-by-Block Trace ===")
    O_fa1, O_fa2, lse_on = trace_one_row_online_softmax(s_row, V, B_c=B_c, omit_rescale=False)
    print(f"  Final FA1 vs plain softmax max_abs_diff = {(O_fa1 - O_plain).abs().max().item():.3e}")
    print(f"  Final FA2 vs plain softmax max_abs_diff = {(O_fa2 - O_plain).abs().max().item():.3e}")
    print(f"  Final LSE vs torch.logsumexp abs_diff   = {abs(lse_on - lse_plain):.3e}")

    print("\n=== 2. Negative Control A: Omit exp(m_old - m_new) Rescale ===")
    O_bad1, O_bad2, lse_bad = trace_one_row_online_softmax(s_row, V, B_c=B_c, omit_rescale=True)
    print(f"  [NegCtrl A] FA1 without rescale vs plain softmax max_abs_diff = {(O_bad1 - O_plain).abs().max().item():.3e}")
    print(f"  [NegCtrl A] FA2 without rescale vs plain softmax max_abs_diff = {(O_bad2 - O_plain).abs().max().item():.3e}")
    print(f"  [NegCtrl A] LSE without rescale abs_diff                      = {abs(lse_bad - lse_plain):.3e}\n")

    test_causal_mask_shift_neg_control()

if __name__ == "__main__":
    main()
