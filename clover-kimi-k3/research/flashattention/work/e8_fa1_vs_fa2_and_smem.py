import math
import re
import sys
import torch

class DetailedIOAndOpCounter:
    def __init__(self):
        self.reads = {"Q": 0, "K": 0, "V": 0, "O": 0, "m": 0, "l": 0, "L": 0}
        self.writes = {"O": 0, "m": 0, "l": 0, "L": 0}
        self.tile_writes = {"O": 0, "m": 0, "l": 0, "L": 0}
        self.o_rescale_muls = 0
        self.o_norm_divs = 0

    def read(self, name, tensor):
        self.reads[name] += tensor.numel()
        return tensor.clone()

    def write(self, name, dst, src):
        self.writes[name] += src.numel()
        self.tile_writes[name] += 1
        dst.copy_(src)


def run_fa1_outer_kv(Q, K, V, B_r=64, B_c=64, causal=False):
    """
    FlashAttention-1 loop order (2205.14135 Alg 1):
    Outer loop j = 1..T_c over K_j, V_j; inner loop i = 1..T_r over Q_i, O_i, l_i, m_i.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    T_c = math.ceil(N / B_c)
    T_r = math.ceil(N / B_r)
    ctr = DetailedIOAndOpCounter()

    O_hbm = torch.zeros((N, d), dtype=Q.dtype)
    l_hbm = torch.zeros((N,), dtype=Q.dtype)
    m_hbm = torch.full((N,), float("-inf"), dtype=Q.dtype)

    for j in range(T_c):
        c0, c1 = j * B_c, min((j + 1) * B_c, N)
        Kj = ctr.read("K", K[c0:c1])
        Vj = ctr.read("V", V[c0:c1])
        for i in range(T_r):
            r0, r1 = i * B_r, min((i + 1) * B_r, N)
            if causal and (r1 - 1) < c0:
                continue
            Qi = ctr.read("Q", Q[r0:r1])
            Oi = ctr.read("O", O_hbm[r0:r1])
            li = ctr.read("l", l_hbm[r0:r1])
            mi = ctr.read("m", m_hbm[r0:r1])

            Sij = (Qi @ Kj.T) * scale
            if causal:
                r_idx = torch.arange(r0, r1)[:, None]
                c_idx = torch.arange(c0, c1)[None, :]
                Sij = Sij.masked_fill(r_idx < c_idx, float("-inf"))

            m_tilde = Sij.max(dim=-1).values
            all_inf = torch.isinf(m_tilde) & (m_tilde < 0)
            m_tilde_s = torch.where(all_inf, torch.zeros_like(m_tilde), m_tilde)
            P_tilde = torch.where(all_inf[:, None], torch.zeros_like(Sij), torch.exp(Sij - m_tilde_s[:, None]))
            l_tilde = P_tilde.sum(dim=-1)

            m_new = torch.maximum(mi, m_tilde)
            m_new_s = torch.where(torch.isinf(m_new) & (m_new < 0), torch.zeros_like(m_new), m_new)
            alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(mi), torch.exp(mi - m_new_s))
            beta = torch.where(all_inf, torch.zeros_like(m_tilde), torch.exp(m_tilde - m_new_s))
            l_new = alpha * li + beta * l_tilde

            denom = torch.where(l_new == 0, torch.ones_like(l_new), l_new)
            # FA1 rescales Oi by (li * alpha) and divides by l_new on every inner step
            Oi_new = ((li * alpha)[:, None] * Oi + (beta[:, None] * P_tilde) @ Vj) / denom[:, None]
            ctr.o_rescale_muls += Oi.numel()
            ctr.o_norm_divs += Oi.numel()

            ctr.write("O", O_hbm[r0:r1], Oi_new)
            ctr.write("l", l_hbm[r0:r1], l_new)
            ctr.write("m", m_hbm[r0:r1], m_new)

    return O_hbm, m_hbm + torch.log(l_hbm), ctr


def run_fa2_outer_q(Q, K, V, B_r=64, B_c=64, causal=False, skip_epilogue_div=False):
    """
    FlashAttention-2 loop order (2307.08691 Alg 1):
    Outer loop i = 1..T_r over Q_i; inner loop j = 1..T_c over K_j, V_j.
    O_tilde_i, l_i, m_i stay in on-chip registers; O_i and L_i written to HBM ONCE in epilogue.
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    T_c = math.ceil(N / B_c)
    T_r = math.ceil(N / B_r)
    ctr = DetailedIOAndOpCounter()

    O_hbm = torch.empty((N, d), dtype=Q.dtype)
    L_hbm = torch.empty((N,), dtype=Q.dtype)

    for i in range(T_r):
        r0, r1 = i * B_r, min((i + 1) * B_r, N)
        Qi = ctr.read("Q", Q[r0:r1])
        # On-chip accumulators in registers (zero HBM traffic)
        O_tilde = torch.zeros((r1 - r0, d), dtype=Q.dtype)
        li = torch.zeros((r1 - r0,), dtype=Q.dtype)
        mi = torch.full((r1 - r0,), float("-inf"), dtype=Q.dtype)

        for j in range(T_c):
            c0, c1 = j * B_c, min((j + 1) * B_c, N)
            if causal and (r1 - 1) < c0:
                continue
            Kj = ctr.read("K", K[c0:c1])
            Vj = ctr.read("V", V[c0:c1])

            Sij = (Qi @ Kj.T) * scale
            if causal:
                r_idx = torch.arange(r0, r1)[:, None]
                c_idx = torch.arange(c0, c1)[None, :]
                Sij = Sij.masked_fill(r_idx < c_idx, float("-inf"))

            m_new = torch.maximum(mi, Sij.max(dim=-1).values)
            m_new_s = torch.where(torch.isinf(m_new) & (m_new < 0), torch.zeros_like(m_new), m_new)
            alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(mi), torch.exp(mi - m_new_s))
            P_tilde = torch.exp(Sij - m_new_s[:, None])
            li = alpha * li + P_tilde.sum(dim=-1)
            O_tilde = alpha[:, None] * O_tilde + P_tilde @ Vj
            ctr.o_rescale_muls += O_tilde.numel()
            mi = m_new

        # Epilogue: divide by li once and write O_i and L_i once to HBM
        if skip_epilogue_div:
            Oi_final = O_tilde
        else:
            Oi_final = O_tilde / li[:, None]
            ctr.o_norm_divs += O_tilde.numel()
        Li_final = mi + torch.log(li)
        ctr.write("O", O_hbm[r0:r1], Oi_final)
        ctr.write("L", L_hbm[r0:r1], Li_final)

    return O_hbm, L_hbm, ctr


def parse_cuda_headers_read_not_run(repo_dir):
    print("\n=== 3. [READ, NOT RUN] CUDA Kernel Headers: Tile Sizes & Shared Memory Formula (fp16, sm80) ===")
    kt_path = f"{repo_dir}/csrc/flash_attn/src/kernel_traits.h"
    fwd_path = f"{repo_dir}/csrc/flash_attn/src/flash_fwd_launch_template.h"
    bwd_path = f"{repo_dir}/csrc/flash_attn/src/flash_bwd_launch_template.h"

    kt_lines = open(kt_path).read().splitlines()
    print("  From csrc/flash_attn/src/kernel_traits.h (lines 103-111):")
    for idx in range(102, 111):
        print(f"    L{idx+1}: {kt_lines[idx]}")

    fwd_lines = open(fwd_path).read().splitlines()
    print("\n  From csrc/flash_attn/src/flash_fwd_launch_template.h (lines 203-278, hdim 64, 96, 128):")
    for idx in range(202, min(278, len(fwd_lines))):
        if "//" not in fwd_lines[idx].strip()[:3]:
            print(f"    L{idx+1}: {fwd_lines[idx]}")

    fwd_text = "\n".join(fwd_lines)
    trait_pat = re.compile(
        r"^\s*run_flash_fwd<Flash_fwd_kernel_traits<\s*Headdim\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*T\s*>",
        re.MULTILINE
    )
    print("\n  Computed shared memory from active Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, T> [READ, NOT RUN]:")
    for hdim in (64, 128):
        m_fn = re.search(rf"void run_mha_fwd_hdim{hdim}\([\s\S]*?\n\}}", fwd_text)
        if not m_fn:
            continue
        for m in trait_pat.finditer(m_fn.group(0)):
            bm, bn, nw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            q_reg, share_qk = m.group(4) == "true", m.group(5) == "true"
            smem_q = bm * hdim * 2
            smem_kv = 2 * bn * hdim * 2
            smem_total = max(smem_q, smem_kv) if share_qk else (smem_q + smem_kv)
            ctas_a100 = (164 * 1024) // smem_total
            print(f"    [FWD fp16] d={hdim:3d} kBlockM={bm:3d} kBlockN={bn:3d} kNWarps={nw} Q_in_regs={str(q_reg):5s} Share_QK={str(share_qk):5s} | "
                  f"kSmemQ={smem_q:6,d} B + kSmemKV={smem_kv:6,d} B -> kSmemSize={smem_total:6,d} B ({smem_total//1024:2d} KB), max CTAs/SM(sm80 164KB)={ctas_a100}")

    bwd_text = open(bwd_path).read()
    bwd_pat = re.compile(
        r"^\s*run_flash_bwd<Flash_bwd_kernel_traits<\s*Headdim\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*T\s*>",
        re.MULTILINE
    )
    print("\n  Computed shared memory from active Flash_bwd_kernel_traits (d=64, 128) in flash_bwd_launch_template.h [READ, NOT RUN]:")
    for hdim in (64, 128):
        m_fn = re.search(rf"void run_mha_bwd_hdim{hdim}\([\s\S]*?\n\}}", bwd_text)
        if not m_fn:
            continue
        for m in bwd_pat.finditer(m_fn.group(0)):
            bm, bn, nw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            atom_msdp, atom_ndkv, atom_mdq = int(m.group(4)), int(m.group(5)), int(m.group(6))
            v_in_regs = m.group(7) == "true"
            no_double_buffer = m.group(8) == "true"
            # kernel_traits.h lines 279-289: sQ and sdO share kSmemQdOSize = size(QdO) * (2 if No_double_buffer else 3) * 2 B
            smem_qdo = bm * hdim * (2 if no_double_buffer else 3) * 2
            smem_kv = bn * hdim * 2 * 2
            smem_ds = bm * bn * 2
            smem_p = bm * bn * 2
            if v_in_regs:
                smem_1col = smem_qdo + max(smem_kv, smem_kv // 2 + smem_ds + smem_p)
            else:
                smem_1col = smem_qdo + smem_kv + smem_ds + smem_p
            print(f"    [BWD fp16] d={hdim:3d} kBlockM={bm:3d} kBlockN={bn:3d} kNWarps={nw:2d} AtomLayout(MSdP,NdKV,MdQ)=({atom_msdp},{atom_ndkv},{atom_mdq}) "
                  f"Is_V_in_regs={str(v_in_regs):5s} No_double_buffer={str(no_double_buffer):5s} | "
                  f"QdO={smem_qdo:6,d} KV={smem_kv:6,d} dS={smem_ds:6,d} P={smem_p:6,d} -> kSmemSize1colblock={smem_1col:7,d} B ({smem_1col/1024:6.1f} KB)")


def main():
    torch.manual_seed(2026)
    print("=== 1. FA1 (outer KV) vs FA2 (outer Q) Loop Order Write/Read Counts (B_r=64, B_c=64, d=64) ===")
    for N in [64, 256, 1024]:
        for causal in [False, True]:
            Q = torch.randn(N, 64, dtype=torch.float64)
            K = torch.randn(N, 64, dtype=torch.float64)
            V = torch.randn(N, 64, dtype=torch.float64)
            O1, L1, c1 = run_fa1_outer_kv(Q, K, V, B_r=64, B_c=64, causal=causal)
            O2, L2, c2 = run_fa2_outer_q(Q, K, V, B_r=64, B_c=64, causal=causal)
            diff_O = (O1 - O2).abs().max().item()
            diff_L = (L1 - L2).abs().max().item()
            print(f"  N={N:4d} causal={str(causal):5s} | diff(O1,O2)={diff_O:.2e} diff(L1,L2)={diff_L:.2e} | "
                  f"O_writes: FA1={c1.writes['O']:9,d} ({c1.tile_writes['O']:3d} tiles) vs FA2={c2.writes['O']:6,d} ({c2.tile_writes['O']:2d} tiles) | "
                  f"O_reads: FA1={c1.reads['O']:9,d} vs FA2={c2.reads['O']} | "
                  f"(m,l/L)_writes: FA1={c1.writes['m']+c1.writes['l']:6,d} vs FA2={c2.writes['L']:5,d} | "
                  f"O_norm_divs: FA1={c1.o_norm_divs:9,d} vs FA2={c2.o_norm_divs:6,d}")

    print("\n=== 2. Negative Control on FA2 Unnormalized Accumulator (skip epilogue / l_i division) ===")
    Q = torch.randn(256, 64, dtype=torch.float64)
    K = torch.randn(256, 64, dtype=torch.float64)
    V = torch.randn(256, 64, dtype=torch.float64)
    O_exact, _, _ = run_fa2_outer_q(Q, K, V, B_r=64, B_c=64, causal=False, skip_epilogue_div=False)
    O_nodiv, _, _ = run_fa2_outer_q(Q, K, V, B_r=64, B_c=64, causal=False, skip_epilogue_div=True)
    print(f"  [NegCtrl skip_epilogue_div=True] max_abs_diff = {(O_nodiv - O_exact).abs().max().item():.3e} (must fail > 1.0)")

    parse_cuda_headers_read_not_run(sys.argv[1])

if __name__ == "__main__":
    main()
