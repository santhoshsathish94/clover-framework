"""
E5 (Eyeriss v2 study): the compressed-sparse-column (CSC) format and the sparse PE's processing order, implemented
from the text of Section IV of the paper (paper_main.tex lines 456-556) and checked by round trip and against a dense
reference.  Nothing here is the authors' code.

CHOICES THAT ARE MINE (the text does not say):
  * count restarts at 0 at the beginning of each segment (weights: a column of M0 values; iacts: a C0*U segment);
  * a gap longer than the count can hold (15 for 4 bits) is bridged by padding entries (count=15, data=0), each of which
    advances 16 positions (15 zeros plus its own zero data) - the EIE-style convention;
  * iact segment/flatten order: index = c*U + u; weight column index k = c*S + s;
  * the PE emulation implements U = 1 only (the format functions take any U).
FROM THE TEXT: count-data pair = 4b + 8b = 12b; weight address 7b per column, iact address 4b per segment; address vector
has one entry per segment plus the end (the text reads address[1] and address[2] for column 1); iact is read before the
weight; an iact whose weight column is empty is dropped without a bubble; SIMD fetches two weights per cycle and pads an odd
column with a zero pair; a dense PE does M0*C0*S MACs per window.

usage: python e5_csc.py            (runs everything, prints tables)
"""
import numpy as np

CNT_BITS, DATA_BITS = 4, 8
CMAX = (1 << CNT_BITS) - 1          # 15
ADDR_BITS_W, ADDR_BITS_I = 7, 4


# ---------------------------------------------------------------- format
def encode_segment(vals):
    """vals: 1-D int array. Returns list of (count, data) pairs."""
    out, gap = [], 0
    for v in vals:
        if v == 0:
            gap += 1
        else:
            while gap > CMAX:                    # bridge long gaps
                out.append((CMAX, 0))
                gap -= CMAX + 1
            out.append((gap, int(v)))
            gap = 0
    return out                                    # trailing zeros need no entries (segment length is known)


def decode_segment(pairs, length):
    out = np.zeros(length, dtype=np.int64)
    pos = 0
    for c, d in pairs:
        pos += c
        out[pos] = d                              # a padding entry writes a 0
        pos += 1
    return out


def encode_segments(segs):
    """segs: list of 1-D arrays. Returns (addr vector, flat pair list)."""
    addr, flat = [0], []
    for s in segs:
        flat.extend(encode_segment(s))
        addr.append(len(flat))
    return addr, flat


def decode_segments(addr, flat, length):
    return [decode_segment(flat[addr[i]:addr[i + 1]], length) for i in range(len(addr) - 1)]


def bits(addr, flat, addr_bits):
    return len(flat) * (CNT_BITS + DATA_BITS) + len(addr) * addr_bits


# ---------------------------------------------------------------- round trip + compression sweep
def roundtrip_tests(rng):
    n = 0
    for trial in range(400):
        L = int(rng.integers(1, 80))
        nseg = int(rng.integers(1, 12))
        dens = rng.choice([0.0, 0.01, 0.05, 0.2, 0.5, 1.0])
        segs = [np.where(rng.random(L) < dens, rng.integers(1, 127, L), 0) for _ in range(nseg)]
        addr, flat = encode_segments(segs)
        back = decode_segments(addr, flat, L)
        assert all(np.array_equal(a, b) for a, b in zip(segs, back)), (trial, dens)
        n += 1
    # negative control: change one count -> decode must differ (or raise)
    segs = [np.array([0, 5, 0, 0, 7, 0, 9, 0, 0, 0])]
    addr, flat = encode_segments(segs)
    bad = list(flat); bad[1] = (bad[1][0] + 1, bad[1][1])
    try:
        differs = not np.array_equal(decode_segments(addr, bad, 10)[0], segs[0])
    except IndexError:
        differs = True
    return n, differs


def compression_sweep(rng):
    print("\n== compression: weight matrix M0=32 x K=15 (C0=5,S=3), 8b values; 200 random matrices per density")
    M0, K = 32, 15
    dense_bits = 8 * M0 * K
    print("density  csc_bits(mean)  dense_bits  ratio   padding entries(mean)  predicted_ratio")
    rows = []
    for dens in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.647, 0.7, 0.8, 1.0]:
        tot_bits = tot_pad = 0
        for _ in range(200):
            mat = np.where(rng.random((M0, K)) < dens, rng.integers(1, 127, (M0, K)), 0)
            addr, flat = encode_segments([mat[:, k] for k in range(K)])
            tot_bits += bits(addr, flat, ADDR_BITS_W)
            tot_pad += sum(1 for c, d in flat if d == 0)
        mean_bits = tot_bits / 200
        pred = (12 * dens * M0 * K + ADDR_BITS_W * (K + 1)) / dense_bits      # ignores padding and the all-zero tail
        rows.append((dens, mean_bits / dense_bits))
        print(f"{dens:7.3f}  {mean_bits:14.1f}  {dense_bits:10d}  {mean_bits/dense_bits:5.3f}  {tot_pad/200:21.3f}  {pred:15.3f}")
    # break-even from the formula: 12*d*M0*K + 7*(K+1) = 8*M0*K
    d_star = (8 * M0 * K - ADDR_BITS_W * (K + 1)) / (12 * M0 * K)
    print(f"formula break-even density d* = (8*M0*K - 7*(K+1)) / (12*M0*K) = {d_star:.4f}")
    # empirical crossing by linear interpolation of the swept ratios
    for (d0, r0), (d1, r1) in zip(rows, rows[1:]):
        if r0 < 1.0 <= r1:
            print(f"empirical crossing of ratio=1.0 between density {d0} and {d1}: {d0 + (1-r0)*(d1-d0)/(r1-r0):.4f}")

    print("\n== long gaps: one segment of length 256, a single non-zero at the end; count field = 4 bits")
    seg = np.zeros(256, dtype=np.int64); seg[255] = 9
    addr, flat = encode_segments([seg])
    print(f"entries={len(flat)} (padding={sum(1 for c,d in flat if d==0)}) -> decoded equal: {np.array_equal(decode_segment(flat,256), seg)}; bits={bits(addr, flat, ADDR_BITS_W)} vs dense {8*256}")
    print(f"predicted padding entries = floor(255/16) = {255//16}")


# ---------------------------------------------------------------- sparse PE processing order
def sparse_pe(Wm, I, S):
    """Wm: (M0, C0*S) weights; I: (C0, W) iacts; U=1. Returns psums (E, M0) and counters."""
    M0, K = Wm.shape
    C0, Wd = I.shape
    E = Wd - S + 1
    w_addr, w_flat = encode_segments([Wm[:, k] for k in range(K)])
    i_addr, i_flat = encode_segments([I[:, p] for p in range(Wd)])     # one segment per position, C0 values
    ps = np.zeros((E, M0), dtype=np.int64)
    c = dict(macs=0, iact_reads=0, weight_reads=0, addr_pair_reads=0, simd2_cycles=0, visited_nz_iact=0, dropped_iact=0)
    for x in range(E):
        for s in range(S):
            p = x + s
            pos = 0
            for cnt, a in i_flat[i_addr[p]:i_addr[p + 1]]:           # read iact first
                pos += cnt
                ch = pos
                pos += 1
                c["iact_reads"] += 1
                if a == 0:
                    continue                                        # padding entry
                c["visited_nz_iact"] += 1
                k = ch * S + s
                c["addr_pair_reads"] += 1                           # address[k], address[k+1]
                col = w_flat[w_addr[k]:w_addr[k + 1]]
                nzw = [e for e in col if e[1] != 0]
                if not nzw:
                    c["dropped_iact"] += 1
                    continue
                c["simd2_cycles"] += -(-len(nzw) // 2)
                m = 0
                for wc, wv in col:
                    m += wc
                    c["weight_reads"] += 1
                    if wv != 0:
                        ps[x, m] += a * wv
                        c["macs"] += 1
                    m += 1
    return ps, c


def dense_ref(Wm, I, S):
    M0, K = Wm.shape
    C0, Wd = I.shape
    E = Wd - S + 1
    out = np.zeros((E, M0), dtype=np.int64)
    for x in range(E):
        for ch in range(C0):
            for s in range(S):
                out[x] += Wm[:, ch * S + s] * I[ch, x + s]
    return out


def pe_sweep(rng):
    M0, C0, S, Wd = 32, 4, 3, 40
    E = Wd - S + 1
    dense_macs = E * M0 * C0 * S
    print(f"\n== sparse PE processing order: M0={M0} C0={C0} S={S} W={Wd} (E={E} windows); dense PE = {dense_macs} MACs; 5 seeds per cell")
    print("w_dens i_dens | MACs/dense  w*i   | simd1_cycles(=MACs)  simd2_cycles  MACs/2  bound MACs/2+nzvisits/2 | exact  dropped_iact  weight_reads/MAC")
    ok_all = True
    for wd in (1.0, 0.5, 0.25, 0.1):
        for idn in (1.0, 0.6, 0.3):
            tm = ts = tv = tw = td = 0
            for seed in range(5):
                Wm = np.where(rng.random((M0, C0 * S)) < wd, rng.integers(1, 100, (M0, C0 * S)), 0)
                I = np.where(rng.random((C0, Wd)) < idn, rng.integers(1, 100, (C0, Wd)), 0)
                ps, c = sparse_pe(Wm, I, S)
                exact = np.array_equal(ps, dense_ref(Wm, I, S))
                ok_all &= exact
                bound = c["macs"] / 2 + c["visited_nz_iact"] / 2
                ok_all &= (c["macs"] / 2 <= c["simd2_cycles"] <= bound)
                tm += c["macs"]; ts += c["simd2_cycles"]; tv += c["visited_nz_iact"]; tw += c["weight_reads"]; td += c["dropped_iact"]
            n = 5
            print(f"{wd:6.2f} {idn:6.2f} | {tm/n/dense_macs:9.4f}  {wd*idn:5.3f}  | {tm/n:19.1f}  {ts/n:12.1f}  {tm/n/2:6.1f}  {tm/n/2 + tv/n/2:22.1f} | {str(exact):5}  {td/n:12.1f}  {tw/max(tm,1):10.3f}")
    print("all cells: PE emulation == dense reference and MACs/2 <= simd2_cycles <= bound:", ok_all)
    # negative control on the weight CSC: corrupt one count
    Wm = np.where(rng.random((M0, C0 * S)) < 0.5, rng.integers(1, 100, (M0, C0 * S)), 0)
    I = np.where(rng.random((C0, Wd)) < 0.6, rng.integers(1, 100, (C0, Wd)), 0)
    ref = dense_ref(Wm, I, S)
    Wm2 = Wm.copy()
    nzr = np.argwhere(Wm2 != 0)[0]
    Wm2[nzr[0], nzr[1]] = 0
    Wm2[(nzr[0] + 1) % M0, nzr[1]] = 77                 # a non-zero moved by one row = a changed count
    ps_bad, _ = sparse_pe(Wm2, I, S)
    print("negative control (one weight moved one row): emulation differs from original reference:", not np.array_equal(ps_bad, ref))


if __name__ == "__main__":
    rng = np.random.default_rng(20261026)
    n, differs = roundtrip_tests(rng)
    print(f"round trip: {n} random segment sets decode exactly; corrupted count changes the decode: {differs}")
    compression_sweep(rng)
    pe_sweep(rng)
