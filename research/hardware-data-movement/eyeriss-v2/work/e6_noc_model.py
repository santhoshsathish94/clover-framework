"""
E6 (Eyeriss v2 study): arithmetic of the paper's argument for a flexible two-level network, with every assumption stated.
THIS IS MY TOY MODEL OF THE ARGUMENT IN SECTION III (paper_main.tex lines 297-330), NOT THE AUTHORS' ANALYTICAL MODEL
(which searches mappings with data-distribution and bandwidth limits and was not available to me).

MODEL (all mine unless marked):
  * P PEs, each consuming one iact and one weight per cycle (SPad reuse ignored).
  * A mapping runs `m` output channels x `x` output positions in parallel, m*x <= P, m <= M_avail, x <= X_avail.
    An iact is shared by the m PEs that differ only in output channel; a weight is shared by the x PEs that differ only in position.
    So distinct source words per cycle: iact = x, weight = m.
  * Rigid array R x C (the paper's example of a fixed design: 1-D horizontal weight multicast + 1-D vertical iact multicast, i.e.
    m = R and x = C, or the transpose): utilisation = min(m_fixed, M_avail) * min(x_fixed, X_avail) / P, best of the two orientations.
  * Flexible: any (m, x); source words per cycle are limited by the number of source ports * values per port.
    FROM THE PAPER TEXT: 3 iact routers and 3 weight routers per cluster, 16 clusters, 24-bit ports = three 8-bit values.
    I take the cap to be 48 ports * 3 values = 144 words/cycle per data type (my reading; the paper does not state it as a limit).
  * Layer shapes (M_avail, X_avail) at batch 1 are ILLUSTRATIVE numbers chosen by me, not the paper's.
"""
import itertools

P = 192
R_ARR, C_ARR = 12, 16
CAP = 48 * 3

LAYERS = [
    ("CONV large (M=384, 13x13)", 384, 169),
    ("CONV few channels (M=16, 55x55)", 16, 3025),
    ("pointwise 1x1 (M=64, 56x56)", 64, 3136),
    ("depth-wise 3x3 (M_avail=1, 56x56)", 1, 3136),
    ("FC batch 1 (M=4096, X=1)", 4096, 1),
    ("FC batch 4 (M=4096, X=4)", 4096, 4),
]


def rigid(M, X):
    best = 0
    for (a, b) in ((R_ARR, C_ARR), (C_ARR, R_ARR)):
        best = max(best, min(a, M) * min(b, X))
    return best / P, best


def flexible(M, X, cap=None):
    best = (0, 0, 0)
    for m in range(1, min(M, P) + 1):
        for x in range(1, min(X, P // m) + 1):
            if cap is not None and (m > cap or x > cap):
                continue
            if m * x > best[0] * best[1] or (m * x == best[0] * best[1] and m + x < best[0] + best[1]):
                best = (m, x, m * x)
    m, x, used = best
    return used / P, m, x


def table():
    print(f"P={P} PEs; rigid array {R_ARR}x{C_ARR}; flexible cap {CAP} words/cycle per data type\n")
    print(f"{'layer':36} | rigid util | flexible util (m x x)  src words/cycle (iact+weight) | flexible, capped util (m x x) | mode implied (iact / weight)")
    for name, M, X in LAYERS:
        ru, _ = rigid(M, X)
        fu, m, x = flexible(M, X)
        cu, mc, xc = flexible(M, X, CAP)
        iact_mode = "broadcast" if x == 1 else ("unicast" if m == 1 else f"multicast x{m}")
        w_mode = "broadcast" if m == 1 and x > 1 else ("unicast" if x == 1 else f"multicast x{x}")
        print(f"{name:36} | {ru*100:8.1f}% | {fu*100:6.1f}% ({m:3d} x {x:3d})   {x + m:6d} ({x}+{m})        | {cu*100:6.1f}% ({mc:3d} x {xc:3d})        | {iact_mode} / {w_mode}")


def conservation_from_e3():
    print("\ncross-check with the RTL numbers of E3 (router_pe_4_clusters_tb, 3x3 kernel on 5x5 map, 4 clusters):")
    inj_w, inj_i = 9, 100          # words injected at the sources (E3 observation)
    fan_w, fan_i = 4, 1            # broadcast to 4 clusters / unicast per cluster
    print(f"  weights: injected {inj_w} x fan-out {fan_w} = delivered {inj_w*fan_w}  (E3 observed 36)")
    print(f"  iacts:   injected {inj_i} x fan-out {fan_i} = delivered {inj_i*fan_i}  (E3 observed 100)")
    m, x = 1, 4
    print(f"  model: this is the depth-wise-like case (m={m}, x={x}): iact streams = x = {x} per cycle, weight streams = m = {m}; "
          f"E3: 4 iact words per cycle injected over 25 cycles = 100, 1 weight word per cycle over 9 cycles = 9")


if __name__ == "__main__":
    table()
    conservation_from_e3()
