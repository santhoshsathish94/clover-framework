"""
E5 - independent count model for the three dataflows as implemented by the E3/E4 testbenches.

HOW THE FORMULAS WERE OBTAINED (honest provenance)
  Read from the testbench loops (tb_rs_conv2d_e3.sv, tb_ws_conv2d_e4.sv, tb_os_conv2d_e4.sv), NOT fitted to the
  printed numbers:
    RS: PE[r][c] (R x E) holds filter row r for every (m,ch): M*C*R words; holds ifmap row r+c for every ch: C*W words;
        per (m,ch,y,s) one MAC. Vertical accumulation is done by the testbench: (R-1) adds per (m, c, y).
    WS: PE[r][s] (R x R) holds weight W[m][ch][r][s] for every (m,ch): M*C words; sliding-window pixels
        I[ch][x+r][y+s] are streamed once per (ch,x,y) (shared by all m). (R*R-1) adds per (m,x,y).
    OS: PE[x][y] (E x F) receives all M*C*R*R weights and the C*R*R window pixels of its own output pixel.
  Per-MAC read counts and psum accesses are 1 per active MAC by construction of those loops (zero-gating aside).

WHAT THIS CAN AND CANNOT SHOW
  Writes (w_wr, i_wr) are counted in the testbenches at the PE scratchpad write-enable (a real signal in src/PE.v),
  but the *schedule that decides how many writes happen* is written by the testbench. `vert_add` is arithmetic done
  in the testbench loop ((R-1) or (R*R-1) per output pixel), not an event counted in hardware.
  So agreement here checks that the counters count the schedule correctly; it is not evidence about the paper's
  dataflow energy.

USAGE
  python e5_model.py predict            -> print predictions for the shapes listed in SHAPES_*
  python e5_model.py check FILE [FILE]  -> parse RUN lines of RTL outputs and compare with the model
"""
import sys, re


def macs(R, H, W, C, M):
    E, F = H - R + 1, W - R + 1
    return M * C * E * F * R * R


def model(df, R, H, W, C, M):
    E, F = H - R + 1, W - R + 1
    mac = macs(R, H, W, C, M)
    if df == "RS":
        pes = R * E
        w_wr = R * E * M * C * R
        i_wr = R * E * C * W
        vert = (R - 1) * M * E * F
    elif df == "WS":
        pes = R * R
        w_wr = R * R * M * C
        i_wr = R * R * C * E * F
        vert = (R * R - 1) * M * E * F
    elif df == "OS":
        pes = E * F
        w_wr = E * F * M * C * R * R
        i_wr = E * F * C * R * R
        vert = 0
    else:
        raise ValueError(df)
    return dict(PEs=pes, MACs=mac, w_wr=w_wr, i_wr=i_wr, vert_add=vert)


# shapes already run by E3/E4 (R,H,W,C,M) and held-out shapes that were NOT run when the model was written
SHAPES_SEEN = [(3, 5, 5, 1, 1), (3, 7, 7, 1, 1), (3, 8, 8, 1, 1), (3, 5, 5, 2, 1), (3, 5, 5, 1, 2), (5, 7, 7, 1, 1)]
SHAPES_HELDOUT = [(3, 6, 6, 2, 2), (4, 7, 7, 1, 1), (3, 6, 5, 1, 1), (2, 5, 5, 1, 1)]

RUN = re.compile(r"RUN R=(\d+) H=(\d+)x(\d+) C=(\d+) M=(\d+) PEs=(\d+) SPARSE=(\d) CORRUPT=(\d) \| MACs=(\d+) "
                 r"w_wr=(\d+)\S* i_wr=(\d+)\S* .*?vert_add=(\d+)\S* gated=(\d+) \| golden_mismatches=(\d+)/(\d+)")


def predict():
    for df in ("RS", "WS", "OS"):
        print(f"== {df}")
        for s in SHAPES_SEEN + SHAPES_HELDOUT:
            tag = "held-out" if s in SHAPES_HELDOUT else "seen"
            m = model(df, *s)
            print(f"  R={s[0]} H={s[1]}x{s[2]} C={s[3]} M={s[4]} [{tag}] " + " ".join(f"{k}={v}" for k, v in m.items()))


def check(files):
    bad = n = 0
    for path in files:
        df = None
        for line in open(path, encoding="utf-8", errors="replace"):
            u = line.upper()
            if line.startswith("==="):
                if "WEIGHT-STATIONARY" in u or line.startswith("=== WS"):
                    df = "WS"
                elif "OUTPUT-STATIONARY" in u or line.startswith("=== OS"):
                    df = "OS"
                elif "ROW-STATIONARY" in u or line.startswith("=== RS"):
                    df = "RS"
                elif "UNMODIFIED COUNTER" in u:
                    df = None  # deliberately broken run, not modelled
                continue
            mt = RUN.search(line)
            if not mt or df is None:
                continue
            R, H, W, C, M, pes, sp, corr, mac, w_wr, i_wr, vert, gated, mism, tot = map(int, mt.groups())
            if sp or corr:
                continue  # sparse / corrupted runs are control runs
            p = model(df, R, H, W, C, M)
            got = dict(PEs=pes, MACs=mac, w_wr=w_wr, i_wr=i_wr, vert_add=vert)
            ok = got == p and mism == 0
            n += 1
            bad += 0 if ok else 1
            print(f"{'OK ' if ok else 'BAD'} {df} R={R} H={H}x{W} C={C} M={M} golden_mismatch={mism}/{tot}" +
                  ("" if ok else f"  model={p} rtl={got}"))
    print(f"checked {n} runs, {bad} disagree")


# AlexNet shapes from ISCA 2016 Table II (H is the padded size): name, H, R, E, C, M, U
ALEXNET = [("CONV1", 227, 11, 55, 3, 96, 4), ("CONV2", 31, 5, 27, 48, 256, 1), ("CONV3", 15, 3, 13, 256, 384, 1),
           ("CONV4", 15, 3, 13, 192, 384, 1), ("CONV5", 15, 3, 13, 192, 256, 1)]


def alexnet():
    print("layer   MACs(N=1)     ifmap  filter   ofmap | logical PE set RS(RxE)  WS(RxR)  OS(ExE) | per-PE words, TB mapping: filter rows (p=M,q=C) | ratio MAC/ifmap  MAC/filter  MAC/ofmap")
    for name, H, R, E, C, M, U in ALEXNET:
        mac = M * C * E * E * R * R
        ifm, flt, ofm = C * H * H, M * C * R * R, M * E * E
        print(f"{name:6} {mac:12,} {ifm:9,} {flt:8,} {ofm:8,} | {R:>3}x{E:<3}={R*E:<5}       {R*R:<5}   {E*E:<6} | M*C*R = {M*C*R:<8,} | {mac/ifm:9.1f} {mac/flt:10.1f} {mac/ofm:10.1f}")
    print("\nSpad sizes from JSSC 2017 (words of 16 b): filter 224, ifmap 12, psum 24; array 12x14 = 168 PEs.")
    for name, H, R, E, C, M, U in ALEXNET:
        print(f"{name}: RS logical PE set {R}x{E} vs array 12x14 -> rows fit: {R <= 12}, columns fit: {E <= 14}; "
              f"1D primitive needs filter {R}, ifmap window {R}, psum 1 words (JSSC text); "
              f"TB-style full-row ifmap needs C*W = {C*H} words (> 12: {C*H > 12})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        predict()
    elif len(sys.argv) >= 3 and sys.argv[1] == "check":
        check(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == "alexnet":
        alexnet()
    else:
        print(__doc__)
