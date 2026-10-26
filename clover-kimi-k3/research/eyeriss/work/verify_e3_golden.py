#!/usr/bin/env python3
import sys
import re

def golden_conv(R, H, W, C_IN, M_OUT, SPARSE):
    E = H - R + 1
    F = W - R + 1
    I = [[[( (h * 3 + w * 2 + ch + 1) % 5 ) + 1 for w in range(W)] for h in range(H)] for ch in range(C_IN)]
    FLT = [[[[( (r * 2 + s + m * 3 + ch * 2 + 1) % 4 ) + 1 for s in range(R)] for r in range(R)] for ch in range(C_IN)] for m in range(M_OUT)]
    if SPARSE:
        I[0][1][2] = 0
        I[0][3][0] = 0
        FLT[0][0][0][2] = 0
    out = {}
    for m in range(M_OUT):
        for x in range(E):
            for y in range(F):
                acc = 0
                for ch in range(C_IN):
                    for r in range(R):
                        for s in range(R):
                            acc += I[ch][x + r][y + s] * FLT[m][ch][r][s]
                out[(m, x, y)] = acc
    return out

def main():
    log_path = sys.argv[1]
    expect_fail = (len(sys.argv) > 2 and sys.argv[2] == "--expect-fail")
    with open(log_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    cfg = {}
    counts = {}
    rtl_out = {}
    for ln in lines:
        if ln.startswith("CFG "):
            for kv in ln.split()[1:]:
                k, v = kv.split("=")
                cfg[k] = int(v)
        elif ln.startswith("COUNTS "):
            for kv in ln.split()[1:]:
                k, v = kv.split("=")
                counts[k] = int(v)
        elif ln.startswith("OUT "):
            parts = dict(kv.split("=") for kv in ln.split()[1:])
            rtl_out[(int(parts["m"]), int(parts["x"]), int(parts["y"]))] = int(parts["val"])

    gold = golden_conv(cfg["R"], cfg["H"], cfg["W"], cfg["C"], cfg["M"], cfg["SPARSE"])
    mismatches = 0
    for k, gval in sorted(gold.items()):
        rval = rtl_out.get(k, None)
        if rval != gval:
            mismatches += 1

    macs = counts["macs"]
    print(f"RUN R={cfg['R']} H={cfg['H']}x{cfg['W']} C={cfg['C']} M={cfg['M']} PEs={cfg['PEs']} SPARSE={cfg['SPARSE']} CORRUPT={cfg['CORRUPT']} | "
          f"MACs={macs} w_wr={counts['w_wr']}({counts['w_wr']/macs:.4f}/MAC) "
          f"i_wr={counts['i_wr']}({counts['i_wr']/macs:.4f}/MAC) "
          f"w_rd={counts['w_rd']}({counts['w_rd']/macs:.4f}/MAC) "
          f"i_rd={counts['i_rd']}({counts['i_rd']/macs:.4f}/MAC) "
          f"psum_rd={counts['psum_rd']}({counts['psum_rd']/macs:.4f}/MAC) "
          f"psum_wr={counts['psum_wr']}({counts['psum_wr']/macs:.4f}/MAC) "
          f"vert_add={counts['vert_adds']}({counts['vert_adds']/macs:.4f}/MAC) "
          f"gated={counts['mac_gated']} | golden_mismatches={mismatches}/{len(gold)}")

    if expect_fail:
        if mismatches == 0:
            print("ERROR: Negative control with corrupted weight did NOT fail!")
            sys.exit(1)
    else:
        if mismatches != 0:
            print(f"ERROR: {mismatches} pixel mismatches against Python golden!")
            sys.exit(1)

if __name__ == "__main__":
    main()
