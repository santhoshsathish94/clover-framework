#!/usr/bin/env python3
"""The complete per-stage picture of one prompt: where every second goes.

Reads the K3_TRACE timeline, which carries one row per operator invocation,
one per trunk slot fetch, one per pipeline wait, and one span per layer.
"""
import collections
import sys

R = []
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) != 7 or p[0] == "kind":
        continue
    R.append((p[0], int(p[1]), p[2], p[3], float(p[4]), float(p[5]), int(p[6])))

wall = max(r[4] + r[5] for r in R)
tk = collections.Counter()
tn = collections.Counter()
for k, L, lab, det, t0, dt, b in R:
    tk[k] += dt
    tn[k] += 1
lay = tk["layer"]

print("=" * 78)
print("COMPLETE STAGE PROFILE - one prompt, 5 tokens, 93 layers, 2455 trunk stages")
print("=" * 78)
print("trace rows %d    timeline span %.2f s" % (len(R), wall))
print()

print("--- 1. WHERE THE TIME GOES (layer rows are spans containing the rest) ---")
for k in ("fetch", "op", "wait"):
    print("  %-8s %7d events  %9.3f s   %5.1f%% of layer time"
          % (k, tn[k], tk[k], 100 * tk[k] / lay))
acc = tk["fetch"] + tk["op"] + tk["wait"]
print("  %-8s %7s  %9.3f s   %5.1f%%" % ("sum", "", acc, 100 * acc / lay))
print("  %-8s %7d  %9.3f s   unaccounted %.3f s (%.1f%%)"
      % ("layers", tn["layer"], lay, lay - acc, 100 * (lay - acc) / lay))
print()

print("--- 2. FETCH, every trunk slot type ---")
f = collections.defaultdict(lambda: [0, 0.0, 0])
for k, L, lab, det, t0, dt, b in R:
    if k == "fetch":
        f[lab][0] += 1; f[lab][1] += dt; f[lab][2] += b
print("  %-9s %6s %10s %11s %10s %9s" % ("slot", "count", "total s", "MB each", "GB total", "GB/s"))
for lab in sorted(f, key=lambda x: -f[x][1]):
    c, s, b = f[lab]
    print("  %-9s %6d %10.3f %11.2f %10.2f %9.2f"
          % (lab, c, s, b / c / 1e6, b / 1e9, b / 1e9 / s if s else 0))
print()

print("--- 3. OPERATORS, every kind ---")
o = collections.defaultdict(lambda: [0, 0.0, 0])
for k, L, lab, det, t0, dt, b in R:
    if k == "op":
        o[lab][0] += 1; o[lab][1] += dt; o[lab][2] += b
print("  %-26s %7s %10s %10s" % ("operator", "calls", "total s", "GB"))
for lab in sorted(o, key=lambda x: -o[x][1]):
    c, s, b = o[lab]
    print("  %-26s %7d %10.3f %10.2f" % (lab, c, s, b / 1e9))
print()

print("--- 4. FETCH AGAINST COMPUTE, the four big trunk stages of one KDA layer ---")
print("  a stage is fetched, then used; nothing overlaps")
seen = 0
for k, L, lab, det, t0, dt, b in R:
    if L != 1:
        continue
    if k == "fetch" and b > 4e7:
        print("  fetch %-6s %9.4f s  %7.2f MB  %6.2f GB/s   at t=%.4f" % (lab, dt, b / 1e6, b / 1e9 / dt, t0))
        seen += 1
    elif k == "op" and lab.startswith("Q ") and seen:
        print("        %-6s %9.4f s  %7.2f MB  %6.2f GB/s   at t=%.4f   <- %.0fx faster than its fetch"
              % ("use", dt, b / 1e6, b / 1e9 / dt, t0, 0))
print()

print("--- 5. PER LAYER, all 93 ---")
pl = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
for k, L, lab, det, t0, dt, b in R:
    if k == "layer":   pl[L][0] = dt
    elif k == "fetch": pl[L][1] += dt
    elif k == "op":    pl[L][2] += dt
    elif k == "wait":  pl[L][3] += dt
print("  %-6s %9s %9s %9s %9s %9s" % ("layer", "span s", "fetch s", "op s", "wait s", "other s"))
for L in sorted(pl):
    s, fe, op, wa = pl[L]
    print("  %-6d %9.3f %9.3f %9.3f %9.3f %9.3f" % (L, s, fe, op, wa, s - fe - op - wa))
tot = [sum(pl[L][i] for L in pl) for i in range(4)]
print("  %-6s %9.3f %9.3f %9.3f %9.3f %9.3f"
      % ("TOTAL", tot[0], tot[1], tot[2], tot[3], tot[0] - tot[1] - tot[2] - tot[3]))
