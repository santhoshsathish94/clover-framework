#!/usr/bin/env python3
"""Where the Q projection's 1.38 s actually goes, call site by call site.

"Q int8 projection: 1.38 s" is an aggregate over about 1159 calls a token, and
an aggregate cannot say what to change. This enumerates every Qm() call in the
layer body from clover-one.c, with the vectors it maps between and the bytes it
must stream, then checks the total against the 53.8251 GB OPSTAT reports. If the
enumeration is wrong the total will not match and this says so rather than
producing a plausible table.

Every int8 matrix is out x (in + 4) bytes: `in` weights a row plus a float32
row scale. At decode each call is a matrix-vector product - one `in`-length
activation against the matrix, producing `out` floats - so it moves the whole
matrix to do 2 FLOP a byte.

usage: q-breakdown.py
"""

E, P, H, D = 7168, 12288, 96, 128
QH, VH, KVL, KVD, KVW, QLORA = 192, 128, 512, 256, 576, 1536
LAT, SI, DI = 3584, 6144, 33792

MEASURED_GB = 53.8251
MEASURED_S = 1.3804
N_MLA, N_KDA, N_MOE = 24, 68, 92


def b(out, inn):
    return out * (inn + 4)


# (group, label, in, out, how many layers of that kind, calls each)
CALLS = [
    ("KDA attention", "Q  raw      x -> heads", E, P, N_KDA, 1),
    ("KDA attention", "K  raw      x -> heads", E, P, N_KDA, 1),
    ("KDA attention", "V  raw      x -> heads", E, P, N_KDA, 1),
    ("KDA attention", "G  gate     x -> heads", E, P, N_KDA, 1),
    ("KDA attention", "O  out   heads -> x   ", P, E, N_KDA, 1),
    ("KDA small",     "B  beta     x -> head ", E, H, N_KDA, 1),
    ("KDA small",     "FA          x -> D    ", E, D, N_KDA, 1),
    ("KDA small",     "FB          D -> heads", D, P, N_KDA, 1),
    ("MLA attention", "G  gate     x -> heads", E, H * VH, N_MLA, 1),
    ("MLA attention", "O  out   heads -> x   ", H * VH, E, N_MLA, 1),
    ("MLA low-rank",  "QA   x -> q latent    ", E, QLORA, N_MLA, 1),
    ("MLA low-rank",  "QB  q latent -> heads ", QLORA, H * QH, N_MLA, 1),
    ("MLA low-rank",  "KA   x -> kv latent   ", E, KVW, N_MLA, 1),
    ("MLA low-rank",  "KB kv latent -> heads ", KVL, H * KVD, N_MLA, 1),
    ("MoE latent",    "DN   x -> latent      ", E, LAT, N_MOE, 1),
    ("MoE latent",    "UP latent -> x        ", LAT, E, N_MOE, 1),
    ("MoE shared",    "SH1 gate x -> inter   ", E, SI, N_MOE, 1),
    ("MoE shared",    "SH3 up   x -> inter   ", E, SI, N_MOE, 1),
    ("MoE shared",    "SH2 down inter -> x   ", SI, E, N_MOE, 1),
    ("layer 0 dense", "MGATE  x -> 33792     ", E, DI, 1, 1),
    ("layer 0 dense", "MUP    x -> 33792     ", E, DI, 1, 1),
    ("layer 0 dense", "MDOWN 33792 -> x      ", DI, E, 1, 1),
]

rows = []
total = 0
calls = 0
for group, label, inn, out, layers, each in CALLS:
    per = b(out, inn)
    tot = per * layers * each
    total += tot
    calls += layers * each
    rows.append((group, label, inn, out, per, layers, tot))

print("%-15s %-24s %7s %7s %10s %7s %12s %7s"
      % ("group", "projection", "in", "out", "MB each", "layers", "GB total", "share"))
for group, label, inn, out, per, layers, tot in sorted(rows, key=lambda r: -r[6]):
    print("%-15s %-24s %7d %7d %10.2f %7d %12.3f %6.1f%%"
          % (group, label, inn, out, per / 1e6, layers, tot / 1e9, 100.0 * tot / total))

print()
print("%-40s %12.3f GB in %d calls" % ("enumerated total", total / 1e9, calls))
print("%-40s %12.4f GB in %d calls" % ("OPSTAT reports", MEASURED_GB, 1159))
d = abs(total / 1e9 - MEASURED_GB) / MEASURED_GB
print("%-40s %12.2f%%" % ("difference", 100.0 * d))
if d > 0.05:
    print("ENUMERATION IS WRONG - more than 5% off, do not trust the table")
else:
    print("reconciles; the remainder is layer 0's attention and the lm_head")

print()
print("time at the measured %.4f s, apportioned by bytes" % MEASURED_S)
byg = {}
for group, label, inn, out, per, layers, tot in rows:
    byg[group] = byg.get(group, 0) + tot
print("  %-16s %10s %9s %9s" % ("group", "GB", "share", "seconds"))
for g, v in sorted(byg.items(), key=lambda kv: -kv[1]):
    print("  %-16s %10.3f %8.1f%% %9.3f" % (g, v / 1e9, 100.0 * v / total,
                                            MEASURED_S * v / total))
