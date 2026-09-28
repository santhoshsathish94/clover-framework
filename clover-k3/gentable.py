#!/usr/bin/env python3
"""Generate a continuation for every prompt in prompts.tsv, with prefix reuse.

Writes one row per prompt as it goes, so a partial run is still usable.
  gentable.py [tokens]
"""
import os
import re
import subprocess
import sys
import time

HERE = "/opt/clover-k3"
OUT = "/root/k3gen"
sys.path.insert(0, HERE)
import gen  # noqa: E402  the driver, for show() and step()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

os.makedirs(OUT, exist_ok=True)
os.makedirs(gen.WORK, exist_ok=True)
done = os.path.join(OUT, "DONE")
tsv = os.path.join(OUT, "gen.tsv")
if os.path.exists(done):
    os.remove(done)

rows = []
for line in open(os.path.join(HERE, "prompts.tsv")):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 4:
        rows.append((int(f[0]), int(f[1]), [int(x) for x in f[2].split(",")], f[3]))

with open(tsv, "w", encoding="utf-8") as fh:
    fh.write("idx\tnpos\tntok\ttotal_s\ts_per_tok\tprompt\tgenerated\n")

for idx, npos, ids, text in rows:
    a = os.path.join(gen.WORK, "a.bin")
    b = os.path.join(gen.WORK, "b.bin")
    out, total, load = [], 0.0, None
    t0 = time.time()
    for i in range(N):
        dst = a if i % 2 == 0 else b
        tok, dt, _ = gen.step(ids + out, load, dst)
        total += dt
        out.append(tok)
        load = dst
    wall = time.time() - t0
    gentext = "".join(gen.show(t) for t in out)
    with open(tsv, "a", encoding="utf-8") as fh:
        fh.write("%d\t%d\t%d\t%.1f\t%.2f\t%s\t%s\n"
                 % (idx, npos, N, wall, wall / N, text,
                    gentext.replace("\t", "\\t").replace("\n", "\\n")))
    with open(os.path.join(OUT, "progress.txt"), "a", encoding="utf-8") as fh:
        fh.write("%s p%d %.1fs %r\n" % (time.strftime("%H:%M:%S"), idx, wall, gentext))
    print("p%-2d %6.1f s  %r" % (idx, wall, gentext), flush=True)

open(done, "w").close()
