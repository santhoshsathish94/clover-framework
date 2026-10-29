#!/usr/bin/env python3
"""Driver v2: how many host steps run in each gate mode (0 free running, 1 barrier = >= 16 chunks and all 16 workers must map, 2 serial = the calling thread maps every chunk), counted from the
configurations ('pi_driver cfg' prints the gate of every host step). Usage: gates.py ./pi_driver [H=600,HX=400,...]   (the counts of one run, default as in run.sh)"""
import collections, re, subprocess, sys

driver = sys.argv[1]
counts = {"H": 600, "HX": 400, "PR": 200, "M": 400}
if len(sys.argv) > 2:
    counts = {}
    for kv in sys.argv[2].split(","):
        k, v = kv.split("=")
        counts[k] = int(v)
tot = collections.Counter()
for scn, n in counts.items():
    c = collections.Counter()
    for seed in range(1, n + 1):
        out = subprocess.run([driver, "cfg", "--scn", scn, "--seed", str(seed)], stdout=subprocess.PIPE, check=True).stdout.decode().strip()
        steps = out.split(" | ")[1:]
        gates = re.findall(r"gate=(\d)", out)
        for part, g in zip([p for p in steps if p.startswith("H ")], gates):
            kv = dict(re.findall(r"(\w+)=(\S+)", part))
            nch = int(kv["bytes"]) // int(kv["raw"]) // 32
            c["host steps"] += 1
            c["gate %s" % g] += 1
            c["chunks>=16"] += nch >= 16
            c["chunks>=2"] += nch >= 2
            if g == "2":
                c["serial with chunks>=2"] += 1
    print(scn, dict(sorted(c.items())))
    tot.update(c)
print("all", dict(sorted(tot.items())))
