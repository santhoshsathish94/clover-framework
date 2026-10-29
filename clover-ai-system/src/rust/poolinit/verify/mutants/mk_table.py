#!/usr/bin/env python3
# Builds the result table of verify/mutants/RESULTS.md from out/mut/results.tsv (analyze.py), md5list.txt (mk_mutants.py) and verdicts.tsv (hand-written verdict + note for every
# mutant that is not simply "caught by driver"). Prints markdown to stdout.
# usage: mk_table.py [results.tsv] [verdicts.tsv]
import os, sys

V = os.path.expanduser("~/rustmig/poolinit/verify")
res = {}
for l in open(sys.argv[1] if len(sys.argv) > 1 else V + "/out/mut/results.tsv", errors="replace"):
    p = l.rstrip("\n").split("\t")
    res[p[0]] = p
desc = {}
for l in open(V + "/mutants/md5list.txt"):
    n, md5, d = l.rstrip("\n").split(" ", 2)
    desc[n] = (md5, d)
ver = {}
vf = sys.argv[2] if len(sys.argv) > 2 else V + "/mutants/verdicts.tsv"
if os.path.exists(vf):
    for l in open(vf):
        if l.strip() and not l.startswith("#"):
            p = l.rstrip("\n").split("\t")
            ver[p[0]] = p[1:]

def key(n):
    return int(n[1:]) if n[1:].isdigit() else 999

print("| mutant | change | .so md5 | verdict | failing units / 3000 | failing per scenario | first failure |")
print("|---|---|---|---|---|---|---|")
for n in sorted(res, key=key):
    r = res[n]
    md5, d = desc.get(n, ("?", "?"))
    v = ver.get(n)
    verdict = v[0] if v else ("control, passes" if n == "M00" else "caught by driver")
    first = r[7].replace("|", "\\|").replace("`", "'")[:230] if len(r) > 7 and r[7] else "-"
    if r[6] != "-":
        first = r[6] + ": " + first
    print("| %s | %s | %s | %s | %s | %s | %s |" % (n, d.replace("|", "\\|"), md5, verdict, r[4], r[5], first))
