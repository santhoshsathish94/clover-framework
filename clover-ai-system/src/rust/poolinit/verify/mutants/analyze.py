#!/usr/bin/env python3
# Reads out/mut/<name>/ of one mutant run and prints / records: UNIT counts (duplicates, missing), failing units per scenario, the first failure
# (lowest seed of the first scenario in the order H HX P PX PR M that has a failure; its two differing lines from the .diff, or the VIOLATION line of the Rust log),
# and copies the raw logs of that first failing unit to verify/mutants/logs/<name>/ (small files only). Appends one TSV line to out/mut/results.tsv.
# usage: analyze.py NAME [md5]
import os, re, shutil, sys

V = os.path.expanduser("~/rustmig/poolinit/verify")
ORDER = ["H", "HX", "P", "PX", "PR", "M"]
SUB = os.environ.get("SUB", "mut")
MULT = int(os.environ.get("MULT", "1"))
EXPECT = {s: n * MULT for s, n in {"H": 600, "HX": 400, "P": 600, "PX": 800, "PR": 200, "M": 400}.items()}

def first_lines(out, scn, seed):
    base = "%s/%s/u%d" % (out, scn, seed)
    res = []
    if os.path.exists(base + ".diff"):
        for l in open(base + ".diff", errors="replace"):
            if l.startswith(">>"):
                res.append(l.strip()[:260])
    rl = base + ".rust.log"
    if os.path.exists(rl):
        for l in open(rl, errors="replace"):
            if l.startswith("VIOLATION"):
                res.append("RUST " + l.strip()[:300]); break
    return res

def main():
    name = sys.argv[1]
    md5 = sys.argv[2] if len(sys.argv) > 2 else "-"
    out = "%s/out/%s/%s" % (V, SUB, name)
    ul = out + "/units.log"
    seen, fails, tot, reason = {}, {}, {}, {}
    dup = 0
    if os.path.exists(ul):
        for l in open(ul, errors="replace"):
            if not l.startswith("UNIT "):
                continue
            p = l.split()
            scn, seed, res = p[1], int(p[2]), p[3]
            if (scn, seed) in seen:
                dup += 1
            seen[(scn, seed)] = res
            reason[(scn, seed)] = " ".join(p[5:])
    for (scn, seed), res in seen.items():
        tot[scn] = tot.get(scn, 0) + 1
        if res != "PASS":
            fails.setdefault(scn, []).append(seed)
    nf = sum(len(v) for v in fails.values())
    per = " ".join("%s=%d/%d" % (s, len(fails.get(s, [])), tot.get(s, 0)) for s in ORDER)
    miss = {s: EXPECT[s] - tot.get(s, 0) for s in ORDER if tot.get(s, 0) != EXPECT[s]}
    first = "-"; fl = []
    for s in ORDER:
        if fails.get(s):
            sd = min(fails[s]); first = "%s %d" % (s, sd); fl = first_lines(out, s, sd) or [reason[(s, sd)]]
            d = "%s/mutants/logs/%s" % (V, name)
            os.makedirs(d, exist_ok=True)
            for ext in (".ref.log", ".rust.log", ".ref.err", ".rust.err", ".diff", ".res"):
                f = "%s/%s/u%d%s" % (out, s, sd, ext)
                if os.path.exists(f) and os.path.getsize(f) < 5 * 1024 * 1024:
                    shutil.copy(f, "%s/%s_u%d%s" % (d, s, sd, ext))
            break
    verdict = "CAUGHT" if nf else "NOT CAUGHT"
    txt = "%s %s md5=%s units=%d dup=%d missing=%s fail=%d %s first=%s" % (name, verdict, md5, len(seen), dup, miss or "none", nf, per, first)
    print(txt)
    for l in fl:
        print("   " + l)
    with open("%s/out/%s/results.tsv" % (V, SUB), "a") as f:
        f.write("\t".join([name, md5, verdict, str(len(seen)), str(nf), per, first, " || ".join(fl)[:700].replace("\t", " ")]) + "\n")

main()
