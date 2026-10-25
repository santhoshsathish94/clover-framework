#!/usr/bin/env python3
# Counts caught units of every mutant run from out/mut/<name>/units.log (UNIT <scenario> <seed> PASS|FAIL ...) and prints the first failure with the first difference.
# "first failure" = the lowest failing seed of the first scenario (order P1..P6) that has a failing unit.
# usage: analyze.py [--md] NAME...   (run in verify/)   --md prints one markdown table row per mutant (needs mutants/<name>.diff first line for the description)
import os, re, sys, hashlib, collections
ORDER = ["P1", "P2", "P3", "P4", "P5", "P6"]
EXPECT = {"P1": 500, "P2": 500, "P3": 36, "P4": 500, "P5": 600, "P6": 100}
args = sys.argv[1:]
md = "--md" in args
names = [a for a in args if a != "--md"]
for name in names:
    base = "out/mut/" + name
    so = "work/mut/%s/target/release/libplanner.so" % name
    md5 = hashlib.md5(open(so, "rb").read()).hexdigest() if os.path.exists(so) else "-"
    desc = open("mutants/%s.diff" % name).readline().strip("# \n") if os.path.exists("mutants/%s.diff" % name) else ""
    units = {}
    dup = 0
    try:
        for line in open(base + "/units.log", errors="replace"):
            m = re.match(r"UNIT (\S+) (\d+) (PASS|FAIL) (.*)", line)
            if m:
                k = (m.group(1), int(m.group(2)))
                if k in units: dup += 1
                units[k] = (m.group(3), m.group(4).strip())
    except FileNotFoundError:
        print(name, "NO units.log"); continue
    tot = collections.Counter(k[0] for k in units)
    fail = collections.Counter(k[0] for k, v in units.items() if v[0] == "FAIL")
    nf = sum(fail.values())
    first = None; refl = rustl = viol = ""
    for s in ORDER:
        fs = sorted(k[1] for k, v in units.items() if k[0] == s and v[0] == "FAIL")
        if not fs: continue
        sd = fs[0]; first = (s, sd, units[(s, sd)][1])
        d = "%s/%s/u%d.diff" % (base, s, sd)
        if os.path.exists(d):
            ls = [l for l in open(d, errors="replace").read().splitlines() if l.startswith(">>")]
            if len(ls) >= 2: refl, rustl = ls[0], ls[1]
        for who in ("rust", "ref"):
            lg = "%s/%s/u%d.%s.log" % (base, s, sd, who)
            if os.path.exists(lg):
                for l in open(lg, errors="replace"):
                    if l.startswith("VIOLATION"):
                        viol += "%s: %s" % (who, l.strip()[:200]); break
        break
    cnt = " ".join("%s=%d/%d" % (s, fail[s], tot[s]) for s in ORDER)
    if md:
        print("| %s | %s | %s | %s | %d/%d | %s | %s |" % (name, desc, md5, "CAUGHT" if nf else "NOT CAUGHT", nf, len(units), cnt,
              ("%s %d" % (first[0], first[1])) if first else "-"))
        continue
    print("== %s md5=%s units=%d (expected %d) PASS=%d FAIL=%d dup=%d  %s" % (name, md5, len(units), sum(EXPECT.values()), len(units) - nf, nf, dup, "CAUGHT" if nf else "NOT CAUGHT"))
    print("   per scenario fail/total: " + cnt)
    if first:
        print("   first failing: %s %d: %s" % first)
        if refl: print("      ref  " + refl[:240]); print("      rust " + rustl[:240])
        if viol: print("      " + viol)
