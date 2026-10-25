#!/usr/bin/env python3
# Reads verify/out/mut/<name>.<tag>/units.log and prints one line per mutant and run: failing units / unit lines, failing per scenario, first failure (lowest seed of the first failing scenario).
# usage: analyze.py [NAME ...]   (default: every directory in out/mut)
import glob, os, re, sys
ORDER = ["barrier", "wait", "stress", "race", "init2", "fifo", "open", "shm", "natural", "real", "fatal"]
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "mut")
names = sys.argv[1:]
runs = {}
for d in sorted(glob.glob(os.path.join(base, "*.*"))):
    if not os.path.isdir(d):
        continue
    nm, tag = os.path.basename(d).rsplit(".", 1)
    if names and nm not in names:
        continue
    runs.setdefault(nm, []).append((tag, d))
for nm in sorted(runs):
    for tag, d in sorted(runs[nm]):
        lines = [l.rstrip("\n") for l in open(os.path.join(d, "units.log"))] if os.path.exists(os.path.join(d, "units.log")) else []
        units = [l for l in lines if l.startswith("UNIT ")]
        keys = [tuple(l.split()[1:3]) for l in units]
        dup = len(keys) - len(set(keys))
        jobs = sum(1 for _ in open(os.path.join(d, "jobs.txt"))) if os.path.exists(os.path.join(d, "jobs.txt")) else -1
        bad = [l for l in units if not re.match(r"UNIT \S+ \d+ PASS ", l)]
        fails = [l for l in bad if re.match(r"UNIT \S+ \d+ FAIL ", l)]
        refbad = [l for l in bad if re.match(r"UNIT \S+ \d+ REFBAD ", l)]
        per = {}
        for l in fails:
            per[l.split()[1]] = per.get(l.split()[1], 0) + 1
        tot = {}
        for l in units:
            tot[l.split()[1]] = tot.get(l.split()[1], 0) + 1
        first = ""
        for s in ORDER:
            c = sorted((int(l.split()[2]), l) for l in fails if l.split()[1] == s)
            if c:
                first = c[0][1].split(" | ", 1)
                first = "%s %s: %s" % (s, c[0][0], first[1] if len(first) > 1 else "")
                break
        pers = " ".join("%s=%d/%d" % (s, per.get(s, 0), tot.get(s, 0)) for s in ORDER if s in tot)
        print("%s.%s units=%d/%d dup=%d FAIL=%d REFBAD=%d | %s | first: %s" % (nm, tag, len(units), jobs, dup, len(fails), len(refbad), pers, first[:260]))
