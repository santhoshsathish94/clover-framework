#!/usr/bin/env python3
# Reads the gzipped units.log of every driver-v2 run (verify/out/mut2/units/NAME.TAG.units.log.gz, written by run_mut2.sh) and prints one line per run in the format of analyze.py (failing units / unit
# lines, failing per scenario, first failure), then, with --compare <analysis.txt of the first campaign>, one line per mutant: caught before (any run with a failing unit) vs caught now, and
# the mutants whose verdict changed. usage: analyze2.py [--dir DIR] [--compare OLD_ANALYSIS] [--tags TAG,TAG] [NAME ...]
import glob, gzip, os, re, sys
ORDER = ["barrier", "wait", "stress", "race", "init2", "fifo", "open", "shm", "natural", "real", "fatal", "book", "hotwait", "race2", "big", "fd0", "stack", "guard"]
OLD = ["barrier", "wait", "stress", "race", "init2", "fifo", "open", "shm", "natural", "real", "fatal"]
args = sys.argv[1:]
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "mut2", "units"); old = None; tags = None; names = []
while args:
    a = args.pop(0)
    if a == "--dir": d = args.pop(0)
    elif a == "--compare": old = args.pop(0)
    elif a == "--tags": tags = args.pop(0).split(",")
    else: names.append(a)
runs = {}
for f in sorted(glob.glob(os.path.join(d, "*.units.log.gz"))):
    nm, tag = os.path.basename(f)[:-len(".units.log.gz")].rsplit(".", 1)
    if names and nm not in names: continue
    if tags and tag not in tags: continue
    runs.setdefault(nm, {})[tag] = [l.rstrip("\n") for l in gzip.open(f, "rt", errors="replace")]
new_any = {}; new_fail = {}
for nm in sorted(runs):
    for tag in sorted(runs[nm]):
        units = [l for l in runs[nm][tag] if l.startswith("UNIT ")]
        keys = [tuple(l.split()[1:3]) for l in units]
        dup = len(keys) - len(set(keys))
        fails = [l for l in units if re.match(r"UNIT \S+ \d+ FAIL ", l)]
        refbad = [l for l in units if re.match(r"UNIT \S+ \d+ REFBAD ", l)]
        per = {}; tot = {}
        for l in fails: per[l.split()[1]] = per.get(l.split()[1], 0) + 1
        for l in units: tot[l.split()[1]] = tot.get(l.split()[1], 0) + 1
        first = ""
        for s in ORDER:
            c = sorted((int(l.split()[2]), l) for l in fails if l.split()[1] == s)
            if c:
                p = c[0][1].split(" | ", 1)
                first = "%s %s: %s" % (s, c[0][0], p[1] if len(p) > 1 else ""); break
        pers = " ".join("%s=%d/%d" % (s, per.get(s, 0), tot.get(s, 0)) for s in ORDER if s in tot)
        print("%s.%s units=%d dup=%d FAIL=%d REFBAD=%d | %s | first: %s" % (nm, tag, len(units), dup, len(fails), len(refbad), pers, first[:260]))
        new_fail[(nm, tag)] = len(fails)
        new_any[nm] = new_any.get(nm, 0) + len(fails)
if old:
    old_any = {}; old_runs = {}
    for l in open(old):
        m = re.match(r"(\S+)\.(\S+) units=(\d+)/(\d+) dup=\d+ FAIL=(\d+) ", l)
        if m:
            old_any[m.group(1)] = old_any.get(m.group(1), 0) + int(m.group(5)); old_runs.setdefault(m.group(1), []).append("%s:%s/%s" % (m.group(2), m.group(5), m.group(3)))
    print("\n== comparison with the first campaign (%s)" % old)
    changed = []; lost = []
    allnames = sorted(new_any)   # only what this campaign ran (the probe builds PM* of the first campaign are not mutants)
    for nm in allnames:
        o = old_any.get(nm); n = new_any.get(nm)
        ov = "n/a" if o is None else ("caught" if o > 0 else "not caught"); nv = "n/a" if n is None else ("caught" if n > 0 else "not caught")
        mark = "" if ov == nv else "  <== CHANGED"
        if ov != nv: changed.append(nm)
        if ov == "caught" and nv == "not caught": lost.append(nm)
        print("CMP %-5s before: %-10s (%s)  now: %-10s (failing units over all my runs %s)%s" % (nm, ov, " ".join(old_runs.get(nm, [])), nv, n, mark))
    print("caught before: %d  caught now: %d  changed: %s  caught before and not now (regressions): %s" % (sum(1 for n in allnames if old_any.get(n, 0) > 0), sum(1 for n in allnames if new_any.get(n, 0) > 0), changed, lost))
