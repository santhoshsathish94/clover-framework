#!/usr/bin/env python3
# Builds the comparison table of the driver-v2 campaign (appended to RESULTS.md) from the gzipped units.log files of verify/mutants/v2out/units (or --dir) and the analysis.txt of the first campaign.
# usage: mk_results2.py [--dir DIR] [--old analysis.txt] [--md5old md5-so.txt --md5new md5-so-now.txt]
import glob, gzip, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
ORDER = ["barrier", "wait", "stress", "race", "init2", "fifo", "open", "shm", "natural", "real", "fatal", "book", "hotwait", "race2", "big", "fd0", "stack", "guard"]
NEW = ["book", "hotwait", "race2", "big", "fd0", "stack", "guard"]
args = sys.argv[1:]; d = os.path.join(here, "v2out", "units"); old = os.path.join(here, "analysis.txt"); m_old = os.path.join(here, "md5-so.txt"); m_new = os.path.join(here, "v2out", "md5-so-now.txt")
while args:
    a = args.pop(0)
    if a == "--dir": d = args.pop(0)
    elif a == "--old": old = args.pop(0)
    elif a == "--md5old": m_old = args.pop(0)
    elif a == "--md5new": m_new = args.pop(0)
descr = {}
for l in open(os.path.join(here, "md5.txt")):
    p = l.rstrip("\n").split("  ", 2)
    if len(p) == 3: descr[p[1]] = p[2]
runs = {}
for f in sorted(glob.glob(os.path.join(d, "*.units.log.gz"))):
    nm, tag = os.path.basename(f)[:-len(".units.log.gz")].rsplit(".", 1)
    runs.setdefault(nm, {})[tag] = [l.rstrip("\n") for l in gzip.open(f, "rt", errors="replace") if l.startswith("UNIT ")]
oldr = {}
for l in open(old):
    m = re.match(r"(\S+)\.(\S+) units=(\d+)/(\d+) dup=\d+ FAIL=(\d+) ", l)
    if m: oldr.setdefault(m.group(1), {})[m.group(2)] = (int(m.group(5)), int(m.group(3)), int(m.group(4)))
def md5s(p):
    r = {}
    if os.path.exists(p):
        for l in open(p):
            q = l.split()
            if len(q) == 2: r[os.path.basename(q[1])[:-3]] = q[0]
    return r
mo, mn = md5s(m_old), md5s(m_new)
def failing(units): return [l for l in units if re.match(r"UNIT \S+ \d+ FAIL ", l)]
def first(units):
    f = failing(units)
    for s in ORDER:
        c = sorted((int(l.split()[2]), l) for l in f if l.split()[1] == s)
        if c:
            p = c[0][1].split(" | ", 1); return "%s %s: %s" % (s, c[0][0], (p[1] if len(p) > 1 else "")[:150])
    return "-"
def per(units):
    f = failing(units); r = {}
    for l in f: r[l.split()[1]] = r.get(l.split()[1], 0) + 1
    return " ".join("%s=%d" % (s, r[s]) for s in ORDER if s in r)
rows = []; changed = []; regress = []; nb = nn = 0
for nm in sorted(set(runs), key=lambda n: ([0 if n == "M00" else 1, int(re.sub(r"\D", "", n) or 0) if n[:2] == "M" and n[1].isdigit() else 999, n])):
    o = oldr.get(nm, {}); ob = sum(v[0] for v in o.values()) > 0 if o else None
    r = runs.get(nm, {}); a = r.get("v2a"); nbool = any(failing(u) for u in r.values() if u) if r else None
    if nm == "M00": ob = False
    if ob: nb += 1
    if nbool: nn += 1
    if ob is not None and nbool is not None and ob != nbool: changed.append(nm)
    if ob and nbool is False: regress.append(nm)
    o1 = o.get("r1"); before = ("%d / %d" % (o1[0], o1[1])) if o1 else "-"
    cnt = "-"
    if a is not None:
        cnt = "%d / %d" % (len(failing(a)), len(a))
        reps = [t for t in ("v2a", "v2b", "v2c") if t in r]
        if len(reps) > 1: cnt += " (runs catching: %d/%d)" % (sum(1 for t in reps if failing(r[t])), len(reps))
        if "v2s" in r: cnt += "; soak %d / %d" % (len(failing(r["v2s"])), len(r["v2s"]))
    same = "" if (nm not in mo or nm not in mn) else ("same" if mo[nm] == mn[nm] else "DIFFERENT")
    rows.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (nm, descr.get(nm, "")[:80].replace("|", "/"), "caught" if ob else ("not caught" if ob is not None else "-"), before, "caught" if nbool else ("not caught" if nbool is not None else "-"), cnt, per(a) if a else "-", first(a)[:200].replace("|", "/") if a else "-"))
print("| mutant | change | before (first campaign) | failing units / units run, first campaign r1 | now | failing units / units run, driver v2 (v2a; repeat runs; soak N=1000) | failing per scenario (v2a) | first failure now (lowest seed of the first failing scenario) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows: print(r)
print()
print("caught before: %d, caught now: %d (of %d mutants, the control M00 not counted); verdict changed: %s; caught before and not caught now (regressions): %s" % (nb, nn, len([n for n in runs if n != "M00"]), ", ".join(changed) or "none", ", ".join(regress) or "none"))
diff_md5 = [n for n in mn if n in mo and mo[n] != mn[n]]
print("md5 of the rebuilt .so against the first campaign's md5-so.txt: %d compared, %d differ%s" % (len([n for n in mn if n in mo]), len(diff_md5), (": " + ", ".join(sorted(diff_md5))) if diff_md5 else ""))
