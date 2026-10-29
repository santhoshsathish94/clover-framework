#!/usr/bin/env python3
# Reads out/mut/<name>/units.log of every mutant and writes mutants/logs/table.md (one row per mutant) and mutants/logs/analysis.txt (first failure lines).
# CAUGHT = at least one `UNIT <scn> <seed> FAIL` line in units.log. The counts are re-done with grep in count_check.sh.
# "failing per scenario" lists only scenarios with failures (totals: Y1-Y4, Y6-Y9, T1 60 units each; Y5, YF 80 units each).
import os, re, sys, glob

HOME = os.path.expanduser("~")
V = os.path.join(HOME, "rustmig/supply/verify")
OUT = os.path.join(V, "out/mut")
LOGS = os.path.join(V, "mutants/logs")
os.makedirs(LOGS, exist_ok=True)
ORDER = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "YF", "T1"]
desc = {}; md5 = {}
for p in sorted(glob.glob(os.path.join(V, "mutants/M*.diff"))):
    n = os.path.basename(p)[:-5]
    first = open(p).readline().strip()
    desc[n] = first.split(": ", 1)[1] if ": " in first else first
for line in open(os.path.join(OUT, "progress.txt")):
    mm = re.match(r"\S+ (\S+) start md5=([0-9a-f]{32})", line)
    if mm: md5[mm.group(1)] = mm.group(2)

def first_failure(name, fails):
    for scn in ORDER:
        seeds = sorted(s for (c, s, _) in fails if c == scn)
        if not seeds: continue
        s = seeds[0]
        reason = [r for (c, sd, r) in fails if c == scn and sd == s][0]
        d = os.path.join(OUT, name, scn, "u%d.diff" % s)
        lines = []
        if os.path.exists(d):
            for l in open(d, errors="replace"):
                if l.startswith(">>"): lines.append(re.sub(r"^>>\d+: ", "", l.rstrip("\n")))
        return scn, s, reason, lines
    return None

rows = []; ana = []
for name in sorted(desc):
    ul = os.path.join(OUT, name, "units.log")
    if not os.path.exists(ul):
        rows.append("| %s | %s | %s | NOT RUN | - | - | - |" % (name, desc[name], md5.get(name, "-"))); continue
    seen = {}; fails = []
    for l in open(ul, errors="replace"):
        mm = re.match(r"UNIT (\S+) (\d+) (PASS|FAIL) lines=\S+ (.*)$", l.rstrip("\n"))
        if not mm: continue
        scn, sd, res, rest = mm.group(1), int(mm.group(2)), mm.group(3), mm.group(4)
        seen[(scn, sd)] = seen.get((scn, sd), 0) + 1
        if res == "FAIL":
            fails.append((scn, sd, rest.split(" cov=", 1)[-1].split(" ", 1)[-1] if " cov=" in rest else rest))
    total = len(seen); dup = sum(1 for v in seen.values() if v > 1)
    per = " ".join("%s=%d" % (s, sum(1 for f in fails if f[0] == s)) for s in ORDER if any(f[0] == s for f in fails)) or "-"
    ff = first_failure(name, fails)
    verdict = "driver FAIL" if fails else "not caught by driver"
    if total != 700 or dup: verdict += " (UNIT lines %d, dup %d)" % (total, dup)
    if ff:
        scn, sd, reason, ls = ff
        ana.append("== %s (%s) first failure %s %d: reason=%s" % (name, desc[name], scn, sd, reason))
        for x in ls: ana.append("   " + x[:300])
        def short(x): return (x[:70] + "..") if len(x) > 72 else x
        if ls:
            first = "%s %d: ref `%s` / rust `%s`" % (scn, sd, short(ls[0]), short(ls[1]) if len(ls) > 1 else "")
        else:
            first = "%s %d: %s" % (scn, sd, reason)
    else:
        first = "-"
    rows.append("| %s | %s | %s | %s | %d / %d | %s | %s |" % (name, desc[name], md5.get(name, "-"), verdict, len(fails), total, per, first.replace("|", "/")))
open(os.path.join(LOGS, "table.md"), "w").write("| mutant | change | .so md5 | driver | failing units / total | failing units per scenario | first failure (ref / rust) |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
open(os.path.join(LOGS, "analysis.txt"), "w").write("\n".join(ana) + "\n")
print("written", len(rows), "rows;", sum(1 for r in rows if "| driver FAIL" in r), "caught;", sum(1 for r in rows if "| not caught" in r), "not caught")
