#!/usr/bin/env python3
# Driver v2: reads out/mut/<name>/units.log of every mutant (run on the v2 driver) and compares with the verdicts of the independent verifier's first pass
# (mutants/logs/table.md, read only). Writes v2/logs/table-v2.md and v2/logs/compare-v2.txt (nothing under mutants/ is written).
# CAUGHT = at least one `UNIT <scn> <seed> FAIL` line in units.log (same rule as mutants/analyze.py).
import glob, os, re, sys

V = os.path.expanduser("~/rustmig/supply/verify")
OUT = os.path.join(V, "out/mut")
LOGS = os.path.join(V, "v2/logs")
os.makedirs(LOGS, exist_ok=True)
OLD_ORDER = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "YF", "T1"]
NEW_ORDER = ["Y10", "Y11", "Y12", "Y13", "Y14", "YG", "T2"]
ORDER = OLD_ORDER + NEW_ORDER
EXPECT = int(sys.argv[1]) if len(sys.argv) > 1 else 1120

old = {}
for l in open(os.path.join(V, "mutants/logs/table.md"), errors="replace"):
    m = re.match(r"\| (M\d\d) \| (.*) \| (?:[0-9a-f]{32}|-) \| (driver FAIL|not caught by driver)[^|]*\| (\d+) / (\d+) \|", l)
    if m:
        old[m.group(1)] = (m.group(3) == "driver FAIL", int(m.group(4)), m.group(2))

def first_failure(name, fails, prefer):
    for scn in prefer + [s for s in ORDER if s not in prefer]:
        seeds = sorted(s for (c, s, _) in fails if c == scn)
        if not seeds:
            continue
        s = seeds[0]
        d = os.path.join(OUT, name, scn, "u%d.diff" % s)
        lines = []
        if os.path.exists(d):
            for x in open(d, errors="replace"):
                if x.startswith(">>"):
                    lines.append(re.sub(r"^>>\d+: ", "", x.rstrip("\n")))
        reason = [r for (c, sd, r) in fails if c == scn and sd == s][0]
        return scn, s, lines, reason
    return None

rows, cmp_lines, new = [], [], {}
for d in sorted(glob.glob(OUT + "/M[0-9]*/")):
    name = os.path.basename(d.rstrip("/"))
    seen, fails = {}, []
    for l in open(d + "units.log", errors="replace"):
        m = re.match(r"UNIT (\S+) (\d+) (PASS|FAIL) lines=\S+ (.*)$", l.rstrip("\n"))
        if not m:
            continue
        scn, sd, res, rest = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        seen[(scn, sd)] = seen.get((scn, sd), 0) + 1
        if res == "FAIL":
            fails.append((scn, sd, rest.split(" cov=", 1)[-1].split(" ", 1)[-1] if " cov=" in rest else rest))
    total, dup = len(seen), sum(1 for v in seen.values() if v > 1)
    per = " ".join("%s=%d" % (s, sum(1 for f in fails if f[0] == s)) for s in ORDER if any(f[0] == s for f in fails)) or "-"
    was = old.get(name)
    prefer = NEW_ORDER if (was and not was[0]) else []
    ff = first_failure(name, fails, prefer)
    if ff:
        scn, sd, ls, reason = ff
        short = lambda x: (x[:70] + "..") if len(x) > 72 else x
        first = ("%s %d: ref `%s` / rust `%s`" % (scn, sd, short(ls[0]), short(ls[1]) if len(ls) > 1 else "")) if ls else "%s %d: %s" % (scn, sd, reason)
    else:
        first = "-"
    new[name] = (bool(fails), len(fails), total, dup)
    rows.append("| %s | %s | %s | %d / %d | %s | %s |" % (name, was[2] if was else "?", "driver FAIL" if fails else "not caught by driver", len(fails), total, per, first.replace("|", "/")))
    if total != EXPECT or dup:
        cmp_lines.append("WARNING %s: UNIT lines %d (expected %d), duplicates %d" % (name, total, EXPECT, dup))

open(os.path.join(LOGS, "table-v2.md"), "w").write("| mutant | change | driver v2 | failing units / total | failing units per scenario | first failure (ref / rust) |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")

names = sorted(set(old) | set(new))
mut_names = [n for n in names if n != "M00"]
caught_old = [n for n in mut_names if old.get(n, (False,))[0]]
caught_new = [n for n in mut_names if new.get(n, (False,))[0]]
regress = [n for n in caught_old if n not in caught_new]
gained = [n for n in caught_new if n not in caught_old]
lost_equiv = [n for n in mut_names if n not in caught_old and n not in caught_new]
cmp_lines.append("mutants compared: %d (old table %d, new runs %d)" % (len(mut_names), len([n for n in old if n != 'M00']), len([n for n in new if n != 'M00'])))
cmp_lines.append("caught before: %d   caught now: %d" % (len(caught_old), len(caught_new)))
cmp_lines.append("caught before and NOT caught now (regressions): %s" % (regress or "none"))
cmp_lines.append("not caught before, caught now (%d): %s" % (len(gained), " ".join(gained)))
cmp_lines.append("not caught before, not caught now (%d): %s" % (len(lost_equiv), " ".join(lost_equiv)))
if "M00" in new:
    cmp_lines.append("control M00: failing units %d / %d" % (new["M00"][1], new["M00"][2]))
cmp_lines.append("")
cmp_lines.append("per mutant: old failing/700 -> new failing/total (only mutants whose caught/not-caught verdict changed, or whose count is below 10 now)")
for n in mut_names:
    o, nw = old.get(n), new.get(n)
    if not o or not nw:
        cmp_lines.append("  %s: old %s new %s" % (n, o and o[1], nw and nw[1]))
        continue
    if o[0] != nw[0] or nw[1] < 10:
        cmp_lines.append("  %s: %s %d/700 -> %s %d/%d" % (n, "caught" if o[0] else "uncaught", o[1], "caught" if nw[0] else "uncaught", nw[1], nw[2]))
open(os.path.join(LOGS, "compare-v2.txt"), "w").write("\n".join(cmp_lines) + "\n")
print("\n".join(cmp_lines))

# the extra controls X* of this pass (mk_extra.py): same rule, no old verdict
for d in sorted(glob.glob(OUT + "/X[0-9]*/")):
    name = os.path.basename(d.rstrip("/"))
    seen, fails = {}, []
    for l in open(d + "units.log", errors="replace"):
        m = re.match(r"UNIT (\S+) (\d+) (PASS|FAIL) lines=\S+ (.*)$", l.rstrip("\n"))
        if m:
            seen[(m.group(1), int(m.group(2)))] = 1
            if m.group(3) == "FAIL":
                fails.append((m.group(1), int(m.group(2)), m.group(4)))
    per = " ".join("%s=%d" % (s, sum(1 for f in fails if f[0] == s)) for s in ORDER if any(f[0] == s for f in fails)) or "-"
    ff = first_failure(name, [(a, b, c.rsplit(" ", 1)[-1]) for a, b, c in fails], NEW_ORDER)
    first = ("%s %d: %s" % (ff[0], ff[1], " / ".join(ff[2][:2]) if ff[2] else ff[3])) if ff else "-"
    line = "extra control %s: %s %d / %d  [%s]  first failure: %s" % (name, "driver FAIL" if fails else "not caught by driver", len(fails), len(seen), per, first[:300])
    print(line)
    open(os.path.join(LOGS, "compare-v2.txt"), "a").write(line + "\n")
