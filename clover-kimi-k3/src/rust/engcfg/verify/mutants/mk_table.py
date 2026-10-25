#!/usr/bin/env python3
# Builds the result table of RESULTS.md from the run's own files (run on the box): out/mut/summary.txt, work/mut/diffs/*.txt, out/mut/<name>/fail/*.txt.
# Usage: mk_table.py > table.md
import glob, os, re, sys

HOME = os.path.expanduser("~")
V = os.path.join(HOME, "rustmig/engcfg/verify")
O = os.path.join(V, "out/mut")
D = os.path.join(V, "work/mut/diffs")

info = {}
for line in open(os.path.join(O, "summary.txt")):
    m = re.match(r"^(M\d+)(-kg)? build\.exit=(\d+) warnings=(\d+) so_md5=(\w*)", line)
    if m:
        info.setdefault(m.group(1), {})["md5"] = m.group(5); info[m.group(1)]["build"] = m.group(3); continue
    m = re.match(r"^(M\d+)(-kg)? done in (\d+)s: units_logged=(\d+) failing=(\d+) by_scenario=\[([^\]]*)\]", line)
    if m:
        k = "kg" if m.group(2) else "def"
        info.setdefault(m.group(1), {})[k] = (int(m.group(4)), int(m.group(5)), m.group(6).strip())

def first_failure(name):
    files = sorted(glob.glob(os.path.join(O, name, "fail", "*.txt")), key=os.path.getmtime)
    if not files:
        return "-"
    txt = open(files[0], errors="replace").read().split("\n")
    m = re.match(r"MISMATCH (.*) unit (\S+) seed (\d+) cmp #(\d+)", txt[0])
    if not m:
        return os.path.basename(files[0]) + ": " + txt[0][:100]
    what, unit, seed = m.group(1), m.group(2), m.group(3)
    what = what.replace("composite: verbatim configure() vs contract sequence on the candidate", "E7 composite A-vs-C").replace("composite: verbatim configure() vs contract sequence on the reference lib", "E7 composite A-vs-B")
    a = txt[1].replace("  reference: ", "") if len(txt) > 1 else ""
    b = txt[2].replace("  candidate: ", "") if len(txt) > 2 else ""
    out = "%s seed %s: %s" % (unit, seed, what)
    if len(a) <= 60 and len(b) <= 90:
        out += " - ref %s, cand %s" % (a, b)
    else:
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        lo = max(0, i - 25)
        out += " - first difference near: ref `%s` / cand `%s`" % (a[lo:i + 35], b[lo:i + 35])
    return out.replace("|", "\\|")

names = sorted(info.keys(), key=lambda s: int(s[1:]))
verd = {}
vf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verdicts.txt")
if os.path.exists(vf):
    for l in open(vf):
        if "|" in l:
            k, v = l.rstrip("\n").split("|", 1)
            verd[k.strip()] = v.strip()
print("| mutant | change | .so md5 | verdict | failing units / 4386 (--keep-going) | default run (units run, failed) | first failure |")
print("|---|---|---|---|---|---|---|")
for n in names:
    i = info[n]
    desc = open(os.path.join(D, n + ".txt")).read().strip().replace("|", "\\|")
    md5 = i.get("md5", "")[:8]
    if i.get("build") != "0":
        print("| %s | %s | - | BUILD FAILED | - | - | - |" % (n, desc)); continue
    dflt = i.get("def"); kg = i.get("kg")
    caught = dflt and dflt[1] > 0
    verdict = ("CAUGHT" if caught else verd.get(n, "NOT CAUGHT")) if n != "M00" else ("PASS (control)" if not caught else "FAIL (control!)")
    kgs = "%d (%s)" % (kg[1], kg[2].replace(":", " ").replace("  ", " ")) if kg else "-"
    print("| %s | %s | %s | %s | %s | %s | %s |" % (n, desc, md5, verdict, kgs, ("%d run, %d failed" % (dflt[0], dflt[1])) if dflt else "-", first_failure(n) if caught else "-"))
