#!/usr/bin/env python3
# Assembles the result table of RESULTS.md from: md5.txt (description per mutant), md5-so.txt (md5 of every built .so), analysis.txt (analyze.py output, every run), verdicts.txt (by hand).
# usage: mk_results.py > table.md       (run in the mutants directory)
import re
descr = {}
for l in open("md5.txt"):
    h, n, d = l.rstrip("\n").split("  ", 2)
    descr[n] = d
md5 = {}
for l in open("md5-so.txt"):
    h, p = l.split()
    md5[p.split("/")[-1][:-3]] = h
runs = {}
for l in open("analysis.txt"):
    m = re.match(r"(\S+)\.(\S+) units=(\d+)/(\d+) dup=(\d+) FAIL=(\d+) REFBAD=(\d+) \| (.*?) \| first: (.*)", l.rstrip("\n"))
    if not m:
        continue
    runs.setdefault(m.group(1), []).append(dict(tag=m.group(2), units=int(m.group(3)), jobs=int(m.group(4)), dup=int(m.group(5)), fail=int(m.group(6)), refbad=int(m.group(7)), per=m.group(8), first=m.group(9)))
ver = {}
for l in open("verdicts.txt"):
    if l.strip() and not l.startswith("#"):
        p = [x.strip() for x in l.rstrip("\n").split("|")]
        ver[p[0]] = p[1]
print("| mutant | change | .so md5 | verdict | failing units / units run (r1) | failing per scenario (r1, scenarios with a failure) | runs that caught it (r1-r3; soak) | first failure (lowest seed of the first failing scenario, r1) |")
print("|---|---|---|---|---|---|---|---|")
tot = dict(units=0, bad_dup=0, bad_ref=0)
for n in sorted(descr):
    rs = sorted(runs.get(n, []), key=lambda r: r["tag"])
    r1 = [r for r in rs if r["tag"] == "r1"][0]
    rr = [r for r in rs if re.fullmatch(r"r\d", r["tag"])]
    soak = [r for r in rs if r["tag"] == "soak"]
    caught = sum(1 for r in rr if r["fail"] > 0)
    v = ver.get(n, "CAUGHT" if caught else "NOT CAUGHT")
    per = " ".join(x for x in r1["per"].split() if not re.match(r".*=0/\d+$", x))
    part = " (partial: MAXFAIL=100, %d of %d units run)" % (r1["units"], r1["jobs"]) if r1["units"] < r1["jobs"] else ""
    cnt = "%d/%d" % (caught, len(rr)) + ("; soak %d/%d" % (soak[0]["fail"], soak[0]["units"]) if soak else "")
    first = r1["first"].replace("|", "\\|")
    first = first if len(first) < 230 else first[:230] + " ..."
    if n == "M00":
        v = "control (passes)"
    for r in rs:
        tot["units"] += r["units"]; tot["bad_dup"] += r["dup"]; tot["bad_ref"] += r["refbad"]
    print("| %s | %s | %s | %s | %d / %d%s | %s | %s | %s |" % (n, descr[n].replace("|", "\\|"), md5[n], v, r1["fail"], r1["units"], part, per or "-", cnt, first or "-"))
print()
print("<!-- unit lines over all runs: %d, duplicate lines: %d, REFBAD: %d -->" % (tot["units"], tot["bad_dup"], tot["bad_ref"]))
