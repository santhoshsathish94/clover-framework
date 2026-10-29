#!/usr/bin/env python3
# Summarises one driver output directory (and its .stdout): per scenario number of failing units, the failing unit with the lowest op index, its raw first lines, and S9/Q5 case failures.
# Usage: summarize.py <name>-quick   (relative to verify/out/mut)
import glob, os, re, sys

base = os.path.join(os.path.expanduser("~/rustmig/slotcache/verify/out/mut"), sys.argv[1])
fails = []
for f in glob.glob(base + "/fail-*.txt"):
    t = open(f, errors="replace").read()
    m = re.match(r"MISMATCH: scenario (\S+) seed (\d+) op #(\d+)", t)
    if not m:
        continue
    fails.append((int(m.group(3)), m.group(1), int(m.group(2)), t, os.path.basename(f)))
stdout = open(base + ".stdout", errors="replace").read() if os.path.exists(base + ".stdout") else ""
ab = [l for l in stdout.splitlines() if "ended abnormally" in l or "did not finish" in l]
print("DIR", sys.argv[1], "failing units:", len(fails))
per = {}
for op, sc, sd, t, fn in fails:
    per.setdefault(re.sub(r"[a-z]?$", "", sc) if False else sc, []).append(op)
print("per scenario (failing units / min op index):", ", ".join("%s %d/%d" % (k, len(v), min(v)) for k, v in sorted(per.items())))
for l in ab[:6]:
    print("ABNORMAL:", l.strip())
case_fail = [l.strip()[:200] for l in stdout.splitlines() if l.strip().startswith("FAIL ") and "agree=" in l]
for l in stdout.splitlines():
    if re.match(r"^(S9|Q5)\s+cases=", l):
        print(l.strip())
for l in case_fail:
    print("CASEFAIL:", l)
if fails:
    fails.sort(key=lambda x: (x[0], x[1], x[2]))
    op, sc, sd, t, fn = fails[0]
    print("LOWEST-OP FAIL FILE:", fn)
    print("\n".join(t.splitlines()[:9])[:1800])
