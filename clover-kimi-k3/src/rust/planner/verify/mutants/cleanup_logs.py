#!/usr/bin/env python3
# Frees the disk used by my own mutant runs: deletes every raw/gzipped unit log under out/mut except the logs of the first failing unit of each mutant (named in analysis.txt).
# units.log, .diff, .res, summary.txt, .stdout, event files stay. Runs are deterministic and can be repeated with run_all.sh.
import os, re
os.chdir(os.path.expanduser("~/rustmig/planner/verify/out/mut"))
keep = set(); cur = None
for l in open("analysis.txt"):
    m = re.match(r"== (\S+) ", l)
    if m: cur = m.group(1)
    m = re.match(r"\s+first failing: (P\d) (\d+):", l)
    if m: keep.add("%s/%s/u%s." % (cur, m.group(1), m.group(2)))
freed = n = 0
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".log") or f.endswith(".log.gz"):
            p = os.path.join(root, f)[2:]
            if any(p.startswith(k) for k in keep): continue
            try:
                freed += os.path.getsize(p); os.remove(p); n += 1
            except OSError: pass
print("deleted", n, "files,", freed // (1 << 20), "MB; kept logs for", len(keep), "first-failing units")
