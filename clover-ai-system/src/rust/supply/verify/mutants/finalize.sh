#!/bin/bash
# End of the campaign: keeps what is worth keeping (first failing unit of every caught mutant, units.log / summary.txt of every run, the .so md5 list) and deletes the scratch
# (out/mut, work/ with the mutant crates, targets and shared data, and the stray files described in RESULTS.md). Run from anywhere; prints what it deleted.
cd "$(dirname "$0")/.."
L=mutants/logs
mkdir -p $L/first-failures
python3 - <<'EOF'
import os, re, glob, shutil
V = os.path.expanduser("~/rustmig/supply/verify"); OUT = V + "/out/mut"; DST = V + "/mutants/logs/first-failures"
ORDER = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "YF", "T1"]
for d in sorted(glob.glob(OUT + "/M[0-9]*/")):
    n = os.path.basename(d.rstrip("/")); fails = []
    for l in open(d + "units.log", errors="replace"):
        m = re.match(r"UNIT (\S+) (\d+) FAIL ", l)
        if m: fails.append((ORDER.index(m.group(1)), int(m.group(2)), m.group(1)))
    if not fails: continue
    _, seed, scn = sorted(fails)[0]
    for ext in ("diff", "res"):
        src = "%s%s/u%d.%s" % (d, scn, seed, ext)
        if os.path.exists(src): shutil.copy(src, "%s/%s.%s.u%d.%s" % (DST, n, scn, seed, ext))
EOF
( cd out/mut && tar czf ../../mutants/logs/units-logs.tar.gz M*/units.log M*/summary.txt rerun*/units.log rerun*/summary.txt P*/units.log P*/summary.txt progress.txt )
grep -h 'start md5' out/mut/progress.txt | awk '{print $3, $2}' | sed 's/md5=//' > $L/md5-so-mutants.txt
cut -c1-80 $L/probe-build.out | awk '{print $2, $1}' > $L/md5-so-probes.txt
echo "stray files in verify/ (newer than mk_mutants.py): $(find . -maxdepth 1 -type f -newer mutants/mk_mutants.py | wc -l), of which empty $(find . -maxdepth 1 -type f -newer mutants/mk_mutants.py -size 0 | wc -l)"
find . -maxdepth 1 -type f -newer mutants/mk_mutants.py -delete
rm -rf out/mut work
echo "done; $(du -xsh mutants | cut -f1) in mutants/, df: $(df -h / | tail -1 | awk '{print $4" free "$5" used"}')"
