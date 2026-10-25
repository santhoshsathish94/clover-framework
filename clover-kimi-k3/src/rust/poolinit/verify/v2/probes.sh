#!/bin/bash
# Driver v2: re-verifies that the mutants argued equivalent (M47 M48 M49 M59 M67 M68) and the one outside the contract (M86) still pass, AND that the mutated line still runs under the new schedules
# (gates, barrier, serial): the verifier's probe builds (mutant + a line appended to $PI_PROBE whenever the changed path runs) are rebuilt with the verifier's generator (mutants/mk_probes.py), run on
# the new driver (3,000 units each, mutants/run_probes.sh) and summarised with mutants/probe_count.py. The files of mutants/ are backed up first and compared afterwards (restored if they differ).
# usage (box, ~/rustmig/poolinit/verify): bash v2/probes.sh [probe names]      default: M47p M48p M49p M59p M67p M68p M86p       output out/v2/probes.txt
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
names="$*"; [ -z "$names" ] && names="M47p M48p M49p M59p M67p M68p M86p"
B=$(mktemp -d /tmp/v2probes.XXXXXX)
for n in $names; do cp -p mutants/$n.diff $B/; done
python3 mutants/mk_probes.py $names | cut -c1-100 > out/v2/probes-build.txt
for n in $names; do cmp -s mutants/$n.diff $B/$n.diff && echo "$n.diff unchanged" || { echo "$n.diff DIFFERS: restored"; cp -p $B/$n.diff mutants/$n.diff; }; done > out/v2/probes-diffs.txt
ls mutants/md5list.txt > /dev/null
JOBS=8 bash mutants/run_probes.sh $names > out/v2/probes-run.txt 2>&1
for n in $names; do python3 mutants/probe_count.py out/probe/$n.events; done > out/v2/probes-counts.txt 2>&1
cat out/v2/probes-build.txt out/v2/probes-diffs.txt out/v2/probes-run.txt out/v2/probes-counts.txt > out/v2/probes.txt
for n in $names; do rm -rf work/mut/$n/target; done
rm -rf out/probe $B
cat out/v2/probes.txt
