#!/bin/bash
# Evidence extraction after run_all.sh: for each mutant the first failure report of the default run (oldest file in fail/), and the per-run lines of summary.txt.
# Output goes to stdout (collect.txt).
O=$HOME/rustmig/engcfg/verify/out/mut
for n in $(cd $HOME/rustmig/engcfg/verify/work/mut && ls -d M[0-9]* | sort); do
  echo "=== $n: $(cat $HOME/rustmig/engcfg/verify/work/mut/diffs/$n.txt 2>/dev/null)"
  grep -E "^$n( |-kg )" $O/summary.txt | cut -c1-400
  f=$(ls -tr $O/$n/fail/*.txt 2>/dev/null | head -1)
  if [ -n "$f" ]; then echo "first failure file: $(basename $f)"; head -3 "$f" | cut -c1-300; else echo "no failure report"; fi
done
