#!/bin/bash
# Independent recount of the failing units of every mutant straight from units.log (grep), to compare with analyze.py's table: prints "<name> <UNIT lines> <FAIL lines> <duplicate (scn,seed)>".
cd "$(dirname "$0")/../out/mut"
for d in M[0-9]*/; do
  n=${d%/}
  echo "$n $(grep -c '^UNIT ' $n/units.log) $(grep -c '^UNIT .* FAIL ' $n/units.log) $(grep '^UNIT ' $n/units.log | awk '{print $2,$3}' | sort | uniq -d | wc -l)"
done
