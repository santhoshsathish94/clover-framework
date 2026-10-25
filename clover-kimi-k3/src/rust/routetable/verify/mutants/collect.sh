#!/bin/bash
# Collects the evidence of the campaign from the raw logs (run on the box).
O=$HOME/rustmig/routetable/verify/out/mut
W=$HOME/rustmig/routetable/verify/work/mut
for n in M0 M1 M2 M3 M4 M5 M6 M7 M8 M8L M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19; do
  echo "##### $n"
  echo "desc: $(cat $W/diffs/$n.txt)"
  echo "hunks: $(grep -c '^@@' $W/diffs/$n.diff)   so_md5: $(md5sum < $W/$n/target/release/libroutetable.so | cut -d' ' -f1)"
  echo "default run: $(grep -E '^overall' $O/$n/stdout.txt) / $(tail -1 $O/$n/stdout.txt)"
  awk '/^FIRST FAILURE/{f=1;c=0} f&&c<12{print substr($0,1,330);c++}' $O/$n/stdout.txt
  grep -h '^CRASH' $O/$n/units.log | head -2 | cut -c1-300
  for m in "" "-kg"; do
    [ "$m" = "-kg" ] && echo "keep-going run: units completed=$(grep -c '^UNIT' $O/$n-kg/units.log) FAIL: $(awk '$1=="UNIT"&&$9=="FAIL"{c[$2]++} END{for(k in c) printf "%s=%d ",k,c[k]}' $O/$n-kg/units.log) CRASH: $(awk '$1=="CRASH"{c[$2]++} END{for(k in c) printf "%s=%d ",k,c[k]}' $O/$n-kg/units.log) $(grep -E '^overall' $O/$n-kg/stdout.txt)"
  done
done
