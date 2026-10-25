#!/bin/bash
# Independent count of the logged operations of a run: awk/wc over the RAW logs (zcat -f reads the compressed ones), not the driver's own numbers.
# usage: count.sh out/NAME     (P6 counts the first run of each library: u*.ref1.log / u*.rust1.log)
d=$1
for s in P1 P2 P3 P4 P5 P6; do
  [ -d "$d/$s" ] || continue
  for k in ref rust; do
    pat="u*.$k.log*"; [ "$s" = P6 ] && pat="u*.${k}1.log*"
    files=$(ls $d/$s/$pat 2>/dev/null)
    n=$(echo "$files" | grep -c .)
    echo -n "$s $k logs=$n "
    zcat -f $files | awk '{ t++; if ($1 ~ /^[dp][A-Z]+$/) ops++; else if ($1 ~ /^Z$/) z++; else if ($1=="CFG"||$1=="END") hdr++; else other++ } END { printf "lines=%d ops=%d stats=%d cfg_end=%d other_text=%d\n", t, ops, z, hdr, other }'
  done
done
