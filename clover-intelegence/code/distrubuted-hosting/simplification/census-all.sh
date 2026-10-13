#!/usr/bin/env bash
# Capture, census, delete - one prompt at a time, so peak disk stays near one dump
# rather than the ~26 GB all twelve would need at once.
set -uo pipefail
ulimit -c 0
C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
OUT=/tmp/census
rm -rf $OUT; mkdir -p $OUT
printf 'prompt\tposition\tblock\toffset\tpod\tattn\tout_norm\trel_attn\trel_fold\trel_moe\tin_norm\tspread\tratio\n' > $OUT/rows.tsv

cd "$C/pipeline" || exit 1
while IFS= read -r line; do
  [ -z "$line" ] && continue
  name=${line%%:*}; ids=${line##*:}
  rm -rf /tmp/cap; mkdir -p /tmp/cap
  env CLOVER_VECTORS=/tmp/cap/v CLOVER_TRACE=/tmp/cap/trace.csv \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
      sh -c "echo \"1 $ids\" | ./bin/pipeline-serve $C serve" > $OUT/$name.log 2>&1
  python3 /tmp/block-census.py /tmp/cap/v "$name" $OUT/rows.tsv
  # The trace is a second instrument on the same run; keep its per-stage hash counts.
  awk -F, -v p="$name" '$1>=1 && $1<=92 {
      key=$3; if (NR>1 && $5!=prev[key]) { by[$2]++ } prev[key]=$5
  } END { for (l in by) printf "%s\t%s\t%d\n", p, l, by[l] }' /tmp/cap/trace.csv >> $OUT/trace-counts.tsv
  rows=$(grep -c "^$name	" $OUT/rows.tsv)
  echo "$name done, $rows rows, $(du -sh /tmp/cap | cut -f1) captured and removed" >&2
  rm -rf /tmp/cap
done <<"EOP"
france:1008 10484 318 15383 387
japan:1008 10484 318 10417 387
germany:1008 10484 318 16458 387
once:13426 7625 261 1169 1459
thethe:1008 276 276 276 276
water:29349 154199 623 10113 16525
hello:19180
question:5376 387 276 12156 6664 318 94916
nonsense:36324 291 79896 275 47369 304 57045 350 60142
french:3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68
genesis:767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252
n2:1008 10484
EOP
echo "ALLDONE $(wc -l < $OUT/rows.tsv) rows" >&2
