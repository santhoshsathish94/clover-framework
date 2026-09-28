#!/bin/bash
# Reusable A/B: clover-k3.c against a candidate source.
#
# Bit-exactness is the gate, not a tiebreak: a candidate that changes the
# logits is not faster, it is a different program. Timing is interleaved so
# machine drift lands on both arms equally, and n=4 because the run-to-run
# spread at 64 tokens is about 0.1 s within a session and up to 2 s across
# sessions. Nothing below that spread is a result.
#
#   ./ab.sh <tag> <candidate.c>
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/config.env"
B="$HERE/build"
TAG=${1:-run}
CAND=${2:-}
[ -n "$CAND" ] && [ -f "$CAND" ] || { echo "usage: ./ab.sh <tag> <candidate.c>" >&2; exit 1; }
[ -f "$B/eqidx.bin" ] || { echo "build it first: ./build.sh" >&2; exit 1; }

L="$B/ab_$TAG.log"
: > "$L"

BASE5=23d162dcefb18211a7540ef12948f1eb
EV="OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} \
K3_PREFETCH=4 K3_TRUNKRAM=0 K3_TRUNKPATH=$K3_TRUNKPATH K3_NREADER=14 \
K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 K3_INDEX=$B/eqidx.bin"
P5="1008,10484,318,15383,387"
P64="1008,10484,318,15383,387,17374,11,261,5243,418,276,18002,163561,473,924,11447,559,276,29652,5243,689,276,2226,13,1073,3707,1012,1201,276,163561,473,3391,290,276,5243,418,276,18002,11,290,276,2226,290,276,5243,418,261,7645,19620,4972,658,1178,12547,15757,290,276,2226,13,1073,3707,1012,1201,276,5243"

for n in 5 64; do
  gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS=$n -o "$B/ab_ref_$n" "$HERE/clover-k3.c" -lm 2>>"$L" \
    || { echo "REF BUILD FAIL" >> "$L"; cat "$L"; exit 1; }
  gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS=$n -o "$B/ab_new_$n" "$CAND" -lm 2>>"$L" \
    || { echo "NEW BUILD FAIL" >> "$L"; cat "$L"; exit 1; }
done
echo "built both" >> "$L"

echo >> "$L"; echo "=== EXACTNESS ===" >> "$L"
env $EV K3_IDS="$P5"  K3_LOGITS="$B/ab_r5.bin"  "$B/ab_ref_5"  >/dev/null 2>&1
env $EV K3_IDS="$P5"  K3_LOGITS="$B/ab_n5.bin"  "$B/ab_new_5"  >/dev/null 2>&1
env $EV K3_IDS="$P64" K3_LOGITS="$B/ab_r64.bin" "$B/ab_ref_64" >/dev/null 2>&1
env $EV K3_IDS="$P64" K3_LOGITS="$B/ab_n64.bin" "$B/ab_new_64" >/dev/null 2>&1
m=$(md5sum "$B/ab_n5.bin" | cut -d' ' -f1)
if [ "$m" = "$BASE5" ]; then echo "  5 tok new == PRESERVED BASELINE" >> "$L"
else echo "  5 tok new DIFFERS FROM BASELINE ($m)" >> "$L"; fi
cmp -s "$B/ab_r5.bin"  "$B/ab_n5.bin"  && echo "  5 tok  ref == new  IDENTICAL" >> "$L" || echo "  5 tok  ref != new  *** DIFFER ***" >> "$L"
cmp -s "$B/ab_r64.bin" "$B/ab_n64.bin" && echo " 64 tok  ref == new  IDENTICAL" >> "$L" || echo " 64 tok  ref != new  *** DIFFER ***" >> "$L"

echo >> "$L"; echo "=== TIMING, interleaved n=4 ===" >> "$L"
printf "%-6s %-8s %-9s %-9s %-9s\n" which wall5 wall64 SUMops64 resid64 >> "$L"
one () {
  w5=$(env $EV K3_IDS="$P5" K3_LOGITS=/dev/null "$B/ab_${1}_5" 2>&1 | grep 'total wall' | awk '{print $5}')
  r=$(env $EV K3_IDS="$P64" K3_LOGITS=/dev/null "$B/ab_${1}_64" 2>&1)
  w64=$(echo "$r" | grep 'total wall'        | awk '{print $5}')
  s=$(echo "$r"   | grep '^SUM of operators' | awk '{print $4}')
  rd=$(echo "$r"  | grep 'of which: stall'   | sed 's/.*residual //')
  printf "%-6s %-8s %-9s %-9s %-9s\n" "$1" "$w5" "$w64" "$s" "$rd" >> "$L"
}
for i in 1 2 3 4; do one ref; one new; done

echo >> "$L"
awk '$1=="ref" && $2+0>0 {a+=$2; b+=$3; c+=$4; d+=$5; n++} END {if(n) printf "ref mean: wall5 %.3f  wall64 %.3f  SUMops %.3f  resid %.3f  (n=%d)\n", a/n, b/n, c/n, d/n, n}' "$L" >> "$L"
awk '$1=="new" && $2+0>0 {a+=$2; b+=$3; c+=$4; d+=$5; n++} END {if(n) printf "new mean: wall5 %.3f  wall64 %.3f  SUMops %.3f  resid %.3f  (n=%d)\n", a/n, b/n, c/n, d/n, n}' "$L" >> "$L"

cat "$L"
