#!/bin/bash
# Probe inputs run on the reference, the real crate and selected mutants (own tool probe.cpp; the driver is not used).
cd ~/rustmig/routetable/verify/work/mut
REAL=~/rustmig/routetable/target/release/libroutetable.so
REF=~/rustmig/routetable/ref/libroutetable_ref.so
mkdir -p probe-in probe-out
printf '\f5 6 7:1\n\v3 4 9\n2\f3 4\n7\v8 9 1\n' > probe-in/ff.trace
printf '1 1 7 7\n1 1 7\n2 1 7 7 7\n1 1 7\n' > probe-in/dup.trace
printf '1 2 - 3\n' > probe-in/sign.trace
printf '1 2 -3 +4 5\n' > probe-in/sign2.trace
OPS_DUP="n107 c0 n107 c1 n107 c3 n107 c-1 n107 c0 n107 c1 n107 c2 n107 c0 n107 n108 c-5 n108 c1 n207 n107"
OPS_DUP2="n107 c0 n107 c-1 n107 c-1 n207 c2 n207 c-1 n207 c0 n107 c-1 n107"
OPS_UNK="n999 c5 n999 c-1 n107 n999 c0 n107 n999"
run() { # case lib-name lib-path file E ops...
  c=$1; l=$2; p=$3; f=$4; E=$5; shift 5
  ( ulimit -v 1000000; ./probe "$p" "$f" "$E" "$@" > probe-out/$c.$l.txt 2>&1; echo "exit $?" >> probe-out/$c.$l.txt )
}
for l in REF REAL M4 M5 M19; do
  case $l in REF) p=$REF;; REAL) p=$REAL;; *) p=~/rustmig/routetable/verify/work/mut/$l/target/release/libroutetable.so;; esac
  run ff $l $p probe-in/ff.trace 100
  run dup $l $p probe-in/dup.trace 100 $OPS_DUP
  run dupunk $l $p probe-in/dup.trace 100 $OPS_UNK
  run dup2 $l $p probe-in/dup.trace 100 $OPS_DUP2
done
for l in REF REAL M8 M8L; do
  case $l in REF) p=$REF;; REAL) p=$REAL;; *) p=~/rustmig/routetable/verify/work/mut/$l/target/release/libroutetable.so;; esac
  run sign $l $p probe-in/sign.trace 100
  run sign2 $l $p probe-in/sign2.trace 100
done
for c in ff dup dup2 dupunk sign sign2; do
  echo "=== $c"
  for l in REAL M4 M5 M8 M8L M19; do
    [ -f probe-out/$c.$l.txt ] || continue
    if cmp -s probe-out/$c.REF.txt probe-out/$c.$l.txt; then echo "$c $l == REF"; else echo "$c $l DIFFERS from REF"; fi
  done
done
echo "=== bytes 0x0b/0x0c in the real trace (read only): $(LC_ALL=C grep -c -P '[\x0b\x0c]' ~/k3gpu/results/p1d.trace)"
