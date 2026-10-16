#!/bin/sh
# Which observation hooks are actually reachable, and how big each per-layer
# slice is. A name in the source is not a wired feature - checked twice now.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting

echo "===== is covc ever allocated? (if not, cover() is dead) ====="
grep -n 'covc *=' $S/clover-one.c
echo "(no output above means never assigned)"

echo
echo "===== every fopen and what guards it ====="
grep -n 'fopen(' $S/clover-one.c

echo
echo "===== which of the observation FILE* are ever opened ====="
for v in prov_fp logit_fp lstat_fp stg_fp dumplay pfx_f pfx_of; do
  n=$(grep -c "$v *=" $S/clover-one.c)
  echo "$v assigned $n times"
done

echo
echo "===== trunk_ram: assigned anywhere? ====="
grep -n 'trunk_ram' $S/clover-one.c

echo
echo "===== per-layer slice sizes on disk ====="
D=/opt/clover-k3/clover-intelegence/dataset
ls -d $D/trunk-* 2>/dev/null | wc -l
echo "total:"
du -sh $D 2>/dev/null
echo "first few slices:"
for n in 0 1 2 3 11 12 13 46 59 90 91 92; do
  [ -d $D/trunk-$n ] && printf "trunk-%-3s %s\n" "$n" "$(du -sh $D/trunk-$n 2>/dev/null | cut -f1)"
done
echo "contents of one slice:"
ls -la $D/trunk-46 2>/dev/null | head -20
