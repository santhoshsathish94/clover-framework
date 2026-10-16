#!/bin/sh
# What can already be observed without changing a line, and how big each
# per-layer slice actually is.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting

echo "===== OPSTAT_JSON: what the per-op report prints ====="
sed -n '2995,3020p' $S/clover-one.c

echo
echo "===== K3_PROV emit: does it carry a size per slot? ====="
grep -n 'prov_fp' $S/clover-one.c | head -20

echo
echo "===== K3_COVER: how it is enabled and reported ====="
grep -n 'covc\|K3_COVER' $S/clover-one.c

echo
echo "===== every K3_ env var the engine reads ====="
grep -o 'getenv("K3_[A-Z0-9_]*")' $S/clover-one.c | sort -u

echo
echo "===== per-layer trunk slices on disk ====="
P=$(grep -o 'K3_PREPARED_DATA[^)]*' $S/clover-one.c | head -1)
echo "(env name: $P)"
for d in /opt/clover-k3/prepared /opt/clover-k3/data/prepared /mnt/*/prepared; do
  [ -d "$d" ] && echo "FOUND $d" && du -sh $d 2>/dev/null && ls $d | head -5
done
echo "--- any trunk-N directories anywhere:"
find /opt/clover-k3 -maxdepth 4 -type d -name 'trunk-*' 2>/dev/null | head -5
