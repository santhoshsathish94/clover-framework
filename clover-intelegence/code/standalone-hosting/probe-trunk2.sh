#!/bin/sh
# The trunk's actual shape: slots, ops, per-layer slices, and the instruments
# that already exist for "does a run need all of it".
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting

echo "===== trunk slot names (SLOTN) ====="
sed -n '150,190p' $S/clover-one.c

echo
echo "===== op table (OPN) ====="
sed -n '285,305p' $S/clover-one.c

echo
echo "===== K3_COVER: per-page touch counting ====="
sed -n '107,146p' $S/clover-one.c

echo
echo "===== trunk -> RAM path and K3_TRUNKRAM ====="
sed -n '440,460p' $S/clover-one.c
grep -n 'trunk_ram\|K3_TRUNKRAM\|K3_PREFETCH\|K3_NX' $S/clover-one.c

echo
echo "===== where the Q op reads its weights (per layer) ====="
sed -n '1340,1360p' $S/clover-one.c
