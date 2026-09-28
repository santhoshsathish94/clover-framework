#!/bin/sh
# One row per expert against one row per tensor, same binary, same layer.
#
# Interleaved rather than run in blocks, so a drift in machine state cannot be
# mistaken for a difference between the schemas.
set -e
cd /opt/clover-k3
L=/srv/k3/run/ab_schema.log
: > "$L"
IDS=1008,10484,318,15383,387
BASE=23d162dcefb18211a7540ef12948f1eb

rm -rf /srv/k3/sel_none /srv/k3/sel_exp /srv/k3/sel_part
mkdir -p /srv/k3/sel_none /srv/k3/sel_exp /srv/k3/sel_part
ln -s /srv/k3/stores/L01.db      /srv/k3/sel_exp/L01.db
ln -s /srv/k3/stores_part/L01.db /srv/k3/sel_part/L01.db

say() { echo "$@" | tee -a "$L"; }

one() {
    rm -f build/srv.bin
    out=$(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
        K3_PREFETCH=4 K3_TRUNKRAM=0 K3_SLICES=/srv/k3/slices \
        K3_STORES="$2" K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
        K3_INDEX=build/eqidx.bin K3_IDS="$IDS" K3_LOGITS=build/srv.bin \
        ./build/clover-server-k3 2>&1)
    w=$(echo "$out" | grep "total wall" | awk '{print $5}')
    q=$(echo "$out" | grep -A1 "of which SQLite" | tail -1 | tr -s ' ')
    if [ ! -f build/srv.bin ]; then say "$1  NO OUTPUT"; return; fi
    m=$(md5sum build/srv.bin | cut -c1-8)
    if [ "$m" = "$(echo $BASE | cut -c1-8)" ]; then g=PASS; else g="FAIL $m"; fi
    say "  $1  ${w}s  $g $q"
}

# a warm-up cycle first, not recorded, so the first cold read of each store
# does not land on whichever schema happens to go first
say "warming"
one "none  " /srv/k3/sel_none
one "expert" /srv/k3/sel_exp
one "part  " /srv/k3/sel_part
say ""

for c in 1 2 3; do
    say "cycle $c"
    one "none  " /srv/k3/sel_none
    one "expert" /srv/k3/sel_exp
    one "part  " /srv/k3/sel_part
    say ""
done
rm -rf /srv/k3/sel_none /srv/k3/sel_exp /srv/k3/sel_part
touch /srv/k3/run/DONE_AB2
