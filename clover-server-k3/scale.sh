#!/bin/sh
# Does the SQLite cost scale with the number of converted layers?
#
# The only variable is how many stores are visible, controlled by symlinking a
# subset into a selection directory. n=0 is the pure checkpoint control and
# uses the same binary, so nothing but the store count differs.
#
# Three runs per setting. The first after adding a store is cold for it, which
# is reported rather than discarded.
set -e
cd /opt/clover-k3
L=/srv/k3/run/scale.log
: > "$L"
IDS=1008,10484,318,15383,387
BASE=23d162dcefb18211a7540ef12948f1eb
SEL=/srv/k3/sel

say() { echo "$@" | tee -a "$L"; }
say "n  run   wall     md5       sqlite"

for n in 0 1 2 3 4; do
    rm -rf "$SEL"; mkdir -p "$SEL"
    i=1
    while [ "$i" -le "$n" ]; do
        ln -s "/srv/k3/stores/L0$i.db" "$SEL/L0$i.db"
        i=$((i + 1))
    done
    for r in 1 2 3; do
        rm -f build/srv.bin
        out=$(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
            K3_PREFETCH=4 K3_TRUNKRAM=0 K3_SLICES=/srv/k3/slices \
            K3_STORES="$SEL" K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
            K3_INDEX=build/eqidx.bin K3_IDS="$IDS" K3_LOGITS=build/srv.bin \
            ./build/clover-server-k3 2>&1)
        w=$(echo "$out" | grep "total wall" | awk '{print $5}')
        q=$(echo "$out" | grep "of which SQLite" | sed 's/.*: //')
        if [ ! -f build/srv.bin ]; then say "$n  $r    NO OUTPUT"; continue; fi
        m=$(md5sum build/srv.bin | cut -c1-8)
        if [ "$m" = "$(echo $BASE | cut -c1-8)" ]; then g=PASS; else g="FAIL $m"; fi
        say "$n  $r   ${w}s   $g   $q"
    done
    say ""
done
rm -rf "$SEL"
touch /srv/k3/run/DONE_SCALE
