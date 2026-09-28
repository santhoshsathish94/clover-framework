#!/bin/sh
# Same-session control for the trunk split. The reference and the slice build
# cannot both be tmpfs-resident at once (54.47 GB each, 124 GB of RAM), so they
# run in sequence, not interleaved. That is a real weakness of this comparison
# and it is stated in the log rather than hidden.
#
# Every run deletes its logits file first. A stale file from an earlier run
# passing the md5 check is exactly how the first version of this script
# reported three PASSes for runs that never happened.
set -e
cd /opt/clover-k3
. ./config.env
L=/srv/k3/run/v1.log
: > "$L"
IDS=1008,10484,318,15383,387
BASE=23d162dcefb18211a7540ef12948f1eb

say() { echo "$@" | tee -a "$L"; }

# $1 label, $2 binary, $3 logits path, $4.. extra NAME=VALUE
one() {
    n="$1"; bin="$2"; out="$3"; shift 3
    rm -f "$out"
    w=$(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
        K3_PREFETCH=4 K3_TRUNKRAM=0 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 \
        K3_HUGE=1 K3_INDEX=build/eqidx.bin K3_IDS="$IDS" K3_LOGITS="$out" \
        "$@" "$bin" 2>&1 | grep "total wall" | awk '{print $5}')
    if [ ! -f "$out" ]; then say "  run $n  NO OUTPUT - the run did not happen"; return; fi
    m=$(md5sum "$out" | cut -d' ' -f1)
    if [ "$m" = "$BASE" ]; then g=PASS; else g=FAIL; fi
    say "  run $n  wall ${w}s  md5 $m  $g"
}

say "binaries:"
say "  $(ls -l build/clover-k3 2>/dev/null || echo 'build/clover-k3 MISSING')"
say "  $(ls -l build/clover-server-k3 2>/dev/null || echo 'build/clover-server-k3 MISSING')"
say ""

say "=== A. reference, trunk from tmpfs (control) ==="
for i in 1 2 3; do
    one "$i" ./build/clover-k3 build/ref.bin K3_TRUNKPATH=/dev/shm/trunk.bin
done

say ""
say "=== B. slices from disk ==="
for i in 1 2 3; do
    one "$i" ./build/clover-server-k3 build/srv.bin K3_SLICES=/srv/k3/slices
done

say ""
say "=== C. slices from tmpfs - the like-for-like case ==="
say "  freeing /dev/shm/trunk.bin (a copy; /root/k3trunk_i8/trunk.bin is the original)"
rm -f /dev/shm/trunk.bin
mkdir -p /dev/shm/slices
cp /srv/k3/slices/L*.bin /dev/shm/slices/
say "  /dev/shm now: $(du -sh /dev/shm | cut -f1)"
for i in 1 2 3; do
    one "$i" ./build/clover-server-k3 build/srv.bin K3_SLICES=/dev/shm/slices
done

say ""
say "=== restoring /dev/shm/trunk.bin so the reference still runs ==="
rm -rf /dev/shm/slices
cp /root/k3trunk_i8/trunk.bin /dev/shm/trunk.bin
say "  /dev/shm now: $(du -sh /dev/shm | cut -f1)"
touch /srv/k3/run/DONE_V1
