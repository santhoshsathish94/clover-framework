#!/bin/sh
# Pinning the trunk costs the expert cache. On a 124 GB box holding 54.47 GB
# for the trunk takes it from the 1.45 TB expert store, so the two must be
# measured together and interleaved, not one after the other.
set -e
cd /opt/clover-k3
L=/srv/k3/run/pin.log
: > "$L"
IDS=1008,10484,318,15383,387
BASE=23d162dc

say() { echo "$@" | tee -a "$L"; }

resid() {   # GB of the expert stores currently in page cache
    python3 - <<'EOF'
import subprocess, glob
tot=res=0.0
for f in sorted(glob.glob("/srv/k3/db/expert/*.db")):
    o = subprocess.run(["/tmp/resid", f], capture_output=True, text=True).stdout.split()
    tot += float(o[1]); res += float(o[4])
print("%.1f" % res)
EOF
}

one() {
    rm -f /tmp/p.bin
    out=$(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
        K3_PREFETCH=4 K3_DB=/srv/k3/db $2 K3_NREADER=14 K3_XDEC=2 \
        K3_PLGRAN=1 K3_HUGE=1 K3_INDEX=build/eqidx.bin K3_IDS="$IDS" \
        K3_LOGITS=/tmp/p.bin ./build/csk3 2>&1)
    w=$(echo "$out"  | grep "total wall" | awk '{print $5}')
    sq=$(echo "$out" | grep "of which SQLite" | awk '{print $8}')
    gb=$(echo "$out" | grep "of which SQLite" | awk '{print $11}')
    m=$(md5sum /tmp/p.bin | cut -c1-8)
    if [ "$m" = "$BASE" ]; then g=PASS; else g="FAIL $m"; fi
    say "  %-9s wall ${w}s  sqlite ${sq} thread-s @ ${gb} GB/s  expert cache $(resid) GB  $g" \
        | sed "s/%-9s/$1/"
}

say "expert store cache before anything: $(resid) GB of 1450.84"
say ""
for c in 1 2 3; do
    say "cycle $c"
    one "pin-off" "K3_PIN=0"
    one "pin-on " "K3_PIN=1"
    say ""
done
touch /srv/k3/run/DONE_PIN
