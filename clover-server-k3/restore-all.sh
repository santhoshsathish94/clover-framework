#!/bin/bash
# Put every layer's experts back into the checkpoint, then reclaim its store.
#
# Order matters and is not negotiable:
#   restore -> verify byte-for-byte -> move the store aside -> gate -> delete
#
# A layer's store is deleted only after restore_shard.py has verified every
# tensor. Peak extra disk is one layer, about 15.7 GB, because the store goes
# away as soon as the shard has the bytes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
B=build
BASE5=23d162dcefb18211a7540ef12948f1eb
P5="1008,10484,318,15383,387"
EDIR=/srv/k3/db/expert
LOG=$B/restore-all.tsv
GATE_EVERY=10

[ -f "$LOG" ] || printf 'layer\tstore_gb\tfree_gb_after\tverdict\n' > "$LOG"

gate() {
    rm -f $B/g.bin
    env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
        K3_PREFETCH=4 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
        K3_DB=/srv/k3/db K3_RAW=/srv/k3/raw K3_RAWMODE=1 \
        K3_INDEX=$B/eqidx.bin K3_IDS="$P5" K3_LOGITS=$B/g.bin \
        $B/csk3 > /dev/null 2>&1
    [ -s $B/g.bin ] && [ "$(md5sum $B/g.bin | cut -d' ' -f1)" = "$BASE5" ]
}

for L in $(seq -w 91 -1 1 2>/dev/null || seq 91 -1 1); do
    L=$((10#$L))
    db=$EDIR/L$(printf %02d $L).db
    [ -f "$db" ] || { echo "L$L: no store, skipping"; continue; }
    gb=$(du -BG "$db" | cut -f1 | tr -d G)

    if ! python3 restore_shard.py --layer "$L" --apply > $B/restore-L$L.log 2>&1; then
        echo "L$L: RESTORE FAILED - stopping, store left in place"
        tail -5 $B/restore-L$L.log
        printf '%d\t%s\t%s\tRESTORE_FAILED\n' "$L" "$gb" "$(df -BG / | tail -1 | awk '{print $4}' | tr -d G)" >> "$LOG"
        exit 1
    fi
    if ! grep -q "verified byte-for-byte" $B/restore-L$L.log; then
        echo "L$L: verification line missing - stopping"
        exit 1
    fi

    mv "$db" "$db.hold"
    if [ $((L % GATE_EVERY)) -eq 0 ] || [ "$L" -le 2 ]; then
        if gate; then v=PASS; else
            echo "L$L: GATE FAILED - putting the store back and stopping"
            mv "$db.hold" "$db"
            exit 1
        fi
    else
        v=verified
    fi
    rm -f "$db.hold"

    free=$(df -BG / | tail -1 | awk '{print $4}' | tr -d G)
    printf '%d\t%s\t%s\t%s\n' "$L" "$gb" "$free" "$v" >> "$LOG"
    echo "L$L done  store ${gb}GB reclaimed  free ${free}GB  $v"
done

echo
echo "=== final gate, every layer from the checkpoint"
if gate; then echo "PASS  $BASE5"; else echo "FAIL"; exit 1; fi
df -h / | tail -1
