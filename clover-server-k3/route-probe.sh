#!/bin/bash
# Is the routing decision a function of the prefix alone?
#
# Three claims, each falsifiable:
#   R  same prompt twice          -> identical routing          (determinism)
#   P0 two prompts, same token 0  -> identical routing at pos 0 (position 0 is closed)
#   PX two prompts, same 0..6     -> identical routing at 0..6  (causal prefix)
#
# K3_PROV writes one S record per (layer, position, rank): the chosen expert and
# its weight. Timing from a PROV run means nothing; only the identities matter.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
B=build
mkdir -p $B/route

run() {   # tag npos ids
    local tag=$1 n=$2 ids=$3
    [ -x "$B/csk3-$n" ] || gcc -O3 -march=native -ffp-contract=off -fopenmp \
        -DNPOS="$n" -o "$B/csk3-$n" clover-server-k3.c -lm -lsqlite3 -lpthread || return 1
    rm -f "$B/route/$tag.tsv" "$B/route/$tag.bin"
    env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
        K3_PREFETCH=4 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
        K3_DB=/srv/k3/db K3_RAW=/srv/k3/raw K3_RAWMODE=1 \
        K3_INDEX=$B/eqidx.bin K3_IDS="$ids" K3_LOGITS="$B/route/$tag.bin" \
        K3_PROV="$B/route/$tag.tsv" "$B/csk3-$n" 2>&1 | grep -E "emitted token"
    echo "   $tag: $(grep -c '^S' "$B/route/$tag.tsv") S records"
}

echo "=== p1a  The capital of France is"
run p1a 5 1008,10484,318,15383,387
echo "=== p1b  same prompt again"
run p1b 5 1008,10484,318,15383,387
echo "=== p2   The chemical symbol for gold is"
run p2 6 1008,15548,9821,395,9148,387
echo "=== p22  Paris is the capital of France and Berlin is the capital of"
run p22 12 113476,387,276,10484,318,15383,316,28202,387,276,10484,318
echo "=== p23  Paris is the capital of France and Rome is the capital of"
run p23 12 113476,387,276,10484,318,15383,316,31082,387,276,10484,318
echo
echo "done"
