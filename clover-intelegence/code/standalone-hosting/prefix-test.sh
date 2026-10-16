#!/bin/sh
# Does a loaded prefix give the same answer as computing it?
#
# The layer body already skipped positions below TLO and read or wrote the MLA
# K/V and the KDA recurrent state around them; the file and the boundary were
# the only missing pieces. This checks the wiring on the smallest prompt that
# can have a prefix at all.
#
#   baseline  5 tokens, nothing cached
#   save      5 tokens, write a cache covering the first 4
#   load      5 tokens, read that cache, compute only position 4
#
# The gate is token identity. A prefix cache is supposed to be exact - unlike
# quantisation it changes no arithmetic, it only avoids repeating it - so
# anything other than identical output is a bug, not a tradeoff.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
BIN=${BIN:-/tmp/clover-one-pfx}
IDS="1008,10484,318,15383,387"
N=${N:-4}

run() {
    name=$1; shift
    echo "{\"input_ids\":[$IDS],\"max_new_tokens\":3}" > /tmp/pfx-$name.json
    env -i PATH="$PATH" \
        CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$S/bin/configs/generation.json \
        K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
        "$@" \
        $BIN $S/bin/dataset < /tmp/pfx-$name.json > /tmp/pfx-$name.out 2>/tmp/pfx-$name.err
    printf "%-9s exit %d  tokens: %s\n" "$name" $? \
        "$(grep -o '"token":[0-9]*' /tmp/pfx-$name.out | tr '\n' ' ')"
    [ -s /tmp/pfx-$name.err ] && head -2 /tmp/pfx-$name.err | cut -c1-120
}

run baseline
run save K3_PFXSAVE=/tmp/pfx.bin K3_PFXN=$N
echo "cache: $(ls -la /tmp/pfx.bin 2>/dev/null | awk '{print $5}') bytes for $N positions"
run load K3_PFXLOAD=/tmp/pfx.bin K3_PFXN=$N

echo
if diff <(grep -o '"token":[0-9]*' /tmp/pfx-baseline.out) \
        <(grep -o '"token":[0-9]*' /tmp/pfx-load.out) >/dev/null 2>&1; then
    echo "PASS - a loaded prefix gives the same tokens as computing it"
else
    echo "FAIL - loaded prefix changed the answer"
fi
