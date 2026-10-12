#!/usr/bin/env bash
# Capture two prompts with the KDA internals recorded, for the suspect layer and a control.
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

run_one() {
    local name="$1" ids="$2"
    echo "=== $name ===" >&2
    rm -rf "/tmp/kda-$name"; mkdir -p "/tmp/kda-$name"
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS="/tmp/kda-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $R serve" > "/tmp/kda-$name.log" 2>&1
    grep -o '\[.*\]' "/tmp/kda-$name.log" | head -1 >&2
}

run_one genesis "767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252"
run_one french  "3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68"

echo "=== internal records present? ===" >&2
for L in 12 24; do
    printf "layer %s: " "$L" >&2
    for loc in 5001 5004 5005 5006 5007 5008 5009; do
        n=$(/tmp/vector-compare "/tmp/kda-genesis/t.$L.bin" "/tmp/kda-genesis/t.$L.bin" $loc 4 2>/dev/null | cut -f2)
        printf "%s=%s " "$loc" "${n:-NA}" >&2
    done
    echo >&2
done
echo "=== done ===" >&2
