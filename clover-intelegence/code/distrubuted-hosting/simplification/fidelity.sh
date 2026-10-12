#!/usr/bin/env bash
# Run the prompt set through the production fleet and record, per prompt, the generated
# token and pod 92's output vector. Called twice: once for a baseline, once with a
# method's environment set, then compared by compare.sh.
#
# usage: fidelity.sh <tag> [env assignments...]
#   fidelity.sh baseline
#   fidelity.sh methodA CLOVER_FREEZE=/tmp/freeze/t CLOVER_FREEZE_FROM=0 CLOVER_FREEZE_TO=6
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
TAG=${1:?tag required}
shift

OUT=/tmp/fid-$TAG
rm -rf "$OUT"; mkdir -p "$OUT"

# Twelve prompts rather than two, because a "nothing changed" claim is only as good as
# the set it did not change on. Mixed length, language, and whether the text means
# anything. All are plain token id lists so no tokenizer runs inside the loop.
prompts="
france:1008 10484 318 15383 387
japan:1008 10484 318 10417 387
germany:1008 10484 318 16458 387
once:13426 7625 261 1169 1459
thethe:1008 276 276 276 276
water:29349 154199 623 10113 16525
hello:19180
question:5376 387 276 12156 6664 318 94916
nonsense:36324 291 79896 275 47369 304 57045 350 60142
french:3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68
genesis:767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252
n2:1008 10484
"

cd "$CODE/pipeline" || exit 1
echo "$prompts" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    name=${line%%:*}; ids=${line##*:}
    env CLOVER_VECTORS="$OUT/$name" CLOVER_VECTORS_OWNER=92 \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=false "$@" \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $CODE serve" > "$OUT/$name.log" 2>&1
    token=$(grep -o 'OK [0-9.]*s \[.*\]' "$OUT/$name.log" | sed 's/^OK [0-9.]*s //')
    printf '%s\t%s\n' "$name" "${token:-MISSING}" >> "$OUT/tokens.tsv"
    echo "  $name -> ${token:-MISSING}" >&2
done

echo "=== $TAG done, $(du -sh "$OUT" | cut -f1) ===" >&2
