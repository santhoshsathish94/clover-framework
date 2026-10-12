#!/usr/bin/env bash
# A deliberately varied set. Everything observed so far came from five prompts that were
# all English, all exactly five tokens, and all with an obvious completion. These vary
# length, language, and whether the text means anything, so the earlier claims can be
# put under some strain before they are trusted.
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

run_one() {
    local name="$1" ids="$2"
    echo "=== $name ($(echo $ids | wc -w) tokens) ===" >&2
    rm -rf "/tmp/keep-$name"; mkdir -p "/tmp/keep-$name"
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS="/tmp/keep-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $R serve" > "/tmp/keep-$name.log" 2>&1
    grep -o '\[.*\]' "/tmp/keep-$name.log" | head -1 >&2
    du -sh "/tmp/keep-$name" >&2
}

run_one hello    "19180"
run_one question "5376 387 276 12156 6664 318 94916"
run_one nonsense "36324 291 79896 275 47369 304 57045 350 60142"
run_one french   "3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68"
run_one genesis  "767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252"

echo "=== done ===" >&2
df -h / | tail -1 >&2
