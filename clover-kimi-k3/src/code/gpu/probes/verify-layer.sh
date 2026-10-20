#!/bin/bash
# verify.sh CANDIDATE_BIN LAYER [LAYER ...]
#
# The common acceptance check for a changed layer program (agent g7). For every LAYER, one after another, on card 7 only:
#   1. the baseline binary (~/probe/model-gpu-base) and CANDIDATE_BIN each run that one layer from the real activations in ~/inputs
#      (MODEL_GPU_ONLY_LAYER=L, MODEL_GPU_REPEAT=8, MODEL_GPU_ROUTES dump of the 16 chosen experts per token);
#   2. one line is printed: layer, median compute ms of runs 2..8 for baseline and candidate (run 1 is warm-up), speedup (baseline / candidate),
#      the candidate's output against the FULL run's output of that layer (the card-0 chain: ~/inputs/in-(L+1).f32), whether the routes files are
#      byte-identical (cmp), and a verdict.
#   PASS  the candidate's output is bit-for-bit the full run's (100.00% identical, max abs difference 0, relative RMS 0, no non-finite) AND its routes
#         equal the baseline's.
#   DIFF  it ran, the routes are equal, the output is not bit-identical: the reason gives bit-identical %, non-finite count, max abs, relative RMS, lowest cosine.
#   FAIL  the candidate (or the baseline) crashed, timed out, printed no comparison, the inputs are missing, or the routes differ / were not written.
#   Exit status: 0 if every layer is PASS, 1 otherwise.
#
# Call (detached, as the creator requires; the lock below serialises callers because one layer needs ~17 GiB of the 24 GiB card):
#   ssh -o BatchMode=yes gpu '~/probe/run-observed.sh NAME ~/agents/g7/verify.sh /home/usergpu/agents/gN/model-gpu-agent 1 2 3'
#   ssh -o BatchMode=yes gpu 'cat ~/runs/NAME/done 2>/dev/null || echo running; cat ~/runs/NAME/job.log'
# Optional environment: VERIFY_CAND_ENV="A=1 B=2" (extra variables for the candidate run only), VERIFY_TIMEOUT=seconds per run (default 900).
# Per-run logs and routes: ~/agents/g7/verify-runs/<time>-<pid>/L<layer>/{base,cand}.log, base/ and cand/ routes-L.bin.
#
# What it does NOT show: it compares against one recorded window (1000 tokens of one prompt); a candidate that is exact there can still differ on
# other inputs. The baseline's own output is compared with the same reference, so a PASS is only as strong as "baseline == full run" on that layer
# (the line says "base-exact=no" if the baseline itself is not bit-identical to the full run). Timings are one process each, baseline first, then the
# candidate, on a card shared only with callers waiting on the lock; they are not a stable microbenchmark.
set -u
BASE=$HOME/probe/model-gpu-base
DATASET=/opt/clover-k3/datasets/computed
REFS=$DATASET/engine-inputs/refs
INPUTS=$HOME/inputs
CARD=7
REPS=8
TMO=${VERIFY_TIMEOUT:-900}
WINDOW_BYTES=$((1000 * 7168 * 4))

if [ $# -lt 2 ]; then sed -n '2,3p' "$0"; echo "usage: verify.sh CANDIDATE_BIN LAYER [LAYER ...]"; exit 2; fi
CAND=$(readlink -f "$1"); shift
[ -x "$CAND" ] || { echo "FAIL: candidate $CAND is not an executable file"; exit 2; }
[ -x "$BASE" ] || { echo "FAIL: baseline $BASE is not an executable file"; exit 2; }
for L in "$@"; do case "$L" in ''|*[!0-9]*) echo "FAIL: layer '$L' is not a number"; exit 2;; esac; done

exec 9>$HOME/agents/g7/verify.lock
if ! flock -n 9; then echo "waiting for card $CARD (another verify.sh holds the lock)"; flock 9; fi

OUT=$HOME/agents/g7/verify-runs/$(date +%Y%m%d-%H%M%S)-$$
mkdir -p "$OUT"
echo "verify.sh: card $CARD, baseline $BASE (sha256 $(sha256sum "$BASE" | cut -c1-12)), candidate $CAND (sha256 $(sha256sum "$CAND" | cut -c1-12), $(date -r "$CAND" '+%F %T')), $REPS runs per layer, logs $OUT"
echo "speedup = baseline median / candidate median over runs 2..8; verdict is against the full run's recorded output (1000-token window)"

# run_one BIN NAME LAYER [extra env ...] -> $OUT/L/NAME.log, routes in $OUT/L/NAME/ ; returns the exit code (124 = timeout)
run_one() {
    local bin=$1 name=$2 L=$3; shift 3
    mkdir -p "$OUT/L$L/$name"
    env CUDA_VISIBLE_DEVICES=$CARD MODEL_GPU_ONLY_LAYER=$L MODEL_GPU_INPUTS=$INPUTS MODEL_GPU_REPEAT=$REPS MODEL_GPU_ROUTES="$OUT/L$L/$name" "$@" \
        timeout "$TMO" "$bin" $DATASET $REFS $L > "$OUT/L$L/$name.log" 2>&1
}
# median of "run N: compute X ms" for N >= 2
median_ms() {
    grep -E '^ +run [0-9]+: compute ' "$1" | awk '{ n=$2+0; if (n >= 2) print $4 }' | sort -n | awk '{ v[NR]=$1 } END { if (NR==0) { print "NA"; exit } if (NR%2) printf "%.1f", v[(NR+1)/2]; else printf "%.1f", (v[NR/2]+v[NR/2+1])/2 }'
}
# fields of the comparison line: pct nonfinite maxabs rms cos   (empty if the line is missing)
cmp_fields() {
    grep -E 'output of layer [0-9]+ against the full run' "$1" | tail -1 | sed -E 's/.*bit-identical ([0-9.]+)%, non-finite ([0-9]+), max abs difference ([^,]+), relative RMS difference ([^,]+), lowest cosine of a position ([-0-9.e+]+).*/\1 \2 \3 \4 \5/'
}
last_line() { grep -v '^ *$' "$1" | tail -1 | cut -c1-200; }

allpass=1
for L in "$@"; do
    verdict=PASS; reason=""
    prefix="layer $L"
    miss=""
    for f in "$INPUTS/in-$L.f32" "$INPUTS/in-$((L+1)).f32"; do
        if [ ! -f "$f" ]; then miss="$miss $(basename "$f") (missing)"; elif [ "$(stat -c %s "$f")" != "$WINDOW_BYTES" ]; then miss="$miss $(basename "$f") (size $(stat -c %s "$f"), not yet fully written?)"; fi
    done
    if [ -n "$miss" ]; then echo "$prefix | FAIL | inputs not usable:$miss"; allpass=0; continue; fi

    run_one "$BASE" base "$L"; rb=$?
    # shellcheck disable=SC2086
    run_one "$CAND" cand "$L" ${VERIFY_CAND_ENV:-}; rc=$?
    bms=$(median_ms "$OUT/L$L/base.log"); cms=$(median_ms "$OUT/L$L/cand.log")
    if [ "$bms" != NA ] && [ "$cms" != NA ]; then sp=$(awk -v b="$bms" -v c="$cms" 'BEGIN { printf "%.3fx", b / c }'); else sp="NA"; fi
    bf=$(cmp_fields "$OUT/L$L/base.log"); cf=$(cmp_fields "$OUT/L$L/cand.log")
    read -r bpct bnf bmax brms bcos <<< "$bf"
    read -r cpct cnf cmax crms ccos <<< "$cf"

    baseexact=no; [ "$bpct" = "100.00" ] && [ "$bnf" = "0" ] && [ "$bmax" = "0" ] && [ "$brms" = "0" ] && baseexact=yes

    routes="n/a"
    if [ "$L" != "0" ]; then
        rbf="$OUT/L$L/base/routes-$L.bin"; rcf="$OUT/L$L/cand/routes-$L.bin"
        if [ ! -s "$rbf" ]; then routes="baseline routes not written"
        elif [ ! -s "$rcf" ]; then routes="candidate routes not written"
        elif cmp -s "$rbf" "$rcf"; then routes="identical"
        else routes="DIFFER ($(cmp "$rbf" "$rcf" 2>&1 | head -1 | sed 's/^.*differ: //'); $(cmp -l "$rbf" "$rcf" 2>/dev/null | wc -l) bytes differ)"; fi
    fi

    if [ $rb -ne 0 ]; then verdict=FAIL; reason="baseline exit code $rb ($(last_line "$OUT/L$L/base.log"))"
    elif [ $rc -eq 124 ]; then verdict=FAIL; reason="candidate timed out after ${TMO}s"
    elif [ $rc -ne 0 ]; then verdict=FAIL; reason="candidate exit code $rc ($(last_line "$OUT/L$L/cand.log"))"
    elif [ -z "$cf" ]; then verdict=FAIL; reason="candidate printed no comparison with the full run"
    elif [ "$routes" != "identical" ] && [ "$routes" != "n/a" ]; then verdict=FAIL; reason="routes: $routes"
    elif [ "$cpct" = "100.00" ] && [ "$cnf" = "0" ] && [ "$cmax" = "0" ] && [ "$crms" = "0" ]; then verdict=PASS
    else verdict=DIFF; reason="not bit-identical: $cpct% identical, non-finite $cnf, max abs $cmax, relative RMS $crms, lowest cosine $ccos"
    fi
    [ "$verdict" != PASS ] && allpass=0
    note=""; [ "$baseexact" = no ] && note=" (base-exact=no: baseline itself vs full run: ${bpct:-?}% identical, max abs ${bmax:-?}, rel RMS ${brms:-?})"
    echo "$prefix | base ${bms} ms | cand ${cms} ms | speedup ${sp} | output bit-identical ${cpct:-NA}% | routes ${routes} | $verdict${reason:+: $reason}${note}"
done
echo "verify.sh done: $([ $allpass = 1 ] && echo 'all PASS' || echo 'NOT all PASS'); logs $OUT"
[ $allpass = 1 ]
