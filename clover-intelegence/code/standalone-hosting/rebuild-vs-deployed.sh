#!/usr/bin/env bash
# Does rebuilding from current source, with the expert cache that the deployed binary
# does not contain, actually make a token cheaper -- and does it still say the same thing?
#
# WHAT WAS FOUND
#   bin/clover-one   2026-10-08 05:40
#   clover-one.c     2026-10-10 16:41   <- two days newer
#   live-root.h      2026-10-10 15:35
#   strings bin/clover-one | grep K3_EXPERT_CACHE  ->  nothing
#
# The deployed binary predates the per-layer expert cache entirely, so K3_EXPERT_CACHE
# was being silently ignored: 0, 24 and 48 produced byte-identical disk traffic
# (1647.83 / 1647.83 / 1647.73 GB) and identical peak RSS (76.46 GB three times) when
# capacity 24 should have added 38.5 GB of anonymous memory. The recorded 3.05 s/token
# is attributed to that cache, which is why it has not been reproducible: the engine on
# the box cannot do it.
#
# THE GATE THAT MATTERS
# Speed is not the first question. The old and new binaries must produce the SAME
# TOKENS for the same prompt. A faster engine that answers differently is not an
# improvement, it is a regression with a good stopwatch. Token ids are compared before
# any timing is believed, and the run is declared failed if they differ.
#
# The deployed binary is NOT replaced by this script. It is backed up at
# /tmp/clover-one.BACKUP-2026-10-15 and the new build lives at /tmp/clover-one.new.
set -u

OLD=/opt/clover-k3/clover-intelegence/code/standalone-hosting/bin/clover-one
NEW=/tmp/clover-one.new
TOKENS=${TOKENS:-32}

run() {
    local label=$1 bin=$2 cache=$3
    echo "######## $label : $(basename "$bin") cache=$cache ########"
    BIN="$bin" TOKENS="$TOKENS" CACHE="$cache" TAG="cmp-$label" \
        bash /tmp/standalone-cost.sh > "/tmp/cmp-$label.log" 2>&1
    if ! grep -q "generated tokens" "/tmp/cmp-$label.log"; then
        echo "  *** FAILED"; tail -6 "/tmp/cmp-$label.log"; return 1
    fi
    grep -E "answer:|prompt pass  |generated tokens|bytes read|device busy|rate while|^cpu |peak RSS" "/tmp/cmp-$label.log"
    grep -h "^TOKENS_PRODUCED:" "/tmp/cmp-$label.log" | cut -c1-120
    echo
}

run old-nocache  "$OLD" 0
run new-nocache  "$NEW" 0
run new-cache24  "$NEW" 24

echo "######## CORRECTNESS GATE ########"
a=$(grep -h "^TOKENS_PRODUCED:" /tmp/cmp-old-nocache.log 2>/dev/null)
b=$(grep -h "^TOKENS_PRODUCED:" /tmp/cmp-new-nocache.log 2>/dev/null)
c=$(grep -h "^TOKENS_PRODUCED:" /tmp/cmp-new-cache24.log 2>/dev/null)
if [ -z "$a" ]; then
    echo "no token list from the old binary -- cannot gate, treat timings as unverified"
elif [ "$a" = "$b" ] && [ "$a" = "$c" ]; then
    echo "PASS: all three runs produced identical token ids over $TOKENS tokens"
else
    echo "*** FAIL: token ids differ. Any speed difference below is meaningless."
    echo "  old-nocache : $a"
    echo "  new-nocache : $b"
    echo "  new-cache24 : $c"
fi
touch /tmp/rebuild.done
