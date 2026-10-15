#!/usr/bin/env bash
# Does the per-token cost fall as an answer gets longer, and where does it settle?
#
# WHY THIS EXISTS
# I measured seven generated tokens, saw 4.46-4.60 s, and reported 4.512 s as the
# steady state -- an hour after writing up the fact that the one-token benchmark was a
# transient that had misled everything tuned against it. Same mistake, one regime up.
#
# The project owner's correction: the recorded 3.05 s/token comes from a long run,
# because most of the work is done in the first cycle. The mechanism is the expert
# cache. It holds K3_EXPERT_CACHE experts per layer against roughly 8 routed per token,
# so after seven tokens it has barely filled and cannot yet be hitting. Flat-and-warming
# looks exactly like flat-and-settled over a window that short.
#
# So: run long, keep EVERY position, and report the curve in blocks rather than one
# median over the whole thing. If the cost declines and settles, the settled value is
# the real per-token number and everything measured over eight tokens is warm-up.
#
# Written as a file and scp'd rather than inlined into an ssh command, because the
# previous attempt at this sweep died with `bash: -c: line 2: syntax error` from nested
# quoting and silently never ran, leaving a dependent job waiting on a marker that
# would never appear.
set -u

TOKENS=${TOKENS:-64}
CACHES=${CACHES:-"24"}

for cache in $CACHES; do
    echo "######################## K3_EXPERT_CACHE=$cache, $TOKENS tokens ########################"
    TOKENS="$TOKENS" CACHE="$cache" TAG="trend$cache" bash /tmp/standalone-cost.sh > "/tmp/trend-$cache.log" 2>&1

    if ! grep -q "generated tokens" "/tmp/trend-$cache.log"; then
        echo "*** FAILED"; tail -5 "/tmp/trend-$cache.log"; continue
    fi

    grep -E "answer:|prompt pass|wall |bytes read|device busy|rate while|^cpu " "/tmp/trend-$cache.log"

    # The curve, in blocks of eight, so warm-up and settled value are both visible.
    awk '/^  position/ {
            n++; if (n == 1) { printf "  prompt pass        %s s\n", $3; next }
            g++; sum += $3; if (g % 8 == 0) {
                printf "  generated %3d-%3d   mean %7.3f s\n", g-7, g, sum/8; sum = 0
            }
         }
         END { if (g % 8) printf "  generated %3d-%3d   mean %7.3f s\n", g-(g%8)+1, g, sum/(g%8) }' \
        "/tmp/trend-$cache.log"

    # First eight generated against last eight: the whole point in one line.
    awk '/^  position/ { n++; if (n > 1) v[++g] = $3 }
         END {
            if (g < 16) { print "  too few tokens to compare first and last blocks"; exit }
            for (i = 1; i <= 8; i++) a += v[i]
            for (i = g-7; i <= g; i++) b += v[i]
            printf "\n  FIRST 8 generated %7.3f s   LAST 8 %7.3f s   change %+.1f %%\n",
                   a/8, b/8, 100*(b/8 - a/8)/(a/8)
         }' "/tmp/trend-$cache.log"
    echo
done
touch /tmp/trend.done
