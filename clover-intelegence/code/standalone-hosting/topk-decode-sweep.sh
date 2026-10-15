#!/usr/bin/env bash
# Does a reduced expert count survive DECODE, or only the prefill token?
#
# WHAT IS ALREADY RECORDED, and what it explicitly is not
# k3-data-problem.md measured top-k of 16 on one prompt and reports, for the PREFILL
# token only: k=16/12/8/4 all emit token 17374, k=2 emits 261, k=1 emits 276. With the
# softmax read rather than the raw margin, k=8 costs no confidence at all (p(top1)
# 0.790 against 0.792) while k=4 keeps the token but drops confidence to 0.539.
# The document states its own limits plainly: "One prompt, one token", "Prefill only.
# ... Whether k = 4 also preserves the decode token 20829 is untested", and "The
# threshold is not established."
#
# This closes the decode half. The observable is not the first token and not a margin:
# it is THE FIRST POSITION AT WHICH THE TOKEN SEQUENCE DIVERGES from k=16 over a long
# generation. A k that matches for 32 tokens is a different claim from a k that matches
# for one, and divergence-at-position-N is more useful than pass/fail because it says
# how long the approximation holds rather than merely whether it did.
#
# k=16 is the control and is built from the same copied source as the others, so a
# difference is k and not the build.
#
# MECHANISM CHECK: bytes per token must fall roughly in proportion to k. If they do not,
# TOPK did not take effect and the token comparison below is meaningless -- a bare
# `#define TOPK 16` already silently defeated one attempt at this.
set -u

OUT=/tmp/topk
TOKENS=${TOKENS:-32}
CACHE=${CACHE:-24}
PROMPT_IDS=${PROMPT_IDS:-"1008 10484 318 15383 387"}
LABEL=${LABEL:-france}

echo "######## prompt=$LABEL ids=[$PROMPT_IDS] tokens=$TOKENS cache=$CACHE ########"
printf "%4s %11s %13s %12s %s\n" "k" "s/token" "GB/token" "32 tokens" "first divergence from k=16"

ref=""
for k in 16 12 8 6 4; do
    BIN="$OUT/clover-one-k$k" TOKENS="$TOKENS" CACHE="$CACHE" TAG="k$k-$LABEL" \
        PROMPT_IDS="$PROMPT_IDS" bash /tmp/standalone-cost.sh > "$OUT/run-k$k-$LABEL.log" 2>&1

    if ! grep -q "generated tokens" "$OUT/run-k$k-$LABEL.log"; then
        printf "%4s  *** FAILED: %s\n" "$k" "$(tail -1 "$OUT/run-k$k-$LABEL.log")"
        continue
    fi
    spt=$(awk '/generated tokens/ {print $5}' "$OUT/run-k$k-$LABEL.log")
    gpt=$(awk '/bytes read/ {print $6}' "$OUT/run-k$k-$LABEL.log")
    tot=$(awk '/^answer:/ {print $4}' "$OUT/run-k$k-$LABEL.log")
    seq=$(grep -h "^TOKENS_PRODUCED:" "$OUT/run-k$k-$LABEL.log" | cut -d' ' -f2-)

    if [ "$k" = 16 ]; then
        ref="$seq"; note="control"
    elif [ "$seq" = "$ref" ]; then
        note="IDENTICAL for all $TOKENS tokens"
    else
        # first position where they differ, 1-based over generated tokens
        note=$(python3 -c '
import sys
a = sys.argv[1].split(); b = sys.argv[2].split()
for i in range(min(len(a), len(b))):
    if a[i] != b[i]:
        print("diverges at token %d of %d  (%s vs %s)" % (i+1, len(a), a[i], b[i])); break
else:
    print("prefix identical, lengths %d vs %d" % (len(a), len(b)))
' "$ref" "$seq")
    fi
    printf "%4s %11s %13s %12s %s\n" "$k" "$spt" "$gpt" "$tot" "$note"
done

echo
echo "bytes/token must fall roughly with k. If it does not, TOPK did not take effect"
echo "and the divergence column above means nothing."
touch /tmp/topk-$LABEL.done
