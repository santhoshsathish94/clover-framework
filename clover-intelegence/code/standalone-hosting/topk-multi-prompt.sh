#!/usr/bin/env bash
# Does k=12 stay identical to k=16 across prompts, and for longer than 32 tokens?
#
# WHAT THE SINGLE-PROMPT SWEEP FOUND
#   k=12  identical for all 32 tokens        2.577 s/token  13.77 GB/token
#   k=8   diverges at token 23               2.180          9.39
#   k=6   diverges at token 4                1.984          7.28
#   k=4   diverges at token 2                1.765          5.04
#
# Two things that matters. First, it answers the question k3-data-problem.md left open:
# k=4 preserves the PREFILL token 17374 but emits 5046 instead of 20829 at the very
# first decode step, so the prefill result did not generalise. Second, k=8 was the
# setting that document called "nearly free" on confidence grounds, 0.790 against
# 0.792 -- and it still diverges by token 23. **Equal confidence on one token does not
# survive a long answer.** Error compounds, and only a long decode shows it.
#
# So k=12 is the one adoption candidate, and one prompt over 32 tokens is not enough to
# adopt anything. The divergence points above show how abruptly this turns: 23, 4, 2.
# This runs several real prompts from the repository's own prompts.tsv, at twice the
# length, and reports where each one first differs.
#
# The gate for adoption is IDENTICAL, not "close". A diverged sequence may well still be
# good text, but that is a judgement about quality that this harness cannot make and
# should not pretend to.
set -u

OUT=/tmp/topk
TOKENS=${TOKENS:-64}
CACHE=${CACHE:-24}
KS=${KS:-"12"}
ROWS=${ROWS:-"1 2 4 6 8"}
TSV=/opt/clover-k3/prompts.tsv

printf "%-42s %4s %10s %11s  %s\n" "prompt" "k" "s/token" "GB/token" "first divergence from k=16"

for row in $ROWS; do
    line=$(awk -v r="$row" -F'\t' '$1 == r {print}' "$TSV")
    [ -z "$line" ] && { echo "row $row not found"; continue; }
    ids=$(echo "$line" | cut -f3 | tr ',' ' ')
    text=$(echo "$line" | cut -f4 | cut -c1-40)

    ref=""
    for k in 16 $KS; do
        BIN="$OUT/clover-one-k$k" TOKENS="$TOKENS" CACHE="$CACHE" TAG="p$row-k$k" \
            PROMPT_IDS="$ids" bash /tmp/standalone-cost.sh > "$OUT/p$row-k$k.log" 2>&1
        if ! grep -q "generated tokens" "$OUT/p$row-k$k.log"; then
            printf "%-42s %4s  *** FAILED: %s\n" "$text" "$k" "$(tail -n 1 "$OUT/p$row-k$k.log")"
            continue
        fi
        spt=$(awk '/generated tokens/ {print $5}' "$OUT/p$row-k$k.log")
        gpt=$(awk '/bytes read/ {print $6}' "$OUT/p$row-k$k.log")
        seq=$(grep -h "^TOKENS_PRODUCED:" "$OUT/p$row-k$k.log" | cut -d' ' -f2-)

        if [ "$k" = 16 ]; then
            ref="$seq"
            printf "%-42s %4s %10s %11s  reference\n" "$text" "$k" "$spt" "$gpt"
        elif [ "$seq" = "$ref" ]; then
            printf "%-42s %4s %10s %11s  IDENTICAL for all %s tokens\n" "" "$k" "$spt" "$gpt" "$TOKENS"
        else
            d=$(python3 -c '
import sys
a, b = sys.argv[1].split(), sys.argv[2].split()
for i in range(min(len(a), len(b))):
    if a[i] != b[i]:
        print("diverges at token %d of %d (%s vs %s)" % (i+1, len(a), a[i], b[i])); break
else:
    print("prefix identical, lengths %d vs %d" % (len(a), len(b)))
' "$ref" "$seq")
            printf "%-42s %4s %10s %11s  %s\n" "" "$k" "$spt" "$gpt" "$d"
        fi
    done
done
echo
echo "Adoption gate is IDENTICAL across every prompt. One divergence anywhere means k=12"
echo "changes answers, and the speed is then a separate question about acceptable drift."
touch /tmp/multiprompt.done
