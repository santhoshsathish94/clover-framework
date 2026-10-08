#!/bin/sh
# Prompt-length sweep: prefixes of one sentence, 2..15 input tokens, 64 output tokens each.
# One resident process, so startup is paid once.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
SENT="The theory of general relativity explains gravity as the curvature of spacetime caused by mass"
ALL=$(python3 /tmp/ask.py encode "$SENT" 2>/dev/null)
echo "sentence ids: $ALL"
{
  N=2
  while [ "$N" -le 15 ]; do
    echo "$ALL" | cut -d, -f1-"$N" | while read -r IDS; do
      printf '{"input_ids":[%s],"max_new_tokens":64}\n' "$IDS"
    done
    N=$((N + 1))
  done
} > /tmp/sweep.jsonl
wc -l < /tmp/sweep.jsonl
"$S/run.sh" < /tmp/sweep.jsonl > /tmp/sweep.out 2> /tmp/sweep.err
echo "run rc=$? -- results in /tmp/sweep.out"
