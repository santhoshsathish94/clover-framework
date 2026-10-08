#!/bin/sh
# If prompts longer than NPOS_SLOTS skip their leading positions, the continuation will
# ignore them. Same sentence truncated to 8 and to 15 tokens; compare what comes out.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
SENT=1008,17296,318,6903,1900,67244,24231,43458,559,276,133690,318,1284,352,8554
for N in 8 15; do
  IDS=$(echo "$SENT" | cut -d, -f1-"$N")
  printf '{"input_ids":[%s],"max_new_tokens":12}\n' "$IDS" \
    | "$S/run.sh" >/tmp/txt$N.out 2>/dev/null
  echo "--- in=$N ---"
  printf 'prompt : '
  python3 /tmp/ask.py decode $(echo "$IDS" | tr ',' ' ')
  printf 'output : '
  python3 /tmp/ask.py decode $(grep -o '"index":[0-9]*,"token":[0-9]*' /tmp/txt$N.out \
    | grep -o 'token":[0-9]*' | cut -d: -f2 | tr '\n' ' ')
done
