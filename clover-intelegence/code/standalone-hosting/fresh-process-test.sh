#!/bin/sh
# Does the engine carry sequence state across requests? Run each length in its own
# process, so nothing can be inherited, and compare against the sweep.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
SENT=1008,17296,318,6903,1900,67244,24231,43458,559,276,133690,318,1284,352,8554
for N in 8 9 15; do
  IDS=$(echo "$SENT" | cut -d, -f1-"$N")
  printf '{"input_ids":[%s],"max_new_tokens":1}\n' "$IDS" \
    | "$S/run.sh" >/tmp/fresh$N.out 2>/tmp/fresh$N.err
  echo "fresh process, in=$N"
  grep -o 'STEP_JSON {"position":[0-9]*,"token":[0-9]*,"seconds":[0-9.]*' /tmp/fresh$N.err
done
