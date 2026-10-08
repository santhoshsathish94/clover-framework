#!/bin/sh
# Ask one free-text question and print the generated continuation.
# usage: askrun.sh "Gravity is" 32
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TEXT="$1"
WANT="${2:-32}"
IDS=$(python3 /tmp/ask.py encode "$TEXT" 2>/dev/null)
printf '{"input_ids":[%s],"max_new_tokens":%s}\n' "$IDS" "$WANT" \
  | CLOVER_SNAPFOLD=score CLOVER_SNAPSHOT=layer "$S/run.sh" >/tmp/ask.out 2>/tmp/ask.err
OUT=$(grep -oE '"index":[0-9]+,"token":[0-9]+' /tmp/ask.out | grep -oE 'token":[0-9]+' | cut -d: -f2 | tr '\n' ' ')
echo "prompt : $TEXT"
echo "ids    : $IDS"
printf 'answer : '
python3 /tmp/ask.py decode $OUT
echo "stop   : $(grep -o 'stop_reason":"[a-z]*' /tmp/ask.out | cut -d'"' -f3)"
echo "time   : $(grep -oE '"seconds":[0-9.]+' /tmp/ask.out | tail -1 | cut -d: -f2) s"
