#!/bin/bash
# Checks the chat session with a small context window so that the paths run: a fact recalled from an earlier turn, a long message summarised first
# (LONG_INPUT=1 adds that turn, about 20 minutes), and the oldest turns summarised when the window fills. PF_SESSION_VERIFY=1 makes every
# continued turn also run one pass over the whole conversation and compare the two (the line in status.txt says bit-identical or DIFFERENT).
# One model run; run it detached, the result is in work/results/check-chat-session.result.
# usage: [LONG_INPUT=1] check-chat-session.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
NAME=check; DIR=$RESULTS_DIR/session-$NAME; OUT=$RESULTS_DIR/check-chat-session.result
export PF_CONTEXT=1024 PF_CHUNK=256 PF_REPLY_RESERVE=256 PF_INPUT_MAX=400 PF_SUMMARY_TOKENS=120 PF_DECODE=40 PF_SESSION_VERIFY=1
"$SCRIPTS_DIR/chat-start.sh" $NAME "Answer briefly." > "$OUT" 2>&1 || exit 1
for _ in $(seq 1 240); do [ -f "$DIR/ready" ] && break; sleep 5; done
[ -f "$DIR/ready" ] || { echo "the model was not ready after 20 minutes"; tail -5 "$DIR/engine.log"; } > "$OUT"
[ -f "$DIR/ready" ] || exit 1
LOG=$ENGINE_DIR/docs/clover-kimi-k3-cpu-context.md
{
  echo "context $PF_CONTEXT positions, chunk $PF_CHUNK, reply reserve $PF_REPLY_RESERVE, longest message $PF_INPUT_MAX tokens, reply limit $PF_DECODE tokens"
  say() { echo; echo "=== USER: $(head -c 160 "$1" | tr '\n' ' ')..."; WAIT=3000 "$SCRIPTS_DIR/chat-say.sh" $NAME -f "$1"; }
  T=$DIR/inputs; mkdir -p "$T"
  piece() { { echo "Here is part $1 of my project log:"; echo; tail -c +$2 "$LOG" | head -c 1200; echo; echo; echo "Question: in one sentence, what does this part say?"; } > $T/$3.txt; }
  echo "My name is Santhosh and I am building a C inference engine for Kimi K3. Please remember this. Answer in one short sentence." > $T/1.txt
  piece 1 1 2; piece 2 2501 3; piece 3 5001 4
  echo "What was my name again, and what did I say I am building?" > $T/5.txt
  { echo "Here is a longer part of my project log:"; echo; tail -c +8001 "$LOG" | head -c 3600; echo; echo; echo "Question: in one sentence, what does this say about speed?"; } > $T/long.txt
  say $T/1.txt
  [ "${LONG_INPUT:-0}" = 1 ] && say $T/long.txt
  say $T/2.txt; say $T/3.txt; say $T/4.txt; say $T/5.txt
  echo; echo "=== STATUS"; cat "$DIR/status.txt"
  "$SCRIPTS_DIR/chat-stop.sh" $NAME
  echo "done $(date)"
} > "$OUT" 2>&1