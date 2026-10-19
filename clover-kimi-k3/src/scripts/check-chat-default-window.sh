#!/bin/bash
# The chat at the DEFAULT window (8192 positions, prompt passes of 1024 tokens) with real content, four turns in one model run:
#  1. a long first message (PROMPT_FILE). If REFERENCE_REPLY is given, the start of the reply must equal the start of that earlier reply, which was made by
#     one prompt pass over the whole prompt: that compares the chunked pass and the chat path with the one-shot path, token for token.
#  2. a short follow-up: a continued prompt pass, checked against one pass over everything (PF_SESSION_VERIFY=1).
#  3. a message that does not fit beside the reply reserve (PF_REPLY_RESERVE=6500): the earlier turns are summarised and the state is rebuilt, at the 8192 window.
#  4. a question that needs a fact from turn 1, which now only the summary holds.
# Memory is sampled after turn 1. Run it detached; the result is work/results/check-chat-default-window.result.
# usage: check-chat-default-window.sh PROMPT_FILE [REFERENCE_REPLY]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
PROMPT=$(readlink -f "${1:?prompt file}"); REF=${2:+$(readlink -f "$2")}
cd "$RESULTS_DIR"
NAME=defaultwindow; DIR=$RESULTS_DIR/session-$NAME; OUT=$RESULTS_DIR/check-chat-default-window.result
export PF_REPLY_RESERVE=6500 PF_DECODE=200 PF_SUMMARY_TOKENS=200 PF_SESSION_VERIFY=1
unset PF_CONTEXT PF_CHUNK PF_INPUT_MAX
"$SCRIPTS_DIR/chat-start.sh" $NAME > "$OUT" 2>&1 || exit 1
for _ in $(seq 1 240); do [ -f "$DIR/ready" ] && break; sleep 5; done
[ -f "$DIR/ready" ] || { echo "the model was not ready after 20 minutes"; tail -5 "$DIR/engine.log"; } > "$OUT"
[ -f "$DIR/ready" ] || exit 1
{
  grep -m1 SESSION "$DIR/engine.log" | cut -c1-200
  say() { echo; echo "=== USER: $(head -c 140 "$1" | tr '\n' ' ')..."; WAIT=4500 "$SCRIPTS_DIR/chat-say.sh" $NAME -f "$1"; }
  T=$DIR/inputs; mkdir -p "$T"
  echo "In one sentence: what was the single most important cause?" > $T/2.txt
  { echo "Here is another part of my project log:"; echo; tail -c +4001 "$ENGINE_DIR/docs/current-status.md" | head -c 1500; echo; echo; echo "Question: in one sentence, what does this part say?"; } > $T/3.txt
  echo "What was the incident I asked about in my first message, and in which year did it happen?" > $T/4.txt
  say "$PROMPT"
  echo "--- memory after turn 1: $(ps -o rss= -C whole-model | awk '{printf "process resident %.0f GB", $1/1048576}'), $(free -g | awk '/Mem:/ {print "used " $3 " GB, buffers/cache " $6 " GB, available " $7 " GB"}')"
  if [ -n "$REF" ]; then
    n=$(wc -c < "$DIR/reply-0001.txt")
    if head -c "$n" "$REF" | cmp -s - "$DIR/reply-0001.txt"; then echo "--- REFERENCE: the first $n bytes of the reply equal the start of the one-shot reply: IDENTICAL"; else echo "--- REFERENCE: the reply differs from the one-shot reply within its first $n bytes: DIFFERENT"; fi
  fi
  say $T/2.txt; say $T/3.txt; say $T/4.txt
  echo; echo "=== STATUS"; cat "$DIR/status.txt"
  "$SCRIPTS_DIR/chat-stop.sh" $NAME
  echo "done $(date)"
} > "$OUT" 2>&1
