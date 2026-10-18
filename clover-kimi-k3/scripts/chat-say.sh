#!/bin/bash
# One turn of a chat session: sends a message, waits up to WAIT seconds (60) for the reply and prints it.
# usage: [WAIT=seconds] chat-say.sh NAME "message"        or        chat-say.sh NAME -f FILE
# A reply that takes longer stays in session-NAME/reply-NNNN.txt; session-NAME/status.txt has one line per finished turn.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
NAME=$1; DIR="$RESULTS_DIR/session-$NAME"
[ -f "$DIR/ready" ] || { echo "the session is not ready (no $DIR/ready)"; exit 1; }
N=$(( $(ls "$DIR"/turn-*.txt 2>/dev/null | wc -l) + 1 )); T=$(printf %04d $N)
if [ "${2:-}" = "-f" ]; then cp "$3" "$DIR/turn-$T.tmp"; else printf '%s\n' "$2" > "$DIR/turn-$T.tmp"; fi
mv "$DIR/turn-$T.tmp" "$DIR/turn-$T.txt"
echo "turn $N sent at $(date +%T)"
for _ in $(seq 1 ${WAIT:-60}); do
  [ -f "$DIR/reply-$T.txt" ] && { cat "$DIR/reply-$T.txt"; echo; tail -1 "$DIR/status.txt"; exit 0; }
  sleep 1
done
echo "no reply yet; later: cat $DIR/reply-$T.txt   (progress: tail $DIR/status.txt)"
