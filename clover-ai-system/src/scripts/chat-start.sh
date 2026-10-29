#!/bin/bash
# Starts a chat session, detached: the model stays loaded and keeps the conversation in memory between turns (see run_session in src/code/common/whole-model.c).
# usage: chat-start.sh [NAME] ["system message"]       then chat-say.sh NAME "message", and chat-stop.sh NAME
# Settings (environment): PF_CONTEXT positions the layers hold (8192), PF_DECODE the longest reply (2500), PF_CHUNK tokens per prompt pass (1024),
#   PF_REPLY_RESERVE room kept for the reply before old turns are summarised (1024), PF_INPUT_MAX longest message before it is summarised first (PF_CONTEXT/2),
#   PF_SUMMARY_TOKENS longest summary (320), BUILD=1 to build first.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
NAME=${1:-chat}; DIR="$RESULTS_DIR/session-$NAME"
for _ in $(seq 1 36); do pgrep -x whole-model > /dev/null || break; sleep 5; done   # a session that was just stopped needs a moment to exit
if pgrep -x whole-model > /dev/null; then echo "a model is already running on this host"; exit 1; fi
rm -rf "$DIR"; mkdir -p "$DIR"
echo "hello" > "$DIR/placeholder.txt"
(PF_SESSION_DIR="$DIR" PF_CHAT=1 PF_CACHE=16 PF_DECODE="${PF_DECODE:-2500}" SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8 PF_SYSTEM="${2:-}" BUILD="${BUILD:-0}" \
  setsid nohup "$SCRIPTS_DIR/run-model.sh" text "$DIR/placeholder.txt" > "$DIR/engine.log" 2>&1 < /dev/null &)
echo "starting; the file $DIR/ready appears when the model is loaded"
