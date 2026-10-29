#!/bin/bash
# Ends a chat session after its current turn; the model unloads.
# usage: chat-stop.sh [NAME]
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
DIR="$RESULTS_DIR/session-${1:-chat}"
touch "$DIR/quit" && echo "asked session ${1:-chat} to end"
