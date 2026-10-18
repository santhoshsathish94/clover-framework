#!/bin/bash
# Chat answers (no thinking) to three questions under exact and use_4bit_trunk, up to 150 new tokens each, to compare the length and style of the replies.
# usage: compare-chat-answers.sh      (the first run builds)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
printf "%s" "What is the boiling point of water?" > chat-question-1.txt
printf "%s" "Who wrote Pride and Prejudice?" > chat-question-2.txt
printf "%s" "Explain what a hash table is." > chat-question-3.txt
n=0
for q in chat-question-1 chat-question-2 chat-question-3; do
  n=$((n+1))
  if [ $n = 1 ]; then B=1; else B=0; fi
  BUILD=$B MODES="exact use_4bit_trunk" PREFIX=$q PF_CHAT=nothink "$SCRIPTS_DIR/compare-exact-and-4bit-trunk.sh" $q.txt 150 > $q.out 2>&1
  echo "=== $(cat $q.txt)"; cat $q.out
done
