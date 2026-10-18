#!/bin/bash
# Chat answers (no thinking) to a few questions under exact and use_4bit_trunk (USE_4BIT_TRUNK=1), up to 150 new tokens each, to compare the LENGTH and style of the replies.
# usage: chat-compare.sh      (needs quant-compare.sh/.py, full-run.sh in /tmp/one-layer; the first run builds)
cd /tmp/one-layer
printf "%s" "What is the boiling point of water?" > cq1.txt
printf "%s" "Who wrote Pride and Prejudice?" > cq2.txt
printf "%s" "Explain what a hash table is." > cq3.txt
n=0
for q in cq1 cq2 cq3; do
  n=$((n+1))
  if [ $n = 1 ]; then NB=""; else NB=1; fi
  NOBUILD=$NB MODES="exact use_4bit_trunk" PREFIX=$q PF_CHAT=nothink ./quant-compare.sh $q.txt 150 > $q.out 2>&1
  echo "=== $(cat $q.txt)"; cat $q.out
done
